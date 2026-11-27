"""One evidence-based applicability contract shared by observers and the host."""
from copy import deepcopy
from pathlib import Path

from .storage import Pending, key, read, sha, write

VERSION = 'speech_voice_applicability_v1'
VOCAL_CHECK = 'nonverbal_vocal_alignment'


def obj(value):
    return value if isinstance(value, dict) else {}


def rows(value):
    return value if isinstance(value, list) else []


def enabled(review):
    return obj(obj(review).get('review_context')).get('applicability_contract') == VERSION


def bind(store, cid, candidate, config):
    if config.get('review_system', {}).get('applicability_contract') != VERSION:
        return
    if candidate.get('review_applicability_version'):
        return
    base = f'review_panel/{cid}/{candidate["take"]}/{candidate.get("review_protocol")}/{candidate.get("review_policy_version")}'
    submitted = (candidate.get('review') or (store.run/'artifacts'/base).exists()
        or any((store.run/'calls'/key(base+'/'+role+'/format_1')/'request.json').exists()
               for role in ('visual', 'audiovisual')))
    candidate['review_applicability_version'] = 'legacy' if submitted else VERSION
    store.save()


def no_speech(p):
    return (p.get('status') == 'observed'
        and obj(p.get('speech_observation')).get('status') == 'ABSENT'
        and bool(obj(p.get('speech_observation')).get('reason'))
        and obj(p.get('voice_observation')).get('status') in ('PRESENT', 'ABSENT')
        and bool(obj(p.get('voice_observation')).get('reason'))
        and isinstance(p.get('transcript'), list)
        and not p['transcript'] and not p.get('utterances'))


def nonverbal(p):
    return no_speech(p) and obj(p.get('voice_observation')).get('status') == 'PRESENT'


def checks(review, role, base):
    p = obj(obj(review.get('opinions')).get(role))
    return tuple(base) + ((VOCAL_CHECK,) if role == 'audiovisual' and nonverbal(p) else ())


def check_findings(review, role):
    p = obj(obj(review.get('opinions')).get(role))
    field = 'perceptual_checks' if role == 'visual' else 'alignment_checks'
    existing = {x.get('id') for x in rows(p.get('findings')) if isinstance(x,dict)}
    result = []
    for c in rows(p.get(field)):
        if not isinstance(c,dict) or c.get('status') != 'defect' or not c.get('observation') or not c.get('category'):
            continue
        ident = f'{role}:check_defect:{c["category"]}'
        if ident not in existing:
            result.append({'id':ident,'category':c['category'],'observation':c['observation'],
                'time_s':c.get('time_s'),'severity':'minor' if c.get('severity') == 'minor' else 'major',
                'origin':'actual_observer_check','evidence_refs':[f'{role}:check:{c["category"]}']})
    return result


def assess(review, role, category, check):
    p = obj(obj(review.get('opinions')).get(role))
    status = check.get('status'); observed = bool(check.get('observation'))
    value = {'role': role, 'category': category, 'source_status': status,
             'valid': observed and status in ('clear', 'defect'), 'reason_code': 'actual_check',
             'evidence_ref': f'{role}:check:{category}'}
    if status == 'defect':
        value['finding_id'] = f'{role}:check_defect:{category}'
    if status in ('not_observed', 'outside_observation') and observed:
        value.update(valid=True, reason_code='outside_actual_observation')
    if status != 'not_applicable':
        return value
    value.update(valid=False, reason_code='unsupported_not_applicable')
    if not observed or p.get('status') != 'observed':
        return value
    if category in ('lip_sync', 'speech_intelligibility'):
        value.update(valid=no_speech(p), reason_code='actual_speech_absent')
    elif category == 'speaker_binding':
        value.update(valid=no_speech(p) and obj(p.get('voice_observation')).get('status') == 'ABSENT',
                     reason_code='actual_human_voice_absent')
    elif category == VOCAL_CHECK:
        # Never use absence of lexical speech to exempt nonverbal vocal timing.
        value.update(valid=False, reason_code='nonverbal_voice_requires_its_own_alignment_check')
    elif category == 'sound_event_alignment':
        from .closed_loop import cited
        d = next((x for x in rows(obj(review.get('feedback_decision')).get('applicability'))
                  if isinstance(x, dict) and x.get('category') == category), {})
        value.update(valid=(d.get('required_by_target') is False and d.get('observed_discrete_event') is False
                     and bool(d.get('reason')) and cited(d, review.get('evidence_catalog', {}), 'audiovisual')),
                     reason_code='no_target_or_observed_discrete_event')
    return value


def fact_issues(review):
    p = obj(obj(review.get('opinions')).get('audiovisual'))
    voice = obj(p.get('voice_observation')).get('status')
    speech = obj(p.get('speech_observation')).get('status')
    words = bool(p.get('transcript')) or bool(p.get('utterances'))
    issues = []
    if (speech == 'PRESENT' and voice == 'ABSENT') or (speech == 'ABSENT' and words):
        issues.append({'id':'audiovisual:contradictory_speech_facts', 'role':'audiovisual',
            'kind':'observation_contract_error', 'state':'open', 'category':'speech_state',
            'observation':'Voice/speech state conflicts with actual transcript or utterances'})
    if obj(review.get('review_context')).get('target_has_speech') is True and no_speech(p):
        issues.append({'id':'target:missing_required_speech','role':'audiovisual',
            'kind':'content_defect','state':'open','category':'dialogue_fidelity',
            'observation':'Actual speech is absent although the source requires dialogue'})
    return issues


def audit(review):
    from .collaborative_review import AV_CHECKS
    from .review import PERCEPTUAL_CHECKS
    out = []
    for role, base, field in [('visual', PERCEPTUAL_CHECKS, 'perceptual_checks'),
                              ('audiovisual', AV_CHECKS, 'alignment_checks')]:
        values = rows(obj(obj(review.get('opinions')).get(role)).get(field))
        for category in checks(review, role, base):
            c = next((x for x in values if isinstance(x, dict) and x.get('category') == category), {})
            out.append(assess(review, role, category, c))
    return out


def clarify(agent, cid, candidate, review, operation):
    """One cached full-current-media observation; no director prose substitutes."""
    if not enabled(review):
        return review
    required = [x['category'] for x in audit(review) if x['role'] == 'audiovisual'
                and not x['valid'] and x['category'] in ('lip_sync', 'speaker_binding', VOCAL_CHECK)]
    if not required:
        return review
    from .decision_contract import digest
    from . import prompts
    receipt = candidate['receipt']
    binding = {'clip_id':cid, 'take':candidate['take'], 'media_sha256':receipt['sha256'],
        'audio_sha256':receipt.get('audio_sha256'), 'original_observation_sha256':digest(review['opinions']['audiovisual']),
        'requested_checks':required, 'contract':VERSION}
    path = agent.s.run/'review_media'/cid/str(candidate['take'])/VERSION/'binding.json'
    old = read(path)
    if old is not None and old != binding:
        raise Pending('Applicability observation binding changed; reconcile the original receipt')
    if sha(receipt['path']) != receipt['sha256'] or sha(receipt['audio_path']) != receipt['audio_sha256']:
        raise Pending('Applicability observation media differs from its original receipt')
    if old is None:
        write(path, binding)
    raw = agent.artifact(operation, lambda: agent.advisory(operation, prompts.APPLICABILITY_OBSERVER,
        {'binding':binding,'actual_seconds':receipt['seconds'],'requested_checks':required,
         'retained_actual_observations':review['opinions']['audiovisual'],
         'instruction':'Observe these exact checks across the complete current media. Do not give a film acceptance decision.'},
        provider='gemini', media=[{'path':receipt['path'],'label':'Complete CURRENT actual video'},
            {'path':receipt['audio_path'],'label':'Same native WAV; one event, not repetition'}], max_tokens=4000))
    value = deepcopy(review); before = deepcopy(value['opinions']['audiovisual']['alignment_checks'])
    adopted = []
    if isinstance(raw, dict) and raw.get('status') == 'observed':
        for category in required:
            matches = [x for x in rows(raw.get('alignment_checks')) if isinstance(x, dict) and x.get('category') == category]
            if len(matches) != 1 or not matches[0].get('observation'):
                continue
            c = matches[0]
            from .closed_loop import intervals
            times = intervals(c.get('time_s'))
            if category == VOCAL_CHECK and c.get('status') in ('clear','defect') and not times:
                continue
            if times and not all(0 <= a <= b <= receipt['seconds'] for a,b in times):
                continue
            check_rows = value['opinions']['audiovisual']['alignment_checks']
            check_rows[:] = [x for x in check_rows if isinstance(x,dict) and x.get('category') != category]
            check_rows.append({**deepcopy(c), 'observed_by':operation, 'media_sha256':receipt['sha256']})
            adopted.append(category)
        for index, finding in enumerate(rows(raw.get('findings'))):
            if (not isinstance(finding,dict) or finding.get('kind','defect') != 'defect'
                    or not finding.get('observation') or not finding.get('category')):
                continue
            value['opinions']['audiovisual'].setdefault('findings', []).append({**deepcopy(finding),
                'id':f'audiovisual:applicability:{index+1}', 'origin':'actual_applicability_observation'})
    value['applicability_resolution'] = {'version':VERSION,'binding':binding,'operation':operation,
        'original_checks':before,'adopted_checks':adopted,'raw_reply_artifact':operation,
        'whole_current_clip_observed':True,'media_regenerated':False}
    write(path.parent/'resolution.json', value['applicability_resolution'])
    return value
