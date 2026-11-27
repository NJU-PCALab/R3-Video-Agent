"""Two observers and a director with bounded, media-bound conflict investigation."""
from copy import deepcopy
from pathlib import Path
import subprocess

from . import prompts
from . import closed_loop
from . import evidence_policy
from . import decision_contract
from . import attempt_budget
from . import applicability, speech_contract
from .content import closed_loop_enabled
from .content import PROTOCOL, caption_issues
from .review import PERCEPTUAL_CHECKS, opinion, visual_frames
from .storage import Unavailable, key, read, score, sha, write

AV_CHECKS = ('lip_sync', 'speaker_binding', 'sound_event_alignment', 'speech_intelligibility',
             'audio_quality', 'unexpected_music')
AV_DIMS = ('dialogue_fidelity', 'speaker_consistency', 'audio_quality', 'audiovisual_sync', 'narrative_continuity')


def normalized(role, raw):
    result = opinion(role, raw)
    if not isinstance(result.get('uncertainties'),list):
        result['uncertainties']=[{'kind':'malformed_uncertainty_output','raw_value':result.get('uncertainties')}]
    if role == 'audiovisual':
        scores = raw.get('scores') if isinstance(raw, dict) and isinstance(raw.get('scores'), dict) else {}
        result['scores'] = {d: scores.get(d) if score(scores.get(d)) else None for d in AV_DIMS}
        result['alignment_checks'] = raw.get('alignment_checks', []) if isinstance(raw, dict) and isinstance(raw.get('alignment_checks'), list) else []
        result['utterances'] = raw.get('utterances', []) if isinstance(raw, dict) and isinstance(raw.get('utterances'), list) else []
        result['transcript'] = raw.get('transcript', result['utterances']) if isinstance(raw, dict) else []
        if isinstance(raw, dict) and 'speech_observation' in raw:
            result['speech_observation'] = raw['speech_observation']
        for index, c in enumerate(result['alignment_checks'] if isinstance(result['alignment_checks'], list) else []):
            if isinstance(c, dict) and c.get('status') == 'defect' and not any(f.get('category') == c.get('category') for f in result['findings']):
                result['findings'].append({'id': f'{role}:alignment:{index+1}', **deepcopy(c), 'scope': 'local'})
    return result


def catalogue(panel, investigations=()):
    rows = {}
    for role, p in panel.items():
        for index, item in enumerate(p['observed_events']):
            ident = f'{role}:observation:{index+1}'
            rows[ident] = {'role': role, 'observation': item, 'source': 'initial_observation'}
        for item in p['findings']:
            rows[item['id']] = {'role': role, 'observation': item.get('observation'), 'source': 'finding'}
        for i, item in enumerate(p.get('uncertainties') or []):
            rows[f'{role}:uncertainty:{i+1}'] = {'role':role, 'observation':item, 'source':'observer_uncertainty'}
        for c in list(p.get('perceptual_checks') or []) + list(p.get('alignment_checks') or []):
            if isinstance(c, dict):
                rows[f'{role}:check:{c.get("category")}'] = {'role': role, 'observation': c.get('observation'),
                    'time_s': c.get('time_s'), 'status': c.get('status'), 'source': 'check'}
    for i, inv in enumerate(investigations):
        rows[f'investigation:{i+1}'] = {'role': inv['role'], 'observation': inv['raw_reply'].get('observed_events'),
            'source': 'targeted_actual_media_reobservation', 'media_binding': inv['media_binding']}
    return rows


def evidence_problems(panel, decision, refs, investigations=()):
    problems = []
    if not isinstance(decision, dict):
        return [{'kind': 'decision_unavailable'}]
    dispositions = {d.get('finding_id'): d for d in closed_loop.rows(decision.get('finding_dispositions')) if isinstance(d, dict) and isinstance(d.get('finding_id'),str)}
    for role, p in panel.items():
        for f in p['findings']:
            d = dispositions.get(f['id'], {})
            accepted = d.get('disposition') in ('not_a_defect', 'covered_by_adjacent', 'accepted_tradeoff')
            if not accepted:
                continue
            cited = d.get('evidence_refs', [])
            if not isinstance(cited, list) or not cited or any(c not in refs for c in cited if isinstance(c, str)) or any(not isinstance(c, str) for c in cited):
                problems.append({'kind': 'unsupported_evidence_citation', 'finding_id': f['id']})
            if f.get('severity') == 'major':
                refuted = any(r.get('finding_id') == f['id'] and r.get('status') == 'refuted' and r.get('observation')
                    for inv in investigations if inv['role'] == role
                    for r in closed_loop.rows(inv['raw_reply'].get('finding_resolutions')) if isinstance(r, dict))
                if not refuted:
                    problems.append({'kind': 'major_defect_without_actual_refutation', 'finding_id': f['id']})
    return problems


def quality(review):
    if review.get('observation_protocol') == closed_loop.VERSION:
        return closed_loop.quality(review)
    panel, decision = review.get('opinions', {}), review.get('feedback_decision') or {}
    outstanding = []
    known = all(panel.get(r, {}).get('status') == 'observed' for r in ('visual', 'audiovisual'))
    for role, categories, field in [('visual', PERCEPTUAL_CHECKS, 'perceptual_checks'), ('audiovisual', AV_CHECKS, 'alignment_checks')]:
        checks = {c.get('category'): c for c in panel.get(role, {}).get(field, []) if isinstance(c, dict)}
        for category in categories:
            c = checks.get(category, {})
            allowed_na = category in ('lip_sync', 'speaker_binding', 'speech_intelligibility') and (panel.get(role, {}).get('voice_observation') or {}).get('status') == 'ABSENT'
            valid = c.get('status') in ('clear', 'defect') or (c.get('status') == 'not_applicable' and allowed_na)
            if not valid or not c.get('observation'):
                outstanding.append({'id': f'{role}:unknown:{category}', 'category': category,
                    'severity': 'uncertain', 'observation': c.get('observation', 'Observation missing')})
    dispositions = {x.get('finding_id'): x for x in (decision.get('finding_dispositions') or []) if isinstance(x, dict)}
    audit = evidence_problems(panel, decision, review.get('evidence_catalog', {}), review.get('investigations', []))
    bad_ids = {x.get('finding_id') for x in audit}
    for p in panel.values():
        for f in p.get('findings', []):
            d = dispositions.get(f['id'], {})
            if f['id'] in bad_ids or d.get('disposition') not in ('not_a_defect', 'covered_by_adjacent', 'accepted_tradeoff') or not d.get('reason'):
                outstanding.append(f)
        if p.get('uncertainties') and not (isinstance(decision.get('uncertainty_resolution_refs'), list) and decision['uncertainty_resolution_refs'] and all(isinstance(x,str) and x.startswith('investigation:') and x in review.get('evidence_catalog',{}) for x in decision['uncertainty_resolution_refs'])):
            outstanding.append({'category': 'unresolved_observer_uncertainty', 'severity': 'uncertain', 'observation': p['uncertainties']})
    coverage = decision.get('target_coverage') if isinstance(decision.get('target_coverage'), dict) else {}
    if coverage.get('current_event') != 'covered' or coverage.get('scripted_speech') not in ('covered', 'not_applicable') or coverage.get('progression') != 'new_event':
        outstanding.append({'category': 'target_or_progression_unverified', 'severity': 'uncertain', 'observation': coverage})
    for problem in review.get('content_conflicts', []):
        outstanding.append({'category': problem['kind'], 'severity': 'major', 'observation': problem})
    outstanding.extend(audit)
    action = decision.get('decision')
    complete = known and action in ('accept', 'revise', 'keep_needs_review') and bool(decision.get('reason'))
    return {'ok': complete and action == 'accept' and not outstanding, 'complete': complete,
        'rank': None, 'quality_status': 'OK' if complete and action == 'accept' and not outstanding else 'draft/needs_review',
        'review_system': 'feedback_v1', 'acceptance_basis': 'two_actual_media_observers_and_grounded_conflict_resolution',
        'unresolved_findings': outstanding, 'feedback_decision': action, 'not_applicable_dimensions': []}


def investigation_media(agent, receipt, question, role, name):
    """Retain a real excerpt with absolute time origin. Never synthesize evidence."""
    if sha(receipt['path']) != receipt['sha256']:
        raise Unavailable('Investigation media differs from original receipt')
    try:
        start = max(0., min(float(question.get('start_s', 0)), float(receipt['seconds']) - 1/24))
        end = min(float(receipt['seconds']), max(start + 1/24, float(question.get('end_s', receipt['seconds']))))
    except (ValueError, TypeError):
        start, end = 0., float(receipt['seconds'])
    end = min(end, start + 4.)
    folder = Path(agent.s.run) / 'review_media' / name
    manifest = folder / 'investigation.json'
    old = read(manifest)
    if old:
        if old['source_sha256'] != receipt['sha256'] or any(sha(x['path']) != x['sha256'] for x in old['files']):
            raise Unavailable('Retained investigation changed; reconcile original evidence')
        return old
    folder.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(Path(agent.config['media_bin']) / 'ffmpeg')
    files = []
    if role == 'visual':
        cmd = [ffmpeg, '-v', 'error', '-threads', '1', '-ss', str(start), '-i', receipt['path'], '-t', str(end-start),
            '-vf', 'fps=12,scale=1024:-2', '-threads', '1', '-q:v', '3', str(folder/'%04d.jpg')]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode:
            raise Unavailable('Targeted visual extraction failed: ' + proc.stderr.decode(errors='replace')[-500:])
        files = [{'path': str(p), 'sha256': sha(p), 'label': f'Current actual silent frame at {start+i/12:.3f}s'}
                 for i,p in enumerate(sorted(folder.glob('*.jpg')))]
    else:
        for source, filename, options in [(receipt['path'], 'current_excerpt.mp4', ['-map','0:v:0','-map','0:a:0','-c:v','libx264','-preset','fast','-crf','20','-c:a','aac']),
            (receipt['audio_path'], 'current_native_excerpt.wav', ['-vn','-c:a','pcm_s16le'])]:
            target = folder/filename
            proc = subprocess.run([ffmpeg, '-y','-v','error','-threads','1','-ss',str(start),'-i',source,'-t',str(end-start),*options,str(target)], capture_output=True, timeout=120)
            if proc.returncode:
                raise Unavailable('Targeted audiovisual extraction failed: ' + proc.stderr.decode(errors='replace')[-500:])
            files.append({'path':str(target), 'sha256':sha(target), 'label': f'Current actual excerpt, absolute time origin {start:.3f}s; WAV is the same native audio, not a second event'})
    if not files:
        raise Unavailable('No real investigation evidence extracted')
    value = {'source_sha256': receipt['sha256'], 'native_audio_sha256': receipt.get('audio_sha256'),
        'start_s':start, 'end_s':end, 'files':files, 'scope':'current actual media; timestamps returned in original clip coordinates'}
    write(manifest,value)
    return value


class CollaborativePanel:
    def __init__(self, agent):
        self.a = agent

    def review(self, cid, candidate):
        a=self.a; take=candidate['take']; protocol=candidate.get('review_protocol', a.config.get('review_system', {}).get('observation_protocol', PROTOCOL)); v4=protocol==closed_loop.VERSION
        name=f'review_panel/{cid}/{take}/{protocol}'
        observed_only = candidate.get('review_policy_version') == evidence_policy.VERSION
        if observed_only:
            name += '/' + evidence_policy.VERSION
        app_contract = candidate.get('review_applicability_version') == applicability.VERSION
        if app_contract:
            name += '/' + applicability.VERSION
        previous = a.s.data['selected'][-1] if a.s.data['selected'] else None
        actual_facts = evidence_policy.media_facts(candidate, previous) if observed_only else None
        filter_audit, decision_audit = [], []
        panel, frames, visual_media = {}, {}, []
        visual_problem = None
        try:
            for label, item in ([('previous',previous)] if previous else []) + [('current',candidate)]:
                receipt=item['receipt']; f=visual_frames(a.config,a.s.run,receipt, f'{cid}_{take}_{label}_{protocol}')
                frames[label]=f
                visual_media += [{'path':x['path'],'label':f'{label.upper()} complete silent sequence: {x["time_s"]:.3f}s'} for x in f['frames']]
        except Unavailable as exc:
            visual_problem=str(exc)
            a.s.warning(name+'/visual_evidence',exc)
        visual_media += [{'path':r['path'],'label':'Actual identity/setting reference, not a film frame: '+r['id']}
                         for r in candidate['references'] if r['kind']=='image']
        av_media=[{'path':candidate['receipt']['path'],'label':'CURRENT complete actual audiovisual clip'},
                  {'path':candidate['receipt']['audio_path'],'label':'CURRENT native WAV, same soundtrack as the video, not another event'}]
        context={'current_seconds':candidate['receipt']['seconds'], 'previous_seconds':previous['receipt']['seconds'] if previous else None,
            'reference_assets_are_not_film_frames':True, 'source_and_caption_withheld':True,
            'sound_policy':{'background_music':(a.config.get('editing_policy') or {}).get('background_music')}}
        for role,prompt,provider,media in [('visual',prompts.VISUAL_V3,'doubao',visual_media),('audiovisual',prompts.AV_V3,'gemini',av_media)]:
            if role=='visual' and visual_problem:
                raw={'status':'unknown','observed_events':[],'findings':[],'uncertainties':[visual_problem]}
            else:
                if observed_only:
                    prompt = prompts.OBSERVED_VISUAL if role == 'visual' else prompts.OBSERVED_AV
                    if app_contract and role == 'audiovisual':
                        prompt += prompts.APPLICABILITY_OBSERVER
                elif v4:
                    prompt += prompts.OBSERVER_CLOSED_LOOP
                payload=context if role=='visual' else {k:v for k,v in context.items() if k!='previous_seconds'}
                raw=a.artifact(name+'/'+role,lambda p=prompt,pr=provider,m=media,u=payload,r=role:
                    a.advisory(name+'/'+r,p,u,provider=pr,media=m,max_tokens=6500))
            panel[role]=normalized(role,raw)
        if observed_only:
            panel, filter_audit = evidence_policy.observed_panel(panel)
        app_context = {'applicability_contract':applicability.VERSION} if app_contract else {}
        app_resolution = None
        if app_contract:
            prepared = applicability.clarify(a,cid,candidate,{'opinions':panel,
                'review_context':{**app_context,'target_has_speech':bool(candidate['source_clip'].get('dialogue'))}},
                name+'/applicability_observation')
            panel = prepared['opinions']
            app_resolution = prepared.get('applicability_resolution')
        speech_verification = speech_contract.verify(a, cid, candidate)
        content_conflicts=caption_issues(candidate['source_clip'],candidate['caption'],a.s.data['selected'])
        investigations=[]
        def evidence_refs():
            refs = catalogue(panel, investigations)
            if observed_only:
                refs['media:current'] = {'role':'host','source':'actual_media_receipt','observation':actual_facts['current']}
                if actual_facts['previous']:
                    refs['media:previous'] = {'role':'host','source':'actual_previous_media_receipt','observation':actual_facts['previous']}
            return refs
        def decide(stage):
            refs=evidence_refs()
            system = ((prompts.OBSERVED_DECISION + (prompts.DECISION_CONTRACT if decision_contract.enabled(a.config) else ''))
                      if observed_only else prompts.DECISION_V3 + (prompts.DECISION_CLOSED_LOOP if v4 else ''))
            if app_contract:
                system += prompts.APPLICABILITY_DECISION
            result = a.artifact(name+'/'+stage,lambda:a.advisory(name+'/'+stage,a.prompt(system),
                {'current_source_event':candidate['source_clip'],'caption_is_only_a_target':candidate['caption'],
                 'independent_speech_verification':speech_verification,
                 'critical_speech_release_issues':speech_contract.critical_issues(speech_verification),
                 'critical_gate_instruction':'An open speech-release issue requires revise/keep_needs_review, never accept. Subjective improvement cannot clear missing words or wrong voice.',
                 **({'actual_media_facts':actual_facts,'reviewer_authority':{'audio_mouth_and_av':'audiovisual','cross_clip_visual_and_identity':'visual'},'review_errors_not_media_defects':decision_audit} if observed_only else {}),
                 'actual_previous_clip':previous['clip'] if previous else None,
                 'opinions':panel,'evidence_catalog':refs,'investigations':investigations,
                 **({'applicability_audit':applicability.audit({'opinions':panel,'review_context':app_context})} if app_contract else {}),
                 **({'host_check_findings':[f for role in ('visual','audiovisual') for f in applicability.check_findings({'opinions':panel},role)]} if app_contract else {}),
                 'content_conflicts':content_conflicts,'remaining_real_takes':attempt_budget.capabilities(a.s,cid,take)['remaining_after_current_slot'],
                 'runtime_capabilities':attempt_budget.capabilities(a.s,cid,take) if v4 else None,
                 'host_reconciliation': reconciliation if stage=='decision_reconciled' else None,
                 'review_context': {**app_context,'source_and_caption_withheld':True,'previous_available':bool(previous), 'target_has_speech':bool(candidate['source_clip'].get('dialogue')), 'concrete_uncertainties_only':v4, 'background_music':context['sound_policy']['background_music']},
                 'retrieved_failure_experience':a.memory.retrieve('local', {'clip':candidate['source_clip'],
                    'findings':[f for p in panel.values() for f in p['findings']], 'conflict_investigation':True}, name+'/'+stage),
                 'evidence_problems_in_initial_decision': evidence_problems(panel,decision,refs,investigations) if stage!='decision_initial' else [],
                 'scope':'Only current clip local fixes. No future plan substitutes for current evidence; no global prefix repairs.'},max_tokens=7000,
                **({'validate':lambda d:decision_contract.validate_applicability(d,require_sound_event=applicability.enabled({'review_context':app_context}) and decision_contract.needs_sound_applicability({'opinions':panel}))} if app_contract else {})))
            if observed_only:
                result, errors = evidence_policy.checked_decision(result, actual_facts)
                decision_audit.extend({'stage':stage,**x} for x in errors)
            return result
        decision=decide('decision_initial')
        questions=decision.get('investigations',[])
        questions=questions if isinstance(questions,list) else []
        # An attempted dismissal of a major finding requires actual re-observation, not prose votes.
        audit=evidence_problems(panel,decision,evidence_refs(),[])
        for issue in audit:
            if issue['kind']=='major_defect_without_actual_refutation':
                role=issue['finding_id'].split(':')[0]
                target_finding=next((f for f in panel[role]['findings'] if f['id']==issue['finding_id']),{})
                times=closed_loop.intervals(target_finding.get('time_s')) if v4 else []
                start=max(0,times[0][0]-.25) if times else 0
                questions.append({'role':role,'start_s':start,'end_s':min(candidate['receipt']['seconds'],start+4),
                    'question':'Check the retained major finding directly; confirm, refute or leave unknown with visible/audible evidence.'})
        av=panel['audiovisual']
        unknown_alignment=any(c.get('status')=='unknown' for c in av.get('alignment_checks',[]) if isinstance(c,dict))
        if unknown_alignment and decision.get('decision') != 'keep_needs_review':
            unknown_check=next(c for c in av.get('alignment_checks',[]) if isinstance(c,dict) and c.get('status')=='unknown')
            times=closed_loop.intervals(unknown_check.get('time_s')) if v4 else []
            start=max(0,times[0][0]-.25) if times else 0
            questions.append({'role':'audiovisual','start_s':start,'end_s':min(start+4,candidate['receipt']['seconds']),
                'question':'Resolve the unknown speaker/mouth/sound alignment from actual current media. A transcription alone is not lip synchronization.'})
        seen=set()
        for q in questions:
            if not isinstance(q,dict) or q.get('role') not in ('visual','audiovisual') or q['role'] in seen:
                continue
            role=q['role'];seen.add(role)
            try:
                binding=investigation_media(a,candidate['receipt'],q,role,f'{cid}_{take}_{role}_investigation')
                media=[{'path':x['path'],'label':x['label']} for x in binding['files']]
                if role=='visual' and previous:
                    media=[{'path':x['path'],'label':f'PREVIOUS complete silent sequence {x["time_s"]:.3f}s'} for x in frames.get('previous',{}).get('frames',[])]+media
                raw=a.artifact(name+'/investigate_'+role,lambda r=role,m=media,b=binding,question=q:
                    a.advisory(name+'/investigate_'+r,prompts.INVESTIGATE_V3 + (prompts.INVESTIGATION_CLOSED_LOOP if v4 else ''),
                        {'question':question,'absolute_time_origin':b['start_s'],'role':r,
                         'retained_findings':panel[r]['findings']+(applicability.check_findings({'opinions':panel},r) if app_contract else []),'initial_observations_are_not_truth':panel[r]['observed_events'],
                         'retained_uncertainties':[{'uncertainty_id':f'{r}:uncertainty:{i+1}','observation':x} for i,x in enumerate(closed_loop.rows(panel[r].get('uncertainties')))]},
                        provider='doubao' if r=='visual' else 'gemini',media=m,max_tokens=5000))
                binding['previous_media_sha256'] = previous['receipt']['sha256'] if role=='visual' and previous else None
                investigations.append({'role':role,'question':q,'media_binding':binding,'raw_reply':raw})
                # Investigation prose is evidence, not automatically a new defect.
                # New findings must be explicitly typed; every original finding stays in the issue ledger.
                fresh_raw = deepcopy(raw)
                if v4:
                    fresh_raw['findings'] = [f for f in closed_loop.rows(raw.get('findings')) if isinstance(f,dict) and f.get('kind', 'defect')=='defect']
                if v4:
                    panel[role]['uncertainties'] = closed_loop.rows(panel[role].get('uncertainties')) + closed_loop.rows(raw.get('uncertainties'))
                fresh = normalized(role, fresh_raw)
                for index, finding in enumerate(fresh['findings']):
                    panel[role]['findings'].append({**finding, 'id': f'{role}:investigation:{index+1}', 'origin': 'actual_targeted_investigation'})
            except Unavailable as exc:
                a.s.warning(name+'/investigate_'+role,exc)
                investigations.append({'role':role,'question':q,'media_binding':{'status':'unknown'},
                    'raw_reply':{'status':'unknown','uncertainties':[str(exc)],'observed_events':[]}})
        # An investigated unknown is resolved only by an actual explicit check on the same category.
        for inv in investigations:
            raw=inv['raw_reply']
            if raw.get('status')!='observed':
                continue
            for field in ('alignment_checks','perceptual_checks'):
                for fresh in (raw.get(field) or []):
                    if not isinstance(fresh,dict) or not fresh.get('observation'):
                        continue
                    original=next((c for c in panel[inv['role']].get(field,[]) if isinstance(c,dict) and c.get('category')==fresh.get('category')),None)
                    if v4:
                        # A four-second excerpt cannot certify an unknown whole eight-second interval.
                        required = closed_loop.intervals(original.get('time_s')) if original else []
                        binding = inv['media_binding']
                        if not required or not all(binding.get('start_s',1e9)<=x<=y<=binding.get('end_s',-1) for x,y in required):
                            continue
                    if original and original.get('status')=='unknown' and fresh.get('status')=='clear':
                        original.update(status='clear',observation=fresh['observation'],resolved_by='actual_targeted_investigation')
        if investigations or audit or decision.get('decision')=='investigate':
            decision=decide('decision_after_investigation')
        if observed_only:
            panel, extra_filter_audit = evidence_policy.observed_panel(panel)
            filter_audit.extend(extra_filter_audit)
        refs=evidence_refs()
        boundary={'status':'ready' if not visual_problem else 'unknown',
            'previous_clip_id':previous['clip']['id'] if previous else None,
            'previous_media_sha256':previous['receipt']['sha256'] if previous else None,
            'current_media_sha256':candidate['receipt']['sha256'],
            'visual_scope':'complete sampled previous and current sequences at 4fps',
            'audio_scope':'current clip video and native WAV only',
            'images':[{'path':f['path'],'label':f'previous tail {f["time_s"]:.3f}s'} for f in frames.get('previous',{}).get('frames',[])[-3:]]+
                     [{'path':f['path'],'label':f'current opening {f["time_s"]:.3f}s'} for f in frames.get('current',{}).get('frames',[])[:3]]}
        write(Path(a.s.run)/'review_media'/f'{cid}_{take}'/'boundary.json',boundary)
        review={'review_system':'feedback_v1','observation_protocol':protocol,'opinions':panel,
            'feedback_decision':decision,'scores':{d:v for p in panel.values() for d,v in p['scores'].items()},
            'voice_observation':av.get('voice_observation'),'transcript':av.get('transcript',[]),
            'observed_events':[x for p in panel.values() for x in p['observed_events']],
            'issues':[x for p in panel.values() for x in p['findings']], 'repair_suggestions':decision.get('revision_plan',{}),
            'investigations':investigations,'evidence_catalog':refs,'content_conflicts':content_conflicts,
            'boundary_evidence':boundary,'review_context': {**app_context,'source_and_caption_withheld':True, 'previous_available':bool(previous), 'target_has_speech':bool(candidate['source_clip'].get('dialogue')), 'concrete_uncertainties_only':v4, 'background_music':context['sound_policy']['background_music']},'limitations':['Silent visual frames sample both full clips at 4fps; targeted frames at 12fps.',
                'Speech timestamps and synchronization judgments are model observations, not independent word/phoneme alignment.'],
            'evidence_credit':'grounded_model_assessments_not_independent_quality_proof'}
        if speech_verification is not None:
            review['speech_contract'] = speech_verification
        if app_resolution:
            review['applicability_resolution'] = app_resolution
        if v4:
            if observed_only and decision_contract.enabled(a.config):
                review['review_context']['decision_contract_version'] = decision_contract.VERSION
            if observed_only:
                review['review_context']['evidence_policy'] = evidence_policy.VERSION
                review.update(actual_media_facts=actual_facts, observation_filter_audit=filter_audit,
                              decision_audit=decision_audit)
            initial_quality = quality(review)
            if not app_contract and (decision.get('decision') == 'investigate' or not initial_quality['decision_consistent'] or (observed_only and decision_audit and not initial_quality['unresolved_findings'] and decision.get('decision')!='accept')):
                reconciliation = {'quality':initial_quality, 'instruction':'Resolve each exact host discrepancy with actual evidence; if unresolved use revise or keep_needs_review. No new media or invented observations.'}
                review['decision_before_reconciliation'] = deepcopy(decision)
                decision = decide('decision_reconciled')
                review['feedback_decision'] = decision
                review['repair_suggestions'] = decision.get('revision_plan', {})
                review['reconciliation'] = {'before':initial_quality,'after':quality(review),'bounded_text_rounds':1}
            review['issue_ledger'] = closed_loop.issue_view(review)
            if observed_only and decision_contract.enabled(a.config):
                review = decision_contract.reconcile(a, cid, candidate, review)
                review['issue_ledger'] = closed_loop.issue_view(review)
        case = \
            {'clip':candidate['source_clip'],'receipt':candidate['receipt'],'review':review,'quality':quality(review),
             'failure_patterns':[f.get('category') for p in panel.values() for f in p['findings']],
             'investigation_count':len(investigations),'not_verified_success_template':True}
        a.memory.record('local',name+'/case','observed_local_review_case',evidence_policy.memory_view(case) if observed_only else case)
        return review
