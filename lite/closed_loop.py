"""Derived issue lifecycle; raw observations, attempts and selections remain immutable."""
from copy import deepcopy
import math
import re

VERSION = 'content_rsi_v4'


def rows(value):
    return value if isinstance(value, list) else []


def obj(value):
    return value if isinstance(value, dict) else {}


def cited(value, refs, role=None):
    ids = value.get('evidence_refs')
    return (isinstance(ids, list) and bool(ids) and all(isinstance(i, str) and i in refs for i in ids)
            and (role is None or any(refs[i].get('role') == role for i in ids)))


def intervals(value):
    # Model time notation is presentation, not provenance; normalize unambiguous forms.
    if isinstance(value,str):
        value=value.strip().replace('–','-').replace('—','-').replace('至','-')
        def point(text):
            text=text.strip().removesuffix('s').strip()
            parts=text.split(':')
            if len(parts)>3 or not all(re.fullmatch(r'[0-9]+(?:[.][0-9]+)?',x) for x in parts):return None
            result=sum(float(x)*60**i for i,x in enumerate(reversed(parts)))
            return result if math.isfinite(result) else None
        pieces=value.split('-')
        if len(pieces) in (1,2):
            points=[point(x) for x in pieces]
            if all(x is not None for x in points):
                return intervals(points[0] if len(points)==1 else points)
        return []
    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
        return [(float(value),float(value))]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        return [(float(value[0]), float(value[1]))] if value[1] >= value[0] else []
    return [pair for x in rows(value) for pair in intervals(x)]


def reobserved(review, role, finding, status, field='finding_resolutions', id_field='finding_id'):
    """A same-role, same-media, time-covering re-observation can settle a finding."""
    media_sha = obj(review.get('boundary_evidence')).get('current_media_sha256')
    for index, inv in enumerate(rows(review.get('investigations'))):
        binding, raw = obj(inv.get('media_binding')), obj(inv.get('raw_reply'))
        if (inv.get('role') != role or raw.get('status') != 'observed' or not media_sha
                or binding.get('source_sha256') != media_sha or not binding.get('files')):
            continue
        previous_sha=obj(review.get('boundary_evidence')).get('previous_media_sha256')
        if role=='visual' and previous_sha and binding.get('previous_media_sha256')!=previous_sha:
            continue
        for resolution in rows(raw.get(field)):
            if not isinstance(resolution, dict) or resolution.get(id_field) != finding['id'] or resolution.get('status') != status or not resolution.get('observation'):
                continue
            times = intervals(resolution.get('time_s'))
            required = intervals(finding.get('time_s'))
            start, end = binding.get('start_s'), binding.get('end_s')
            if (not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not times
                    or not all(start <= a <= b <= end for a, b in times)
                    or required and not all(any(a <= x and y <= b for a, b in times) for x, y in required)):
                continue
            return {'evidence_ref': f'investigation:{index+1}', 'resolution': resolution}
    return None


def issue_view(review):
    from . import applicability
    from .collaborative_review import AV_CHECKS
    from .review import PERCEPTUAL_CHECKS
    panel, decision = obj(review.get('opinions')), obj(review.get('feedback_decision'))
    refs = obj(review.get('evidence_catalog'))
    dispositions = {x.get('finding_id'): x for x in rows(decision.get('finding_dispositions')) if isinstance(x, dict) and isinstance(x.get('finding_id'),str)}
    uncertainty_decisions = {x.get('uncertainty_id'): x for x in rows(decision.get('uncertainty_dispositions')) if isinstance(x, dict) and isinstance(x.get('uncertainty_id'),str)}
    context = obj(review.get('review_context'))
    ledger = []
    for role in ('visual', 'audiovisual'):
        p = obj(panel.get(role))
        if p.get('status') != 'observed':
            ledger.append({'id':role+':observer_unavailable', 'kind':'missing_evidence', 'state':'open', 'role':role})
        actual_findings = rows(p.get('findings')) + (applicability.check_findings(review,role) if applicability.enabled(review) else [])
        for finding in actual_findings:
            if not isinstance(finding, dict) or not isinstance(finding.get('id'),str) or not finding['id']:
                ledger.append({'id':role+':malformed_finding', 'kind':'missing_evidence','state':'open','role':role})
                continue
            d = obj(dispositions.get(finding['id']))
            state, resolution = 'open', None
            refutation = reobserved(review, role, finding, 'refuted')
            confirmation = reobserved(review, role, finding, 'confirmed')
            grounded = bool(d.get('reason')) and cited(d, refs, role)
            if grounded and d.get('disposition') == 'not_a_defect' and not confirmation and (refutation or finding.get('severity') != 'major'):
                state, resolution = 'refuted', refutation or d
            elif grounded and d.get('disposition') == 'accepted_tradeoff' and finding.get('severity') == 'minor' and not (finding.get('category')=='unexpected_music' and context.get('background_music')=='none'):
                state, resolution = 'accepted_tradeoff', d
            ledger.append({'id':finding['id'],'role':role,'kind':'content_defect','state':state,
                'finding':finding,'disposition':d,'resolution':resolution,'confirmed':bool(confirmation)})
        for i, uncertainty in enumerate(rows(p.get('uncertainties'))):
            ident = f'{role}:uncertainty:{i+1}'; d = obj(uncertainty_decisions.get(ident))
            code = d.get('scope_code')
            allowed = (code == 'audio_unavailable' and role == 'visual'
                or code == 'source_withheld' and context.get('source_and_caption_withheld') is True and d.get('generic_method_limit') is True
                or code == 'no_previous' and context.get('previous_available') is False and role == 'visual'
                or code == 'sampling_limit' and role == 'visual' and d.get('generic_method_limit') is True)
            linked_findings={f.get('id') for f in rows(p.get('findings')) if isinstance(f,dict) and isinstance(f.get('id'),str)}
            detail=obj(uncertainty)
            concrete = (detail.get('kind') in ('evidence_gap','media_question','defect')
                or detail.get('category') in ('identity_continuity','lip_sync','speaker_binding','transient_artifacts')
                or bool(detail.get('related_finding_ids')) or bool(d.get('related_finding_ids')))
            # The prompt's concrete-only convention is guidance, not evidence about
            # what this actual item says. A valid method limit may be misplaced here.
            # Reclassifying that item never dismisses a concrete finding/check.
            if concrete or any(ref in linked_findings for ref in rows(d.get('evidence_refs')) if isinstance(ref,str)):
                allowed=False
            scope_limit = allowed and d.get('kind') == 'scope_limit' and bool(d.get('reason')) and cited(d, refs, role)
            # A generic method limit never dismisses a concrete finding or an unknown check.
            resolved = (d.get('kind')=='resolved' and bool(d.get('reason')) and cited(d,refs,role)
                and reobserved(review,role,{'id':ident,'time_s':obj(uncertainty).get('time_s')},'resolved','uncertainty_resolutions','uncertainty_id'))
            ledger.append({'id':ident,'role':role,'kind':'scope_limit' if scope_limit else 'missing_evidence',
                'state':'resolved_by_investigation' if resolved else 'declared_limit' if scope_limit else 'open','observation':uncertainty,'disposition':d,'resolution':resolved or None})
        base_checks = PERCEPTUAL_CHECKS if role == 'visual' else AV_CHECKS
        for category in (applicability.checks(review,role,base_checks) if applicability.enabled(review) else base_checks):
            field = 'perceptual_checks' if role == 'visual' else 'alignment_checks'
            c = next((x for x in rows(p.get(field)) if isinstance(x, dict) and x.get('category') == category), {})
            status = c.get('status'); valid = status in ('clear', 'defect') and bool(c.get('observation'))
            if applicability.enabled(review):
                valid = applicability.assess(review,role,category,c)['valid']
                if not valid:
                    ledger.append({'id':f'{role}:unknown:{category}','role':role,'kind':'missing_evidence',
                        'state':'open','category':category,'observation':c})
                continue
            if context.get('evidence_policy') == 'observed_facts_v1' and status in ('not_observed', 'outside_observation') and c.get('observation'):
                valid = True
            if status == 'not_applicable' and c.get('observation'):
                if category in ('lip_sync','speaker_binding','speech_intelligibility'):
                    valid = (obj(p.get('voice_observation')).get('status') == 'ABSENT'
                        and not any(obj(t).get('text') for t in rows(p.get('transcript')))
                        and context.get('target_has_speech') is False)
                    # Breathing/grunts are vocal sound, not necessarily lexical speech.
                    # Only intelligibility gets this exemption; mouth/speaker binding
                    # still requires its own actual check when nonverbal voice exists.
                    if category == 'speech_intelligibility' and context.get('decision_contract_version') == 'grounded_dispositions_v1':
                        valid = (p.get('status') == 'observed' and context.get('target_has_speech') is False
                            and isinstance(p.get('transcript'), list) and not any(obj(t).get('text') for t in p['transcript'])
                            and obj(p.get('voice_observation')).get('status') in ('ABSENT','PRESENT')
                            and obj(p.get('speech_observation')).get('status') not in ('PRESENT','UNKNOWN'))
                elif category == 'sound_event_alignment':
                    sound_applicability = next((x for x in rows(decision.get('applicability')) if isinstance(x,dict) and x.get('category') == category), {})
                    valid = (sound_applicability.get('required_by_target') is False
                        and sound_applicability.get('observed_discrete_event') is False and bool(sound_applicability.get('reason'))
                        and cited(sound_applicability, refs, 'audiovisual'))
            if not valid:
                ledger.append({'id':f'{role}:unknown:{category}','role':role,'kind':'missing_evidence','state':'open','category':category,'observation':c})
    if applicability.enabled(review):
        ledger.extend(applicability.fact_issues(review))
    coverage = obj(decision.get('target_coverage'))
    speech_valid = coverage.get('scripted_speech') == 'covered' or coverage.get('scripted_speech') == 'not_applicable' and context.get('target_has_speech') is False
    if coverage.get('current_event') != 'covered' or not speech_valid or coverage.get('progression') != 'new_event':
        ledger.append({'id':'target:coverage','kind':'target_or_progression_unverified','state':'open','observation':coverage})
    for i, conflict in enumerate(rows(review.get('content_conflicts'))):
        ledger.append({'id':f'target:conflict:{i+1}','kind':'content_conflict','state':'open','observation':conflict})
    for i, finding in enumerate(rows(decision.get('additional_findings'))):
        if isinstance(finding,dict):
            ledger.append({'id':f'director:new:{i+1}','kind':'additional_grounded_finding' if cited(finding,refs,finding.get('role')) else 'unsupported_additional_finding',
                'state':'open','finding':finding})
    # Malformed/missing citations are explicit, even if all observers happen to report clear.
    known_ids = {f.get('id') for p in panel.values() if isinstance(p,dict) for f in rows(p.get('findings')) if isinstance(f,dict) and isinstance(f.get('id'),str)}
    if applicability.enabled(review):
        known_ids.update(f['id'] for role in ('visual','audiovisual') for f in applicability.check_findings(review,role))
    for ident,d in dispositions.items():
        if ident not in known_ids:
            # Extra explanations cannot dismiss any original issue. Keep them as annotations;
            # a newly identified defect still enters the open ledger. Do not block on an invented label alone.
            new_defect=d.get('disposition') in ('fix_local','unresolved')
            ledger.append({'id':str(ident)+':annotation','kind':'director_detected_issue' if new_defect else 'decision_annotation',
                'state':'open' if new_defect else 'annotation_only','observation':d,'citations_valid':cited(d,refs)})
        elif not cited(d,refs):
            ledger.append({'id':str(ident)+':citation','kind':'unsupported_disposition','state':'open','observation':d})
    return ledger


def quality(review):
    from . import applicability
    from .decision_contract import errors
    decision = obj(review.get('feedback_decision')); ledger = issue_view(review)
    outstanding = [x for x in ledger if x['state'] == 'open']
    decision_errors = errors(review)
    action = decision.get('decision')
    complete = all(obj(obj(review.get('opinions')).get(r)).get('status') == 'observed' for r in ('visual','audiovisual')) and action in ('accept','revise','keep_needs_review') and bool(decision.get('reason'))
    ok = complete and action == 'accept' and not outstanding and not decision_errors
    value = {'ok':bool(ok),'complete':bool(complete),'rank':None,'quality_status':'OK' if ok else 'draft/needs_review',
        'review_system':'feedback_v1','acceptance_basis':VERSION,'feedback_decision':action,
        'decision_consistent':not (action == 'accept' and (outstanding or decision_errors)),
        'unresolved_findings':outstanding,'issue_ledger':ledger,
        'retained_tradeoffs':[x for x in ledger if x['state']=='accepted_tradeoff'], 'not_applicable_dimensions':[]}
    if obj(review.get('review_context')).get('decision_contract_version') == 'grounded_dispositions_v1':
        value['decision_errors'] = decision_errors
    if applicability.enabled(review):
        value['applicability_audit'] = applicability.audit(review)
        value['not_applicable_checks'] = [x['category'] for x in value['applicability_audit'] if x['valid'] and x['source_status']=='not_applicable']
    from .speech_contract import apply_quality
    return apply_quality(review, value)


def capabilities(candidates, take=None, *, window_start=1, max_takes=6):
    used = {c['take'] for c in candidates if c.get('receipt') or c.get('technical_failure') or c.get('caption')}
    active = used.intersection(range(window_start, window_start + max_takes))
    return {'max_real_takes':max_takes,'consumed_or_reserved_slots':sorted(active),
        'remaining_unreserved_slots':max(0,max_takes-len(used if max_takes<6 else active)), 'current_slot':take,
        'active_slot_range':[window_start, window_start+max_takes-1], 'lifetime_consumed_or_reserved_slots':sorted(used),
        'remaining_after_current_slot':max(0,window_start+max_takes-1-take) if take is not None else None,
        'all_previous_candidates_retained':True,'can_select_any_retained_usable_candidate':True,
        'usable_takes':[c['take'] for c in candidates if c.get('receipt') and c.get('review')],
        'postproduction_edit_executor':False,
        'instruction':'Rendering a later take does not delete or prevent selecting earlier media. The configured total is a ceiling, not a target.'}


def comparison_credit(parent, current, comparison, absolute_quality):
    distinct = parent['receipt']['sha256'] != current['receipt']['sha256']
    evidence = bool(comparison.get('evidence')) and distinct
    gains = bool(comparison.get('resolved_findings')) and evidence
    regressions = bool(comparison.get('regressions')) and evidence
    outcome = comparison.get('outcome')
    if outcome == 'regressed' and regressions:
        credit = 'opinion_supported_regression'
    elif gains and (regressions or outcome == 'mixed'):
        credit = 'opinion_supported_mixed_change'
    elif gains and outcome == 'improved' and not regressions:
        credit = 'opinion_supported_media_improvement' if absolute_quality['ok'] else 'opinion_supported_partial_improvement'
    else:
        credit = 'unconfirmed_hypothesis'
    return {'evidence':credit,'model_comparison_outcome':outcome, 'relative_outcome':'mixed' if gains and regressions else outcome if outcome in ('improved','mixed','unchanged','regressed','unknown') else 'unknown',
        'absolute_quality_status':absolute_quality['quality_status'],'remaining_issues':absolute_quality.get('unresolved_findings',[]),
        'distinct_media':distinct,'success_template':False,'causal_claim':'none; retained model comparisons, not independent or single-factor proof'}


def bind_review_protocol(store, cid, candidate, requested, evidence_policy=None):
    """An upgrade cannot rename an already submitted review or evade an unknown POST."""
    from .storage import Pending, key
    if candidate.get('review_protocol') and (not evidence_policy or candidate.get('review_policy_version')):
        return candidate['review_protocol']
    found=[]
    for protocol in ('content_rsi_v3', VERSION):
        base=f'review_panel/{cid}/{candidate["take"]}/{protocol}'
        if any((store.run/'calls'/key(base+'/'+role+'/format_1')/'request.json').exists()
               or (store.run/'artifacts'/(base+'/'+role+'.json')).exists() for role in ('visual','audiovisual')):
            found.append(protocol)
    if len(found)>1:
        raise Pending('Multiple original review protocols exist; reconcile retained requests before choosing a route')
    candidate.setdefault('review_protocol',found[0] if found else requested)
    if evidence_policy and not candidate.get('review_policy_version'):
        # A candidate that already owns any old review input must finish its
        # original observer/director operation, including after an interruption.
        candidate['review_policy_version']='legacy' if found or candidate.get('review') else evidence_policy
    store.save()
    return candidate['review_protocol']
