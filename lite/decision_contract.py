"""Bounded correction of report contradictions; never synthesize a media PASS."""
from copy import deepcopy
import hashlib
import json

from .storage import Pending, read, write

VERSION = 'grounded_dispositions_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def enabled(config):
    return config.get('review_system', {}).get('decision_contract') == VERSION


def validate_applicability(value, *, require_sound_event=False):
    """Reject prose in executable predicates; never infer a boolean from words."""
    rows = value.get('applicability', [])
    if not isinstance(rows, list):
        raise ValueError('applicability must be an array')
    sound = [r for r in rows if isinstance(r, dict) and r.get('category') == 'sound_event_alignment']
    if len(sound) > 1 or (require_sound_event and len(sound) != 1):
        raise ValueError('not_applicable sound_event_alignment needs one explicit applicability row')
    for row in sound:
        for field in ('required_by_target', 'observed_discrete_event'):
            if type(row.get(field)) is not bool:
                raise ValueError('sound_event_alignment.' + field + ' must be a JSON boolean true/false, not prose; retain actual evidence in reason')
    return value


def needs_sound_applicability(review):
    return any(isinstance(c, dict) and c.get('category') == 'sound_event_alignment'
               and c.get('status') == 'not_applicable'
               for c in (review.get('opinions', {}).get('audiovisual', {}).get('alignment_checks') or []))


def errors(review):
    if review.get('review_context', {}).get('decision_contract_version') != VERSION:
        return []
    d = review.get('feedback_decision') or {}
    if d.get('decision') != 'accept':
        return []
    result = []
    for row in d.get('finding_dispositions') or []:
        if isinstance(row, dict) and row.get('disposition') in ('fix_local', 'unresolved'):
            result.append({'kind': 'accept_with_repair_disposition', 'finding_id': row.get('finding_id'),
                           'disposition': row['disposition'], 'scope': 'report_error_not_a_new_media_defect'})
    plan = d.get('revision_plan') or {}
    if isinstance(plan, dict) and any(plan.get(k) for k in ('caption_changes', 'reference_changes', 'duration_changes', 'asset_changes')):
        result.append({'kind': 'accept_with_revision_plan', 'scope': 'report_error_not_a_new_media_defect'})
    return result


def reconcile(agent, cid, candidate, review):
    """One cached text operation, bound to exact original review and media bytes."""
    from .collaborative_review import quality
    from .closed_loop import issue_view
    from . import prompts, evidence_policy, applicability
    original = deepcopy(review)
    value = deepcopy(review)
    value.setdefault('review_context', {})['decision_contract_version'] = VERSION
    q = quality(value)
    problems = errors(value)
    if not problems and q['decision_consistent']:
        value['issue_ledger'] = issue_view(value)
        return value
    name = f'review_panel/{cid}/{candidate["take"]}/{VERSION}/decision_correction'
    binding = {'review_sha256': digest(original), 'media_sha256': candidate['receipt']['sha256'],
               'audio_sha256': candidate['receipt'].get('audio_sha256'), 'clip_id': cid, 'take': candidate['take']}
    path = agent.s.run / 'review_media' / cid / str(candidate['take']) / VERSION / 'binding.json'
    previous = read(path)
    if previous is not None and previous != binding:
        raise Pending('Decision correction binding changed; reconcile the original text receipt')
    if previous is None:
        write(path, binding)
    system = prompts.OBSERVED_DECISION + prompts.DECISION_CONTRACT + (prompts.APPLICABILITY_DECISION if applicability.enabled(value) else '')
    payload = {'current_source_event': candidate['source_clip'], 'original_review': evidence_policy.memory_view(original),
         'actual_media_facts': original.get('actual_media_facts') or evidence_policy.media_facts(candidate),
         'binding': binding, 'host_discrepancies': problems, 'host_quality': q,
         'instruction': 'Correct the structured decision from these SAME original observations. Do not request media or generation. '
                        'Historical repaired defects are not current fix_local issues. If evidence remains open, retain revise/keep_needs_review.'}
    def validate(d):
        validate_applicability(d, require_sound_event=applicability.enabled(value) and needs_sound_applicability(value))
        # This is still the same bounded correction operation. A valid boolean
        # alone cannot make a known required/observed event's missing check clear.
        sound_gap = applicability.enabled(value) and any(x.get('id') == 'audiovisual:unknown:sound_event_alignment' for x in q.get('unresolved_findings', []))
        original_row = next((x for x in (original.get('feedback_decision', {}).get('applicability') or []) if isinstance(x, dict) and x.get('category') == 'sound_event_alignment'), {})
        if sound_gap and any(original_row.get(k) is True for k in ('required_by_target', 'observed_discrete_event')):
            proposed = next((x for x in (d.get('applicability') or []) if isinstance(x, dict) and x.get('category') == 'sound_event_alignment'), {})
            if any(original_row.get(k) is True and proposed.get(k) is not True for k in ('required_by_target', 'observed_discrete_event')):
                raise ValueError('Retain the original true sound-event predicates; text correction cannot erase the existing observation gap')
            if d.get('decision') == 'accept':
                raise ValueError('accept contradicts the retained sound_event_alignment evidence gap; use revise or keep_needs_review with the gap preserved, not new media or invented observations')
        return d
    corrected = agent.artifact(name, lambda: agent.advisory(name, system, payload, max_tokens=7000, validate=validate))
    # A retained raw artifact may predate this validator. Re-enter the SAME text
    # operation: format_1 reuses its original answer, then bounded format_2/3
    # correct structure. Never rename/re-POST it or rewrite the paid raw reply.
    try:
        validate(corrected)
    except (ValueError, TypeError, KeyError):
        recovery_path = path.parent / 'typed_structure_recovery.json'
        recovery = read(recovery_path)
        raw_digest = digest(corrected)
        if recovery is not None:
            if recovery.get('binding') != binding or recovery.get('original_correction_sha256') != raw_digest:
                raise Pending('Typed correction recovery binding changed; reconcile original replies')
            corrected = validate(recovery['decision'])
        else:
            from .storage import Unavailable
            try:
                typed = agent.models.json(name, system, payload, max_tokens=7000, validate=validate)
            except Unavailable as exc:
                raise Pending('Original bounded text format recovery remains unresolved: ' + str(exc), status='needs_decision_clarification') from exc
            recovery = {'binding': binding, 'original_correction_sha256': raw_digest, 'decision': typed,
                        'same_logical_operation': name, 'original_raw_reply_unchanged': True,
                        'new_media_requested': False, 'format_slots_bounded_by_existing_three_reply_limit': True}
            write(recovery_path, recovery)
            corrected = typed
    checked, audit = evidence_policy.checked_decision(corrected, original.get('actual_media_facts') or evidence_policy.media_facts(candidate))
    value['decision_before_contract'] = deepcopy(original.get('feedback_decision'))
    value['feedback_decision'] = checked
    value['repair_suggestions'] = checked.get('revision_plan', {})
    value['decision_contract'] = {'version': VERSION, 'binding': binding, 'operation': name,
                                  'before_quality': q, 'correction_audit': audit, 'bounded_text_rounds': 1}
    after = quality(value)
    if errors(value) or not after['decision_consistent'] or not after['complete']:
        from . import applicability
        evidence_only = (applicability.enabled(value) and not errors(value) and bool(after['unresolved_findings'])
            and all(x.get('kind') in ('missing_evidence','observation_contract_error') for x in after['unresolved_findings']))
        write(path.parent / 'unresolved.json', {'binding': binding, 'raw_correction_artifact': name,
              'errors': errors(value), 'quality': after, 'new_media_requested': False,
              'classification':'observation_evidence_missing' if evidence_only else 'decision_or_content_disagreement'})
        raise Pending('Bounded decision correction remains inconsistent; retain original evidence and do not render another take',
                      status='needs_observation_clarification' if evidence_only else 'needs_decision_clarification')
    value['decision_contract']['status'] = 'consistent'
    value['issue_ledger'] = issue_view(value)
    return value
