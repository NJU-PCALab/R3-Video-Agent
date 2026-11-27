"""Bounded lessons from retained observations; no paid comparison or causal credit."""
from copy import deepcopy
from .storage import key, candidate_quality

VERSION = 'lightweight_experience_v1'
EXPERIENCE_LIMIT = 3


def enabled(config):
    return config.get('completion_policy', {}).get('experience_loop') == VERSION


def export(agent, operation, outcome, lesson, stage='shared'):
    try:
        # Keep source context in state as well as the optional exported episode.
        details = deepcopy(lesson)
        details.setdefault('source_run', str(agent.s.run))
        return agent.memory.record(stage, operation, outcome, details, nonblocking=True)
    except OSError as exc:
        agent.s.warning(operation, 'Optional experience export deferred; authoritative state retained: ' + str(exc))
        return None


def recall(agent, stage, query, operation):
    cache = agent.s.data.setdefault('completion_experience_consumption', {})
    if operation not in cache:
        try:
            episodes = agent.memory.retrieve(stage, query, operation, limit=EXPERIENCE_LIMIT,
                include_stages=('local', 'global', 'asset', 'shared'))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            agent.s.warning(operation, 'Optional experience retrieval unavailable: ' + str(exc))
            episodes = []
        # State is the authoritative fallback when the optional export was unavailable.
        recent = []
        try:
            corrections=agent.memory._corrections()
            for lesson in agent.s.data.get('local_lessons', [])[-2:]:
                episode = {'id':key(lesson),'stage':'shared','source_run':lesson.get('source_run', str(agent.s.run)),
                    'context': lesson.get('memory_context', {}), 'outcome':'observed_candidate_change','details':lesson}
                if not agent.memory.exclusion(episode):
                    viewed=agent.memory._evidence_view(episode,corrections)
                    recent.append(viewed['details'])
        except (OSError,ValueError,KeyError,TypeError) as exc:
            agent.s.warning(operation,'Recent lesson advice withheld pending correction evidence: '+str(exc))
            episodes=[]
        cache[operation] = {'episodes': episodes, 'recent_in_run_lessons': deepcopy(recent),
            'use': 'Advice only; preserve exact script, identity, budget and unresolved defects.'}
        agent.s.save()
    result = deepcopy(cache[operation])
    result['episodes'] = agent.memory.eligible(result.get('episodes', []), operation)[:EXPERIENCE_LIMIT]
    available = EXPERIENCE_LIMIT - len(result['episodes'])
    result['recent_in_run_lessons'] = [lesson for lesson in result.get('recent_in_run_lessons', [])
        if not agent.memory.exclusion({'source_run': lesson.get('source_run', str(agent.s.run)),
            'context': lesson.get('memory_context', {}), 'details': lesson})][:available]
    result['scope'] = agent.config.get('memory_policy', {}).get('scope', 'within_story')
    return result


def _findings(candidate):
    return candidate_quality(candidate).get('unresolved_findings', [])


def _category(finding):
    return finding.get('category') or finding.get('finding', {}).get('category')


def _clear(review, category):
    checks = [o.get('checks', {}).get(category) for o in review.get('opinions', {}).values() if isinstance(o, dict)]
    return any(isinstance(c, dict) and c.get('status') == 'clear' and c.get('observation') for c in checks)


def _protocol(candidate):
    review = candidate.get('review') or {}
    return review.get('observation_protocol') or candidate.get('observation_protocol') or review.get('completion_review_version')


def learn(agent, cid, candidates, current):
    lessons = agent.s.data.setdefault('local_lessons', [])
    operation = f'completion/lesson/{cid}/{current["take"]}'
    saved = next((x for x in lessons if x.get('operation') == operation), None)
    if saved is None:
        parents = [c for c in candidates if c['take'] < current['take'] and c.get('review') and c.get('receipt')]
        parent = max(parents, key=lambda c: c['take']) if parents else None
        problems, remaining = _findings(parent) if parent else [], _findings(current)
        comparable = bool(parent and parent['receipt']['sha256'] != current['receipt']['sha256']
            and _protocol(parent) and _protocol(parent) == _protocol(current)
            and parent.get('generation_policy_version') == current.get('generation_policy_version'))
        unresolved = {_category(f) for f in remaining}
        resolved = [deepcopy(f) for f in problems if comparable and _category(f)
            and _category(f) not in unresolved and _clear(current['review'], _category(f))]
        major = any(f.get('severity', f.get('finding', {}).get('severity')) in ('major', 'critical') for f in remaining)
        before = parent or {}
        changes = {name: {'before': deepcopy(before.get(name)), 'after': deepcopy(current.get(name))}
            for name in ('caption', 'references') if parent and before.get(name) != current.get(name)}
        # Avoid recursively embedding old reviews and retrieved memory into lessons.
        intervention = deepcopy(current.get('intervention')) if isinstance(current.get('intervention'), dict) else {
            'kind': 'initial_generation' if not parent else 'legacy_change_execution_unverified' if changes else 'resample',
            'execution_verified': False, 'reason': current.get('plan', {}).get('reason'),
            'changed_components': list(changes), 'render_spec': current.get('plan', {}).get('render_spec'),
            'reference_ids': [r.get('id') for r in current.get('references', [])]}
        intervention_id = intervention.get('intervention_id') or key({'operation': operation, 'intervention': intervention})
        outcome_id = key({'operation': operation, 'media': current['receipt']['sha256'],
                          'protocol': _protocol(current)})
        saved = {'operation': operation, 'clip_id': cid, 'parent_take': parent['take'] if parent else None,
            'source_run': str(agent.s.run), 'memory_context': agent.memory.context(),
            'context': {'clip': deepcopy(current.get('source_clip')), 'parent_media_sha256': before.get('receipt', {}).get('sha256')},
            'feedback': deepcopy(problems), 'intervention_id': intervention_id, 'outcome_id': outcome_id,
            'take': current['take'], 'problem': deepcopy(problems), 'intervention': intervention,
            'relative_outcome': 'observation_supported_improvement' if resolved and not major else 'unconfirmed',
            'resolved_in_observations': resolved, 'remaining_issues': deepcopy(remaining),
            'absolute_quality_status': candidate_quality(current)['quality_status'],
            'parent_media_sha256': parent['receipt']['sha256'] if parent else None,
            'media_sha256': current['receipt']['sha256'], 'audio_sha256': current['receipt'].get('audio_sha256'),
            'observation_protocol': _protocol(current),
            'observed_outcome': {'media_sha256': current['receipt']['sha256'], 'intervention_id': intervention_id,
                'resolved_findings': deepcopy(resolved), 'remaining_findings': deepcopy(remaining),
                'credit': 'observed_after_intervention_not_causal_proof'},
            'reuse_hypothesis': 'Consider the recorded intervention only for matching observed problems; verify the next actual media.',
            'evidence': 'existing_observations_only; comparable categories, not independent comparison or causal proof',
            'success_template': False, 'causal_claim': 'none', 'paid_comparison': False}
        lessons.append(saved)
        agent.s.save()
    export(agent, operation, 'observed_candidate_change', saved)


def global_lesson(agent, count, proposal, applied, receipts):
    lesson = {'selected_count': count, 'analysis': proposal.get('analysis'),
        'operation': f'completion/global/{count}', 'source_run': str(agent.s.run),
        'intervention_id': key({'operation': f'completion/global/{count}', 'applied': applied, 'receipts': [r['id'] for r in receipts]}),
        'intervention_kind': 'global_replan', 'memory_context': agent.memory.context(),
        'target_clip_ids': sorted({x['clip_id'] for x in applied if x.get('clip_id')} |
            {b['clip_id'] for r in receipts for b in r.get('reference_bindings', []) if b.get('clip_id')}),
        'lessons': proposal.get('lessons', []), 'applied_future_directions': applied,
        'asset_receipt_ids': [r['id'] for r in receipts],
        'asset_execution': any(r.get('executed') for r in receipts),
        'evidence': 'future_plan_and_actual_execution_receipts; no observed improvement credited',
        'causal_claim': 'none', 'success_template': False}
    previous=next((x for x in agent.s.data['global_lessons'] if x.get('selected_count')==count),None)
    if previous is None:
        agent.s.data['global_lessons'].append(lesson)
        agent.s.save()
    else:lesson=previous
    export(agent, f'completion/global/{count}/lesson', 'global_revision_and_asset_receipts', lesson)


def followup(agent, count):
    """Append observations until actual targets are seen; exports stay immutable."""
    for lesson in agent.s.data.get('global_lessons', []):
        start = lesson.get('selected_count', count)
        if start >= count:
            continue
        targets = set(lesson.get('target_clip_ids', [])) | {
            x['clip_id'] for x in lesson.get('applied_future_directions', []) if x.get('clip_id')}
        observations = lesson.get('followup') or []
        seen = {(r['clip_id'], r.get('media_sha256')) for r in observations}
        covered = set(lesson.get('followup_target_coverage', [])) | {
            r['clip_id'] for r in observations if r['clip_id'] in targets}
        # With no applied target, retain the original one-checkpoint contextual
        # followup. Otherwise continue only until the declared targets are seen.
        if not observations or targets - covered:
            rows = [s for s in agent.s.data['selected'][start:count]
                    if (s['clip']['id'], s['receipt']['sha256']) not in seen]
            if rows:
                observations.extend({'clip_id': s['clip']['id'], 'media_sha256': s['receipt']['sha256'],
                    'intervention_id': lesson.get('intervention_id'), 'was_direct_target': s['clip']['id'] in targets,
                    'absolute_quality_status': candidate_quality(s)['quality_status'],
                    'remaining_issues': _findings(s)} for s in rows)
                lesson['followup'] = observations
                lesson['followup_target_coverage'] = sorted(covered | {
                    s['clip']['id'] for s in rows if s['clip']['id'] in targets})
                lesson['followup_credit'] = 'later_clips_observed; different_media_context_not_controlled_before_after'
                lesson['followup_observed_through'] = count
                agent.s.save()
        # Re-entry can retry an optional export after state was saved, without
        # replacing an earlier checkpoint's immutable memory episode.
        if lesson.get('followup_observed_through') == count:
            export(agent, f'completion/global/{start}/followup/{count}',
                   'global_followup_observations', lesson)
