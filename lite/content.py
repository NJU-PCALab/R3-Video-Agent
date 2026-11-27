from . import duration_budget
"""Small content invariants: stable story events, faithful captions, future-only planning."""
from copy import deepcopy
import difflib
import json
import re

from .storage import clips, key, write

PROTOCOL = 'content_rsi_v3'

def closed_loop_enabled(config):
    return config.get('review_system', {}).get('observation_protocol') == 'content_rsi_v4'


def enabled(config):
    return config.get('review_system', {}).get('observation_protocol') in (PROTOCOL, 'content_rsi_v4')


def normal(text):
    return ''.join(re.findall(r'[\w\u4e00-\u9fff]+', str(text).lower()))


def dialogue(clip):
    return [d for d in clip.get('dialogue', []) if isinstance(d, dict) and isinstance(d.get('text'), str)]


def event_signature(clip):
    return {'scene': normal(clip.get('scene', '')), 'action': normal(clip.get('action', '')),
            'speech': [(normal(d.get('speaker')), normal(d['text'])) for d in dialogue(clip)]}


def bind_events(items, previous=()):
    result = deepcopy(items)
    for c in result:
        sig = event_signature(c)
        match = next((p for p in previous if event_signature(p) == sig), None)
        if match is None:
            match = next((p for p in previous if c.get('title') and c.get('title') == p.get('title')
                and c.get('scene') == p.get('scene') and difflib.SequenceMatcher(
                    None, normal(c['action']), normal(p['action']), autojunk=False).ratio() > .75), None)
        # Supplied ids are advisory. The host owns the binding, without asking the model to echo hashes.
        c['event_id'] = match.get('event_id') if match and match.get('event_id') else 'event_' + key(sig)
    return result


def repeated_events(proposed, prefix=()):
    seen, issues = list(prefix), []
    for c in proposed:
        for old in seen:
            speech = {normal(d['text']) for d in dialogue(c) if len(normal(d['text'])) >= 8}
            repeated_speech = speech & {normal(d['text']) for d in dialogue(old)}
            same_action = normal(c.get('action')) == normal(old.get('action')) and bool(normal(c.get('action')))
            same_event = c.get('event_id') and c.get('event_id') == old.get('event_id')
            if same_action or same_event or repeated_speech:
                issues.append({'kind': 'replayed_story_event', 'clip_id': c.get('id'),
                    'previous_clip_id': old.get('id'), 'repeated_speech': sorted(repeated_speech),
                    'reason': 'An already scheduled/selected occurrence cannot be repaired by appending the same event.'})
                break
        seen.append(c)
    return issues


def caption_issues(clip, caption, selected=()):
    issues, text = [], normal(caption)
    required = [normal(d['text']) for d in dialogue(clip)]
    for line in required:
        if line and line not in text:
            issues.append({'kind': 'required_speech_missing', 'speech': line})
    tagged = re.findall(r'<d>(.*?)</d>', str(caption), flags=re.S | re.I)
    for line in tagged:
        spoken = normal(re.sub(r'\[[^\]]*\]', '', line))
        if spoken and spoken not in required:
            issues.append({'kind': 'borrowed_or_extra_speech', 'speech': spoken})
    for s in selected:
        for d in dialogue(s['clip']):
            line = normal(d['text'])
            if len(line) >= 8 and line in text and line not in required:
                issues.append({'kind': 'replayed_prefix_speech', 'previous_clip_id': s['clip']['id'], 'speech': line})
    return issues


def speech_units(text):
    """Keep Chinese character timing; use words for space-delimited English."""
    if re.search(r'[\u3400-\u9fff]', text) or not re.search(r'[A-Za-z]', text):
        return 'characters', len(normal(text))
    return 'words', len(re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*", text))


def speech_budget(clip, references):
    """Timing guidance, not a quality score or a reason to stop a film."""
    rows = []
    for d in dialogue(clip):
        ref = next((r for r in references if r.get('speaker') == d.get('speaker') and r.get('text') and r.get('seconds')), None)
        unit, size = speech_units(d['text'])
        low, high, default = (1.5, 3.5, 2.5) if unit == 'words' else (2.5, 5.5, 4.5)
        compatible = ref and speech_units(ref['text'])[0] == unit
        rate = speech_units(ref['text'])[1] / float(ref['seconds']) if compatible else default
        rows.append({'speaker': d.get('speaker'), 'text': d['text'],
            'estimated_speech_seconds': round(size / max(low, min(high, rate)), 2),
            'timing_unit': unit,
            'basis': 'actual_timbre_sample_rate_guidance' if compatible else 'planning_estimate_not_forced_alignment'})
    return {'utterances': rows, 'estimated_speech_seconds': round(sum(x['estimated_speech_seconds'] for x in rows), 2),
            'instruction': 'Reserve action/pauses and editing handles; speech and visible performance must share one timeline. Do not delay all speech until second 4 of a 10-second clip.'}


def prefix_facts(selected):
    return [{'clip_id': s['clip']['id'], 'event_id': s['clip'].get('event_id'),
        'source_event_is_only_target': s['clip'], 'media_sha256': s['receipt']['sha256'],
        'observed_events': s['review'].get('observed_events', []),
        'observed_speech': s['review'].get('transcript', []),
        'evidence_credit': 'model_observations_of_selected_media_not_independent_truth',
        'local_status': s['quality']['quality_status']} for s in selected]


def correct_local_plan(agent, clip, plan, operation):
    from . import prompts
    issues = caption_issues(clip, plan.get('caption', ''), agent.s.data['selected'])
    if not issues:
        return plan
    agent.memory.record('local', operation + '/content_conflict', 'planning_content_conflict',
        {'clip': clip, 'issues': issues, 'original_plan': plan, 'not_a_media_improvement': True})
    revised = agent.artifact(operation + '_content_correction', lambda: agent.advisory(
        operation + '/content_correction', agent.prompt(prompts.LOCAL_V3),
        {'current_source_event': clip, 'original_plan': plan, 'precise_content_conflicts': issues,
         'actual_asset_catalog': agent.catalog(), 'h3_guide': agent.guide,
         'instruction': 'Correct only this current event; never substitute the next event or repeat a selected event.'}, max_tokens=10000))
    remaining = caption_issues(clip, revised.get('caption', ''), agent.s.data['selected'])
    if not remaining:
        return revised
    # Use the retained director's actual target, not an invented replacement or false PASS.
    agent.s.warning(operation, 'Enhanced caption remains inconsistent; using retained model-authored source event for real generation/review')
    return {**plan, 'caption': json.dumps(clip, ensure_ascii=False), 'seconds': clip['seconds'],
            'content_fallback': {'reason': remaining, 'source': 'retained_model_director_event', 'not_quality_pass': True}}


def global_checkpoint(agent):
    from . import prompts
    s = agent.s
    count = len(s.data['selected'])
    if not count or count % 3 or count in s.data['checkpoints']:
        return
    suffix = s.data['plan']['clips'][count:]
    if not suffix:
        s.data['checkpoints'].append(count); s.save()
        s.event('final_checkpoint_archived', selected_clips=count, whole_film_review='not_requested')
        return
    prefix = [row['clip'] for row in s.data['selected']]
    payload = {'source': agent.source, 'narrative_prefix': prefix_facts(s.data['selected']),
        'planned_suffix': suffix, 'actual_assets': agent.catalog(),
        'unavailable_assets': agent.registry.unavailable(),
        'previous_global_lessons': [{'selected_count': x['selected_count'], 'lessons': x.get('lessons', [])}
                                    for x in s.data['global_lessons'][-3:]],
        'retrieved_experience': agent.memory.retrieve('global', {'story': agent.source, 'planned_suffix': suffix,
            'purpose': 'causal completeness continuity no repeated events'}, f'global/{count}'),
        'asset_action_receipts':list(s.data.get('asset_action_receipts',{}).values()),
        'actual_prefix_seconds': sum(x['receipt']['seconds'] for x in s.data['selected']),
        'target_total_seconds': agent.config.get('target_seconds', 240),
        'scope': 'Future narrative continuity/completeness/causality and future assets only. Local defects remain owned by local RSI, never repaired or closed by appending a replay.'}
    result = agent.artifact(f'global_{count}', lambda: agent.advisory(f'global/{count}',
        agent.prompt(prompts.GLOBAL_V3), payload, max_tokens=24000))
    changes, issues = None, []
    if result.get('decision') == 'revise':
        try:
            changes = bind_events(clips(result.get('suffix'), start=count), suffix)
            issues = repeated_events(changes, prefix) + duration_budget.issues(agent.config,changes,s.data['selected'])
        except (ValueError, TypeError) as exc:
            issues = [{'kind': 'unusable_suffix_shape', 'reason': str(exc)}]
        if issues:
            original = result
            result = agent.artifact(f'global_{count}_content_correction', lambda: agent.advisory(
                f'global/{count}/content_correction', agent.prompt(prompts.GLOBAL_V3),
                {**payload, 'original_proposal': original, 'precise_conflicts': issues,
                 'instruction': 'Return a complete future suffix without replaying prefix events. Do not repair any selected clip.'}, max_tokens=24000))
            try:
                changes = bind_events(clips(result.get('suffix'), start=count), suffix) if result.get('decision') == 'revise' else None
                issues = repeated_events(changes, prefix) + duration_budget.issues(agent.config,changes,s.data['selected']) if changes else []
            except (ValueError, TypeError) as exc:
                issues = [{'kind': 'unusable_suffix_shape', 'reason': str(exc)}]
            agent.memory.record('global', f'global/{count}/content_conflict', 'global_proposal_conflict',
                {'issues': issues, 'original_proposal': original, 'correction': result,
                 'scope': 'proposal only; no prefix defect resolved'})
        if changes and not issues:
            s.data['plan']['clips'] = deepcopy(prefix) + changes
            s.data['plan_revision'] += 1
            if not closed_loop_enabled(agent.config):
                agent.asset_updates(result.get('asset_updates', []))
            write(s.run / 'plans' / f'revision_{s.data["plan_revision"]}.json', s.data['plan'])
            s.event('global_revision_applied', selected_prefix_frozen=count, suffix_clips=len(changes),
                plan_revision=s.data['plan_revision'], scope='future_narrative_only', analysis=result.get('analysis'))
        elif issues:
            s.warning(f'global/{count}', 'Retaining prior future plan; new proposal repeats selected events or is unusable: ' + json.dumps(issues, ensure_ascii=False))
    if closed_loop_enabled(agent.config):
        agent.asset_updates(result.get('asset_updates', []), operation=f'global/{count}/assets')
    s.data['global_lessons'].append({'selected_count': count, 'lessons': result.get('lessons', []),
        'observed_gaps': result.get('observed_gaps'), 'scope': 'future_narrative_only',
        'evidence': 'unvalidated_global_proposal_not_prefix_repair', 'proposal_conflicts': issues})
    agent.memory.record('global', f'global/{count}/proposal', 'unvalidated_global_proposal',
        {'selected_prefix': count, 'proposal': result, 'proposal_conflicts': issues,
         'scope': 'future_narrative_only; selected defects stay open', 'plan_revision': s.data['plan_revision']})
    s.data['checkpoints'].append(count); s.save(); agent.ensure_assets()
