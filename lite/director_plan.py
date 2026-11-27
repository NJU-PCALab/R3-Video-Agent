"""Compile one explicit per-shot direction into the agent's source plan."""

import math
import re

from . import binding_contract, binding_timeline, character_stages, duration_budget, screen_text, shot_policy, task_parameters, voice_lineage
from .storage import Pending, clips


def scene_code(value):
    match = re.match(r'^\s*(S[1-9][0-9]*)\b', value) if isinstance(value, str) else None
    return match.group(1) if match else None


def source_scene_timing(source):
    pattern = r'^\s*\|\s*(S[1-9][0-9]*)\s*\|\s*(\d{1,3}):([0-5]\d)\s*[–—-]\s*(\d{1,3}):([0-5]\d)\s*\|'
    windows = []
    for match in re.finditer(pattern, source, re.MULTILINE):
        code, start_min, start_sec, end_min, end_sec = match.groups()
        start = 60 * int(start_min) + int(start_sec)
        end = 60 * int(end_min) + int(end_sec)
        windows.append((code, start, end))
    if not windows or any(code != f'S{i + 1}' or start != (windows[i - 1][2] if i else 0) or end <= start
                          for i, (code, start, end) in enumerate(windows)):
        return {}
    return {code: end - start for code, start, end in windows}


def suggested_shot_durations(scene_seconds):
    """Give the director one feasible scene schedule using whole-second shots."""
    if scene_seconds < 4:
        return []
    shot_count = max(math.ceil(scene_seconds / 10), (scene_seconds + 4) // 8)
    short, longer = divmod(scene_seconds, shot_count)
    return [short + 1] * longer + [short] * (shot_count - longer)


def longform_handoff(screenplay, config, source=''):
    """Give the director a compact, factual inventory from the current screenplay."""
    if not task_parameters.longform(config):
        return {}
    storyboard = screenplay.get('scenes')
    milestones = []
    for scene in storyboard if isinstance(storyboard, list) else ():
        if not isinstance(scene, dict):
            continue
        scene_id = next((code for field in ('scene_id', 'id', 'title') if (code := scene_code(scene.get(field)))), None)
        raw = scene.get('shots')
        shots = raw if isinstance(raw, list) else []
        durations = [next((s[field] for field in ('seconds', 'dur', 'duration') if field in s), None)
                     if isinstance(s, dict) else None for s in shots]
        seconds = sum(durations) if durations and all(
            isinstance(d, (float, int)) and not isinstance(d, bool) and 4 <= d <= 10 for d in durations
        ) else None
        milestones.append({'scene': scene_id, 'screenplay_shots': len(shots), 'screenplay_seconds': seconds})
    ids = [row['scene'] for row in milestones]
    source_timing = source_scene_timing(source)
    if sum(source_timing.values()) != config['target_seconds']:
        source_timing = {}
    minimum = task_parameters.LONGFORM_MIN_SECONDS
    known_ids = ids if None not in ids and len(ids) == len(set(ids)) else []
    return {'required_scene_order': list(source_timing) or known_ids,
            'source_scene_seconds': source_timing,
            'suggested_shot_seconds_by_scene': {scene: suggested_shot_durations(seconds)
                                                for scene, seconds in source_timing.items()},
            'screenplay_scene_inventory': milestones, 'target_seconds': config['target_seconds'],
            'duration_bounds_seconds': [minimum, config['max_final_seconds']],
            'minimum_shot_count': math.ceil(minimum / 10)}


def validate_longform(clips, config, handoff, *, require_duration=True):
    """Reject a correctly shaped excerpt before reviewing it as a complete film."""
    if not handoff.get('source_scene_seconds'):
        # Unstructured sources continue through the existing duration correction path.
        return
    summary = duration_budget.director_summary(config, clips, source_scene_ids=handoff['source_scene_seconds'])
    lower, upper = handoff['duration_bounds_seconds']
    issues = []
    if require_duration and (not summary['within_hard_bounds'] or len(clips) < handoff['minimum_shot_count']):
        issues.append(f"Returned {len(clips)} shots totaling {summary['total_seconds']:g} seconds. "
                      f"Supply the complete film: {lower:g}–{upper:g} seconds and at least "
                      f"{handoff['minimum_shot_count']} shots (each at most 10 seconds).")
        planned_seconds = {row['scene']: row['seconds'] for row in summary['by_scene']}
        timing = ', '.join(f'{scene} {planned_seconds.get(scene, 0):g}/{seconds:g}'
                           for scene, seconds in handoff['source_scene_seconds'].items())
        issues.append(f'Scene seconds (returned/source target): {timing}. '
                      'Use suggested_shot_seconds_by_scene to budget the actual story beats before returning each scene.')
    required = handoff['required_scene_order']
    if required:
        observed = []
        for clip in clips:
            scene = scene_code(clip.get('scene_id') or clip.get('scene'))
            if not observed or scene != observed[-1]:
                observed.append(scene)
        if observed != required:
            issues.append(f'Use the exact ordered scene IDs {required} in each shot. Observed {observed[:16]}.')
    if issues:
        raise ValueError(' '.join(issues))


_MATCH = r'\b((?:match(?:ing|es)?|matched to|same as|identical composition to)\s+(?:the\s+)?)'
_SOURCE_SHOT_NOUN = r'(?:baseline|reference|(?:(?:fixed|frontal|wide|medium|close)\s+)?shot|composition|framing)'
_CAMERA_BASELINE = re.compile(_MATCH + rf'(S[1-9][0-9]*)(?=\s+{_SOURCE_SHOT_NOUN}\b)', re.IGNORECASE)
_CAMERA_SHOT = re.compile(_MATCH + r'(S[1-9][0-9]*)-([0-9]+)\b', re.IGNORECASE)
_TRANSITION = re.compile(r'\b((?:into|toward)\s+)(S[1-9][0-9]*)(?=[.!?]?\s*$)', re.IGNORECASE)
_STAGE_SCENE_SUFFIX = re.compile(
    r'([,;]\s*)(S[1-9][0-9]*)(?:\s*[-–—]\s*(S[1-9][0-9]*)|\s+(onwards?))(?=[.!?]?\s*$)',
    re.IGNORECASE,
)


def describe_source_camera_references(plan, scene_ids):
    """Spell out qualified image matches and verified scene handoffs."""
    rows = plan.get('clips')
    if not scene_ids or not isinstance(rows, list):
        return plan

    def replace(match):
        lead, code, *shot = match.groups()
        if code.upper() not in scene_ids:
            return match.group()
        direction = f'{lead}scene {int(code[1:])}'
        return f'{direction}, shot {int(shot[0])}' if shot else direction

    def directions(spec, fields):
        updated = spec
        for key in fields:
            raw = spec.get(key)
            if isinstance(raw, str):
                direction = _CAMERA_SHOT.sub(replace, _CAMERA_BASELINE.sub(replace, raw))
                if direction != raw:
                    updated = {**updated, key: direction}
        return updated

    def normalize_handoff(value, next_scene):
        if not isinstance(value, str):
            return value
        return _TRANSITION.sub(lambda m: replace(m) if m.group(2).upper() == next_scene else m.group(), value)

    normalized = []
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            row = dict(row)
            for field in ('render_spec', 'shot_design'):
                spec = row.get(field)
                if isinstance(spec, dict):
                    row[field] = directions(spec, ('opening', 'camera', 'closing'))
            next_row = rows[index + 1] if index + 1 < len(rows) else None
            current_scene = scene_code(row.get('scene_id') or row.get('scene'))
            next_scene = scene_code(next_row.get('scene_id') or next_row.get('scene')) if isinstance(next_row, dict) else None
            if current_scene in scene_ids and next_scene in scene_ids and int(next_scene[1:]) == int(current_scene[1:]) + 1:
                if 'exit_state' in row:
                    row['exit_state'] = normalize_handoff(row['exit_state'], next_scene)
                beats = row.get('action_beats')
                if isinstance(beats, list):
                    row['action_beats'] = [
                        {**beat, 'visible_result': normalize_handoff(beat['visible_result'], next_scene)}
                        if isinstance(beat, dict) and 'visible_result' in beat else beat for beat in beats
                    ]
        normalized.append(row)
    return {**plan, 'clips': normalized}


def describe_stage_scene_suffixes(plan, config, scene_ids):
    """Spell out a known source schedule before the identity allocator runs."""
    table = plan.get('character_reference_stages')
    if not scene_ids or 'character_stage_allocation_receipt' in plan or not isinstance(table, (dict, list)):
        return plan
    cast = config.get('generation_contract', {}).get('cast', {})

    def scheduled(match):
        delimiter, first, last, onward = match.groups()
        if first.upper() not in scene_ids or last and (
            last.upper() not in scene_ids or int(last[1:]) < int(first[1:])
        ):
            return match[0]
        start = int(first[1:])
        if last:
            return f'{delimiter}from scene {start} through scene {int(last[1:])}'
        return f'{delimiter}scene {start} {onward}'

    def family_description(role, family):
        if not isinstance(role, str) or role not in cast:
            return family
        stages = family.get('stages') if isinstance(family, dict) else family
        if not isinstance(stages, list):
            return family
        normalized = []
        for stage in stages:
            desc = stage.get('description') if isinstance(stage, dict) else None
            if isinstance(desc, str):
                stage = {**stage, 'description': _STAGE_SCENE_SUFFIX.sub(scheduled, desc)}
            normalized.append(stage)
        return {**family, 'stages': normalized} if isinstance(family, dict) else normalized

    if isinstance(table, dict):
        table = {role: family_description(role, row) for role, row in table.items()}
    else:
        table = [family_description(row.get('canonical_role'), row) if isinstance(row, dict) else row for row in table]
    return {**plan, 'character_reference_stages': table}


def require_render_specs(value):
    rows = value.get('clips')
    if not isinstance(rows, list):
        return
    problems = []
    for position, row in enumerate(rows, start=1):
        spec = row.get('render_spec') if isinstance(row, dict) else None
        if not isinstance(spec, dict):
            problems.append(f'clip {position}: missing per-shot render_spec')
            continue
        for field in ('opening', 'action', 'camera', 'closing'):
            try:
                if not isinstance(spec.get(field), str) or not binding_contract.english_direction(spec[field]):
                    raise ValueError('must be nonempty English physical direction')
            except ValueError as exc:
                problems.append(f'clip {position} render_spec.{field}: {exc}')
    if problems:
        raise ValueError(f'{len(problems)} invalid per-shot render field(s): ' + '; '.join(problems[:8]))


def project_render_specs(plan):
    """Adapt a complete legacy shot locally. Film-wide notes cannot supply shot data."""
    rows = plan.get('clips')
    if not isinstance(rows, list):
        return plan
    result = []
    projected = []
    for position, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            result.append(raw)
            continue
        row = dict(raw)
        for field in ('dialogue', 'dialogue_timeline'):
            if field in row:
                row[field] = binding_timeline.project_visibility(row[field])
        design = row.get('shot_design')
        if 'render_spec' not in row and isinstance(design, dict):
            spec = {key: design.get(key) for key in ('opening', 'camera', 'closing')}
            spec['action'] = row.get('action')
            if all(isinstance(value, str) and value.strip() for value in spec.values()):
                row['render_spec'] = spec
                projected.append(position)
        spec = row.get('render_spec')
        if 'action' not in row and isinstance(spec, dict):
            action = spec.get('action')
            if isinstance(action, str) and action.strip():
                row['action'] = action
        result.append(row)
    normalized = {**plan, 'clips': result}
    if projected:
        normalized['host_render_spec_projection'] = {
            'clip_positions': projected,
            'fields': {'opening': 'shot_design.opening', 'action': 'action',
                       'camera': 'shot_design.camera', 'closing': 'shot_design.closing'},
        }
    return normalized


def validate(value, config, *, source_scene_ids=()):
    """Use the same source compiler for initial, revised and duration-corrected plans."""
    bound = binding_contract.enabled(config)
    value = dict(value)
    if bound:
        value = project_render_specs(describe_source_camera_references(value, source_scene_ids))
    value = shot_policy.project_missing_action_beats(value, config, director_layout=bound)
    if shot_policy.contract_enabled(config):
        value = screen_text.project_plan(value)
        value = voice_lineage.project_canonical_roles(value, config)
    if bound:
        require_render_specs(value)
    if shot_policy.contract_enabled(config):
        binding_contract.validate_source(value, config)
    value['clips'] = clips(value.get('clips'))
    value['assets'] = value.get('assets', []) if isinstance(value.get('assets', []), list) else []
    binding_contract.validate_source(value, config)
    value = describe_stage_scene_suffixes(value, config, source_scene_ids)
    try:
        value = character_stages.prepare_plan(value, config)
    except Pending as exc:
        # Reuse the existing structure-repair attempts before any assets are dispatched.
        raise ValueError(str(exc)) from exc
    return binding_contract.validate_source(value, config)
