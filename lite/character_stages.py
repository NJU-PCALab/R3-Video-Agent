"""Plan-owned same-person image stages; finite shared slots, no young fallback.

This module does not call a model. Stage images use the existing registry/image
worker. The initial table owns identity, stage order and the aggregate quota;
local/global suggestions can select declared stages, never invent a new family.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re

from .storage import Pending, sha, operation_recorded

VERSION = 'character_reference_stages_v1'
ALLOCATION_AUTHORITY = 'host_family_cap_v1'


def enabled(config):
    return config.get('character_reference_policy', {}).get('version') == VERSION


def instructions(config):
    if not enabled(config):
        return ''
    policy = config['character_reference_policy']
    budget = policy.get('max_candidates_per_character', 2)
    maximum = policy.get('max_stages_per_character', 3)
    if not _integer(maximum) or maximum > 3 or not _integer(budget):
        _fail('Character stage limits require explicit bounded configuration')
    maximum = min(maximum, budget)
    required = ('Declare a reference family for every canonical cast role, including stable characters '
        'with one stage and candidate_limit:2. The initial script/plan review must check visible aging '
        'and time jumps and assign stages explicitly; prose alone is not a stage binding. '
        if policy.get('require_stage_declarations') is True else '')
    age_guidance = ('For a visible age change, state the direction and any story-specified target age in the stage '
                    'description. Use older or elderly according to the actual story. ')
    allocation = ('The host, not the model, owns candidate allocation: retain distinct predeclared '
        'stage image IDs; a two-stage person shares two slots as 1+1, never merge stage IDs to satisfy '
        'the cap. Stable one-stage people keep at most two slots. Later age-stage image IDs are '
        'allowed children of the canonical identity anchor, not new cast roles. '
        if policy.get('allocation_authority') == ALLOCATION_AUTHORITY else '')
    modes = _role_modes(config)
    reversible = [role for role, mode in modes.items() if mode == 'reversible_appearance']
    transitions = ''
    if reversible:
        transitions = ('Only the following roles may return to an earlier declared appearance on '
            'the main timeline with an explicit character_states selection: '
            + json.dumps(reversible, ensure_ascii=False) + '. All other roles progress chronologically. ')
    elif modes:
        transitions = 'All roles progress chronologically on the main timeline. '
    return (required + allocation + transitions + age_guidance + 'For a character who visibly ages or changes life stage, declare an upfront '
        'character_reference_stages table keyed by the exact canonical cast role. Each entry has '
        'canonical_image_asset_id and stages:[{id, image_asset_id, description, candidate_limit}]. '
        'Write the character image entries in assets[].prompt in English, including the canonical image. '
        'Any optional stage.prompt is also English image direction. Keep original spoken words in dialogue. '
        'Describe the visible appearance in English. For scene timing, write e.g. from scene 1 through scene 5. '
        'Use the canonical image asset for the first stage; later stages are images of that same '
        'person derived from the canonical image, not new people. Describe age, hair, clothing and '
        'identity invariants in English. Use at most '+str(maximum)+' stages and share the original '
        +str(budget)+' image candidate slots across the whole person (normally 1 young + 1 older); '
        'do not increase or reset asset limits. For every first appearance and time jump, set '
        'clip.character_states:{canonical_role:{stage:declared_stage_id,wardrobe:English_state}}. '
        'Only stage plus short English wardrobe/appearance strings are allowed in this state; '
        'age is owned by the declared stage, never a separate overriding age field. '
        'After a time jump, later shots inherit that stage; never return to the young image by '
        'default. A finite, initially declared clip with timeline_transition:"flashback" may '
        'explicitly select an earlier already established stage for that clip only; the next '
        'mainline clip resumes the later age. Keep dialogue, speaker, order and story facts intact. '
        'Do not invent an age change from an unmarked passage of time; identify the source time '
        'jump during planning. No stage image or identity quality is guaranteed by registration.')


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _fail(message):
    raise Pending(message, status='needs_character_reference_plan')


def _integer(value, minimum=1):
    return type(value) is int and value >= minimum


def _english(value):
    return isinstance(value, str) and bool(value.strip()) and not re.search(r'[\u3400-\u9fff]|<\s*(?:Subject|Picture|Audio|Video|d)\b|\bS\d+\b', value, re.I)


def _flashback_ids(plan):
    # Host-owned lossless speech partitions retain the initial parent event's authority.
    return list(dict.fromkeys(c.get('completion_split', {}).get('parent_clip_id', c['id'])
        for c in plan.get('clips', []) if c.get('timeline_transition') == 'flashback'))


def _role_modes(config):
    modes = config.get('character_reference_policy', {}).get('role_transition_modes', {})
    cast = config.get('generation_contract', {}).get('cast', {})
    if (not isinstance(modes, dict) or any(role not in cast or mode not in
            ('chronological_age', 'reversible_appearance') for role,mode in modes.items())):
        _fail('Character transition policy needs canonical cast roles and finite host-authorized modes')
    return deepcopy(modes)


def _allocation_projection(table, config):
    cast = config.get('generation_contract', {}).get('cast', {})
    if not isinstance(table, dict):
        _fail('character_reference_stages must be a role-keyed table')
    result = {}
    for role, family in table.items():
        if role not in cast or not isinstance(family, dict):
            _fail('Character stage role must be an existing canonical cast role')
        rows = family.get('stages')
        if not isinstance(rows, list) or not rows or not all(isinstance(s, dict) for s in rows):
            _fail('Each character needs a finite declared stage list')
        if any(not _integer(s.get('candidate_limit', 1)) or s.get('candidate_limit', 1) > 2 for s in rows):
            _fail('Proposed image candidate allocations must be bounded positive integers')
        result[role] = {'canonical_image_asset_id':family.get('canonical_image_asset_id', cast[role]['image_asset_id']),
            'stages':[{'id':s.get('id'), 'image_asset_id':s.get('image_asset_id'),
                       'candidate_limit':s.get('candidate_limit', 1)} for s in rows]}
    return result


def _role_keyed_table(raw, cast):
    if isinstance(raw, dict):
        return deepcopy(raw)
    if not isinstance(raw, list) or not raw:
        _fail('character_reference_stages must be a role-keyed table')
    table = {}
    for entry in raw:
        role = entry.get('canonical_role') if isinstance(entry, dict) else None
        if not isinstance(role, str) or role not in cast or role in table:
            _fail('Stage family list must name each existing canonical role at most once')
        table[role] = {key:deepcopy(value) for key,value in entry.items() if key != 'canonical_role'}
    return table


def _offscreen_occurrences(clip, role):
    """Require a matching explicit off-screen mark for every utterance by this role."""
    from .binding_timeline import visibility
    lines = clip.get('dialogue', [])
    timeline = clip.get('dialogue_timeline', [])
    seen = 0
    for index, line in enumerate(lines):
        if not isinstance(line, dict) or line.get('speaker') != role:
            continue
        seen += 1
        timing = timeline[index] if isinstance(timeline, list) and index < len(timeline) else {}
        if not isinstance(timing, dict) or timing.get('speaker', role) != role:
            return False
        marks = [v for v in (line.get('on_screen_or_offscreen'), timing.get('on_screen_or_offscreen')) if v]
        if not marks or any(visibility(v) != 'off_screen' for v in marks):
            return False
    return bool(seen)


def _voice_only_fallback(role, spec, assets, clips):
    voice = assets.get(spec.get('voice_asset_id'), {})
    if voice.get('kind') != 'audio' or voice.get('speaker', role) != role:
        return False
    spoken = False
    for clip in clips:
        if (spec['image_asset_id'] in clip.get('asset_ids', [])
                or role in (clip.get('character_states') or {}) or role in (clip.get('character_stage_bindings') or {})):
            return False
        if any(line.get('speaker') == role for line in clip.get('dialogue', [])):
            if not _offscreen_occurrences(clip, role):
                return False
            spoken = True
    return spoken


def _stable_family_fallbacks(table, plan, config):
    policy = config['character_reference_policy']
    roles = policy.get('stable_single_stage_roles', [])
    cast = config.get('generation_contract', {}).get('cast', {})
    if not isinstance(roles, list) or any(not isinstance(role, str) or role not in cast for role in roles) or len(set(roles)) != len(roles):
        _fail('Stable single-stage fallback requires an explicit list of canonical roles')
    missing = [role for role in roles if role not in table]
    if not missing:
        return [], []
    voice_only = []
    assets = {asset['id']:asset for asset in plan.get('assets', []) if isinstance(asset, dict) and 'id' in asset}
    for role in missing:
        canonical = cast[role]['image_asset_id']
        image = assets.get(canonical, {})
        description = cast[role].get('description')
        if not image and _voice_only_fallback(role, cast[role], assets, plan.get('clips', [])):
            voice_only.append(role)
        elif (image.get('kind') != 'image' or image.get('character_role', role) != role
                or image.get('canonical_image_asset_id', canonical) != canonical
                or (image.get('role') != 'character' and image.get('generator', 'qwen_image') != 'qwen_image')
                or image.get('role') in ('scene', 'setting', 'prop', 'environment', 'background')
                or not _english(image.get('prompt'))):
            _fail(f'Stable role {role} needs its existing canonical image {canonical} or explicit off-screen dialogue')
        if (not _english(description)
                or any((clip.get('character_states') or {}).get(role) not in (None, {'stage':'stable'})
                       for clip in plan.get('clips', []))):
            _fail('A stable role fallback needs its existing canonical image and no explicit stage selection')
        table[role] = {'canonical_image_asset_id':canonical, 'stages':[{
            'id':'stable', 'image_asset_id':canonical, 'description':description, 'candidate_limit':2}]}
    return missing, voice_only


def _normalize_allocation(plan, config):
    """Reduce a retained proposal to existing host slots; never alter its identities."""
    policy = config.get('character_reference_policy', {})
    if policy.get('allocation_authority') != ALLOCATION_AUTHORITY:
        return
    old = plan.get('character_stage_allocation_receipt')
    if old is not None and not isinstance(old, dict):
        _fail('Retained host allocation audit is malformed')
    raw = deepcopy(old.get('proposed_table') if old is not None else plan.get('character_reference_stages', {}))
    cast = config.get('generation_contract', {}).get('cast', {})
    table = _role_keyed_table(raw, cast); additions = []
    if 'time_jump_note' in table and 'time_jump_note' not in cast:
        note = table.pop('time_jump_note')
        if not isinstance(note, str) or not note.strip() or len(note) > 4000:
            _fail('time_jump_note is one bounded annotation string, never an unknown cast role')
        additions.append({'source_field':'character_reference_stages.time_jump_note', 'value':note})
    shapes = []
    for role, family in list(table.items()):
        if role in cast and isinstance(family, list):
            canonical = cast[role]['image_asset_id']
            wrapper = family[0] if len(family) == 1 and isinstance(family[0], dict) else {}
            if 'stages' in wrapper:
                if set(wrapper) != {'canonical_image_asset_id', 'stages'}:
                    _fail('Wrapped stage family permits only canonical_image_asset_id and stages')
                if wrapper['canonical_image_asset_id'] != canonical:
                    _fail('Stage table cannot change canonical person identity')
                table[role] = deepcopy(wrapper)
                original_shape = 'singleton_family_wrapper'
            else:
                # A stage list can fill its existing canonical role's stages field.
                table[role] = {'canonical_image_asset_id':canonical, 'stages':deepcopy(family)}
                original_shape = 'stage_list'
            shapes.append({'role':role, 'from':original_shape, 'to':'family.stages',
                           'canonical_image_asset_id':canonical})
    stable_roles, voice_only = _stable_family_fallbacks(table, plan, config)
    before_notes = deepcopy(old.get('notes_before', []) if old is not None else plan.get('character_reference_stage_notes', []))
    if not isinstance(before_notes, list):
        _fail('Retained character reference notes must be a list')
    budget = policy.get('max_candidates_per_character', 2)
    maximum = policy.get('max_stages_per_character', 3)
    if not _integer(budget) or not _integer(maximum) or maximum > 3:
        _fail('Character stage limits require explicit bounded configuration')
    proposed = _allocation_projection(table, config)
    for role,family in table.items():
        rows = family['stages']; desired = [s['candidate_limit'] for s in proposed[role]['stages']]
        if len(rows) > min(maximum, budget):
            _fail('Declared stage count exceeds the unchanged host family slots; do not merge stage IDs')
        effective = [1] * len(rows)
        remaining = budget - len(rows)
        for i,limit in enumerate(desired):
            added = min(limit - 1, remaining); effective[i] += added; remaining -= added
        for row,limit in zip(rows, effective):
            row['candidate_limit'] = limit
    receipt = {'version':ALLOCATION_AUTHORITY, 'family_candidate_limit':budget,
        'proposed_table':raw, 'proposed_table_sha256':_digest(raw), 'proposed':proposed,
        'effective':_allocation_projection(table, config), 'notes_before':before_notes,
        'retained_annotation_notes':additions, 'budget_increased':False,
        'scope':'allocation_only; stage identities, source facts and dialogue unchanged'}
    if shapes:
        receipt['retained_family_shape_normalizations'] = shapes
        receipt['scope'] = 'lossless family shape and bounded allocation; identities, source facts and dialogue unchanged'
    if isinstance(raw, list):
        receipt['retained_table_shape_normalization'] = 'canonical_role_list_to_role_keyed_table'
    if voice_only:
        receipt['offscreen_voice_only_roles'] = voice_only
    if stable_roles:
        receipt['host_declared_stable_single_stage_roles'] = stable_roles
    if old is not None:
        if (old != receipt or _allocation_projection(plan.get('character_reference_stages', {}), config) != receipt['effective']
                or plan.get('character_reference_stage_notes', []) != before_notes + additions):
            _fail('Retained host allocation audit/effective stage identities are immutable')
        return
    plan['character_reference_stages'] = table
    plan['character_reference_stage_notes'] = before_notes + additions
    plan['character_stage_allocation_receipt'] = receipt


def _table(plan, config):
    table = plan.get('character_reference_stages', {})
    if not isinstance(table, dict):
        _fail('character_reference_stages must be a role-keyed table')
    cast = config.get('generation_contract', {}).get('cast', {})
    policy = config.get('character_reference_policy', {})
    _role_modes(config)
    maximum = policy.get('max_stages_per_character', 3)
    budget = policy.get('max_candidates_per_character', 2)
    if not _integer(maximum) or maximum > 3 or not _integer(budget):
        _fail('Character stage limits require explicit bounded configuration')
    required = policy.get('require_stage_declarations', False)
    if type(required) is not bool:
        _fail('require_stage_declarations must be an explicit boolean')
    if required and (not cast or set(table) != set(cast)):
        _fail('Initial reference stages must declare every canonical cast role; no omitted-table young fallback')
    output = {}; occupied = set()
    canonical_ids = {v.get('image_asset_id') for v in cast.values()}
    for role, raw in table.items():
        if role not in cast or not isinstance(raw, dict):
            _fail('Character stage role must be an existing canonical cast role')
        canonical = cast[role]['image_asset_id']
        if raw.get('canonical_image_asset_id', canonical) != canonical:
            _fail('Stage table cannot change canonical person identity')
        stages = raw.get('stages')
        if not isinstance(stages, list) or not 1 <= len(stages) <= maximum:
            _fail('Character stage count exceeds the initial configured bound')
        rows = []; stage_ids = set()
        for rank, stage in enumerate(stages):
            if not isinstance(stage, dict):
                _fail('Each reference stage must be an object')
            ident = stage.get('id'); asset = stage.get('image_asset_id'); limit = stage.get('candidate_limit', 1)
            entry = f"Invalid/duplicate stage entry for {role!r} at index {rank}"
            if not isinstance(ident, str) or not re.fullmatch(r'[a-zA-Z0-9_\-]{1,48}', ident) or ident in stage_ids:
                _fail(f"{entry}: id must be a distinct stage identifier within the family")
            if not isinstance(asset, str) or not re.fullmatch(r'[a-zA-Z0-9_\-]{1,96}', asset):
                _fail(f"{entry} ({ident}): image_asset_id must be a valid asset identifier")
            if asset in occupied:
                _fail(f"{entry} ({ident}): image_asset_id {asset!r} is already assigned. "
                      "Use a distinct image ID for each declared appearance.")
            if not _integer(limit) or limit > 2:
                _fail(f"{entry} ({ident}): candidate_limit must be 1 or 2")
            if not _integer(stage.get('rank', rank), 0) or stage.get('rank', rank) != rank:
                _fail(f"{entry} ({ident}): rank must equal the zero-based stage index {rank}")
            if not _english(stage.get('description')):
                _fail(f"{entry} ({ident}): description must describe appearance in English. "
                      "Spell out source scene numbers as words or digits after the word scene. "
                      "Keep H3 binding labels out of this field.")
            if rank == 0 and asset != canonical:
                _fail('The first stage must use the canonical image asset')
            if rank and asset in canonical_ids:
                _fail('A later stage cannot use another canonical character image')
            stage_ids.add(ident); occupied.add(asset)
            rows.append({**deepcopy(stage), 'id': ident, 'rank': rank,
                         'image_asset_id': asset, 'candidate_limit': limit,
                         'description': stage['description'].strip()})
        if sum(s['candidate_limit'] for s in rows) > budget:
            _fail('Stage allocations exceed the unchanged shared character-image candidate budget')
        output[role] = {'canonical_image_asset_id': canonical,
                        'family_candidate_limit': budget, 'stages': rows}
    return output


def prepare_plan(plan, config):
    """Validate before paid assets, add stable stage specs and carry state forward."""
    result = deepcopy(plan)
    if not enabled(config):
        return result
    _normalize_allocation(result, config)
    table = _table(result, config)
    if not table:
        if any(c.get('character_states') or c.get('character_stage_bindings') for c in result.get('clips', [])):
            _fail('Explicit character stage requires its upfront declared family')
        return result
    assets = result.setdefault('assets', [])
    if not isinstance(assets, list):
        _fail('Stage image declarations need an asset list')
    by_id = {a.get('id'): a for a in assets if isinstance(a, dict)}
    voice_only = set(result.get('character_stage_allocation_receipt', {}).get('offscreen_voice_only_roles', []))
    for role, family in table.items():
        if role in voice_only:
            continue
        canonical = family['canonical_image_asset_id']
        for stage in family['stages']:
            ident = stage['image_asset_id']; current = by_id.get(ident, {})
            if current and (current.get('kind', 'image') != 'image'
                            or current.get('character_role', role) != role
                            or current.get('canonical_image_asset_id', canonical) != canonical):
                _fail('Stage image conflicts with an existing asset identity')
            if current.get('reference_asset_id') not in (None, canonical):
                _fail('Stage image parent must be the unchanged canonical person')
            prompt = stage.get('prompt') or (current.get('prompt') if stage['rank'] == 0 else None) or stage['description']
            if not _english(prompt):
                if stage.get('prompt'):
                    field = f"character_reference_stages[{role!r}].stages[{stage['rank']}].prompt"
                elif stage['rank'] == 0 and current.get('prompt'):
                    field = f"assets[{ident!r}].prompt"
                else:
                    field = f"character_reference_stages[{role!r}].stages[{stage['rank']}].description"
                _fail(f"Stage image generation requires explicit English directions in {field} for {role!r}. "
                      "Translate that image direction, retaining all appearance details, IDs and dialogue.")
            stage['prompt'] = prompt  # Retain the unprefixed plan input; preparation is idempotent.
            spec = {**deepcopy(current), 'id': ident, 'kind': 'image', 'role': 'character',
                    'generator': 'qwen_image', 'character_role': role,
                    'canonical_image_asset_id': canonical, 'character_stage': stage['id'],
                    'stage_rank': stage['rank'], 'stage_candidate_limit': stage['candidate_limit'],
                    'description': stage['description'], 'prompt': prompt}
            if current.get('role') and current['role'] != 'character':
                spec.setdefault('source_role', current['role'])
            if ident != canonical:
                spec['reference_asset_id'] = canonical
                spec['prompt'] = ('Same person as the supplied canonical reference: preserve facial identity, '
                                  'bone structure and distinctive features; change age/wardrobe only as specified. '
                                  + prompt)
            by_id[ident] = spec
    # Canonical images precede their child stages even if the writer listed assets otherwise.
    stage_ids = {s['image_asset_id'] for f in table.values() for s in f['stages']}
    canonical = [by_id[f['canonical_image_asset_id']] for role, f in table.items() if role not in voice_only]
    children = [by_id[s['image_asset_id']] for f in table.values() for s in f['stages'][1:]]
    result['assets'] = canonical + children + [a for a in assets if not isinstance(a, dict) or a.get('id') not in stage_ids]
    result['character_reference_stages'] = table
    carried = {}; highest = {}; modes = _role_modes(config)
    first_from_image = config['character_reference_policy'].get('first_stage_from_bound_image_roles', [])
    if (not isinstance(first_from_image, list) or any(not isinstance(role, str) or role not in table
                                                     for role in first_from_image)):
        _fail('First stage from an image needs explicit declared canonical roles')
    for clip in result.get('clips', []):
        explicit = clip.get('character_states', {})
        if not isinstance(explicit, dict):
            _fail('character_states must map a canonical role to its declared stage')
        if any(role not in config.get('generation_contract', {}).get('cast', {}) for role in explicit):
            _fail('Unknown role in character state')
        for role, state in explicit.items():
            if role not in table:
                _fail('Explicit character stage requires its upfront declared family')
            if (not isinstance(state, dict) or set(state) - {'stage', 'wardrobe', 'appearance'}
                    or any(not _english(v) or len(v) > 512 for k,v in state.items() if k != 'stage')):
                _fail('Character appearance state allows only stage and short English wardrobe/appearance strings')
        present = {d.get('speaker') for d in clip.get('dialogue', [])}
        offscreen = sorted(voice_only & present)
        if offscreen:
            clip['offscreen_voice_only_roles'] = offscreen
        else:
            clip.pop('offscreen_voice_only_roles', None)
        present.difference_update(voice_only)
        present.update(explicit)
        present.update(role for role, f in table.items()
                       if any(s['image_asset_id'] in clip.get('asset_ids', []) for s in f['stages']))
        bindings = {}; flashback = clip.get('timeline_transition') == 'flashback'
        current = deepcopy(carried)
        for role, family in table.items():
            state = explicit.get(role)
            if state is not None:
                if not isinstance(state, dict) or not isinstance(state.get('stage'), str):
                    _fail('Every explicit character state needs one declared stage')
                stage = next((s for s in family['stages'] if s['id'] == state['stage']), None)
                if stage is None:
                    _fail('Unknown age/reference stage; no implicit canonical-young fallback')
                if flashback:
                    if role not in carried:
                        _fail('A flashback requires an already established mainline character stage')
                    if stage['rank'] > highest[role]:
                        _fail('A flashback cannot introduce a future character stage')
                    current[role] = deepcopy(state)
                else:
                    if stage['rank'] < highest.get(role, 0) and modes.get(role) != 'reversible_appearance':
                        _fail('Chronological character stage regressed; correct the initial timeline explicitly')
                    highest[role] = max(stage['rank'], highest.get(role, 0)); carried[role] = deepcopy(state)
                    current[role] = deepcopy(state)
            if role not in present:
                continue
            if role not in current:
                if len(family['stages']) > 1:
                    bound = {s['image_asset_id'] for s in family['stages'] if s['image_asset_id'] in clip.get('asset_ids', [])}
                    if flashback or role not in first_from_image or bound != {family['canonical_image_asset_id']}:
                        _fail('First visible occurrence of a multi-stage character needs an explicit stage')
                current[role] = {'stage': family['stages'][0]['id']}
                if not flashback:
                    carried[role] = deepcopy(current[role]); highest[role] = 0
            state = current[role]; stage = next(s for s in family['stages'] if s['id'] == state['stage'])
            bindings[role] = {'canonical_image_asset_id': family['canonical_image_asset_id'],
                              'image_asset_id': stage['image_asset_id'], 'stage': stage['id'],
                              'stage_rank': stage['rank'], 'description': stage['description'],
                              'appearance': deepcopy({k:v for k,v in state.items() if k != 'stage'})}
        clip['character_states'] = {**deepcopy(explicit), **{r:deepcopy(current[r]) for r in bindings}}
        clip['character_stage_bindings'] = bindings
    return result


def freeze_plan(store, plan, config):
    """Pin the upfront family allocation; a new ID/version cannot mint slots."""
    if not enabled(config):
        return None
    table = _table(plan, config)
    receipt = {'version': VERSION, 'families': table, 'sha256': _digest(table),
               'flashback_clip_ids': _flashback_ids(plan),
               'generation_authority': 'initial_plan_only', 'budget_reset_allowed': False}
    if 'role_transition_modes' in config.get('character_reference_policy', {}):
        receipt['role_transition_modes'] = _role_modes(config)
    if config.get('character_reference_policy', {}).get('allocation_authority') == ALLOCATION_AUTHORITY:
        checked = deepcopy(plan); _normalize_allocation(checked, config)
        if checked.get('character_stage_allocation_receipt') != plan.get('character_stage_allocation_receipt'):
            _fail('Host allocation must be prepared and audited before publication')
        receipt['allocation_audit'] = deepcopy(checked['character_stage_allocation_receipt'])
    old = store.data.get('character_reference_plan_receipt')
    if old is not None and old != receipt:
        _fail('Initial character stage identity/allocation is immutable after publication')
    if old is None:
        store.data['character_reference_plan_receipt'] = deepcopy(receipt)
        store.save()
    return receipt


def image_ids_for_clip(clip, config):
    return [b['image_asset_id'] for b in clip.get('character_stage_bindings', {}).values()] if enabled(config) else []


def assert_clip_bindings(clip, config):
    """Check the normalized authority at dispatch; never reconstruct lost state."""
    if not enabled(config) or config['character_reference_policy'].get('require_stage_declarations') is not True:
        return
    cast = config.get('generation_contract', {}).get('cast', {})
    states = clip.get('character_states', {}); bindings = clip.get('character_stage_bindings', {})
    if not isinstance(states, dict) or not isinstance(bindings, dict):
        _fail('Required character stage binding/state is missing or malformed')
    present = set(states) | set(bindings)
    present.update(d.get('speaker') for d in clip.get('dialogue', []) if isinstance(d, dict))
    present.update(role for role,spec in cast.items() if spec['image_asset_id'] in clip.get('asset_ids', []))
    for role in clip.get('offscreen_voice_only_roles', []):
        if (role not in cast or role not in config['character_reference_policy'].get('stable_single_stage_roles', [])
                or role in states or role in bindings or cast[role]['image_asset_id'] in clip.get('asset_ids', [])
                or not _offscreen_occurrences(clip, role)):
            _fail('Voice-only identity requires explicit off-screen speech and no visual stage or reference')
        present.discard(role)
    for role in present:
        state = states.get(role); binding = bindings.get(role)
        if (role not in cast or not isinstance(state, dict) or not isinstance(binding, dict)
                or not isinstance(state.get('stage'), str) or not state['stage']
                or binding.get('stage') != state['stage']
                or binding.get('canonical_image_asset_id') != cast[role]['image_asset_id']
                or not isinstance(binding.get('image_asset_id'), str) or not binding['image_asset_id']
                or not _integer(binding.get('stage_rank'), 0) or not _english(binding.get('description'))
                or binding.get('appearance') != {k:v for k,v in state.items() if k != 'stage'}):
            _fail('Required character stage binding differs from its normalized state/canonical identity')


def required_image_ids(clip, config):
    """Include canonical parent for lazy child preparation, not for H3 rendering."""
    if not enabled(config):
        return []
    return list(dict.fromkeys(i for b in clip.get('character_stage_bindings', {}).values()
                             for i in (b['canonical_image_asset_id'], b['image_asset_id'])))


def _asset_stage(plan, config, ident):
    for role, family in _table(plan, config).items():
        for stage in family['stages']:
            if stage['image_asset_id'] == ident:
                return role, family, stage
    return None


def assert_generation_allowed(agent, ident, spec):
    if not enabled(agent.config):
        return
    table = _table(agent.s.data.get('plan', {}), agent.config)
    item = _asset_stage(agent.s.data.get('plan', {}), agent.config, ident)
    if any(spec.get(k) is not None and not isinstance(spec[k], str)
           for k in ('canonical_image_asset_id', 'reference_asset_id', 'character_role')):
        _fail('Character stage lineage fields must be explicit asset/role strings')
    parents = {spec.get('canonical_image_asset_id'), spec.get('reference_asset_id')}
    family_ids = {s['image_asset_id'] for f in table.values() for s in f['stages']}
    cast_ids = {r['image_asset_id'] for r in agent.config.get('generation_contract', {}).get('cast', {}).values()}
    if not item and (parents & family_ids or spec.get('character_role') in table
                    or spec.get('role') == 'character' and ident not in cast_ids):
        _fail('Undeclared character-stage descendant cannot refresh the family quota')
    if item:
        role, family, stage = item
        if (spec.get('character_role') != role or spec.get('character_stage') != stage['id']
                or spec.get('canonical_image_asset_id') != family['canonical_image_asset_id']):
            _fail('Stage asset metadata differs from the frozen same-person plan')


def candidate_takes(agent, ident, row):
    """Return original/reconcilable slots or remaining allocation across ALL versions."""
    if not enabled(agent.config):
        return None
    item = _asset_stage(agent.s.data.get('plan', {}), agent.config, ident)
    if item is None:
        assert_generation_allowed(agent, ident, row.get('spec', {}))
        return None
    assert_generation_allowed(agent, ident, row['spec'])
    role, family, stage = item
    frozen = agent.s.data.get('character_reference_plan_receipt')
    if not frozen or frozen.get('sha256') != _digest(_table(agent.s.data['plan'], agent.config)):
        _fail('Stage family allocation must be frozen before an image dispatch')
    consumed = set()
    for s in family['stages']:
        asset = s['image_asset_id']; asset_row = agent.s.data.get('assets', {}).get(asset, {})
        for name in asset_row.get('lifecycle_attempts', {}):
            try:version, take = map(int, name.split('/'))
            except (ValueError, AttributeError):_fail('Malformed retained character image attempt ledger')
            consumed.add((asset, version, take))
        suffix = hashlib.sha256(asset.encode()).hexdigest()[:12]
        root = agent.s.run/'media/images'
        accounted = agent.s.data.get('accounted_operations', ())
        folders = set(root.glob('*_'+suffix)) | {Path(p) for p in accounted if Path(p).parent == root}
        for folder in folders:
            match = re.fullmatch(r'(\d+)_(\d+)_'+suffix, folder.name)
            if match and (operation_recorded(folder, accounted) or (folder/'process.json').is_file()):
                consumed.add((asset, int(match[1]), int(match[2])))
    version = row.get('pending_version') or row.get('generation', 0) + 1
    per_stage = sum(x[0] == ident for x in consumed); total = len(consumed); result = []
    for take in range(1, stage['candidate_limit'] + 1):
        operation = (ident, version, take)
        if operation in consumed:
            result.append(take)  # Reconcile the exact original request, never a new slot.
        elif per_stage < stage['candidate_limit'] and total < family['family_candidate_limit']:
            result.append(take); per_stage += 1; total += 1
    if not result:
        raise Pending('Character stage allocation exhausted across retained versions; no fresh ID or generation', status='needs_asset_repair')
    return tuple(result)


def annotate_catalog(catalog, assets, plan, config):
    result = deepcopy(catalog)
    if not enabled(config):
        return result
    for actual in result:
        item = _asset_stage(plan, config, actual['id'])
        if item is None:
            continue
        role, family, stage = item
        spec = assets.get(actual['id'], {}).get('spec', {})
        if (spec.get('canonical_image_asset_id') != family['canonical_image_asset_id']
                or spec.get('character_stage') != stage['id'] or spec.get('character_role') != role):
            _fail('Registered stage image is not in its original same-person lineage')
        version = next((v for v in assets.get(actual['id'], {}).get('versions', []) if v.get('sha256') == actual['sha256']), {})
        provenance = version.get('character_stage_generation', {})
        parent_sha = provenance.get('parent_sha256')
        if stage['rank']:
            parent_versions = assets.get(family['canonical_image_asset_id'], {}).get('versions', [])
            parent_intact = any(v.get('sha256') == parent_sha and Path(v.get('path', '')).is_file()
                                and sha(v['path']) == parent_sha for v in parent_versions)
            if (provenance.get('role') != role or provenance.get('stage') != stage['id']
                    or provenance.get('canonical_image_asset_id') != family['canonical_image_asset_id']
                    or provenance.get('image_asset_id') != actual['id']
                    or provenance.get('image_sha256') != actual['sha256'] or not parent_intact):
                _fail('Later-stage reference lacks its retained physical canonical-parent SHA lineage')
        actual['character_stage_authority'] = {'role': role, 'stage': stage['id'], 'stage_rank': stage['rank'],
            'canonical_image_asset_id': family['canonical_image_asset_id'], 'asset_id': actual['id'],
            'sha256': actual['sha256'], 'parent_sha256': parent_sha,
            'content_acceptance_not_guaranteed': True}
    return result


def resolve_image(clip, role, by_id, config):
    """None means legacy role; declared stage absence is never a legacy fallback."""
    if not enabled(config) or role not in clip.get('character_stage_bindings', {}):
        return None
    b = clip['character_stage_bindings'][role]; cast = config['generation_contract']['cast'][role]
    if b.get('canonical_image_asset_id') != cast['image_asset_id']:
        raise ValueError('Stage binding changed canonical person identity')
    image = by_id.get(b['image_asset_id'])
    if image is None:
        raise ValueError('Required character stage image missing; cannot fall back to the young canonical image')
    a = image.get('character_stage_authority', {})
    if (image.get('kind') != 'image' or a.get('role') != role or a.get('stage') != b['stage']
            or a.get('canonical_image_asset_id') != b['canonical_image_asset_id']
            or a.get('asset_id') != image['id'] or a.get('sha256') != image.get('sha256')):
        raise ValueError('Stage reference lacks its exact same-person/stage/SHA authority')
    others = [r for r in by_id.values() if r['id'] != image['id']
              and (r['id'] == b['canonical_image_asset_id'] or
                   r.get('character_stage_authority', {}).get('canonical_image_asset_id') == b['canonical_image_asset_id'])]
    if others:
        raise ValueError('Ambiguous multiple age references for the same character')
    return image


AGE_REVIEW_GUIDE = (
    '\nCompare the edited image against the canonical identity and requested age. '
    'Return character_age_review with identity, age_change and natural_texture. '
    'Each is an object with status ("clear", "defect" or "unknown") and visible evidence. '
    'Flag the wrong person, missing or reversed requested age change, and visibly waxy, '
    'painted or over-sharpened skin as defects. If the evidence is inconclusive, say unknown.'
)


def age_edit_mode(spec):
    """Derive age intent from the plan-owned stage, independently of repair prompts."""
    description = spec['description'].casefold()
    younger = re.search(r'\b(younger|de[- ]aged?|rejuvenat\w*)\b|年轻版|变年轻|年轻时|青年时期|回春', description)
    older = re.search(r'\b(aged?|grows?|looks?|appears?)\s+(visibly\s+)?older\b|\bolder\s+(version|self|adult|appearance)\b|\bolder\b(?!\s+(brother|sister|sibling|son|daughter|friend|coat|jacket|garment|clothes|clothing|costume))|变老|年长版|年长时|成年后', description)
    elderly = re.search(r'\b(elderly|old age|senior citizen|(?:late )?seventies|eighties|nineties|[7-9][0-9][ -]year[s -]*(?:old)?)\b|老年|高龄|七[十旬]|八[十旬]|九[十旬]', description)
    if younger and (older or elderly):
        return None
    if younger:
        return 'younger'
    if elderly:
        return 'elderly'
    if older:
        return 'older'
    return None


def age_edit_prompt(stage_description, mode, repair_prompt):
    cues = {
        'younger': 'Use fewer established lines and less under-eye hollow, with age-appropriate skin texture and facial volume.',
        'older': 'Show the requested increase in apparent age with natural lines, age-appropriate hair changes and subtle facial volume changes.',
        'elderly': 'Make old age visibly legible with naturally greying hair, established forehead and eye lines, and age-appropriate cheeks and jawline.',
    }
    return (
        f'Edit the supplied portrait as the exact same person at the requested life stage. '
        f'Target: {stage_description}. {cues[mode]} Keep recognisable eye spacing, nose, mouth and '
        'underlying facial structure. Use a realistic photographic portrait with soft natural light, '
        'restrained contrast and natural skin. Preserve requested wardrobe and setting. '
        'Follow any compatible production detail below, giving the target age and identity priority.\n'
        f'{repair_prompt}'
    )


def age_review_defects(review):
    report = review.get('character_age_review')
    if not isinstance(report, dict):
        return []
    defects = []
    for dimension in ('identity', 'age_change', 'natural_texture'):
        finding = report.get(dimension)
        if isinstance(finding, dict) and finding.get('status') == 'defect':
            evidence = finding.get('evidence')
            if isinstance(evidence, str) and evidence.strip():
                defects.append({'dimension': dimension, 'evidence': evidence.strip()})
    return defects


def generation_parent(agent, ident):
    """Return the real canonical parent for an age edit; no target-only parent."""
    if not enabled(agent.config):
        return None
    item = _asset_stage(agent.s.data.get('plan', {}), agent.config, ident)
    if item is None or item[2]['rank'] == 0:
        return None
    role, family, stage = item
    parent = next((a for a in agent.catalog() if a['id'] == family['canonical_image_asset_id']), None)
    if not parent or parent.get('kind') != 'image':
        raise Pending('Prepare the canonical identity image before its later age stage', status='needs_asset_repair')
    if not Path(parent.get('path', '')).is_file() or sha(parent['path']) != parent.get('sha256'):
        _fail('Canonical parent image SHA changed before stage generation')
    return deepcopy(parent)


def stamp_generated(agent, ident, actual, parent=None):
    """Record physical parent provenance, not a claim that image review passed."""
    result = deepcopy(actual)
    if not enabled(agent.config):
        return result
    item = _asset_stage(agent.s.data.get('plan', {}), agent.config, ident)
    if item is None:
        return result
    role, family, stage = item
    if stage['rank'] and (not parent or parent.get('id') != family['canonical_image_asset_id']):
        _fail('Later-stage image must have the real canonical image supplied to its worker')
    result['character_stage_generation'] = {
        'role': role, 'stage': stage['id'], 'canonical_image_asset_id': family['canonical_image_asset_id'],
        'image_asset_id': ident, 'image_sha256': actual['sha256'],
        'parent_sha256': parent.get('sha256') if parent else None,
        'parent_version': parent.get('version') if parent else None,
        'identity_and_age_review_separate': True}
    return result
