"""Compile H3 reference labels and speaking roles from host-owned structured data."""
from copy import deepcopy
import hashlib
import json
import re

from .assets import reference_labels
from .evidence_policy import scripted, dialogue_texts, normal
from .storage import Pending, sha
from . import task_parameters, binding_timeline, voice_lineage, dialogue_integrity

VERSION = 'cast_reference_contract_v2'
LEGACY = 'cast_reference_contract_v1'
SILENT_POLICY_VERSION = 'h3_nonverbal_scene_v1'
SILENT_POLICY = 'No intelligible speech. Use only nonverbal environment sounds and physical foley.'


def enabled(config):
    return config.get('generation_contract', {}).get('version') in (VERSION, LEGACY)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def text(value):
    value = str(value or '').strip()
    if re.search(r'<\s*(?:Subject|Picture|Audio|Video|d)\b|\bS\d+\b', value, re.I):
        raise ValueError('Structured shot direction cannot introduce independent reference/subject labels')
    return value


def english_direction(value):
    value=text(value)
    if re.search(r'[\u3400-\u9fff]',value):
        raise ValueError('H3 guide: non-dialogue direction must be English; keep original speech only inside <d>')
    return value


def validate_cast(cast):
    if not isinstance(cast, dict) or not cast:
        raise ValueError('Source plan requires an explicit canonical cast')
    if any(not isinstance(row, dict) or 'voice_asset_id' not in row for row in cast.values()):
        raise ValueError('Canonical roles need explicit image and voice asset ids; use null for a non-speaking role')
    images = [row.get('image_asset_id') for row in cast.values()]
    voices = [row['voice_asset_id'] for row in cast.values() if row['voice_asset_id'] is not None]
    for values in (images, voices):
        if any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError('Canonical roles need distinct explicit image and non-null voice asset ids')


def validate_source(plan, config):
    """Validate source-only contracts before assets; no paths/models are needed."""
    from . import shot_policy
    shot_policy.validate_config(config)
    if not enabled(config):
        return plan
    cast = config.get('generation_contract', {}).get('cast', {})
    validate_cast(cast)
    if not isinstance(plan.get('assets', []), list) or not isinstance(plan.get('clips'), list):
        raise ValueError('Source assets and clips must be explicit lists')
    assets = {}
    for asset in plan.get('assets', []):
        if not isinstance(asset, dict) or not isinstance(asset.get('id'), str):
            raise ValueError('Every planned asset needs a stable explicit id')
        ident = asset['id']
        if ident in assets:
            raise ValueError('Duplicate source asset id; cannot hide competing identity/constraints')
        assets[ident] = asset
    for clip in plan.get('clips', []):
        shot_policy.validate_clip(clip, config)
        direction = clip.get('render_spec')
        if not isinstance(direction, dict) or not all(english_direction(direction.get(k)) for k in ('opening', 'action', 'camera', 'closing')):
            raise ValueError('Source needs complete executable English render_spec before assets')
        raw_lines = clip.get('dialogue', [])
        if not isinstance(raw_lines, list) or any(not isinstance(row, dict) or not isinstance(row.get('speaker'), str) or not isinstance(row.get('text'), str) or not row['text'].strip() for row in raw_lines):
            raise ValueError('Literal dialogue must be a list of explicit speaker/text occurrences')
        asset_ids = clip.get('asset_ids', [])
        if not isinstance(asset_ids, list) or any(not isinstance(ident,str) for ident in asset_ids):
            raise ValueError('Source asset_ids must be a list of stable ids')
        lines = scripted(clip)
        if dialogue_integrity.enabled(config):
            dialogue_integrity.validate_source_language(lines, task_parameters.language(config))
        timeline = clip.get('dialogue_timeline', [])
        if not isinstance(timeline, list) or any(not isinstance(row,dict) for row in timeline):
            raise ValueError('Source timeline must be a list of structured occurrences')
        for index, line in enumerate(lines):
            role = line.get('speaker')
            if role not in cast:
                raise ValueError('Source dialogue names an unknown canonical role')
            if cast[role]['voice_asset_id'] is None:
                raise ValueError('Source dialogue assigns speech to a non-speaking canonical role')
            row = timeline[index] if index < len(timeline) else {}
            if row.get('speaker', role) != role:
                raise ValueError('Source timeline and literal dialogue disagree on speaker')
            left = line.get('on_screen_or_offscreen')
            right = row.get('on_screen_or_offscreen')
            lv = binding_timeline.visibility(left) if left else None
            rv = binding_timeline.visibility(right) if right else None
            if lv and rv and lv != rv:
                raise ValueError('Source dialogue and timeline have contradictory visibility')
    return plan


def compile(clip, references, config, plan):
    from . import character_stages, shot_policy
    character_stages.assert_clip_bindings(clip, config)
    cast = config['generation_contract'].get('cast', {})
    if not isinstance(cast, dict) or not cast:
        raise ValueError('An explicit cast table is required')
    refs = reference_labels(references)
    by_id = {r['id']: r for r in refs}
    if len(by_id) != len(refs):
        raise ValueError('Duplicate reference asset id')
    if character_stages.enabled(config) and config.get('character_reference_policy', {}).get('require_stage_declarations'):
        for reference in refs:
            authority = reference.get('character_stage_authority', {})
            if authority and clip.get('character_stage_bindings', {}).get(authority.get('role'), {}).get('image_asset_id') != reference['id']:
                raise ValueError('Actual stage reference differs from the source-owned clip binding')
    lines = scripted(clip)
    silent = not lines
    if silent and any(r['kind'] == 'video' and r.get('include_audio') for r in refs):
        raise ValueError('A nonverbal scene cannot take a reference video with audio')
    subjects, definitions, table, actual_images = {}, [], [], {}
    for role in sorted(cast):
        spec=cast[role]
        stage_bound = role in clip.get('character_stage_bindings', {}) and character_stages.enabled(config)
        image = character_stages.resolve_image(clip, role, by_id, config) if stage_bound else by_id.get(spec['image_asset_id'])
        if not stage_bound and image is not None and any(a.get('asset_id') in by_id for a in spec.get('approved_image_aliases', [])):
            raise ValueError('Ambiguous canonical and variant images for one cast role')
        if image is None and not stage_bound:
            aliases=[]
            for approved in spec.get('approved_image_aliases', []):
                actual=by_id.get(approved.get('asset_id'))
                if actual is None:
                    continue
                if (approved.get('parent_image_asset_id') != spec['image_asset_id']
                        or actual.get('kind') != 'image' or actual.get('sha256') != approved.get('sha256')):
                    raise ValueError('Character image variant differs from its explicit SHA-bound canonical approval')
                aliases.append(actual)
            if len(aliases)>1:
                raise ValueError('Ambiguous character image variants; one actual image per cast role is required')
            image=aliases[0] if aliases else None
        if image:
            if image['kind'] != 'image':
                raise ValueError('Cast image binding is not an image')
            subjects[role] = f'<Subject {len(subjects)+1}>'
            actual_images[role]=image
            if silent:
                definitions.append(f"{subjects[role]} is identified by {image['h3_label']}; preserve the exact pictured face, age and visible identifying features.")
            elif dialogue_integrity.enabled(config) or stage_bound:
                # Cast descriptions contain plot analysis in imported benchmark roles.
                # The bound image is the identity authority; never recite role biography.
                stage = clip.get('character_stage_bindings', {}).get(role) if stage_bound else None
                appearance = ('Preserve the same person facial identity and the age shown in this exact stage reference: '
                              + english_direction(stage['description']) + '. ' +
                              ' '.join(english_direction(v) for v in stage.get('appearance', {}).values())) if stage else 'Preserve the pictured facial identity, age and appearance.'
                definitions.append(f"{subjects[role]} is the character identified by {image['h3_label']}; {appearance}")
            else:
                definitions.append(f"{subjects[role]} is {english_direction(spec.get('description', 'the specified adult character'))}, identified by {image['h3_label']}.")
    for ref in refs:
        if ref['kind'] == 'image' and ref['id'] not in {v['id'] for v in actual_images.values()}:
            definitions.append(f"{ref['h3_label']} is the setting/prop reference. Match its architecture, lighting, materials and prop placement; do not display the reference image itself.")
    if any(line['speaker'] in cast and cast[line['speaker']].get('voice_asset_id') is None for line in lines):
        raise ValueError('Cannot compile speech for a non-speaking canonical role')
    if dialogue_integrity.enabled(config):
        dialogue_integrity.validate_source_language(lines, task_parameters.language(config))
    vocal_ids={role:'S'+str(i+1) for i,role in enumerate(sorted(cast))}
    timelines = binding_timeline.groups(lines, plan.get('dialogue_timeline', []),
        clip.get('dialogue_timeline', []), allow_partition=config['generation_contract']['version']==VERSION)
    speech = []
    for index, (line, parts) in enumerate(zip(lines, timelines)):
        timing=parts[0]
        role = line['speaker']; spec = cast.get(role)
        if not spec or timing.get('speaker') != role:
            raise ValueError('Script and timeline disagree on speaker')
        visibility = timing.get('on_screen_or_offscreen')
        segments=[]
        for part in parts:
            visible=binding_timeline.visibility(part.get('on_screen_or_offscreen'))
            begin,finish=part.get('start_s'),part.get('end_s')
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in (begin,finish)) or not 0<=begin<finish<=clip['seconds']-.5:
                raise ValueError('Speech timeline must reserve a real closing handle')
            if segments and begin<segments[-1]['end_s']:
                raise ValueError('Partition times must be ordered and non-overlapping')
            segments.append({'start_s':begin,'end_s':finish,'visibility':'on_screen' if visible=='on_screen' else 'off_screen'})
        on_screen = any(x['visibility']=='on_screen' for x in segments)
        authority=clip.get('dialogue_timeline',[])
        if index<len(authority) and authority[index].get('on_screen_or_offscreen'):
            required=binding_timeline.visibility(authority[index]['on_screen_or_offscreen'])
            if any(x['visibility']!=required for x in segments):
                raise ValueError('Local timing cannot silently change the source event speaking visibility')
        if on_screen and role not in subjects:
            raise ValueError('Visible speaker has no bound character image')
        voice = voice_lineage.resolve(by_id,spec['voice_asset_id'],role)
        if not voice or voice['kind'] != 'audio' or voice.get('speaker') != role or voice.get('track_type') not in ('synthetic_voice', 'isolated_voice'):
            raise ValueError('Speaker needs its own isolated, explicitly labelled voice reference')
        start, end = segments[0]['start_s'], segments[-1]['end_s']
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (start, end)) or not 0 <= start < end <= clip['seconds'] - .5:
            raise ValueError('Speech timeline must reserve a real closing handle')
        fallback = 'the off-screen speaker' if task_parameters.enabled(config) else 'the off-screen adult speaker'
        actor = subjects.get(role, fallback)+' ('+vocal_ids[role]+')'
        definitions.append(f"{voice['h3_label']} is ONLY the voice timbre of {actor}; it never belongs to another visible person.")
        if len(parts)==1:
            delivery='visibly with synchronized lips' if on_screen else 'off-screen; visible listeners remain silent'
        else:
            delivery='through one continuous scripted occurrence, with these explicit visibility intervals: '+ '; '.join(
                f"{x['start_s']:g}s to {x['end_s']:g}s " + ('on-screen with synchronized lips' if x['visibility']=='on_screen' else 'off-screen; visible listeners remain silent') for x in segments)
        speech.append(f"From approximately {start:g}s to {end:g}s, {actor} speaks {delivery} in the timbre of {voice['h3_label']}, naturally and completely: <d>[{task_parameters.language(config)}] {line['text'].strip()}</d>")
        image_id = actual_images.get(role, {}).get('id', spec['image_asset_id'])
        if character_stages.enabled(config) and role in clip.get('offscreen_voice_only_roles', []):
            image_id = None
        table.append({'script_index': index, 'speaker': role, 'vocal_id':vocal_ids[role], 'subject_label': subjects.get(role),
                      'image_asset_id': image_id, 'image_label': actual_images.get(role,{}).get('h3_label'),
                      'voice_asset_id': voice['id'], 'voice_label': voice['h3_label'], 'voice_sha256': voice['sha256'],
                      'visibility': ('on_screen' if on_screen else 'off_screen') if len(parts)==1 else 'segmented', 'start_s': start, 'end_s': end})
        if voice['id']!=spec['voice_asset_id']:
            table[-1]['canonical_voice_asset_id']=spec['voice_asset_id']
            table[-1]['voice_lineage_authority']=deepcopy(voice['canonical_voice_authority'])
        if len(parts)>1:
            table[-1]['visibility_segments']=segments
    allowed_voices = {row['voice_asset_id'] for row in table}
    from . import completion_memory, shot_policy
    sounds=[]
    for r in refs:
        if r['kind']!='audio' or r['id'] in allowed_voices:
            continue
        a=r.get('sound_reference_authority',{})
        if (not completion_memory.enabled(config) or r['id'] not in clip.get('sound_asset_ids',[])
                or a.get('asset_id')!=r['id'] or a.get('role') not in ('music','ambience')
                or a.get('speech_present') is not False or a.get('human_vocalization') is not False
                or not a.get('asset_action_receipt_id') or a.get('selected_sha256')!=r['sha256']
                or sha(a['path'])!=a['sha256'] or r.get('review_status')!='PASS'
                or (a['role']=='music' and (shot_policy.no_score(config) or clip.get('music_asset_id')!=r['id']))):
            raise ValueError('Unassigned voice/music reference cannot enter this generation')
        sounds.append({'asset_id':r['id'],'label':r['h3_label'],'authority':deepcopy(a)})
        if silent:
            definitions.append(f"{r['h3_label']} guides only the {a['role']} sound texture.")
        else:
            definitions.append(f"{r['h3_label']} guides only the {a['role']} sound texture; contains no speech or human vocalization. Do not copy its signal or words.")
    music=any(r['authority']['role']=='music' for r in sounds)
    direction = shot_policy.execution_direction(clip, plan)
    if not isinstance(direction, dict):
        raise ValueError('Local planner must provide structured render_spec, not free reference-numbered caption')
    if dialogue_integrity.enabled(config) or silent:
        direction, direction_audit = dialogue_integrity.visual_only(direction, silent=silent)
    else:
        direction_audit = None
    fields = [english_direction(direction.get(k)) for k in ('opening', 'action', 'camera', 'closing')]
    if not all(fields):
        raise ValueError('render_spec needs opening, action, camera and closing')
    for field in fields:
        if '<d>' in field or any(normal(row['text']) in normal(field) for row in lines):
            raise ValueError('Only the host speech block may quote the scripted line')
    retention='\n'.join((f"{label}: preserve the same person and the exact declared age stage; apply the declared clip wardrobe/appearance instead of freezing portrait clothing."
        if character_stages.enabled(config) and role in clip.get('character_stage_bindings', {}) else
        f"{label}: fully_preserved - preserve the bound character identity, clothing and appearance.") for role,label in subjects.items())
    retention+='\nVisual setting references guide the location and props.'
    if not silent:
        retention+=' Audio references guide timbre only; do not copy their words or original signal.'
    caption = ('subject_definitions:\n' + '\n'.join(definitions) +
               f"\n\nsummary:\nOne complete {task_parameters.shot_context(config)} shot, approximately {clip['seconds']:g} seconds. No background music.\n\ndetailed_description:\n[Shot 1] " +
               ' '.join(fields) + '\n' + shot_policy.source_conditions(clip, config) + '\n' + '\n'.join(speech) +
               '\nOnly the listed speakers speak, once per listed occurrence. Preserve the full final syllable and a settled silent closing handle. Do not rush, stutter, repeat, add words, narration or music.\n'
               'overall_soundscape:\nNative dialogue, quiet scene ambience and physically justified foley. No non-diegetic music.\n\nnon_diegetic_music:\nNone.\n')
    if silent:
        shot_context = dialogue_integrity.clean_silent_direction(task_parameters.shot_context(config)) or 'cinematic'
        caption = (
            'subject_definitions:\n' + '\n'.join(definitions)
            + f"\n\nsummary:\nOne complete {shot_context} shot, approximately {clip['seconds']:g} seconds."
            + '\n\nspoken_content_policy:\n' + SILENT_POLICY
            + '\n\nretention_analysis:\n' + retention
            + '\n\ndetailed_description:\n[Shot 1] ' + ' '.join(fields)
            + '\n' + shot_policy.source_conditions(clip, config, silent=True)
            + '\n\noverall_soundscape:\nQuiet, physically justified scene ambience and foley.'
            + (' One quiet, purely instrumental cue guided by the bound reference.' if music else '')
            + '\n\nnon_diegetic_music:\n' + ('One short instrumental cue.' if music else 'None.') + '\n'
        )
    elif dialogue_integrity.enabled(config):
        lang = task_parameters.language(config)
        spoken = ('Only the literal text inside the delimited dialogue blocks is spoken, exactly once, in '+lang+'. '
                  'Visual directions and reference descriptions are silent instructions. '
                  'Story location, nationality and accent never change the spoken language. No translation or narrator.')
        caption = caption.replace('detailed_description:\n', 'spoken_content_policy:\n'+spoken+'\n\ndetailed_description:\n')
    if music and not silent:
        caption=caption.replace('No background music.','One quiet sparse instrumental cue, below the native dialogue and foley.')
        caption=caption.replace('add words, narration or music.','add words or narration. Keep the instrumental cue quiet.')
        caption=caption.replace('No non-diegetic music.','The isolated instrumental cue gently fades out before this shot ends; do not imply an adjacent continuing score.')
        caption=caption.replace('non_diegetic_music:\nNone.','non_diegetic_music:\nOne short quiet instrumental cue guided by the explicitly bound music reference.')
    if not silent:
        caption=caption.replace('\n\ndetailed_description:' , '\n\nretention_analysis:\n'+retention+'\n\ndetailed_description:',1)
    if dialogue_texts(caption) != [normal(x['text']) for x in lines]:
        raise ValueError('Canonical dialogue occurrences changed during compilation')
    if re.search(r'[\u3400-\u9fff]',re.sub(r'<d>.*?</d>','',caption,flags=re.S)):
        raise ValueError('Unspoken Chinese metadata leaked into final H3 caption')
    record = {'version': VERSION, 'source_event_sha256': digest(clip), 'render_spec_sha256': digest(direction),
              'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(), 'cast_sha256': digest(cast),
              'ordered_references': [{'id': r['id'], 'kind': r['kind'], 'label': r['h3_label'], 'sha256': r['sha256']} for r in refs],
              'speaking_bindings': table, 'character_stage_bindings': deepcopy(clip.get('character_stage_bindings', {})),
              'source_action_state_targets': {'action_beats':deepcopy(clip.get('action_beats', [])), 'state_checkpoints':deepcopy(clip.get('state_checkpoints', {}))}, **({'sound_bindings':sounds} if sounds else {}),
              **({'dialogue_integrity_version':dialogue_integrity.VERSION,'visual_direction_audit':direction_audit} if direction_audit is not None else {})}
    if silent:
        record['nonverbal_policy'] = SILENT_POLICY_VERSION
    return caption, record


def validate(caption, clip, references, config, record, *, allow_original_legacy=False):
    cast = config['generation_contract']['cast']
    lines = scripted(clip)
    if any(not cast.get(line['speaker'], {}).get('voice_asset_id') for line in lines):
        raise Pending('A scripted speaker lacks a canonical voice asset; do not submit H3')
    expected = [{'id': r['id'], 'kind': r['kind'], 'label': r['h3_label'], 'sha256': r['sha256']} for r in reference_labels(references)]
    permitted={VERSION,LEGACY} if allow_original_legacy else {VERSION}
    if (not isinstance(record, dict) or record.get('version') not in permitted or record.get('ordered_references') != expected
            or record.get('caption_sha256') != hashlib.sha256(caption.encode()).hexdigest()
            or record.get('source_event_sha256') != digest(clip)
            or record.get('cast_sha256') != digest(cast)
            or dialogue_texts(caption) != [normal(x['text']) for x in lines]):
        raise Pending('Final cast/reference/speech binding changed; do not submit H3')
    if not lines and not allow_original_legacy:
        direction = caption.partition('\n\ndetailed_description:\n')[2].partition('\n\noverall_soundscape:\n')[0]
        if (record.get('nonverbal_policy') != SILENT_POLICY_VERSION or record.get('speaking_bindings')
                or caption.count(SILENT_POLICY) != 1 or not direction
                or dialogue_integrity.silent_speech_cue(direction)
                or re.search(r'<\s*/?\s*d\b', caption, re.I)
                or any(r['kind'] == 'video' and r.get('include_audio') for r in references)):
            raise Pending('Nonverbal scene needs a clean H3 prompt and visual-only video references')
        sound_ids = {sound['asset_id'] for sound in record.get('sound_bindings', [])}
        if any(ref['kind'] == 'audio' and ref['id'] not in sound_ids for ref in references):
            raise Pending('Nonverbal scene has an unapproved audio reference')
    for row in record.get('speaking_bindings',[]):
        if row.get('canonical_voice_asset_id'):
            ref=next((x for x in references if x['id']==row['voice_asset_id']),None)
            try:
                if ref is None or voice_lineage.check(ref,config['generation_contract']['cast'][row['speaker']]['voice_asset_id'],row['speaker'])!=row.get('voice_lineage_authority'):raise ValueError('Voice lineage differs from compiled binding')
            except (ValueError,OSError,KeyError) as exc:
                raise Pending('Voice clone lineage changed before H3 submission: '+str(exc)) from exc
    for sound in record.get('sound_bindings',[]):
        ref=next((r for r in references if r['id']==sound['asset_id']),None)
        a=sound['authority']
        if ref is None or ref.get('sound_reference_authority')!=a or sha(a['path'])!=a['sha256']:
            raise Pending('Background audio authority changed before generation')
    for ref in references:
        if sha(ref['path']) != ref['sha256']:
            raise Pending('Reference bytes changed after canonical binding')
    return True


def admit_gpu(store, physical, config=None):
    import subprocess
    config=config or {}
    if not task_parameters.admission(config,physical):
        raise Pending('Worker device is outside its explicit physical GPU authorization',status='awaiting_resource')
    target=str(physical)
    try:
        free = int(subprocess.check_output(['nvidia-smi','-i',target,'--query-gpu=memory.free','--format=csv,noheader,nounits'],timeout=15).decode().strip())
        compute = subprocess.check_output(['nvidia-smi','-i',target,'--query-compute-apps=pid','--format=csv,noheader,nounits'],timeout=15).decode().strip()
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        raise Pending(f'GPU{target} admission unavailable: '+str(exc),status='awaiting_resource') from exc
    if free < 79000 or compute:
        raise Pending(f'GPU{target} waiting for >=79000MiB free and no compute processes; free={free}MiB',status='awaiting_resource')
    store.event('repair_gpu_admitted',physical_gpu=int(target),free_mib=free)
