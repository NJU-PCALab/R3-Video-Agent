"""Opt-in editorial instructions and reference capabilities, not content PASS rules."""
from copy import deepcopy


VERSION = 'shot_cuts_v1'


def enabled(config):
    policy = config.get('editing_policy')
    return isinstance(policy, dict) and policy.get('version') == VERSION


def no_score(config):
    return enabled(config) and (config.get('editing_policy') or {}).get('background_music', 'none') == 'none'


def is_music(asset):
    return asset.get('role') in ('music', 'background_music', 'score') or asset.get('generator') == 'music' or asset.get('track_type') == 'generated_music'


def excluded_reason(asset, config):
    if not enabled(config):
        return None
    if asset.get('role') == 'continuity' or str(asset.get('track_type') or '').startswith('mixed'):
        return 'Previous mixed soundtrack/pose/motion is observation-only, not a continuation condition.'
    if str(asset.get('id', '')).startswith(('last_clip_', 'motion_clip_', 'audio_clip_')):
        return 'Derived previous-clip references do not constrain the next first frame.'
    if no_score(config) and is_music(asset):
        return 'This experiment uses no non-diegetic score; retain speech and diegetic ambience.'
    return None


def visible_catalog(catalog, config):
    return [a for a in catalog if excluded_reason(a, config) is None]


def context(config):
    if not enabled(config):
        return {}
    return {'version': VERSION, 'authority': 'user_authorized_generation_instructions_not_observations',
            'generation': 'Ref2VA self-contained editorial shots; no automatic FL2VA switch',
            'boundary': 'New shot at clip opening; finish a complete action/utterance before clip end.',
            'continuity': 'Preserve narrative state, identity, props, screen direction and cause; do not promise pixel/pose matching.',
            'forbidden_continuation': 'Do not replay the previous clip or split a continuous camera move/action across generations.',
            'background_music': 'none' if no_score(config) else 'explicit_model_planned_cues_only',
            'target_total_seconds': config.get('target_seconds', 240),
            'past_experience': 'Retain old lessons but this current editing/music policy overrides incompatible continuation/music advice.'}


def instructions(config, *, director_layout=False):
    if not enabled(config):
        return ''
    text = '''\n当前优先制作策略 shot_cuts_v1（覆盖旧指南中的自动衔接建议）：
每个clip是一个完整、可剪接的镜头单元，不是把同一连续长镜头分成多个生成块。
clip开头必须是有意设计的新镜头：换景别、明显换机位、正反打、道具插入或有动机的时间/场景切换；
不要沿用同一构图再重置姿态，不要求新首帧复制上一尾帧，不重演前片段已完成的动作或台词。
每段写shot_design:{opening,closing,camera,cut_from_previous,cut_reason,carry_state}，它是拍摄意图，不是已观察事实。
前段应完成动作结果、整句台词并留下短暂可剪接的稳定结尾；后段从结果后的新动作开始。
不得把同一走路、抬手、开门、递物动作或一句话硬劈在两个生成段中；复杂动作需重排成建立/反应/细节等完整镜头。
多段对白可用有意义的正反打，但每段句子说完再切，保留说话人、视线轴线、道具位置和时间逻辑。
Ref2VA参考用于人物、服装、场景、音色，不是精确首尾帧条件。上一段真实媒体只供观察与剪辑判断。
本轮不自动切换FL2VA，不输出first_frame_id/last_frame_id，也不选择last_clip_、motion_clip_、audio_clip_衔接素材。
原有asset_ids和音色能力仍可用；缺少shot_design字段不制造宿主异常，不把制作意图当评审PASS。
全局RSI可重排未生成后缀的镜头、时长和资产；冻结已选前缀，为下一新镜头设计有效切点，不承诺修复旧画面。
评审必须区分合理硬切与同机位的动作重放/姿态跳变：新镜头首尾画面不同本身不是缺陷。
'''
    if director_layout:
        text = ''.join(line for line in text.splitlines(keepends=True) if 'shot_design' not in line)
    if no_score(config):
        text += '''本轮非剧情背景配乐关闭：不要设计或引用music/score资产，不复制上一混合声轨，不让每段重启同一乐句。
保留原生对白、脚步、风、衣料、门声等剧情内声音；不要用配乐掩盖必要音效。caption明确写无背景配乐。
原稿确有可见乐器演奏时，才作为剧情内声音描述；不得自行添歌唱、旁白或乐器演奏。
音频/声画评审将意外配乐、跨切点配乐突起、句子被切断等作为具体finding，按意见迭代，分数只作参考。
'''
    else:
        text += '''配乐不是每段必选参考。先规划叙事用途、情绪、进入/退出点、留白和对白遮蔽风险。
只有有明确用途的镜头才选music资产；连续音乐段选择明确reference_windows，不要每clip默认从第0秒重启。
参考窗口不能保证输出乐句连续，实际音频评审发现突起、错情绪或遮蔽时，局部可取消配乐参考，
全局调整未生成后缀的配乐安排，不把同一音频参考当成已连续的配乐，不复制上一混合声轨。
'''
    return text


def effective_caption(caption, clip, config):
    if not enabled(config):
        return caption
    # Raw model caption remains in plan. These are explicit production commands,
    # not inferred model decisions or claims about the generated media.
    overlay = ('EDITORIAL PRODUCTION INSTRUCTIONS: Begin a fresh, clearly intentional camera shot. '
               'Preserve character identity and the established narrative/prop state, not the previous framing or pose. '
               'Do not replay the preceding action. Keep each action beat and spoken sentence complete within this clip; '
               'end on its completed result with a short settled editing handle. '
               'Do not simulate a seamless continuation across independently generated clips.')
    design = clip.get('shot_design')
    if isinstance(design, dict):
        # The model's concrete shot choices stay distinguishable from the
        # fixed host instruction, and are never promoted to observed evidence.
        choices = [(key, design[key]) for key in ('camera', 'opening', 'closing')
                   if isinstance(design.get(key), str) and design[key].strip()]
        if choices:
            overlay += ' MODEL-PLANNED SHOT INTENT: ' + '; '.join(key + ': ' + value for key, value in choices) + '.'
    if no_score(config):
        overlay += (' SOUND POLICY: No non-diegetic music, background score, musical underscore, songs or musical stings. '
                    'Use only the scripted dialogue and plausible diegetic ambience/foley; do not replace required sounds with music.')
    return caption.rstrip() + '\n\n' + overlay


def intent(clip, config):
    return {'policy': context(config), 'model_shot_design': deepcopy(clip.get('shot_design')),
            'observed_boundary_quality': None}


CONTRACT_VERSION = 'bounded_shot_inputs_v1'


def contract_enabled(config):
    return config.get('production_contract', {}).get('version') == CONTRACT_VERSION


def validate_config(config):
    """Resolve one policy authority before any provider call; labels never override it."""
    from .storage import Pending
    from . import character_stages
    if character_stages.enabled(config) and config.get('execution_contract') != 'asset_worker_lifecycle_v1':
        raise Pending('Character-stage reference policy requires the shared asset/worker lifecycle quota contract', status='needs_configuration')
    if enabled(config) and config['editing_policy'].get('background_music', 'none') not in ('none', 'explicit_model_planned_cues_only'):
        raise Pending('Unknown background_music policy; no competing label may supply a default', status='needs_configuration')
    if contract_enabled(config):
        cap = config['production_contract'].get('max_action_beats', 3)
        if type(cap) is not int or not 1 <= cap <= 3:
            raise Pending('Action beats must use a bounded initial limit from 1 to 3', status='needs_configuration')


def contract_instructions(config):
    if not contract_enabled(config):
        return ''
    return """
Executable source contract bounded_shot_inputs_v1: each initial shot declares action_beats,
a list of at most 3 {action,visible_result} objects in English, in execution order.
Use one main action and only its essential support; separate overloaded actions into
complete shots BEFORE generation, never split an already submitted shot to gain attempts.
Optional state_checkpoints maps stable role/prop keys to {entry,exit} English physical
states. All entry_state/exit_state must also be English physical targets.
Use clip.on_screen_text, a JSON list of exact strings, for lettering on signs, screens,
balances, or placards, preserving its original language and numbers. Describe the
screen and its placement in English in render_spec, beats and state fields. Write
spoken monetary amounts in English in staging and keep the original words in dialogue.
For each canonical voice reference, use kind="audio", role="voice", the registered
voice asset ID and its exact speaker. Put descriptive usage labels in the name field.
These are planned obligations, not observed facts. Keep literal speech separate.
Local repairs may clarify physical staging, but cannot drop these source obligations.
Global planning sees actual prefix observations; a requested exit state is never proof
of the next entry state. Preserve uncertainty and address missing results in the existing
uncommitted suffix. Do not add a new reviewer, vote or retry loop.
"""


def project_missing_action_beats(plan, config, *, director_layout=False):
    """Copy the physical action and result already supplied by the director."""
    if not contract_enabled(config) or (
        not director_layout and config['production_contract'].get('project_missing_action_beats') is not True
    ):
        return plan
    clips = plan.get('clips')
    if not isinstance(clips, list):
        return plan
    projected = []
    positions = []
    for index, clip in enumerate(clips):
        if isinstance(clip, dict) and 'action_beats' not in clip:
            spec = clip.get('render_spec')
            if isinstance(spec, dict) and all(isinstance(spec.get(field), str) and spec[field].strip()
                                              for field in ('action', 'closing')):
                clip = {**clip, 'action_beats':[{'action':spec['action'], 'visible_result':spec['closing']}]}
                positions.append(index)
        projected.append(clip)
    if not positions:
        return plan
    return {**plan, 'clips':projected, 'host_action_beat_projection':{
        'clip_positions':positions, 'action_source':'render_spec.action', 'result_source':'render_spec.closing'}}


def validate_clip(clip, config):
    if not contract_enabled(config):
        return
    from .binding_contract import english_direction
    beats = clip.get('action_beats')
    if not isinstance(beats, list) or not 1 <= len(beats) <= config['production_contract'].get('max_action_beats', 3):
        raise ValueError('Initial shot needs 1-3 bounded action_beats before any asset generation')
    for beat in beats:
        if not isinstance(beat, dict) or set(beat) - {'action', 'visible_result'}:
            raise ValueError('Action beat has unsupported fields')
        if not all(english_direction(beat.get(k)) for k in ('action', 'visible_result')):
            raise ValueError('Each action needs its observable completed result')
    for field in ('entry_state', 'exit_state'):
        if clip.get(field) and not english_direction(clip[field]):
            raise ValueError('Shot entry/exit state must be physical English targets')
    states = clip.get('state_checkpoints', {})
    if not isinstance(states, dict) or len(states) > 8:
        raise ValueError('State checkpoints must be a bounded role/prop map')
    for ident, row in states.items():
        if not isinstance(ident, str) or not ident or not isinstance(row, dict) or set(row) != {'entry', 'exit'}:
            raise ValueError('Each state checkpoint needs one entry and exit target')
        if not all(english_direction(row[k]) for k in ('entry', 'exit')):
            raise ValueError('Checkpoint physical state must be explicit English')


def execution_direction(clip, plan):
    """Only bounded local framing may amend an approved global physical direction."""
    from .binding_contract import digest
    spec = clip.get('render_spec')
    if not clip.get('_replan_direction') or clip['_replan_direction'] != digest(spec):
        local = plan.get('render_spec') or spec
        if clip.get('_replan_endpoints') and isinstance(local, dict):
            return {**local, **clip['_replan_endpoints']}
        return local
    direction = deepcopy(spec)
    emphasis = plan.get('framing_emphasis')
    if emphasis == 'action':
        direction['camera'] += ' Keep the approved action visible throughout.'
    elif emphasis == 'result':
        direction['camera'] += ' Keep the final physical result unobstructed.'
    return direction


def source_conditions(clip, config, *, silent=False):
    """Compile frozen source obligations independently of optional local prose."""
    if not contract_enabled(config):
        return ''
    validate_clip(clip, config)
    if silent:
        from .dialogue_integrity import clean_silent_direction
        physical_fields = []
        for index, beat in enumerate(clip['action_beats'], start=1):
            physical_fields.extend(((f'Beat {index}', beat['action']), ('Visible result', beat['visible_result'])))
        physical_fields.extend((field, clip.get(field)) for field in ('entry_state', 'exit_state'))
        for row in clip.get('state_checkpoints', {}).values():
            physical_fields.extend((('Starting pose', row['entry']), ('Ending pose', row['exit'])))
        physical = [(key, clean_silent_direction(value)) for key, value in physical_fields]
        return ' '.join(f"{key}: {value.rstrip('.')}." for key, value in physical if value)
    lines = ['SOURCE ACTION/STATE TARGETS (not observations):']
    for n, beat in enumerate(clip['action_beats'], 1):
        lines.append(f"Beat {n}: {beat['action']}. Completed visible result: {beat['visible_result']}.")
    for field in ('entry_state', 'exit_state'):
        if clip.get(field):
            lines.append(field + ' target: ' + clip[field] + '.')
    for ident, row in clip.get('state_checkpoints', {}).items():
        # IDs are audit keys only; never send potentially non-English role names.
        lines.append(f"Physical state: begin with {row['entry']}; end with {row['exit']}.")
    return ' '.join(lines)
