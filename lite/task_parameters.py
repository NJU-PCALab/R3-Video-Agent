"""Bind an authorized benchmark task to devices, language and existing prompts.

No planning, review, repair or selection policy is implemented here.
Legacy operations without this opt-in retain their original parameters.
"""
from copy import deepcopy
from pathlib import Path
import json
from .storage import Pending

VERSION = 'benchmark_task_parameters_v1'
REVISED_VERSION = 'benchmark_task_parameters_v2'
LONGFORM_VERSION = 'longform_task_parameters_v1'
LONGFORM_MIN_SECONDS = 570
LONGFORM_TARGET_SECONDS = 600
LONGFORM_MAX_SECONDS = 630


def longform(config):
    return config.get('task_contract', {}).get('version') == LONGFORM_VERSION


def enabled(config):
    return config.get('task_contract', {}).get('version') in (VERSION, REVISED_VERSION, LONGFORM_VERSION)


def language(config):
    return config['task_contract']['language'] if enabled(config) else 'Chinese'


def admission(config, physical):
    if not enabled(config):
        return str(physical) == '1'
    task = config['task_contract']
    assigned = task.get('physical_gpu')
    if isinstance(assigned, bool) or assigned not in (0, 1) or str(physical) != str(assigned):
        return False
    if any(str(config.get(name, {}).get('device')) != str(assigned)
           for name in ('image_provider', 'audio_provider')):
        return False
    try:
        auth = json.loads(Path(config['authorization']).read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if longform(config) and (not isinstance(auth, dict) or not task.get('benchmark_id')
                            or not isinstance(auth.get('physical_gpus'), list)
                            or not isinstance(auth.get('story_indices'), list)
                            or task.get('story_index') is None):
        return False
    return (auth.get('benchmark_id') == task.get('benchmark_id')
            and assigned in auth.get('physical_gpus', [])
            and task.get('story_index') in auth.get('story_indices', []))


def validate(config):
    if not enabled(config):
        return
    task = config['task_contract']
    if task.get('language') not in ('Chinese', 'English'):
        raise Pending('Benchmark task needs its explicit Chinese or English language', status='needs_configuration')
    if not admission(config, task.get('physical_gpu')):
        raise Pending('Assigned physical GPU differs from this task authorization or worker devices', status='needs_configuration')
    lo, hi = task.get('min_seconds'), config.get('max_final_seconds')
    target = config.get('target_seconds')
    if longform(config):
        auth = json.loads(Path(config['authorization']).read_text())
        cap = task.get('director_max_output_tokens', 24000)
        if type(cap) is not int or not 24000 <= cap <= 32768:
            raise Pending('Long-film director output limit must be 24000–32768 tokens', status='needs_configuration')
        if (lo != LONGFORM_MIN_SECONDS or hi != LONGFORM_MAX_SECONDS
                or task.get('max_seconds') != hi or target != LONGFORM_TARGET_SECONDS
                or auth.get('duration_seconds') != [lo, hi]
                or config.get('duration_admission') != 'short_story_frames_v1'):
            raise Pending('Ten-minute films require a 600-second target, an authorized '
                          '570–630-second window and 24-fps duration admission', status='needs_configuration')
        return
    expected_max = 300 if task.get('version') == REVISED_VERSION else 180
    if task.get('version') == REVISED_VERSION:
        auth = json.loads(Path(config['authorization']).read_text())
        if task.get('max_seconds') != 300 or auth.get('duration_seconds') != [150, 300]:
            raise Pending('Revised duration requires the authorized 150–300 contract', status='needs_configuration')
    if (lo != 150 or hi != expected_max or isinstance(target, bool)
            or not isinstance(target, (int, float)) or not lo <= target <= hi):
        raise Pending(f'Benchmark duration parameters must bind 150–{expected_max} seconds', status='needs_configuration')


def prompt(value, config, *, screenplay=False):
    if not enabled(config):
        return value
    revised = config['task_contract'].get('version') == REVISED_VERSION
    if longform(config):
        bounds = f'{LONGFORM_MIN_SECONDS}–{LONGFORM_MAX_SECONDS}'
        plan_guidance = ('Plan close to 600 seconds, preferably 590–605, in complete 4-to-10-second shots. '
                         'Use compact JSON, concise descriptions and canonical asset IDs; omit redundant '
                         'restatements. Include the full ordered shot list and use source scene timings for pacing. '
                         'Preserve complete speech and the ending.')
        if screenplay:
            plan_guidance = ('Work at scene level. In a screenplay, provide one scene object per source scene with its '
                             'scene_id, actions, changes in appearance, and complete attributed dialogue. '
                             'Follow source timing, causal order and ending. The director expands these scenes into shots. '
                             'In an editorial review, return only the specified review fields.')
        value = value.replace('通常全片180至300秒', f'全片{bounds}秒')
        value = value.replace('Short-story hard limit:', 'Film duration limit:')
    else:
        bounds = '150–300' if revised else '150–180'
        plan_guidance = ('Plan 210–240 seconds initially to retain repair margin; preserve complete speech and ending.' if revised
                         else 'Plan 155–165 seconds initially to retain repair margin; preserve complete speech and ending.')
    value = value.replace('总时长目标约300秒', f'总时长遵循当前任务的{bounds}秒要求')
    value = value.replace('所有人物是成年人。', '人物年龄与故事输入一致，保留其明确要求的儿童、衰老和变形。')
    value = value.replace('描述成年角色音色', '描述故事中该角色的音色')
    value = value.replace('中文对白是否真实说出', '指定语言的对白是否真实说出')
    return value + ('\nTask parameters are authoritative production inputs, not observations: '
        + f'All character dialogue and any narration must be {language(config)}. '
        + 'Retain the source culture, setting, ages and declared transformations. '
        + 'Choose appearances and visual style consistently with the source; do not inherit another story. '
        + f'Target {config["target_seconds"]} seconds; complete final film {bounds} seconds. '
        + plan_guidance)


def voice_spec(spec, config):
    if not enabled(config) or spec.get('role') != 'voice':
        return spec
    from .audio_parameters import tts_language
    value = deepcopy(spec)
    if not value.get('language'):
        value['language'] = language(config)
    if tts_language(value.get('language')) != language(config):
        raise Pending('Voice asset language conflicts with its source task; correct the asset input', status='needs_asset_repair')
    return value


def asr_request(request, config):
    if not enabled(config):
        return request
    return {**request, 'language': language(config).lower()}


def shot_context(config):
    if not enabled(config):
        return 'live-action ancient-Chinese drama'
    return 'narrative film'


def duration_correction(config):
    if longform(config):
        window = '590-605'
    elif not enabled(config):
        window = '135-150'
    else:
        window = '210-240' if config['task_contract'].get('version') == REVISED_VERSION else '155-165'
    return f'Return a complete executable plan including assets and clips, {window} seconds total, preserve ending and all dialogue. Do not cut media.'
