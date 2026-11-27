"""Opt-in frame budget for new short stories. Never truncates generated media."""
import math
import re
from copy import deepcopy
from .storage import Pending,read,write
from . import task_parameters

def enabled(config):
    return config.get('duration_admission') == 'short_story_frames_v1'

def _frames(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('Duration must be finite and positive')
    return math.ceil(value*24-1e-7)

def receipt_frames(receipt):
    """Native frame counts, never rounded seconds or boolean-valued integers."""
    if not isinstance(receipt,dict):raise ValueError('Native receipt must be an object')
    frames=receipt.get('frames')
    if isinstance(frames,bool) or not isinstance(frames,int) or frames<=0 or receipt.get('fps')!=24:
        raise ValueError('Actual media requires positive integer 24fps native frames')
    return frames

def issues(config, proposed, selected=()):
    extended = task_parameters.longform(config)
    if extended:
        task_parameters.validate(config)
    if not enabled(config):return []
    try:
        limit=_frames(config['max_final_seconds'])
        revised = config.get('task_contract', {}).get('version') == task_parameters.REVISED_VERSION
        if revised: task_parameters.validate(config)
        maximum = task_parameters.LONGFORM_MAX_SECONDS * 24 if extended else (7200 if revised else 4320)
        if limit>maximum:raise ValueError(f'Story limit must not exceed {maximum//24} seconds')
        if not isinstance(proposed,list):raise ValueError('Future plan must be a clips list')
        prefix=0
        for row in selected:
            prefix+=receipt_frames(row['receipt'])
        future=sum(_frames(row['seconds']) for row in proposed)
        if extended and prefix + future < task_parameters.LONGFORM_MIN_SECONDS * 24:
            return [{'kind': 'duration_budget_below_minimum', 'actual_prefix_frames': prefix,
                     'future_frames': future, 'minimum_frames': task_parameters.LONGFORM_MIN_SECONDS * 24,
                     'instruction': 'Budget complete actions, dialogue and reactions across the remaining scenes; never pad or slow final media.'}]
        if prefix+future>limit:
            return [{'kind':'duration_budget_exceeded','actual_prefix_frames':prefix,'future_frames':future,'limit_frames':limit,'instruction':'Revise text before GPU submission; preserve complete speech and ending; never truncate media'}]
        return []
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        return [{'kind':'invalid_duration_budget','reason':str(exc)}]

def director_summary(config, clips, *, source_scene_ids=()):
    """Host arithmetic over normalized shot targets, before any media exists."""
    if not task_parameters.longform(config):
        return None
    problems = issues(config, clips)
    scene_totals = {}
    for clip in clips:
        scene = clip.get('scene_id') or clip.get('scene')
        if not isinstance(scene, (str, int)) or isinstance(scene, bool):
            scene = None
        if isinstance(scene, str):
            scene = scene.strip() or None
        if source_scene_ids and isinstance(scene, str):
            shot_code = re.fullmatch(r'(S[1-9][0-9]*)-[0-9]+[A-Za-z]?', scene)
            if shot_code and shot_code[1] in source_scene_ids:
                scene = shot_code[1]
        row = scene_totals.setdefault(scene, {'scene': scene, 'shot_count': 0, 'frames': 0})
        row['shot_count'] += 1
        row['frames'] += _frames(clip['seconds'])
    for row in scene_totals.values():
        row['seconds'] = row['frames'] / 24
    total = sum(row['frames'] for row in scene_totals.values())
    return {'scope': 'planned_shot_reservations', 'fps': 24, 'shot_count': len(clips),
            'total_frames': total, 'total_seconds': total / 24,
            'hard_range_seconds': [task_parameters.LONGFORM_MIN_SECONDS, config['max_final_seconds']],
            'preferred_range_seconds': [590, 605], 'within_hard_bounds': not problems,
            'by_scene': list(scene_totals.values()),
            'instruction': 'Use these host-calculated frame counts for all duration judgments. '
                'Do not add or cut time to satisfy a conflicting estimated total. '
                'Evaluate pacing, dialogue, scene coverage, visuals and music on their own evidence.'}


def admit(config, proposed, selected=()):
    bad=issues(config,proposed,selected)
    if bad:raise Pending('Short-story text budget requires correction: '+str(bad),status='needs_duration_plan')

def record(agent,proposed,selected=None,*,committed=True,save=True):
    """Small deterministic ledger; no new locks, model calls or timing authority.

    Failed staging writes a diagnostic only, so an uncommitted hold reduction
    cannot be mistaken for a production state revision.
    """
    if not enabled(agent.config):return None
    selected=agent.s.data.get('selected',[]) if selected is None else selected
    rows=[]
    for clip in proposed:
        detail=deepcopy(clip.get('_frame_budget',{}))
        rows.append({'clip_id':clip.get('id'),**detail,'reserved_frames':_frames(clip['seconds'])})
    prefix=sum(receipt_frames(row['receipt']) for row in selected)
    future=sum(row['reserved_frames'] for row in rows);limit=_frames(agent.config['max_final_seconds'])
    value={'version':'integer_frame_ledger_v1','fps':24,'budget_admitted':committed,
        'scope':'frame_reservation_snapshot_not_a_plan_commit',
        'actual_selected_frames':prefix,'future_reserved_frames':future,'total_reserved_frames':prefix+future,
        'remaining_frames':limit-prefix-future,'limit_frames':limit,'future':rows,
        'minimum_future_frames':sum(row.get('minimum_future_frames') or 0 for row in rows),
        'unmeasured_native_tails_are_unknown':True,'does_not_authorize_dialogue_edits':True}
    root=getattr(agent.s,'run',None)
    if root and read(root/'duration_ledger.json')!=value:write(root/'duration_ledger.json',value)
    if committed and agent.s.data.get('duration_ledger')!=value:
        agent.s.data['duration_ledger']=value
        save_state=getattr(agent.s,'save',None)
        if save and callable(save_state):save_state()
    return value

def local_reservations(agent, clip, duration):
    """Account for the current shot, locked alternatives and the entire unused suffix."""
    chosen={x['clip']['id'] for x in agent.s.data['selected']}
    suffix=[x for x in agent.s.data['plan']['clips'] if x['id'] not in chosen and x['id']!=clip['id']]
    from .completion_plan import budget_clips
    planned=next((x for x in agent.s.data['plan']['clips'] if x['id']==clip['id']),{})
    current={**planned,**clip,'seconds':duration}
    return budget_clips(agent,[current]+suffix)

def local(agent, clip, duration, receipt=None):
    if not enabled(agent.config):return
    if receipt is not None:
        try:duration=receipt_frames(receipt)/24
        except (ValueError,TypeError) as exc:
            raise Pending('Native media requires 24fps positive frames',status='needs_duration_plan') from exc
    proposed=local_reservations(agent,clip,duration)
    bad=issues(agent.config,proposed,agent.s.data['selected'])
    record(agent,proposed,committed=not bool(bad))
    if bad:raise Pending('Short-story native/request frame budget requires correction: '+str(bad),status='needs_duration_plan')

def final(config, selected):
    admit(config,[],selected)
