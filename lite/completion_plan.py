"""Lossless, unsubmitted-only speech partitioning before a real GPU request."""
from copy import deepcopy
import math
import re

from . import content, duration_budget
from .storage import Pending, key, read, sha, write


def _ceil(value):
    return math.ceil(value * 24 - 1e-7) / 24


def occurrences(clips):
    return [(d.get('speaker'), d['text']) for c in clips for d in content.dialogue(c)]


def _segments(line, seconds):
    text = line['text']
    if seconds <= 9.2:
        return [(deepcopy(line), seconds)]
    sentences = [x for x in re.split(r'(?<=[。！？!?；;])|(?<=\.)(?=\s)', text) if x]
    # A long sentence may already contain a deliberate spoken clause boundary.
    # Preserve the dash and every space; never split hyphenated words, numeric
    # ranges or arbitrary words merely to evade the shot/attempt budget.
    total = max(1, content.speech_units(text)[1])
    sentences = [part for sentence in sentences for part in (
        re.split(r'(?<= —)(?=\s)|(?<= –)(?=\s)', sentence)
        if seconds * content.speech_units(sentence)[1] / total > 9.2
        else [sentence]) if part]
    # Retain existing clause punctuation and every character. Only overlong,
    # still-unsubmitted speech uses these finer boundaries; digits such as
    # 1,000 and 7:30 have no following whitespace and remain indivisible.
    sentences = [part for sentence in sentences for part in (
        re.split(r'(?<=[,:])(?=\s)|(?<=[，：])', sentence)
        if seconds * content.speech_units(sentence)[1] / total > 9.2
        else [sentence]) if part]
    if ''.join(sentences) != text or len(sentences) < 2:
        raise Pending('One complete sentence exceeds the planning window; source-preserving text planning is required', status='needs_duration_plan')
    result = []
    for sentence in sentences:
        duration = seconds * content.speech_units(sentence)[1] / total
        if duration > 9.2:
            raise Pending('A complete sentence cannot fit a 10-second source-preserving shot', status='needs_duration_plan')
        result.append(({**deepcopy(line), 'text': sentence}, duration))
    return result


def _pack(clip, catalog):
    estimates = content.speech_budget(clip, catalog)['utterances']
    units = [unit for line, estimate in zip(content.dialogue(clip), estimates)
             for unit in _segments(line, estimate['estimated_speech_seconds'])]
    groups, group, length = [], [], .8
    for unit in units:
        gap = .12 if group else 0
        if group and length + gap + unit[1] > 10 + 1e-7:
            groups.append((group, max(4, _ceil(length))))
            group, length, gap = [], .8, 0
        group.append(unit[0]); length += gap + unit[1]
    if group:
        groups.append((group, max(4, _ceil(length))))
    return groups


def _no_render_claim(agent, cid):
    if any(c.get('receipt') or c.get('technical_failure') or c.get('caption')
            for c in agent.s.data.get('candidates', {}).get(cid, [])):
        return False
    # request.json is already an owned immutable input even if the submission
    # acknowledgement has not yet arrived. Never rename/split it on re-entry.
    return not any(
            any((agent.s.run/'media/videos').glob(cid + '_take_*/'+name))
            for name in ('request.json','submitted.json','receipt.json'))


def budget_clips(agent, clips):
    try:return _budget_clips(agent,clips)
    except Pending:raise
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        raise Pending('Frame reservation needs valid native/request/plan durations; preserve original inputs',status='needs_duration_plan') from exc


def _budget_clips(agent, clips):
    """Reserve each logical clip once using integer frames, never rewrite input.

    Actual candidate receipts and owned request durations are alternatives for
    the SAME shot, not appendable shots. Retain the largest reservation until a
    candidate is selected. The selected prefix is accounted separately.
    """
    result = deepcopy(clips)
    catalog=agent.catalog() if callable(getattr(agent,'catalog',None)) else []
    for clip in result:
        planned=duration_budget._frames(clip['seconds']);reserved=planned
        sources=[];claimed=False
        for candidate in agent.s.data.get('candidates',{}).get(clip['id'],[]):
            claimed=claimed or bool(candidate.get('caption') or candidate.get('receipt') or candidate.get('technical_failure'))
            if candidate.get('receipt'):
                try:frames=duration_budget.receipt_frames(candidate['receipt'])
                except (ValueError,TypeError,KeyError) as exc:
                    raise Pending('Retained state receipt cannot bound native duration; reconcile original media',status='needs_duration_plan') from exc
                reserved=max(reserved,frames)
                sources.append({'kind':'retained_state_receipt','take':candidate.get('take'),'frames':frames,
                    'media_validation':'reservation_only; media bytes are verified by the render/review path'})
        root=getattr(agent.s,'run',None)
        folders=sorted((root/'media/videos').glob(clip['id']+'_take_*')) if root else []
        for folder in folders:
            if any((folder/name).exists() for name in ('request.json','submitted.json','receipt.json')):claimed=True
            if (folder/'request.json').is_file():
                try:request=read(folder/'request.json');seconds=request.get('target',{}).get('duration_seconds')
                except (OSError,ValueError,TypeError,AttributeError) as exc:
                    raise Pending('Owned render request cannot be read; preserve it for reconciliation',status='needs_duration_plan') from exc
                if seconds is not None:
                    try:frames=duration_budget._frames(seconds)
                    except (ValueError,TypeError,OverflowError) as exc:
                        raise Pending('Owned request duration is invalid; preserve it for reconciliation',status='needs_duration_plan') from exc
                    reserved=max(reserved,frames)
                    sources.append({'kind':'owned_request','frames':frames,'path':str(folder/'request.json'),
                        'actual_sentence_tail':'unknown_until_native_receipt'})
            path=folder/'receipt.json'
            if not path.is_file():continue
            try:
                receipt=read(path);frames=duration_budget.receipt_frames(receipt)
                if (receipt.get('native_audio') is not True or sha(receipt['path'])!=receipt.get('sha256')
                        or sha(receipt['audio_path'])!=receipt.get('audio_sha256')):
                    raise ValueError('Retained media digest/native audio mismatch')
            except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
                raise Pending('Retained native receipt differs; do not regenerate',status='needs_duration_plan') from exc
            reserved=max(reserved,frames)
            sources.append({'kind':'verified_native_receipt','frames':frames,'path':str(path)})
        minimum=max(96,duration_budget._frames(clip.get('minimum_seconds',4)))
        if not claimed:
            speech=content.speech_budget(clip,catalog)
            utterances=speech['utterances']
            if utterances:
                minimum=max(minimum,duration_budget._frames(sum(u['estimated_speech_seconds'] for u in utterances)+.8+max(0,len(utterances)-1)*.12))
            reserved=max(reserved,minimum)
        clip['seconds']=reserved/24
        clip['_frame_budget']={'planned_frames':planned,'reserved_frames':reserved,
            'minimum_future_frames':minimum if not claimed else None,'owned_input':claimed,
            'minimum_basis':'planning estimate/explicit hold, not observed speech duration',
            'sources':sources}
    return result


def prepare(agent):
    """An internal plan revision never renames an existing paid operation."""
    agent.s.check_stop()
    selected = {s['clip']['id'] for s in agent.s.data['selected']}
    items = agent.s.data['plan']['clips']
    catalog = agent.catalog()
    number = max([int(c['id'].rsplit('_', 1)[1]) for c in items], default=0)
    result, revisions = [], []
    for clip in items:
        if clip['id'] in selected or not _no_render_claim(agent, clip['id']):
            result.append(deepcopy(clip)); continue
        groups = _pack(clip, catalog)
        if not groups:
            result.append(deepcopy(clip)); continue
        if len(groups) == 1 and groups[0][0] == content.dialogue(clip):
            # A small safe duration extension does not change event, text or IDs.
            duration = max(clip['seconds'], groups[0][1])
            copy = deepcopy(clip)
            if duration != clip['seconds']:
                copy['seconds'] = duration
                revisions.append({'parent_clip_id':clip['id'],'kind':'duration_extension',
                    'before':clip['seconds'],'after':duration,'original_source':deepcopy(clip)})
            result.append(copy); continue
        original_clip = clip
        from . import completion_flow
        clip = completion_flow.execution_clip(agent, clip)
        receipt_id = key({'source': original_clip, 'groups': groups})
        children = []
        for index, (lines, duration) in enumerate(groups):
            child = deepcopy(clip)
            child.pop('_replan_direction', None)
            child.pop('_replan_endpoints', None)
            if index and clip.get('sound_asset_ids'):
                # One sparse cue must not multiply when a future event is split.
                child['asset_ids']=[i for i in child.get('asset_ids',[]) if i not in clip['sound_asset_ids']]
                child.pop('music_asset_id',None);child.pop('sound_asset_ids',None)
            if index:
                number += 1
                child['id'] = f'clip_{number:03d}'
            child.update(dialogue=lines, seconds=duration,
                event_id=clip.get('event_id', clip['id'])+f':part_{index+1}',
                completion_split={'parent_clip_id':clip['id'],'parent_event_id':clip.get('event_id'),
                    'part':index+1,'parts':len(groups),'revision_receipt_id':receipt_id,
                    'original_source':deepcopy(clip),'text_is_exact_partition':True})
            first, last = index == 0, index == len(groups)-1
            old = clip.get('render_spec', {})
            child['action'] = ('Begin' if first else 'Continue') + ' this same event through only the assigned dialogue units; ' + (
                'complete its final consequence once.' if last else 'leave its final consequence for the following assigned shot.')
            child['render_spec'] = {
                'opening': old.get('opening', 'A clear continuous shot in the retained scene.') if first else
                    'A deliberate new shot in the same scene immediately after the preceding assigned speech; preserve positions and current physical state.',
                'action': 'Perform only the assigned speech naturally and completely. ' + (
                    'Complete the retained final consequence once: ' + old.get('action', '') if last else
                    'Advance the opening action without repeating a completed movement or performing the final consequence early.'),
                'camera': old.get('camera', 'A stable single shot.'),
                'closing': old.get('closing', 'Hold the completed action and silent mouth.') if last else
                    'Finish the assigned speech and hold a stable silent closing moment; the same event continues in the next deliberate shot.'}
            child['entry_state']=clip.get('entry_state','') if first else 'Continue the prior assigned shot in the same scene and physical state.'
            child['exit_state']=clip.get('exit_state','') if last else 'Assigned speech completed; final parent consequence remains for the following shot.'
            child['shot_design']={'opening':child['render_spec']['opening'],'closing':child['render_spec']['closing'],
                'camera':child['render_spec']['camera'],'cut_reason':'Deliberate dialogue coverage cut without missing or repeated speech',
                'carry_state':child['entry_state']}
            from . import shot_policy
            if shot_policy.contract_enabled(agent.config) and not last:
                # Original physical consequence is assigned ONCE to the final part.
                # Earlier coverage preserves state and only performs assigned speech.
                child['action_beats']=[{'action':'Keep the established physical state while performing only the assigned speech',
                    'visible_result':'The assigned speech ends completely; the final parent action remains unperformed'}]
                child['state_checkpoints']={ident:{'entry':state['entry'],'exit':state['entry']}
                    for ident,state in clip.get('state_checkpoints',{}).items()}
            child['microshots']=[{'action':child['action'],'camera':child['render_spec']['camera'],'sound':'Only the assigned speech, retained natural action sounds and ambience.'}]
            child['parent_event_action'] = clip['action']
            child.pop('dialogue_timeline', None)
            child['title'] = clip.get('title', clip['id']) + f' ({index+1}/{len(groups)})'
            children.append(child)
        # Bind every child speech fragment to its original speaker occurrence and
        # exact character interval. Deliberate duplicate lines retain distinct IDs.
        originals=content.dialogue(clip);ordinal=offset=0
        for child in children:
            for line in child['dialogue']:
                parent=originals[ordinal];end=offset+len(line['text'])
                if parent['speaker']!=line['speaker'] or parent['text'][offset:end]!=line['text']:
                    raise ValueError('Speech partition changed role, order or occurrence')
                line['completion_source_occurrence']={'parent_clip_id':clip['id'],'script_index':ordinal,'start_character':offset,'end_character':end}
                authority=clip.get('dialogue_timeline',[])
                if ordinal<len(authority) and authority[ordinal].get('on_screen_or_offscreen'):
                    line['on_screen_or_offscreen']=authority[ordinal]['on_screen_or_offscreen']
                offset=end
                if offset==len(parent['text']):ordinal+=1;offset=0
        if ordinal!=len(originals) or offset:
            raise ValueError('Speech partition omitted a source occurrence')
        # Lossless partition of a speaker occurrence may add a cut, never words.
        expected = ''.join(d['text'] for d in content.dialogue(clip))
        actual = ''.join(d['text'] for c in children for d in content.dialogue(c))
        if actual != expected:
            raise ValueError('Speech partition lost or added source text')
        result.extend(children)
        revisions.append({'parent_clip_id':clip['id'],'kind':'lossless_speech_partition',
            'receipt_id':receipt_id,'original_source':deepcopy(original_clip),'children':deepcopy(children),
            'gpu_submissions_before_partition':0,'original_paid_text_operations_retained':True})
    future = [c for c in result if c['id'] not in selected]
    budgeted=budget_clips(agent,future)
    problems = duration_budget.issues(agent.config,budgeted,agent.s.data['selected'])
    if not revisions and not problems:
        duration_budget.record(agent,budgeted)
        return False
    if problems:
        # Only future silence/action holds may release planning slack. No generated
        # frame, dialogue, action description or ending is removed or slowed down.
        excess = sum(duration_budget._frames(c['seconds']) for c in budgeted) + sum(s['receipt']['frames'] for s in agent.s.data['selected']) - duration_budget._frames(agent.config['max_final_seconds'])
        for c in sorted(future, key=lambda x:x['seconds'], reverse=True):
            if excess <= 0:
                break
            if content.dialogue(c) or not _no_render_claim(agent, c['id']):
                continue
            before_frames=duration_budget._frames(c['seconds'])
            removable=max(0,before_frames-max(96,duration_budget._frames(c.get('minimum_seconds',4))))
            released = min(excess, removable)
            if released:
                before=c['seconds'];c['seconds']=(before_frames-released)/24;excess-=released
                revisions.append({'parent_clip_id':c['id'],'kind':'future_action_hold_reallocation',
                    'before':before,'after':c['seconds'],'action_and_ending_unchanged':True,'media_edited':False})
    budgeted=budget_clips(agent,future)
    try:duration_budget.admit(agent.config,budgeted,agent.s.data['selected'])
    except Pending:
        duration_budget.record(agent,budgeted,committed=False)
        raise
    ids = [c['id'] for c in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Lossless plan revision introduced duplicate clip IDs')
    prefix = [c for c in result if c['id'] in selected]
    original_prefix = [c for c in items if c['id'] in selected]
    if prefix != original_prefix:
        raise ValueError('Committed prefix changed during speech planning')
    revision = agent.s.data['plan_revision'] + 1
    receipt = {'version':'lossless_unsubmitted_plan_v1','revision':revision,'changes':revisions,
        'actual_prefix_frames':sum(s['receipt']['frames'] for s in agent.s.data['selected']),
        'planned_future_frames':sum(round(c['seconds']*24) for c in future),
        'source_prompt_unchanged':True,'original_media_and_call_history_unchanged':True,'take_counts_reset':False}
    # Write recovery evidence before the atomic state commit. Re-entry computes
    # exactly the same children, never another set of clips or another request.
    path=agent.s.run/'plans'/f'completion_revision_{revision}.json'
    write(path, {'receipt':receipt,'before':deepcopy(items),'after':deepcopy(result)})
    agent.s.data['plan']['clips'] = result
    agent.s.data['plan_revision'] = revision
    agent.s.data.setdefault('completion_plan_revisions',[]).append({'path':str(path),'receipt':receipt})
    duration_budget.record(agent,budgeted,save=False)
    agent.s.save()
    agent.s.event('completion_plan_committed', revision=revision, evidence=str(path), changes=len(revisions), gpu_dispatches=0)
    return True
