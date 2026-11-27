"""A bounded completion path. Review uncertainty is data, never a receipt retry."""
from copy import deepcopy
import math
import re

from .storage import Pending, Unavailable, key, operation_recorded, read, write
from . import character_stages, shot_policy, completion_memory, task_parameters
from .binding_contract import digest

VERSION = 'completion_first_v1'


def enabled(config):
    return config.get('completion_policy', {}).get('version') == VERSION


def required_ids(agent):
    """Use the same dependencies as the actual reference resolver, including pending assets."""
    current = getattr(agent, '_completion_clip', None)
    selected = {s['clip']['id'] for s in agent.s.data['selected']}
    targets = [current] if current else [c for c in agent.s.data.get('plan', {}).get('clips', [])
                                      if c['id'] not in selected]
    return {ident for clip in targets for ident in (reference_ids(agent, clip, include_missing=True) + character_stages.required_image_ids(clip, agent.config))}


def defer_assets(agent, updates, operation):
    if not updates:
        return []
    ident = key({'operation': operation, 'updates': updates})
    receipts = agent.s.data.setdefault('completion_deferred_assets', {})
    if ident not in receipts:
        receipts[ident] = {'status': 'deferred_optional_optimization', 'operation': operation,
            'proposals': deepcopy(updates), 'executed': False,
            'reason': 'Stable cast and existing references retained; optional optimization is outside the render critical path.'}
        agent.s.save()
        agent.s.event('optional_assets_deferred', receipt=receipts[ident])
    return [receipts[ident]]


def execution_clip(agent, clip):
    """Build the effective shot from applied, same-run edits without rewriting receipts."""
    spec = None
    states = {}
    endpoints = {}
    state_to_frame = {'entry_state':'opening', 'exit_state':'closing'}
    # Standalone plan compilation can be used before a run state is materialized.
    run_state = agent.s.data
    checkpoints = run_state.get('replan_receipts', {}) if isinstance(run_state, dict) else {}
    for checkpoint in sorted((k for k in checkpoints if str(k).isdigit()), key=int):
        for row in checkpoints[checkpoint].get('receipts', []):
            if row.get('clip_id') != clip['id'] or row.get('status') != 'applied':
                continue
            changes = row['changes']
            if 'render_spec' in changes:
                new_spec = changes['render_spec']
                # A camera-only follow-up does not revoke accepted boundaries.
                for state, field in state_to_frame.items():
                    if spec is None or spec[field] != new_spec[field]:
                        states.pop(state, None)
                        endpoints.pop(field, None)
                spec = new_spec
            for state, field in state_to_frame.items():
                if state in changes:
                    states[state] = changes[state]
                    if 'render_spec' not in changes:
                        endpoints[field] = changes[state]
                    else:
                        endpoints.pop(field, None)
    if spec is None:
        endpoints = {field: endpoints[field] for state,field in state_to_frame.items()
                     if field in endpoints and (clip.get(state) == endpoints[field] or
                     field in clip.get('_replan_endpoints', {}) and clip['_replan_endpoints'][field] == clip.get(state))}
    raw_spec = clip.get('render_spec')
    if spec is None and not endpoints and not ('_replan_direction' in clip or '_replan_endpoints' in clip):
        return clip
    effective = {**(spec or raw_spec or {}), **endpoints}
    # Older receipts may predate literal speech hygiene. Clean the source once
    # before deriving its duplicate H3 action/beat and endpoint instructions.
    from . import dialogue_integrity
    if dialogue_integrity.enabled(agent.config):
        effective, _ = dialogue_integrity.visual_only(effective)
        endpoints = {k:effective[k] for k in endpoints}
        if states:
            bounds, _ = dialogue_integrity.visual_only({state_to_frame[k]:v for k,v in states.items()})
            states = {k:bounds[state_to_frame[k]] for k in states}
    # A split can retain the parent's ID while replacing its render_spec.
    full = spec is not None and (spec == raw_spec or
        (clip.get('_replan_direction') == digest(effective) and raw_spec == effective))
    if not full and (spec is not None or not endpoints or not isinstance(raw_spec, dict)):
        if '_replan_direction' in clip or '_replan_endpoints' in clip:
            return {k:v for k,v in clip.items() if k not in ('_replan_direction','_replan_endpoints')}
        return clip
    view = deepcopy(clip)
    if isinstance(view.get('completion_split'), dict):
        view['completion_split'].pop('original_source', None)
    view['render_spec'] = effective
    view.pop('_replan_direction', None)
    view.pop('_replan_endpoints', None)
    if full:
        view['_replan_direction'] = digest(effective)
        view['action'] = effective['action']
        view['action_beats'] = [{'action':effective['action'], 'visible_result':effective['closing']}]
        view['entry_state'] = states.get('entry_state', effective['opening'])
        view['exit_state'] = states.get('exit_state', effective['closing'])
        view['shot_design'] = {k:effective[k] for k in ('opening', 'camera', 'closing')}
        if 'microshots' in view:
            view['microshots'] = [{'action':effective['action'], 'camera':effective['camera']}]
        if 'parent_event_action' in view:
            view['parent_event_action'] = effective['action']
    else:
        view['_replan_endpoints'] = endpoints
        for state, field in state_to_frame.items():
            if field in endpoints:
                view[state] = endpoints[field]
        if isinstance(view.get('shot_design'), dict):
            view['shot_design'].update(endpoints)
            if 'opening' in endpoints:
                view['shot_design']['carry_state'] = endpoints['opening']
        if 'closing' in endpoints and view.get('action_beats'):
            view['action_beats'][-1]['visible_result'] = endpoints['closing']
    return view


def generation_plan(agent, clip, take, legacy_name, candidates):
    """Reuse an owned old plan only for its unchanged input; new input gets a receipt."""
    from . import content
    clip = execution_clip(agent, clip)
    old = read(agent.s.run / 'artifacts' / (legacy_name + '.json'))
    if old is not None and not clip.get('completion_split'):
        return old
    reviewed = [c for c in candidates if c.get('review') and c.get('receipt')]
    if reviewed:
        from .storage import candidate_quality
        useful = any(f.get('severity', f.get('finding', {}).get('severity')) in ('major', 'critical')
            for f in candidate_quality(reviewed[-1]).get('unresolved_findings', []))
        if candidate_quality(reviewed[-1])['ok'] or not useful:
            return {'decision': 'select_best', 'stop_reason': 'accepted' if candidate_quality(reviewed[-1])['ok'] else 'no_useful_local_change',
                'reason': 'Retain real media; report uncertainty does not require a new GPU candidate.'}
    operation = f'completion/local/{clip["id"]}/{take}'
    if clip.get('completion_split'):
        operation += '/plan_' + clip['completion_split']['revision_receipt_id']
    prompt = '''Plan ONE continuous shot from this exact source event. Return only
{reason,render_spec:{opening,action,camera,closing},reference_ids:[],issue_refs:[]}.
reference_ids is optional and changes only existing scene/prop image references; omit
it to retain references. Canonical character images/voices remain fixed. All directions are English;
no Subject/Picture/Audio labels, no quoted speech, no caption, no asset changes.
Host owns role/reference labels, exact dialogue, duration and speaking windows.
Address confirmed major defects from retained observations; do not invent new events,
repeat the selected prefix or move this dialogue elsewhere. For a split event, use
only the assigned speech units and allocated opening/closing; no repeated parent action.'''
    from . import dialogue_integrity
    if dialogue_integrity.enabled(agent.config):
        prompt += dialogue_integrity.PLANNER_GUIDE
    if clip.get('_replan_direction'):
        prompt += ('\nThe approved Replan shot direction owns the physical action. Keep it unchanged. '
                   'To improve visibility, optionally set framing_emphasis to action or result.')
    experience = {}
    if completion_memory.enabled(agent.config):
        prompt += ASSET_GUIDE + "\nLocal optional updates: at most one, target this shot only."
        experience = {'retrieved_experience':completion_memory.recall(agent,'local',
            {'clip':clip,'problems':[c.get('review',{}).get('findings',[]) for c in reviewed]},operation+'/memory'),
            'actual_asset_catalog':agent.catalog(), 'editing_policy':shot_policy.context(agent.config)}
        prompt = prompt.replace('no asset changes.', 'optional asset_updates follow the schema below.')
    duration_parent = reviewed[-1] if task_parameters.longform(agent.config) and reviewed else None
    if duration_parent and duration_parent['take'] < take:
        prompt += ('\nOptional tail extension: if the preceding take audibly truncated a line or visibly '
                   'cut off an action, return tail_extension_finding with one exact major or critical '
                   'finding ID from that take. The host may add up to one second after the existing '
                   'speech windows. For unrelated issues omit the field.')
    plan = agent.artifact(operation, lambda: agent.advisory(operation, prompt,
        {'current_event': clip, 'previous_selected': content.prefix_facts(agent.s.data['selected'][-1:]),
         'actual_candidates': [{'take': c['take'], 'review': c.get('review'), 'quality': c.get('quality'),
             'previous_applied_intervention':c.get('intervention'),
             'visual_direction_audit':c.get('plan',{}).get('visual_direction_audit')} for c in reviewed],
         'speech_budget_is_planning_only': content.speech_budget(clip, agent.catalog()),
         'optional_reference_ids_allowed': optional_reference_ids(agent), **experience}, max_tokens=3000))
    plan = plan if isinstance(plan,dict) else {}
    direction = (shot_policy.execution_direction(clip, plan)
                 if clip.get('_replan_direction') or clip.get('_replan_endpoints') else plan.get('render_spec'))
    if not isinstance(direction, dict) or not all(isinstance(direction.get(k), str) and direction[k].strip()
            and not re.search(r'[\u3400-\u9fff]|<\s*(?:Subject|Picture|Audio|Video|d)\b|\bS\d+\b', direction[k])
            for k in ('opening', 'action', 'camera', 'closing')):
        direction = clip.get('render_spec')
        agent.s.warning(operation, 'Known malformed local directions; retained source render_spec used. No media PASS inferred.')
    return {'decision': 'generate', 'reason': plan.get('reason', 'Host-compiled retained source event'),
        'render_spec': deepcopy(direction), 'seconds': clip['seconds'], 'asset_ids': reference_ids(agent, clip),
        'asset_updates':plan.get('asset_updates',[]) if completion_memory.enabled(agent.config) else [],
        **({'tail_extension_proposal': {'finding_id': plan['tail_extension_finding'], 'parent_take': duration_parent['take']}}
           if duration_parent and duration_parent['take'] < take and isinstance(plan.get('tail_extension_finding'),str) else {}),
        **{k:deepcopy(plan[k]) for k in ('reference_ids','issue_refs','framing_emphasis') if k in plan}}


def reference_ids(agent, clip, include_missing=False):
    """Canonical cast anchors plus used scene/sound references; unused voices are not dependencies."""
    character_stages.assert_clip_bindings(clip, agent.config)
    catalog = {r['id']: r for r in agent.catalog()}
    declared = agent.s.data.get('assets', {})
    cast = agent.config.get('generation_contract', {}).get('cast', {})
    ids = []
    stages = clip.get('character_stage_bindings', {}) if character_stages.enabled(agent.config) else {}
    ids.extend(character_stages.image_ids_for_clip(clip, agent.config))
    for line in clip.get('dialogue', []):
        spec = cast.get(line.get('speaker'), {})
        alias = clip.get('voice_reference_ids', {}).get(line.get('speaker')) if completion_memory.enabled(agent.config) else None
        if alias:
            from . import voice_lineage
            approved = voice_lineage.annotate([catalog[alias]] if alias in catalog else [], list(catalog.values()), agent.s.data, agent.config)
            alias = alias if approved and approved[0].get('canonical_voice_authority') else None
        if not character_stages.enabled(agent.config) or line.get('speaker') not in clip.get('offscreen_voice_only_roles', []):
            ids.append(stages.get(line.get('speaker'), {}).get('image_asset_id', spec.get('image_asset_id')))
        ids.append(alias or spec.get('voice_asset_id'))
    for ident in clip.get('asset_ids', []):
        if character_stages.enabled(agent.config):
            family_role = next((role for role, family in agent.s.data.get('plan', {}).get('character_reference_stages', {}).items()
                if ident in {stage['image_asset_id'] for stage in family['stages']}), None)
            if family_role in stages:
                ident = stages[family_role]['image_asset_id']
        row = catalog.get(ident) or declared.get(ident, {}).get('spec', {})
        if row.get('kind') != 'audio' and (include_missing or ident in catalog):
            ids.append(ident)
    if completion_memory.enabled(agent.config):
        ids.extend(clip.get('sound_asset_ids', []))
    return list(dict.fromkeys(i for i in ids if isinstance(i, str) and i))


def optional_reference_ids(agent):
    cast = agent.config.get('generation_contract', {}).get('cast', {})
    anchors = {v.get(k) for v in cast.values() for k in ('image_asset_id', 'voice_asset_id')}
    anchors.update(stage['image_asset_id'] for family in agent.s.data.get('plan', {}).get('character_reference_stages', {}).values() for stage in family['stages'])
    return [a['id'] for a in agent.catalog() if a.get('kind') == 'image' and a['id'] not in anchors]


def reference_patch_error(agent, requested):
    if not isinstance(requested,list):return 'reference_ids must be a list'
    catalog={r['id']:r for r in agent.catalog()}
    anchors={v.get(k) for v in agent.config.get('generation_contract',{}).get('cast',{}).values()
             for k in ('image_asset_id','voice_asset_id')}
    anchors.update(stage['image_asset_id'] for family in agent.s.data.get('plan', {}).get('character_reference_stages', {}).values() for stage in family['stages'])
    if any(not isinstance(i,str) or i not in catalog
           or catalog[i].get('kind')!='image' and i not in anchors for i in requested):
        return 'reference_ids contains unavailable or unsupported optional media'
    return None


def planned_reference_ids(agent, clip, plan):
    """Allow only existing optional images to change; canonical identities and voices remain host-owned."""
    base = reference_ids(agent, clip)
    requested = plan.get('reference_ids')
    if reference_patch_error(agent,requested):
        return base
    catalog = {r['id']:r for r in agent.catalog()}
    cast = agent.config.get('generation_contract', {}).get('cast', {})
    anchors = {v.get(k) for v in cast.values() for k in ('image_asset_id','voice_asset_id')}
    anchors.update(stage['image_asset_id'] for family in agent.s.data.get('plan', {}).get('character_reference_stages', {}).values() for stage in family['stages'])
    fixed = [i for i in base if i in anchors or catalog.get(i, {}).get('kind') != 'image']
    optional = [i for i in requested if isinstance(i,str) and i in catalog
                and catalog[i].get('kind')=='image' and i not in anchors]
    return list(dict.fromkeys(fixed+optional))


def apply_tail_extension(agent, clip, result):
    """Use one reviewed timing defect to budget extra tail room on the next take."""
    from . import duration_budget, speech_contract
    from .closed_loop import intervals
    proposal = result.get('tail_extension_proposal')
    if not task_parameters.longform(agent.config) or not isinstance(proposal, dict):
        return
    parent = next((row for row in agent.s.data['candidates'].get(clip['id'], [])
                   if row.get('take') == proposal.get('parent_take')), None)
    result['tail_extension_decision'] = 'original_duration'
    if not parent:
        return
    source, review, receipt, parent_plan = (parent.get(k) for k in ('source_clip', 'review', 'receipt', 'plan'))
    if not all(isinstance(x, dict) for x in (source, review, receipt, parent_plan)):
        return
    if (not receipt.get('sha256') or review.get('media_sha256') != receipt['sha256']
            or not receipt.get('audio_sha256') or review.get('audio_sha256') != receipt['audio_sha256']
            or any(source.get(k) != clip.get(k) for k in ('id', 'scene', 'action', 'dialogue'))
            or parent_plan.get('dialogue_timeline') != result['dialogue_timeline']):
        return
    # A prior revision may have changed the request, and preserving the end of
    # speech can leave more native frames than that request originally asked for.
    submitted = source.get('seconds')
    if type(submitted) not in (int, float) or not 4 <= submitted <= 10:
        return
    try:
        native_frames = duration_budget.receipt_frames(receipt)
    except ValueError:
        return
    if not 96 <= native_frames <= 240:
        return
    native_seconds = native_frames / 24
    eligible = any(isinstance(f, dict) and f.get('id') == proposal.get('finding_id')
                   and f.get('kind') == 'observed_content_defect' and f.get('state') == 'open'
                   and f.get('severity') in ('major', 'critical') and f.get('observation')
                   and f.get('category') in ('sentence_ending', 'speech_completeness', 'action_completion')
                   and any(0 <= start <= end and native_seconds - 1 <= end <= native_seconds + 1/24
                           for start, end in intervals(f.get('time_s')))
                   for f in review.get('findings') or [])
    if not eligible:
        return
    parent_frames = max(round(clip['seconds'] * 24), round(submitted * 24), native_frames)
    result['seconds'] = parent_frames / 24
    if parent_frames == 240:
        result['tail_extension_decision'] = 'at_h3_limit'
        return
    extended = min(240, parent_frames + 24) / 24
    # Native speech-tail preservation may legitimately retain up to ten seconds.
    # Reserve that headroom before this optional edit; regular admission still
    # checks the exact dispatched request and the selected native receipt.
    reserve = 10 if speech_contract.enabled(agent.config) else extended
    future = duration_budget.local_reservations(agent, clip, reserve)
    if duration_budget.issues(agent.config, future, agent.s.data['selected']):
        result['tail_extension_decision'] = 'insufficient_duration_budget'
        return
    result.update(seconds=extended, tail_extension_decision='applied')


def compile_plan(agent, clip, plan):
    """Timing is host planning, not an assertion about generated speech."""
    from . import content
    from .binding_contract import english_direction
    def valid(spec, check_literal=True):
        if not isinstance(spec,dict):return False
        try:
            fields=[english_direction(spec.get(k)) for k in ('opening','action','camera','closing')]
            return all(fields) and (not check_literal or not any(content.normal(d['text']) in content.normal(field)
                for d in content.dialogue(clip) for field in fields))
        except ValueError:return False
    clip = execution_clip(agent, clip)
    character_stages.freeze_plan(agent.s, agent.s.data.get('plan', {}), agent.config)
    result = deepcopy(plan)
    if 'reference_ids' in result and reference_patch_error(agent,result['reference_ids']):
        result.setdefault('rejected_fields',[]).append('reference_ids')
    from . import dialogue_integrity
    strict = dialogue_integrity.enabled(agent.config)
    def sanitized(spec):
        # Syntax first; strip verbal descriptions BEFORE checking literal overlap.
        # A source paraphrase ("two hundred pounds") is the very content to remove.
        if strict and valid(spec, check_literal=False):
            return dialogue_integrity.visual_only(spec)
        return spec, None
    if clip.get('_replan_direction') or clip.get('_replan_endpoints'):
        result['render_spec'] = shot_policy.execution_direction(clip, result)
        # Legacy local artifacts can contain free-text directions outside render_spec.
        result.pop('shot_design', None)
        result.pop('caption', None)
    spec, audit = sanitized(result.get('render_spec'))
    if not valid(spec):
        spec, audit = sanitized(clip.get('render_spec'))
        if not valid(spec):
            raise Pending('Source needs executable English shot directions; no caption-label correction loop',status='needs_source_plan')
        agent.s.warning('completion/compile/'+clip['id'],'Invalid optional local directions discarded; retained source direction used, original model reply kept.')
    if audit and audit.get('fallback_fields'):
        source_spec, source_audit = sanitized(clip.get('render_spec'))
        if isinstance(source_spec,dict):
            for field in list(audit['fallback_fields']):
                if not source_audit or field not in source_audit.get('fallback_fields',[]):
                    spec[field]=source_spec[field]
                    audit.setdefault('recovered_source_fields',[]).append(field)
            audit['unrecovered_fields']=[f for f in audit['fallback_fields'] if f not in audit.get('recovered_source_fields',[])]
        if audit.get('unrecovered_fields',audit['fallback_fields']):
            agent.s.warning('completion/compile/'+clip['id'], 'Visual constraints lost during prompt hygiene; retained audit for the next existing planning call.')
    result['render_spec'] = deepcopy(spec)
    if audit is not None:
        result['visual_direction_audit'] = audit
    result.update(seconds=clip['seconds'], asset_ids=planned_reference_ids(agent, clip, result),
        asset_updates=result.get('asset_updates',[]) if completion_memory.enabled(agent.config) else [])
    source_lines = content.dialogue(clip)
    estimates = content.speech_budget(clip, agent.catalog())['utterances']
    total = sum(x['estimated_speech_seconds'] for x in estimates)
    room = clip['seconds'] - .8 - max(0, len(source_lines)-1)*.12
    if total > room + 1e-6:
        raise Pending('Unsubmitted shot requires a committed lossless speech split before GPU dispatch', status='needs_duration_plan')
    scale = min(1.0, room / total) if total else 1
    cursor = .2
    if dialogue_integrity.enabled(agent.config):
        requested = (plan.get('dialogue_timeline') or clip.get('dialogue_timeline') or [{}])[0].get('start_s')
        if type(requested) in (int,float) and math.isfinite(requested) and requested >= .2:
            cursor = min(requested, max(.2, clip['seconds']-.6-total-max(0,len(source_lines)-1)*.12))
    result['dialogue_timeline'] = []
    for index,(line, estimate) in enumerate(zip(source_lines, estimates)):
        end = cursor + estimate['estimated_speech_seconds'] * scale
        result['dialogue_timeline'].append({'speaker': line['speaker'], 'text': line['text'],
            'start_s': round(cursor, 6), 'end_s': round(end, 6), 'on_screen_or_offscreen': line.get('on_screen_or_offscreen') or (clip.get('dialogue_timeline',[])[index].get('on_screen_or_offscreen') if index<len(clip.get('dialogue_timeline',[])) else None) or 'on_screen'})
        cursor = end + .12
    apply_tail_extension(agent, clip, result)
    return result


def reconcile_prepared_direction(agent, clip, candidate):
    """Under runner.lock, upgrade only a locally staged, provably undispatched take."""
    from . import binding_contract
    from .assets import reference_labels
    from .media import video_request
    binding = candidate.get('binding_contract') or {}
    from .evidence_policy import scripted
    nonverbal_upgrade = (binding_contract.enabled(agent.config) and not scripted(clip)
                         and binding.get('nonverbal_policy') != binding_contract.SILENT_POLICY_VERSION)
    revised_direction = clip.get('_replan_direction') or clip.get('_replan_endpoints')
    if not (revised_direction or nonverbal_upgrade) or candidate.get('receipt'):
        return
    take = candidate['take']
    folder = agent.s.run/'media/videos'/f'{clip["id"]}_take_{take}'
    if operation_recorded(folder, agent.s.data.get('accounted_operations', [])):
        return  # The original worker owns this input; resume its receipt path.
    old_source = candidate['source_clip']
    source = {**deepcopy(clip), 'seconds': old_source['seconds']} if revised_direction else old_source
    if not nonverbal_upgrade and source == old_source and binding.get('source_event_sha256') == digest(source):
        return
    if not binding_contract.enabled(agent.config):
        raise Pending('Prepared Replan take requires its original binding contract',status='needs_worker_reconciliation')
    # Only these files can exist before the run accounts the take and records submission.
    allowed = {'request.json','reference_bindings.json','cast_binding_contract.json','resource_queue.json'}
    if folder.is_dir() and any(p.name not in allowed for p in folder.iterdir()):
        raise Pending('Unknown worker evidence on a prepared Replan take',status='needs_worker_reconciliation')
    refs, plan = candidate['references'], deepcopy(candidate['plan'])
    plan['render_spec'] = shot_policy.execution_direction(source, plan)
    plan.pop('shot_design', None)
    plan.pop('caption', None)
    caption, binding = binding_contract.compile(source, refs, agent.config, plan)
    seed = 10000 + int(clip['id'].rsplit('_',1)[1])*100 + take
    original_request = video_request(candidate['caption'], old_source['seconds'], refs, seed,
                                     agent.config.get('video_aspect_ratio', '16:9'))
    repaired_request = {**original_request, 'prompt':caption}
    audit_file = agent.s.run/'plans/replan_pre_dispatch'/folder.name/(digest(candidate)+'.json')
    audit = read(audit_file, attempts=1)
    request_file, binding_file = folder/'request.json', folder/'cast_binding_contract.json'
    request = read(request_file, attempts=1)
    stored_binding = read(binding_file, attempts=1)
    ref_labels = read(folder/'reference_bindings.json', attempts=1)
    acceptable_requests = [None,original_request] + ([repaired_request] if audit else [])
    acceptable_bindings = [None,candidate.get('binding_contract')] + ([binding] if audit else [])
    if (request not in acceptable_requests or stored_binding not in acceptable_bindings
            or ref_labels not in (None, reference_labels(refs))):
        raise Pending('Prepared Replan take differs from the original saved inputs',status='needs_worker_reconciliation')
    if audit is None:
        write(audit_file,{'original_candidate':candidate, 'request':request, 'binding':stored_binding,
                          'repaired_caption_sha256':binding['caption_sha256']})
    if request is not None and request != repaired_request:
        write(request_file, repaired_request)
    if stored_binding is not None and stored_binding != binding:
        write(binding_file, binding)
    candidate.update(caption=caption, source_clip=source, plan=plan, binding_contract=binding,
                     replan_pre_dispatch_evidence=str(audit_file))
    if shot_policy.enabled(agent.config):
        candidate['editing_intent'] = shot_policy.intent(source,agent.config)
    candidate['intervention'] = candidate_intervention(agent,clip,candidate,agent.s.data['candidates'][clip['id']])
    agent.s.save()


def stop_plan(plan, candidates):
    from .storage import candidate_quality
    if plan.get('decision') != 'select_best':
        return plan
    result = deepcopy(plan)
    accepted = any(c.get('review') and c.get('receipt') and candidate_quality(c)['ok'] for c in candidates)
    if result.get('stop_reason') == 'accepted' and not accepted:
        result.update(stop_reason='no_useful_local_change',
            completion_stop_audit='Original accepted label contradicted retained quality. Original artifact kept; no PASS manufactured.')
    return result


def choose(candidates):
    from .storage import candidate_quality
    def rank(candidate):
        q = candidate_quality(candidate)
        issues = q.get('unresolved_findings', [])
        severity = lambda f: f.get('severity', f.get('finding', {}).get('severity', 'uncertain'))
        major = sum(severity(f) in ('major', 'critical') for f in issues)
        from . import dialogue_integrity
        core = {f.get('category', f.get('finding',{}).get('category')) for f in issues
                if severity(f) in ('major','critical') and f.get('category', f.get('finding',{}).get('category')) in dialogue_integrity.CORE}
        if not candidate.get('dialogue_integrity_audit') and not candidate.get('dialogue_integrity_version'):
            core = set()
        # Severe observed lexical errors take precedence among fallback takes.
        # An uncertain ASR difference does not enter this set.
        lexical_defect = bool(core.intersection(dialogue_integrity.CHECKS))
        return (bool(q['ok']), not lexical_defect, -len(core), -major, -len(issues), candidate['take'])
    return max(candidates, key=rank)


def learn(agent, cid, candidates, current):
    """Record existing observations without a paid comparison on the progress path."""
    if completion_memory.enabled(agent.config):
        return completion_memory.learn(agent,cid,candidates,current)
    from .storage import candidate_quality
    if any(x.get('clip_id') == cid and x.get('take') == current['take'] for x in agent.s.data['local_lessons']):
        return
    previous = [x for x in candidates if x['take'] < current['take'] and x.get('review')]
    if not previous:
        return
    parent = previous[-1]
    lesson = {'clip_id': cid, 'parent_take': parent['take'], 'take': current['take'],
        'evidence': 'retained_observer_status_comparison_not_independent_media_improvement',
        'relative_outcome': 'unconfirmed', 'absolute_quality_status': candidate_quality(current)['quality_status'],
        'parent_media_sha256': parent['receipt']['sha256'], 'media_sha256': current['receipt']['sha256'],
        'remaining_issues': candidate_quality(current).get('unresolved_findings', []),
        'lesson': current.get('plan', {}).get('reason'), 'causal_claim': 'none', 'paid_comparison': False}
    agent.s.data['local_lessons'].append(lesson)
    agent.s.save()


def candidate_intervention(agent, clip, candidate, candidates):
    """Describe the real generator inputs, not the director's intent or seed alone."""
    def conditions(row):
        refs=row.get('references',[])
        keyframes=bool(refs) and all(r.get('kind')=='image' and r.get('frame_index') is not None for r in refs)
        return {'caption':row.get('caption'), 'seconds':row.get('source_clip',{}).get('seconds'),
            'references':[{'type':'video_audio' if r.get('kind')=='video' and r.get('include_audio') else r.get('kind'),
                           'sha256':r.get('sha256'),'frame_index':r.get('frame_index') if keyframes else None}
                          for r in refs]}
    parents=[c for c in candidates if c.get('receipt') and c.get('take',0)<candidate['take']]
    parent=parents[-1] if parents else None
    before=conditions(parent) if parent else {}
    after=conditions(candidate)
    changed={k:{'before':before.get(k),'after':v} for k,v in after.items() if before.get(k)!=v}
    plan=candidate.get('plan',{})
    from .storage import candidate_quality
    issues=candidate_quality(parent).get('unresolved_findings',[]) if parent and parent.get('review') else []
    kind='initial_generation' if parent is None else 'local_revision' if changed else 'resample'
    ident=key({'run':str(agent.s.run),'clip_id':clip['id'],'take':candidate['take'],'conditions':after})
    return {'version':'executed_intervention_v1','intervention_id':ident,'kind':kind,
        'status':'prepared_not_submitted','issue_refs':[f.get('id') for f in issues if f.get('id')],
        'parent_media_sha256':parent['receipt'].get('sha256') if parent else None,
        'proposed_patch':{k:deepcopy(plan[k]) for k in ('render_spec','reference_ids','issue_refs','tail_extension_proposal') if k in plan},
        'applied_patch':changed if parent else {},
        'rejected_fields':deepcopy(plan.get('rejected_fields',[])),
        'request_conditions_sha256':digest(after),'causal_claim':'none',
        'interpretation':'A resample uses another existing candidate slot; it is not a directed repair.' if kind=='resample' else
                         'Input changes are executable; their content effect remains unobserved until review.'}


def realized_prefix(agent):
    """A cheap full-prefix evidence ledger; targets never become observed facts."""
    rows=[]
    for selected in agent.s.data['selected']:
        clip=selected['clip'];review=selected.get('review',{});receipt=selected.get('receipt',{})
        base=clip['id']+'/'+str(selected.get('take',1))
        evidence=[]
        for ident,item in review.get('evidence_catalog',{}).items():
            if isinstance(item,dict) and item.get('observation'):
                evidence.append({'id':base+'/'+ident,'observation':deepcopy(item['observation'])})
        if not evidence:
            for i,item in enumerate(review.get('observed_events',[])):
                evidence.append({'id':base+'/event/'+str(i+1),'observation':deepcopy(item)})
        narrative=review.get('narrative_observation',{})
        rows.append({'clip_id':clip['id'],'event_id':clip.get('event_id'),
            'target_only':{'action':clip.get('action'),'entry_state':clip.get('entry_state'),
                           'exit_state':clip.get('exit_state')},
            'media_sha256':receipt.get('sha256'),'audio_sha256':receipt.get('audio_sha256'),
            'frames':receipt.get('frames'),'evidence':evidence[:8],
            'narrative_observation':deepcopy(narrative) if isinstance(narrative,dict) else {},
            'remaining_issues':[{k:f.get(k) for k in ('id','category','severity','observation')}
                for f in selected.get('quality',{}).get('unresolved_findings',[])[:4]],
            'coverage':'observer_report_available' if evidence else 'unknown_not_inferred_from_target'})
    ledger={'version':'realized_prefix_evidence_v1','selected_count':len(rows),'shots':rows,
        'prefix_sha256':digest([{'id':r['clip_id'],'video':r['media_sha256'],'audio':r['audio_sha256']} for r in rows]),
        'actual_prefix_frames':sum(r['frames'] or 0 for r in rows),
        'instruction':'Use observations for realization, target_only for intended goals. Missing/uncertain evidence is not proof of coverage or failure.'}
    agent.s.data['realized_prefix']=ledger
    return ledger


def _global_patch(agent, future, proposal):
    """Stage bounded future edits, then admit once before touching production state."""
    from . import content, duration_budget, completion_plan, dialogue_integrity
    staged=deepcopy(future);applied=[];receipts=[];seen=set()
    for row in proposal.get('suggestions',[]) if isinstance(proposal.get('suggestions'),list) else []:
        if not isinstance(row,dict):continue
        cid=row.get('clip_id');clip=next((c for c in staged if c['id']==cid),None)
        rejected=[];before=deepcopy(clip);changes={}
        if not clip or cid in seen:
            receipts.append({'clip_id':cid,'status':'rejected','reason':'not_uncommitted_future_or_duplicate'});continue
        seen.add(cid)
        forbidden=set(row)-{'clip_id','reason','evidence_refs','render_spec','entry_state','exit_state','reference_ids','seconds'}
        if forbidden:rejected.extend(sorted(forbidden))
        if not rejected and 'render_spec' in row:
            spec=row['render_spec']
            if (not isinstance(spec,dict) or not all(isinstance(spec.get(k),str) and spec[k].strip()
                and not re.search(r'[\u3400-\u9fff]|<\s*(?:Subject|Picture|Audio|Video|d)\b|\bS\d+\b',spec[k])
                for k in ('opening','action','camera','closing'))
                or any(content.normal(d['text']) in content.normal(f) for d in content.dialogue(clip)
                       for f in spec.values() if isinstance(f,str))):
                rejected.append('render_spec')
            else:
                if dialogue_integrity.enabled(agent.config):
                    spec,audit=dialogue_integrity.visual_only(spec)
                    if audit.get('fallback_fields'):rejected.append('render_spec_constraint_loss')
                if not rejected and spec!=clip.get('render_spec'):changes['render_spec']=deepcopy(spec)
        for field in ('entry_state','exit_state'):
            if field in row:
                if (not isinstance(row[field],str) or not row[field].strip()
                    or (shot_policy.contract_enabled(agent.config) and re.search(r'[\u3400-\u9fff]|<\s*(?:Subject|Picture|Audio|Video|d)\b|\bS\d+\b',row[field]))):rejected.append(field)
                elif row[field]!=clip.get(field):changes[field]=row[field]
        if 'reference_ids' in row:
            if reference_patch_error(agent,row['reference_ids']):rejected.append('reference_ids')
            else:
                refs=planned_reference_ids(agent,clip,row)
                if refs!=reference_ids(agent,clip):changes['asset_ids']=refs
        if 'seconds' in row:
            value=row['seconds']
            if (content.dialogue(clip) or type(value) not in (int,float) or not math.isfinite(value)
                or not max(4,clip.get('minimum_seconds',4))<=value<=clip['seconds']
                or abs(value*24-round(value*24))>1e-6):
                rejected.append('seconds')
            elif value!=clip['seconds']:changes['seconds']=value
        if rejected:
            receipts.append({'clip_id':cid,'status':'rejected','fields':rejected,'reason':row.get('reason')});continue
        clip.update(changes)
        record={'clip_id':cid,'status':'applied' if changes else 'noop','reason':row.get('reason'),
                'changes':deepcopy(changes),'before_sha256':digest(before),'after_sha256':digest(clip),
                'evidence_refs':row.get('evidence_refs',[]),'outcome':'unvalidated_future_effect'}
        receipts.append(record)
        if changes:applied.append({'clip_id':cid,**deepcopy(changes),'reason':row.get('reason')})
    # Include all locked unselected requests in the same real-frame account.
    selected={s['clip']['id'] for s in agent.s.data['selected']}
    replacements={c['id']:c for c in staged}
    all_future=[replacements.get(c['id'],c) for c in agent.s.data['plan']['clips'] if c['id'] not in selected]
    bad=duration_budget.issues(agent.config,completion_plan.budget_clips(agent,all_future),agent.s.data['selected'])
    if bad:
        return [],[{'status':'rejected','reason':'optional_global_frame_budget','details':bad}],future
    return applied,receipts,staged


def _uncertain_handoff(previous, next_clip):
    """Match an explicit carry state and the review's typed uncertainty."""
    exit_state = previous['clip'].get('exit_state')
    entry_state = next_clip.get('entry_state')
    if not (isinstance(exit_state, str) and isinstance(entry_state, str)
            and exit_state.strip() and exit_state.strip() == entry_state.strip()):
        return None
    review = previous.get('review')
    if not isinstance(review, dict):
        return None
    observation = review.get('narrative_observation')
    if not isinstance(observation, dict) or observation.get('status') != 'unknown':
        return None
    decision = review.get('feedback_decision')
    narrative = decision.get('narrative_observation') if isinstance(decision, dict) else None
    # Normalization also yields unknown for absent or unsupported observations.
    # Escalate only when the director explicitly names the carried expectation.
    if not isinstance(narrative, dict) or narrative.get('status') != 'unknown':
        return None
    target = narrative.get('remaining_target')
    if not isinstance(target, str) or target.strip() != exit_state.strip():
        return None
    return {'reason':'carry_state_observation_unknown','previous_clip_id':previous['clip']['id'],
            'next_clip_id':next_clip['id'],'carry_state_target':entry_state}


def global_checkpoint(agent):
    """One existing call reviews the full realized prefix and changes only its unlocked suffix."""
    from . import completion_plan, content
    selected = agent.s.data['selected']
    count = len(selected)
    if not count or count in agent.s.data['checkpoints']:return
    selected_ids={s['clip']['id'] for s in selected}
    pending=[c for c in agent.s.data['plan']['clips'] if c['id'] not in selected_ids]
    saved=agent.s.data.get('replan_receipts',{}).get(str(count))
    trigger={}
    if saved is None and count % 3:
        if not pending:return
        trigger=_uncertain_handoff(selected[-1],pending[0])
        if not trigger or not completion_plan._no_render_claim(agent,pending[0]['id']):return
    ledger=realized_prefix(agent)
    future=[c for c in pending if completion_plan._no_render_claim(agent,c['id'])]
    name=f'completion/global/{count}'
    if saved is not None:
        # The patch can itself remove the original trigger. Resume from the
        # retained proposal and never acquire a replacement request.
        if saved.get('prefix_sha256')!=ledger['prefix_sha256']:
            raise Pending('Retained Replan receipt has a different selected prefix',
                          status='needs_replan_reconciliation')
        try:
            proposal=read(agent.s.run/'artifacts'/(name+'.json'), required=True)
        except (OSError,ValueError) as exc:
            raise Pending('Retained Replan proposal is unavailable',
                          status='needs_replan_reconciliation') from exc
        proposal=proposal if isinstance(proposal,dict) else {}
        if saved.get('proposal_sha256')!=digest(proposal):
            raise Pending('Retained Replan proposal does not match its receipt',
                          status='needs_replan_reconciliation')
    if completion_memory.enabled(agent.config):completion_memory.followup(agent,count)
    if future or saved is not None:
        if saved is None:
            future_view = [execution_clip(agent, c) for c in future]
            experience={}
            if completion_memory.enabled(agent.config):
                experience={'retrieved_experience':completion_memory.recall(agent,'global',
                    {'future':future_view,'prefix':ledger},name+'/memory'),
                    'actual_asset_catalog':agent.catalog(),'editing_policy':shot_policy.context(agent.config)}
            prompt='''Assess narrative causality, coverage and pacing against the FULL realized_prefix.
Target-only goals are not proof that an event occurred; cite actual shot/evidence ids,
retain uncertainty and do not replay completed actions. Address earlier missing goals
through clearer UNCOMMITTED future acting/entry/exit conditions without adding events
or changing the script. Return {analysis,suggestions:[{clip_id,reason,evidence_refs:[],
render_spec:{opening,action,camera,closing},entry_state,exit_state,reference_ids:[]}],lessons:[]}.
All edit fields are optional. A new render_spec is the complete replacement for the shot's
physical staging. Include all still-required physical beats and preserve all named role/prop
checkpoints, character identities, literal dialogue and native sound policy. English
physical shot directions only, one continuous shot.
reference_ids may change existing optional scene/prop images; canonical identities stay fixed.
seconds may only reduce a pure non-dialogue hold to its declared minimum (>=4 seconds),
on the 24fps grid. No dialogue-shot timing changes. Preserve dialogue, speakers, order,
events, ending, ids, selected media and all paid inputs. Invalid/noop suggestions retain
the current plan and receive an explicit receipt. No asset updates unless allowed below.'''
            if completion_memory.enabled(agent.config):prompt+=ASSET_GUIDE
            from . import dialogue_integrity
            if dialogue_integrity.enabled(agent.config):prompt+=dialogue_integrity.PLANNER_GUIDE
            proposal=agent.artifact(name,lambda:agent.advisory(name,prompt,
                {'realized_prefix':ledger,'selected_observations':ledger['shots'],
                 'future':future_view,'script_locked':True,
                 **({'checkpoint_trigger':trigger} if trigger else {}),
                 'optional_reference_ids_allowed':optional_reference_ids(agent),
                 'prior_patch_receipts':list(agent.s.data.get('replan_receipts', {}).values())[-1:],**experience},max_tokens=5000))
            proposal=proposal if isinstance(proposal,dict) else {}
            applied,receipts,staged=_global_patch(agent,future,proposal)
            replacements={c['id']:c for c in staged}
            agent.s.data['plan']['clips']=[replacements.get(c['id'],c) for c in agent.s.data['plan']['clips']]
            saved={'version':'bounded_future_patch_v1','prefix_sha256':ledger['prefix_sha256'],
                'applied':applied,'receipts':receipts,'proposal_sha256':digest(proposal),
                'outcome':'unvalidated_until_subsequent_observation'}
            if trigger:saved['checkpoint_trigger']=trigger
            agent.s.data.setdefault('replan_receipts',{})[str(count)]=saved
            agent.s.save()
        if completion_memory.enabled(agent.config):
            assets=agent.asset_updates(proposal.get('asset_updates',[]),name+'/assets')
            completion_memory.global_lesson(agent,count,proposal,saved['applied'],assets)
        else:
            agent.s.data['global_lessons'].append({'selected_count':count,'proposal':proposal,
                'evidence':'optional_future_plan_not_observed_media_improvement','asset_execution':False,
                'applied_directions':saved['applied']})
    agent.s.data['checkpoints'].append(count)
    agent.s.save()


def record_observation(agent, name, value):
    try:
        agent.memory.record('local', name, 'observed_candidate', value, nonblocking=completion_memory.enabled(agent.config))
    except OSError as exc:
        # The actual candidate, receipt and accounting are already fsynced in state.
        agent.s.warning(name, 'Optional experience export failed; authoritative candidate retained: ' + str(exc))


ASSET_GUIDE = """
Return asset_updates:[] or at most TWO useful optional intents with schema
{id,action:keep|inspect|new|update,kind:image|audio,role,reason,prompt,target_clip_ids:[]}.
Retrieve lessons as hypotheses, never proof of success. Use current registry ids.
Do not change script, dialogue, roles, canonical identity, duration, events or ending.
Voice improvements are direct clones of the SAME canonical voice, never a new actor.
Scene/prop image updates are allowed. keep plus omit_reference:true removes an optional reference from targets, never deletes its asset or a canonical identity anchor. Audio music uses the existing music tool only
if task policy permits: quiet short isolated instrumental cues, at least two
unscored shots between cues and at most four scored shots per film. Ambience is native H3 sound direction or an explicitly
approved pure audio import; there is no general ambience synthesis tool. No arbitrary
file paths. Failed optional assets retain old references and never demand a retry.
Only uncommitted future clips may change; preserve all selected media and observations.
"""
