"""Bounded optional asset execution through the existing registry and worker receipts."""
from copy import deepcopy
import re
from . import asset_actions, binding_contract, completion_memory, completion_plan, shot_policy, asset_permissions
from .storage import Pending, key, sha, write

LIMITS = {'global': 2, 'local': 1, 'film': 6}


def targets(agent, raw, operation):
    selected = {s['clip']['id'] for s in agent.s.data['selected']}
    wanted = raw.get('target_clip_ids', [])
    if not isinstance(wanted, list):
        return []
    local = getattr(agent, '_completion_clip', None)
    result = []
    for c in agent.s.data.get('plan', {}).get('clips', []):
        if c['id'] not in wanted or c['id'] in selected:
            continue
        safe = completion_plan._no_render_claim(agent, c['id'])
        if local and c['id'] == local['id'] and ('/local/' in operation or operation.startswith('local_')):
            rows = agent.s.data.get('candidates', {}).get(c['id'], [])
            safe = all(x.get('technical_failure') or (x.get('receipt') and x.get('review')) for x in rows)
            terminal={c['id']+'_take_'+str(x['take']) for x in rows if x.get('technical_failure') or (x.get('receipt') and x.get('review'))}
            safe = safe and not any(p.parent.name not in terminal for p in (agent.s.run/'media/videos').glob(c['id']+'_take_*/submitted.json'))
        if safe:
            result.append(c)
    return result


def _character_image_ids(agent):
    cast = agent.config.get('generation_contract', {}).get('cast', {})
    families = agent.s.data.get('plan', {}).get('character_reference_stages', {})
    return {role.get('image_asset_id') for role in cast.values() if role.get('image_asset_id')} | {
        stage['image_asset_id'] for family in families.values() for stage in family['stages']}


def _prepare(agent, raw):
    if (not isinstance(raw, dict) or raw.get('action') not in ('new', 'update', 'keep', 'inspect')
            or not isinstance(raw.get('id'), str) or not re.fullmatch(r'[a-zA-Z0-9_\-]{1,96}', raw['id'])
            or not isinstance(raw.get('reason'), str) or not raw['reason'].strip()):
        return None, 'Malformed optional intent retained without clarification call'
    spec = {k: deepcopy(raw[k]) for k in ('id','action','reason','kind','role','speaker','prompt','description','generator','reference_asset_id','path','seconds','text','language') if k in raw}
    rows = agent.s.data['assets']; old = rows.get(spec['id'], {}).get('spec', {})
    if spec['action'] in ('keep', 'inspect'):
        return spec, None
    protected = _character_image_ids(agent)
    if spec['id'] in protected:
        return None, 'Optional asset updates cannot change a canonical or stage character image'
    if spec['action'] == 'update' and not old:
        return None, 'Unknown update target'
    if spec['action'] == 'new' and spec['id'] in rows:
        return None, 'Existing id cannot be redeclared as new'
    if old and any(k in spec and old.get(k) is not None and spec[k]!=old[k] for k in ('kind','role','speaker')):
        return None, 'Asset update cannot change its existing kind, role or speaker'
    spec = {**deepcopy(old), **spec}
    if not isinstance(spec.get('prompt') or spec.get('description') or spec.get('path'), str):
        return None, 'Complete asset target required'
    cast = agent.config.get('generation_contract', {}).get('cast', {})
    canonical = {v['voice_asset_id']: role for role,v in cast.items()}
    if spec['id'] in canonical or spec.get('role') == 'voice':
        role = canonical.get(spec['id']) or spec.get('speaker')
        parent = cast.get(role, {}).get('voice_asset_id')
        if not parent or (spec.get('reference_asset_id') and spec['reference_asset_id'] != parent):
            return None, 'Voice variants must clone the unchanged canonical voice of the same role'
        spec.update(kind='audio', role='voice', speaker=role, generator='voice_clone', reference_asset_id=parent)
        spec.pop('path', None)
        spec['id'] = parent + '_variant_' + key(spec)[:12]
        spec['action'] = 'new'
    elif spec.get('kind', 'image') == 'audio':
        if spec.get('role') not in ('music', 'ambience'):
            return None, 'Non-voice audio needs an explicit music or ambience role'
        if spec.get('role')=='music' and agent.config.get('editing_policy',{}).get('background_music')!='explicit_model_planned_cues_only':
            return None, 'Music needs explicit task permission'
        if shot_policy.no_score(agent.config) and shot_policy.is_music(spec):
            return None, 'Explicit no-score policy'
        if spec['role'] == 'music':
            spec.update(generator='music')
        else:
            # ACE-Step is music, not an ambience/SFX synthesizer. Only approved pure imports.
            approved = agent.config.get('completion_policy', {}).get('approved_ambience_imports', [])
            source = next((a for a in approved if a.get('path') == spec.get('path')), None)
            try:intact=source and sha(source['path'])==source.get('sha256')
            except OSError:intact=False
            if not intact:
                return None, 'No approved ambience source; use native scene ambience in render_spec'
            spec.update(generator='import')
    elif spec.get('kind', 'image') != 'image':
        return None, 'Only existing image, canonical voice clone, music and approved ambience tools supported'
    else:
        parent = spec.get('reference_asset_id')
        if spec.get('role') == 'character' or isinstance(parent, str) and parent in protected:
            return None, 'Optional asset updates cannot change a canonical or stage character image'
        spec.update(kind='image', generator='qwen_image')
    if asset_permissions.generation_blocked(agent, spec['id'], {'spec': spec}):
        return None, 'Reuse-only voice permission forbids new root/variant/clone generation'
    return spec, None


def sound_ok(asset, row):
    if asset.get('kind') != 'audio' or asset.get('review_status') != 'PASS':
        return False
    v = next((v for v in row.get('versions', []) if v.get('sha256') == asset.get('sha256')), {})
    r = v.get('review', {})
    return (r.get('speech_present') is False and r.get('human_vocalization') is False
        and (row['spec'].get('role') != 'ambience' or r.get('background_music') is False))


def _bind(agent, receipt, clips, spec):
    asset = next((a for a in agent.catalog() if a['id'] == spec['id'] and a.get('review_status') == 'PASS'), None)
    if not asset or sha(asset['path']) != asset['sha256']:
        return []
    row = agent.s.data['assets'][spec['id']]
    audio = asset['kind'] == 'audio'; voice = spec.get('role') == 'voice'
    if audio and not voice and not sound_ok(asset, row):
        return []
    bindings = []
    all_clips = agent.s.data['plan']['clips']
    selected_music = {s['clip']['id'] for s in agent.s.data['selected'] if s['clip'].get('music_asset_id')}
    for clip in clips:
        if voice and spec.get('speaker') not in {d.get('speaker') for d in clip.get('dialogue', [])}:
            continue
        speakers = {d.get('speaker') for d in clip.get('dialogue', [])}
        if audio and not voice and (len(speakers) >= 3 or clip.get('sound_asset_ids')):
            continue
        if spec.get('role') == 'music':
            index = all_clips.index(clip)
            scored = [i for i,c in enumerate(all_clips) if c.get('music_asset_id') or c['id'] in selected_music]
            if len(scored)>=4 or any(abs(index - i) < 3 for i in scored):
                continue
            clip['music_asset_id'] = spec['id']
        if voice:
            clip.setdefault('voice_reference_ids', {})[spec['speaker']] = spec['id']
        elif audio:
            clip.setdefault('sound_asset_ids', []).append(spec['id'])
        if spec['id'] not in clip.setdefault('asset_ids', []):
            clip['asset_ids'].append(spec['id'])
        bindings.append({'clip_id':clip['id'], 'asset_id':spec['id'], 'sha256':asset['sha256'], 'kind':asset['kind'],'status':'planned_future_reference; actual usage is in candidate.references and asset_usage'})
    return bindings


def apply(agent, updates, operation):
    if not isinstance(updates, list):
        return []
    operation = operation or 'completion/assets/'+key(updates)
    state = agent.s.data.setdefault('completion_asset_receipts', {})
    result = []
    local = '/local/' in operation or operation.startswith('local_')
    scope = ('local/'+getattr(agent, '_completion_clip', {}).get('id', operation)) if local else operation
    for index, raw in enumerate(updates):
        ident = key({'operation':operation,'index':index,'intent':raw})
        receipt = state.get(ident)
        if receipt and receipt['status'] != 'executing':
            result.append(deepcopy(receipt)); continue
        receipt = receipt or {'id':ident,'operation':operation,'scope':scope,'original_intent':deepcopy(raw),
            'executed':False,'reference_bindings':[],'status':'received','causal_claim':'none'}
        state[ident] = receipt; agent.s.save(); agent.s.check_stop()
        spec, error = (deepcopy(receipt['proposal']),None) if receipt.get('reserved') else _prepare(agent, raw)
        clips = targets(agent, raw if isinstance(raw,dict) else {}, operation)
        # Recheck reserved intents too: a saved proposal is not a generation permit.
        if spec and spec.get('action') not in ('keep', 'inspect') and asset_permissions.generation_blocked(agent, spec['id'], {'spec': spec}):
            error = 'Reuse-only voice permission forbids reserved root/variant/clone generation'
        if error or not clips:
            receipt.update(status='deferred', reason=error or 'No uncommitted eligible target clips')
        elif spec['action']=='keep' and raw.get('omit_reference') is True:
            if spec['id'] not in _character_image_ids(agent):
                for clip in clips:
                    clip['asset_ids']=[i for i in clip.get('asset_ids',[]) if i!=spec['id']]
                    clip['sound_asset_ids']=[i for i in clip.get('sound_asset_ids',[]) if i!=spec['id']]
                    if clip.get('music_asset_id')==spec['id']:clip.pop('music_asset_id')
                    clip['voice_reference_ids']={r:i for r,i in clip.get('voice_reference_ids',{}).items() if i!=spec['id']}
                receipt.update(status='optional_reference_omitted',asset_id=spec['id'],reason='Existing asset and prior media retained')
            else:receipt.update(status='deferred',reason='Canonical and stage character images cannot be omitted')
        elif spec['action'] in ('keep','inspect'):
            # Inspect reuses real registered observations; no paid inspection/followup loop.
            existing = agent.s.data['assets'].get(spec['id'], {})
            bindings = [] if spec['id'] in _character_image_ids(agent) else _bind(
                agent, receipt, clips, {**existing.get('spec', {}), **spec})
            receipt.update(status='kept_existing_observations', asset_id=spec['id'],
                observation=deepcopy(existing.get('versions', [])[-1:]), reference_bindings=bindings)
        else:
            signature = key(spec)
            prior = next((r for r in state.values() if r['id'] != ident and r.get('signature') == signature), None)
            reserved = [r for r in state.values() if r.get('reserved') and r['id'] != ident]
            if prior:
                receipt.update(status='already_recorded', original_receipt_id=prior['id'],
                    reference_bindings=_bind(agent,receipt,clips,spec) if prior.get('executed') else [],
                    reason='Same target reuses accepted asset only; attempts unchanged')
            elif (index >= LIMITS['local' if local else 'global'] or len(reserved) >= LIMITS['film']
                    or sum(r.get('scope') == scope for r in reserved) >= LIMITS['local' if local else 'global']):
                receipt.update(status='deferred_budget', reason='Optional asset bound reached; render continues with existing references')
            else:
                if not receipt.get('reserved'):
                    try:
                        physical = agent.config.get('task_contract', {}).get('physical_gpu', agent.config.get('audio_provider', {}).get('device',1))
                        binding_contract.admit_gpu(agent.s, physical, agent.config)
                    except Pending as exc:
                        if exc.status != 'awaiting_resource':
                            raise
                        receipt.update(status='deferred_resource', reason=str(exc)); agent.s.save()
                        result.append(deepcopy(receipt)); continue
                    receipt.update(status='executing',reserved=True,signature=signature,asset_id=spec['id'],proposal=deepcopy(spec))
                    agent.s.save()
                if not receipt.get('asset_action_receipt_ids'):
                    declared = asset_actions.apply(agent,[spec],operation+'/'+ident)
                    receipt['asset_action_receipt_ids'] = [r['id'] for r in declared]; agent.s.save()
                try:
                    agent.ensure_assets(only_ids=[spec['id']], optional=True)
                except Pending as exc:
                    if exc.status not in ('needs_asset_observation','needs_voice_observation'):
                        raise
                    # Missing optional observations never buy another asset or block the film.
                    row=agent.s.data['assets'][spec['id']]
                    row.update(dependency_state='optional_observation_unavailable',
                        generation=row.get('pending_version') or row.get('generation',0),
                        pending_prompt=None,pending_version=None,last_generation_failure=str(exc))
                    receipt['observation_gap']=str(exc)
                    agent.registry.save()
                    agent.s.warning(operation,'Optional asset observation unavailable; retained current references: '+str(exc))
                asset_actions.settle(agent)
                actions = [agent.s.data['asset_action_receipts'][i] for i in receipt['asset_action_receipt_ids']]
                accepted = any(r.get('status') == 'published_accepted' for r in actions)
                receipt.update(status='published_accepted' if accepted else 'optional_unavailable', executed=accepted,
                    reference_bindings=_bind(agent, receipt, clips, spec) if accepted else [])
        agent.s.save()
        write(agent.s.run/'completion_assets'/f'{ident}.json', receipt)
        completion_memory.export(agent, 'completion/assets/'+ident, 'asset_execution_result', receipt)
        result.append(deepcopy(receipt))
    return result


def annotate_sounds(agent, references):
    for ref in references:
        row=agent.s.data['assets'].get(ref['id'],{})
        if row.get('spec',{}).get('role') not in ('music','ambience'):
            continue
        actual=next((a for a in agent.catalog() if a['id']==ref['id']),None)
        actions=[r for r in agent.s.data.get('asset_action_receipts',{}).values()
            if r.get('asset_id')==ref['id'] and r.get('status')=='published_accepted'
            and r.get('generated_version',{}).get('sha256')==(actual or {}).get('sha256')]
        if actual and sound_ok(actual,row) and actions and sha(actual['path'])==actual['sha256']:
            ref['sound_reference_authority']={'role':row['spec']['role'],'asset_id':actual['id'],
                'path':actual['path'],'sha256':actual['sha256'],'selected_sha256':ref['sha256'],
                'asset_action_receipt_id':actions[-1]['id'],'speech_present':False,'human_vocalization':False}
    return references
