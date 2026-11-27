"""Replay-safe receipts for optional future asset intents, independent of story edits."""
from copy import deepcopy
from . import prompts, shot_policy
from .closed_loop import rows
from .storage import key, write, sha, Unavailable


def apply(agent, updates, operation, depth=0):
    s = agent.s
    intents = s.data.setdefault('asset_action_receipts', {})
    result = []
    for index, original in enumerate(rows(updates)):
        ident = key({'operation':operation,'index':index,'intent':original})
        if ident in intents and intents[ident].get('status') not in ('received','inspection_followup_pending'):
            result.append(deepcopy(intents[ident])); continue
        receipt = intents.get(ident) or {'id':ident,'operation':operation,'original_intent':deepcopy(original),
            'status':'received','scope':'future_asset_versions_only','selected_prefix_unchanged':True}
        intents[ident]=receipt; s.save()
        raw = original if isinstance(original,dict) else {}
        if raw.get('action') not in ('keep','inspect','update','new') or not raw.get('id'):
            proposal = agent.artifact('asset_actions/'+ident+'/clarification', lambda: agent.advisory(
                'asset_actions/'+ident+'/clarification',prompts.ASSET_ACTIONS,
                {'original_intent':original,'actual_assets':agent.catalog(),
                 'unavailable_assets':agent.registry.unavailable(),'scope':'future assets only'},max_tokens=4500))
            tasks = rows(proposal.get('tasks'))
            if len(tasks)!=1 or not isinstance(tasks[0],dict):
                receipt.update(status='needs_clarification',reason=proposal.get('unresolved_intents') or 'No single executable task returned')
                s.save(); result.append(deepcopy(receipt)); continue
            raw=tasks[0]
        asset_id, action = raw.get('id'), raw.get('action')
        if not isinstance(asset_id,str) or not asset_id or action not in ('keep','inspect','update','new') or not raw.get('reason'):
            receipt.update(status='rejected',reason='Task needs actual id, action and reason',proposal=raw)
        elif action!='new' and asset_id not in s.data['assets']:
            receipt.update(status='rejected',reason='Unknown asset id',proposal=raw)
        elif action=='new' and asset_id in s.data['assets']:
            receipt.update(status='rejected',reason='New asset id already registered; use explicit update',proposal=raw)
        elif action=='keep':
            receipt.update(status='kept',asset_id=asset_id,reason=raw['reason'])
        elif action=='inspect' and depth:
            receipt.update(status='needs_clarification',asset_id=asset_id,reason='Bounded asset followup cannot recursively inspect again')
        elif action=='inspect':
            asset=next((a for a in agent.catalog() if a['id']==asset_id),None)
            if asset is None:
                receipt.update(status='needs_clarification',asset_id=asset_id,reason='No published usable file to inspect')
            else:
                if sha(asset['path']) != asset['sha256']:
                    raise Unavailable('Registered asset bytes changed; reconcile original asset before inspection')
                observation=agent.artifact('asset_actions/'+ident+'/inspection',lambda: agent.advisory(
                    'asset_actions/'+ident+'/inspection',prompts.ASSET_FEEDBACK,
                    {'target':s.data['assets'][asset_id]['spec'],'question':raw['reason']},
                    provider='doubao' if asset['kind']=='image' else 'gemini',
                    media=[{'path':asset['path'],'label':'Actual registered asset to inspect'}],max_tokens=4000))
                receipt.update(status='inspection_followup_pending' if observation.get('decision')=='revise' else 'inspected',asset_id=asset_id,media_sha256=asset['sha256'],
                    observation=observation,next_action_required=observation.get('decision')!='accept',reason=raw['reason'])
                s.save()
                if observation.get('decision')=='revise':
                    followup=agent.artifact('asset_actions/'+ident+'/followup',lambda:agent.advisory(
                        'asset_actions/'+ident+'/followup',prompts.ASSET_ACTIONS,
                        {'actual_inspection':observation,'current_asset':asset,'original_intent':original,
                         'actual_assets':agent.catalog(),'instruction':'Return one update with a complete new generation target, or explicit keep with unresolved issues. No recursive inspect or new asset id.'},max_tokens=4500))
                    tasks=[t for t in rows(followup.get('tasks')) if isinstance(t,dict) and t.get('id')==asset_id and t.get('action') in ('update','keep')]
                    if tasks:
                        receipt['followup_receipts']=apply(agent,tasks[:1],operation+'/inspection/'+ident,depth=1)
                        receipt.update(status='inspected_followup_recorded',next_action_required=True)
                    else:
                        receipt.update(status='needs_clarification',reason='Inspection found a defect but no executable update was returned')
        else:
            spec={k:deepcopy(v) for k,v in raw.items() if k not in ('action','reason')}
            target=spec.get('prompt') or spec.get('description') or spec.get('path')
            if not isinstance(target,str) or not target.strip():
                receipt.update(status='needs_clarification',reason='Update/new requires complete prompt, description or source path',proposal=raw)
            elif shot_policy.no_score(agent.config) and shot_policy.is_music(spec):
                receipt.update(status='omitted_by_policy',reason='Current explicit no-score policy',proposal=raw)
            else:
                # Semantic identity persists across operation names; replay cannot buy new slots.
                signature=key({'asset_id':asset_id,'spec':spec})
                previous=next((v for k,v in intents.items() if k!=ident and v.get('intent_signature')==signature and v.get('status') not in ('rejected','needs_clarification')),None)
                if previous:
                    receipt.update(status='already_recorded',original_receipt=previous['id'],asset_id=asset_id,intent_signature=signature)
                else:
                    row=s.data['assets'].get(asset_id,{})
                    old_target=row.get('spec',{}).get('prompt') or row.get('spec',{}).get('description') or row.get('spec',{}).get('path')
                    if action=='update' and old_target==target and all(row.get('spec',{}).get(k)==v for k,v in spec.items() if k!='id'):
                        receipt.update(status='no_change',asset_id=asset_id,reason='Same generation target retained; attempts cannot be refreshed',intent_signature=signature)
                    else:
                        receipt.update(status='declared',asset_id=asset_id,intent_signature=signature,
                            base_generation=row.get('generation',0),proposal=raw)
                        # Save intent and pending spec in the same Store transaction via declare().
                        agent.registry.declare(asset_id,spec)
        s.save(); s.event('asset_action_recorded',receipt=deepcopy(receipt)); result.append(deepcopy(receipt))
        write(s.run/'asset_actions'/f'{ident}.json',receipt)
    return result


def settle(agent):
    for receipt in agent.s.data.get('asset_action_receipts',{}).values():
        if receipt.get('status')!='declared':continue
        row=agent.s.data['assets'].get(receipt['asset_id'],{})
        if row.get('pending_prompt'):continue
        version=next((v for v in reversed(row.get('versions',[])) if v.get('version',0)>receipt['base_generation']),None)
        receipt.update(status=('published_accepted' if version.get('feedback_accepted') else 'published_needs_review') if version else 'failed_or_omitted',
            generated_version=deepcopy(version),generation=row.get('generation'),no_attempts_reset=True)
        agent.s.save();write(agent.s.run/'asset_actions'/f'{receipt["id"]}.json',receipt)
        agent.s.event('asset_action_settled',receipt=deepcopy(receipt))

    for receipt in agent.s.data.get('asset_action_receipts',{}).values():
        if receipt.get('status')!='inspected_followup_recorded':continue
        children=[agent.s.data['asset_action_receipts'].get(x['id'],x) for x in receipt.get('followup_receipts',[])]
        if children and all(x.get('status')=='published_accepted' for x in children):
            receipt.update(status='inspected_repaired_future_asset',next_action_required=False,followup_receipts=deepcopy(children))
            agent.s.save();write(agent.s.run/'asset_actions'/f'{receipt["id"]}.json',receipt)
            agent.s.event('asset_inspection_followup_settled',receipt=deepcopy(receipt))
