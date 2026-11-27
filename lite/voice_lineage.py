"""Host-owned, SHA-bound authority for an accepted direct canonical voice clone."""
from copy import deepcopy
import hashlib,json,math
from .storage import sha
from .asset_lifecycle import voice_report_usable

VERSION='published_direct_voice_clone_v1'

class VoiceLineageError(ValueError):
    """Invalid host evidence for a selected voice reference."""

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def project_canonical_roles(plan, config):
    """Map a director's descriptive use label to the role consumed by voice review."""
    cast = config.get('generation_contract', {}).get('cast', {})
    assets = plan.get('assets')
    if not isinstance(cast, dict) or not isinstance(assets, list):
        return plan
    voices = {spec['voice_asset_id']: role for role, spec in cast.items()
              if isinstance(spec, dict) and isinstance(spec.get('voice_asset_id'), str)}
    rows = {a['id']: a for a in assets if isinstance(a, dict) and isinstance(a.get('id'), str)}
    normalized = []
    for asset in assets:
        role = voices.get(asset.get('id')) if isinstance(asset, dict) and isinstance(asset.get('id'), str) else None
        if role is None:
            normalized.append(asset)
            continue
        ident = asset['id']
        if 'speaker' in asset and asset['speaker'] != role:
            raise ValueError(f'Canonical voice asset {ident} has a speaker different from its registered role')
        usage = asset.get('role')
        mode = asset.get('generator')
        parent_id = asset.get('reference_asset_id')
        parent = rows.get(parent_id) if isinstance(parent_id, str) else None
        if mode == 'voice_clone' and parent and (parent is asset or voices.get(parent['id'], role) != role
                                                  or parent.get('speaker', role) != role):
            raise ValueError(f'Canonical voice clone {ident} needs a direct same-speaker reference')
        if usage == 'voice':
            normalized.append(asset)
            continue
        if (asset.get('speaker') != role or asset.get('kind') != 'audio'
                or mode not in ('voice_design', 'voice_clone')
                or asset.get('track_type') not in (None, 'synthetic_voice', 'isolated_voice')
                or not isinstance(usage, str) or not usage.strip()
                or usage.lower() in ('music', 'background_music', 'score', 'ambience', 'foley', 'sound_effect', 'scene', 'prop', 'character')
                or ('source_role' in asset and asset['source_role'] != usage)):
            raise ValueError(f'Canonical voice asset {ident} needs matching speaker, audio kind, and an explicit voice role or generator')
        if mode == 'voice_clone':
            # A matching target speaker alone cannot authorize cloning a different person.
            if (not parent or parent is asset or parent.get('speaker') != role or parent.get('kind') != 'audio'
                    or voices.get(parent['id'], role) != role or parent.get('generator') == 'voice_clone'):
                raise ValueError(f'Canonical voice clone {ident} needs an explicitly declared direct same-speaker reference')
        normalized.append({**asset, 'source_role': usage, 'role': 'voice'})
    return {**plan, 'assets': normalized}


def annotate(references,catalog,state,config):
    if config.get('generation_contract',{}).get('version')!='cast_reference_contract_v2':return references
    by_id={x['id']:x for x in catalog};cast=config['generation_contract'].get('cast',{});result=deepcopy(references)
    for ref in result:
        item=by_id.get(ref['id']);row=state.get('assets',{}).get(ref['id'],{})
        if not item or item.get('generator')!='voice_clone' or item.get('review_status')!='PASS' or item.get('kind')!='audio':continue
        spec=row.get('spec',{});parent_id=spec.get('reference_asset_id');parent=by_id.get(parent_id)
        roles=[r for r,s in cast.items() if s.get('voice_asset_id')==parent_id and r==spec.get('speaker')]
        if len(roles)!=1 or not parent or parent.get('speaker')!=roles[0] or parent.get('review_status')!='PASS' or parent.get('kind')!='audio':continue
        role=roles[0]
        if item.get('speaker')!=role or item.get('track_type') not in ('synthetic_voice','isolated_voice') or parent.get('track_type') not in ('synthetic_voice','isolated_voice'):continue
        if item.get('provenance',{}).get('reference_sha256')!=parent['sha256']:continue
        approvals=[a for a in state.get('asset_action_receipts',{}).values() if a.get('asset_id')==item['id'] and a.get('status')=='published_accepted' and a.get('proposal',{}).get('reference_asset_id')==parent_id and a.get('generated_version',{}).get('sha256')==item['sha256'] and a.get('generated_version',{}).get('effective_reference_asset_id')==parent_id]
        if len(approvals)!=1:continue
        approval=approvals[0];v=approval['generated_version']
        if not v.get('feedback_accepted') or not voice_report_usable(v,v.get('review',{})):continue
        try:
            intact=sha(parent['path'])==parent['sha256'] and sha(item['path'])==item['sha256'] and sha(ref['path'])==ref['sha256']
        except OSError:intact=False
        if not intact:continue
        if ref['sha256']!=item['sha256']:
            w=ref.get('reference_window',{});start,duration=w.get('start_s'),w.get('seconds')
            if (w.get('source_sha256')!=item['sha256'] or w.get('source_path')!=item['path']
                    or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in (start,duration))
                    or not 0<=start<start+duration<=item['seconds'] or duration<2):continue
        ref['canonical_voice_authority']={'version':VERSION,'canonical_voice_asset_id':parent_id,'canonical_voice_path':parent['path'],'canonical_voice_sha256':parent['sha256'],'speaker':role,'asset_id':item['id'],'asset_path':item['path'],'asset_sha256':item['sha256'],'selected_sha256':ref['sha256'],'asset_action_receipt_id':approval['id'],'asset_action_receipt_sha256':digest(approval),'published_accepted':True}
    return result

def check(ref,canonical_id,role):
    a=ref.get('canonical_voice_authority',{})
    if (a.get('version')!=VERSION or a.get('canonical_voice_asset_id')!=canonical_id or a.get('speaker')!=role
            or ref.get('speaker')!=role or ref.get('generator')!='voice_clone' or ref.get('kind')!='audio'
            or ref.get('track_type') not in ('synthetic_voice','isolated_voice') or ref.get('review_status')!='PASS'
            or a.get('asset_id')!=ref['id'] or a.get('selected_sha256')!=ref.get('sha256') or a.get('published_accepted') is not True
            or not a.get('asset_action_receipt_id') or not a.get('asset_action_receipt_sha256')
            or ref.get('provenance',{}).get('reference_sha256')!=a.get('canonical_voice_sha256')):
        raise VoiceLineageError('Voice clone lacks exact canonical parent/role/SHA and accepted asset receipt')
    try:
        intact=sha(a['canonical_voice_path'])==a['canonical_voice_sha256'] and sha(a['asset_path'])==a['asset_sha256'] and sha(ref['path'])==ref['sha256']
    except (OSError,KeyError):intact=False
    if not intact:raise VoiceLineageError('Canonical or cloned voice bytes changed after lineage approval')
    if ref['sha256']!=a['asset_sha256']:
        w=ref.get('reference_window',{})
        if w.get('source_sha256')!=a['asset_sha256'] or w.get('source_path')!=a['asset_path']:
            raise VoiceLineageError('Voice clone excerpt lost its original SHA binding')
    return a

def resolve(by_id,canonical_id,role):
    canonical=by_id.get(canonical_id);aliases=[]
    for ref in by_id.values():
        if ref.get('canonical_voice_authority',{}).get('version')==VERSION and ref['canonical_voice_authority'].get('canonical_voice_asset_id')==canonical_id:
            check(ref,canonical_id,role);aliases.append(ref)
    if canonical is not None and aliases or len(aliases)>1:
        raise ValueError('Ambiguous canonical and cloned voice references for one speaker')
    return canonical if canonical is not None else aliases[0] if aliases else None
