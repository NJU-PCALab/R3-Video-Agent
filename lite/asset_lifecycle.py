"""Required-asset readiness and exact owned-worker handoff for server stories."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,math,time
from .storage import Pending,Stopped,Unavailable,read,sha,write
from .audio_parameters import tts_language
from . import task_parameters

VERSION='asset_worker_lifecycle_v1'
CALIBRATION='independent_voice_calibration_v1'
CALIBRATION_REVIEW_GUIDE = '''
For independent_voice_calibration_v1 the requested text is a neutral calibration
utterance, not scene dialogue. Assess its actual text, stable single-speaker identity,
intelligibility, naturalness and absence of background music. Scene text and dramatic
scene transitions are outside this sample's purpose: do not fail it for not speaking
scene dialogue or performing a scene action. Record unvalidated dramatic performance
separately, never as a validated capability. Actual timbre, speech, noise, music or
multi-speaker defects still remain defects. Return background_music as boolean or null.
'''

def enabled(config):return config.get('execution_contract')==VERSION

def calibration_spec(spec,config):
    spec=task_parameters.voice_spec(spec,config)
    if not enabled(config) or spec.get('role')!='voice' or spec.get('generator','voice_design') not in ('voice_design','voice_clone'):
        return deepcopy(spec)
    language=tts_language(spec.get('language')).lower()
    text=spec.get('calibration_text')
    if text is None:
        if language not in ('zh','chinese','中文','en','english'):
            raise Unavailable('This voice language needs explicit independent calibration_text')
        text=('早晨的风吹过窗前，我把桌上的书轻轻合上。请慢慢说，把事情清清楚楚地告诉我。' if language in ('zh','chinese','中文') else
              'The morning breeze comes through the open window. I close the book on the table and listen carefully. Please take your time and tell me what happened.')
    if not isinstance(text,str) or len(text.strip())<20:
        raise Unavailable('Independent voice calibration text is missing or too short; revise its input, never pad or loop the waveform')
    return {**deepcopy(spec),'scene_text':spec.get('scene_text',spec.get('text')),'text':text.strip(),'calibration_policy':CALIBRATION,
            'calibration_purpose':'clean canonical timbre sample; not dialogue added to the film'}


def review_target(spec, config):
    """Expose the text actually generated, without a competing scene-text target."""
    target = calibration_spec(spec, config)
    if target.get('calibration_policy') == CALIBRATION:
        target.pop('scene_text', None)
        target['review_scope'] = 'calibration text, identity and purity; scene performance unvalidated'
    return target


def normalize_voice_review(review):
    """Only unambiguous wire-type aliases; preserve original replies and defects."""
    result = deepcopy(review)
    value = result.get('background_music')
    if isinstance(value, str) and value.strip().lower() in ('none', 'false', 'true'):
        result['background_music'] = value.strip().lower() == 'true'
        result['protocol_normalization'] = {
            'version': 'voice_boolean_alias_v1', 'original_review': deepcopy(review),
            'conversions': [{'field': 'background_music', 'original': value,
                             'effective': result['background_music']}],
            'content_rejudged': False}
    return result


def input_digest(row):
    spec=calibration_spec(row['spec'],{'execution_contract':VERSION})
    values={k:spec.get(k) for k in ('generator','text','calibration_policy','language','reference_asset_id','path','seconds')}
    values['prompt']=row.get('pending_prompt') or spec.get('prompt') or spec.get('description')
    return hashlib.sha256(json.dumps(values,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def admit_pending(row):
    if row.get('dependency_state')=='blocked' and row.get('blocked_input_sha256')==input_digest(row):
        raise Pending('Required asset has retained failed candidates; supply a corrected distinct asset revision, not repeated resume',status='needs_asset_repair')

def candidate_takes(agent,ident,row):
    """One explicit run/asset/input-bound exception; retain the two failed slots."""
    from . import character_stages
    stage_takes = character_stages.candidate_takes(agent,ident,row)
    if stage_takes is not None:
        return stage_takes
    permit=agent.config.get('asset_attempt_exception') or {}
    if permit.get('asset_id')!=ident:
        admit_pending(row)
        return (1,2)
    agent.s.check_stop()
    path=Path(permit.get('receipt',''))
    if not path.is_file() or sha(path)!=permit.get('sha256'):
        raise Pending('Asset exception receipt is missing or changed',status='needs_configuration')
    receipt=read(path)
    version=row.get('pending_version')
    originals={k:row.get('lifecycle_attempts',{}).get(k) for k in ('1/1','1/2')}
    digest=hashlib.sha256(json.dumps(originals,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    valid=(receipt.get('version')=='one_asset_candidate_exception_v1'
           and receipt.get('explicit_user_approval') is True
           and receipt.get('run')==str(agent.s.run.resolve())
           and receipt.get('asset_id')==ident and receipt.get('asset_version')==version==1
           and receipt.get('authorized_take')==3 and receipt.get('additional_candidates')==1
           and receipt.get('input_sha256')==row.get('blocked_input_sha256')==input_digest(row)
           and receipt.get('original_attempts_sha256')==digest
           and all(r and r.get('failure') and r.get('actual') is None for r in originals.values()))
    if not valid:
        raise Pending('Asset exception does not match the retained run/input/failures',status='needs_configuration')
    if f'{version}/3' in row.get('lifecycle_attempts',{}):
        raise Pending('The single authorized asset candidate is already terminal; no fourth slot',status='needs_asset_repair')
    claim=agent.s.run/'asset_authorizations'/(permit['sha256']+'.json')
    value={'authorization_receipt':str(path),'authorization_sha256':permit['sha256'],
           'run':str(agent.s.run.resolve()),'asset_id':ident,'asset_version':version,
           'take':3,'original_attempts_sha256':digest,'prior_attempts_retained':True}
    previous=read(claim)
    if previous is not None and previous!=value:
        raise Pending('Existing asset authorization claim differs; reconcile it',status='needs_configuration')
    if previous is None:
        write(claim,value)
        agent.s.event('explicit_asset_exception_admitted',**value)
    return (3,)

def requeue_declared(row,before):
    if before.get('dependency_state')!='blocked' or input_digest(row)==before.get('blocked_input_sha256'):return
    consumed=max(before.get('generation',0),before.get('pending_version') or 0)
    row.update(generation=consumed,pending_version=consumed+1,dependency_state='repair_requested')
    row.setdefault('repair_revisions',[]).append({'from_version':consumed,'to_version':consumed+1,'previous_input_sha256':before.get('blocked_input_sha256'),'new_input_sha256':input_digest(row)})

def voiced_seconds(observation,source_sha,duration):
    if observation.get('source_audio_sha256')!=source_sha or not isinstance(observation.get('vad'),list):
        raise Pending('Independent calibration VAD lacks its exact audio binding',status='needs_voice_observation')
    intervals=[]
    for row in observation['vad']:
        a,b=row.get('start'),row.get('end')
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (a,b)) or not 0<=a<b<=duration+.001:
            raise Pending('Independent calibration VAD has invalid time windows',status='needs_voice_observation')
        intervals.append((a,b))
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    seconds=sum(b-a for a,b in merged)
    if seconds<2:raise Unavailable('Actual unpadded calibration speech is shorter than 2 seconds; original waveform and observation retained')
    if not isinstance(observation.get('text'),str) or not observation['text'].strip():
        raise Pending('VAD found speech but blind transcript is missing; do not regenerate for an observation gap',status='needs_voice_observation')
    return seconds

def required_ids(agent):
    from . import completion_flow
    if completion_flow.enabled(agent.config):return completion_flow.required_ids(agent)
    ids=set()
    for spec in agent.config.get('generation_contract',{}).get('cast',{}).values():
        ids.update(x for x in (spec.get('image_asset_id'),spec.get('voice_asset_id')) if x)
    for clip in agent.s.data.get('plan',{}).get('clips',[]):
        ids.update(x for x in clip.get('asset_ids',[]) if isinstance(x,str))
    return ids

def voice_report_usable(actual,review):
    review=normalize_voice_review(review)
    kind=review.get('content_type')
    native=actual.get('track_type')=='synthetic_voice' and actual.get('generator') in ('voice_design','voice_clone')
    # Existing native-TTS reviews legitimately use speech; imported audio still needs explicit single-speaker evidence.
    typed=kind=='single_speaker_voice' or (kind=='speech' and native)
    return typed and review.get('background_music') is False and review.get('speaker_count',1)==1 and not any(x.get('severity') in ('major','critical') for x in review.get('findings',[]) if isinstance(x,dict))

def clarify_voice_review(agent,name,spec,actual,review,media):
    review=normalize_voice_review(review)
    if not enabled(agent.config) or spec.get('role')!='voice' or review.get('decision')!='accept':return review
    if any(x.get('severity') in ('major','critical') for x in review.get('findings',[]) if isinstance(x,dict)):return review
    if review.get('background_music') is True or review.get('speaker_count',1) not in (None,1):return review
    native=actual.get('track_type')=='synthetic_voice' and actual.get('generator') in ('voice_design','voice_clone')
    kind=review.get('content_type');gap=review.get('background_music') not in (True,False) or kind in (None,'UNKNOWN','unknown','audio') or (kind=='speech' and not native)
    if not gap:return review
    from . import prompts
    binding={'asset_sha256':actual['sha256'],'original_review_sha256':hashlib.sha256(json.dumps(review,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    operation=name+'_voice_observation_'+actual['sha256'][:12]
    extra=agent.artifact(operation,lambda:agent.advisory(operation,prompts.ASSET_FEEDBACK+CALIBRATION_REVIEW_GUIDE+'\nFor this same audio only, explicitly return content_type single_speaker_voice|speech|music|noise|unknown, background_music boolean or null, speaker_count, observed_events and findings. An observation gap is not a reason for new generation.',{'original_review':review,'actual_binding':binding,'purpose':'clarify missing purity observations only; preserve actual defects'},provider='gemini',media=media,max_tokens=3500))
    extra=normalize_voice_review(extra)
    merged=deepcopy(review)
    for field in ('content_type','background_music','speaker_count'):
        if field in extra:merged[field]=extra[field]
    merged['findings']=deepcopy(review.get('findings',[]))+[x for x in extra.get('findings',[]) if x not in review.get('findings',[])]
    merged['observation_clarification']={'binding':binding,'operation':operation,'raw_clarification':deepcopy(extra),'original_review_retained':True}
    remaining=merged.get('background_music') not in (True,False) or merged.get('content_type') in (None,'UNKNOWN','unknown','audio') or (merged.get('content_type')=='speech' and not native)
    if remaining:raise Pending('Bounded same-asset observation remains unknown; no new generation',status='needs_asset_observation')
    return merged

def reconcile_blocked_voice_observation(agent,ident,row):
    """Reconcile retained generic-type reports; never request another candidate."""
    if row.get('dependency_state')!='blocked' or row.get('spec',{}).get('role')!='voice' or row.get('blocked_input_sha256')!=input_digest(row):return False
    version=row.get('pending_version')
    records=sorted(row.get('lifecycle_attempts',{}).values(),key=lambda r:r.get('take',0),reverse=True)
    for record in records:
        original=record.get('actual') or {};review=original.get('review') or {}
        if record.get('version')!=version or review.get('decision')!='accept' or review.get('content_type')!='audio' or review.get('background_music') is not False or review.get('speaker_count',1) not in (None,1):continue
        if any(x.get('severity') in ('major','critical') for x in review.get('findings',[]) if isinstance(x,dict)):continue
        path=Path(original.get('path',''))
        if not path.is_file() or sha(path)!=original.get('sha256'):
            raise Pending('Retained voice observation media changed; do not regenerate',status='needs_asset_observation')
        name=f'{version}_{record["take"]}_{hashlib.sha256(ident.encode()).hexdigest()[:12]}'
        effective=clarify_voice_review(agent,'asset_review_'+name,row['spec'],original,review,[{'path':str(path),'label':'Actual reference asset'}])
        derived={**deepcopy(original),'review':effective,'feedback_accepted':bool(effective.get('reason') and effective.get('observed_events') and voice_report_usable(original,effective)),
                 'derived_observation_reconciliation':{'original_review_retained':True,'original_candidate_take':record['take'],'new_generation':False}}
        record_attempt(row,version,record['take'],actual=derived)
        if derived['feedback_accepted']:
            finish_asset(agent,ident,row,[derived],version)
            agent.s.event('retained_voice_observation_reconciled',asset_id=ident,version=version,take=record['take'],media_sha256=original['sha256'],new_generation=False)
            return True
    agent.registry.save()
    return False

def usable_version(row):
    voice=row.get('spec',{}).get('role')=='voice'
    for v in reversed(row.get('versions',[])):
        if v.get('character_age_defects'):continue
        path=Path(v.get('path',''))
        if not path.is_file() or not v.get('sha256') or sha(path)!=v['sha256']:continue
        if voice:
            seconds=v.get('seconds');review=v.get('review',{})
            if isinstance(seconds,bool) or not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or not 2<=seconds<=15:continue
            if v.get('kind')!='audio' or v.get('track_type') not in ('synthetic_voice','isolated_voice'):continue
            from .calibration_recovery import valid as calibration_reference_valid
            if calibration_reference_valid(v):return v
            accepted=v.get('feedback_accepted') is True or (review.get('verdict')=='PASS' and v.get('feedback_accepted') is not False)
            if not accepted:
                from .explicit_recovery import voice_draft_valid
                accepted = voice_draft_valid(v)
            if not accepted or not voice_report_usable(v,review):continue
            if any(x.get('severity') in ('major','critical') for x in review.get('findings',[]) if isinstance(x,dict)):continue
        return v
    return None

def readiness_issues(agent):
    if not enabled(agent.config):return []
    issues=[]
    for ident in sorted(required_ids(agent)):
        row=agent.s.data['assets'].get(ident)
        if not row:
            issues.append({'asset_id':ident,'reason':'Required asset was not declared; correct the text asset plan'});continue
        if row['spec'].get('role')=='music' and agent.config.get('editing_policy',{}).get('background_music')=='none':continue
        if usable_version(row) is None:
            issues.append({'asset_id':ident,'reason':row.get('last_generation_failure','No verified usable required media'),'state':row.get('dependency_state','missing')})
    return issues

def admit_assets(agent):
    issues=readiness_issues(agent)
    if issues:
        agent.s.data['asset_readiness']={'status':'blocked','issues':issues};agent.s.save()
        raise Pending('Required assets are not ready: '+json.dumps(issues,ensure_ascii=False),status='needs_asset_repair')
    if enabled(agent.config):
        warnings=[{'asset_id':i,'reason':'Existing reference has retained review warnings; readiness is not content acceptance'} for i in required_ids(agent) if i in agent.s.data['assets'] and usable_version(agent.s.data['assets'][i]) and usable_version(agent.s.data['assets'][i]).get('feedback_accepted') is False]
        agent.s.data['asset_readiness']={'status':'ready_with_warnings' if warnings else 'ready','required_ids':sorted(required_ids(agent)),'warnings':warnings};agent.s.save()

def record_attempt(row,version,take,*,failure=None,actual=None):
    records=row.setdefault('lifecycle_attempts',{});ident=f'{version}/{take}'
    value={'version':version,'take':take,'failure':str(failure) if failure is not None else None,'actual':deepcopy(actual)}
    old=records.get(ident)
    if old is None:records[ident]=value
    elif old!=value:
        # Preserve a prior technical failure and later review as distinct observations.
        observations=row.setdefault('lifecycle_attempt_observations',[])
        if value not in observations:observations.append(value)

def finish_asset(agent,ident,row,successful,version, *, optional=False):
    if row['spec'].get('role')=='voice':
        accepted=[v for v in successful if usable_version({'spec':row['spec'],'versions':[v]})]
    else:accepted=[v for v in successful if not v.get('character_age_defects')]
    if optional:accepted=[v for v in accepted if v.get('feedback_accepted')]
    if accepted:
        best=next((v for v in reversed(accepted) if v.get('feedback_accepted')),accepted[-1])
        agent.registry.publish(ident,best);row['active_reference_version']=best['version'];row['dependency_state']='ready';row.pop('last_generation_failure',None)
    elif usable_version(row) is not None:
        row['dependency_state']='ready_previous_version';row['failed_update_version']=version;row['active_reference_version']=usable_version(row)['version']
    elif not optional and ident in required_ids(agent):
        row['dependency_state']='blocked';row['pending_version']=version
        row['blocked_input_sha256']=input_digest(row)
        row['pending_prompt']=row.get('pending_prompt') or row['spec'].get('prompt') or row['spec'].get('description') or 'Required asset repair'
        agent.registry.save()
        raise Pending('Required asset failed its bounded candidates; original attempts retained: '+ident,status='needs_asset_repair')
    else:
        row['dependency_state']='optional_unavailable'
    row['generation']=version;row['pending_version']=None;row['pending_prompt']=None;agent.registry.save()

def _positive(policy,name,default):
    x=policy.get(name,default)
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:
        raise Pending('Invalid bounded lifecycle policy '+name,status='needs_configuration')
    return float(x)

def await_gpu(store,config,folder,admit):
    policy=config.get('worker_lifecycle_policy') or {};limit=_positive(policy,'queue_timeout_seconds',21600);poll=_positive(policy,'resource_poll_seconds',5)
    started=time.monotonic();waiting=False;last_reason=None
    while True:
        store.check_stop()
        try:admit()
        except Pending as exc:
            if exc.status!='awaiting_resource':raise
            if time.monotonic()-started>=limit:
                write(folder/'resource_queue.json',{'status':'bounded_wait_exhausted','reason':str(exc),'new_submission':False})
                raise Pending('Known GPU resource queue reached its bounded wait; no worker was submitted',status='awaiting_resource') from exc
            if str(exc)!=last_reason or not waiting:
                store.status('awaiting_resource',stage='worker.resource_queue',folder=str(folder),reason=str(exc),runner_waiting=True)
                write(folder/'resource_queue.json',{'status':'waiting','reason':str(exc),'new_submission':False,'observed_at':time.time()});last_reason=str(exc)
            waiting=True;time.sleep(max(0,min(poll,limit-(time.monotonic()-started))));continue
        if waiting:
            write(folder/'resource_queue.json',{'status':'admitted','wait_seconds':time.monotonic()-started,'new_submission':False,'observed_at':time.time()})
            store.status('running',stage='worker.resource_admitted',folder=str(folder))
        return

def await_exit(store,config,folder):
    from .worker_wait import process_identity,same_process,group_members,reconcile_launch
    expected=read(folder/'process.json')
    if expected is None:
        if (folder/'submitted.json').exists():expected=reconcile_launch(folder)
        else:raise Pending('Terminal worker result lacks original launch identity',status='needs_worker_reconciliation')
    submitted_command = None
    if expected.get('start_ticks') and expected.get('argv') == ['']:
        submission = read(folder / 'submitted.json', {})
        command = submission.get('command') if isinstance(submission, dict) else None
        if isinstance(command, list) and command and command[0] and all(isinstance(arg, str) for arg in command):
            submitted_command = command
    policy=config.get('worker_lifecycle_policy') or {};limit=_positive(policy,'handoff_timeout_seconds',90);poll=_positive(policy,'exit_poll_seconds',.25);started=time.monotonic();announced=False
    while True:
        store.check_stop();actual=process_identity(expected['pid']);members=group_members(expected['pid'])
        # The launch snapshot can have a birth time before cmdline is visible.
        if submitted_command and same_process(actual, {**expected, 'argv': submitted_command}):
            write(folder / 'launch_identity_reconciled.json', {
                'process': expected, 'observed': actual, 'time': time.time(),
                'result_retained': True, 'accepted_before_exit': False,
            })
            expected = actual
            submitted_command = None
        # Linux can clear the exiting task's cmdline before it becomes a zombie.
        # The unchanged birth identity is not PID reuse. Keep waiting (bounded)
        # without accepting the result or signalling a process with incomplete argv.
        exit_transition=bool(actual and expected.get('start_ticks') and actual.get('pid')==expected['pid'] and actual.get('start_ticks')==expected['start_ticks'] and actual.get('argv')==[''])
        if exit_transition and not (folder/'exit_identity_transition.json').exists():
            write(folder/'exit_identity_transition.json',{'process':expected,'observed':actual,'time':time.time(),'result_retained':True,'accepted_before_exit':False})
        if actual is not None and not same_process(actual,expected) and not exit_transition:
            raise Pending('Original worker PID was reused; preserve result and reconcile identity',status='needs_worker_reconciliation')
        if actual is None and not members:
            write(folder/'exit_handoff.json',{'process':expected,'verified_at':time.time(),'owned_process_and_group_ended':True,'wait_seconds':time.monotonic()-started,'result_retained':True});return
        if not announced:
            store.event('worker_result_waiting_for_exit',folder=str(folder),process=expected);announced=True
        if time.monotonic()-started>=limit:
            write(folder/'exit_handoff_pending.json',{'process':expected,'observed_at':time.time(),'live_owned_members':list(members.values()),'result_retained':True})
            raise Pending('Original result retained; owned worker has not completed exit handoff',status='awaiting_worker_exit')
        time.sleep(poll)

def cleanup_user_stop(store,folder):
    from .worker_wait import stop_owned_worker
    expected=read(folder/'process.json')
    if expected is None:return
    outcome=stop_owned_worker(folder,expected,ready=lambda:False,grace=5)
    write(folder/'user_stop_handoff.json',{'process':expected,'outcome':outcome,'time':time.time(),'STOP_retained':True})
    if outcome['outcome']!='stopped':raise Pending('User STOP retained; original worker shutdown requires reconciliation',status='stop_cleanup_unconfirmed')

def admit_entry(config,*,new_story=False):
    from . import binding_contract, shot_policy
    shot_policy.validate_config(config)
    if not enabled(config):
        if new_story:raise Pending('New stories require the deployed asset/worker execution contract',status='needs_configuration')
        return
    from .attempt_budget import limit
    cap = limit(config)
    if config.get('max_real_takes_per_clip')!=cap or config.get('review_system',{}).get('max_real_takes_per_clip')!=cap:
        raise Pending('Video candidate limits disagree across execution and review configuration',status='needs_configuration')
    if not config.get('generation_contract',{}).get('cast'):
        raise Pending('Execution contract needs an explicit canonical cast',status='needs_configuration')
    if binding_contract.enabled(config):
        try:
            binding_contract.validate_cast(config['generation_contract']['cast'])
        except ValueError as exc:
            raise Pending(str(exc), status='needs_configuration') from exc
    if task_parameters.enabled(config):
        task_parameters.validate(config)
        if config.get('gpu_resource_admission')!='assigned_gpu_idle_v1':raise Pending('Assigned GPU resource admission is required',status='needs_configuration')
    elif config.get('gpu_resource_admission')!='gpu1_idle_v1':raise Pending('GPU1 resource admission is required',status='needs_configuration')

def verify_runtime(config):
    """Verify the workspace and the installed backend that will actually execute."""
    root = Path(__file__).resolve().parents[1]
    cap_path = root / 'execution_capabilities.json'
    try:
        manifest = read(Path(config['framework_release']) / 'manifest.json', required=True)['files']
        if sha(cap_path) != manifest['execution_capabilities.json']:
            return False
        cap = read(cap_path, required=True)
        if cap.get('version') != VERSION or not cap.get('files'):
            return False
        for name, digest in cap['files'].items():
            if manifest.get(name) != digest or sha(root / name) != digest:
                return False
        shared = Path(config['production_root'])
        backend = [(name[len('backend/'):], digest) for name, digest in manifest.items()
                   if name.startswith('backend/')]
        return bool(backend) and all(sha(shared / name) == digest for name, digest in backend)
    except (OSError, ValueError, KeyError, TypeError):
        return False
