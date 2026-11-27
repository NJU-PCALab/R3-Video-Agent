"""One observation set and one advisory director; media findings remain explicit."""
from copy import deepcopy
from pathlib import Path
import json
import re

from .storage import Pending, Unavailable, key, sha, write
from . import evidence_policy

VERSION='completion_observed_v1'
CHECK_CONTRACT='dimension_evidence_v2'
OBSERVATION_PROTOCOL=VERSION+'/'+CHECK_CONTRACT
VISUAL=('identity_continuity','scene_continuity','visual_quality')
AV=('speech_completeness','dialogue_naturalness','sentence_ending','source_binding',
    'lip_sync','sound_event_alignment','audio_quality','unexpected_music')


def can_start(agent,cid,take):
    """No new protocol may replace an already submitted original review."""
    from .storage import read
    prefix=f'review_panel/{cid}/{take}/'
    return not any(read(p,{}).get('name','').startswith(prefix)
        for p in (agent.s.run/'calls').glob('*/request.json'))


def _dict(value):
    return value if isinstance(value,dict) else {}


def _rows(value):
    return value if isinstance(value,list) else []


def _grounded(item):
    observation = item.get('observation')
    def content(value):
        if isinstance(value,str):
            return value.strip().lower().strip('.!。 ') not in ('','clear','defect','unknown','observed','not_applicable','not applicable','n/a','none')
        return isinstance(value,dict) and any(content(v) for k,v in value.items() if k not in ('status','time_s','start_s','end_s'))
    return bool(content(observation) and not evidence_policy.generic_limit(item))


def _checks(raw):
    """Canonical request view; a conflicting alias can never override a defect."""
    checks={};audit=[]
    severity={'defect':3,'unknown':2,'not_applicable':1,'clear':0}
    for original,item in _dict(raw.get('checks')).items():
        category=evidence_policy.canonical_category(original)
        item=_dict(item)
        if category in checks and checks[category] != item:
            audit.append({'reason':'conflicting_check_aliases','category':category,'source_category':original,
                'original_checks':deepcopy(raw.get('checks'))})
            prior=checks[category]
            # Missing evidence is unknown even when the status string says clear.
            rank=lambda row:severity.get(row.get('status'),2) if _grounded(row) else 2
            if rank(item)>rank(prior):checks[category]=deepcopy(item)
        else:checks[category]=deepcopy(item)
    return checks,audit


def normalize(role,raw,required):
    raw=_dict(raw);checks,audit=_checks(raw);findings=[];protocol_gaps=[]
    # Event summaries are useful for Replan, but independent dimension evidence
    # must not become eleven unknown content findings when this redundant list is
    # empty. Do not synthesize narrative events from check text.
    observed=raw.get('status')=='observed' and (bool(_rows(raw.get('observed_events')))
        or any(_grounded(item) for item in checks.values()))
    if raw.get('status')=='observed' and not _rows(raw.get('observed_events')):
        protocol_gaps.append({'kind':'missing_event_overview','role':role,
            'observation':'No event overview supplied; individual check evidence is evaluated separately.'})
    for category in required:
        item=_dict(checks.get(category));status=item.get('status');grounded=_grounded(item)
        if status=='defect' and grounded:
            findings.append({'id':f'{role}:check:{category}','role':role,'category':category,
                'severity':item.get('severity') if item.get('severity') in ('minor','major','critical') else 'major',
                'observation':item['observation'],'time_s':item.get('time_s'),'state':'open','kind':'observed_content_defect'})
        elif status not in (('clear','not_applicable') if role=='audiovisual' else ('clear',)) or not grounded:
            findings.append({'id':f'{role}:check:{category}','role':role,'category':category,
                'severity':'uncertain','observation':item.get('observation','Required actual observation is missing'),
                'state':'open','kind':'review_uncertain'})
    for index,f in enumerate(_rows(raw.get('findings'))):
        if not isinstance(f,dict) or not f.get('observation') or evidence_policy.generic_limit(f):
            continue
        # Only explicitly typed observed defects request a new GPU candidate.
        if evidence_policy.owner(f.get('category')) not in (None,role):
            audit.append({'reason':'outside_reviewer_authority','original':deepcopy(f)});continue
        severity=f.get('severity','uncertain')
        findings.append({**deepcopy(f),'id':f'{role}:finding:{index+1}','role':role,'state':'open',
            'severity':severity if severity in ('major','critical','minor') else 'uncertain',
            'kind':'observed_content_defect' if severity in ('major','critical','minor') else 'review_uncertain'})
    return {'status':'observed' if observed else 'unknown','observed_events':_rows(raw.get('observed_events')),
        'checks':checks,'findings':findings,'filter_audit':audit,'raw_reply':deepcopy(raw),
        'check_contract':CHECK_CONTRACT,'protocol_gaps':protocol_gaps,
        'transcript':raw.get('heard_text',raw.get('transcript'))}


def _offscreen_lip_na(source,check):
    """An absent expected mouth needs no lip score, but still needs speech checks."""
    if not _grounded(check):return False
    observation=str(check.get('observation',''))
    if not re.search(r'off[ _-]?screen|out of (?:the )?frame|not visible|no (?:speaking )?mouth|画外|看不到.*嘴|无可见.*嘴',observation,re.I):
        return False
    lines=_rows(source.get('dialogue'));timeline=_rows(source.get('dialogue_timeline'))
    def visibility(index,line):
        value=line.get('on_screen_or_offscreen') or (_dict(timeline[index]).get('on_screen_or_offscreen') if index<len(timeline) else None)
        return str(value or '').lower().replace('-','_').replace(' ','_')
    return bool(lines) and all(visibility(index,_dict(line)) in ('off_screen','offscreen')
        for index,line in enumerate(lines))


def decision_findings(findings):
    """Normalize/deduplicate only ranking data; all original evidence is retained.

    Exact normalized descriptions plus overlapping times identify duplicates.
    Different wording, disjoint intervals and uncertain-vs-defect observations
    remain separate; there is no extra model reconciliation or voting.
    """
    from .closed_loop import intervals
    result=[]
    for finding in _rows(findings):
        if not isinstance(finding,dict):continue
        item=deepcopy(finding);original=item.get('category')
        item['category']=evidence_policy.canonical_category(original)
        if item['category']!=original:item['original_category']=original
        observation=item.get('observation')
        normalized=evidence_policy.normal(json.dumps(observation,ensure_ascii=False,sort_keys=True)
            if isinstance(observation,dict) else observation or '')
        times=intervals(item.get('time_s'))
        def duplicate(other):
            other_observation=other.get('observation')
            other_text=evidence_policy.normal(json.dumps(other_observation,ensure_ascii=False,sort_keys=True)
                if isinstance(other_observation,dict) else other_observation or '')
            other_times=intervals(other.get('time_s'))
            same_kind=(item.get('severity')=='uncertain')==(other.get('severity')=='uncertain')
            overlap=(not times and not other_times) or any(a<=d and c<=b for a,b in times for c,d in other_times)
            return bool(normalized and normalized==other_text and item.get('category')==other.get('category') and same_kind and overlap)
        prior=next((other for other in result if duplicate(other)),None)
        if prior is None:
            item['source_finding_ids']=[item['id']] if item.get('id') else []
            result.append(item)
        else:
            prior['source_finding_ids']=list(dict.fromkeys(prior['source_finding_ids']+([item['id']] if item.get('id') else [])))
            if {'uncertain':0,'minor':1,'major':2,'critical':3}.get(item.get('severity'),0)>{'uncertain':0,'minor':1,'major':2,'critical':3}.get(prior.get('severity'),0):
                prior['severity']=item['severity']
    return result


def quality(review):
    issues=decision_findings(review.get('findings'))
    complete=all(_dict(_dict(review.get('opinions')).get(r)).get('status')=='observed' for r in ('visual','audiovisual'))
    ok=complete and not issues
    return {'ok':ok,'complete':complete,'rank':None,'quality_status':'OK' if ok else 'draft/needs_review',
        'review_system':VERSION,'acceptance_basis':'actual_observations_not_director_label',
        'feedback_decision':'accept' if ok else 'revise' if any(f.get('severity') in ('major','critical') for f in issues) else 'keep_needs_review',
        'unresolved_findings':issues,'issue_ledger':issues,'retained_tradeoffs':[],
        'not_applicable_dimensions':list(dict.fromkeys(category
            for panel in _dict(review.get('opinions')).values() for category,item in _dict(_dict(panel).get('checks')).items()
            if isinstance(item,dict) and item.get('status')=='not_applicable' and _grounded(item)
            and not any(f.get('id')=='speech:unsupported_na:'+category for f in issues))),
        'protocol_gaps':[deepcopy(gap) for panel in _dict(review.get('opinions')).values() for gap in _rows(_dict(panel).get('protocol_gaps'))],
        'content_release_eligible':ok,
        'technical_progress_independent_of_content_acceptance':True}



VOICE_REFERENCE_AUTHORITY='host_voice_reference_authority_v1'


def voice_reference_authority(source,cast,voices,labels,audio_sha256,render_bindings=()):
    """Facts of the actual blind-reference order, never a role-ID naming guess.

    This view is made only after the blind request. Canonical references used by
    the blind observer and an approved render variant are different authorities;
    neither a voice_r1 suffix nor a clone name can renumber neutral REF labels.
    """
    references=[{'reference_label':labels.get(r['id']),
        'canonical_voice_asset_id':r['id'],'audio_sha256':r['sha256'],
        'canonical_speakers':sorted(role for role,spec in cast.items()
            if spec.get('voice_asset_id')==r['id'])} for r in voices]
    occurrences=[{'script_index':i,'speaker':line.get('speaker'),
        'canonical_voice_asset_id':cast.get(line.get('speaker'),{}).get('voice_asset_id'),
        'expected_reference':labels.get(cast.get(line.get('speaker'),{}).get('voice_asset_id'))}
        for i,line in enumerate(_rows(source.get('dialogue')))]
    value={'version':VOICE_REFERENCE_AUTHORITY,'authority':'host_actual_reference_bytes',
        'current_audio_sha256':audio_sha256,'references':references,
        'expected_source_occurrences':occurrences,
        'actual_render_voice_bindings':deepcopy(_rows(render_bindings)),
        'neutral_labels_are_not_asset_ids':True,'role_id_suffix_is_not_reference_number':True,
        'comparison_scope':'canonical timbre references; an approved render variant does not renumber them',
        'timbre_observation_is_model_evidence_not_identity_ground_truth':True}
    value['id']=key(value)
    return value


def voice_identity_findings(raw,authority):
    """One host-owned reading of the retained blind result; UNKNOWN stays unknown."""
    raw=_dict(raw);refs={r['reference_label']:r for r in authority['references']}
    occurrences=authority['expected_source_occurrences'];expected=list(occurrences)
    utterances=_rows(raw.get('utterances'))
    if len(expected)==1 and utterances:expected=expected*len(utterances)
    if raw.get('status')!='observed' or len(utterances)!=len(expected) or any(x.get('expected_reference') is None for x in expected):
        return [{'id':'speech:voice_observation','category':'voice_identity','severity':'uncertain',
            'state':'open','kind':'review_uncertain','host_reference_authority_id':authority['id'],
            'observation':'Independent current voice observation or canonical reference is incomplete; no identity inferred from an asset name.'}]
    findings=[]
    for i,(utterance,occurrence) in enumerate(zip(utterances,expected)):
        utterance=_dict(utterance);label=occurrence['expected_reference'];match=utterance.get('match')
        known=isinstance(match,str) and match in set(refs)|{'none','mixed'}
        grounded=bool(utterance.get('observation') and utterance.get('contrastive_observation'))
        determinate=grounded and known and utterance.get('status') in ('clear','defect')
        if determinate and utterance.get('status')=='clear' and match==label:continue
        wanted=refs.get(label,{});observed=refs.get(match,{}) if isinstance(match,str) else {}
        findings.append({'id':f'speech:voice_identity:{i}','category':'voice_identity','state':'open',
            'severity':'major' if determinate and (utterance.get('status')=='defect' or match!=label) else 'uncertain',
            'kind':'observed_content_defect' if determinate else 'review_uncertain',
            'observation':deepcopy(utterance),'expected_reference':label,'observed_reference':match,
            'source_script_index':occurrence['script_index'],
            'expected_canonical_voice_asset_id':wanted.get('canonical_voice_asset_id'),
            'expected_reference_audio_sha256':wanted.get('audio_sha256'),
            'observed_canonical_voice_asset_id':observed.get('canonical_voice_asset_id'),
            'observed_reference_audio_sha256':observed.get('audio_sha256'),
            'host_reference_authority_id':authority['id'],'mapping_reinterpretation_allowed':False})
    return findings


def guard_voice_decision(raw,authority):
    """Keep director prose raw in the audit; it cannot become reference authority."""
    clean=deepcopy(_dict(raw));audit=[]
    if authority is None:return clean,audit
    original_reason=clean.get('reason')
    audit.append({'kind':'director_voice_commentary_is_advisory',
        'original_director_reason':original_reason,
        'host_voice_reference_authority_id':authority['id'],
        'reason':'Only submitted reference labels and media SHA define the mapping; original director artifact is retained.'})
    kept=[]
    for finding in _rows(clean.get('additional_findings')):
        if isinstance(finding,dict) and finding.get('category') in (
                'voice_identity','voice_reference_identity','voice_reference_mapping','speaker_consistency'):
            audit.append({'kind':'outside_director_voice_authority','original':deepcopy(finding),
                'host_voice_reference_authority_id':authority['id']})
        else:kept.append(finding)
    clean['additional_findings']=kept
    mapping='; '.join(str(r['reference_label'])+'='+r['canonical_voice_asset_id']+
        ' [audio SHA '+r['audio_sha256']+']' for r in authority['references']) or 'canonical references unavailable'
    expected='; '.join(str(r['speaker'])+' expects '+str(r['expected_reference'])
        for r in authority['expected_source_occurrences'])
    clean['reason']='Host voice-reference authority: '+mapping+'. '+expected+(
        '. Neutral labels cannot be inferred from role-ID suffixes or approved variant names. '
        'Retained blind findings remain unresolved unless their actual evidence supports a match; '
        'other recommendations and checked narrative observations remain separate.')
    clean.update(host_voice_reference_authority_id=authority['id'],
        voice_mapping_reinterpretation_allowed=False,
        voice_reference_proposals_require_registry_validation=True)
    return clean,audit

def review(agent,cid,candidate):
    from .review import visual_frames
    receipt=candidate['receipt'];source=candidate['source_clip'];take=candidate['take']
    if sha(receipt['path'])!=receipt['sha256'] or sha(receipt['audio_path'])!=receipt['audio_sha256']:
        raise Pending('Completion review media bytes differ from the original receipt')
    from . import dialogue_integrity
    strict = candidate.get('dialogue_integrity_version') == dialogue_integrity.VERSION
    protocol = OBSERVATION_PROTOCOL + '/literal_speech_v1' if strict else OBSERVATION_PROTOCOL
    name=f'completion/review/{cid}/{take}/{protocol}'
    previous=agent.s.data['selected'][-1] if agent.s.data['selected'] else None
    actual=evidence_policy.media_facts(candidate,previous)
    try:
        visual=[]
        for label,item in ([('previous',previous)] if previous else [])+[('current',candidate)]:
            frames=visual_frames(agent.config,agent.s.run,item['receipt'],f'completion_{cid}_{take}_{label}')
            visual.extend({'path':f['path'],'label':f'{label} actual silent frame at {f["time_s"]:.3f}s'} for f in frames['frames'])
        v=agent.artifact(name+'/visual',lambda:agent.advisory(name+'/visual',
            '''Observe complete sampled previous/current silent video sequences at 4fps.
Assess only visible identity, scene continuity and visual quality. No audio/mouth verdict,
no defect merely because unsampled frames cannot be ruled out. Describe actual visible actions, completed results, age/wardrobe and object states in chronological order with time spans in observed_events.
Preserve uncertainty about unsampled actions. Do not infer an expected result, age or identity from a target;
report concrete visible discontinuities and repeated actions so the director can compare them with source obligations.
Return {status:"observed|unknown",
observed_events:[],checks:{identity_continuity:{status:"clear|defect|unknown",observation,time_s},
scene_continuity:{status,observation,time_s},visual_quality:{status,observation,time_s}},
findings:[{category,severity:"major|minor|uncertain",observation,time_s}]}.
Do not invent facts from a reference or target. Actual content observations only.''',
            {'actual_media_facts':actual,'previous_available':bool(previous)},provider='doubao',media=visual,max_tokens=3500))
    except Unavailable as exc:
        agent.s.warning(name+'/visual',exc);v={'status':'unknown','reason':str(exc)}
    blind=None
    if source.get('dialogue') or strict:
        try:
            blind=agent.media.transcribe(receipt['audio_path'],f'{cid}_{take}_literal_export',detect_language=True) if strict else agent.media.transcribe(receipt['audio_path'],f'{cid}_{take}_export')
        except Unavailable as exc:
            agent.s.warning(name+'/blind_asr',exc);blind={'status':'failed','reason':str(exc),'target_dialogue_supplied':False}
    media=[{'path':receipt['path'],'label':'CURRENT complete actual video'},
        {'path':receipt['audio_path'],'label':'CURRENT native WAV, the SAME soundtrack, not another event'}]
    av=agent.artifact(name+'/audiovisual',lambda:agent.advisory(name+'/audiovisual',
        '''Watch the full current video and listen to its same native WAV. Independently
transcribe actual speech before comparing with the target, which is an expectation only.
Do not fill unspoken words from the script or forced alignment. Distinguish clear speech
from complete speech, natural pace from rushing, and actual final syllable from a cut tail.
ASR text differences alone are not audible defects. Only report actually observable issues.
Check mouth, visible/offscreen source, sound-event alignment and unexpected music.
No lexical speech: completeness/naturalness/ending/lip_sync may be not_applicable;
nonverbal voice still needs source binding and actual vocal/action alignment.
Offscreen scripted speech: lip_sync may be not_applicable with an actual observation
that no speaking mouth is visible; speech completeness/ending/source binding still apply.
Return {status:"observed|unknown",observed_events:[],heard_text,checks:{
speech_completeness:{status:"clear|defect|unknown|not_applicable",observation,time_s},
dialogue_naturalness:{status,observation,time_s},sentence_ending:{status,observation,time_s},
source_binding:{status,observation,time_s},lip_sync:{status,observation,time_s},
sound_event_alignment:{status,observation,time_s},audio_quality:{status,observation,time_s},
unexpected_music:{status,observation,time_s}},findings:[{category,severity,observation,time_s}]}.
No mandatory ASR discrepancy enum. Missing observations remain unknown; do not invent PASS.''' + (dialogue_integrity.REVIEW_GUIDE if strict else ''),
        {'target_only':source,'actual_media_facts':actual,'blind_asr_without_target':blind,
         'sound_policy':agent.config.get('editing_policy',{}),
         **({'required_spoken_language':__import__('lite.task_parameters',fromlist=['language']).language(agent.config),
             'speech_policy':'scripted_only' if source.get('dialogue') else 'no_lexical_speech',
             'literal_text_comparison':dialogue_integrity.text_evidence(source,blind,{}),
             'visual_instructions_are_never_spoken':True} if strict else {})},provider='gemini',media=media,max_tokens=4500))
    panel={'visual':normalize('visual',v,VISUAL),'audiovisual':normalize('audiovisual',av,AV+dialogue_integrity.CHECKS if strict else AV)}
    findings=panel['visual']['findings']+panel['audiovisual']['findings']
    av=_dict(av)
    # N/A cannot hide expected speech; this is a retained quality issue, not an exit.
    if source.get('dialogue'):
        if not isinstance(av.get('heard_text'),str) or not av['heard_text'].strip():
            findings.append({'id':'speech:missing_actual_transcript','category':'speech_completeness','severity':'uncertain','state':'open','kind':'review_uncertain','observation':'Target words are present but independently heard speech was not provided.'})
        if not isinstance(blind,dict) or not blind.get('audio_sha256') or blind.get('target_dialogue_supplied') is not False:
            findings.append({'id':'speech:blind_asr_missing','category':'speech_completeness','severity':'uncertain','state':'open','kind':'review_uncertain','observation':'Independent target-free CPU transcript is unavailable; no PASS inferred.'})
        for category in ('speech_completeness','dialogue_naturalness','sentence_ending','source_binding','lip_sync',*dialogue_integrity.CHECKS):
            check=_dict(panel['audiovisual']['checks'].get(category))
            if check.get('status')=='not_applicable' and not (category=='lip_sync' and _offscreen_lip_na(source,check)):
                findings.append({'id':'speech:unsupported_na:'+category,'category':category,'role':'audiovisual',
                    'severity':'uncertain','state':'open','kind':'review_uncertain',
                    'observation':'Scripted speech cannot be exempted without actual speech evidence.'})
    voice=None;voice_authority=None
    if source.get('dialogue'):
        cast=agent.config['generation_contract']['cast'];catalog={r['id']:r for r in agent.catalog()}
        voices=[r for role,spec in cast.items() if (r:=catalog.get(spec.get('voice_asset_id')))
            and Path(r.get('path','')).is_file() and r.get('speaker')==role and r.get('track_type') in ('synthetic_voice','isolated_voice') and sha(r['path'])==r['sha256']]
        voices.sort(key=lambda r:r['sha256']);labels={r['id']:f'REF_{i+1}' for i,r in enumerate(voices)}
        voice=agent.artifact(name+'/voice_identity',lambda:agent.advisory(name+'/voice_identity',
            '''Audio-only blind contrastive voice observation. CURRENT is actual audio;
neutral REF samples carry no role names, target text or video. For each actual spoken
utterance identify the matching timbre REF label or none/unknown/mixed. Compare texture,
resonance, pitch behavior and articulation, never infer identity from age/gender or words.
Return {status:"observed|unknown",utterances:[{match,status:"clear|defect|unknown",observation,
contrastive_observation,start_s,end_s}]}. Unknown is valid, not a reason to invent a match.''',
            {'reference_labels':list(labels.values()),'actual_seconds':receipt['seconds']},provider='gemini',
            media=[{'path':receipt['audio_path'],'label':'CURRENT actual audio'}]+
                  [{'path':r['path'],'label':labels[r['id']]+' neutral isolated reference'} for r in voices],max_tokens=3000))
        voice=_dict(voice)
        voice_authority=voice_reference_authority(source,cast,voices,labels,receipt['audio_sha256'],
            _dict(candidate.get('binding_contract')).get('speaking_bindings',[]))
        findings.extend(voice_identity_findings(voice,voice_authority))
    refs={f['id']:f for f in findings}
    for role,p in panel.items():
        refs.update({f'{role}:observation:{i+1}':{'role':role,'observation':ob,'evidence_status':p['status']}
            for i,ob in enumerate(p['observed_events']) if isinstance(ob,str) and ob.strip()})
        for category,item in p['checks'].items():
            if _grounded(item) and item.get('status') in ('clear','defect','not_applicable'):
                refs.setdefault(f'{role}:check:{category}',{**deepcopy(item),'role':role,'category':category,
                    'evidence_status':p['status']})
    decision=agent.artifact(name+'/director',lambda:agent.advisory(name+'/director',
        '''Use observer responsibilities: visual continuity/identity from visual,
mouth/audio/speech from audiovisual, voice identity from the blind contrastive report.
The host_voice_reference_authority table is the only REF-to-asset/SHA mapping.
REF numbers are sorted by submitted audio SHA, NOT role names or voice_r1/r2/r3 IDs.
A same-person render variant does not rename the canonical neutral REF samples.
Never reinterpret that mapping, clear a retained blind mismatch by a naming guess,
or make a new voice-identity judgment. Describe non-voice narrative evidence separately.
Return {reason,revision_plan:{caption_changes,reference_changes,duration_changes},
additional_findings:[{category,severity:"major|minor|uncertain",observation,time_s,evidence_refs:[]}],
narrative_observation:{status:"observed|unknown",observed_progress,remaining_target,evidence_refs:[]}}.
Assess target actions/causality using actual cited observations; source/caption is not
proof an action occurred. No accept/revise vote, no finding disposition protocol,
no investigations or asset changes. Never invent actual duration, frames or observations.
Optional narrative_observation: describe only actual observed progress with observer
evidence IDs. remaining_target is an unmet expectation, never proof of occurrence.
Report conflicts as uncertainty. Recommendations do not rewrite original evidence.''',
        {'source_target_only':source,'opinions':panel,'voice_identity':voice,'findings':findings,
         'evidence_catalog':refs,'actual_media_facts':actual,
         'voice_reference_authority':voice_authority},max_tokens=3500))
    decision,decision_audit=evidence_policy.checked_decision(_dict(decision),actual)
    decision,voice_audit=guard_voice_decision(decision,voice_authority)
    decision_audit.extend(voice_audit)
    for i,f in enumerate(_rows(decision.get('additional_findings'))):
        if not isinstance(f,dict) or not f.get('observation'):
            continue
        citations=_rows(f.get('evidence_refs'));grounded=bool(citations) and all(x in refs for x in citations if isinstance(x,str)) and all(isinstance(x,str) for x in citations)
        findings.append({**deepcopy(f),'id':f'director:additional:{i+1}','state':'open',
            'kind':'observed_content_defect' if grounded else 'review_uncertain',
            'severity':f.get('severity','uncertain') if grounded else 'uncertain'})
    if decision.get('advisory_unavailable') or not decision.get('reason'):
        findings.append({'id':'director:unavailable','severity':'uncertain','state':'open','kind':'review_uncertain',
            'observation':'Known director report failure retained; no content PASS inferred.'})
    narrative=_dict(decision.get('narrative_observation'));citations=_rows(narrative.get('evidence_refs'))
    progress=narrative.get('observed_progress')
    grounded=(narrative.get('status')=='observed' and isinstance(progress,str) and bool(progress.strip())
        and bool(citations) and all(isinstance(ref,str) and ref in refs
            and refs[ref].get('kind')!='review_uncertain'
            and refs[ref].get('evidence_status')!='unknown' for ref in citations))
    narrative_view={'status':'observed' if grounded else 'unknown',
        'observed_progress':progress[:2000] if grounded else None,
        'remaining_target':str(narrative.get('remaining_target') or '')[:2000],
        'remaining_target_is_expectation_only':True,
        'evidence_refs':citations if grounded else [],'new_model_calls':0}
    value={'completion_review_version':VERSION,'review_system':'feedback_v1','observation_protocol':OBSERVATION_PROTOCOL,
        'opinions':panel,'findings':findings,'issue_ledger':findings,'scores':{},
        'observed_events':panel['visual']['observed_events']+panel['audiovisual']['observed_events'],
        'decision_audit':decision_audit,'voice_identity':voice,'voice_reference_authority':voice_authority,
        'blind_asr':blind,'actual_media_facts':actual,
        'feedback_decision':decision,'repair_suggestions':decision.get('revision_plan',{}),
        'evidence_catalog':refs,'media_sha256':receipt['sha256'],'audio_sha256':receipt['audio_sha256'],
        'narrative_observation':narrative_view,
        'no_cross_role_votes_or_report_reconciliation':True,'content_acceptance_not_progress_gate':True}
    write(agent.s.run/'review_media'/cid/str(take)/'completion_review.json',value)
    return value
