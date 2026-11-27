"""Independent blind ASR, complete-sentence checks and contrastive voice identity."""
from copy import deepcopy
import math

from .evidence_policy import normal, scripted
from . import speech_structure
from .storage import Pending, Unavailable, key, sha, write

VERSION = 'grounded_speech_release_v1'
CHECKS = ('speech_completeness', 'dialogue_naturalness', 'sentence_ending', 'source_binding')


def enabled(config):
    return config.get('speech_verification', {}).get('version') == VERSION


def export_duration(target, native_seconds, observation, *, maximum=10):
    """Do not cut detected speech merely to match a planned duration."""
    ends = [x['end'] for x in observation.get('vad', []) if isinstance(x, dict) and isinstance(x.get('end'), (int, float))]
    for word in observation.get('words', []):
        pair = word.get('timestamp') if isinstance(word, dict) else None
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and isinstance(pair[1], (int, float)):
            ends.append(pair[1])
    speech_end = max(ends, default=0)
    wanted = max(target, speech_end + .5)
    frames = min(math.floor(min(native_seconds, maximum) * 24 + 1e-7), math.ceil(wanted * 24 - 1e-7))
    if frames < round(target * 24):
        raise Unavailable('Native output is shorter than the requested complete shot')
    return frames / 24


def critical_issues(contract):
    if not isinstance(contract, dict) or contract.get('version') != VERSION:
        return []
    issues = []
    def add(category, observation, uncertain=False):
        issues.append({'id': 'speech_contract:' + category, 'kind': 'speech_release_contract',
                       'category': category, 'state': 'open', 'severity': 'uncertain' if uncertain else 'major',
                       'observation': observation, 'content_release_blocking': True})
    report = contract.get('content_report', {})
    checks = report.get('checks') if isinstance(report.get('checks'), dict) else {}
    if report.get('status') != 'observed':
        add('observation_missing', 'Actual current-video/WAV speech verification missing', True)
    for category in CHECKS:
        c = checks.get(category, {})
        if c.get('status') != 'clear' or not c.get('observation'):
            add(category, c or 'Observation missing', c.get('status') != 'defect')
    if not contract.get('blind_asr', {}).get('audio_sha256') or contract.get('blind_asr', {}).get('target_dialogue_supplied') is not False:
        add('blind_transcript_missing', 'Target-free independent transcription is required', True)
    if contract.get('asr_text_disagrees'):
        resolution = report.get('asr_discrepancy', {})
        if resolution.get('classification') not in ('homophone_transcription', 'asr_error') or not resolution.get('actual_heard_text') or not resolution.get('observation'):
            add('transcript_disagreement', {'blind_asr': contract.get('blind_asr', {}).get('text'), 'resolution': resolution}, True)
    voice = contract.get('voice_report', {})
    required = contract.get('required_voice_matches', [])
    observed = voice.get('utterances', []) if isinstance(voice.get('utterances'), list) else []
    if len(required)==1 and observed:
        required = required * len(observed)
    if voice.get('status') != 'observed' or len(observed) != len(required):
        add('voice_identity_missing', 'Independent audio-only contrastive voice observation is incomplete', True)
    else:
        for index, (actual, expected) in enumerate(zip(observed, required)):
            if (actual.get('status') != 'clear' or actual.get('match') != expected['reference_label']
                    or not actual.get('observation') or not actual.get('contrastive_observation')):
                known_labels={x['reference_label'] for x in contract.get('voice_references',[])}
                known_labels.update(x['reference_label'] for x in required)
                confirmed_wrong=(actual.get('status')=='clear' and bool(actual.get('observation'))
                    and bool(actual.get('contrastive_observation')) and actual.get('match') in known_labels|{'none','mixed'}
                    and actual.get('match')!=expected['reference_label'])
                add('voice_identity_' + str(index), {'actual': actual, 'expected_reference': expected['reference_label']},
                    not confirmed_wrong and actual.get('status') != 'defect')
    return issues


def apply_quality(review, value):
    contract = review.get('speech_contract')
    if not contract:
        return value
    issues = critical_issues(contract)
    value = deepcopy(value)
    value['speech_release_gate'] = {'version': VERSION, 'passed': not issues,
                                   'media_sha256': contract.get('media_sha256'), 'audio_sha256': contract.get('audio_sha256')}
    value['critical_release_issues'] = issues
    value['content_release_eligible'] = bool(value.get('ok')) and not issues
    if issues:
        value['ok'] = False; value['quality_status'] = 'draft/needs_review'
        value['unresolved_findings'] += issues
        if value.get('feedback_decision') == 'accept':
            value['decision_consistent'] = False
    return value


def verify(agent, cid, candidate):
    if not enabled(agent.config) or not scripted(candidate['source_clip']):
        return None
    receipt = candidate['receipt']; source = candidate['source_clip']; name = f'speech_contract/{cid}/{candidate["take"]}/{VERSION}'
    if sha(receipt['path']) != receipt['sha256'] or sha(receipt['audio_path']) != receipt['audio_sha256']:
        raise Pending('Speech verification source differs from original media')
    blind = agent.media.transcribe(receipt['audio_path'], f'{cid}_{candidate["take"]}_export')
    cast = agent.config['generation_contract']['cast']
    catalog = {r['id']: r for r in agent.catalog()}
    voices = []
    current_speakers = {line['speaker'] for line in scripted(source)}
    for role, spec in cast.items():
        # An explicitly non-speaking cast entry has no timbre reference. It is
        # exempt only while absent from this clip's scripted speaking roles.
        if 'voice_asset_id' in spec and spec['voice_asset_id'] is None and role not in current_speakers:
            continue
        r = catalog.get(spec.get('voice_asset_id'))
        if not r or r.get('speaker') != role or r.get('track_type') not in ('synthetic_voice', 'isolated_voice') or sha(r['path']) != r['sha256']:
            raise Pending('Canonical isolated voice sample is unavailable or changed')
        voices.append(r)
    voices.sort(key=lambda r: r['sha256'])
    mapping = {r['id']: 'REF_' + str(i+1) for i, r in enumerate(voices)}
    current = [{'path': receipt['path'], 'label': 'CURRENT full actual video'},
               {'path': receipt['audio_path'], 'label': 'CURRENT full native WAV, the SAME soundtrack'}]
    expected = scripted(source)
    content = agent.artifact(name + '/content', lambda: agent.advisory(name + '/content',
        '''核验当前真实视频与同一WAV。剧本只是期望，不得把没听到的词补入转写。逐项区分听得清和说完整、自然语速和赶词、句尾完整和音量正常、画内/画外声源与实际谁在动嘴。不要从音色年龄猜身份。独立ASR和VAD是线索而非事实，强制对齐不证明词已说出。守/手等同音转写差异不算实际发音缺陷。返回 {status:"observed|unknown",heard_text,checks:{speech_completeness:{status:"clear|defect|unknown",observation,time_s},dialogue_naturalness:{status,observation,time_s},sentence_ending:{status,observation,time_s},source_binding:{status,observation,time_s}},asr_discrepancy:{classification:"none|homophone_transcription|asr_error|actual_defect|unknown",actual_heard_text,observation}}。实际漏末词、非剧本重复、明显赶词、尾音被切、错误可见说话人均是defect；没有实际证据写unknown。''',
        {'target_only_not_observation': source, 'binding_table': candidate['binding_contract']['speaking_bindings'],
         'actual_seconds': receipt['seconds'], 'blind_asr_without_target': blind,
         'instruction': 'Listen independently. Assess the full current actual audiovisual interval.'},
        provider='gemini', media=current, max_tokens=5000))
    voice = agent.artifact(name + '/voice_identity', lambda: agent.advisory(name + '/voice_identity',
        '''Audio-only blind contrastive voice check. Listen to CURRENT and neutral reference samples. No video, role names, expected speaker, or dialogue target is supplied. For every actual spoken utterance in CURRENT, identify which reference has the same speaker timbre, or none/unknown/mixed. Compare voice texture, resonance, pitch behavior and articulation; do not infer identity from age/gender or matching words. A short/noisy sample may be unknown. A generic 'young voice fits' is insufficient. Return {status:"observed|unknown",utterances:[{start_s,end_s,match:"REF_1|REF_2|none|unknown|mixed",status:"clear|defect|unknown",observation,contrastive_observation}]} in temporal order. If one uninterrupted sentence has pauses, keep it as one utterance; explicit extra repeated sentences must remain separate.''',
        {'reference_labels': list(mapping.values()), 'current_seconds': receipt['seconds']}, provider='gemini',
        media=[{'path': receipt['audio_path'], 'label': 'CURRENT actual soundtrack'}] +
              [{'path': r['path'], 'label': mapping[r['id']] + ' isolated timbre reference'} for r in voices], max_tokens=4000))
    contract = {'version': VERSION, 'media_sha256': receipt['sha256'], 'audio_sha256': receipt['audio_sha256'],
                'blind_asr': blind, 'content_report': content, 'voice_report': voice,
                'asr_text_disagrees': normal(blind.get('text')) != normal(''.join(x['text'] for x in expected)),
                'voice_references': [{'reference_label': mapping[r['id']], 'asset_id': r['id'], 'speaker': r['speaker'], 'sha256': r['sha256']} for r in voices],
                'required_voice_matches': [{'script_index': i, 'reference_label': mapping[cast[x['speaker']]['voice_asset_id']]} for i, x in enumerate(expected)],
                'no_expected_speaker_or_video_supplied_to_voice_check': True}
    issues=critical_issues(contract)
    if issues and all(x['severity']=='uncertain' for x in issues):
        voice_uncertain=any(x['category'].startswith('voice_identity') for x in issues)
        field='voice_report' if voice_uncertain else 'content_report'
        media=([{'path':receipt['audio_path'],'label':'CURRENT actual soundtrack'}]+[{'path':r['path'],'label':mapping[r['id']]+' isolated voice'} for r in voices]) if voice_uncertain else current
        prompt=('Recheck this SAME audio-only blind contrastive voice task. Return status and utterances with start_s,end_s,match,status,observation,contrastive_observation. Use REF labels; no age/gender guesses. Unknown stays unknown.' if voice_uncertain else
                'Recheck this SAME full current video/WAV. Return status,heard_text,checks for speech_completeness/dialogue_naturalness/sentence_ending/source_binding each with status clear/defect/unknown, observation,time_s, and asr_discrepancy classification/actual_heard_text/observation. Target is not observed truth; do not fill missing words. Same-sound Chinese homophones are not audible defects.')
        payload={'previous_observation':contract[field],'reference_labels':list(mapping.values())} if voice_uncertain else {'target_only':source,'previous_observation':content,'independent_blind_asr':blind}
        clarification=agent.artifact(name+'/bounded_observation_clarification',lambda:agent.advisory(name+'/bounded_observation_clarification',prompt,payload,provider='gemini',media=media,max_tokens=5000))
        if field=='content_report':
            clarification=speech_structure.reconcile(agent,name+'/bounded_observation_clarification',prompt,payload,media,clarification,contract,cid,candidate['take'])
        contract['before_clarification']=deepcopy(contract[field]);contract[field]=clarification;contract['observation_clarification_rounds']=1
    write(agent.s.run / 'review_media' / cid / str(candidate['take']) / 'speech_contract.json', contract)
    remaining=critical_issues(contract)
    if remaining and all(x['severity']=='uncertain' for x in remaining):
        raise Pending('Bounded same-media speech observation remains unknown; do not regenerate for a report gap',status='needs_speech_observation_clarification')
    return contract
