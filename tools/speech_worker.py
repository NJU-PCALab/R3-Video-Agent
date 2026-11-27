"""Offline, CPU-only blind Whisper + Silero VAD. Never receives script or voices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from numbers import Integral


def vad_seconds(rows, sample_count, sample_rate):
    """Convert actual sample indices without Silero's one-decimal rounding."""
    result = []
    for row in rows:
        start, end = row['start'], row['end']
        if not all(isinstance(x, Integral) and not isinstance(x, bool) for x in (start, end)) or not 0 <= start < end <= sample_count:
            raise ValueError('VAD sample indices exceed the actual input waveform')
        result.append({'start': int(start) / sample_rate, 'end': int(end) / sample_rate})
    return result


def request_language(request):
    required={'audio_path','audio_sha256','model_path'}
    if not required <= set(request) or set(request)-required-{'language'}:
        raise ValueError('Blind ASR must not receive target text, prompts or reference audio')
    language=request.get('language','chinese')
    if language not in ('chinese','english','auto'):
        raise ValueError('Blind ASR task language must be Chinese or English')
    return language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    parser.add_argument('--vad-only', action='store_true')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    language=request_language(request)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import soundfile as sf
    from transformers import pipeline
    from silero_vad import load_silero_vad, get_speech_timestamps
    torch.set_num_threads(4)
    p = Path(request['audio_path'])
    if hashlib.sha256(p.read_bytes()).hexdigest() != request['audio_sha256']:
        raise ValueError('Blind ASR audio differs from original receipt')
    audio, rate = sf.read(p, dtype='float32')
    if rate != 16000 or audio.ndim != 1:
        raise ValueError('Blind ASR needs retained 16kHz mono WAV')
    begin = time.time()
    samples = get_speech_timestamps(torch.from_numpy(audio), load_silero_vad(), sampling_rate=rate, return_seconds=False)
    vad = vad_seconds(samples, len(audio), rate)
    if args.vad_only:
        result = {}
    else:
        transcribe = pipeline('automatic-speech-recognition', model=request['model_path'], device=-1,
                              torch_dtype=torch.float32, model_kwargs={'attn_implementation': 'eager'})
        result = transcribe({'raw': audio, 'sampling_rate': rate}, return_timestamps='word',
                            generate_kwargs={**({'language':language} if language!='auto' else {}), 'task': 'transcribe', 'do_sample': False, 'num_beams': 5})
    value = {'status': 'completed', 'audio_sha256': request['audio_sha256'], 'model_path': request['model_path'],
             'text': result.get('text'), 'words': result.get('chunks', []), 'vad': vad,
             'vad_unit': 'seconds_from_sample_indices_v1', 'sample_rate': rate, 'sample_count': len(audio),
             'observation_kind': 'vad_only' if args.vad_only else 'blind_asr_and_vad',
             'seconds': len(audio) / rate, 'wall_seconds': time.time() - begin, 'device': 'cpu',
             'target_dialogue_supplied': False, 'references_supplied': False,
             'timestamp_credit': 'model_estimates_not_proof_of_words_or_phonemes',
             'model_config_sha256': hashlib.sha256((Path(request['model_path'])/'config.json').read_bytes()).hexdigest()}
    output = Path(args.response); temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); os.replace(temp, output)


if __name__ == '__main__':
    main()
