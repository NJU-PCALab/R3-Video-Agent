"""Bounded CPU VAD precision recheck of a retained rounded observation."""
from pathlib import Path
import math
import wave
from .storage import Pending, read, sha, write

UNIT = 'seconds_from_sample_indices_v1'


def recheck(media, folder, request, original):
    rows = original.get('vad')
    duration = original.get('seconds')
    if original.get('vad_unit') == UNIT or not isinstance(rows, list) or not isinstance(duration, (int, float)):
        return original
    # Only the known legacy one-decimal endpoint overflow is a precision fault.
    overflow = [row.get('end') for row in rows if isinstance(row, dict) and isinstance(row.get('end'), (int, float)) and row['end'] > duration + .001]
    if not overflow or any(not math.isfinite(x) or abs(x - round(duration, 1)) > 1e-8 or x - duration > .050001 for x in overflow):
        return original
    if sha(request['audio_path']) != request['audio_sha256']:
        raise Pending('VAD precision input differs from retained audio', status='needs_voice_observation')
    with wave.open(request['audio_path']) as wav:
        samples, rate = wav.getnframes(), wav.getframerate()
        valid = rate == 16000 and wav.getnchannels() == 1 and abs(samples / rate - duration) <= 1 / rate
    if not valid:
        raise Pending('VAD precision duration differs from actual samples', status='needs_voice_observation')
    operation = folder / 'vad_sample_precision_v1'
    command = [media.config['speech_verification']['asr_python'], str(Path(media.config['lite_root']) / 'tools/speech_worker.py'),
               '--vad-only', '--request', str(operation / 'request.json'), '--response', str(operation / 'worker_response.json')]
    result = media._worker(operation, request, command, counter='cpu_vad_runs', amount=1, response_name='worker_response.json', device='')
    if result.get('audio_sha256') != request['audio_sha256'] or result.get('observation_kind') != 'vad_only' or result.get('vad_unit') != UNIT or result.get('sample_count') != samples or result.get('sample_rate') != rate:
        raise Pending('VAD precision reply lacks exact original sample binding', status='needs_voice_observation')
    corrected = result.get('vad')
    if not isinstance(corrected, list) or any(not isinstance(row, dict) or not all(isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool) and math.isfinite(row[k]) for k in ['start', 'end']) or not 0 <= row['start'] < row['end'] <= samples / rate for row in corrected):
        raise Pending('VAD precision reply still has invalid windows', status='needs_voice_observation')
    binding = {'original_response_sha256': sha(folder / 'worker_response.json'), 'input_audio_sha256': request['audio_sha256'],
               'recheck_response_sha256': sha(operation / 'worker_response.json'), 'record': str(operation),
               'original_vad': rows, 'effective_vad': corrected, 'asr_repeated': False, 'audio_regenerated': False}
    audit = folder / 'vad_sample_precision_audit.json'
    if audit.exists() and read(audit) != binding:
        raise Pending('Retained VAD precision binding changed', status='needs_voice_observation')
    if not audit.exists():
        write(audit, binding)
    return {**original, 'vad': corrected, 'vad_unit': UNIT, 'vad_precision_recheck': binding}
