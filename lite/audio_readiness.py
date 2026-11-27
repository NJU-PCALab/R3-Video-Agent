"""Identity of the audio stack exercised by a readiness probe (no credentials)."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re

from . import task_parameters
from .storage import sha

MODES = ('voice_design', 'voice_clone', 'music')


def _file(path):
    path = Path(path)
    return [str(path.resolve(strict=True)), sha(path)]


def _checkpoint(path):
    """Cheap identity for deployed weights; no multi-GB reads in preflight.

    Bind declared revisions separately. Also detect replacement at the same
    path via resolved file identity, size and change/modify timestamps.
    Ignore downloader bookkeeping and transient interpreter caches.
    """
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Expected an audio checkpoint directory')
    rows = []
    def fail(exc):
        raise exc
    for folder, dirs, names in os.walk(root, onerror=fail):
        dirs[:] = [n for n in dirs if not n.startswith('.') and n != '__pycache__']
        for name in names:
            if name.startswith('.') or not name.lower().endswith(('.safetensors', '.bin', '.pt', '.pth', '.model', '.json', '.yaml', '.yml', '.txt', '.tiktoken', '.py')):
                continue
            p = Path(folder) / name
            st = p.stat()
            rows.append([str(p.relative_to(root)), str(p.resolve()), st.st_dev, st.st_ino,
                         st.st_size, st.st_mtime_ns, st.st_ctime_ns])
    if not rows:
        raise ValueError('Expected model files in audio checkpoint directory')
    return [str(root), sorted(rows)]


def fingerprint(config):
    workspace = Path(config['lite_root'])
    audio = config['audio_provider']
    route = config.get('media_adapters', {}).get('audio')
    body = {'protocol': 'audio-probe-1',
            'driver': _file(workspace / 'tools/probe_audio_runtime.py'),
            'parameters': _file(workspace / 'lite/audio_parameters.py'),
            'transport': _file(workspace / 'lite/audio_assets.py'),
            'calibration': config.get('execution_contract'),
            'language': task_parameters.language(config),
            'device': audio.get('device'),
            'media_bin': str(Path(config['media_bin']).resolve())}
    if route is not None:
        body['adapter'] = [_file(route['executable']), route['model']]
    else:
        body['local'] = {
            'worker': _file(workspace / 'tools/audio_worker.py'),
            'interpreter': [str(Path(audio['python']).absolute()), _file(audio['python'])],
            'music_source': str(Path(audio['music_source']).resolve(strict=True)),
            'models': {k: [_checkpoint(audio['models'][k]),
                            _file(workspace / 'model_manifests' / (k + '.json'))] for k in MODES}}
    speech = config.get('speech_verification') or {}
    if speech.get('asr_model_path'):
        body['speech'] = {'worker': _file(workspace / 'tools/speech_worker.py'),
                          'interpreter': [str(Path(speech['asr_python']).absolute()), _file(speech['asr_python'])],
                          'model': _checkpoint(speech['asr_model_path']), 'version': speech.get('version')}
    return sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def matches(config, record):
    if not isinstance(record, dict) or record.get('runtime_ready') is not True:
        return False
    try:
        if record['audio_fingerprint'] != fingerprint(config):
            return False
        for mode in MODES:
            row = record['runtime_evidence'][mode]
            if not all(re.fullmatch(r'[0-9a-f]{64}', row[k]) for k in
                       ('waveform_sha256', 'original_sha256', 'worker_receipt_sha256')):
                return False
            seconds = row['seconds']
            if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 2:
                return False
        return True
    except (KeyError, TypeError, ValueError, OSError):
        return False
