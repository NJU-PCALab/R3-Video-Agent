"""Verify actual local audio workers and retain acoustic/hash evidence for preflight."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lite import asset_lifecycle, audio_readiness, task_parameters
from lite.media import Media
from lite.storage import Store, lock, read, sha, write


PROBES = {
    'voice_design': {
        'kind': 'audio', 'role': 'voice', 'generator': 'voice_design',
        'speaker': 'Runtime probe speaker',
        'prompt': 'A fictional adult with a calm, clear speaking voice. No background noise or music.',
    },
    'voice_clone': {
        'kind': 'audio', 'role': 'voice', 'generator': 'voice_clone',
        'speaker': 'Runtime probe speaker',
    },
    'music': {
        'kind': 'audio', 'role': 'music', 'generator': 'music',
        'prompt': 'A brief instrumental piano phrase with no speech or singing.',
        'seconds': 8,
    },
}


def probe_specs(config):
    for mode, spec in PROBES.items():
        yield mode, ({**spec, 'language': task_parameters.language(config)}
                     if spec['role'] == 'voice' else spec)


def probe_run(output, identity):
    """Never move an unknown submission into a new request namespace."""
    run = output / 'runs' / identity
    prior = [output / 'run', *(output / 'runs').glob('*')]
    for other in prior:
        if other == run or not other.is_dir():
            continue
        accounted = read(other / 'state.json', {}, attempts=1).get('accounted_operations', {})
        for mode in PROBES:
            folder = other / 'media/audio_assets' / ('probe_' + mode)
            if not folder.exists() and str(folder) not in accounted:
                continue
            response = read(folder / 'worker_response.json', {}, attempts=1)
            # A local process exit does not establish a remote job's status.
            if response.get('needs_reconciliation') is True or response.get('ok') not in (True, False):
                raise RuntimeError('Reconcile the previous audio probe submission before changing the audio setup: ' + str(folder))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    config = read(parser.parse_args().config, required=True)
    asset_lifecycle.admit_entry(config, new_story=True)
    workspace = Path(config['lite_root']).resolve()
    if not (workspace / 'tools/audio_worker.py').is_file():
        parser.error('lite_root must contain the frozen harness checkout')
    output = workspace / 'verification/audio_runtime_probe'
    if Path(config['audio_provider']['readiness']).resolve() != output / 'readiness.json':
        parser.error('Point audio_provider.readiness at the probe workspace readiness.json')
    with lock(output / 'probe.lock', nonblocking=True):
        identity = audio_readiness.fingerprint(config)
        run = probe_run(output, identity)
        media = Media(Store(run), config)
        evidence = {}
        original_voice = None
        for mode, spec in probe_specs(config):
            reference = original_voice if mode == 'voice_clone' else None
            receipt = media.audio('probe_' + mode, spec, reference, seed=6701)
            waveform = Path(receipt['path'])
            original = Path(receipt['full_path'])
            worker = Path(receipt['provenance']['worker_response'])
            measured = media.audio_info(waveform)
            if (read(worker, required=True).get('ok') is not True
                    or sha(waveform) != receipt['sha256']
                    or sha(original) != receipt['full_sha256']
                    or measured['seconds'] < 2
                    or mode == 'voice_clone'
                    and receipt['provenance'].get('reference_sha256') != original_voice['sha256']):
                raise RuntimeError('Audio probe evidence differs for ' + mode)
            if mode == 'voice_design':
                original_voice = receipt
            evidence[mode] = {'waveform_sha256': sha(waveform), 'original_sha256': sha(original),
                              'worker_receipt_sha256': sha(worker), 'seconds': measured['seconds']}
        if identity != audio_readiness.fingerprint(config):
            raise RuntimeError('The audio setup changed during its probe; retain the original receipts')
        write(output / 'readiness.json', {'runtime_ready': True, 'audio_fingerprint': identity, 'time': time.time(),
                                          'runtime_evidence': evidence, 'audio_backend': 'adapter' if 'audio' in config.get('media_adapters', {}) else 'local'})
    print(output / 'readiness.json')


if __name__ == '__main__':
    main()
