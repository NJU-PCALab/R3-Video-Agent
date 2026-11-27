from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
import traceback

from . import audio_readiness, media_ports
from .agent import FilmAgent
from . import asset_lifecycle, finalization
from .media import Media, process_identity, sampling_size
from .providers import Models, gemini_credentials, gemini_endpoint
from .credentials import CredentialSourceError, resolve_openai_credentials
from .storage import Pending, Stopped, Store, Unavailable, lock, read, sha, write


def verify_source(config, digest, *, resuming=False):
    task_hash = (config.get('task_contract') or {}).get('source_sha256')
    if task_hash is not None and task_hash != digest:
        raise Pending('Story source differs from its authorized benchmark task hash', status='needs_input_reconciliation')
    if resuming and config.get('source_text_sha256') not in (None, digest):
        raise Pending('Saved story source differs from its authorized hash', status='needs_input_reconciliation')


def preflight(config):
    root = Path(config['production_root'])
    adapters = config.get('media_adapters', {})
    checks = {'known_media_adapters': not (set(adapters) - set(media_ports.KINDS.values()))}
    if asset_lifecycle.enabled(config):
        try:
            asset_lifecycle.admit_entry(config)
            checks['asset_worker_execution_contract']=asset_lifecycle.verify_runtime(config)
        except Pending:
            checks['asset_worker_execution_contract']=False
        if not checks['asset_worker_execution_contract']:
            # Do not import or execute a shared backend that failed verification.
            return {'ready': False, 'checks': checks, 'paid_calls': 0, 'gpu_dispatches': 0,
                    'credential_values_included': False}
    for kind in adapters:
        checks[kind + '_adapter'] = media_ports.ready(config, kind)
    for name, path in [('h3_worker', root / 'scripts/h3_diffsynth_worker.py'),
                       ('image_worker', root / 'scripts/avh_model_worker.py'),
                       ('image_python', config['image_provider']['python']),
                       ('image_weights', config['image_provider']['model_path']),
                       ('ffmpeg', Path(config['media_bin']) / 'ffmpeg'),
                       ('ffprobe', Path(config['media_bin']) / 'ffprobe'),
                       ('h3_config', config['h3_config'])]:
        if name.startswith('image_') and 'image' in adapters or name == 'h3_worker' and 'video' in adapters:
            continue
        checks[name] = Path(path).exists()
    for name, route in [('text', 'text_route'), ('doubao', 'doubao_route')]:
        if name == 'doubao' and config.get('review_system', {}).get('version') != 'feedback_v1':
            continue
        try:
            checks[name + '_credentials'] = bool(resolve_openai_credentials(config[route]).api_key)
        except CredentialSourceError:
            checks[name + '_credentials'] = False
    try:
        checks['gemini_credentials'] = bool(gemini_endpoint(config) and gemini_credentials(config))
    except CredentialSourceError:
        checks['gemini_credentials'] = False
    sys.path.insert(0, str(root))
    from lib.h3_av.config import load_config
    h3 = load_config(config['h3_config'])
    devices = {}
    for mode in ('ref2va', 't2va', 'fl2va'):
        cfg = h3.effective_config(mode)
        devices[mode] = str(cfg.cuda_device)
        if 'video_aspect_ratio' in config:
            try:
                sampling_size(cfg, config['video_aspect_ratio'])
                checks[mode + '_film_canvas'] = True
            except ValueError:
                checks[mode + '_film_canvas'] = False
        checks[mode + '_device_matches_worker'] = str(cfg.cuda_device) == str(config['image_provider'].get('device', '0'))
        if 'video' in adapters:
            continue
        checks[mode + '_pdd8_exact_fa3'] = cfg.inference_steps == 8 and cfg.acceleration_profile == 'pdd8_exact' and cfg.attention_backend == 'flash_attn_3'
        checks[mode + '_weights'] = Path(cfg.ref2va_weights if mode == 'ref2va' else cfg.fl2va_weights).is_dir()
        checks[mode + '_python'] = Path(cfg.diffsynth_python).is_file()
        checks[mode + '_adapter'] = Path(cfg.pdd_lora_path).is_file()
    audio = config.get('audio_provider')
    if audio:
        if 'audio' not in adapters:
            checks['audio_python'] = Path(audio.get('python', '')).is_file()
            if config.get('runtime_import_probe') in ('host_runtime_import_v1','host_runtime_compute_v2'):
                from .runtime_imports import audio_imports
                checks['audio_imports_on_this_host'] = audio_imports(config)
            for name in ('voice_design', 'voice_clone', 'music'):
                checks[name + '_weights'] = Path(audio.get('models', {}).get(name, '')).is_dir()
        ready = read(audio.get('readiness', ''), {})
        # Runtime availability is separate from a particular asset's quality.
        # Failed style/voice reviews remain FAIL and still enter asset revision.
        checks['actual_audio_model_probes'] = audio_readiness.matches(config, ready)
        checks['audio_lock_matches_device'] = Path(audio.get('lock_file', '')) == root / 'service_logs' / ('gpu_' + str(audio.get('device', '1')) + '.lock')
    if config.get('speech_verification', {}).get('version') == 'grounded_speech_release_v1':
        speech = config['speech_verification']
        checks['offline_asr_python'] = Path(speech.get('asr_python','')).is_file()
        checks['offline_asr_weights'] = (Path(speech.get('asr_model_path',''))/'model.safetensors').is_file()
        checks['canonical_cast_table'] = bool(config.get('generation_contract',{}).get('cast'))
    guide = Path(config['lite_root']) / 'resources/ref-en.txt'
    checks['prompt_guide'] = guide.is_file() and sha(guide) == '1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7'
    return {'ready': all(checks.values()), 'checks': checks, 'paid_calls': 0, 'gpu_dispatches': 0,
            'credential_values_included': False, 'h3_devices': devices,
            'image_device': str(config['image_provider'].get('device', '0')),
            'shared_gpu_lock': str(root / 'service_logs' / ('gpu_' + devices['ref2va'] + '.lock')),
            'audio_gpu_lock': audio.get('lock_file') if audio else None}


def main():
    parser = argparse.ArgumentParser(description='Lite long-film content evolution')
    parser.add_argument('command', choices=['start', 'resume', 'status', 'preflight', 'assets', 'memory'])
    parser.add_argument('--config')
    parser.add_argument('--run')
    parser.add_argument('--source')
    args = parser.parse_args()
    if args.command == 'assets':
        value = read(Path(args.run) / 'asset_registry.json', {})
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'memory':
        config = read(args.config, required=True)
        rows = [read(p, required=True) for p in sorted((Path(config['lite_root']) / 'memory/episodes').glob('*.json'))]
        print(json.dumps(rows[-12:], ensure_ascii=False, indent=2))
        return 0
    if args.command == 'status':
        value = read(Path(args.run) / 'state.json', {})
        print(json.dumps({k: value.get(k) for k in ('status', 'current', 'counts', 'delivery', 'updated_at')}, ensure_ascii=False))
        return 0
    if args.command == 'preflight':
        config = read(args.config, required=True)
        result = preflight(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['ready'] else 1
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with lock(run / 'runner.lock', nonblocking=True):
        retained_final = None
        if args.command == 'start':
            if any((run / name).exists() for name in ('state.json','source.txt','config.json','process.json','report.json')) or any(
                    (run / name).is_dir() and any((run / name).iterdir())
                    for name in ('calls','media','artifacts','launches','exits')):
                raise ValueError('Run already exists; use resume to retain requests, media and counters')
            config = read(args.config, required=True)
            source_bytes = Path(args.source).read_bytes()
            source_digest = sha256(source_bytes).hexdigest()
            verify_source(config, source_digest)
            asset_lifecycle.admit_entry(config,new_story=True)
            readiness = preflight(config)
            write(run / 'preflight.json', readiness)
            if not readiness['ready']:
                raise Unavailable('Required real model runtime unavailable; see preflight.json')
            (run / 'source.txt').write_bytes(source_bytes)
            config['source_text_sha256'] = source_digest
            write(run / 'config.json', config)
        else:
            state=read(run/'state.json', required=True)
            absent = object()
            report = read(run / 'report.json', absent)
            if state.get('status') == 'finished_stopped' or (
                state.get('status') in finalization.TERMINAL and report is absent
            ):
                raise Pending('Completed result is immutable; use an explicitly authorized derivative run',status='completed_run_immutable')
            config = read(run / 'config.json', required=True)
            if Path(config['lite_root']).resolve() != Path(__file__).resolve().parents[1]:
                raise ValueError('Resume the run from the workspace recorded in its configuration.')
            verify_source(config, sha(run / 'source.txt'), resuming=True)
            if report is not absent:
                retained_final = finalization.retained(run, config, state, report)
            else:
                asset_lifecycle.admit_entry(config,new_story=True)
                if asset_lifecycle.enabled(config) and not asset_lifecycle.verify_runtime(config):
                    raise Pending('Runtime source differs from its tested execution capability manifest',status='needs_configuration')
                audio = config.get('audio_provider')
                if audio and not audio_readiness.matches(config, read(audio['readiness'], {})):
                    raise Pending('Audio readiness does not match this run; probe its saved audio configuration before resuming', status='needs_configuration')
        store = Store(run, existing=args.command == 'resume')
        store.data.setdefault('framework_version', 'r3')
        if args.command == 'start':
            # Persist the empty ledger before creating an execution witness.
            store.save()
        identity = process_identity(os.getpid()) or {'pid': os.getpid(), 'argv': sys.argv}
        write(run / 'process.json', {**identity, 'started_at': time.time()})
        write(run / 'launches' / f'{time.time_ns()}.json', identity)
        write(Path(config['lite_root']) / 'current.json', {'run': str(run), 'process': identity})
        store.event('runner_started', process=identity, resume=args.command == 'resume')
        code = 0
        try:
            store.check_stop()
            if retained_final:
                delivery, status, previous = retained_final
                store.event('retained_final_verified', previous_process=previous, final_sha256=delivery['sha256'])
                store.data['delivery'] = delivery
                store.status(status, stage='film.finished', path=delivery['path'])
            else:
                agent = FilmAgent(store, Models(store, config), Media(store, config), config)
                agent.run()
        except Stopped as exc:
            store.status('stopped', reason=str(exc))
        except Pending as exc:
            code = 2
            store.status(exc.status, reason=str(exc), automatic_resubmission=False)
        except Unavailable as exc:
            code = 3
            if 'agent' in locals():
                agent.partial_result(exc)
            store.status('needs_attention', reason=str(exc), evidence_retained=True)
        except Exception as exc:
            code = 1
            trace = traceback.format_exc()
            failure = {'type': type(exc).__name__, 'message': str(exc), 'traceback': trace, 'time': time.time()}
            write(run / 'failures' / f'{time.time_ns()}.json', failure)
            store.status('host_error', reason=str(exc), evidence_retained=True)
            print(trace, file=sys.stderr, flush=True)
        finally:
            write(run / 'exits' / f'{time.time_ns()}.json', {'process': identity, 'returncode': code,
                'status': store.data['status'], 'finished_at': time.time()})
        return code


if __name__ == '__main__':
    raise SystemExit(main())
