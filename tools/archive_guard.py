"""Attach-only delivery guard for one declared existing run; never dispatches work.

Run with the pinned full release: python -B tools/archive_guard.py --config FILE.
The immutable config pins run/run_id, expected_runner, host, guard_dir,
archive_dir and result metadata. Optional memory_dir/provenance_files preserve
extra original evidence. This tool does not import an experiment or a launcher.
"""
from pathlib import Path, PurePosixPath
from fractions import Fraction
import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lite.storage import lock, operation_recorded, read, sha, write
from lite import duration_budget, task_parameters
from lite.worker_wait import process_identity, same_process, group_members


def check(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected):
    return same_process({k: actual.get(k) for k in ('pid', 'start_ticks', 'argv')}
                        if isinstance(actual, dict) else None, expected)


def validate(config):
    check(config.get('host') == socket.gethostname(), 'Guard host differs from explicit task authorization')
    check(isinstance(config.get('run_id'), str) and re.fullmatch(r'[A-Za-z0-9_\-]{1,160}', config['run_id']), 'Invalid run identity')
    process = config.get('expected_runner', {})
    check(type(process.get('pid')) is int and process['pid'] > 0 and str(process.get('start_ticks', '')).isdigit()
          and isinstance(process.get('argv'), list) and bool(process['argv']), 'Exact original runner identity is required')
    paths = {name: Path(config[name]).resolve() for name in ('run', 'guard_dir', 'archive_dir')}
    check(paths['run'].is_dir() and paths['run'].name == config['run_id'], 'Existing run path differs')
    check(paths['archive_dir'].name == config['run_id'], 'Archive must belong to the exact run identity')
    check(not any(paths['run'] == paths[k] or paths['run'] in paths[k].parents for k in ('guard_dir', 'archive_dir')),
          'Guard/archive output must be outside the immutable run')
    check(config.get('classification') in ('new_generated_result', 'human_assisted_recovery_result'), 'Result classification must be explicit')
    for name in ('index', 'story_id', 'title'):
        check(name in config, 'Missing result metadata: ' + name)
    return paths


def exact_exit(run, expected):
    rows = [row for path in (run / 'exits').glob('*.json')
            if (row := read(path, {})) and same(row.get('process'), expected)]
    check(len(rows) <= 1, 'Ambiguous exact original runner exits; success cannot be assumed')
    return rows[0] if rows else None


def probe(config, path):
    return json.loads(subprocess.check_output([
        str(Path(config['media_bin']) / 'ffprobe'), '-v', 'error', '-count_frames',
        '-show_streams', '-of', 'json', str(path)], timeout=180))['streams']


def review_replacement_evidence(run, folder, state=None):
    """A retained unknown original can accompany one verified replacement answer."""
    from types import SimpleNamespace
    from lite.explicit_recovery import replacement_claim
    from lite.providers import invalid_json_constant, model_answer, visible
    from lite.storage import Pending, Unavailable, key

    run, folder = Path(run).resolve(), Path(folder).resolve()
    call = folder.parent
    if folder.name != 'physical_1' or call.parent != run / 'calls':
        return None
    cfg = read(run / 'config.json', required=True)
    if not cfg.get('unknown_request_recovery'):
        return None
    try:
        claim = replacement_claim(SimpleNamespace(run=run), cfg, call)
        if not claim:
            return None
        replacement = call / 'physical_2'
        if {path.name for path in call.glob('physical_*')} != {'physical_1', 'physical_2'}:
            return None
        if any((folder / name).exists() for name in ('response.body', 'response_metadata.json', 'response.json')):
            return None
        failure = read(folder / 'failure.json', required=True)
        if any(key not in failure or failure[key] is not None for key in ('usage', 'cost')):
            return None
        request = read(call / 'request.json', required=True)
        provider, name = request['provider'], request['name']
        if provider not in ('doubao', 'glm', 'gemini') or key(name) != call.name:
            return None
        for attempt in (folder, replacement):
            submitted = read(attempt / 'submitted.json', required=True)
            if submitted.get('provider') != provider or submitted.get('name') != name:
                return None
        if (replacement / 'failure.json').exists():
            return None
        raw_body = (replacement / 'response.body').read_bytes()
        raw_sha = hashlib.sha256(raw_body).hexdigest()
        meta = read(replacement / 'response_metadata.json', required=True)
        status = meta.get('http_status')
        if (meta.get('raw_body_sha256') != raw_sha or meta.get('request_sha256') != claim['request_sha256']
                or meta.get('byte_count') != len(raw_body) or type(status) is not int or not 200 <= status < 300):
            return None
        raw = visible(json.loads(raw_body, parse_constant=invalid_json_constant))
        if read(replacement / 'response.json', required=True) != raw:
            return None
        text, usage = model_answer(provider, raw)
        value = text.strip()
        fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', value, flags=re.DOTALL)
        if not isinstance(json.loads(fence.group(1) if fence else value, parse_constant=invalid_json_constant), dict):
            return None
        if read(call / 'answer.json', required=True) != {'text': text, 'usage': usage}:
            return None
        state = read(run / 'state.json', required=True) if state is None else state
        accounted = state.get('accounted_operations', [])
        uses = state.get('usage', {})
        if (str(folder) not in accounted or str(replacement) not in accounted
                or uses.get(str(call)) != usage or uses.get(str(folder)) is not None):
            return None
        return {'call_id': call.name, 'request_sha256': claim['request_sha256'],
                'permission': claim['permission'], 'permission_sha256': claim['permission_sha256'],
                'original_http_status': failure['http_status'], 'original_execution_unknown': True,
                'original_usage': None, 'original_cost': None, 'replacement_attempt': 2,
                'replacement_raw_body_sha256': raw_sha, 'answer_sha256': sha(call / 'answer.json')}
    except (OSError, ValueError, TypeError, AttributeError, KeyError, Pending, Unavailable):
        return None


def verify_run(config):
    """Require exact exit and physical native source media, never a status label alone."""
    run = Path(config['run']); expected = config['expected_runner']
    process = read(run / 'process.json', required=True)
    check(same(process, expected), 'Current process record differs from the pinned original runner')
    check(not same(process_identity(expected['pid']), expected), 'Original runner is still active')
    terminal = exact_exit(run, expected)
    check(terminal is not None and terminal.get('returncode') == 0, 'Exact successful original runner exit is required')
    for path in (run / 'media').rglob('process.json'):
        worker = read(path, required=True)
        check(not same(process_identity(worker['pid']), worker) and not group_members(worker['pid']), 'Owned worker or its group is still active: ' + str(path))
    for path in (run / 'media').rglob('submitted.json'):
        check(read(path.parent / 'process.json') is not None, 'Original worker launch identity is unresolved: ' + str(path))
    replacements = []
    for folder in (run / 'calls').glob('*/physical_*'):
        failure = read(folder / 'failure.json', {})
        if (not operation_recorded(folder) or read(folder / 'response.json') is not None
                or failure.get('definitive') or failure.get('http_status') == 429):
            continue
        replacement = review_replacement_evidence(run, folder)
        check(replacement, 'Original model submission remains unresolved: ' + str(folder))
        replacements.append(replacement)
    cfg = read(run / 'config.json', required=True); state = read(run / 'state.json', required=True)
    report = read(run / 'report.json', required=True); selected = state['selected']; clips = state['plan']['clips']
    check(state['status'] in ('completed', 'completed_with_warnings') and len(selected) == len(clips) and bool(selected), 'Run is not technically complete')
    check([s['clip']['id'] for s in selected] == [c['id'] for c in clips], 'Selected sources are out of plan order')
    delivery = report['delivery']; final = Path(delivery['path'])
    check(sha(final) == delivery['sha256'], 'Final actual media SHA differs')
    streams = probe(cfg, final); video = [s for s in streams if s['codec_type'] == 'video']; audio = [s for s in streams if s['codec_type'] == 'audio']
    check(len(video) == len(audio) == 1 and Fraction(video[0]['avg_frame_rate']) == 24, 'Final stream count or fps differs')
    frames = int(video[0]['nb_read_frames'])
    if task_parameters.longform(cfg):
        duration_budget.final(cfg, selected)
    else:
        check(3600 <= frames <= 7200, 'Final frame/duration mismatch')
    check(frames == delivery['frames'] == sum(s['receipt']['frames'] for s in selected), 'Final frame/duration mismatch')
    check(len(delivery['source_clips']) == len(selected), 'Final ordered source count differs')
    ordered = []
    for position, selected_clip in enumerate(selected):
        item = selected_clip['receipt']; video = [s for s in probe(cfg, item['path']) if s['codec_type'] == 'video']; audio = [s for s in probe(cfg, item['audio_path']) if s['codec_type'] == 'audio']
        check(len(video) == len(audio) == 1 and sha(item['path']) == item['sha256'] and sha(item['audio_path']) == item['audio_sha256']
              and item.get('native_audio') is True and int(video[0]['nb_read_frames']) == item['frames']
              and Fraction(video[0]['avg_frame_rate']) == 24 and audio[0]['codec_name'].startswith('pcm_')
              and delivery['source_clips'][position]['sha256'] == item['sha256'], 'Ordered source video/native PCM verification failed')
        ordered.append({'clip_id': selected_clip['clip']['id'], 'take': selected_clip['take'], 'frames': item['frames'],
                        'video_sha256': item['sha256'], 'audio_sha256': item['audio_sha256']})
    return state, report, {'technical_delivery_passed': True, 'ordered_sources': ordered,
        'state_sha256': sha(run / 'state.json'), 'report_sha256': sha(run / 'report.json'),
        'actual_frames': frames, 'actual_seconds': frames / 24, 'content_acceptance_passed': report.get('content_acceptance_passed'),
        'independent_quality_improvement_proven': False, 'all_candidate_media_retained': True, 'currency_cost': None,
        **({'authorized_review_replacements': replacements} if replacements else {})}


def preserve(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    check(not target.is_symlink(), 'Frozen target is a symlink')
    if target.exists():
        check(sha(source) == sha(target), 'Existing frozen bytes differ; originals preserved: ' + str(target))
    else:
        temporary = target.with_name(target.name + '.' + str(os.getpid()) + '.tmp')
        shutil.copy2(source, temporary)
        check(sha(source) == sha(temporary), 'Original bytes changed during freeze')
        os.replace(temporary, target)


def freeze(config):
    """Receipt is the last atomic publication; interrupted staging can be reconciled."""
    state, report, verification = verify_run(config)
    run = Path(config['run']); folder = Path(config['archive_dir']); folder.mkdir(parents=True, exist_ok=True)
    package = folder / 'package'; receipt_path = folder / 'remote_archive_receipt.json'
    previous = read(receipt_path)
    if previous is not None:
        check(previous.get('run_id') == config['run_id'] and previous.get('final_sha256') == report['delivery']['sha256'], 'Existing archive receipt differs')
        check(sha(folder / 'delivery.tar.gz') == previous['archive_sha256'] and sha(folder / 'package_manifest.json') == previous['manifest_sha256'], 'Existing archive bytes differ')
        files = read(folder / 'package_manifest.json', required=True)
        check(all(sha(package / name) == digest for name, digest in files.items()), 'Existing frozen package differs')
        return previous
    for source in run.rglob('*'):
        if source.is_file() and not source.name.endswith('.tmp'):
            preserve(source, package / source.relative_to(run))
    preserve(Path(report['delivery']['path']), package / 'final.mp4')
    memory = Path(config['memory_dir']) if config.get('memory_dir') else None
    if memory:
        for source in memory.rglob('*'):
            if source.is_file(): preserve(source, package / 'production_memory' / source.relative_to(memory))
    for selected in state['selected']:
        cid = selected['clip']['id']; candidate = next(c for c in state['candidates'][cid] if c['take'] == selected['take'])
        destination = package / 'selected' / cid
        for name, value in (('source_clip.json', candidate['source_clip']), ('review.json', candidate['review']), ('references.json', candidate['references'])):
            path = destination / name; old = read(path)
            check(old is None or old == value, 'Frozen selected evidence differs')
            if old is None: write(path, value)
        caption = destination / 'caption.txt'
        check(not caption.exists() or caption.read_text() == candidate['caption'], 'Frozen caption differs')
        if not caption.exists(): caption.write_text(candidate['caption'])
        for ref in candidate['references']:
            check(sha(ref['path']) == ref['sha256'], 'Original reference SHA differs')
            preserve(Path(ref['path']), package / 'reference_assets' / (ref['sha256'] + Path(ref['path']).suffix))
    for name, source in config.get('provenance_files', {}).items():
        check(str(PurePosixPath(name)) == name and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts, 'Invalid provenance package path')
        preserve(Path(source), package / 'provenance' / name)
    for replacement in verification.get('authorized_review_replacements', []):
        target = package / 'provenance' / ('authorized_review_replacement_' + replacement['call_id'] + '.json')
        preserve(Path(replacement['permission']), target)
        check(sha(target) == replacement['permission_sha256'], 'Frozen replacement authorization differs')
    write(package / 'technical_verification.json', verification)
    manifest = {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()}
    write(folder / 'package_manifest.json', manifest)
    archive = folder / 'delivery.tar.gz'; temporary = folder / ('delivery.' + str(os.getpid()) + '.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as bundle:
        for name in manifest: bundle.add(package / name, arcname=name, recursive=False)
    with temporary.open('rb') as handle: os.fsync(handle.fileno())
    os.replace(temporary, archive)
    receipt = {k: config[k] for k in ('run_id', 'index', 'story_id', 'title', 'classification', 'host')}
    receipt.update(success=True, run=str(run), archive=str(archive), archive_sha256=sha(archive),
        manifest=str(folder / 'package_manifest.json'), manifest_sha256=sha(folder / 'package_manifest.json'),
        final_sha256=report['delivery']['sha256'], content_acceptance_passed=report.get('content_acceptance_passed'), time=time.time())
    if config.get('baseline_run_id'): receipt['baseline_run_id'] = config['baseline_run_id']
    write(receipt_path, receipt)
    return receipt


def observe_once(config, verified_receipt=None):
    """A missing live snapshot/exit/ACK is pending; malformed JSON stays explicit."""
    run = Path(config['run']); guard = Path(config['guard_dir']); expected = config['expected_runner']
    check(same(read(run / 'process.json', required=True), expected), 'Original run process changed; no automatic reattach')
    if any(Path(path).exists() for path in [run / 'STOP', guard / 'STOP', *config.get('stop_files', [])]):
        return {'status': 'stopped', 'automatic_resume': False}
    if same(process_identity(expected['pid']), expected):
        state = read(run / 'state.json', {})
        return {'status': 'waiting_original_runner', 'actual_status': state.get('status', 'snapshot_temporarily_unavailable'), 'runner_live_exact': True}
    terminal = exact_exit(run, expected)
    if terminal is None: return {'status': 'awaiting_exact_original_exit', 'no_automatic_redispatch': True}
    write(guard / 'runner_observed_exit.json', terminal)
    if terminal.get('returncode') != 0:
        return {'status': 'needs_attention', 'exact_runner_exit': terminal, 'automatic_resume': False, 'not_content_failure_by_default': True}
    # The process-local cache is populated only after this guard's complete
    # native verification. Long transfers do not decode every source every 10s.
    receipt = verified_receipt if verified_receipt is not None else freeze(config)
    ack = read(Path(config['archive_dir']) / 'local_delivery_ack.json')
    if ack is None: return {'status': 'awaiting_local_full_ack', 'archive': config['archive_dir']}
    check(ack.get('all_files_sha_verified') is True and ack.get('run_id') == config['run_id']
          and all(ack.get(k) == receipt[k] for k in ('archive_sha256', 'manifest_sha256', 'final_sha256')), 'Full local acknowledgement differs')
    return {'status': 'completed_archived', 'archive': config['archive_dir'], 'content_acceptance_passed': receipt.get('content_acceptance_passed'), 'classification': receipt['classification']}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = read(args.config, required=True); paths = validate(config)
    guard = paths['guard_dir']; guard.mkdir(parents=True, exist_ok=True)
    with lock(guard / 'archive_guard.lock', nonblocking=True):
        process = process_identity(os.getpid()); check(process is not None, 'Linux exact guard identity is unavailable')
        claim = guard / 'claim'; claim.mkdir()
        write(claim / 'config.json', config); write(guard / 'service_process.json', process); code = 0
        verified_receipt = None
        try:
            write(guard / 'handoff.json', {'process': process, 'original_runner': config['expected_runner'], 'runner_started': False, 'runner_resumed': False, 'config_sha256': sha(args.config)})
            while True:
                outcome = observe_once(config, verified_receipt); write(guard / 'outcome.json', outcome)
                if outcome['status'] == 'awaiting_local_full_ack' and verified_receipt is None:
                    verified_receipt = read(Path(config['archive_dir']) / 'remote_archive_receipt.json', required=True)
                write(guard / 'heartbeat.json', {'time': time.time(), 'guard_only': True, **outcome})
                if outcome['status'] in ('completed_archived', 'needs_attention', 'stopped'):
                    code = 0 if outcome['status'] == 'completed_archived' else 2; break
                time.sleep(10)
        except Exception as exc:
            code = 1; write(guard / 'service_error.json', {'error': str(exc), 'type': type(exc).__name__, 'traceback': traceback.format_exc(), 'no_automatic_redispatch': True}); raise
        finally:
            write(guard / 'service_exit.json', {'process': process, 'returncode': code, 'time': time.time()})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
