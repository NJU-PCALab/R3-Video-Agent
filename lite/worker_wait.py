"""Bounded local-worker waits; no new dispatcher or source-repair controller."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import time

from .storage import Pending, read, write


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        tail = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if tail[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': tail[19],
                'argv': (path / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except (OSError, ValueError, IndexError):
        return None


def same_process(actual, expected):
    return bool(actual and expected.get('start_ticks') and actual == {
        k: expected.get(k) for k in ('pid', 'start_ticks', 'argv')})


def reconcile_launch(folder):
    """Recover a lost local launch acknowledgement by exact saved argv only."""
    submitted = read(folder / 'submitted.json', {})
    argv = submitted.get('command')
    if not isinstance(argv, list) or not argv:
        raise Pending('Original worker launch lacks its exact command: ' + str(folder))
    matches = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            identity = process_identity(int(path.name))
            if identity and identity['argv'] == argv:
                matches.append(identity)
    if len(matches) != 1:
        raise Pending('Original local launch has zero or multiple matching workers; do not redispatch: ' + str(folder))
    write(folder / 'process.json', matches[0])
    write(folder / 'launch_reconciled.json', {'process': matches[0], 'time': time.time(), 'redispatched': False})
    return matches[0]


def progress(folder, process):
    files = {}
    for name in ('worker.log', 'worker_metadata.json', 'native.mp4', 'native.tmp.mp4', 'generated.png'):
        path = folder / name
        if path.is_file():
            stat = path.stat()
            files[name] = [stat.st_size, stat.st_mtime_ns]
    ticks = None
    try:
        fields = (Path('/proc') / str(process['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        ticks = None
    return {'files': files, 'cpu_ticks': ticks}


def positive(value, default):
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else default


def observe_wait(folder, process, status, config, *, kind, now=None, activity=None):
    now = time.time() if now is None else now
    policy = config.get('worker_wait_policy', {})
    policy = policy if isinstance(policy, dict) else {}
    queue_limit = positive(policy.get('queue_timeout_seconds'), 21600)
    quiet_limit = positive(policy.get('progress_timeout_seconds'), 1800)
    runtime_limit = positive(policy.get(kind + '_timeout_seconds'), 7200 if kind == 'video' else 3600)
    state = str(status.get('status') or status.get('state') or '').lower()
    phase = 'queued' if state in ('queued', 'waiting_gpu_lock', 'waiting_for_gpu') else 'running'
    previous = read(folder / 'wait_state.json', {})
    if previous.get('process') != process:
        previous = {}
    activity = progress(folder, process) if activity is None else activity
    changed = previous.get('phase') != phase
    active = changed or previous.get('activity') != activity
    value = {'process': process, 'phase': phase, 'observed_at': now, 'activity': activity,
             'last_progress_at': now if active else previous.get('last_progress_at', now),
             'queued_since': (previous.get('queued_since') or now) if phase == 'queued' else None,
             'running_since': (previous.get('running_since') or now) if phase != 'queued' else None}
    decision = 'wait'
    if phase == 'queued':
        if now - value['queued_since'] >= queue_limit:
            decision = 'queue_timeout'
    elif now - value['running_since'] >= runtime_limit:
        decision = 'runtime_timeout'
    elif now - value['last_progress_at'] >= quiet_limit:
        decision = 'progress_timeout'
    if active or now - previous.get('observed_at', 0) >= 30 or decision != 'wait':
        write(folder / 'wait_state.json', value)
    return {**value, 'decision': decision, 'phase_changed': changed,
            'queue_limit_seconds': queue_limit, 'quiet_limit_seconds': quiet_limit,
            'runtime_limit_seconds': runtime_limit}


def group_members(group):
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == group:
                identity = process_identity(int(path.name))
                if identity:
                    result[identity['pid']] = identity
        except (OSError, ValueError, IndexError):
            continue
    return result


def stop_owned_worker(folder, expected, *, ready, grace=15, check_stop=lambda: None):
    """A new candidate is allowed only after the original worker is gone."""
    if ready():
        return {'outcome': 'terminal_receipt', 'signal_sent': False}
    actual = process_identity(expected['pid'])
    if not actual:
        return {'outcome': 'stopped' if not group_members(expected['pid']) else 'unconfirmed', 'signal_sent': False}
    if not same_process(actual, expected):
        return {'outcome': 'identity_changed', 'signal_sent': False}
    try:
        if os.getpgid(expected['pid']) != expected['pid']:
            return {'outcome': 'not_private_process_group', 'signal_sent': False}
    except ProcessLookupError:
        return {'outcome': 'stopped' if not group_members(expected['pid']) else 'unconfirmed', 'signal_sent': False}
    owned = group_members(expected['pid'])
    write(folder / 'cancel_request.json', {'process': expected, 'owned_members': list(owned.values()), 'time': time.time()})
    # Recheck identity immediately before signalling this worker's private group.
    if not same_process(process_identity(expected['pid']), expected):
        return {'outcome': 'identity_changed', 'signal_sent': False}
    try:
        os.killpg(expected['pid'], signal.SIGTERM)
    except ProcessLookupError:
        return {'outcome': 'stopped' if not group_members(expected['pid']) else 'unconfirmed', 'signal_sent': False}
    except PermissionError:
        return {'outcome': 'permission_denied', 'signal_sent': False}
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        check_stop()
        if ready():
            return {'outcome': 'terminal_receipt', 'signal_sent': True}
        if not group_members(expected['pid']):
            return {'outcome': 'stopped', 'signal_sent': True}
        time.sleep(.25)
    members = group_members(expected['pid'])
    if any(not same_process(identity, owned.get(pid, {})) for pid, identity in members.items()):
        return {'outcome': 'unconfirmed_group_member', 'signal_sent': True}
    for pid, identity in members.items():
        if same_process(process_identity(pid), identity):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError:
                return {'outcome': 'permission_denied', 'signal_sent': True}
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        check_stop()
        if ready():
            return {'outcome': 'terminal_receipt', 'signal_sent': True}
        if not group_members(expected['pid']):
            return {'outcome': 'stopped', 'signal_sent': True}
        time.sleep(.1)
    return {'outcome': 'unconfirmed', 'signal_sent': True}
