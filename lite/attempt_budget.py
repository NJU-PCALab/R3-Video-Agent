"""An explicit user reset grants one new six-slot window; history stays intact."""
import hashlib
import json
import subprocess

from .storage import Pending, read, sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def start(store, cid):
    reset = store.data.get('attempt_budget_resets', {}).get(cid)
    if reset is None:
        return 1
    receipt = read(reset['authorization_receipt'], {})
    if not receipt:
        raise Pending('Attempt reset authorization receipt is missing')
    old = [c for c in store.data['candidates'][cid] if c['take'] <= 6]
    valid = (sha(reset['authorization_receipt']) == reset['authorization_sha256']
        and receipt.get('run') == str(store.run) and receipt.get('clip_id') == cid
        and receipt.get('authorization') == 'explicit_oom_reset_and_resume'
        and receipt.get('new_slots') == 6 and receipt.get('take_start') == 7
        and reset.get('take_start') == 7 and receipt.get('original_candidates_sha256') == digest(old)
        and len(old) == 6 and {c['take'] for c in old} == set(range(1, 7))
        and all('CUDA out of memory' in c.get('technical_failure', '')
                and not c.get('receipt') and not c.get('review') for c in old))
    if not valid:
        raise Pending('Attempt reset differs from its exact original OOM history or authorization')
    if any(c['take'] > 12 for c in store.data['candidates'][cid]):
        raise Pending('Authorized reset allows only slots 7 through 12')
    return 7


def limit(config):
    value = config.get('completion_policy', {}).get('max_video_takes', 6)
    if type(value) is not int or not 1 <= value <= 6:
        raise Pending('Video candidate limit must be an integer from 1 through 6', status='needs_configuration')
    return value


def slots(store, config, cid):
    first = start(store, cid)
    cap = limit(config)
    if cap < 6 and first != 1:
        raise Pending('Reduced lifetime budget cannot combine with a legacy attempt reset', status='needs_configuration')
    existing = {c['take'] for c in store.data.get('candidates', {}).get(cid, [])}
    # Reconcile original reserved media/review, including pre-change takes above cap.
    # Those slots are NOT permission to submit a new request.
    return sorted(set(range(first, first + cap)) | existing)


def capabilities(store, cid, take=None):
    from .closed_loop import capabilities as describe
    return describe(store.data.get('candidates', {}).get(cid, []), take, window_start=start(store, cid), max_takes=limit(read(store.run/'config.json', {})))


def admit_gpu(store, config, cid, take):
    cap = limit(config)
    if cap < 6 and (take > cap or start(store, cid) != 1):
        folder = store.run / 'media/videos' / f'{cid}_take_{take}'
        # Only the owned request/worker may be reconciled beyond the new cap.
        if not (folder/'submitted.json').exists():
            raise Pending('Reduced video budget forbids a new submission beyond its lifetime limit', status='video_budget_exhausted')
    if start(store, cid) == 1:
        return
    if (store.run / 'media/videos' / f'{cid}_take_{take}' / 'submitted.json').exists():
        # Existing worker identity/result must be reconciled even if GPU is busy.
        return
    # This recovery is solely GPU1. Do not launch a worker into another job's memory.
    if str(config.get('image_provider', {}).get('device')) != '1':
        raise Pending('Authorized OOM recovery is restricted to physical GPU1')
    try:
        free = int(subprocess.check_output(['nvidia-smi', '-i', '1',
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], timeout=15).decode().strip())
        compute = subprocess.check_output(['nvidia-smi', '-i', '1',
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], timeout=15).decode().strip()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise Pending('GPU1 recovery admission could not verify resources: ' + str(exc), status='awaiting_resource') from exc
    if free < 79000 or compute:
        raise Pending(f'GPU1 recovery waiting for free memory and no compute processes; free={free}MiB', status='awaiting_resource')
    store.event('gpu_recovery_admitted', clip_id=cid, physical_gpu=1, free_mib=free,
                authorization=store.data['attempt_budget_resets'][cid]['authorization_receipt'])
