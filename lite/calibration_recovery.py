"""Exact retained calibration reference; original performance review stays failed."""
from copy import deepcopy
from pathlib import Path
import wave

from .storage import Pending, Unavailable, read, sha
from .explicit_recovery import digest, permission

VERSION = 'retained_calibration_reference_v1'


def valid(actual):
    binding = actual.get('authorized_calibration_reference') or {}
    path = Path(binding.get('receipt', ''))
    if not path.is_file() or sha(path) != binding.get('sha256'):
        return False
    r = read(path)
    try:
        Path(actual['path']).resolve().relative_to(Path(r['run']).resolve())
        source = Path(actual['full_path'])
        observation = actual['provenance']['voice_activity']['observation']
        from .asset_lifecycle import voiced_seconds
        duration = voiced_seconds(observation, actual['full_sha256'], actual['seconds'])
        media_ok = sha(actual['path']) == actual['sha256'] and sha(source) == actual['full_sha256']
        with wave.open(str(source), 'rb') as pcm:
            media_ok = media_ok and pcm.getcomptype() == 'NONE' and pcm.getnchannels() == 1 \
                and abs(pcm.getnframes() / pcm.getframerate() - actual['seconds']) < .05
    except (OSError, KeyError, ValueError, EOFError, Pending, Unavailable, wave.Error):
        return False
    return bool(r.get('version') == VERSION and r.get('same_run_restart_authorized') is True
        and r.get('new_generation_allowed') is False and r.get('content_acceptance_passed') is False
        and r.get('asset_id') == binding.get('asset_id')
        and r.get('candidate_take') == binding.get('candidate_take')
        and r.get('asset_version') == actual.get('version')
        and r.get('media_sha256') == actual.get('sha256')
        and r.get('full_media_sha256') == actual.get('full_sha256')
        and r.get('review_sha256') == digest(actual.get('review'))
        and r.get('observation_sha256') == digest(observation)
        and r.get('retained_review_defect') == 'missing_requested_dramatic_voice_transition'
        and actual.get('generator') == 'voice_design'
        and actual.get('track_type') == 'synthetic_voice'
        and actual.get('kind') == 'audio' and 2 <= actual.get('seconds', 0) <= 15
        and duration >= 2 and media_ok
        and actual.get('feedback_accepted') is False
        and actual.get('review', {}).get('decision') == 'revise')


def adopt(agent, ident, row):
    r = permission(agent.config, 'retained_calibration_reference')
    if r is None or r.get('asset_id') != ident:
        return False
    agent.s.check_stop()
    from .asset_lifecycle import input_digest, usable_version
    record = row.get('lifecycle_attempts', {}).get(f"{r.get('asset_version')}/{r.get('candidate_take')}")
    original = (record or {}).get('actual')
    if (r.get('run') != str(agent.s.run.resolve()) or not original or record.get('failure')
            or r.get('input_sha256') != row.get('blocked_input_sha256')
            or r.get('input_sha256') != input_digest(row) or row.get('versions')
            or row.get('spec', {}).get('role') != 'voice'):
        raise Pending('Retained calibration reference differs from its original evidence', status='needs_asset_repair')
    actual = deepcopy(original)
    actual['authorized_calibration_reference'] = {
        **agent.config['retained_calibration_reference'], 'asset_id': ident,
        'candidate_take': r['candidate_take'], 'content_acceptance_passed': False}
    if not valid(actual):
        raise Pending('Retained calibration media/observation binding failed', status='needs_asset_repair')
    agent.registry.publish(ident, actual)
    row.update(dependency_state='ready_with_warnings', active_reference_version=actual['version'],
               generation=actual['version'], pending_version=None, pending_prompt=None)
    agent.registry.save()
    agent.s.warning('retained_calibration_reference',
        f'{ident}: original dramatic voice defect retained; existing calibration sample only; content acceptance remains false')
    agent.s.event('retained_calibration_reference_adopted', asset_id=ident, take=r['candidate_take'],
        media_sha256=actual['sha256'], new_generation=False, original_review_unchanged=True)
    return True


def generation_blocked(agent, ident, row):
    """Variant/clone IDs inherit the exact root's no-generation scope."""
    r = permission(agent.config, 'retained_calibration_reference')
    if r is None:
        return False
    root = r['asset_id']
    root_spec = agent.s.data['assets'].get(root, {}).get('spec', {})
    seen, cursor = set(), ident
    while cursor and cursor not in seen:
        if cursor == root:
            return True
        seen.add(cursor)
        spec = agent.s.data['assets'].get(cursor, {}).get('spec', {})
        cursor = spec.get('reference_asset_id')
    spec = row.get('spec', {})
    return bool(spec.get('role') == 'voice' and root_spec.get('speaker')
                and spec.get('speaker') == root_spec['speaker'])
