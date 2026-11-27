"""Exact, user-authorized recovery receipts; never implicit generation retries."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os

from .storage import Pending, read, sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def permission(config, field):
    binding = config.get(field)
    if not binding:
        return None
    p = Path(binding.get('receipt', ''))
    if not p.is_file() or sha(p) != binding.get('sha256'):
        raise Pending('Explicit recovery permission is missing or changed', status='needs_configuration')
    return read(p)


def request_permission(store, config, folder):
    r = permission(config, 'unknown_request_recovery')
    if r is None or r.get('call_id') != folder.name:
        return None
    original = folder / 'physical_1'
    failure = read(original / 'failure.json', {})
    # Receipts predating this field authorized HTTP 502 only.
    approved_status = r.get('original_http_status', 502)
    valid = (r.get('version') == 'one_unknown_request_replacement_v1'
             and r.get('explicit_user_approval') is True
             and r.get('run') == str(store.run.resolve())
             and r.get('request_sha256') == sha(folder / 'request.json')
             and r.get('submitted_sha256') == sha(original / 'submitted.json')
             and r.get('failure_sha256') == sha(original / 'failure.json')
             and r.get('replacement_attempt') == 2 and r.get('additional_attempts') == 1
             and r.get('original_execution_and_cost_remain_unknown') is True
             and type(approved_status) is int and approved_status in (500, 502)
             and failure.get('http_status') == approved_status
             and failure.get('definitive') is False)
    if not valid:
        raise Pending('Unknown-request permission does not match original evidence', status='needs_configuration')
    return r


def replacement_claim(store, config, folder, *, create=False):
    r = request_permission(store, config, folder)
    if r is None:
        return None
    binding = config['unknown_request_recovery']
    value = {'permission': binding['receipt'], 'permission_sha256': binding['sha256'],
             'run': r['run'], 'call_id': r['call_id'], 'request_sha256': r['request_sha256'],
             'original_attempt': 1, 'replacement_attempt': 2,
             'original_execution_and_cost_remain_unknown': True}
    p = folder / 'explicit_replacement_claim.json'
    previous = read(p)
    if previous is not None and previous != value:
        raise Pending('Existing replacement claim differs; do not resubmit', status='needs_configuration')
    if previous is None and create:
        original = folder / 'physical_1'
        if (original / 'response.body').is_file() or (original / 'response.json').is_file():
            return None  # Reconcile late original evidence before choosing a replacement.
        # Exclusive durable claim precedes any new physical submission. A crash
        # after this claim is reconciled through physical_2, never a third slot.
        with p.open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        store.event('explicit_unknown_request_replacement_authorized', **value)
        previous = value
    return previous


def authorized_original_preserved(run, folder):
    """For archive eligibility only; never declares original usage zero."""
    from types import SimpleNamespace
    config = read(Path(run) / 'config.json')
    claim = replacement_claim(SimpleNamespace(run=Path(run)), config, folder)
    return bool(claim and (folder/'physical_2/response.json').is_file()
                and (folder/'answer.json').is_file())


def voice_draft_valid(actual):
    binding = actual.get('authorized_draft_reference') or {}
    path = Path(binding.get('receipt', ''))
    if not path.is_file() or sha(path) != binding.get('sha256'):
        return False
    r = read(path)
    try:
        Path(actual['path']).resolve().relative_to(Path(r['run']).resolve())
    except (KeyError, ValueError):
        return False
    return (r.get('version') == 'retained_minor_voice_draft_v1'
            and r.get('explicit_user_approval') is True and r.get('new_generation_allowed') is False
            and r.get('asset_id') == binding.get('asset_id')
            and r.get('candidate_take') == binding.get('candidate_take')
            and r.get('asset_version') == actual.get('version')
            and r.get('media_sha256') == actual.get('sha256')
            and r.get('full_media_sha256') == actual.get('full_sha256')
            and r.get('review_sha256') == digest(actual.get('review'))
            and actual.get('feedback_accepted') is False
            and actual.get('review', {}).get('decision') == 'revise')


def adopt_retained_voice(agent, ident, row):
    from .asset_lifecycle import input_digest, usable_version, voice_report_usable
    r = permission(agent.config, 'retained_voice_draft')
    if r is None or r.get('asset_id') != ident:
        return False
    agent.s.check_stop()
    record = row.get('lifecycle_attempts', {}).get(f"{r.get('asset_version')}/{r.get('candidate_take')}")
    original = (record or {}).get('actual')
    if (r.get('run') != str(agent.s.run.resolve()) or not original or record.get('failure')
            or r.get('input_sha256') != row.get('blocked_input_sha256') or r.get('input_sha256') != input_digest(row)
            or row.get('versions') or row.get('spec', {}).get('role') != 'voice'):
        raise Pending('Retained voice permission does not match blocked original asset', status='needs_configuration')
    review = original.get('review', {})
    findings = review.get('findings')
    if (not voice_report_usable(original, review) or not isinstance(findings, list)
            or not findings or any(not isinstance(x, dict) or x.get('severity') != 'minor' for x in findings)):
        raise Pending('Only explicitly bound minor style defects may be retained', status='needs_asset_repair')
    actual = deepcopy(original)
    actual['authorized_draft_reference'] = {**agent.config['retained_voice_draft'],
        'asset_id': ident, 'candidate_take': r['candidate_take'], 'content_acceptance_passed': False}
    if not voice_draft_valid(actual) or usable_version({'spec': row['spec'], 'versions': [actual]}) is None:
        raise Pending('Retained voice media or review failed verification', status='needs_asset_repair')
    agent.registry.publish(ident, actual)
    row.update(dependency_state='ready_with_warnings', active_reference_version=actual['version'],
               generation=actual['version'], pending_version=None, pending_prompt=None)
    agent.registry.save()
    agent.s.warning('retained_voice_draft', f'{ident}: original minor voice style defect retained; no new candidate; content acceptance remains false')
    agent.s.event('retained_voice_draft_adopted', asset_id=ident, take=r['candidate_take'],
                  media_sha256=actual['sha256'], new_generation=False, original_review_unchanged=True)
    return True
