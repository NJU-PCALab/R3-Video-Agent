from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from time import sleep as _read_pause


def read(path, default=None, *, required=False, attempts=3):
    """Read one JSON snapshot without a separate exists/read race.

    Shared storage may briefly make an atomic-replaced file unavailable. Only
    that missing-file condition is retried; malformed JSON, permissions and
    other I/O errors remain explicit. Optional absence is still a normal API
    result. Required run state/config must never silently become a default.
    A caller that already owns a bounded polling loop uses attempts=1, retaining
    that loop's original deadline instead of stacking a second retry policy.
    """
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts not in (1, 2, 3):
        raise ValueError('JSON snapshot reads allow only one to three bounded attempts')
    path = Path(path)
    last_missing = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError as exc:
            last_missing = exc
            if attempt + 1 < attempts:
                _read_pause(.05)
    if required:
        raise FileNotFoundError(f'Required JSON snapshot unavailable after bounded reads: {path}') from last_missing
    return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def operation_recorded(folder, accounted_operations=()):
    """Independent execution witnesses survive a missing submission snapshot.

    A missing response is normal while work is in flight. These markers only
    forbid replacing original input or dispatching an unknown operation again;
    they do not manufacture a successful receipt.
    """
    folder = Path(folder)
    return str(folder) in accounted_operations or any((folder / name).is_file() for name in (
        'submitted.json', 'receipt.json', 'response.body', 'response.json',
        'response_metadata.json', 'failure.json', 'worker_response.json',
        'worker_failure.json', 'launch_failure.json', 'native.mp4'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


@contextmanager
def lock(path, nonblocking=False, *, timeout=60.0):
    """Bound time spent waiting for a held flock. Keep the lock inode intact."""
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError('Expected a finite nonnegative lock timeout')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if nonblocking:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Pending(f'Timed out waiting for lock: {path}', status='awaiting_shared_lock') from None
                time.sleep(min(0.1, remaining))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Pending(RuntimeError):
    """A real submission has no definitive result; do not duplicate it."""

    def __init__(self, message, *, status='awaiting_original_receipt'):
        super().__init__(message)
        self.status = status


class Unavailable(RuntimeError):
    """A known, recorded failure, not a lost acknowledgement."""


class RuntimeUnavailable(Unavailable):
    """A terminal local worker/runtime failure, not a candidate content defect.

    The original slot stays consumed. Repeating it or buying a second candidate
    in the unchanged runtime cannot repair the cause.
    """


class Stopped(RuntimeError):
    pass


class Store:
    def __init__(self, run, *, existing=None):
        self.run = Path(run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        if existing is None:
            # A prepared source/config is a valid new run. Execution evidence is
            # different: losing state after a launch must never reset quota.
            existing = any((self.run / name).is_file() for name in (
                'events.jsonl', 'process.json', 'report.json'))
            existing = existing or any(path.is_file() for name in ('launches', 'exits', 'calls')
                for path in (self.run / name).rglob('*'))
            existing = existing or any(path.is_file() for name in ('submitted.json', 'receipt.json')
                for path in (self.run / 'media').rglob(name))
        self.data = read(self.run / 'state.json', {
            'created_at': time.time(), 'status': 'created', 'selected': [],
            'candidates': {}, 'assets': {}, 'plan_revision': 0, 'checkpoints': [],
            'warnings': [], 'local_lessons': [], 'global_lessons': [],
            'counts': {}, 'accounted_operations': [], 'usage': {}}, required=existing)
        if not isinstance(self.data, dict):
            raise ValueError('Run state must be a JSON object; original snapshot retained')

    def save(self):
        self.data['updated_at'] = time.time()
        write(self.run / 'state.json', self.data)

    def event(self, kind, **value):
        with (self.run / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'kind': kind, **value}, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def status(self, status, **details):
        self.data.update(status=status, current=details)
        self.save()
        self.event('status', status=status, **details)

    def count_once(self, operation, **counts):
        if operation in self.data['accounted_operations']:
            return
        self.data['accounted_operations'].append(operation)
        for name, amount in counts.items():
            self.data['counts'][name] = self.data['counts'].get(name, 0) + amount
        self.save()

    def usage_once(self, operation, usage):
        if operation not in self.data['usage']:
            self.data['usage'][operation] = usage
            self.save()

    def warning(self, stage, message):
        record = {'stage': stage, 'message': str(message), 'time': time.time()}
        self.data['warnings'].append(record)
        self.save()
        self.event('warning', **record)

    def check_stop(self):
        if (self.run / 'STOP').exists():
            raise Stopped('Explicit STOP requested')


DIMENSIONS = ('story_action_fidelity', 'identity_continuity', 'visual_quality',
              'temporal_continuity', 'dialogue_fidelity', 'speaker_consistency',
              'audio_quality', 'audiovisual_sync', 'narrative_continuity')


def score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 5


def review_scores(review):
    value = review.get('scores') if isinstance(review, dict) else None
    return value if isinstance(value, dict) else {}


def validate_video_review(value):
    """Only executable structure is repaired; UNKNOWN observations are valid."""
    if not isinstance(value, dict):
        raise ValueError('Return a review object')
    if value.get('verdict') not in ('PASS', 'FAIL', 'UNKNOWN'):
        raise ValueError('verdict must be PASS, FAIL or UNKNOWN')
    if not isinstance(value.get('scores'), dict):
        raise ValueError('scores must be an object of dimension names and scores/null, not null or an array; use UNKNOWN when evidence is insufficient')
    return value


def quality(review, clip, *, voice_observation=None):
    review = review if isinstance(review, dict) else {}
    if review.get('completion_review_version') == 'completion_observed_v1':
        from .completion_review import quality as completion_quality
        return completion_quality(review)
    if review.get('review_system') == 'feedback_v1':
        from .review import feedback_quality
        return feedback_quality(review)
    clip = clip if isinstance(clip, dict) else {}
    scores = review_scores(review)
    voice = review.get('voice_observation') or voice_observation
    transcript = review.get('transcript', [])
    has_transcribed_voice = any(bool(line.get('text')) if isinstance(line, dict) else bool(line)
                                for line in transcript) if isinstance(transcript, list) else True
    observed_no_voice = (isinstance(voice, dict) and voice.get('status') == 'ABSENT'
                         and isinstance(voice.get('reason'), str) and bool(voice['reason'].strip())
                         and not has_transcribed_voice)
    not_applicable = []
    if not clip.get('dialogue'):
        not_applicable.append('dialogue_fidelity')
        if observed_no_voice:
            not_applicable.append('speaker_consistency')
    required = [d for d in DIMENSIONS if d not in not_applicable]
    known = all(score(scores.get(d)) for d in required)
    ok = (known and review.get('verdict') == 'PASS'
          and all(scores[d] >= (4.5 if d == 'dialogue_fidelity' else 4) for d in required))
    rank = sum(scores[d] for d in required if score(scores.get(d))) / len(required)
    return {'ok': ok, 'complete': known, 'rank': rank, 'quality_status': 'OK' if ok else 'draft/needs_review',
            'not_applicable_dimensions': not_applicable}


def candidate_quality(candidate):
    result = quality(candidate['review'], candidate.get('source_clip', candidate.get('clip', {})),
                     voice_observation=candidate.get('voice_observation'))
    audit = candidate.get('dialogue_integrity_audit', {})
    if audit.get('findings'):
        issues = result.get('unresolved_findings', []) + audit['findings']
        result.update(ok=False,quality_status='draft/needs_review',content_release_eligible=False,
                      unresolved_findings=issues,issue_ledger=issues)
    if candidate.get('selection_uncertainty'):
        result.update(ok=False, quality_status='draft/needs_review', selection_uncertainty=candidate['selection_uncertainty'])
    return result


def seconds(value, default=8):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        value = default
    return round(min(10, max(4, value)) * 24) / 24


def clips(value, start=0):
    """Only executable shape is required. Content is judged by models."""
    if not isinstance(value, list) or not value:
        raise ValueError('Return a nonempty clips list')
    result = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or not isinstance(raw.get('action'), str) or not raw['action'].strip():
            raise ValueError('Each clip needs a nonempty action describing actual audiovisual content')
        item = dict(raw)
        item['id'] = f'clip_{start + index + 1:03d}'
        item['seconds'] = seconds(raw.get('seconds'))
        item['dialogue'] = raw.get('dialogue') if isinstance(raw.get('dialogue'), list) else []
        item['asset_ids'] = raw.get('asset_ids') if isinstance(raw.get('asset_ids'), list) else []
        result.append(item)
    return result
