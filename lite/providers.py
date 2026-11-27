"""Two real model transports with raw replies, usage and conservative replay."""
from __future__ import annotations

import base64
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from .storage import Pending, Unavailable, key, lock, operation_recorded, read, sha, write
from .credentials import resolve_openai_credentials


def invalid_json_constant(value):
    raise ValueError('Non-finite JSON number is not a usable score: ' + value)


def gemini_endpoint(config):
    from .credentials import _endpoint
    url = config.get('gemini_url') or os.getenv(config.get('gemini_url_env', 'R3_AV_URL'))
    return _endpoint(url)


def gemini_credentials(config):
    from .credentials import CredentialSourceError, _key
    name = config.get('gemini_key_env', 'GEMINI_API_KEY')
    token = _key(os.getenv(name, ''))
    mode = config.get('gemini_auth', 'api_key')
    if mode == 'api_key':
        return {'x-goog-api-key': token, 'Content-Type': 'application/json'}
    if mode == 'bearer':
        return {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    raise CredentialSourceError('gemini_auth must be api_key or bearer')


def visible(value):
    if isinstance(value, list):
        return [visible(v) for v in value]
    if isinstance(value, dict):
        return {k: visible(v) for k, v in value.items() if k not in ('reasoning_content', 'reasoning', 'thinking_content', 'thoughtSignature') and not (k == 'text' and value.get('thought') is True)}
    return value


RESPONSE_HEADERS = frozenset(('x-request-id', 'request-id', 'retry-after',
    'server', 'x-envoy-upstream-service-time', 'content-type', 'content-length'))


def safe_headers(headers):
    try:
        items = list(headers.items())
    except (AttributeError, TypeError):
        return {}
    return {str(k): str(v)[:1000] for k, v in items if str(k).lower() in RESPONSE_HEADERS}


def retry_deadline(headers, failed_at):
    """RFC 9110 Retry-After, relative to receipt of the original refusal."""
    value = next((v.strip() for k, v in headers.items() if k.lower() == 'retry-after'), '')
    if re.fullmatch(r'[0-9]+', value):
        seconds = int(value)
        # Integer arithmetic also preserves very large valid server delays.
        return math.ceil(failed_at) + seconds if seconds else failed_at
    try:
        stamp = parsedate_to_datetime(value)
        if stamp.tzinfo is None:  # Obsolete HTTP asctime format denotes GMT.
            stamp = stamp.replace(tzinfo=timezone.utc)
        return max(failed_at, stamp.timestamp())
    except (ValueError, TypeError, OverflowError, OSError):
        return failed_at + 62  # Preserve the existing default backoff.


def wait_for_retry(deadline, check_stop):
    """Wait outside the provider reservation, or yield a resumable long delay."""
    inline_limit = 3600
    stop_waiting = time.monotonic() + inline_limit
    while True:
        check_stop()
        now = time.time()
        if deadline <= now:
            return
        if deadline > now + inline_limit or time.monotonic() >= stop_waiting:
            raise Pending(f'Provider rate limited this request until Unix time {deadline}',
                          status='awaiting_rate_limit')
        time.sleep(min(5, deadline - now))


def persist_body(path, body):
    """Persist original bytes before parsing; the receipt is immutable on replay."""
    if path.exists():
        if path.read_bytes() != body:
            raise Pending('Existing original response body differs; no replacement permitted')
        return
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def response_pending(attempt_dir, reason):
    """A received but unusable envelope cannot justify a new physical request."""
    if attempt_dir is None:
        raise Pending(f'Original response retained but unusable ({reason})')
    path = attempt_dir / 'response_issue.json'
    if not path.exists():
        write(path, {'type': reason, 'http_status': 200, 'definitive': False,
            'usage': None, 'cost': None, 'raw_body_sha256': sha(attempt_dir / 'response.body')
                if (attempt_dir / 'response.body').is_file() else None,
            'recorded_at': time.time(), 'automatic_resubmission_allowed': False})
    raise Pending(f'Original response retained but unusable ({reason}): {attempt_dir}')


def retained_response(attempt_dir, request_path):
    """Recover after body persistence, including a crash before metadata/JSON."""
    body_path = attempt_dir / 'response.body'
    if body_path.is_file():
        meta_path = attempt_dir / 'response_metadata.json'
        try:
            metadata = read(meta_path)
        except (ValueError, UnicodeError):
            raise Pending('Original response metadata is unreadable; no automatic resubmission') from None
        if metadata is not None:
            if (not isinstance(metadata, dict) or metadata.get('raw_body_sha256') != sha(body_path)
                    or metadata.get('request_sha256') != sha(request_path)):
                raise Pending('Original response evidence changed; no automatic resubmission')
        else:
            write(meta_path, {'raw_body_sha256': sha(body_path), 'request_sha256': sha(request_path),
                'byte_count': body_path.stat().st_size, 'http_status': 200, 'received_at': None,
                'reconciled_at': time.time(), 'recovered_after_body_persistence': True})
        try:
            raw = json.loads(body_path.read_bytes(), parse_constant=invalid_json_constant)
        except (ValueError, UnicodeError):
            response_pending(attempt_dir, 'invalid_original_json')
        if not (attempt_dir / 'response.json').is_file():
            write(attempt_dir / 'response.json', visible(raw))
        return visible(raw)
    try:
        return read(attempt_dir / 'response.json')
    except (ValueError, UnicodeError):
        response_pending(attempt_dir, 'invalid_retained_json')


def model_answer(provider, raw, attempt_dir=None):
    """Validate the transport envelope, recording issues only for a live attempt."""
    if provider in ('glm', 'doubao'):
        if not isinstance(raw, dict) or not isinstance(raw.get('choices'), list) or not raw['choices']:
            response_pending(attempt_dir, 'unexpected_original_structure')
        message = raw['choices'][0]
        content = message.get('message') if isinstance(message, dict) else None
        if not isinstance(content, dict) or not isinstance(content.get('content'), str) or (raw.get('usage') is not None and not isinstance(raw['usage'], dict)):
            response_pending(attempt_dir, 'unexpected_original_structure')
        text = content['content']
        usage = {'provider': provider, 'model': raw.get('model'), **(raw.get('usage') or {}),
                 'finish_reason': message.get('finish_reason'), 'cost': None}
    else:
        chunks = raw if isinstance(raw, list) else [raw]
        if not chunks or any(not isinstance(c, dict) for c in chunks):
            response_pending(attempt_dir, 'unexpected_original_structure')
        if any(c.get('usageMetadata') is not None and not isinstance(c['usageMetadata'], dict) for c in chunks):
            response_pending(attempt_dir, 'unexpected_original_structure')
        errors = [c['error'] for c in chunks if 'error' in c]
        if errors:
            if attempt_dir is None:
                raise Unavailable('Gateway returned an explicit model error')
            write(attempt_dir / 'embedded_failure.json', {'errors': errors})
            raise Unavailable(f'Gateway returned an explicit model error: {attempt_dir}')
        parts = []
        for chunk in chunks:
            candidates = chunk.get('candidates', [])
            if not isinstance(candidates, list):
                response_pending(attempt_dir, 'unexpected_original_structure')
            for candidate in candidates:
                content = candidate.get('content') if isinstance(candidate, dict) else None
                if not isinstance(content, dict) or not isinstance(content.get('parts'), list):
                    response_pending(attempt_dir, 'unexpected_original_structure')
                for part in content['parts']:
                    if not isinstance(part, dict):
                        response_pending(attempt_dir, 'unexpected_original_structure')
                    if not part.get('thought') and 'text' in part:
                        if not isinstance(part['text'], str):
                            response_pending(attempt_dir, 'unexpected_original_structure')
                        parts.append(part['text'])
        if not parts:
            response_pending(attempt_dir, 'unexpected_original_structure')
        text = ''.join(parts)
        usage = {'provider': provider, 'model': next((part['modelVersion'] for part in chunks if part.get('modelVersion')), None), 'cost': None,
                 'usage_metadata': next((c['usageMetadata'] for c in reversed(chunks) if c.get('usageMetadata')), {})}
    return text, usage


class Models:
    def __init__(self, store, config):
        self.store, self.config = store, config

    def _wait_rate(self, provider):
        interval = self.config.get(provider + '_interval_seconds', 62 if provider == 'glm' else 8)
        folder = Path(self.config.get('rate_limit_root') or (Path(self.config['lite_root']) / 'runtime'))
        while True:
            self.store.check_stop()
            from .fleet_lock import directory_lock
            reservation=(directory_lock(folder/(provider+'.cross_host.d'),self.store.check_stop)
                if self.config.get('cross_host_rate_reservation')=='atomic_directory_v1' else lock(folder/(provider+'.lock')))
            with reservation:
                path = folder / (provider + '.json')
                now = time.time()
                if self.config.get('cross_host_rate_reservation')=='atomic_directory_v1':
                    from .fleet_lock import read_reservation
                    rate=read_reservation(path,self.store.check_stop)
                else:
                    rate=read(path,{})
                remaining = rate.get('next_allowed_at', 0) - time.time()
                if remaining <= 0:
                    now = time.time()
                    write(path, {'next_allowed_at': now + interval})
                    return
            time.sleep(min(5, remaining))

    def text(self, name, system, user, *, media=(), max_tokens=12000, provider='glm', json_mode=True):
        if provider not in ('glm', 'gemini', 'doubao'):
            raise Unavailable('Unsupported configured model provider')
        folder = self.store.run / 'calls' / key(name)
        folder.mkdir(parents=True, exist_ok=True)
        answer_path = folder / 'answer.json'
        answer = read(answer_path)
        if answer is not None:
            self.store.usage_once(str(folder), answer['usage'])
            return answer['text']
        parts = [{'text': json.dumps(user, ensure_ascii=False) if not isinstance(user, str) else user}]
        for entry in media:
            path = Path(entry['path'] if isinstance(entry, dict) else entry)
            parts.append({'text': 'Actual submitted media: ' + (entry.get('label', path.name) if isinstance(entry, dict) else path.name)})
            parts.append({'inlineData': {'mimeType': mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
                                        'data': base64.b64encode(path.read_bytes()).decode()}})
        if provider in ('glm', 'doubao'):
            route = self.config['text_route' if provider == 'glm' else 'doubao_route']
            content = parts[0]['text'] if provider == 'glm' else [
                {'type': 'text', 'text': p['text']} if 'text' in p else
                {'type': 'image_url', 'image_url': {'url': 'data:' + p['inlineData']['mimeType'] + ';base64,' + p['inlineData']['data']}}
                for p in parts]
            if provider == 'doubao' and any('inlineData' in p and not p['inlineData']['mimeType'].startswith('image/') for p in parts):
                raise Unavailable('Doubao visual reviewer accepts silent sampled images only')
            body = {'model': route['model'],
                    'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
                    'max_tokens': max_tokens, 'temperature': 0.35 if provider == 'glm' else 0.1}
            if provider == 'glm':
                body['reasoning_effort'] = 'low'
            if json_mode:
                body['response_format'] = {'type': 'json_object'}
        else:
            body = {'systemInstruction': {'parts': [{'text': system}]},
                    'contents': [{'role': 'user', 'parts': parts}],
                    'generationConfig': {'maxOutputTokens': max_tokens, 'temperature': 0.2}}
            if json_mode:
                body['generationConfig']['responseMimeType'] = 'application/json'
        # The logical stage owns its first request. Re-entry reuses those bytes;
        # implementation versions and arbitrary model metadata are not gates.
        recorded = any(folder.glob('physical_*')) or any(
            operation.startswith(str(folder) + '/') for operation in self.store.data['accounted_operations'])
        try:
            original = read(folder / 'request.json', required=recorded)
        except FileNotFoundError:
            raise Pending('Original model request is unavailable; retain submitted work and reconcile input',
                status='needs_input_reconciliation') from None
        if original is None:
            write(folder / 'request.json', {'name': name, 'provider': provider, 'body': body})
        else:
            body, provider = original['body'], original['provider']
        binding = folder / 'experience_used.json'
        if not binding.exists():
            found, pending = set(), [body]
            while pending:
                item = pending.pop()
                if isinstance(item, dict):
                    if isinstance(item.get('id'), str) and item.get('source_run') and item.get('outcome'):
                        found.add(item['id'])
                    pending.extend(item.values())
                elif isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, str) and item[:1] in ('{', '['):
                    try:
                        pending.append(json.loads(item))
                    except ValueError:
                        continue
            if found:
                write(binding, {'memory_ids': sorted(found), 'request_sha256': sha(folder / 'request.json'),
                                'name': name, 'evidence': 'episodes_present_in_original_request_body'})
                self.store.event('experience_request_bound', name=name, memory_ids=sorted(found), record=str(binding))
        from .explicit_recovery import request_permission, replacement_claim
        recovery = request_permission(self.store, self.config, folder)
        claimed = replacement_claim(self.store, self.config, folder) if recovery else None
        retry_not_before = None
        for attempt in range(3):
            self.store.check_stop()
            if claimed and attempt == 0:
                continue  # A later-arriving original reply remains evidence, not a new selection.
            if claimed and attempt > 1:
                raise Pending('Explicit replacement allowance exhausted; no third physical attempt')
            attempt_dir = folder / f'physical_{attempt + 1}'
            attempt_dir.mkdir(exist_ok=True)
            raw = retained_response(attempt_dir, folder / 'request.json')
            if raw is None and operation_recorded(attempt_dir, self.store.data['accounted_operations']):
                failure = read(attempt_dir / 'failure.json', {})
                if failure.get('http_status') == 429:
                    # Reconstruct from the original reply, including older receipts.
                    failed_at = failure.get('failed_at')
                    if failed_at is None:
                        failed_at = (attempt_dir / 'failure.json').stat().st_mtime
                    retry_not_before = retry_deadline(failure.get('response_headers', {}), failed_at)
                    continue
                if failure.get('definitive'):
                    if claimed and recovery.get('original_http_status') == 500:
                        raise Pending(f'Authorized HTTP 500 replacement was rejected: {attempt_dir}')
                    raise Unavailable(f'Recorded provider rejection: {failure.get("http_status")}; {attempt_dir}')
                if recovery and attempt == 0:
                    claimed = replacement_claim(self.store, self.config, folder, create=True)
                    if claimed:
                        continue
                    raw = retained_response(attempt_dir, folder / 'request.json')
                if raw is None:
                    raise Pending(f'Original {provider} submission awaits acknowledgement: {attempt_dir}')
            if raw is None:
                if retry_not_before is not None:
                    wait_for_retry(retry_not_before, self.store.check_stop)
                self._wait_rate(provider)
                if provider in ('glm', 'doubao'):
                    cred = resolve_openai_credentials(self.config['text_route' if provider == 'glm' else 'doubao_route'])
                    url, headers = cred.base_url.rstrip('/') + '/chat/completions', cred.headers()
                else:
                    url = gemini_endpoint(self.config)
                    headers = gemini_credentials(self.config)
                counts = {'observations': 1} if provider in ('gemini', 'doubao') else {}
                if provider == 'doubao':
                    counts['doubao_requests'] = 1
                self.store.count_once(str(attempt_dir), llm_requests=1, **counts)
                write(attempt_dir / 'submitted.json', {'time': time.time(), 'provider': provider, 'name': name})
                request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode(), headers=headers)
                transport_started = time.time()
                try:
                    with urllib.request.urlopen(request, timeout=600) as response:
                        original_body = response.read()
                        persist_body(attempt_dir / 'response.body', original_body)
                        write(attempt_dir / 'response_metadata.json', {
                            'raw_body_sha256': sha(attempt_dir / 'response.body'),
                            'request_sha256': sha(folder / 'request.json'), 'byte_count': len(original_body),
                            'http_status': response.status if type(getattr(response, 'status', None)) is int else 200,
                            'response_headers': safe_headers(getattr(response, 'headers', None)),
                            'received_at': time.time(), 'elapsed_seconds': time.time() - transport_started})
                    raw = retained_response(attempt_dir, folder / 'request.json')
                except urllib.error.HTTPError as exc:
                    try:
                        detail = exc.read(4000).decode(errors='replace')
                    except (OSError, IncompleteRead) as read_error:
                        detail = 'HTTP error body unavailable: ' + type(read_error).__name__
                    finally:
                        exc.close()
                    response_headers = safe_headers(exc.headers)
                    failure = {'http_status': exc.code, 'detail': detail, 'definitive': exc.code < 500,
                                                       'response_headers': response_headers, 'failed_at': time.time(),
                                                       'elapsed_seconds': time.time() - transport_started,
                                                       'request_sha256': sha(folder / 'request.json'),
                                                       'usage': None, 'cost': None}
                    if exc.code == 429:
                        retry_not_before = retry_deadline(response_headers, failure['failed_at'])
                        failure['retry_not_before'] = retry_not_before
                    write(attempt_dir / 'failure.json', failure)
                    if exc.code == 429:
                        if claimed:
                            raise Pending('Authorized replacement received 429; its one physical attempt is consumed') from None
                        continue
                    if exc.code >= 500:
                        raise Pending(f'HTTP {exc.code}, original execution/cost unknown: {attempt_dir}') from None
                    if claimed and recovery.get('original_http_status') == 500:
                        raise Pending(f'Authorized HTTP 500 replacement was rejected with HTTP {exc.code}: {attempt_dir}') from None
                    raise Unavailable(f'Provider rejected request HTTP {exc.code}: {attempt_dir}') from None
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    write(attempt_dir / 'failure.json', {'type': type(exc).__name__, 'definitive': False, 'usage': None, 'cost': None})
                    raise Pending(f'Lost or unreadable original response: {attempt_dir}') from None
            text, usage = model_answer(provider, raw, attempt_dir)
            write(answer_path, {'text': text, 'usage': usage})
            self.store.usage_once(str(folder), usage)
            self.store.event('model_completed', name=name, provider=provider, record=str(folder))
            return text
        raise Unavailable(f'Three recorded HTTP429 refusals: {folder}; retained without retrying unknown submissions')

    def json(self, name, system, user, *, validate=None, **options):
        current = user
        last = None
        for attempt in range(3):
            format_name = f'{name}/format_{attempt + 1}'
            text = self.text(format_name, system + '\nReturn a JSON object. No SHA, stage echo or machine provenance is requested.', current, **options)
            value = text.strip()
            fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', value, flags=re.DOTALL)
            if fence:
                value = fence.group(1)
            try:
                result = json.loads(value, parse_constant=invalid_json_constant)
                if not isinstance(result, dict):
                    raise ValueError('Return an object with the requested content fields')
                if validate:
                    result = validate(result)
                if attempt:
                    from .memory import ExperienceMemory
                    ExperienceMemory(self.store, self.config).record('shared', name + '/format_recovered',
                        'bounded_structure_recovery', {'stage': name, 'attempts': attempt + 1,
                          'last_problem': last, 'reply_record': str(self.store.run / 'calls' / key(f'{name}/format_{attempt + 1}')),
                          'not_content_quality_proof': True})
                return result
            except (ValueError, TypeError, KeyError) as exc:
                last = str(exc)
                if self.config.get('unknown_request_recovery'):
                    from .explicit_recovery import replacement_claim
                    folder = self.store.run / 'calls' / key(format_name)
                    if (replacement_claim(self.store, self.config, folder)
                            and read(folder / 'physical_1/failure.json', {}).get('http_status') == 500):
                        raise Pending('Authorized HTTP 500 replacement returned unusable structured content; '
                                      'preserve its reply without another format request',
                                      status='needs_model_response_reconciliation') from exc
                self.store.warning(name, f'Model structure repair {attempt + 1}: {last}')
                from .memory import ExperienceMemory
                ExperienceMemory(self.store, self.config).record('shared', f'{name}/format_{attempt + 1}/failure',
                    'model_structure_failure', {'stage': name, 'problem': last,
                      'reply_record': str(self.store.run / 'calls' / key(f'{name}/format_{attempt + 1}')),
                      'not_a_source_code_bug': True})
                from .task_parameters import longform
                if isinstance(exc, json.JSONDecodeError) and longform(self.config):
                    screenplay = name in ('screenplay/initial', 'screenplay/rewrite_1', 'screenplay/rewrite_2')
                    storyboard = name in ('director/initial', 'director/rewrite', 'director/duration_correction')
                    if screenplay or storyboard:
                        answer = read(self.store.run / 'calls' / key(format_name) / 'answer.json') or {}
                        if (answer.get('usage') or {}).get('finish_reason') == 'length':
                            guidance = ('Write the complete scene-level screenplay and leave shot staging to the director.'
                                        if screenplay else
                                        'Expand every source scene into short complete shots. Meet the original duration and '
                                        'shot-count requirements. Include each shot’s render specification. Use concise field values.')
                            current = {'original_task': user, 'precise_format_problem': last,
                                       'instruction': 'The previous JSON exceeded the output limit. Regenerate one compact '
                                                      'JSON object from the original task, preserving dialogue and ending. '
                                                      + guidance + ' Close the JSON object immediately after the last entry.'}
                            continue
                current = {'original_task': user, 'original_response': text, 'precise_format_problem': last,
                           'instruction': 'Correct the executable structure, retaining the substantive content. Extra explanatory fields are allowed.'}
        raise Unavailable(f'Three retained model-structure replies require attention: {name}: {last}')
