"""Evidence-linked content experience; retrieval is advice, never a PASS gate."""
from copy import deepcopy
import json
from pathlib import Path
import re
import time

from .storage import key, lock, read, sha, write

SCOPE_VERSION = 'within_story_v1'

def _run(value):
    return str(Path(value)) if isinstance(value, str) and value else None

def _probe(value):
    value = str(value).lower()
    return '/verification/' in value and 'probe' in value


def terms(value):
    text = json.dumps(value, ensure_ascii=False).lower()
    words = set(re.findall(r'[a-z_]{3,}', text))
    for chunk in re.findall(r'[\u4e00-\u9fff]+', text):
        words.update(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return words


class ExperienceMemory:
    def __init__(self, store, config):
        self.store = store
        self.root = Path(config['lite_root']) / 'memory'
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        store.data.setdefault('memory_consumption', {})

    def context(self):
        """Small applicability contract; no model or database is needed to apply it."""
        completion = self.config.get('completion_policy', {}).get('version') == 'completion_first_v1'
        if completion:
            from . import completion_review
            protocol = getattr(completion_review, 'OBSERVATION_PROTOCOL', completion_review.VERSION)
        else:
            protocol = self.config.get('review_system', {}).get('observation_protocol')
        language = self.config.get('task_contract', {}).get('language') or self.config.get('language') or 'Chinese'
        return {'scope_version': SCOPE_VERSION, 'language': language,
                'observation_protocol': protocol,
                'origin_kind': 'deployment_probe' if _probe(self.store.run) else 'story'}

    def exclusion(self, episode):
        """Check origins before lexical ranking, including a legacy import's origin."""
        if not isinstance(episode, dict):
            return 'malformed_episode'
        origin = _run(episode.get('source_run'))
        details = episode.get('details') or {}
        context = episode.get('context') or {}
        chain = episode.get('origin_chain') or []
        if not isinstance(context, dict) or not isinstance(chain, list):
            return 'malformed_origin_context'
        origins = [origin] + [_run(x.get('source_run')) for x in chain if isinstance(x, dict)]
        if isinstance(details, dict) and details.get('source_run'):
            origins.append(_run(details['source_run']))
        if isinstance(details, dict) and isinstance(details.get('legacy_record'), dict):
            legacy = details['legacy_record']
            origins.append(_run(legacy.get('source_run') or legacy.get('run')))
        if not origin or any(x is None for x in origins):
            return 'unknown_origin'
        if context.get('origin_kind') == 'deployment_probe' or any(_probe(x) for x in origins):
            return 'deployment_probe'
        transfer = self.config.get('memory_policy', {}).get('scope') == 'cross_story'
        same_run = all(x == _run(str(self.store.run)) for x in origins)
        if not same_run and not transfer:
            return 'outside_current_story_run'
        current = self.context()
        for field in ('language', 'observation_protocol'):
            actual = context.get(field)
            # Old same-run rows can lack metadata; their provenance remains
            # local, but known conflicting metadata must never be ignored.
            if actual is None and isinstance(details, dict):
                actual = details.get(field)
                if field == 'observation_protocol' and actual is None:
                    actual = (details.get('review') or {}).get(field) if isinstance(details.get('review'), dict) else None
            expected = current[field]
            legacy_family = (same_run and field == 'observation_protocol' and isinstance(actual, str)
                             and '/' not in actual and isinstance(expected, str) and expected.startswith(actual + '/'))
            if actual and expected and actual != expected and not legacy_family:
                return 'incompatible_' + field
            if not same_run and expected and not actual:
                return 'unproven_transfer_' + field
        return None

    def eligible(self, episodes, operation):
        """Filter a cached view without rewriting any frozen request or episode."""
        kept, excluded = [], []
        for episode in episodes:
            reason = self.exclusion(episode)
            if reason:
                excluded.append({'id': episode.get('id') if isinstance(episode, dict) else None, 'reason': reason})
            else:
                kept.append(deepcopy(episode))
        if excluded:
            audits = self.store.data.setdefault('memory_scope_exclusions', {})
            if audits.get(operation) != excluded:
                audits[operation] = excluded
                self.store.save()
                self.store.event('experience_scope_filtered', operation=operation, excluded=excluded)
        return kept

    def record(self, stage, operation, outcome, details, *, nonblocking=False):
        from . import evidence_policy
        if evidence_policy.enabled(self.config):
            details = evidence_policy.memory_view(details)
        ident = key({'run': str(self.store.run), 'stage': stage, 'operation': operation})
        path = self.root / 'episodes' / (ident + '.json')
        with lock(self.root / 'memory.lock', nonblocking=nonblocking):
            if not path.exists():
                write(path, {'id': ident, 'time': time.time(), 'stage': stage, 'operation': operation,
                      'outcome': outcome, 'source_run': str(self.store.run),
                      'origin_chain': [{'source_run': str(self.store.run)}] +
                          ([{'source_run': details['source_run']}] if isinstance(details, dict) and details.get('source_run') else []),
                      'context': self.context(),
                      'source_sha256': self.config.get('source_text_sha256'),
                      'models': deepcopy(self.config.get('text_route', {})).get('model'),
                      'details': deepcopy(details), 'causal_claim': 'none; not a controlled causal experiment'})
        return ident

    def correct_observation(self, media_sha256, reason, corrective_lesson, evidence):
        """Append an evidence-bound erratum; never rewrite an episode or paid reply."""
        if (not isinstance(media_sha256, str) or not re.fullmatch('[a-f0-9]{64}', media_sha256)
                or not reason or not corrective_lesson or not evidence):
            raise ValueError('A correction needs exact media, reason, lesson and retained evidence')
        for item in evidence:
            if not Path(item['path']).is_file() or sha(item['path']) != item['sha256']:
                raise ValueError('Correction evidence unavailable or changed')
        value = {'media_sha256': media_sha256, 'reason': reason,
                 'corrective_lesson': corrective_lesson, 'evidence': deepcopy(evidence),
                 'credit': 'user_report_and_engineering_media_audit',
                 'scope': 'false acceptance of these exact media bytes; not a universal model failure'}
        ident = key(value)
        with lock(self.root / 'memory.lock'):
            path = self.root / 'corrections' / (ident + '.json')
            if not path.exists():
                write(path, {**value, 'id': ident, 'time': time.time()})
        self.record('shared', 'correction/' + ident, 'corrected_false_acceptance', value)
        return ident

    def _corrections(self):
        rows = []
        for path in sorted((self.root / 'corrections').glob('*.json')):
            value = read(path)
            if not isinstance(value, dict) or not value.get('media_sha256') or not value.get('corrective_lesson'):
                raise ValueError('Invalid retained correction: ' + str(path))
            if any(not Path(e['path']).is_file() or sha(e['path']) != e['sha256'] for e in value.get('evidence', [])):
                raise ValueError('Retained correction evidence changed: ' + str(path))
            rows.append(value)
        return rows

    def _evidence_view(self, episode, corrections):
        from . import evidence_policy
        result = deepcopy(episode)
        if evidence_policy.enabled(self.config):
            result = evidence_policy.memory_view(result)
            episode = result
        serialized = json.dumps(episode['details'], ensure_ascii=False)
        related = [c for c in corrections if c['media_sha256'] in serialized]
        full_path = str(self.root / 'episodes' / (episode['id'] + '.json'))
        if related and episode.get('outcome') != 'corrected_false_acceptance':
            result.update(evidence_credit='corrected_observation_do_not_reuse_as_success', corrections=related)
            result['details'] = {'corrective_lessons': [c['corrective_lesson'] for c in related],
                'original_details_withheld_as_positive_advice': True, 'full_episode_path': full_path}
            return result
        result['evidence_credit'] = ('recorded_media_comparison_not_causal_proof'
            if episode.get('outcome') in ('opinion_supported_media_improvement', 'opinion_supported_partial_improvement', 'opinion_supported_mixed_change', 'opinion_supported_regression')
            else 'corrective_evidence' if episode.get('outcome') == 'corrected_false_acceptance'
            else 'model_assessment_or_proposal_not_verified_success')
        details = episode['details']
        if len(serialized) > 5000 and isinstance(details, dict) and episode.get('outcome') in (
                'observed_candidate_change', 'global_revision_and_asset_receipts', 'global_followup_observations'):
            # An intervention's long proposed prompt must not truncate away
            # uncertainty, remaining defects, or the distinction from execution.
            result['details'] = {k: deepcopy(details.get(k)) for k in (
                'operation', 'clip_id', 'selected_count', 'source_run', 'intervention_id', 'outcome_id',
                'relative_outcome', 'resolved_in_observations', 'remaining_issues',
                'absolute_quality_status', 'observation_protocol', 'evidence', 'causal_claim', 'success_template',
                'followup_credit', 'followup_target_coverage') if k in details}
            intervention = details.get('intervention') or details.get('applied_future_directions') or {}
            result['details'].update(intervention_excerpt=json.dumps(intervention, ensure_ascii=False)[:1800],
                truncated=True, full_episode_path=full_path)
        elif len(serialized) > 5000 and isinstance(details, dict) and isinstance(details.get('review'), dict):
            review = details['review']
            # Prior truncation retained a long caption and cut away the actual
            # findings. Send findings/decision first, with a pointer to raw data.
            result['details'] = {'review': {k: review.get(k) for k in
                ('observation_protocol', 'issues', 'feedback_decision', 'observed_events', 'limitations')},
                'quality': details.get('quality'), 'receipt': details.get('receipt'),
                'clip_id': details.get('clip', {}).get('id'), 'caption_excerpt': str(details.get('caption', ''))[:800],
                'truncated': True, 'full_episode_path': full_path}
        elif len(serialized) > 5000:
            result['details'] = {'verbatim_excerpt': serialized[:5000], 'truncated': True, 'full_episode_path': full_path}
        return result

    def retrieve(self, stage, query, operation, limit=4, *, include_stages=None):
        # Freeze each request's selected memories for replay after interruption.
        consumed = self.store.data['memory_consumption'].get(operation)
        if consumed is not None:
            return self.eligible(consumed['episodes'], operation)
        wanted = terms(query)
        perceptual = self.config.get('review_system', {}).get('observation_protocol') in ('perception_first_v2', 'content_rsi_v3', 'content_rsi_v4')
        corrections = []
        if perceptual:
            try:
                corrections = self._corrections()
            except (ValueError, OSError, KeyError, TypeError) as exc:
                # A broken memory source must not halt film generation or turn
                # known bad advice into a cached success. Exclude it explicitly.
                self.store.warning('memory', 'Memory advice withheld until correction evidence is reconciled: ' + str(exc))
                self.store.data['memory_consumption'][operation] = {'stage': stage, 'query': query,
                    'episodes': [], 'time': time.time(), 'status': 'correction_evidence_unavailable'}
                self.store.save()
                return []
        ranked = []
        for path in sorted((self.root / 'episodes').glob('*.json')):
            try:
                episode = read(path)
                if not isinstance(episode, dict) or episode.get('stage') not in (include_stages or (stage, 'shared')) or not episode.get('id') or 'details' not in episode:
                    continue
                if self.exclusion(episode):
                    continue
                overlap = len(wanted & terms(episode.get('details', {})))
                if include_stages and not overlap:
                    continue
                priority = 1 if include_stages and episode.get('outcome') in ('observed_candidate_change', 'global_revision_and_asset_receipts', 'asset_execution_result') else 0
                ranked.append((overlap + priority, episode.get('time', 0), episode))
            except (ValueError, OSError):
                self.store.warning('memory', 'Unreadable experience retained, excluded from retrieval: ' + str(path))
        episodes = [row[2] for row in sorted(ranked, key=lambda x: (x[0], x[1]), reverse=True)[:limit]]
        if perceptual:
            episodes = [self._evidence_view(e, corrections) for e in episodes]
        for episode in episodes if not perceptual else []:
            serialized = json.dumps(episode['details'], ensure_ascii=False)
            if len(serialized) > 5000:
                episode['details'] = {'verbatim_excerpt': serialized[:5000], 'truncated': True,
                                      'full_episode_path': str(self.root / 'episodes' / (episode['id'] + '.json'))}
        self.store.data['memory_consumption'][operation] = {'stage': stage, 'query': query,
                                                         'episodes': deepcopy(episodes), 'time': time.time(),
                                                         'scope': self.config.get('memory_policy', {}).get('scope', 'within_story'),
                                                         'scope_version': SCOPE_VERSION}
        self.store.save()
        self.store.event('experience_retrieved', stage=stage, operation=operation,
                         memory_ids=[e['id'] for e in episodes], request_will_receive=bool(episodes))
        return episodes

    def import_legacy(self, path):
        """Retain old origins, even when unknown; an import is not new evidence."""
        path = Path(path)
        if not path.is_file():
            return 0
        imported = 0
        for number, line in enumerate(path.read_text().splitlines()):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            operation = f'legacy/{path}/{number}/{key(row)}'
            ident = key({'legacy_path': str(path), 'line': number, 'record': row})
            source_run = row.get('source_run') or row.get('run')
            context = deepcopy(row.get('context')) if isinstance(row.get('context'), dict) else {
                'origin_kind': 'deployment_probe' if _probe(source_run) else 'legacy_import',
                'language': row.get('language'), 'observation_protocol': row.get('observation_protocol')}
            value = {'id': ident, 'time': time.time(), 'stage': 'local', 'operation': operation,
                'outcome': 'legacy_' + str(row.get('evidence', 'unconfirmed')),
                'source_run': source_run, 'origin_chain': (deepcopy(row['origin_chain'])
                    if isinstance(row.get('origin_chain'), list) else []) + [{'source_run': source_run}],
                'imported_by_run': str(self.store.run), 'context': context,
                'details': {'legacy_record': deepcopy(row), 'imported_from': str(path),
                    'credit': 'prior_content_experiment', 'not_new_v2_validation': True},
                'causal_claim': 'none; imported historical observation, not new validation'}
            with lock(self.root / 'memory.lock'):
                destination = self.root / 'episodes' / (ident + '.json')
                if not destination.exists():
                    write(destination, value)
            imported += 1
        return imported
