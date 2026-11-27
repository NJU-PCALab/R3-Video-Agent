"""One run-owned asset registry; state.json remains the atomic source of truth."""
from copy import deepcopy
from pathlib import Path

from .storage import key, write
from . import asset_lifecycle


def generator(spec):
    if spec.get('generator'):
        return spec['generator']
    if spec.get('path'):
        return 'import'
    if spec.get('kind', 'image') == 'image':
        return 'qwen_image'
    if spec.get('role') == 'voice':
        return 'voice_clone' if spec.get('reference_asset_id') else 'voice_design'
    return 'music' if spec.get('role') == 'music' else 'import'


class AssetRegistry:
    def __init__(self, store):
        self.store = store
        store.data.setdefault('asset_usage', {})

    def declare(self, ident, spec):
        row = self.store.data['assets'].setdefault(ident, {'spec': {}, 'versions': []})
        before = deepcopy(row)
        row['spec'] = {**row['spec'], **deepcopy(spec), 'id': ident}
        row['spec']['generator'] = generator(row['spec'])
        row['pending_prompt'] = spec.get('prompt') or spec.get('description') or spec.get('path') or ''
        if asset_lifecycle.enabled(getattr(self,'lifecycle_config',{})):asset_lifecycle.requeue_declared(row,before)
        self.save()
        return row

    def publish(self, ident, version):
        row = self.store.data['assets'][ident]
        if not any(v.get('version') == version.get('version') for v in row['versions']):
            row['versions'].append(deepcopy(version))
        self.save()

    def catalog(self):
        result = []
        for ident, row in self.store.data['assets'].items():
            if not row.get('versions'):
                continue
            actual = row['versions'][-1]
            if asset_lifecycle.enabled(getattr(self,'lifecycle_config',{})):
                actual=asset_lifecycle.usable_version(row)
                if actual is None:continue
            if not Path(actual.get('path', '')).is_file():
                continue
            spec = row['spec']
            result.append({**{k: spec[k] for k in ('name', 'description', 'role', 'speaker', 'language') if k in spec},
                           **{k: actual[k] for k in ('kind', 'path', 'sha256', 'seconds', 'version', 'generator',
                                                   'track_type', 'text', 'include_audio', 'provenance') if k in actual},
                           'id': ident, 'name': spec.get('name', ident),
                           'review_status': ('PASS' if actual['feedback_accepted'] else 'NEEDS_REVIEW')
                               if 'feedback_accepted' in actual else actual.get('review', {}).get('verdict', 'UNKNOWN')})
        return result

    def used(self, references, clip_id, take):
        operation = f'{clip_id}/take_{take}'
        if operation not in self.store.data['asset_usage']:
            self.store.data['asset_usage'][operation] = deepcopy(references)
            self.store.event('asset_references_bound', operation=operation, references=references)
            self.save()

    def unavailable(self):
        available = {a['id'] for a in self.catalog()}
        return [{'id': ident, 'name': row['spec'].get('name', ident),
                 'generator': generator(row['spec']), 'status': 'missing',
                 'reason': row.get('last_generation_failure', 'No generated file is available')}
                for ident, row in self.store.data['assets'].items() if ident not in available]

    def save(self):
        self.store.save()
        # Readable export only; the runner never treats this mirror as a gate.
        write(self.store.run / 'asset_registry.json', {'assets': self.store.data['assets'],
              'usage': self.store.data['asset_usage'], 'authority': 'state.json',
              'registered_files': self.catalog(), 'unavailable_assets': self.unavailable()})


def select_references(catalog, ids, *, first_frame_id=None, last_frame_id=None):
    wanted = list(ids) if isinstance(ids, list) else []
    wanted += [a for a in (first_frame_id, last_frame_id) if a]
    by_id = {a['id']: a for a in catalog}
    output, seen, audio_seconds, video_seconds = [], set(), 0, 0
    for ident in wanted:
        item = by_id.get(str(ident))
        if not item or item['path'] in seen or len(output) >= 12:
            continue
        kind = item['kind']
        count = sum(r['kind'] == kind for r in output)
        if kind == 'image':
            if count >= 9:
                continue
        elif kind in ('audio', 'video'):
            duration = item.get('seconds')
            if not isinstance(duration, (int, float)) or not 2 <= duration <= 15 or count >= 3:
                continue
            if kind == 'audio':
                if audio_seconds + duration > 15:
                    continue
                audio_seconds += duration
            else:
                if video_seconds + duration > 15:
                    continue
                video_seconds += duration
        else:
            continue
        seen.add(item['path']);output.append(deepcopy(item))
    # Ref2VA requires a visual reference alongside audio.
    if not any(r['kind'] in ('image', 'video') for r in output):
        output = [r for r in output if r['kind'] != 'audio']
    # Exact boundary control belongs to FL2VA and cannot consume audio/video refs.
    if output and all(r['kind'] == 'image' for r in output) and (first_frame_id or last_frame_id):
        anchors = []
        for ident, index in ((first_frame_id, 0), (last_frame_id, -1)):
            item = by_id.get(ident)
            if item and item['kind'] == 'image':
                anchors.append({**deepcopy(item), 'frame_index': index})
        if anchors:
            return anchors
    return output


def reference_labels(references):
    counts, result = {}, []
    for item in references:
        kind = item['kind'];counts[kind] = counts.get(kind, 0) + 1
        result.append({**deepcopy(item), 'h3_label': f'<{dict(image="Picture", audio="Audio", video="Video")[kind]} {counts[kind]}>'})
    return result
