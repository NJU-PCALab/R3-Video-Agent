"""Audio generation and imported media normalization; no synthetic receipts."""
from pathlib import Path
import shutil

from . import media_ports
from .assets import generator
from .audio_parameters import AudioParameterError, VERSION, tts_language
from .storage import Unavailable, key, read, sha, write
from .worker_wait import positive
from . import asset_lifecycle, asset_permissions


class AudioAssetMixin:
    def audio_info(self, path):
        import json
        value = json.loads(self.command([self.ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]))
        tracks = [s for s in value.get('streams', []) if s.get('codec_type') == 'audio']
        if not tracks:
            raise Unavailable('No actual audio stream: ' + str(path))
        duration = float(value.get('format', {}).get('duration') or tracks[0].get('duration') or 0)
        return {'seconds': duration, 'sample_rate': int(tracks[0]['sample_rate']), 'channels': tracks[0]['channels']}

    def audio(self, name, spec, reference=None, seed=42):
        folder = self.store.run / 'media/audio_assets' / name
        cached = read(folder / 'receipt.json')
        if cached:
            self.audio_info(cached['path'])
            return cached
        # Cached retained media above is reusable; every new physical request is checked.
        asset_permissions.assert_generation_allowed(self.config, self.store, spec, reference)
        spec = asset_lifecycle.calibration_spec(spec,self.config)
        provider = self.config.get('audio_provider', {})
        mode = generator(spec)
        language = tts_language(spec.get('language')) if mode in ('voice_design', 'voice_clone') else spec.get('language', 'Chinese')
        adapter = self.config.get('media_adapters', {}).get('audio')
        model = adapter['model'] if adapter else provider.get('models', {}).get(mode)
        if not model or ('audio' not in self.config.get('media_adapters', {}) and not Path(model).is_dir()):
            raise Unavailable('Audio generator is not deployed: ' + mode)
        if mode == 'voice_clone' and (not reference or reference.get('track_type') not in ('synthetic_voice', 'isolated_voice')):
            raise Unavailable('Voice clone requires a registered isolated voice; mixed native soundtrack is unsuitable')
        text = spec.get('text') or '夜色渐深，风吹过窗前。我会把事情的原委，清清楚楚地告诉你。'
        manifest = Path(self.config['lite_root']) / 'model_manifests' / (mode + '.json')
        model_origin = {} if adapter else read(manifest, {})
        request = {'generator': mode, 'prompt': spec.get('prompt') or spec.get('description', ''),
                   'text': text, 'language': language, 'requested_language': spec.get('language'),
                   'language_adapter': VERSION, 'seed': seed,
                   'seconds': min(180, positive(spec.get('seconds'), 15)), 'model_path': model,
                   'music_source': None if adapter else provider.get('music_source'), 'reference': reference,
                   'model_revision': model_origin.get('revision'),
                   'model_code_revision': model_origin.get('runtime_code_revision'),
                   'model_manifest_sha256': sha(manifest) if not adapter and manifest.is_file() else None}
        if spec.get('calibration_policy'):
            request.update(scene_text=spec.get('scene_text'),calibration_policy=spec['calibration_policy'],calibration_purpose=spec['calibration_purpose'])
        command = [provider['python'], str(Path(self.config['lite_root']) / 'tools/audio_worker.py'),
                   '--request', str(folder / 'request.json'), '--response', str(folder / 'worker_response.json'),
                   '--lock-file', provider['lock_file']]
        asset_permissions.assert_generation_allowed(self.config, self.store, spec, reference)
        try:
            result = self._worker(folder, request, command, counter='audio_candidates', amount=1,
                                  response_name='worker_response.json', device=provider.get('device', '1'))
        except Unavailable as exc:
            failure = read(folder / 'worker_response.json', {})
            if failure.get('error_type') == 'invalid_audio_parameter':
                raise AudioParameterError(str(exc)) from exc
            raise
        full = Path(result['path']);info = self.audio_info(full)
        if info['seconds'] < 2:
            raise Unavailable('Generated audio too short for an H3 reference; original waveform retained')
        voice_activity = None
        if asset_lifecycle.enabled(self.config) and spec.get('role')=='voice':
            if info['seconds']>15:
                raise Unavailable('Calibration utterance exceeds 15 seconds; preserve whole original and revise text instead of cutting speech')
            observation=self.transcribe(full,'voice_calibration_'+name)
            voiced=asset_lifecycle.voiced_seconds(observation,sha(full),info['seconds'])
            voice_activity={'observation':observation,'voiced_seconds':voiced,'source_sha256':sha(full),'target_text_supplied_to_asr':False}
        output = folder / 'reference.wav'
        duration = min(15, info['seconds'])
        self.command([self.ffmpeg, '-y', '-v', 'error', '-i', full, '-t', str(duration),
                      '-map', '0:a:0', '-c:a', 'pcm_s16le', '-ar', '32000', '-ac', '2', output])
        receipt = {**self.audio_info(output), 'kind': 'audio', 'path': str(output), 'sha256': sha(output),
                   'generator': mode, 'text': text if mode != 'music' and info['seconds'] <= 15 else None,
                   'full_text': text if mode != 'music' else None,
                   'track_type': 'synthetic_voice' if mode != 'music' else 'generated_music',
                   'full_path': str(full), 'full_sha256': sha(full),
                   'provenance': {'model_path': model, 'worker_response': str(folder / 'worker_response.json'),
                     'model_manifest': None if adapter else provider.get('manifest'), 'reference_sha256': (reference or {}).get('sha256'),
                     'model_revision': request['model_revision'],
                     'model_code_revision': request['model_code_revision'],
                     'model_manifest_sha256': request['model_manifest_sha256'],
                     'requested_language': request['requested_language'], 'effective_language': language,
                     'language_adapter': VERSION,
                     'free_form_prompt_consumed': mode != 'voice_clone',
                     'calibration_policy':spec.get('calibration_policy'),'scene_text_not_spoken_as_extra_dialogue':spec.get('scene_text'),'voice_activity':voice_activity}}
        write(folder / 'receipt.json', receipt)
        return receipt

    def import_asset(self, name, spec):
        folder = self.store.run / 'media/imports' / name
        cached = read(folder / 'receipt.json')
        if cached:
            if not Path(cached['path']).is_file():
                raise Unavailable('Original imported asset missing')
            return cached
        source = Path(spec.get('path', ''))
        if not source.is_file():
            raise Unavailable('Import needs an actual local media file, not a model-invented path')
        folder.mkdir(parents=True, exist_ok=True)
        kind = spec.get('kind', 'image')
        if kind == 'image':
            from PIL import Image
            output = folder / 'reference.png'
            with Image.open(source) as image:
                image.convert('RGB').save(output)
            receipt = {'kind': kind}
        elif kind == 'audio':
            info = self.audio_info(source)
            if info['seconds'] < 2:
                raise Unavailable('Imported audio is shorter than 2 seconds')
            output = folder / 'reference.wav'
            start = positive(spec.get('start_s'), 0)
            duration = min(15, positive(spec.get('seconds'), info['seconds']))
            self.command([self.ffmpeg, '-y', '-v', 'error', '-ss', str(start), '-i', source,
                          '-t', str(duration),
                          '-c:a', 'pcm_s16le', '-ar', '32000', '-ac', '2', output])
            whole = start == 0 and duration >= info['seconds'] - .001
            receipt = {**self.audio_info(output), 'kind': kind, 'track_type': 'unverified_import',
                       'text': spec.get('text') if whole else None, 'full_text': spec.get('text')}
        elif kind == 'video':
            info = self.probe(source)
            if info['seconds'] < 2:
                raise Unavailable('Imported video is shorter than 2 seconds')
            output = folder / 'reference.mp4'
            audio = ['-c:a', 'aac'] if spec.get('include_audio') and info['audio_streams'] else ['-an']
            self.command([self.ffmpeg, '-y', '-v', 'error', '-ss', str(positive(spec.get('start_s'), 0)), '-i', source,
                          '-t', str(min(15, positive(spec.get('seconds'), info['seconds']))),
                          '-vf', 'fps=24', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', *audio, output])
            receipt = {**self.probe(output), 'kind': kind, 'include_audio': bool(spec.get('include_audio') and info['audio_streams'])}
        else:
            raise Unavailable('Unsupported imported modality: ' + str(kind))
        receipt.update(path=str(output), sha256=sha(output), generator='import',
                       provenance={'source_path': str(source), 'source_sha256': sha(source)})
        write(folder / 'receipt.json', receipt)
        return receipt

    def reference_excerpt(self, asset, *, duration, start=0):
        total = asset['seconds']
        if total < 2:
            raise Unavailable('Reference has less than 2 seconds of actual media')
        start = min(max(0, positive(start, 0)), max(0, total - 2))
        duration = min(15, max(2, positive(duration, 5)), total - start)
        if start == 0 and total <= duration + .001:
            return asset
        try:
            if sha(asset['path']) != asset['sha256']:
                raise Unavailable('Reference source bytes changed before excerpt selection')
        except OSError as exc:
            raise Unavailable('Reference source is unavailable for excerpt selection') from exc
        ident = key({'sha': asset['sha256'], 'start': start, 'duration': duration, 'kind': asset['kind']})
        result = self.import_asset('window_' + ident, {'kind': asset['kind'], 'path': asset['path'],
                 'start_s': start, 'seconds': duration, 'include_audio': asset.get('include_audio', False)})
        try:
            intact = sha(result['path']) == result['sha256']
        except (OSError, KeyError):
            intact = False
        if not intact or result.get('provenance', {}).get('source_sha256') != asset['sha256']:
            raise Unavailable('Reference excerpt receipt does not match its source or output bytes')
        excerpt = {**asset, **result, 'generator': asset.get('generator'), 'track_type': asset.get('track_type'),
                   'text': None, 'reference_window': {'source_sha256': asset['sha256'], 'start_s': start,
                     'seconds': result['seconds'], 'source_path': asset['path']}}
        # The import receipt records the crop; the reference retains the asset's origin.
        if 'provenance' in asset:
            excerpt['provenance'] = asset['provenance']
        if asset['kind'] == 'audio':
            excerpt['full_text'] = asset.get('full_text', asset.get('text'))
        excerpt.pop('canonical_voice_authority', None)
        excerpt.pop('sound_reference_authority', None)
        return excerpt

    def continuity_assets(self, clip_id, receipt):
        folder = self.store.run / 'media/continuity' / clip_id;folder.mkdir(parents=True, exist_ok=True)
        last = folder / 'last_frame.png'
        if not last.exists():
            self.command([self.ffmpeg, '-y', '-v', 'error', '-sseof', '-0.15', '-i', receipt['path'], '-frames:v', '1', last])
        image = {'id': 'last_' + clip_id, 'name': clip_id + ' actual last frame', 'kind': 'image', 'role': 'continuity',
                 'path': str(last), 'sha256': sha(last), 'generator': 'derive', 'version': 1,
                 'provenance': {'source_media_sha256': receipt['sha256']}}
        video = self.import_asset('tail_' + clip_id, {'kind': 'video', 'path': receipt['path'], 'include_audio': False,
                                  'start_s': max(0, receipt['seconds'] - 3), 'seconds': 3})
        video.update(id='motion_' + clip_id, name=clip_id + ' actual motion reference', role='continuity', version=1)
        return [image, video]
