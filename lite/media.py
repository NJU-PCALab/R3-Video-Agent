"""Thin adapters over the existing real workers and their shared GPU lock."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from . import media_ports
from .storage import Pending, Stopped, RuntimeUnavailable, Unavailable, operation_recorded, read, seconds, sha, write
from .worker_wait import observe_wait, process_identity, reconcile_launch, stop_owned_worker
from .audio_assets import AudioAssetMixin
from . import binding_contract, speech_contract, asset_lifecycle, finalization
from .assets import reference_labels


def qwen_edit_seed(run_id, operation, seed, parent_hashes):
    """Generate reproducible edit noise independently of the source portrait."""
    payload = json.dumps(['qwen-image-edit-v1', run_id, operation, seed, list(parent_hashes)],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return (1 << 62) | (int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 62) - 1))

def sampling_size(profile, aspect_ratio):
    width, height = profile.export_width, profile.export_height
    if aspect_ratio == '16:9':
        return width, height
    if aspect_ratio != '9:16':
        raise ValueError('Unsupported film aspect ratio: ' + str(aspect_ratio))
    # H3 uses multiples of 32 and its reference protocol specifies a 768 short edge.
    width, height = sorted((width, height))
    if width != 768 or height <= width or height % 32:
        raise ValueError('Portrait H3 requires a 768-pixel short edge and a longer side on the 32-pixel grid')
    return width, height


def portrait_crop(width, height):
    if width >= height or min(width // 18, height // 32) == 0:
        raise Unavailable('Native H3 shot has no usable portrait canvas')
    unit = min(width // 18, height // 32)
    return 18 * unit, 32 * unit


def verify_viewing_geometry(receipt, config):
    aspect = config.get('video_aspect_ratio')
    if aspect == '9:16' or config.get('exact_landscape_canvas') and aspect == '16:9':
        w, h = map(int, aspect.split(':'))
        if receipt['width'] * h != receipt['height'] * w or receipt.get('sample_aspect_ratio') not in ('1:1', '1/1'):
            raise Unavailable(f'Presentation requires square pixels and an exact {aspect} canvas')


def video_request(caption, duration, references, seed, aspect_ratio='16:9'):
    # References have already been selected from the real asset catalog.
    keyframes = references and all(r['kind'] == 'image' and r.get('frame_index') is not None for r in references)
    mode = 'fl2va' if keyframes else ('ref2va' if references else 't2va')
    return {'task': mode, 'prompt': caption, 'seed': seed,
               'conditions': [{'type': 'video_audio' if r['kind'] == 'video' and r.get('include_audio') else r['kind'],
                   'uri': Path(r['path']).as_uri(), 'role': 'reference',
                   'frame_index': r.get('frame_index') if mode == 'fl2va' else None} for r in references],
               'target': {'short_edge': 768, 'aspect_ratio': aspect_ratio, 'duration_seconds': duration}}


class Media(AudioAssetMixin):
    def __init__(self, store, config):
        self.store, self.config = store, config
        self.root = Path(config['production_root'])
        self.ffmpeg = str(Path(config['media_bin']) / 'ffmpeg')
        self.ffprobe = str(Path(config['media_bin']) / 'ffprobe')
        sys.path.insert(0, str(self.root))
        from lib.h3_av.config import load_config
        self.h3 = load_config(config['h3_config'])

    def command(self, argv):
        try:
            result = subprocess.run([str(a) for a in argv], capture_output=True, timeout=1200)
        except subprocess.TimeoutExpired as exc:
            self.store.event('media_command_failed', reason='timeout', timeout_seconds=1200,
                             command=[str(a) for a in argv])
            raise Unavailable('Media command exceeded 1200 seconds; original inputs retained') from exc
        if result.returncode:
            raise Unavailable('Media command failed: ' + result.stderr.decode(errors='replace')[-2000:])
        return result.stdout

    def probe(self, path):
        path = Path(path)
        if not path.is_file() or not path.stat().st_size:
            raise Unavailable('Real media file missing: ' + str(path))
        value = json.loads(self.command([self.ffprobe, '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', path]))
        videos = [s for s in value['streams'] if s['codec_type'] == 'video']
        audios = [s for s in value['streams'] if s['codec_type'] == 'audio']
        v = videos[0] if videos else {}
        frames = int(v.get('nb_read_frames') or v.get('nb_frames') or 0)
        fps = float(Fraction(v.get('avg_frame_rate') or '0/1'))
        return {'path': str(path), 'sha256': sha(path), 'frames': frames, 'fps': fps,
                'seconds': frames / fps if fps else float(value.get('format', {}).get('duration', 0)),
                'video_streams': len(videos), 'audio_streams': len(audios),
                'width': v.get('width'), 'height': v.get('height'),
                'sample_aspect_ratio': v.get('sample_aspect_ratio')}

    def _worker(self, folder, request, command, *, counter, amount, response_name, device=None):
        return self._worker_impl(folder,request,command,counter=counter,amount=amount,response_name=response_name,device=device)

    def _retained_request(self, folder):
        try:
            return read(folder / 'request.json', required=operation_recorded(
                folder, self.store.data['accounted_operations']))
        except FileNotFoundError:
            raise Pending('Original worker request is unavailable; retain original operation and reconcile input',
                status='needs_input_reconciliation') from None

    def _worker_impl(self, folder, request, command, *, counter, amount, response_name, device=None):
        kind = media_ports.KINDS.get(counter)
        remote = media_ports.command(self.config, kind) if kind else None
        if remote:
            request = {**request, 'provider_model': self.config['media_adapters'][kind]['model']}
            command = [*remote, '--request', str(folder / 'request.json'),
                       '--response', str(folder / response_name), '--output-dir', str(folder)]
        child = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            retained = self._retained_request(folder)
            if asset_lifecycle.enabled(self.config) and retained is not None and retained!=request:
                raise Pending('Original worker request differs; preserve original result and reconcile instead of reusing or redispatching',status='needs_worker_reconciliation')
            child = None
            response_path = folder / response_name
            def ready():
                result = read(response_path)
                return result is not None and (response_name != 'worker_status.json' or result.get('status') in ('completed', 'failed'))
            if retained is None:
                write(folder / 'request.json', request)
            if not ready():
                failure = read(folder / 'worker_failure.json')
                if failure:
                    raise RuntimeUnavailable('Original worker has a definitive retained failure: ' + str(folder))
                process = read(folder / 'process.json')
                if process is None:
                    if (folder / 'launch_failure.json').exists() or (folder / 'worker_failure.json').exists():
                        raise RuntimeUnavailable('Recorded worker failure; original attempt retained: ' + str(folder))
                    if operation_recorded(folder, self.store.data['accounted_operations']):
                        submitted = read(folder / 'submitted.json')
                        if submitted is None:
                            raise Pending('Original worker submission snapshot is unavailable; no redispatch',
                                status='needs_worker_reconciliation')
                        process = reconcile_launch(folder)
                if process is None:
                    self.store.check_stop()
                    if not remote and counter not in ('cpu_asr_runs','cpu_vad_runs') and self.config.get('gpu_resource_admission') in ('gpu1_idle_v1','assigned_gpu_idle_v1'):
                        if asset_lifecycle.enabled(self.config):
                            asset_lifecycle.await_gpu(self.store,self.config,folder,lambda:binding_contract.admit_gpu(self.store,device if device is not None else self.config['image_provider']['device'],self.config))
                        else:
                            binding_contract.admit_gpu(self.store, device if device is not None else self.config['image_provider']['device'],self.config)
                    self.store.count_once(str(folder), **{counter: amount})
                    write(folder / 'submitted.json', {'time': time.time(), 'command': command})
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(device if device is not None else self.config['image_provider'].get('device', '0')),
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                               FFMPEG_BINARY=self.ffmpeg)
                    with (folder / 'worker.log').open('ab') as log:
                        try:
                            child = subprocess.Popen(command, stdout=log, stderr=log, env=env,
                                                     stdin=subprocess.DEVNULL, start_new_session=True)
                        except OSError as exc:
                            write(folder / 'launch_failure.json', {'type': type(exc).__name__, 'definitive': True})
                            raise RuntimeUnavailable('Worker could not launch: ' + str(folder)) from None
                    process = process_identity(child.pid) or {'pid': child.pid, 'argv': command}
                    write(folder / 'process.json', process)
                    self.store.event('worker_started', folder=str(folder), process=process)
                while not ready():
                    self.store.check_stop()
                    current = process_identity(process['pid'])
                    if current is None:
                        # A successful response may arrive during the last poll.
                        if ready():
                            break
                        outcome = stop_owned_worker(folder, process, ready=ready, grace=.1,
                                                    check_stop=self.store.check_stop)
                        if child is not None:
                            child.poll()
                        if ready():
                            break
                        if outcome['outcome'] != 'stopped':
                            raise Pending('Original worker exited but its process group is unresolved: ' + str(folder))
                        if remote:
                            raise Pending('Remote worker exited without a terminal receipt: ' + str(folder))
                        status = read(folder / 'worker_status.json', {})
                        write(folder / 'worker_failure.json', {'definitive': True, 'status': status})
                        raise RuntimeUnavailable('Original worker exited without success; retained logs: ' + str(folder))
                    if process.get('start_ticks') and current.get('start_ticks') != process['start_ticks']:
                        raise Pending('Original worker PID was reused; do not resubmit: ' + str(folder))
                    status = read(folder / 'worker_status.json', {})
                    if status.get('status') == 'failed':
                        raise RuntimeUnavailable('Original worker failed: ' + str(status.get('error'))[:1000])
                    wait = observe_wait(folder, process, status, self.config,
                                        kind='video' if counter == 'h3_seconds' else 'image')
                    if wait['phase_changed']:
                        self.store.event('worker_wait_phase', folder=str(folder), phase=wait['phase'])
                    if wait['decision'] == 'queue_timeout':
                        self.store.event('worker_queue_wait_limit', folder=str(folder), wait=wait)
                        raise Pending('Original worker is still waiting for GPU resources: ' + str(folder), status='awaiting_resource')
                    if wait['decision'] in ('runtime_timeout', 'progress_timeout'):
                        policy = self.config.get('worker_wait_policy') or {}
                        grace = policy.get('cancellation_grace_seconds', 15) if isinstance(policy, dict) else 15
                        grace = min(30, max(.1, grace)) if isinstance(grace, (int, float)) else 15
                        outcome = stop_owned_worker(folder, process, ready=ready, grace=grace, check_stop=self.store.check_stop)
                        if child is not None:
                            child.poll()
                        write(folder / 'cancellation_receipt.json', {'wait': wait, 'outcome': outcome, 'time': time.time()})
                        if ready():
                            break
                        if outcome['outcome'] != 'stopped':
                            raise Pending('Worker timeout cancellation is unconfirmed; no new submission: ' + str(folder))
                        if remote:
                            raise Pending('Reconcile the remote job after local worker cancellation: ' + str(folder))
                        write(folder / 'worker_failure.json', {'definitive': True, 'reason': wait['decision'],
                              'cancelled_original_process': process, 'cancellation': outcome})
                        self.store.event('worker_timeout_confirmed', folder=str(folder), reason=wait['decision'])
                        raise RuntimeUnavailable('Original worker timed out and is confirmed stopped; attempt remains consumed: ' + str(folder))
                    time.sleep(3)
            if child is not None:
                child.poll()
            result = read(response_path, required=True)
            if asset_lifecycle.enabled(self.config):
                asset_lifecycle.await_exit(self.store,self.config,folder)
                if child is not None:child.poll()
            if remote and result.get('needs_reconciliation') is True:
                raise Pending('Remote submission awaits reconciliation: ' + str(folder))
            if result.get('ok') is False or result.get('status') == 'failed':
                raise RuntimeUnavailable('Original worker returned failure: ' + str(result.get('error'))[:1000])
            return result
        except Stopped:
            if asset_lifecycle.enabled(self.config):asset_lifecycle.cleanup_user_stop(self.store,folder)
            raise
        finally:
            if child is not None:child.poll()

    def image(self, name, prompt, parents=(), seed=42, *, age_edit=None):
        folder = self.store.run / 'media' / 'images' / name
        cached = read(folder / 'receipt.json')
        if cached:
            if not Path(cached['path']).is_file():
                raise Unavailable('Previously generated image missing; do not silently regenerate')
            return cached
        provider = self.config['image_provider']
        image_adapter = self.config.get('media_adapters', {}).get('image')
        width, height = (768, 1344) if self.config.get('video_aspect_ratio') == '9:16' else (1344, 768)
        parent_images = [{'path': p['path']} for p in parents[:10]]
        if parent_images and (provider['model'] == 'Qwen-Image-2.1' or 'image' in self.config.get('media_adapters', {})):
            for binding, parent in zip(parent_images, parents):
                binding['sha256'] = parent.get('sha256') or sha(Path(binding['path']))
            # Replaying text-to-image noise in an edit can sharpen the parent and ignore the edit.
            seed = qwen_edit_seed(self.store.run.name, name, seed, (p['sha256'] for p in parent_images))
        request = {'prompt': prompt, 'seed': seed, 'width': width, 'height': height,
                   'parameters': deepcopy(provider.get('parameters', {})),
                   'parent_images': parent_images, 'model': image_adapter['model'] if image_adapter else provider['model']}
        if age_edit is not None:
            if len(parents) != 1 or (provider['model'] != 'Qwen-Image-2.1' and 'image' not in self.config.get('media_adapters', {})):
                raise Unavailable('A Qwen character age edit requires one canonical parent')
            request['character_age_edit'] = age_edit
        response_path = folder / 'worker_response.json'
        command = [provider['python'], str(self.root / 'scripts/avh_model_worker.py'), '--kind', 'image',
                   '--model-path', provider['model_path'], '--request', str(folder / 'request.json'),
                   '--response', str(response_path), '--device', str(provider.get('device', '0'))]
        result = self._worker(folder, request, command, counter='image_candidates', amount=1, response_name='worker_response.json')
        path = Path(result['images'][0]['path'])
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        receipt = {'path': str(path), 'sha256': sha(path), 'kind': 'image', 'native_worker': str(response_path)}
        if age_edit is not None:
            condition = result.get('character_age_condition') or {}
            source = parent_images[0]['sha256']
            if (condition.get('age_edit') != age_edit or condition.get('source_sha256') != source
                    or result.get('parent_hashes') != [source]):
                raise RuntimeUnavailable('Qwen worker did not verify the character age conditioning')
            receipt['character_age_condition'] = condition
        write(folder / 'receipt.json', receipt)
        return receipt

    def render(self, name, caption, duration, references, seed, dialogue_source=None, binding=None):
        folder = self.store.run / 'media' / 'videos' / name
        if binding_contract.enabled(self.config):
            original=self._retained_request(folder)
            original_binding=read(folder/'cast_binding_contract.json')
            submitted = operation_recorded(folder, self.store.data['accounted_operations'])
            allow_legacy=(submitted and original is not None and original.get('prompt')==caption and original_binding==binding)
            binding_contract.validate(caption, dialogue_source, references, self.config, binding,allow_original_legacy=allow_legacy)
        if dialogue_source is not None:
            from .evidence_policy import dialogue_texts, normal, scripted
            if dialogue_texts(caption) != [normal(x['text']) for x in scripted(dialogue_source)]:
                raise Unavailable('Final submitted dialogue does not match current script occurrences')
            retained = self._retained_request(folder)
            if retained and retained.get('prompt') != caption:
                raise Pending('Original H3 prompt differs; reconcile original request, do not replace it')
        cached = read(folder / 'receipt.json')
        if cached:
            verify_viewing_geometry(self.probe(cached['path']), self.config)
            return cached
        duration = seconds(duration)
        aspect_ratio = self.config.get('video_aspect_ratio', '16:9')
        request = video_request(caption, duration, references, seed, aspect_ratio)
        mode = request['task']
        cfg = self.h3.effective_config(mode)
        width, height = sampling_size(cfg, aspect_ratio)
        if binding is not None and not (folder / 'cast_binding_contract.json').exists():
            write(folder / 'cast_binding_contract.json', binding)
        if not (folder / 'reference_bindings.json').exists():
            write(folder / 'reference_bindings.json', reference_labels(references))
        command = [cfg.diffsynth_python, str(self.root / 'scripts/h3_diffsynth_worker.py'),
                   '--request', str(folder / 'request.json'), '--output', str(folder / 'native.mp4'),
                   '--project', cfg.diffsynth_project, '--weights', cfg.ref2va_weights if mode == 'ref2va' else cfg.fl2va_weights,
                   '--width', str(width), '--height', str(height), '--steps', str(cfg.inference_steps),
                   '--acceleration-profile', cfg.acceleration_profile, '--attention-backend', cfg.attention_backend,
                   '--vram-reserve-gb', str(cfg.vram_reserve_gb),
                   '--lock-file', str(self.root / 'service_logs' / f'gpu_{cfg.cuda_device}.lock')]
        if cfg.acceleration_profile == 'pdd8_exact':
            command += ['--pdd-lora-path', cfg.pdd_lora_path, '--pdd-lora-sha256', cfg.pdd_lora_sha256,
                        '--pdd-base-variant', cfg.pdd_base_variant]
        self._worker(folder, request, command, counter='h3_seconds', amount=duration, response_name='worker_status.json')
        return self._finish_video(folder, duration)

    def _finish_video(self, folder, duration):
        source = folder / 'native.mp4'
        info = self.probe(source)
        source_speech = None
        target_duration = duration
        if speech_contract.enabled(getattr(self, 'config', {})):
            source_speech = self.transcribe(source, folder.name + '_native')
            duration = speech_contract.export_duration(duration, info['seconds'], source_speech)
        frames = round(duration * 24)
        if info['frames'] < frames or not info['audio_streams']:
            raise Unavailable('H3 did not return enough real frames with native audio')
        output, audio = folder / 'viewing.mp4', folder / 'native_audio.wav'
        video_filter = 'fps=24'
        if getattr(self, 'config', {}).get('video_aspect_ratio') == '9:16':
            # The 768x1344 sample needs six pixels removed from each side.
            width, height = portrait_crop(info['width'], info['height'])
            video_filter += f',crop={width}:{height},setsar=1'
        elif getattr(self, 'config', {}).get('exact_landscape_canvas'):
            unit = min(info['width'] // 32, info['height'] // 18)
            if unit <= 0:
                raise Unavailable('Native H3 shot has no usable landscape canvas')
            video_filter += f',crop={32 * unit}:{18 * unit},setsar=1'
        self.command([self.ffmpeg, '-y', '-v', 'error', '-i', source, '-t', str(duration),
                      '-map', '0:v:0', '-map', '0:a:0', '-vf', video_filter, '-c:v', 'libx264', '-crf', '18',
                      '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '32000', '-ac', '2', '-movflags', '+faststart', output])
        self.command([self.ffmpeg, '-y', '-v', 'error', '-i', source, '-t', str(duration),
                      '-map', '0:a:0', '-c:a', 'pcm_s16le', '-ar', '32000', '-ac', '2', audio])
        receipt = self.probe(output)
        verify_viewing_geometry(receipt, getattr(self, 'config', {}))
        if receipt['frames'] != frames or receipt['fps'] != 24 or receipt['audio_streams'] != 1:
            raise Unavailable('Viewing export failed basic frame/fps/native-track check')
        receipt.update(kind='video', source_path=str(source), source_sha256=sha(source),
                       audio_path=str(audio), audio_sha256=sha(audio), native_audio=True)
        if source_speech is not None:
            receipt.update(target_seconds=target_duration, export_policy='preserve_detected_sentence_tail_v1', native_blind_speech=source_speech)
        write(folder / 'receipt.json', receipt)
        return receipt

    def transcribe(self, path, name, *, detect_language=False):
        folder = self.store.run / 'media/speech' / name
        folder.mkdir(parents=True, exist_ok=True)
        input_sha = sha(path)
        try:
            bound = read(folder / 'source_binding.json', required=operation_recorded(
                folder, self.store.data['accounted_operations']))
        except FileNotFoundError:
            raise Pending('Original blind ASR source binding is unavailable; no replacement',
                status='needs_input_reconciliation') from None
        if bound and bound['source_sha256'] != input_sha:
            raise Pending('Original blind ASR source changed; do not repeat request')
        if not bound:
            write(folder / 'source_binding.json', {'source_path':str(path),'source_sha256':input_sha})
        wav = folder / 'input_16k.wav'
        if not wav.exists():
            self.command([self.ffmpeg, '-v', 'error', '-y', '-i', path, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav])
        request = {'audio_path':str(wav),'audio_sha256':sha(wav),'model_path':self.config['speech_verification']['asr_model_path']}
        from . import task_parameters
        request = task_parameters.asr_request(request,self.config)
        if detect_language:
            request['language'] = 'auto'
        retained = self._retained_request(folder)
        if retained and retained != request:
            raise Pending('Original blind ASR request differs; retain original operation')
        command = [self.config['speech_verification']['asr_python'],str(Path(self.config['lite_root'])/'tools/speech_worker.py'),
                   '--request',str(folder/'request.json'),'--response',str(folder/'worker_response.json')]
        try:
            result = self._worker(folder,request,command,counter='cpu_asr_runs',amount=1,response_name='worker_response.json',device='')
        except Unavailable as exc:
            raise Pending('Original independent speech worker requires diagnosis, not a new H3 take: '+str(exc),status='needs_speech_runtime') from exc
        if result.get('audio_sha256') != request['audio_sha256']:
            raise Pending('Blind ASR result does not bind original audio')
        from . import vad_precision
        result = vad_precision.recheck(self, folder, request, result)
        return {**result,'source_audio_sha256':input_sha,'record':str(folder)}

    def native_audio_asset(self, clip_id, receipt):
        folder = self.store.run / 'media' / 'audio'
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / (clip_id + '.wav')
        if not output.exists():
            self.command([self.ffmpeg, '-y', '-v', 'error', '-i', receipt['audio_path'], '-t', '3',
                          '-c:a', 'pcm_s16le', '-ar', '32000', '-ac', '2', output])
        return {'id': 'audio_' + clip_id, 'name': clip_id + ' native mixed soundtrack excerpt',
                'kind': 'audio', 'seconds': self.audio_info(output)['seconds'], 'path': str(output), 'sha256': sha(output),
                'track_type': 'mixed_native', 'generator': 'derive', 'version': 1,
                'description': 'Actual mixed native soundtrack, not an isolated speaker or synthetic voice clone.'}

    def _assemble_longform(self, selections, output):
        """Mux the selected viewing tracks with one ordered native PCM stream."""
        # Every viewing export is already libx264, 24 fps and yuv420p. A concat
        # demuxer opens one MP4 at a time; per-file durations come from video
        # frames rather than container AAC padding. Reuse the encoded pictures.
        with TemporaryDirectory(prefix='.assembling-', dir=output.parent) as staging:
            folder = Path(staging)
            playlist = folder / 'shots.ffconcat'
            audio = folder / 'native.s16le'
            total_frames = 0
            padding = []
            with playlist.open('w', encoding='utf-8') as clips, audio.open('wb') as track:
                clips.write('ffconcat version 1.0\n')
                for selected in selections:
                    self.store.check_stop()
                    source = selected['receipt']
                    frames = source['frames']
                    path = str(Path(source['path']).resolve())
                    if type(frames) is not int or frames < 1 or '\n' in path or '\r' in path:
                        raise Unavailable('Invalid frame count or MP4 filename in selected shot')
                    escaped = path.replace("'", "'\\''")
                    clips.write(f"file '{escaped}'\nduration {frames / 24:.9f}\n")
                    first_sample = round(total_frames * 32000 / 24)
                    total_frames += frames
                    samples = round(total_frames * 32000 / 24) - first_sample
                    pcm = self.command([self.ffmpeg, '-nostdin', '-v', 'error', '-threads', '1',
                                        '-i', source['audio_path'], '-map', '0:a:0', '-af',
                                        f'aresample=32000,atrim=end_sample={samples}', '-ac', '2',
                                        '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'])
                    # Native export and video trim may differ by a fraction of
                    # one frame. Keep the next sound at its exact frame boundary.
                    missing = samples * 4 - len(pcm)
                    # _finish_video rounds the target to the nearest frame.
                    # Half a frame at 32 kHz plus one cumulative rounding sample.
                    if not pcm or missing < 0 or missing > 668 * 4 or missing % 4:
                        raise Unavailable('Selected native audio does not cover its video: ' + path)
                    if missing:
                        padding.append({'clip_id': selected['clip']['id'], 'samples_per_channel': missing // 4})
                    track.write(pcm)
                    track.write(bytes(missing))
            self.store.check_stop()
            movie = folder / 'assembled.mp4'
            self.command([self.ffmpeg, '-nostdin', '-y', '-v', 'error', '-f', 'concat', '-safe', '0',
                          '-i', playlist, '-f', 's16le', '-ar', '32000', '-ac', '2', '-i', audio,
                          '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                          '-threads:a', '1', '-movflags', '+faststart', movie])
            receipt = self.probe(movie)
            verify_viewing_geometry(receipt, getattr(self, 'config', {}))
            if receipt['fps'] != 24 or receipt['frames'] != total_frames or receipt['audio_streams'] != 1:
                raise Unavailable('Long-form assembly failed ordered frame/fps/native-track validation')
            os.replace(movie, output)
            receipt['path'] = str(output)
            receipt['native_pcm_alignment'] = {'sample_rate': 32000, 'boundary_padding': padding,
                                                'total_padded_samples_per_channel': sum(
                                                    entry['samples_per_channel'] for entry in padding)}
            return receipt

    def assemble(self, selections, *, partial=False):
        if not partial and not getattr(self, 'config', {}).get('selective_repair'):
            retained = finalization.retain_assembly(self.store.run, selections)
            if retained is not None:
                if getattr(self, 'config', {}).get('video_aspect_ratio') == '9:16':
                    verify_viewing_geometry(self.probe(retained['path']), self.config)
                return retained
        folder = self.store.run / ('partial_delivery' if partial else 'delivery')
        folder.mkdir(exist_ok=True)
        listing = folder / 'sources.txt'
        listing.write_text(''.join("file '" + str(Path(s['receipt']['path']).resolve()).replace("'", "'\\''") + "'\n" for s in selections))
        output = folder / ('partial.mp4' if partial else 'final.mp4')
        if getattr(self, 'config', {}).get('task_contract', {}).get('version') == 'longform_task_parameters_v1':
            receipt = self._assemble_longform(selections, output)
        else:
            receipt = self._assemble_filtered(selections, output)
        receipt.update(native_audio=True, production_complete=not partial, whole_film_review='not_requested', whole_film_semantic_score=None,
                       source_clips=[{'clip_id': s['clip']['id'], 'path': s['receipt']['path'], 'sha256': s['receipt']['sha256']} for s in selections])
        write(folder / 'technical.json', receipt)
        return receipt

    def _assemble_filtered(self, selections, output):
        # Container AAC padding must not advance the next clip's video clock.
        # Join actual frame ranges and the separately retained native PCM tracks.
        inputs, filters, pairs = [], [], []
        for index, selected in enumerate(selections):
            source = selected['receipt']
            frames = source['frames']
            inputs += ['-i', source['path'], '-i', source['audio_path']]
            filters += [f'[{index * 2}:v:0]trim=end_frame={frames},setpts=N/(24*TB)[v{index}]',
                        f'[{index * 2 + 1}:a:0]atrim=end_sample={round(frames * 32000 / 24)},asetpts=PTS-STARTPTS[a{index}]']
            pairs.append(f'[v{index}][a{index}]')
        filters.append(''.join(pairs) + f'concat=n={len(selections)}:v=1:a=1[v][a]')
        self.command([self.ffmpeg, '-y', '-v', 'error', *inputs, '-filter_complex', ';'.join(filters),
                      '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                      '-r', '24', '-frames:v', str(sum(s['receipt']['frames'] for s in selections)),
                      '-c:a', 'aac', '-ar', '32000', '-ac', '2', '-movflags', '+faststart', output])
        receipt = self.probe(output)
        verify_viewing_geometry(receipt, getattr(self, 'config', {}))
        if receipt['fps'] != 24 or receipt['frames'] != sum(s['receipt']['frames'] for s in selections) or receipt['audio_streams'] != 1:
            raise Unavailable('Final file failed ordered-source frame/fps/native-track validation')
        return receipt
