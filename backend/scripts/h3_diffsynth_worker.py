"""One native H3 job in the pre-existing GPU environment; no environment mutation."""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
import time
import hashlib
import statistics
from pathlib import Path
from urllib.parse import urlparse, unquote


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    for name in ('request', 'output', 'project', 'weights'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--width', type=int, default=1344)
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--lock-file')
    p.add_argument('--acceleration-profile', choices=('native', 'adaln_exact', 'pdd8_exact'), default='native')
    p.add_argument('--pdd-lora-path')
    p.add_argument('--pdd-lora-sha256')
    p.add_argument('--pdd-base-variant', choices=('ref2va', 'fl2va'), default='ref2va')
    p.add_argument('--attention-backend', choices=('auto', 'flash_attn', 'flash_attn_3', 'torch_sdpa'), default='auto')
    p.add_argument('--vram-reserve-gb', type=float, default=5)
    p.add_argument('--vram-limit-gb', type=float)
    a = p.parse_args()
    if a.vram_reserve_gb < 4 or (a.vram_limit_gb is not None and a.vram_limit_gb <= 0):
        p.error('Keep at least 4 GiB of activation headroom; explicit VRAM limit must be positive')
    output = Path(a.output)
    status_path = output.parent / 'worker_status.json'
    lock = None
    try:
        request = json.loads(Path(a.request).read_text())
        if a.acceleration_profile == 'pdd8_exact':
            expected_variant = {'ref2va': 'ref2va', 't2va': 'fl2va', 'fl2va': 'fl2va'}.get(request['task'])
            if expected_variant != a.pdd_base_variant or a.steps != 8:
                raise ValueError('PDD adapter partition must match the H3 task at exactly 8 evaluations')
            if not a.pdd_lora_path or not a.pdd_lora_sha256:
                raise ValueError('PDD profile requires a local adapter and expected SHA256')
        if a.lock_file:
            import fcntl
            lock_path = Path(a.lock_file)
            lock_path.parent.mkdir(parents=True,exist_ok=True)
            lock = lock_path.open('a')
            save(status_path,{'status':'queued','reason':'waiting for shared GPU pool'})
            fcntl.flock(lock,fcntl.LOCK_EX)
        save(status_path,{'status':'running'})
        started = time.perf_counter()
        project = Path(a.project)
        sys.path.insert(0, str(project))
        import math
        import torch
        import numpy as np
        from PIL import Image
        if a.attention_backend != 'auto':
            os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION'] = {'flash_attn': 'flash_attention_2',
                'flash_attn_3': 'flash_attention_3', 'torch_sdpa': 'torch'}[a.attention_backend]
        from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
        from diffsynth.core.attention import attention as attention_module
        from diffsynth.utils.data.audio_video import write_video_audio
        if a.attention_backend == 'flash_attn' and not attention_module.FLASH_ATTN_2_AVAILABLE:
            raise RuntimeError('Explicit FlashAttention requested but unavailable; refusing silent fallback')
        if a.attention_backend == 'flash_attn_3' and not attention_module.FLASH_ATTN_3_AVAILABLE:
            raise RuntimeError('Explicit FlashAttention 3 requested but unavailable; refusing silent fallback')
        torch.set_num_threads(8)
        root = Path(a.weights)
        parts = {'dit': list(sorted(str(x) for x in (root/'transformer').glob('model-*.safetensors'))),
                 'text': list(sorted(str(x) for x in (root/'text_encoder').glob('model-*.safetensors'))),
                 'video': str(root/'video_vae/source/model.safetensors'),
                 'audio': str(root/'audio_vae/model.safetensors')}
        for part in parts.values():
            files = part if isinstance(part, list) else [part]
            if not files or any(not Path(f).is_file() for f in files):
                raise FileNotFoundError(f'Incomplete weights: {files}')
        opts = dict(offload_dtype=torch.bfloat16, offload_device='cpu', onload_dtype=torch.bfloat16,
                    onload_device='cpu', preparing_dtype=torch.bfloat16, preparing_device='cuda:0',
                    computation_dtype=torch.bfloat16, computation_device='cuda:0')
        vram_limit = torch.cuda.get_device_properties(0).total_memory/2**30 - a.vram_reserve_gb
        if a.vram_limit_gb is not None:
            if a.vram_limit_gb > vram_limit:
                raise ValueError('Explicit VRAM limit leaves insufficient activation headroom')
            vram_limit = a.vram_limit_gb
        torch.cuda.reset_peak_memory_stats()
        pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda:0',
            model_configs=[ModelConfig(path=value, **opts) for value in parts.values()],
            processor_config=ModelConfig(path=str(root/'processor')), redirect_common_files=False,
            vram_limit=vram_limit)
        pipe.requires_grad_(False)
        pipe.eval()
        pdd = None
        if a.acceleration_profile == 'pdd8_exact':
            from h3_pdd import install_pdd
            pdd = install_pdd(pipe, a.pdd_lora_path, base_weights=root/'transformer', base_variant=a.pdd_base_variant,
                              video_shift=12.0, audio_shift=3.0, expected_sha256=a.pdd_lora_sha256)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        kwargs, references, images, indices = {}, [], [], []
        for condition in request['conditions']:
            parsed = urlparse(condition['uri'])
            if parsed.scheme != 'file':
                raise ValueError('Direct DiffSynth backend requires same-host file assets')
            path = unquote(parsed.path)
            kind = condition['type']
            if kind == 'image':
                image = Image.open(path).convert('RGB')
                if request['task'] == 'fl2va':
                    images.append(image)
                    indices.append(condition['frame_index'])
                else:
                    references.append({'type': 'image', 'image': image})
            elif kind == 'audio':
                import soundfile as sf
                audio, sr = sf.read(path, always_2d=True)
                references.append({'type': 'audio', 'audio': torch.from_numpy(audio.T.copy()).float(), 'sample_rate': sr})
            elif kind in ('video', 'video_audio'):
                import imageio.v3 as iio
                # Host import guarantees 24 fps; the upstream pipeline does not resample.
                frames = [Image.fromarray(f) for f in iio.imiter(path)]
                item = {'type': kind, 'video': frames}
                if kind == 'video_audio':
                    import subprocess
                    raw = subprocess.check_output([os.environ.get('FFMPEG_BINARY','ffmpeg'), '-v','error','-i',path,
                        '-f','f32le','-ac','2','-ar','32000','-'])
                    item.update(audio=torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy().reshape(-1,2).T), sample_rate=32000)
                references.append(item)
        if images:
            kwargs.update(keyframes=images, keyframe_indices=indices)
        if references:
            kwargs['references'] = references
        frames = max(5, math.ceil((request['target']['duration_seconds']*24-5)/17)*17+5)
        metadata = {'backend': 'diffsynth', 'vendor': '7686e54d41d25c0e8ed5f1318acc23b6bb832654',
            'weights': str(root), 'mode': request['task'], 'frames': frames, 'steps': a.steps,
            'width': a.width, 'height': a.height, 'torch': torch.__version__,
            'acceleration_profile': a.acceleration_profile, 'attention_requested': a.attention_backend,
            'attention_actual': attention_module.ATTENTION_IMPLEMENTATION,
            'gpu': torch.cuda.get_device_name(0), 'vram_limit_gb': vram_limit,
            'model_load_seconds': load_seconds, 'cfg_scale': 1.0, 'flow_shift': 12.0, 'audio_flow_shift': 3.0}
        acceleration = None
        if pdd is not None:
            metadata['pdd_base_variant'] = a.pdd_base_variant
            metadata['pdd'] = pdd
        if a.acceleration_profile in ('adaln_exact', 'pdd8_exact'):
            from h3_acceleration import enable_adaln_exact
            acceleration = enable_adaln_exact(pipe)
        step_times = []
        model_fn = pipe.model_fn
        def measured_model_fn(*args, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            result = model_fn(*args, **kw)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t)
            return result
        pipe.model_fn = measured_model_fn
        save(output.parent/'worker_metadata.json', metadata)
        pipeline_started = time.perf_counter()
        video, audio = pipe(prompt=request['prompt'], width=a.width, height=a.height, num_frames=frames,
            num_inference_steps=a.steps, seed=request['seed'], cfg_scale=1.0, flow_shift=12.0, audio_flow_shift=3.0, **kwargs)
        torch.cuda.synchronize()
        metadata.update(pipeline_seconds=time.perf_counter()-pipeline_started, denoise_seconds=sum(step_times),
                        denoise_step_seconds=step_times, median_denoise_step_seconds=statistics.median(step_times),
                        peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30)
        if acceleration is not None:
            metadata['acceleration'] = acceleration
        if pdd is not None:
            metadata['pdd'] = pdd
        video_hash = hashlib.sha256()
        for frame in video:
            video_hash.update(np.asarray(frame).tobytes())
        audio_array = audio.detach().cpu().contiguous().numpy() if torch.is_tensor(audio) else np.asarray(audio)
        metadata['decoded_video_sha256'] = video_hash.hexdigest()
        metadata['decoded_audio_sha256'] = hashlib.sha256(audio_array.tobytes()).hexdigest()
        metadata['audio_array_shape'] = list(audio_array.shape)
        write_started = time.perf_counter()
        temp = output.with_name('native.tmp.mp4')
        write_video_audio(video, audio, str(temp), fps=24, audio_sample_rate=32000)
        os.replace(temp, output)
        metadata.update(write_seconds=time.perf_counter()-write_started, total_seconds=time.perf_counter()-started)
        save(output.parent/'worker_metadata.json', metadata)
        save(status_path, {'status': 'completed', 'path': str(output), 'frames': len(video)})
    except BaseException as exc:
        traceback.print_exc()
        save(status_path, {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
