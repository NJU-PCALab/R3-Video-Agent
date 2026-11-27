"""Isolated audio worker: one request, one result, shared physical GPU lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lite.storage import write
from lite.audio_parameters import AudioParameterError, tts_language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    parser.add_argument('--lock-file', required=True)
    args = parser.parse_args()
    response = Path(args.response);folder = response.parent
    status = folder / 'worker_status.json'
    request = json.loads(Path(args.request).read_text())
    lock_path = Path(args.lock_file);lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if request['generator'] in ('voice_design', 'voice_clone'):
            language = tts_language(request.get('language'))
        with lock_path.open('a') as gpu_lock:
            write(status, {'state': 'waiting_gpu_lock', 'device': os.environ.get('CUDA_VISIBLE_DEVICES')})
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            write(status, {'state': 'running'})
            started = time.time()
            import torch
            import soundfile as sf
            torch.set_num_threads(8)
            torch.manual_seed(request['seed'])
            output = folder / 'generated.wav'
            mode = request['generator']
            if mode in ('voice_design', 'voice_clone'):
                from qwen_tts import Qwen3TTSModel
                model = Qwen3TTSModel.from_pretrained(request['model_path'], device_map='cuda:0',
                            dtype=torch.bfloat16, attn_implementation='sdpa')
                options = {'text': request['text'], 'language': language,
                           'max_new_tokens': 1024}
                if mode == 'voice_design':
                    wavs, sr = model.generate_voice_design(**options, instruct=request['prompt'])
                else:
                    reference = request['reference']
                    wavs, sr = model.generate_voice_clone(**options, ref_audio=reference['path'],
                        ref_text=reference.get('text') or None,
                        x_vector_only_mode=not bool(reference.get('text')))
                sf.write(output, wavs[0], sr, subtype='PCM_16')
            elif mode == 'music':
                sys.path.insert(0, request['music_source'])
                os.environ['ACESTEP_CHECKPOINTS_DIR'] = request['model_path']
                os.environ['ACESTEP_PROJECT_ROOT'] = request['music_source']
                # This deployment uses GLM for planning and never initializes
                # ACE's optional music LM. Check only the three consumed models.
                import acestep.model_downloader as downloader
                downloader.MAIN_MODEL_COMPONENTS = ['acestep-v15-turbo', 'vae', 'Qwen3-Embedding-0.6B']
                from acestep.handler import AceStepHandler
                from acestep.llm_inference import LLMHandler
                from acestep.inference import GenerationParams, GenerationConfig, generate_music
                handler = AceStepHandler()
                message, ok = handler.initialize_service(project_root=request['music_source'],
                    config_path='acestep-v15-turbo', device='cuda', use_flash_attention=False,
                    compile_model=False, offload_to_cpu=False)
                if not ok:
                    raise RuntimeError(str(message))
                params = GenerationParams(caption=request['prompt'], lyrics='[Instrumental]',
                    instrumental=True, duration=max(10, min(180, request.get('seconds', 15))),
                    inference_steps=8, seed=request['seed'], thinking=False, use_cot_metas=False,
                    use_cot_caption=False, use_cot_lyrics=False)
                result = generate_music(handler, LLMHandler(), params,
                    GenerationConfig(batch_size=1, audio_format='wav'), save_dir=str(folder / 'music_output'))
                if not result.success or not result.audios:
                    raise RuntimeError(str(result.error))
                samples, sr = sf.read(result.audios[0]['path'], always_2d=True)
                sf.write(output, samples, sr, subtype='PCM_16')
            else:
                raise ValueError('Unsupported audio generator: ' + str(mode))
            info = sf.info(output)
            if info.frames == 0:
                raise RuntimeError('Audio model returned an empty waveform')
            write(response, {'ok': True, 'path': str(output), 'sample_rate': info.samplerate,
                 'channels': info.channels, 'seconds': info.duration, 'generator': mode,
                 'model_path': request['model_path'], 'elapsed_seconds': time.time() - started,
                 'device': os.environ.get('CUDA_VISIBLE_DEVICES'), 'seed': request['seed']})
            write(status, {'state': 'completed'})
    except Exception as exc:
        write(response, {'ok': False, 'error': str(exc), 'type': type(exc).__name__,
              'error_type': 'invalid_audio_parameter' if isinstance(exc, AudioParameterError) else 'worker_failure',
              'retryable': not isinstance(exc, AudioParameterError), 'traceback': traceback.format_exc()})
        write(status, {'state': 'failed'})
        raise


if __name__ == '__main__':
    main()
