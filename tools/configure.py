"""Create a private, per-story workspace from the public source manifest."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import os
import shutil
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
GUIDE_SHA = '1e574f356716ad55612247ffb7bbccbcdb484ad96599d63c7dca1af186b1fab7'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def shared_rate_limits(install):
    """Publish all three initial provider slots together, once per installation."""
    from lite.fleet_lock import directory_lock, read_reservation

    folder = install / 'rate-limits'
    providers = ('glm', 'doubao', 'gemini')
    with directory_lock(install / '.rate-limits-setup.d'):
        if not folder.exists():
            if (install / 'backend').exists():
                raise ValueError('Shared rate-limits directory is missing from an existing installation; '
                                 'reconcile its saved provider deadlines before continuing')
            with tempfile.TemporaryDirectory(dir=install) as temp:
                staged = Path(temp) / 'rate-limits'
                staged.mkdir()
                for provider in providers:
                    dump(staged / (provider + '.json'), {'next_allowed_at': 0})
                staged.rename(folder)
        for provider in providers:
            with directory_lock(folder / (provider + '.cross_host.d')):
                read_reservation(folder / (provider + '.json'))
    return folder


def shared_backend(install, files):
    """Install worker entry points once so jobs on the same GPU share one lock."""
    dest = install / 'backend'
    names = [Path(key).relative_to('backend') for key in files if key.startswith('backend/')]
    install.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=install) as temp:
            staged = Path(temp) / 'backend'
            for name in names:
                (staged / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / 'backend' / name, staged / name)
            (staged / 'service_logs').mkdir()
            try:
                staged.rename(dest)
            except OSError:
                if not dest.is_dir():
                    raise
    for name in names:
        if digest(dest / name) != files['backend/' + name.as_posix()]:
            raise ValueError('Shared backend differs from this checkout: ' + name.as_posix())
    return dest


def configure(args):
    install, workspace = args.install.resolve(), args.workspace.resolve()
    source, guide = args.source.resolve(), args.guide.resolve()
    if (workspace.exists() or install == workspace or install.is_relative_to(workspace)
            or workspace.is_relative_to(ROOT) or install.is_relative_to(ROOT)
            or ROOT.is_relative_to(workspace)):
        raise ValueError('Choose a new workspace and an installation outside the checkout')
    if digest(guide) != GUIDE_SHA:
        raise ValueError('The upstream full-reference guide has the wrong SHA-256')
    files = json.loads((ROOT / 'manifest.json').read_text())['files']
    for name, expected in files.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Checkout integrity mismatch: ' + name)
    cfg = json.loads((ROOT / 'config.example.json').read_text())
    if args.cast:
        cfg['generation_contract']['cast'] = json.loads(args.cast.read_text())
    elif source.read_bytes() != (ROOT / 'examples/the_brass_key.txt').read_bytes():
        raise ValueError('Supply --cast for your story')
    from lite.binding_contract import validate_cast
    validate_cast(cfg['generation_contract']['cast'])
    rate_limits = shared_rate_limits(install)
    backend = shared_backend(install, files)
    models = install / 'models'
    gpu = str(args.gpu)
    cfg.update(lite_root=str(workspace), framework_release=str(ROOT), production_root=str(backend),
               authorization=str(workspace / 'authorization.json'), h3_config=str(workspace / 'h3.yaml'),
               rate_limit_root=str(rate_limits), media_bin=str(args.ffmpeg_bin.resolve()),
               video_aspect_ratio=args.aspect)
    cfg['task_contract'].update(physical_gpu=args.gpu, language=args.language, source_sha256=digest(source))
    cfg['image_provider'].update(device=gpu, python=str(install/'envs/image/bin/python'),
                                 model_path=str(models/'image'))
    cfg['audio_provider'].update(device=gpu, python=str(install/'envs/audio/bin/python'),
            lock_file=str(backend/'service_logs'/('gpu_'+gpu+'.lock')), music_source=str(install/'ACE-Step-1.5'),
            manifest=str(workspace/'model_manifests/catalog.json'),
            readiness=str(workspace/'verification/audio_runtime_probe/readiness.json'),
            models={mode:str(models/mode) for mode in ('voice_design','voice_clone','music')})
    cfg['speech_verification'].update(asr_python=str(install/'envs/asr/bin/python'), asr_model_path=str(models/'asr'))
    auth = json.loads((ROOT/'examples/authorization.example.json').read_text())
    auth['physical_gpus'] = [args.gpu]
    if args.profile == 'long':
        cfg.update(target_seconds=600, max_final_seconds=630)
        cfg['task_contract'].update(version='longform_task_parameters_v1', min_seconds=570,
                                    max_seconds=630, director_max_output_tokens=32768)
        auth['duration_seconds'] = [570,630]
    h3 = yaml.safe_load((ROOT/'examples/h3.example.yaml').read_text())
    h3.update(cuda_device=gpu, diffsynth_python=str(install/'envs/video/bin/python'),
              diffsynth_project=str(install/'DiffSynth-Studio'), ref2va_weights=str(models/'h3/Ref2VA'),
              fl2va_weights=str(models/'h3/FL2VA'),
              pdd_lora_path=str(models/'adapters/MiniMax-H3-Ref2VA-Acc-8Step.safetensors'),
              fl2va_pdd_lora_path=str(models/'adapters/MiniMax-H3-FL2VA-Acc-8Step.safetensors'))
    catalog = json.loads((ROOT/'examples/model_sources.json').read_text())
    workspace.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workspace.parent) as temp:
        stage = Path(temp)/'workspace'
        stage.mkdir()
        for name in list(files) + ['manifest.json']:
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        (stage/'resources').mkdir(exist_ok=True)
        shutil.copy2(guide, stage/'resources/ref-en.txt')
        shutil.copy2(source, stage/'story.txt')
        (stage/'model_manifests').mkdir()
        dump(stage/'model_manifests/catalog.json', catalog)
        for mode in ('voice_design','voice_clone','music'):
            dump(stage/'model_manifests'/(mode+'.json'), catalog[mode])
        dump(stage/'authorization.json', auth)
        dump(stage/'config.json', cfg)
        (stage/'h3.yaml').write_text(yaml.safe_dump(h3, sort_keys=False))
        # A fully populated destination cannot be overwritten by a concurrent installer.
        stage.rename(workspace)
    print(workspace / 'config.json')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', required=True, type=Path)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--guide', required=True, type=Path)
    parser.add_argument('--source', type=Path, default=ROOT/'examples/the_brass_key.txt')
    parser.add_argument('--cast', type=Path)
    parser.add_argument('--profile', choices=['short','long'], default='short')
    parser.add_argument('--language', choices=['Chinese','English'], default='English')
    parser.add_argument('--aspect', choices=['16:9','9:16'], default='16:9')
    parser.add_argument('--gpu', choices=[0,1], type=int, default=0)
    parser.add_argument('--ffmpeg-bin', required=True, type=Path)
    configure(parser.parse_args())
