"""Optional executables that implement the public media worker protocol."""
from pathlib import Path
import os
import subprocess


KINDS = {'image_candidates': 'image', 'audio_candidates': 'audio', 'h3_seconds': 'video'}


def command(config, kind):
    routes = config.get('media_adapters', {})
    if set(routes) - set(KINDS.values()):
        raise ValueError('media_adapters accepts image, audio and video')
    if kind not in routes:
        return None
    route = routes[kind]
    if set(route) != {'executable', 'model'} or not isinstance(route['model'], str) or not route['model'].strip():
        raise ValueError('Each adapter requires an executable and a public model identifier')
    path = Path(route['executable'])
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('Adapter must name an executable file by absolute path')
    return [str(path), '--kind', kind, '--model', route['model']]


def ready(config, kind):
    try:
        argv = command(config, kind)
        return bool(argv and subprocess.run(argv + ['--check'], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15).returncode == 0)
    except (ValueError, OSError, subprocess.TimeoutExpired):
        return False
