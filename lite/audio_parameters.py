"""Translate planner language names into the deployed TTS interface."""
from .storage import Unavailable, key


VERSION = 'tts_language_v1'
LANGUAGES = ('auto', 'chinese', 'english', 'french', 'german', 'italian',
             'japanese', 'korean', 'portuguese', 'russian', 'spanish')
ALIASES = dict(zip(('zh', 'en', 'fr', 'de', 'it', 'ja', 'ko', 'pt', 'ru', 'es'), LANGUAGES[1:]))
ALIASES.update({'中文': 'chinese', '汉语': 'chinese', '普通话': 'chinese',
                'mandarin': 'chinese', 'cmn': 'chinese'})


class AudioParameterError(Unavailable):
    """Changing the caption or seed cannot correct an unsupported parameter."""


def tts_language(value):
    if value is None or value == '':
        return 'Chinese'
    if not isinstance(value, str):
        raise AudioParameterError('TTS language must be a supported name or language code')
    text = value.strip().casefold().replace('_', '-')
    if not text:
        return 'Chinese'
    language = text if text in LANGUAGES else ALIASES.get(text) or ALIASES.get(text.split('-')[0])
    if language not in LANGUAGES:
        raise AudioParameterError(f'Unsupported TTS language {value!r}; supported: {list(LANGUAGES)}')
    return 'auto' if language == 'auto' else language.title()


def parameter_signature(spec):
    return key({'adapter': VERSION, 'generator': spec.get('generator'), 'language': spec.get('language')})
