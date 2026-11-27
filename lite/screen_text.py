"""Separate literal artwork from the English fields actually compiled for H3."""
from copy import deepcopy
from decimal import Decimal
import re


HAN = r'\u3400-\u9fff'
HAS_HAN = re.compile(f'[{HAN}]')
AMOUNT = re.compile(r'(?<![\w.,])(?P<number>\d+(?:\.\d+)?)(?P<unit>[万亿千元])(?P<yuan>元)?', re.ASCII)
SCALE = {'千': 1000, '万': 10000, '亿': 100000000, '元': 1}
# A Han passage only belongs to artwork after an explicit text/display cue.
# Quotation marks admit mixed-language brand names. Unquoted copy is limited
# to ideographs and numeric terms; subsequent English direction stays intact.
LITERAL = (
    r'"[^"\n]+"' + r"|“[^”\n]+”|‘[^’\n]+’|'[^'\n]+'|「[^」\n]+」|"
    + rf'[{HAN}0-9]+(?:(?:\.(?=[0-9])|、(?=[{HAN}0-9]))[{HAN}0-9]+)*'
)
SURFACE = r'(?:banner|sign(?:age)?|panel|screen|placard|badge|display|poster|card|label|notice|phone|interface|board)'
CUE_PREFIX = (
    rf'(?P<cue>\b(?:'
    rf'(?:{SURFACE}\s+(?:reads?|says?|shows?|displays?|reading|saying|showing|displaying)|'
    rf'(?:Chinese|Mandarin)\s+(?:text|characters|lettering)'
    rf'(?:\s+{SURFACE})?(?:\s+(?:reads?|shows?|displays?|reading|showing|displaying))?)'
    rf')\s*[:=]?\s+)')
TEXT_CUE = re.compile(CUE_PREFIX + rf'(?P<literal>{LITERAL})', re.IGNORECASE | re.ASCII)


def english_amount(match):
    unit = match['unit']
    number = Decimal(match['number']) * SCALE[unit]
    digits = format(number, ',f')
    if '.' in digits:
        digits = digits.rstrip('0').rstrip('.')
    return digits + (' yuan' if unit == '元' or match['yuan'] else '')


def artwork_direction(value, artwork, location):
    if not isinstance(value, str) or not HAS_HAN.search(value):
        return value

    def project(match):
        raw = match['literal']
        literal = raw[1:-1] if raw[0] in '\"“‘\'「' else raw
        if not HAS_HAN.search(literal):
            return match.group()
        if raw == literal and (value[match.end():match.end()+1] or ' ') not in ' .,:;!?，。；！？，)':
            raise ValueError(f'{location}: quote the full display text or move it to on_screen_text')
        if literal not in artwork:
            artwork.append(literal)
        amount = AMOUNT.fullmatch(literal)
        replacement = english_amount(amount) if amount else 'the specified lettering'
        if not amount and re.fullmatch(r'(Chinese|Mandarin) (text|characters|lettering)[:=]?\s+', match['cue'], re.I):
            return 'space reserved for the specified lettering'
        return match['cue'] + replacement

    cue = TEXT_CUE
    multiword = sorted({line for line in artwork if HAS_HAN.search(line) and ' ' in line and line in value},
                       key=len, reverse=True)
    if multiword:
        # An explicit literal fixes the boundary between badge text and subsequent staging.
        declared = '|'.join(re.escape(line) for line in multiword)
        cue = re.compile(CUE_PREFIX + rf'(?P<literal>(?-i:{declared})|{LITERAL})', TEXT_CUE.flags)
    cleaned = AMOUNT.sub(english_amount, cue.sub(project, value))
    if HAS_HAN.search(cleaned):
        raise ValueError(
            f'{location} contains unresolved non-English staging. Write the physical action in English, '
            'put exact visible lettering in on_screen_text (a list of strings), and keep spoken words in dialogue.')
    return cleaned


def project_plan(plan):
    """Copy the plan; no rewriting of the paid response, speech, captions, or IDs."""
    if not isinstance(plan.get('clips'), list):
        return plan
    clips = deepcopy(plan['clips'])
    for pos, clip in enumerate(clips, 1):
        if not isinstance(clip, dict):
            continue
        location = f'clip {pos}'
        artwork = clip.get('on_screen_text', [])
        if not isinstance(artwork, list) or any(not isinstance(item, str) or not item.strip() for item in artwork):
            raise ValueError(f'{location}.on_screen_text must be a list of nonempty literal strings')

        def process(mapping, fields, path):
            if isinstance(mapping, dict):
                for field in fields:
                    if field in mapping:
                        mapping[field] = artwork_direction(mapping[field], artwork, f'{path}.{field}')

        process(clip.get('render_spec'), ('opening', 'action', 'camera', 'closing'), f'{location}.render_spec')
        process(clip, ('entry_state', 'exit_state'), location)
        beats = clip.get('action_beats')
        if isinstance(beats, list):
            for index, beat in enumerate(beats):
                process(beat, ('action', 'visible_result'), f'{location}.action_beats[{index}]')
        checkpoints = clip.get('state_checkpoints')
        if isinstance(checkpoints, dict):
            for name, checkpoint in checkpoints.items():
                process(checkpoint, ('entry', 'exit'), f'{location}.state_checkpoints[{name}]')
        if artwork:
            clip['on_screen_text'] = artwork
    return {**plan, 'clips': clips}
