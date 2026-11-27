"""One dialogue source, observable opinions and small host-owned media facts."""
from copy import deepcopy
import hashlib
import json
import re

VERSION = 'observed_facts_v1'
TAG = re.compile(r'<d>(.*?)</d>', re.S | re.I)
AV_CATEGORIES = {'lip_sync', 'mouth_activity', 'speaker_binding', 'dialogue_fidelity',
    'dialogue_content', 'speech_intelligibility', 'audio_quality', 'unexpected_music',
    'sound_event_alignment', 'audiovisual_sync', 'sound_policy_violation',
    'speech_completeness', 'dialogue_naturalness', 'sentence_ending', 'source_binding',
    'spoken_language', 'unexpected_speech'}
VISUAL_CATEGORIES = {'identity_continuity', 'visual_quality', 'temporal_continuity',
    'visual_continuity', 'transition_integrity', 'reference_image_leakage',
    'style_consistency', 'transient_artifacts', 'cross_clip_continuity', 'scene_continuity'}

# A small vocabulary bridge, not a semantic classifier. Keep the original field
# alongside the canonical category wherever this changes a decision view.
CATEGORY_ALIASES = {'speaker_binding':'source_binding', 'voice_binding':'source_binding',
    'speech_content':'dialogue_fidelity', 'dialogue_accuracy':'dialogue_fidelity',
    'speech_naturalness':'dialogue_naturalness', 'lip_synchronization':'lip_sync',
    'lip_synchronisation':'lip_sync', 'scene_consistency':'scene_continuity'}


def canonical_category(category):
    return CATEGORY_ALIASES.get(category, category) if isinstance(category, str) else category


def enabled(config):
    return config.get('review_system', {}).get('evidence_policy') == VERSION


def normal(text):
    return ''.join(re.findall(r'[\w\u4e00-\u9fff]+', str(text).lower()))


def scripted(clip):
    return [deepcopy(x) for x in clip.get('dialogue', [])
            if isinstance(x, dict) and isinstance(x.get('text'), str) and x['text'].strip()]


def compile_caption(caption, clip):
    """Keep actual shot slots; inject exact script occurrences, never quoted rules."""
    source = str(caption)
    lines = scripted(clip)
    # An absolute rule that quotes the dialogue creates another speech block.
    body = re.sub(r'^.*ABSOLUTE VOICE RULE:.*(?:\n|$)', '', source, flags=re.M | re.I)
    shot = re.search(r'\[Shot\s+\d+\]', body, re.I)
    shot_start = shot.start() if shot else -1
    pending = list(lines)
    retained = []
    def replace(match):
        spoken = normal(re.sub(r'\[[^\]]*\]', '', match.group(1)))
        if pending and match.start() >= shot_start and spoken == normal(pending[0]['text']):
            row = pending.pop(0)
            language = row.get('language') or ('Chinese' if re.search(r'[\u4e00-\u9fff]', row['text']) else 'English')
            retained.append(f"<d>[{language}] {row['text'].strip()}</d>")
            return f'__SCRIPT_SLOT_{len(retained)-1}__'
        return 'the scripted dialogue'
    body = TAG.sub(replace, body)
    # Full quoted dialogue elsewhere (summary, prohibitions, metadata) is not
    # another source. Deliberate repetitions remain separate script entries.
    for row in lines:
        literal = row['text'].strip()
        body = body.replace(literal, 'the scripted dialogue')
        bare = literal.rstrip('。！？!?.,，')
        if len(bare) >= 3:
            body = body.replace(bare, 'the scripted dialogue')
            letters = normal(bare)
            pattern = r'[\s，,。.!！?？、;；:：“”‘’\"\x27]*'.join(re.escape(x) for x in letters)
            if letters and letters[0].isascii():
                pattern = r'(?<!\w)' + pattern
            if letters and letters[-1].isascii():
                pattern += r'(?!\w)'
            body = re.sub(pattern, 'the scripted dialogue', body, flags=re.I)
    if pending:
        extra = []
        for row in pending:
            language = row.get('language') or ('Chinese' if re.search(r'[\u4e00-\u9fff]', row['text']) else 'English')
            retained.append(f"<d>[{language}] {row['text'].strip()}</d>")
            extra.append(f"{row.get('speaker', 'Speaker')}: __SCRIPT_SLOT_{len(retained)-1}__")
        block = '\nScripted speech in this shot, in the listed order:\n' + '\n'.join(extra) + '\n'
        marker = re.search(r'^overall_soundscape\s*:', body, re.M | re.I)
        at = marker.start() if marker else len(body)
        body = body[:at] + block + body[at:]
    for i, text in enumerate(retained):
        body = body.replace(f'__SCRIPT_SLOT_{i}__', text)
    if dialogue_texts(body) != [normal(x['text']) for x in lines]:
        raise ValueError('Compiled speech differs from the retained script occurrences')
    return body, {'version': VERSION, 'input_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'output_sha256': hashlib.sha256(body.encode()).hexdigest(),
        'input_dialogue_blocks': len(TAG.findall(source)), 'output_dialogue_blocks': len(TAG.findall(body)),
        'scripted_occurrences': len(lines), 'changed': body != source,
        'source': 'retained_current_script; each occurrence appears once'}


def dialogue_texts(caption):
    return [normal(re.sub(r'\[[^\]]*\]', '', text)) for text in TAG.findall(str(caption))]


def generic_limit(value):
    """Recognize method disclaimers, never erase a timed visible/audible defect."""
    item = value if isinstance(value, dict) else {}
    if isinstance(value, dict):
        observation = item.get('observation')
        if isinstance(observation, dict):
            return generic_limit(observation)
        if not isinstance(observation, str):
            return False
    text = str(item.get('observation', value)).lower()
    if (item.get('related_finding_ids')
            or re.search(r'\b\d+(?:\.\d+)?\s*s\b|\b\d+:\d+|\d+(?:\.\d+)?\s*秒', text)
            or re.search(r'\b(?:is visible|was observed|appears|changed|shows a)\b|出现了|看到了|发生了', text)):
        return False
    method = re.search(r'4\s*fps|sampl|between.?frame|逐帧|抽帧|采样|未采到|无声|without audio|no audio|silent frames', text)
    limited = re.search(r'cannot|can.t|unable|not possible|not captured|无法|不能|不保证|非逐帧|not frame.by.frame', text)
    withheld = re.search(r'no narrative context|no (?:script|caption|previous clip)|未提供剧本|没有上一|缺少叙事上下文', text)
    return bool(method and limited or withheld)


def owner(category):
    category = canonical_category(category)
    if category in AV_CATEGORIES:
        return 'audiovisual'
    if category in VISUAL_CATEGORIES:
        return 'visual'
    return None


def observed_panel(panel):
    clean = deepcopy(panel)
    audit = []
    for role, p in clean.items():
        p.pop('raw_reply', None)  # Original observer artifact and paid reply remain on disk.
        for field in ('scope_limits', 'limitations'):
            for item in p.pop(field, []) or []:
                audit.append({'role': role, 'source': field, 'reason': 'method_note_not_actionable', 'original': item})
        for field in ('uncertainties', 'findings'):
            kept = []
            for item in p.get(field, []) or []:
                category = item.get('category') if isinstance(item, dict) else None
                if generic_limit(item) or owner(category) not in (None, role):
                    audit.append({'role': role, 'source': field, 'reason': 'method_note_not_actionable' if generic_limit(item) else 'outside_reviewer_authority', 'original': item})
                else:
                    kept.append(item)
            p[field] = kept
        for field in ('perceptual_checks', 'alignment_checks'):
            for check in p.get(field, []) or []:
                if isinstance(check, dict) and check.get('status') == 'unknown' and generic_limit(check.get('observation', '')):
                    audit.append({'role': role, 'source': field, 'reason': 'method_note_not_actionable', 'original': deepcopy(check)})
                    check.update(status='outside_observation', observation='No concrete defect reported in the supplied observable evidence.')
    return clean, audit


def media_facts(candidate, previous=None):
    def one(c, source):
        receipt = c.get('receipt', {})
        return {'clip_id': source.get('id', source.get('clip_id')), 'take': c.get('take'),
            'target_seconds': source.get('seconds'), 'actual_seconds': receipt.get('seconds'),
            'fps': receipt.get('fps'), 'frames': receipt.get('frames'),
            'basis': 'host receipt of these exact media bytes; numbers are not estimates'}
    return {'current': one(candidate, candidate.get('source_clip', {})),
        'previous': one(previous, previous.get('clip', {})) if previous else None}


def checked_decision(decision, facts):
    from .closed_loop import intervals
    clean = deepcopy(decision)
    current = facts['current']; actual = current.get('actual_seconds'); target = current.get('target_seconds')
    tolerance = 1 / max(1, current.get('fps') or 24) + .001
    audit, findings = [], []
    for finding in clean.get('additional_findings', []) or []:
        if not isinstance(finding, dict):
            audit.append({'kind': 'malformed_additional_finding', 'original': finding}); continue
        category = str(finding.get('category', '')).lower()
        if ('duration' in category or '时长' in category) and isinstance(actual, (int, float)) and isinstance(target, (int, float)) and abs(actual-target) <= tolerance:
            audit.append({'kind': 'false_duration_claim', 'original': finding, 'actual_media_facts': current}); continue
        times = intervals(finding.get('time_s'))
        if isinstance(actual, (int, float)) and any(a < 0 or b > actual+tolerance for a,b in times):
            audit.append({'kind': 'outside_current_media', 'original': finding, 'actual_media_facts': current}); continue
        if owner(finding.get('category')) not in (None, finding.get('role')):
            audit.append({'kind': 'outside_reviewer_authority', 'original': finding}); continue
        findings.append(finding)
    clean['additional_findings'] = findings
    if audit:
        audit[0]['original_director_reason'] = clean.get('reason')
        supported = [str(x.get('reason')) for x in clean.get('finding_dispositions', [])
                     if isinstance(x,dict) and x.get('reason')]
        clean['reason'] = ('Host media facts checked: unsupported added opinions are excluded and archived in decision_audit. '
            + 'Retained director action: ' + str(clean.get('decision')) + '. ' + '; '.join(supported))
    return clean, audit


def memory_view(value):
    """Request view only; immutable episodes and frozen request bindings stay intact."""
    if isinstance(value, list):
        return [memory_view(x) for x in value if not generic_limit(x)]
    if isinstance(value, dict):
        return {k: memory_view(v) for k,v in value.items()
                if k not in ('scope_limits', 'limitations', 'observation_filter_audit', 'decision_audit',
                             'decision_before_reconciliation', 'raw_reply')
                and not (isinstance(v, str) and generic_limit(v))}
    return value
