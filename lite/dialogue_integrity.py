"""Single literal speech source; cheap audit of existing, media-bound observations.

The guard reduces prompt contamination. It is not a guarantee of model compliance.
Text mismatch alone is uncertainty; independent agreeing observations can establish
an insertion. No new paid reviewer, transcript rewrite, or automatic script edit.
"""
from copy import deepcopy
import difflib
import re
import unicodedata

VERSION = 'literal_speech_v1'
CHECKS = ('dialogue_fidelity', 'spoken_language', 'unexpected_speech')
CORE = set(CHECKS) | {'speech_completeness','voice_identity','source_binding','speaker_binding'}
PLANNER_GUIDE = '''
SPEECH IS HOST-OWNED. render_spec contains only visible actions, positions, expressions,
lighting and camera movement. Never explain, paraphrase, translate or quote what anyone
says, asks, sings, pleads, thinks or narrates. Do not include dialogue intent, literary
analysis, director commentary, hypothetical outcomes or instructions to the audience.
Use physical acting instead (eyes lower, palm rises, jaw tenses). Never specify a second
speaking schedule in prose: host dialogue_timeline is the sole timing authority.
The task's spoken language is independent of the story's country or historical culture.
When dialogue is empty, there are no intelligible words, crowd dialogue or narration.
Do not repeat production advice in shot content. A revision must remove conflicting
conditions, not stack more negative instructions onto the same contaminated text.
'''
REVIEW_GUIDE = '''
Also check dialogue_fidelity, spoken_language and unexpected_speech with the same
{status:clear|defect|unknown|not_applicable,observation,time_s} schema in checks.
Clearly pronounced extra words are still a dialogue_fidelity defect. Quote the ACTUAL
extra/omitted/repeated words with times, including spoken production directions.
Report the actual language heard, including mixed Japanese/English/Chinese; setting and
nationality do not authorize a different spoken language. Empty dialogue means no
intelligible speech (including crowd talk). Breathing, laughing or music is not a word.
CPU transcription is an independent hint, not proof: homophones, names and punctuation
may differ. If observation and transcript conflict, keep unknown; never fabricate clear.
The supplied comparison is descriptive only. Use the original WAV/video to assess it.
'''


def enabled(config):
    return config.get('completion_policy', {}).get('dialogue_integrity') == VERSION


# Fail closed on common speech paraphrases and nonvisual commentary. Not a semantic
# parser: the planner remains responsible for physical, faithful event descriptions.
SPEECH = re.compile(r'\b(?:say(?:s|ing)?|said|tell(?:s|ing)?|told|explain(?:s|ed|ing)?|ask(?:s|ed|ing)?|reply|replies|replied|remark(?:s|ed|ing)?|declare(?:s|d)?|announce(?:s|d)?|utter(?:s|ed|ing)?(?!\s+(?:stillness|silence|calm|darkness|quiet)\b)|recite(?:s|d)?|narrat(?:e|es|ed|ion)|plead(?:s|ed|ing)?|speak(?:s|ing)?|spoken|sing(?:s|ing)?|sang|whisper(?:s|ed|ing)?|mutter(?:s|ed|ing)?|speech|dialogue|words|sentence|line\s+delivery|chatter|chant(?:s|ing)?|prais(?:e|es|ing)|stat(?:e|es|ed|ing)|report(?:s|ed|ing)?|answer(?:s|ed|ing)?|propos(?:e|es|ed|ing)|jok(?:e|es|ed|ing)|wish(?:es|ed|ing)?|cr(?:y|ies|ied)(?: out)?|call(?:s|ed|ing)?(?: out)?|beg(?:s|ged|ging)?)\b', re.I)
COMMENTARY = re.compile(r'\b(?:is intended|intended to|intention|signif(?:y|ies)|symboli[sz]|represents|realiz(?:e|es|ing)|understands? that|means that|in order to|because|which shows|the audience|the viewer|narrative|causal|story beat|moral)\b', re.I)
TIMING = re.compile(r'\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)[ -]*(?:seconds?|secs?|s)\b|\b(?:speaking window|silent pause|held silence)\b',re.I)
SILENT_SPEECH_CUE = re.compile(
    r'\b(?:voices?|voiced|vocal(?:i[sz]\w*|s)?|voice[- ]?overs?|'
    r'vocifer\w*|syllables?|phonemes?|pronounc\w*|enunciat\w*|accent\w*|'
    r'diction|timbre|pitchy|lip[- ]?sync\w*|mouth(?:s|ing|ed)|'
    r'chats?|chatt\w*|talk\w*|convers\w*|gossip\w*|quarrel\w*|'
    r'argu\w*|complain\w*|exclaim\w*|grumbl\w*|murmur\w*|'
    r'describ\w+\s+(?:buying|how|what|where|why|their)|'
    r'transcri\w*|translat\w*|subtitles?|'
    r'Mandarin|Cantonese|Chinese\s+(?:delivery|questions?|voice)|'
    r'questions?|responds?|responded|responding|answers?|'
    r'shouts?|shouted|shouting|yells?|yelled|yelling|'
    r'greet\w*|scold\w*|whimper\w*|insist\w*|interject\w*|'
    r'pleas?|lyrics?|hums?|humming|singing|(?:move\w*|part\w*)\s+(?:his |her |their )?lips|'
    r'lips\s+(?:move\w*|part\w*))\b',
    re.I,
)
FALLBACK = {'opening':'Hold the established characters and location in a continuous shot.',
 'action':'Preserve the established physical action; use restrained visible facial and hand movements.',
 'camera':'Maintain a stable continuous composition without a cut.',
 'closing':'Hold the final visible pose with quiet natural movement.'}


def silent_speech_cue(value):
    return bool(SPEECH.search(value) or SILENT_SPEECH_CUE.search(value)
                or COMMENTARY.search(value)
                or re.search(r'["“”‘]|(?<!\w)\x27|[\u3040-\u30ff\u3400-\u9fff]', value))


def clean_silent_direction(value):
    """Pass through uncontaminated physical directions, without paraphrasing speech."""
    value = ' '.join(str(value or '').split())
    if not silent_speech_cue(value):
        return value
    kept = []
    for clause in re.split(r'(?<=[.!?;])\s+|,\s+|\s+(?:and|while|as)\s+(?=(?:then\s+)?\w+\b)', value, flags=re.I):
        clause = clause.strip()
        if not clause:
            continue
        if silent_speech_cue(clause):
            continue
        kept.append(clause)
    return ', '.join(kept)


def visual_only(spec, *, silent=False):
    result={};removed=[];trimmed=[];fallback_fields=[]
    for field in FALLBACK:
        raw=str((spec or {}).get(field) or '')
        # Split sentences/clauses; remove the contaminated unit, not translate it
        # into a second, equally speakable paraphrase.
        units=re.split(r'(?<=[.!?;])\s+|,\s+(?=(?:his|her|their|he|she|the)\b)|\s+and\s+(?=(?:then\s+)?(?:he|she|the\s+\w+|explains?|asks?|says?|speaks?)\b)',raw,flags=re.I)
        retained=[]
        for unit in units:
            original=unit
            # Remove an independent temporal aside, preserving the physical
            # composition that follows it. Never reconstruct a speech paraphrase.
            unit=re.sub(r'\b(?:immediately\s+)?(?:after|before|following)\s+(?:his|her|their|the)\s+(?:(?:previous|last|earlier|final)\s+)?(?:words|speech|line|sentence)\s*[:;,]?\s*', '', unit, flags=re.I)
            unit=re.sub(r'\b(?:for|over|during)\s+(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)[ -]*(?:seconds?|secs?|s)\b', '', unit, flags=re.I)
            if unit!=original:
                trimmed.append({'field':field,'original':original,'retained':unit,
                    'reason':'removed_independent_speech_or_duration_aside'})
            reasons=[]
            if SPEECH.search(unit):reasons.append('speech_paraphrase_or_verbal_direction')
            if COMMENTARY.search(unit):reasons.append('nonvisual_commentary')
            if TIMING.search(unit):reasons.append('competing_prose_timeline')
            if re.search(r'["“”]|[\u3040-\u30ff\u3400-\u9fff]',unit):reasons.append('quoted_or_foreign_unspoken_text')
            if reasons:removed.append({'field':field,'original':unit,'reasons':reasons})
            elif unit.strip():retained.append(unit.strip())
        if not retained and raw.strip():fallback_fields.append(field)
        visual = ' '.join(retained)
        if silent:
            cleaned = clean_silent_direction(visual)
            if visual and cleaned != visual:
                trimmed.append({'field':field,'original':visual,'retained':cleaned,
                    'reason':'silent_shot_only_physical_directions'})
            visual = cleaned
            if not visual and raw.strip() and field not in fallback_fields:
                fallback_fields.append(field)
        result[field]=visual or FALLBACK[field]
    return result,{'version':VERSION,'removed_units':removed,'trimmed_units':trimmed,
                   'fallback_fields':fallback_fields,'visual_constraints_lost':bool(fallback_fields),
                   'original':deepcopy(spec),
                   'scope':'prompt_hygiene_not_observed_media_improvement'}


def tokens(text):
    value=unicodedata.normalize('NFKC',str(text or '')).casefold()
    # Individual Han/Kana characters, but English word units. Punctuation is ignored.
    return re.findall(r'[a-z]+(?:[’\'][a-z]+)?|[\u3040-\u30ff\u3400-\u9fff]|\d+',value)


def transcript(value):
    if not isinstance(value,dict):return ''
    text=value.get('text','')
    return text if isinstance(text,str) else ''


def diffs(target, heard):
    a,b=tokens(target),tokens(heard)
    return [{'kind':tag,'expected':a[i:j],'heard':b[k:l]}
            for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if tag!='equal']


def text_evidence(source,blind,av):
    target=' '.join(str(d.get('text','')) for d in source.get('dialogue',[]) if isinstance(d,dict))
    heard=av.get('heard_text','') if isinstance(av,dict) else ''
    if not isinstance(heard,str):heard=''
    independent=transcript(blind)
    return {'expected':target,'blind_asr':independent,'av_heard_text':heard,
            'blind_difference':diffs(target,independent) if independent else [],
            'av_difference':diffs(target,heard) if heard else [],
            'text_difference_alone_is_not_audible_defect':True}


def audit(source,blind,av):
    e=text_evidence(source,blind,av);findings=[]
    independent=tokens(e['blind_asr']);av_tokens=tokens(e['av_heard_text'])
    target=tokens(e['expected'])
    # Exact agreeing additions of >=2 Chinese characters / English words cannot
    # be dismissed as punctuation; divergent substitutions stay uncertain.
    # Avoid annotative/no-speech boilerplate as actual audio evidence.
    empty=re.compile(r'^(?:none|no (?:speech|dialogue|lexical speech)|silence|n/?a|无|没有对白|无对白|无语音|无可辨识言语)[.!。\s]*$',re.I)
    positive=bool(av_tokens) and not empty.fullmatch(e['av_heard_text'].strip())
    positive=positive and not e['av_heard_text'].lstrip().startswith(('[','('))
    if positive and av.get('status') == 'observed':
        expected = e['expected']; heard = e['av_heard_text']
        wrong_kana = not re.search(r'[\u3040-\u30ff]',expected) and len(re.findall(r'[\u3040-\u30ff]',heard)) >= 3
        wrong_english = bool(re.search(r'[\u3400-\u9fff]',expected)) and not re.search(r'[a-zA-Z]',expected) and len(re.findall(r'[a-zA-Z]{2,}',heard)) >= 2
        wrong_han = bool(expected) and not re.search(r'[\u3400-\u9fff]',expected) and len(re.findall(r'[\u3400-\u9fff]',heard)) >= 3
        if wrong_kana or wrong_english or wrong_han:
            findings.append({'id':'speech:observed_language_mismatch','category':'spoken_language','severity':'major',
                'state':'open','kind':'observed_content_defect','observation':{'expected_literal':expected,'actually_heard':heard,
                    'basis':'Original audiovisual observer explicitly transcribed a different script/language; task setting is not speech authority'}})
    if positive and e['av_difference']:
        matching=[]
        for diff in e['av_difference']:
            extra=diff['heard']
            if diff['kind']=='insert' and len(extra)>=2 and any(independent[i:i+len(extra)]==extra for i in range(len(independent)-len(extra)+1)):
                matching.append(diff)
        actual=av.get('status')=='observed'
        # Both are original observers of the SAME bound soundtrack; the report
        # records agreement, not an experimental proof of a successful repair.
        confirmed=actual and bool(matching) and isinstance(blind,dict) and bool(blind.get('source_audio_sha256'))
        findings.append({'id':'speech:literal_transcript_audit','category':'unexpected_speech' if not target else 'dialogue_fidelity',
            'severity':'major' if confirmed else 'uncertain','state':'open',
            'kind':'observed_content_defect' if confirmed else 'review_uncertain',
            'observation':{'comparison':e,'corroborated_insertions':matching,
                'reason':'Independent ASR and audiovisual transcript agree on extra spoken tokens' if confirmed else 'Literal transcript differs; homophones and observer disagreement remain unknown'}})
    return {'version':VERSION,'findings':findings,'comparison':e,'new_paid_calls':0}


def audit_candidate(candidate):
    review=candidate.get('review',{});panel=review.get('opinions',{}).get('audiovisual',{})
    raw=panel.get('raw',panel.get('raw_reply',{}))
    if not raw:raw=panel
    blind = review.get('blind_asr')
    if isinstance(blind,dict) and blind.get('source_audio_sha256') != candidate.get('receipt',{}).get('audio_sha256'):
        blind = None
    return {**audit(candidate.get('source_clip',{}),blind,raw),
            'media_sha256':candidate.get('receipt',{}).get('sha256'),
            'audio_sha256':candidate.get('receipt',{}).get('audio_sha256'),
            'original_review_unchanged':True}


def validate_source_language(lines, language):
    for row in lines:
        text = row['text']
        if re.search(r'[\u3040-\u30ff]',text):
            raise ValueError('Source dialogue contains Japanese script outside the authorized Chinese/English speech contract')
        if language == 'English' and re.search(r'[\u3400-\u9fff]',text):
            raise ValueError('Source dialogue script differs from the English task; preserve original and obtain source correction')
        # Chinese may legitimately contain a proper noun/acronym; do not silently delete it.
