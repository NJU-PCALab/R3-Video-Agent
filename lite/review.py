"""Evidence and feedback drive acceptance; scores remain descriptive metadata."""
from copy import deepcopy
from pathlib import Path
import subprocess

from . import prompts
from . import shot_policy
from .storage import DIMENSIONS, Unavailable, read, score, sha, write

VERSION = 'feedback_v1'
PERCEPTION_PROTOCOL = 'perception_first_v2'
PERCEPTUAL_CHECKS = ('reference_image_leakage', 'style_consistency',
                     'transition_integrity', 'transient_artifacts')
ROLES = {
    'visual': ('story_action_fidelity', 'identity_continuity', 'visual_quality', 'temporal_continuity'),
    'audio': ('dialogue_fidelity', 'speaker_consistency', 'audio_quality'),
    'audiovisual': ('audiovisual_sync', 'narrative_continuity'),
}


def enabled(config):
    return config.get('review_system', {}).get('version') == VERSION


def perception_first(config):
    return enabled(config) and config.get('review_system', {}).get('observation_protocol') == PERCEPTION_PROTOCOL


def opinion(role, raw):
    """Host-owned labels link observations without rewriting the raw reply."""
    value = deepcopy(raw) if isinstance(raw, dict) else {}
    observations = value.get('observed_events', value.get('observations', []))
    observations = observations if isinstance(observations, list) else []
    findings = value.get('findings')
    missing_findings = not isinstance(findings, list)
    findings = findings if isinstance(findings, list) else []
    normalized = []
    for index, item in enumerate(findings):
        item = deepcopy(item) if isinstance(item, dict) else {'observation': str(item)}
        item['id'] = f'{role}:{index + 1}'
        item.setdefault('severity', 'uncertain')
        normalized.append(item)
    checks = value.get('perceptual_checks')
    checks = checks if isinstance(checks, list) else []
    # A model-reported defect cannot disappear because its findings list is empty.
    # Keep the literal observation; the host supplies only a stable finding label.
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or check.get('status') != 'defect':
            continue
        if not any(f.get('category') == check.get('category') for f in normalized):
            normalized.append({'id': f'{role}:check:{index + 1}',
                'category': check.get('category'), 'severity': check.get('severity', 'uncertain'),
                'observation': check.get('observation'), 'time_s': check.get('time_s'),
                'scope': 'local', 'origin': 'literal_model_perceptual_check'})
    unknown = value.get('status') not in ('observed', 'OBSERVED') or not observations or missing_findings
    scores = value.get('scores') if isinstance(value.get('scores'), dict) else {}
    return {'role': role, 'status': 'UNKNOWN' if unknown else 'observed',
            'observed_events': observations, 'findings': normalized,
            'uncertainties': value.get('uncertainties', []),
            'perceptual_checks': checks,
            'scores': {d: scores.get(d) if score(scores.get(d)) else None for d in ROLES[role]},
            'voice_observation': value.get('voice_observation'), 'transcript': value.get('transcript', []),
            'raw_reply': value}


def feedback_quality(review):
    if review.get('observation_protocol') in ('content_rsi_v3', 'content_rsi_v4'):
        from .collaborative_review import quality
        return quality(review)
    panel = review.get('opinions', {})
    panel = panel if isinstance(panel, dict) else {}
    known = all(panel.get(role, {}).get('status') == 'observed' for role in ROLES)
    perceptual = review.get('observation_protocol') == PERCEPTION_PROTOCOL
    if perceptual:
        checks = {c.get('category'): c for c in panel.get('visual', {}).get('perceptual_checks', [])
                  if isinstance(c, dict)}
        # Missing/uncertain observations remain unknown, never a Python failure.
        known = known and all(checks.get(c, {}).get('status') in ('clear', 'defect')
            and bool(checks[c].get('observation')) for c in PERCEPTUAL_CHECKS)
    decision = review.get('feedback_decision') or {}
    decision = decision if isinstance(decision, dict) else {}
    findings = [f for role in ROLES for f in panel.get(role, {}).get('findings', [])]
    dispositions = decision.get('finding_dispositions', [])
    dispositions = dispositions if isinstance(dispositions, list) else []
    addressed, outstanding = {}, []
    for row in dispositions:
        if isinstance(row, dict) and row.get('reason'):
            addressed[row.get('finding_id')] = row
    for finding in findings:
        response = addressed.get(finding['id'], {})
        # An opinion can be contested with explicit evidence, or an explicit
        # minor tradeoff can be accepted. Deferral never erases a prefix defect.
        accepted = response.get('disposition') in ('not_a_defect', 'covered_by_adjacent', 'accepted_tradeoff')
        if (perceptual and finding.get('severity') == 'major'
                and finding.get('category') in PERCEPTUAL_CHECKS):
            # A planned flashback or a director's textual explanation does not
            # remove an observed major rendering defect in these same bytes.
            accepted = False
        grounded = bool(response.get('evidence'))
        if not accepted or not grounded:
            outstanding.append(finding)
    action = decision.get('decision')
    complete = known and action in ('accept', 'revise', 'keep_needs_review') and bool(decision.get('reason'))
    ok = complete and action == 'accept' and not outstanding
    scores = review.get('scores')
    values = [v for v in (scores.values() if isinstance(scores, dict) else []) if score(v)]
    return {'ok': ok, 'complete': complete, 'rank': sum(values) / len(values) if values else None,
            'quality_status': 'OK' if ok else 'draft/needs_review', 'review_system': VERSION,
            'acceptance_basis': 'observations_and_finding_dispositions',
            'unresolved_findings': outstanding, 'feedback_decision': action,
            'not_applicable_dimensions': []}


def visual_frames(config, run, receipt, label):
    """CPU-only 4fps evidence, including the last frame; never claim continuous observation."""
    count, fps = int(receipt['frames']), float(receipt['fps'])
    step = max(1, round(fps / 4))
    indices = sorted(set(range(0, count, step)) | {count - 1})
    folder = Path(run) / 'review_media' / label
    manifest = folder / 'frames.json'
    existing = read(manifest)
    source_sha = sha(receipt['path'])
    if receipt.get('sha256') and source_sha != receipt['sha256']:
        raise Unavailable('Current visual media differs from the retained receipt')
    if existing and existing['source_sha256'] == source_sha:
        if all(Path(f['path']).exists() and sha(f['path']) == f['sha256'] for f in existing['frames']):
            return existing
        raise Unavailable('Retained visual review frames changed; reconcile original evidence')
    folder.mkdir(parents=True, exist_ok=True)
    select = f'not(mod(n\\,{step}))+eq(n\\,{count - 1})'
    command = [str(Path(config['media_bin']) / 'ffmpeg'), '-y', '-v', 'error', '-threads', '1',
               '-i', receipt['path'], '-vf', f"select='{select}',scale=768:-2", '-vsync', 'vfr',
               '-q:v', '4', '-threads', '1', str(folder / '%04d.jpg')]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired:
        raise Unavailable('Visual evidence extraction timed out; review is unknown, not a generation failure') from None
    if result.returncode:
        raise Unavailable('Visual evidence extraction failed: ' + result.stderr.decode(errors='replace')[-1000:])
    paths = sorted(folder.glob('*.jpg'))
    if len(paths) != len(indices):
        raise Unavailable('Visual evidence frame count mismatch; no paid review submitted')
    value = {'source_path': receipt['path'], 'source_sha256': source_sha, 'sample_fps': 4,
             'limitation': 'Sampled silent frames; events between samples may be unknown.',
             'frames': [{'path': str(p), 'sha256': sha(p), 'time_s': n / fps, 'frame_index': n}
                        for p, n in zip(paths, indices)]}
    write(manifest, value)
    return value


class ReviewPanel:
    def __init__(self, agent):
        self.agent = agent

    def review(self, cid, candidate):
        if self.agent.config.get('review_system', {}).get('observation_protocol') in ('content_rsi_v3', 'content_rsi_v4'):
            from .collaborative_review import CollaborativePanel
            return CollaborativePanel(self.agent).review(cid, candidate)
        a = self.agent
        take = candidate['take']
        name = f'review_panel/{cid}/{take}'
        perceptual = perception_first(a.config)
        if perceptual:
            name += '/' + PERCEPTION_PROTOCOL
        previous = a.s.data['selected'][-1] if a.s.data['selected'] else None
        refs = candidate['references']
        clip = candidate['source_clip']
        plan = a.s.data.get('plan', {}).get('clips', [])
        position = next((i for i, c in enumerate(plan) if c['id'] == cid), -1)
        context = {'target': clip, 'caption_is_only_a_target': candidate['caption'],
                   'source_story': a.source, 'next_planned_clips': plan[position + 1:position + 3],
                   'future_plan_is_not_observed': True,
                   'previous_actual_review': previous['review'] if previous else None}
        editorial = shot_policy.enabled(a.config)
        if editorial:
            context['editing_policy'] = shot_policy.context(a.config)
            context['boundary_intent_is_not_observed_fact'] = candidate.get('editing_intent')
        visual_problem = None
        try:
            evidence = visual_frames(a.config, a.s.run, candidate['receipt'], f'{cid}_{take}')
        except Unavailable as exc:
            visual_problem = str(exc)
            evidence = {'frames': []}
            a.s.warning(name + '/visual_evidence', exc)
        visual_media = [{'path': f['path'], 'label': f"Current silent frame {f['time_s']:.3f}s"}
                        for f in evidence['frames']]
        prior = {'frames': []}
        if previous:
            try:
                prior = visual_frames(a.config, a.s.run, previous['receipt'], f'previous_{previous["clip"]["id"]}')
            except Unavailable as exc:
                prior = {'frames': []}
                context['previous_visual_evidence_unavailable'] = str(exc)
                if editorial:
                    visual_problem = 'Previous boundary evidence unavailable: ' + str(exc)
            visual_media += [{'path': f['path'], 'label': f"Previous clip tail {f['time_s']:.3f}s"}
                             for f in prior['frames'][-3:]]
        boundary = None
        if editorial:
            pair = ([{'path': f['path'], 'label': f"BOUNDARY PREVIOUS tail {f['time_s']:.3f}s"}
                     for f in prior['frames'][-3:]] if previous else [])
            pair += [{'path': f['path'], 'label': f"BOUNDARY CURRENT opening {f['time_s']:.3f}s"}
                     for f in evidence['frames'][:3]]
            # Put the temporal pair before the full current frame sequence, so
            # it is explicit rather than buried among identity references.
            visual_media = pair + visual_media
            boundary = {'status': 'ready' if evidence['frames'] and (not previous or prior['frames']) else 'unknown',
                        'previous_clip_id': previous['clip']['id'] if previous else None,
                        'previous_media_sha256': previous['receipt'].get('sha256') if previous else None,
                        'current_media_sha256': candidate['receipt'].get('sha256'),
                        'intended_transition': 'editorial_cut', 'images': pair,
                        'meaning': 'Compare actual action/pose/state across the cut; a deliberate new camera shot is allowed.'}
            context['boundary_review'] = boundary
            write(Path(a.s.run) / 'review_media' / f'{cid}_{take}' / 'boundary.json', boundary)
        visual_media += [{'path': r['path'], 'label': 'Actual identity/setting reference ' + r['id']}
                         for r in refs if r['kind'] == 'image']
        audio_media = [{'path': candidate['receipt']['audio_path'], 'label': 'Full current native soundtrack'}]
        audio_media += [{'path': r['path'], 'label': 'Actual audio reference ' + r['id'] + '; ' + r.get('track_type', 'unknown')}
                        for r in refs if r['kind'] == 'audio']
        if previous:
            audio_media.append({'path': previous['receipt']['audio_path'], 'label': 'Previous actual mixed soundtrack; continuity only'})
        av_media = [{'path': candidate['receipt']['path'], 'label': 'Current actual audiovisual clip'},
                    {'path': candidate['receipt']['audio_path'], 'label': 'Current full native audio'}]
        if previous:
            av_media.append({'path': previous['receipt']['path'], 'label': 'Previous actual clip for transition only'})
        tasks = [('visual', prompts.VISUAL_FEEDBACK, 'doubao', visual_media),
                 ('audio', prompts.AUDIO_FEEDBACK, 'gemini', audio_media),
                 ('audiovisual', prompts.AV_FEEDBACK, 'gemini', av_media)]
        panel = {}
        for role, prompt, provider, media in tasks:
            role_context = context
            if perceptual and role in ('visual', 'audiovisual'):
                prompt = prompts.PERCEPTION_VISUAL if role == 'visual' else prompts.PERCEPTION_AV
                role_context = {'media_scope': 'Only actual labeled current/previous media and reference assets.',
                    'actual_seconds': candidate['receipt']['seconds'],
                    'actual_fps': candidate['receipt']['fps'],
                    'reference_assets_are_not_film_frames': True,
                    'intent_context_withheld': True}
            elif perceptual:
                role_context = {'required_dialogue': clip.get('dialogue', []),
                    'sound_policy': {'background_music': (a.config.get('editing_policy') or {}).get('background_music')},
                    'audio_reference_scope': 'Compare actual supplied timbre references; do not invent visual events.'}
            if editorial and not perceptual:
                prompt += shot_policy.instructions(a.config)
            if role == 'visual' and visual_problem:
                raw = a.artifact(f'{name}/{role}', lambda: {'status': 'unknown', 'observed_events': [],
                    'findings': [], 'uncertainties': [visual_problem], 'scores': {}})
            else:
                raw = a.artifact(f'{name}/{role}', lambda r=role, p=prompt, model=provider, files=media, payload=role_context:
                    a.advisory(f'{name}/{r}', p, payload, provider=model, media=files, max_tokens=6500))
            panel[role] = opinion(role, raw)
        decision_prompt = prompts.PERCEPTION_DECISION if perceptual else prompts.FEEDBACK_DECISION
        decision = a.artifact(f'{name}/decision', lambda: a.advisory(f'{name}/decision', a.prompt(decision_prompt),
            {**context, 'opinions': panel, 'remaining_real_takes': 6 - take,
             'rule': 'Scores are advisory; explicitly address each finding and reviewer uncertainty.'}, max_tokens=6000))
        scores = {d: panel[role]['scores'][d] for role in ROLES for d in ROLES[role]}
        review = {'review_system': VERSION, 'opinions': panel, 'feedback_decision': decision,
                  'scores': scores, 'voice_observation': panel['audio']['voice_observation'],
                  'transcript': panel['audio']['transcript'],
                  'observed_events': [v for role in ROLES for v in panel[role]['observed_events']],
                  'issues': [f for role in ROLES for f in panel[role]['findings']],
                  'repair_suggestions': decision.get('revision_plan', {}),
                  'evidence_manifest': str(Path(a.s.run) / 'review_media' / f'{cid}_{take}' / 'frames.json'),
                  'limitations': ['Visual review samples at 4fps; numeric scores do not determine acceptance.']}
        if perceptual:
            review.update(observation_protocol=PERCEPTION_PROTOCOL,
                observation_context={'visual': 'intent_withheld', 'audiovisual': 'intent_withheld',
                                     'audio': 'scripted_speech_and_sound_policy_only'},
                evidence_credit='model_media_assessment; not an independently verified success template')
        if editorial:
            review['boundary_evidence'] = boundary
        q = feedback_quality(review)
        review['verdict'] = 'PASS' if q['ok'] else 'FAIL' if q['complete'] else 'UNKNOWN'
        return review
