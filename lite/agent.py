from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

from . import prompts, screen_text
from . import content
from . import closed_loop, asset_actions, evidence_policy, decision_contract, attempt_budget, applicability
from . import character_stages, shot_policy, binding_contract, speech_contract, duration_budget, asset_lifecycle
from . import task_parameters, voice_lineage, completion_flow, completion_plan, completion_review, director_plan
from .assets import AssetRegistry, generator, reference_labels, select_references
from .audio_parameters import AudioParameterError, parameter_signature
from .memory import ExperienceMemory
from .review import ReviewPanel, enabled as feedback_enabled, visual_frames
from .storage import (DIMENSIONS, Pending, RuntimeUnavailable, Unavailable, candidate_quality, clips, read,
                      review_scores, seconds, sha, validate_video_review, write, key)


class FilmAgent:
    def __init__(self, store, models, media, config):
        shot_policy.validate_config(config)
        self.s, self.models, self.media, self.config = store, models, media, config
        self.source = (store.run / 'source.txt').read_text()
        self.guide = (Path(config['lite_root']) / 'resources/ref-en.txt').read_text()
        self.registry = AssetRegistry(store)
        self.registry.lifecycle_config = config
        self.memory = ExperienceMemory(store, config)
        self.feedback_enabled = feedback_enabled(config)

    def prompt(self, value, *, director_layout=False):
        addition = '''\n生成绑定契约：另返回render_spec:{opening,action,camera,closing}，非对白render_spec只用英语写具体表演与摄影，不写任何Subject/Picture/Audio标签或S1，不在这些字段引用完整台词。返回dialogue_timeline与原剧本同序，每项speaker/start_s/end_s/on_screen_or_offscreen，句末至少留0.5秒收尾。宿主从真实参考编译标签与唯一对白，原caption仅存档。不能以缩短镜头掩盖重复/缺尾。''' if binding_contract.enabled(self.config) else ''
        if duration_budget.enabled(self.config):
            budget = (f'\nShort-story hard limit: actual selected prefix plus all future shots <={self.config["max_final_seconds"]:g} seconds at 24fps. '
                      if task_parameters.enabled(self.config) else '\nShort-story hard limit: target 150 seconds, initial plan 135-150 seconds; actual selected prefix plus all future shots <=180 seconds at 24fps. ')
            value += budget+'Preserve complete dialogue and ending. Do not add replay events. Use these exact canonical speaking roles and canonical image/voice identity anchors. Predeclared same-person stage child image IDs are allowed only through character_reference_stages; they do not create a new person or candidate budget. Canonical identity anchors: '+json.dumps(self.config['generation_contract']['cast'],ensure_ascii=False)
            if any(role.get('voice_asset_id') is None for role in self.config['generation_contract']['cast'].values()):
                value += '\nCanonical roles with voice_asset_id:null are non-speaking. Keep their visual references and assign them no dialogue or voice assets.'
        return task_parameters.prompt(value + shot_policy.instructions(self.config, director_layout=director_layout) + shot_policy.contract_instructions(self.config) + character_stages.instructions(self.config) + (prompts.CONTENT_POLICY_V3 if content.enabled(self.config) else '') + (prompts.SINGLE_SOURCE_DIALOGUE if evidence_policy.enabled(self.config) else '') + ('' if director_layout else addition), self.config)

    def screenplay_prompt(self, value):
        if not task_parameters.longform(self.config):
            return self.prompt(value)
        cast = self.config.get('generation_contract', {}).get('cast', {})
        roles = {name: {'description': spec.get('description', ''), 'speaks': spec.get('voice_asset_id') is not None}
                 for name, spec in cast.items()}
        return task_parameters.prompt(value + '\nCanonical cast and speaking roles: '
                                      + json.dumps(roles, ensure_ascii=False), self.config, screenplay=True)

    def artifact(self, name, function):
        path = self.s.run / 'artifacts' / (name + '.json')
        result = read(path)
        if result is None:
            self.s.status('running', stage=name)
            result = function()
            write(path, result)
            self.s.event('stage_completed', stage=name, artifact=str(path))
        return result

    def advisory(self, name, system, user, **options):
        try:
            return self.models.json(name, system, user, **options)
        except Unavailable as exc:
            self.s.warning(name, exc)
            return {'verdict': 'UNKNOWN', 'issues': [str(exc)], 'advisory_unavailable': True}

    def plan(self):
        if self.s.data.get('plan'):
            return
        memories = self.memory.retrieve('global', {'source': self.source}, 'screenplay/initial')
        draft = self.artifact('screenplay_0', lambda: self.models.json('screenplay/initial', self.screenplay_prompt(prompts.WRITER),
            {'source': self.source, 'target_seconds': self.config.get('target_seconds', 300),
             'previous_content_experience': memories}, max_tokens=20000))
        for turn in (1, 2):
            review = self.artifact(f'editor_{turn}', lambda t=turn, d=draft: self.advisory(f'screenplay/editor_{t}', self.screenplay_prompt(prompts.EDITOR),
                {'source': self.source, 'screenplay': d}, max_tokens=6000))
            draft = self.artifact(f'screenplay_{turn}', lambda t=turn, d=draft, r=review: self.models.json(
                f'screenplay/rewrite_{t}', self.screenplay_prompt(prompts.WRITER), {'source': self.source, 'previous': d, 'review': r}, max_tokens=20000))
        self.artifact('screenplay_final_review', lambda: self.advisory('screenplay/final_review', self.screenplay_prompt(prompts.EDITOR),
            {'source': self.source, 'screenplay': draft}, max_tokens=6000))
        handoff = director_plan.longform_handoff(draft, self.config, self.source)
        director_input = {'host_storyboard_requirements': handoff} if handoff else {}

        def validate(value, *, require_duration=True):
            value = director_plan.validate(value, self.config, source_scene_ids=handoff.get('source_scene_seconds', ()))
            director_plan.validate_longform(value['clips'], self.config, handoff, require_duration=require_duration)
            return value
        bound = binding_contract.enabled(self.config)
        director_blueprint = prompts.BOUND_DIRECTOR if bound else prompts.DIRECTOR
        if handoff:
            director_blueprint += prompts.COMPLETE_LONGFORM_STORYBOARD
        director_prompt = self.prompt(director_blueprint, director_layout=bound)
        plan_tokens = self.config['task_contract'].get('director_max_output_tokens', 24000) if task_parameters.longform(self.config) else 24000
        plan = self.artifact('initial_plan', lambda: self.models.json('director/initial', director_prompt,
            {'source': self.source, 'screenplay': draft, 'target_seconds': self.config.get('target_seconds', 300), **director_input},
            max_tokens=plan_tokens, validate=validate))
        initial_duration = duration_budget.director_summary(self.config, plan['clips'], source_scene_ids=handoff.get('source_scene_seconds', ()))
        timing = {'host_plan_duration': initial_duration} if initial_duration is not None else {}
        review = self.artifact('plan_review', lambda: self.advisory('director/review', self.prompt(prompts.EDITOR),
            {'source': self.source, 'plan': plan, **timing, 'content_issues': content.repeated_events(content.bind_events(plan['clips'])) if content.enabled(self.config) else []}, max_tokens=6000))
        if review.get('verdict') != 'PASS' or (content.enabled(self.config) and content.repeated_events(content.bind_events(plan['clips']))):
            try:
                revised = self.artifact('plan_revised', lambda: self.models.json('director/rewrite', director_prompt,
                    {'source': self.source, 'previous': plan, 'review': review, **timing, **director_input,
                     **({'screenplay': draft} if handoff else {})}, max_tokens=plan_tokens,
                    validate=lambda value: validate(value, require_duration=False)))
            except Unavailable as exc:
                self.s.warning('director/rewrite', f'Keeping original executable model plan: {exc}')
            else:
                revised_duration = duration_budget.director_summary(self.config, revised['clips'], source_scene_ids=handoff.get('source_scene_seconds', ()))
                if initial_duration and initial_duration['within_hard_bounds'] and not revised_duration['within_hard_bounds']:
                    self.artifact('plan_revision_duration_decision', lambda: {
                        'selected': 'initial_plan', 'rejected': 'plan_revised',
                        'reason': 'The initial plan meets the hard duration contract; the editorial rewrite does not.',
                        'initial_duration': initial_duration, 'revised_duration': revised_duration})
                else:
                    plan = revised
        if content.enabled(self.config):
            plan['clips'] = content.bind_events(plan['clips'])
        if duration_budget.issues(self.config,plan['clips']):
            plan = self.artifact('plan_duration_correction', lambda: self.models.json('director/duration_correction', director_prompt,
                {'source': self.source, 'previous': plan, 'budget_issues': duration_budget.issues(self.config, plan['clips']),
                 'instruction': task_parameters.duration_correction(self.config), **director_input,
                 **({'screenplay': draft} if handoff else {})}, max_tokens=plan_tokens, validate=validate))
            if content.enabled(self.config):plan['clips']=content.bind_events(plan['clips'])
        plan = character_stages.prepare_plan(plan, self.config)
        binding_contract.validate_source(plan, self.config)
        duration_budget.admit(self.config,plan['clips'])
        if initial_duration is not None:
            self.s.check_stop()
        character_stages.freeze_plan(self.s, plan, self.config)
        self.s.data['plan'] = plan
        self.s.data['plan_revision'] = 1
        for index, asset in enumerate(plan['assets']):
            if not isinstance(asset, dict):
                continue
            ident = str(asset.get('id') or f'asset_{index + 1:03d}')
            self.registry.declare(ident, asset)
        self.s.save()
        self.s.event('plan_published', clips=len(plan['clips']), assets=len(self.s.data['assets']),
                     planned_seconds=sum(c['seconds'] for c in plan['clips']))

    def catalog(self):
        return shot_policy.visible_catalog(character_stages.annotate_catalog(self.registry.catalog(), self.s.data.get('assets', {}), self.s.data.get('plan', {}), self.config), self.config)

    def asset_updates(self, updates, operation=None):
        if completion_flow.enabled(self.config):
            from . import completion_memory, completion_assets
            if completion_memory.enabled(self.config):
                return completion_assets.apply(self, updates, operation)
            return completion_flow.defer_assets(self, updates, operation)
        if self.config.get('selective_repair') and updates:
            raise Pending('Selective repair preserves existing character and voice assets; unsupported new asset proposal', status='needs_binding_clarification')
        if content.closed_loop_enabled(self.config):
            return asset_actions.apply(self, updates, operation or 'asset_intents/'+key(updates))
        for index, raw in enumerate(updates if isinstance(updates, list) else []):
            if not isinstance(raw, dict) or not any(raw.get(k) for k in ('prompt', 'description', 'path')):
                continue
            ident = str(raw.get('id') or f'new_asset_{len(self.s.data["assets"]) + index + 1}')
            self.registry.declare(ident, raw)

    def ensure_assets(self, only_ids=None, *, optional=False):
        def remember(stage, operation, outcome, details):
            if optional:
                from .completion_memory import export
                return export(self,operation,outcome,details,stage=stage)
            return self.memory.record(stage,operation,outcome,details)
        # VoiceDesign/imports precede voice clones even if the director listed them later.
        pending = sorted(list(self.s.data['assets'].items()), key=lambda pair: generator(pair[1]['spec']) == 'voice_clone')
        for ident, row in pending:
            if only_ids is not None and ident not in only_ids:
                continue
            if only_ids is None and completion_flow.enabled(self.config) and ident not in completion_flow.required_ids(self):
                continue
            prompt = row.get('pending_prompt')
            if not prompt:
                continue
            takes=(1,2)
            if asset_lifecycle.enabled(self.config):
                from . import calibration_recovery, asset_permissions
                if calibration_recovery.adopt(self,ident,row):continue
                from .explicit_recovery import adopt_retained_voice
                if adopt_retained_voice(self,ident,row):continue
                if asset_lifecycle.reconcile_blocked_voice_observation(self,ident,row):continue
                if asset_permissions.generation_blocked(self,ident,row):
                    if not optional:
                        raise Pending('Reuse-only voice permission forbids new generation',status='asset_generation_forbidden')
                    row.update(dependency_state='optional_unavailable',pending_prompt=None,
                               last_generation_failure='Reuse-only authorization forbids descendant generation')
                    self.registry.save()
                    self.s.event('retained_voice_generation_blocked',asset_id=ident,new_generation=False)
                    continue
                takes=asset_lifecycle.candidate_takes(self,ident,row)
            if shot_policy.no_score(self.config) and shot_policy.is_music(row['spec']):
                self.s.event('asset_omitted_by_music_policy', asset_id=ident, generator=generator(row['spec']),
                             reason='Explicit no-score experiment; raw asset proposal retained, no GPU attempt submitted')
                row['pending_prompt'] = None
                self.registry.save()
                continue
            signature = parameter_signature({**row['spec'], 'generator': generator(row['spec'])})
            if row.get('parameter_failure', {}).get('signature') == signature:
                row['pending_prompt'] = None
                self.registry.save()
                continue
            version = row.get('pending_version') or row.get('generation', 0) + 1
            row['pending_version'] = version
            self.registry.save()
            history_key = f'asset/{ident}/version_{version}'
            spec = row['spec'];mode = generator(spec)
            reference = next((a for a in self.catalog() if a['id'] == spec.get('reference_asset_id')), None)
            experience = self.memory.retrieve('asset', spec, history_key)
            if experience and not optional:
                prepared = self.artifact(history_key.replace('/', '_') + '_prepare', lambda: self.advisory(
                    history_key + '/prepare', prompts.ASSET,
                    {'asset': spec, 'previous_prompt': prompt, 'content_experience': experience,
                     'actual_asset_catalog': self.catalog()}))
                if isinstance(prepared.get('prompt'), str) and prepared['prompt'].strip():
                    prompt = prepared['prompt']
                proposed = next((a for a in self.catalog() if a['id'] == prepared.get('reference_asset_id')
                                 and a.get('track_type') in ('synthetic_voice', 'isolated_voice')), None)
                if mode == 'voice_clone' and proposed:
                    reference = proposed
            stage_parent = character_stages.generation_parent(self, ident)
            parents, successful = ([stage_parent] if stage_parent else row['versions'][-1:]), []
            age_edit = (character_stages.age_edit_mode(spec) if stage_parent and mode == 'qwen_image'
                        and self.config.get('image_provider', {}).get('model') == 'Qwen-Image-2.1' else None)
            for take in takes:
                name = f'{version}_{take}_{__import__("hashlib").sha256(ident.encode()).hexdigest()[:12]}'
                self.s.status('running', stage='asset.generate', asset=ident, version=version, generator=mode)
                try:
                    from .asset_permissions import assert_generation_allowed
                    assert_generation_allowed(self.config,self.s,{**spec,'id':ident},reference)
                    character_stages.assert_generation_allowed(self, ident, spec)
                    generation_prompt = prompt
                    if mode == 'qwen_image' and spec.get('character_stage'):
                        generation_prompt += '\nHOST-OWNED CHARACTER STAGE (takes priority over incompatible optional repair prose): ' + spec['description'] + '. Preserve this same person identity; do not revert to another life stage.'
                    if mode == 'qwen_image':
                        if age_edit:
                            generation_prompt = character_stages.age_edit_prompt(spec['description'], age_edit, prompt)
                            asset = self.media.image(name, generation_prompt, parents=parents,
                                                     seed=42 + version * 10 + take, age_edit=age_edit)
                        else:
                            asset = self.media.image(name, generation_prompt, parents=parents, seed=42 + version * 10 + take)
                    elif mode == 'import':
                        asset = self.media.import_asset(name, spec)
                    else:
                        asset = self.media.audio(name, {**spec, 'prompt': prompt}, reference, seed=42 + version * 10 + take)
                except AudioParameterError as exc:
                    if asset_lifecycle.enabled(self.config):asset_lifecycle.record_attempt(row,version,take,failure=exc)
                    row['parameter_failure'] = {'signature': signature, 'reason': str(exc)}
                    row['last_generation_failure'] = str(exc)
                    self.s.warning(history_key, exc)
                    remember('asset', history_key + f'/take_{take}/parameter_failure', 'technical_failure',
                                       {'asset': spec, 'reason': str(exc), 'generator': mode,
                                        'retry_requires_parameter_change': True})
                    break
                except Unavailable as exc:
                    if asset_lifecycle.enabled(self.config):asset_lifecycle.record_attempt(row,version,take,failure=exc)
                    row['last_generation_failure'] = str(exc)
                    self.s.warning(history_key, exc)
                    remember('asset', history_key + f'/take_{take}/failure', 'technical_failure',
                                       {'asset': spec, 'reason': str(exc), 'generator': mode})
                    if isinstance(exc,RuntimeUnavailable):
                        self.registry.save()
                        raise Pending('Local runtime failure retained; repair the runtime before another asset candidate: '+str(exc),status='needs_runtime_repair') from exc
                    continue
                kind = asset['kind']
                reviewer = 'doubao' if self.feedback_enabled and kind in ('image', 'video') else 'gemini'
                asset_media = [{'path': asset['path'], 'label': 'Actual reference asset'}]
                if (optional or stage_parent) and kind == 'image' and parents:
                    from pathlib import Path
                    asset_media += [{'path':p['path'],'label':'Retained parent reference; preserve character identity if present'}
                        for p in parents[:1] if Path(p.get('path','')).is_file()]
                if self.feedback_enabled and kind == 'video':
                    frames = visual_frames(self.config, self.s.run, asset, 'asset_' + name)
                    asset_media = [{'path': f['path'], 'label': f"Silent reference video frame {f['time_s']:.3f}s"}
                                   for f in frames['frames']]
                asset_prompt = prompts.ASSET_FEEDBACK if self.feedback_enabled else (
                    prompts.IMAGE_REVIEW if kind == 'image' else prompts.AUDIO_REVIEW if kind == 'audio' else prompts.REFERENCE_VIDEO_REVIEW)
                if stage_parent:
                    asset_prompt += '\nCompare the actual generated character image with the supplied canonical parent. Report same-person facial identity and requested life-stage/wardrobe separately with concrete visible evidence. Provenance alone is not visual identity or age acceptance.'
                if age_edit:
                    asset_prompt += character_stages.AGE_REVIEW_GUIDE
                review_target=asset_lifecycle.review_target(spec,self.config)
                if review_target.get('calibration_policy')==asset_lifecycle.CALIBRATION:
                    asset_prompt+=asset_lifecycle.CALIBRATION_REVIEW_GUIDE
                review = self.artifact(f'asset_review_{name}', lambda: self.advisory(
                    history_key + f'/review_{take}', asset_prompt,
                    {'requested_asset': review_target, 'generation_is_only_a_target': prompt, 'modality': kind,
                     **({'sound_purity_required':'Report speech_present and human_vocalization as booleans, background_music and actual observations. No human voices allowed.'} if optional and spec.get('role') in ('music','ambience') else {})},
                    provider=reviewer, media=asset_media, max_tokens=3500))
                review=asset_lifecycle.clarify_voice_review(self,'asset_review_'+name,spec,asset,review,asset_media)
                if mode == 'import' and kind == 'audio' and spec.get('role') == 'voice' and review.get('content_type') == 'single_speaker_voice' and review.get('background_music') is False:
                    asset['track_type'] = 'isolated_voice'
                actual = {**asset, 'generator': mode, 'review': review, 'prompt': generation_prompt, 'version': version,
                          'effective_reference_asset_id': (reference or {}).get('id')}
                actual = character_stages.stamp_generated(self, ident, actual, stage_parent)
                successful.append(actual)
                remember('asset', history_key + f'/take_{take}', 'observed_asset_review',
                    {'asset_spec': spec, 'receipt': asset, 'review': review, 'previous_versions': parents,
                     'retrieved_memory_ids': [m['id'] for m in experience]})
                score = review.get('score')
                asset_ok = (review.get('decision') == 'accept' and bool(review.get('reason'))
                            and bool(review.get('observed_events'))) if self.feedback_enabled else (
                            review.get('verdict') == 'PASS' and isinstance(score, (int, float)) and score >= 4)
                if asset_lifecycle.enabled(self.config) and spec.get('role')=='voice':asset_ok=asset_ok and asset_lifecycle.voice_report_usable(actual,review)
                if optional and spec.get('role') in ('music','ambience'):
                    asset_ok = asset_ok and review.get('speech_present') is False and review.get('human_vocalization') is False
                    if spec['role']=='ambience':asset_ok=asset_ok and review.get('background_music') is False
                if age_edit:
                    actual['character_age_defects'] = character_stages.age_review_defects(review)
                    asset_ok = asset_ok and not actual['character_age_defects']
                actual['feedback_accepted'] = asset_ok
                if asset_lifecycle.enabled(self.config):asset_lifecycle.record_attempt(row,version,take,actual=actual)
                if asset_ok:
                    break
                if take < max(takes) and mode != 'import':
                    change = self.advisory(history_key + '/revise', prompts.ASSET,
                        {'asset': spec, 'review': review, 'previous_prompt': prompt, 'content_experience': experience,
                         'actual_asset_catalog': self.catalog()})
                    prompt = change.get('prompt') or prompt
                    if mode == 'voice_clone' and not optional:
                        proposed = next((a for a in self.catalog() if a['id'] == change.get('reference_asset_id')
                                         and a.get('track_type') in ('synthetic_voice', 'isolated_voice')), None)
                        if proposed:
                            reference = proposed
                    if kind == 'image':
                        parents = ([stage_parent] if age_edit else
                                   [stage_parent, asset] if stage_parent else [asset])
            if asset_lifecycle.enabled(self.config):
                asset_lifecycle.finish_asset(self,ident,row,successful,version,optional=optional)
                continue
            successful = [v for v in successful if not v.get('character_age_defects')]
            if successful:
                best = (next((v for v in reversed(successful) if v.get('feedback_accepted')), successful[-1])
                        if self.feedback_enabled else max(successful, key=lambda v: v['review'].get('score') if isinstance(v['review'].get('score'), (int, float)) else -1))
                self.registry.publish(ident, best)
                row.pop('parameter_failure', None)
                row.pop('last_generation_failure', None)
            else:
                self.s.warning(history_key, 'No new usable asset; retaining previous references without inventing media')
            row['pending_prompt'] = None;row['generation'] = version;row['pending_version'] = None
            self.registry.save()

        if not optional:asset_lifecycle.admit_assets(self)
        if content.closed_loop_enabled(self.config):
            asset_actions.settle(self)

    def references(self, ids, plan=None):
        plan = plan or {}
        catalog = self.catalog()
        wanted = set(str(i) for i in ids) if isinstance(ids, list) else set()
        windows = plan.get('reference_windows')
        windows = {str(w.get('id')): w for w in windows if isinstance(w, dict)} if isinstance(windows, list) else {}
        counts = {kind: sum(a['kind'] == kind and a['id'] in wanted for a in catalog) for kind in ('audio', 'video')}
        prepared = []
        for asset in catalog:
            if asset['id'] in wanted and asset['kind'] in counts and hasattr(self.media, 'reference_excerpt'):
                window = windows.get(asset['id'], {})
                share = 15 / max(1, min(3, counts[asset['kind']]))
                try:
                    asset = self.media.reference_excerpt(asset, duration=window.get('seconds', share), start=window.get('start_s', 0))
                except Unavailable as exc:
                    self.s.warning('reference_window/' + asset['id'], exc)
                    continue
            prepared.append(asset)
        anchors = {} if shot_policy.enabled(self.config) else {
            'first_frame_id': plan.get('first_frame_id'), 'last_frame_id': plan.get('last_frame_id')}
        result = voice_lineage.annotate(select_references(prepared, ids, **anchors), catalog, self.s.data, self.config)
        from . import completion_memory, completion_assets
        return completion_assets.annotate_sounds(self,result) if completion_memory.enabled(self.config) else result

    def ensure_voice_observation(self, candidate):
        review, clip = candidate['review'], candidate.get('source_clip', candidate.get('clip', {}))
        scores = review_scores(review)
        if not isinstance(review, dict) or 'speaker_consistency' not in scores:
            return
        if (clip.get('dialogue') or isinstance(review.get('voice_observation'), dict)
                or candidate.get('voice_observation') is not None
                or scores.get('speaker_consistency') is not None):
            return
        # Empty transcript is not proof of absence. Old reviews lacking an
        # explicit observation get one separate real-audio observation, once.
        cid, take = clip['id'], candidate['take']
        observation = self.artifact(f'voice_observation_{cid}_{take}', lambda: self.advisory(
            f'voice_observation/{cid}/{take}', prompts.VOICE_OBSERVATION,
            {'task': 'Observe the complete actual native audio track; do not rescore the candidate.'},
            provider='gemini', media=[{'path': candidate['receipt']['audio_path'], 'label': 'Complete actual native soundtrack'}],
            max_tokens=1500))
        candidate['voice_observation'] = observation
        self.s.save()

    def refresh_quality(self):
        for candidates in self.s.data['candidates'].values():
            for candidate in candidates:
                if not candidate.get('review') or not candidate.get('receipt'):
                    continue
                self.ensure_voice_observation(candidate)
                self._update_quality(candidate)
        for selected in self.s.data['selected']:
            cid = selected['clip']['id']
            candidate = next((c for c in self.s.data['candidates'].get(cid, [])
                              if c['take'] == selected['take'] and c.get('receipt', {}).get('sha256') == selected['receipt']['sha256']), None)
            if candidate and candidate.get('voice_observation') is not None:
                selected['voice_observation'] = deepcopy(candidate['voice_observation'])
            self._update_quality(selected)
        self.s.save()

    def repair_retained_decisions(self):
        """Explicit engineering receipt: exact selected review, no media regeneration."""
        if not decision_contract.enabled(self.config):
            return
        for task in self.config.get('review_system', {}).get('retained_decision_repairs', []):
            self.s.check_stop()
            operation = 'decision_contract/' + decision_contract.digest(task)
            receipts = self.s.data.setdefault('decision_repair_receipts', {})
            if operation in receipts:
                continue
            cid, take = task['clip_id'], task['take']
            candidate = next(c for c in self.s.data['candidates'][cid] if c['take'] == take)
            if (candidate['receipt']['sha256'] != task['media_sha256']
                    or decision_contract.digest(candidate['review']) != task['review_sha256']):
                from .storage import Pending
                raise Pending('Retained review repair differs from its exact authorized snapshot')
            before = deepcopy(candidate['review'])
            repaired = decision_contract.reconcile(self, cid, candidate, before)
            history = {'operation': operation, 'before': before, 'before_sha256': task['review_sha256'],
                       'after_sha256': decision_contract.digest(repaired), 'media_unchanged': True,
                       'framework_release': self.config.get('framework_release')}
            candidate.setdefault('review_history', []).append(history)
            candidate['review'] = repaired
            self._update_quality(candidate)
            for selected in self.s.data['selected']:
                if selected['clip']['id'] == cid and selected['take'] == take and selected['receipt']['sha256'] == task['media_sha256']:
                    selected.setdefault('review_history', []).append(deepcopy(history))
                    selected['review'] = deepcopy(repaired)
                    self._update_quality(selected)
                    selected.setdefault('selection_record', {}).update(candidate_decision=repaired['feedback_decision']['decision'],
                        remaining_issues=selected['quality']['unresolved_findings'],
                        decision_repair_operation=operation)
            receipts[operation] = {k: v for k, v in history.items() if k != 'before'}
            self.s.save()
            self.s.event('retained_decision_repaired', clip_id=cid, take=take, receipt=receipts[operation],
                         quality=candidate['quality'], original_model_artifacts_unchanged=True)
            self.memory.record('shared', operation, 'engineering_report_correction',
                {'lesson': '修复报告字段冲突不等于修复媒体。当前已消除的问题不用fix_local；呼吸等非言语不要求对白清晰度。实际口型、声源绑定和真实缺陷仍须原观察证据。',
                 'receipt': receipts[operation], 'quality': candidate['quality'], 'new_media_generated': False,
                 'media_improvement_claim': 'none; same original bytes and observations'})

    def _update_quality(self, candidate):
        value = candidate_quality(candidate)
        if candidate.get('quality') != value:
            before = deepcopy(candidate.get('quality'))
            candidate.setdefault('quality_history', []).append({'time': time.time(), 'before': before,
                'after': deepcopy(value),
                'reason': 'derived_quality_recomputed' if content.closed_loop_enabled(self.config) else 'voice_dimension_applicability',
                'framework_release':self.config.get('framework_release'),
                'original_review_unchanged':True})
            candidate['quality'] = value
            clip = candidate.get('source_clip', candidate.get('clip', {}))
            self.s.event('quality_applicability_updated', clip_id=clip.get('id'), take=candidate['take'],
                         before=before, after=value, original_review_unchanged=True)

    def render_clip(self, clip):
        if completion_flow.enabled(self.config):
            clip = completion_flow.execution_clip(self, clip)
        self._completion_clip = clip
        asset_lifecycle.admit_assets(self)
        cid = clip['id']
        candidates = self.s.data['candidates'].setdefault(cid, [])
        from . import dialogue_integrity
        if dialogue_integrity.enabled(self.config):
            for item in candidates:
                if item.get('review'):
                    item['dialogue_integrity_audit'] = dialogue_integrity.audit_candidate(item)
            self.s.save()
        window_start = attempt_budget.start(self.s, cid)
        for take in attempt_budget.slots(self.s, self.config, cid):
            self.s.check_stop()
            existing = next((c for c in candidates if c['take'] == take), None)
            if existing and (existing.get('review') or existing.get('technical_failure')):
                if existing.get('review'):
                    self.learn_local(cid,candidates,existing)
                if existing.get('review') and candidate_quality(existing)['ok']:
                    break
                continue
            if take > attempt_budget.limit(self.config) and window_start == 1:
                folder = self.s.run/'media/videos'/f'{cid}_take_{take}'
                if not existing or not existing.get('receipt') and not (folder/'submitted.json').exists():
                    self.s.event('video_budget_prevents_unsubmitted_candidate', clip_id=cid, take=take)
                    continue
            if existing is not None and completion_flow.enabled(self.config):
                completion_flow.reconcile_prepared_direction(self, clip, existing)
            self.s.status('running', stage='clip.local_plan', clip_id=cid, take=take)
            plan_name = f'local_{cid}_{take}'
            def propose():
                return self.advisory(plan_name, self.prompt((prompts.LOCAL_V3 + (prompts.LOCAL_CLOSED_LOOP if content.closed_loop_enabled(self.config) else '')) if content.enabled(self.config) else prompts.LOCAL + (prompts.FEEDBACK_LOCAL if self.feedback_enabled else '')), {
                    'clip': clip, 'actual_asset_catalog': self.catalog(),
                    'runtime_capabilities':attempt_budget.capabilities(self.s,cid,take) if content.closed_loop_enabled(self.config) else None,
                    'editing_policy': shot_policy.context(self.config),
                    'unavailable_assets': self.registry.unavailable(), 'h3_guide': self.guide,
                    'selected_prefix_observations': (content.prefix_facts(self.s.data['selected'][-3:]) if content.enabled(self.config) else [{'clip': s['clip'], 'review': s['review']} for s in self.s.data['selected'][-3:]]),
                    'speech_budget': content.speech_budget(clip, self.catalog()) if content.enabled(self.config) else None,
                    'actual_candidates': (evidence_policy.memory_view if evidence_policy.enabled(self.config) else lambda x:x)([{**{k: c[k] for k in ('take', 'plan', 'review', 'technical_failure') if k in c}, **({'quality':candidate_quality(c)} if c.get('review') else {}), **({'actual_media_facts':evidence_policy.media_facts(c)} if evidence_policy.enabled(self.config) and c.get('receipt') else {})} for c in candidates]),
                    'content_experience': self.memory.retrieve('local', {'clip': clip, 'reviews': [c.get('review') for c in candidates]}, plan_name)}, max_tokens=10000)
            if (task_parameters.longform(self.config) and existing is not None
                    and completion_flow.enabled(self.config) and isinstance(existing.get('plan'), dict)):
                # The saved plan already binds the caption and H3 request. In
                # particular, retain its admitted length after an interruption.
                plan = deepcopy(existing['plan'])
            else:
                plan = (completion_flow.generation_plan(self, clip, take, plan_name, candidates)
                        if completion_flow.enabled(self.config) else self.artifact(plan_name, propose))
            if completion_flow.enabled(self.config):
                plan = completion_flow.stop_plan(plan, candidates)
            if not completion_flow.enabled(self.config) and content.closed_loop_enabled(self.config) and plan.get('decision')=='select_best' and plan.get('stop_reason') not in ('accepted','no_useful_local_change','attempts_exhausted'):
                plan = self.artifact(plan_name+'_stop_reconciliation_v4',lambda: self.advisory(
                    plan_name+'/stop_reconciliation_v4',self.prompt(prompts.LOCAL_V3+prompts.LOCAL_CLOSED_LOOP),
                    {'clip':clip,'original_stop_proposal':plan,'runtime_capabilities':attempt_budget.capabilities(self.s,cid,take),
                     'actual_candidates':[{'take':c['take'],'quality':candidate_quality(c),'review':c['review']} for c in candidates if c.get('review')],
                     'instruction':'Clarify stop reason against retained candidates; no assumption that a new take prevents rollback.'},max_tokens=10000))
            if not completion_flow.enabled(self.config) and content.enabled(self.config) and plan.get('decision') != 'select_best':
                plan = content.correct_local_plan(self, clip, plan, plan_name)
            if plan.get('decision') == 'select_best' and any(c.get('receipt') and c.get('review') for c in candidates):
                self.s.event('local_stop', clip_id=cid, take=take, reason=plan.get('reason'),
                             stop_reason=plan.get('stop_reason','legacy_model_stop'),
                             remaining_issues=[{'take':c['take'],'issues':candidate_quality(c).get('unresolved_findings',[])} for c in candidates if c.get('review')])
                break
            if completion_flow.enabled(self.config) and existing is None:
                plan = completion_flow.compile_plan(self, clip, plan)
            applied = plan_name + '_assets'
            if not self.s.data.get(applied):
                self.asset_updates(plan.get('asset_updates', []), operation=plan_name+'/assets')
                self.s.data[applied] = True
                self.s.save()
            if clip.get('_replan_direction') or clip.get('_replan_endpoints'):
                # Optional asset operations modify the stored future clip. Re-project
                # after them to keep the new image/voice bindings in the actual take.
                current = next(c for c in self.s.data['plan']['clips'] if c['id'] == cid)
                clip = self._completion_clip = completion_flow.execution_clip(self, current)
            self.ensure_assets()
            from . import completion_memory
            if completion_memory.enabled(self.config) and existing is None:
                plan['asset_ids'] = completion_flow.planned_reference_ids(self, clip, plan)
            caption = plan.get('caption')
            if not isinstance(caption, str) or not caption.strip():
                caption = json.dumps(clip, ensure_ascii=False)
                self.s.warning(plan_name, 'Using retained model-authored clip description; enhanced caption unavailable')
            duration = seconds(plan.get('seconds'), clip['seconds'])
            if binding_contract.enabled(self.config):
                duration = max(duration, clip.get('minimum_seconds', 4))
            duration_budget.local(self,clip,duration)
            references = deepcopy(existing['references']) if existing is not None else self.references(plan.get('asset_ids', clip['asset_ids']), plan)
            requested = plan.get('asset_ids', clip['asset_ids'])
            if not completion_flow.enabled(self.config) and existing is None and isinstance(requested, list) and requested != [r['id'] for r in references]:
                bound = self.artifact(f'caption_binding_{cid}_{take}', lambda: self.advisory(
                    f'caption_binding/{cid}/{take}', self.prompt(prompts.CAPTION_BINDING),
                    {'clip': clip, 'proposed_caption': caption, 'actual_reference_bindings': reference_labels(references),
                     'h3_guide': self.guide}))
                if isinstance(bound.get('caption'), str) and bound['caption'].strip():
                    caption = bound['caption']
                else:
                    self.s.warning(plan_name, 'Reference selection changed; caption binding correction unavailable, retaining model target for real review')
            source = {**deepcopy(clip), 'seconds': duration}
            if shot_policy.enabled(self.config) and isinstance(plan.get('shot_design'), dict):
                source['shot_design'] = deepcopy(plan['shot_design'])
            if existing is None:
                caption = shot_policy.effective_caption(caption, source, self.config)
                compilation = None
                if evidence_policy.enabled(self.config):
                    caption, compilation = evidence_policy.compile_caption(caption, source)
                existing = {'take': take, 'plan': plan, 'caption': caption, 'source_clip': source,
                            'dialogue_integrity_version': self.config.get('completion_policy', {}).get('dialogue_integrity'),
                            'references': references, 'plan_revision': self.s.data['plan_revision']}
                if compilation is not None:
                    existing.update(caption_compilation=compilation, generation_policy_version=evidence_policy.VERSION)
                if binding_contract.enabled(self.config):
                    try:
                        caption, binding = binding_contract.compile(source, references, self.config, plan)
                    except voice_lineage.VoiceLineageError as exc:
                        raise Pending('Host voice lineage validation failed: '+str(exc),status='needs_binding_clarification') from exc
                    except ValueError as exc:
                        binding_error = str(exc)
                        def validate_binding(value):
                            corrected_plan = {**plan, 'render_spec':value.get('render_spec'), 'dialogue_timeline':value.get('dialogue_timeline')}
                            binding_contract.compile(source, references, self.config, corrected_plan)
                            return value
                        try:
                            corrected = self.artifact(plan_name+'_binding_correction', lambda:self.models.json(
                                plan_name+'/binding_correction', self.prompt(prompts.LOCAL_V3),
                                {'current_source_event':source,'previous_local_plan':plan,'actual_reference_bindings':reference_labels(references),
                                 'binding_error':binding_error,'instruction':'Return render_spec and dialogue_timeline only. Preserve current source dialogue, roles, duration and references; no free reference labels.'},
                                max_tokens=6000,validate=validate_binding))
                            validate_binding(corrected)
                            plan = {**plan, 'render_spec':corrected['render_spec'], 'dialogue_timeline':corrected['dialogue_timeline']}
                            caption,binding=binding_contract.compile(source,references,self.config,plan)
                        except (ValueError,Unavailable) as error:
                            raise Pending('Bounded generation binding correction unresolved: '+str(error),status='needs_binding_clarification') from error
                    existing.update(caption=caption, binding_contract=binding, plan=plan)
                    existing['proposed_caption_compilation']=existing.pop('caption_compilation',None)
                if shot_policy.enabled(self.config):
                    existing['editing_intent'] = shot_policy.intent(source, self.config)
                if completion_flow.enabled(self.config):
                    existing['intervention']=completion_flow.candidate_intervention(self,clip,existing,candidates)
                candidates.append(existing)
                self.s.save()
            self.registry.used(existing['references'], cid, take)
            self.s.status('running', stage='clip.render', clip_id=cid, take=take)
            try:
                if not existing.get('receipt'):
                    attempt_budget.admit_gpu(self.s, self.config, cid, take)
                    existing['receipt'] = self.media.render(f'{cid}_take_{take}', existing['caption'],
                        existing['source_clip']['seconds'], existing['references'], seed=10000 + int(cid.rsplit('_', 1)[1]) * 100 + take,
                        **({'dialogue_source':existing['source_clip']} if existing.get('generation_policy_version')==evidence_policy.VERSION else {}),
                        **({'binding':existing['binding_contract']} if existing.get('binding_contract') else {}))
                    if completion_flow.enabled(self.config) and existing.get('intervention'):
                        operation=existing['intervention']
                        request=self.s.run/'media/videos'/f'{cid}_take_{take}'/'request.json'
                        operation.update(status='executed',media_sha256=existing['receipt']['sha256'],
                            audio_sha256=existing['receipt'].get('audio_sha256'),
                            actual_request_sha256=sha(request) if request.is_file() else None)
                    self.s.save()
                    self.s.event('media_completed', clip_id=cid, take=take, receipt=existing['receipt'])
            except Unavailable as exc:
                existing['technical_failure'] = str(exc)
                self.s.save()
                self.s.warning(f'{cid}/take_{take}', exc)
                failed_observation={'clip': existing['source_clip'], 'caption': existing['caption'], 'references': existing['references'], 'reason': str(exc)}
                from . import completion_memory
                if completion_memory.enabled(self.config):
                    completion_memory.export(self,f'{cid}/take_{take}/technical_failure','technical_failure',failed_observation)
                else:
                    self.memory.record('local', f'{cid}/take_{take}/technical_failure', 'technical_failure',failed_observation)
                if isinstance(exc,RuntimeUnavailable):
                    raise Pending('Local runtime failure retained; no content retry in the unchanged runtime: '+str(exc),status='needs_runtime_repair') from exc
                continue
            receipt = existing['receipt']
            if completion_flow.enabled(self.config):
                completion_plan.prepare(self)
            duration_budget.local(self,clip,receipt['seconds'],receipt)
            media = [{'path': receipt['path'], 'label': 'Current actual video to judge'},
                     {'path': receipt['audio_path'], 'label': 'Current actual native soundtrack; listen for exact dialogue'}]
            if self.s.data['selected']:
                media.append({'path': self.s.data['selected'][-1]['receipt']['path'], 'label': 'Previous selected clip for continuity only'})
            media += [{'path': r['path'], 'label': 'Actual requested identity/setting reference: ' + r['id']}
                      for r in existing['references'] if r['kind'] == 'image'][:3]
            self.s.status('running', stage='clip.review', clip_id=cid, take=take)
            if self.feedback_enabled:
                closed_loop.bind_review_protocol(self.s,cid,existing,self.config.get('review_system',{}).get('observation_protocol'),
                    evidence_policy=self.config.get('review_system',{}).get('evidence_policy'))
                applicability.bind(self.s,cid,existing,self.config)
                self.s.save()
                review = self.artifact(f'review_{cid}_{take}', lambda: (completion_review.review(self, cid, existing)
                    if completion_flow.enabled(self.config) and completion_review.can_start(self,cid,take) else ReviewPanel(self).review(cid, existing)))
            else:
                review = self.artifact(f'review_{cid}_{take}', lambda: self.advisory(f'review/{cid}/{take}', self.prompt(prompts.VIDEO_REVIEW),
                    {'target': existing['source_clip'], 'caption_is_only_a_target': existing['caption'],
                     'reference_assets': [{'id': r['id'], 'description': r.get('description')} for r in existing['references']]},
                    provider='gemini', media=media, max_tokens=7000, validate=validate_video_review))
            existing['review'] = review
            self.ensure_voice_observation(existing)
            if dialogue_integrity.enabled(self.config):
                existing['dialogue_integrity_audit'] = dialogue_integrity.audit_candidate(existing)
            existing['quality'] = candidate_quality(existing)
            self.s.save()
            self.learn_local(cid, candidates, existing)
            observed = {'clip': existing['source_clip'], 'caption': existing['caption'], 'references': existing['references'],
                        'receipt': receipt, 'review': review, 'quality': existing['quality']}
            if completion_flow.enabled(self.config):
                completion_flow.record_observation(self, f'{cid}/take_{take}/review', observed)
            else:
                self.memory.record('local', f'{cid}/take_{take}/review', 'observed_candidate', observed)
            self.s.event('review_completed', clip_id=cid, take=take, quality=existing['quality'])
            if existing['quality']['ok']:
                break
        usable = [c for c in candidates if c.get('receipt') and c.get('review')]
        if speech_contract.enabled(self.config) and not completion_flow.enabled(self.config):
            eligible = [c for c in usable if candidate_quality(c)['ok']]
            if not eligible:
                write(self.s.run / 'drafts' / (cid + '.json'), {'clip_id':cid,'candidates':[{'take':c['take'],'quality':candidate_quality(c)} for c in usable], 'release_blocked':True})
                raise Pending('All retained candidates have unresolved content; saved as draft, not accepted: ' + cid, status='content_release_blocked')
            usable = eligible
        if not usable:
            raise Unavailable(f'All {attempt_budget.limit(self.config)} authorized candidate slots lack usable media: {cid}; original history retained')
        if completion_flow.enabled(self.config):
            best = completion_flow.choose(usable)
            selection = {'take':best['take'],'reason':'Host selects retained actual media by acceptance, confirmed defects and remaining uncertainties; content warnings preserved.'}
        else:
            selection = None
            if self.feedback_enabled and len(usable) > 1:
                selection = self.artifact(f'selection_{cid}', lambda: self.advisory(f'selection/{cid}', self.prompt(prompts.FEEDBACK_SELECTION),
                    {'target': clip, 'candidates': [{'take': c['take'], 'review': evidence_policy.memory_view(c['review']) if evidence_policy.enabled(self.config) else c['review'], 'quality': evidence_policy.memory_view(candidate_quality(c)) if evidence_policy.enabled(self.config) else candidate_quality(c), **({'actual_media_facts':evidence_policy.media_facts(c)} if evidence_policy.enabled(self.config) else {})} for c in usable],
                     'runtime_capabilities':attempt_budget.capabilities(self.s,cid),
                     'selection_is_not_candidate_acceptance':True}))
                best = next((c for c in usable if c['take'] == selection.get('take')),
                            max(usable,key=lambda c:(candidate_quality(c)['ok'],c['take'])))
                if not any(c['take'] == selection.get('take') for c in usable):
                    best['selection_uncertainty'] = 'Selection unavailable; retained best known acceptance status then latest real candidate; selection remains uncertain'
            else:
                best = max(usable, key=lambda c: (candidate_quality(c)['ok'], candidate_quality(c)['rank'] or -1))
        selected = {'clip': deepcopy(best['source_clip']), 'take': best['take'], 'receipt': best['receipt'],
                    'review': best['review'], 'quality': candidate_quality(best),
                    'candidate_count': len(candidates), 'plan_revision': best['plan_revision']}
        if content.closed_loop_enabled(self.config):
            selected['selection_record'] = {'selected_take':best['take'],
                'candidate_decision':best['review'].get('feedback_decision',{}).get('decision'),
                'selection_reason':selection.get('reason') if selection else 'Selected retained real candidate by quality status',
                'remaining_issues':selected['quality'].get('unresolved_findings',[]),
                'stop_reason':'accepted' if selected['quality']['ok'] else 'attempts_exhausted' if attempt_budget.capabilities(self.s,cid)['remaining_unreserved_slots']==0 else 'model_stopped_with_issues',
                'all_candidates_retained':True, 'selection_does_not_rewrite_review':True}
        if best.get('selection_uncertainty'):
            selected['selection_uncertainty'] = best['selection_uncertainty']
            selected['quality']['ok'] = False
            selected['quality']['quality_status'] = 'draft/needs_review'
        if best.get('voice_observation') is not None:
            selected['voice_observation'] = deepcopy(best['voice_observation'])
        if best.get('editing_intent') is not None:
            selected['editing_intent'] = deepcopy(best['editing_intent'])
        if not any(s['clip']['id'] == cid for s in self.s.data['selected']):
            self.s.data['selected'].append(selected)
            if not shot_policy.enabled(self.config):
                try:
                    audio = self.media.native_audio_asset(cid, best['receipt'])
                    self.s.data['assets'][audio['id']] = {'spec': audio, 'versions': [audio], 'pending_prompt': None}
                except Unavailable as exc:
                    self.s.warning(cid + '/optional_audio_reference', exc)
            if not shot_policy.enabled(self.config) and hasattr(self.media, 'continuity_assets'):
                try:
                    for asset in self.media.continuity_assets(cid, best['receipt']):
                        self.registry.declare(asset['id'], asset)
                        self.s.data['assets'][asset['id']]['pending_prompt'] = None
                        self.registry.publish(asset['id'], asset)
                except Unavailable as exc:
                    self.s.warning(cid + '/optional_continuity', exc)
            if completion_flow.enabled(self.config):completion_flow.realized_prefix(self)
            self.registry.save()
            self.s.event('clip_selected', clip_id=cid, take=best['take'], quality=selected['quality'])

    def learn_local(self, cid, candidates, current):
        if completion_flow.enabled(self.config):
            return completion_flow.learn(self, cid, candidates, current)
        if any(l.get('clip_id')==cid and l.get('take')==current['take'] for l in self.s.data['local_lessons']):
            return
        previous = [c for c in candidates if c.get('review') and c['take'] < current['take']]
        if not previous:
            return
        parent = previous[-1]
        protocol_changed=(parent['review'].get('observation_protocol') != current['review'].get('observation_protocol')
            or parent['review'].get('review_context',{}).get('evidence_policy') != current['review'].get('review_context',{}).get('evidence_policy')
            or parent['review'].get('review_context',{}).get('applicability_contract') != current['review'].get('review_context',{}).get('applicability_contract'))
        if self.feedback_enabled:
            comparison = self.artifact(f'comparison_{cid}_{current["take"]}', lambda: self.advisory(
                f'comparison/{cid}/{current["take"]}', prompts.FEEDBACK_COMPARISON,
                {'parent': {'take': parent['take'], 'review': evidence_policy.memory_view(parent['review']) if evidence_policy.enabled(self.config) else parent['review'], 'media_sha256': parent['receipt']['sha256'], **({'actual_media_facts':evidence_policy.media_facts(parent)} if evidence_policy.enabled(self.config) else {})},
                 'current': {'take': current['take'], 'review': evidence_policy.memory_view(current['review']) if evidence_policy.enabled(self.config) else current['review'], 'media_sha256': current['receipt']['sha256'], **({'actual_media_facts':evidence_policy.media_facts(current)} if evidence_policy.enabled(self.config) else {})}}))
            distinct = parent['receipt']['sha256'] != current['receipt']['sha256']
            improved = (distinct and comparison.get('outcome') == 'improved'
                        and bool(comparison.get('resolved_findings')) and bool(comparison.get('evidence'))
                        and not comparison.get('regressions')
                        and (not content.enabled(self.config) or (candidate_quality(current)['ok'] and not current['review'].get('content_conflicts'))))
            lesson = {'clip_id': cid, 'parent_take': parent['take'], 'take': current['take'],
                'review_system': 'feedback_v1', 'lesson': comparison.get('lesson'),
                'evidence': 'opinion_supported_media_improvement' if improved else 'unconfirmed_hypothesis',
                'opinion_comparison': comparison, 'distinct_media': distinct,
                'components': {'caption': parent['caption'] != current['caption'],
                    'duration': parent['source_clip']['seconds'] != current['source_clip']['seconds'],
                    'references': parent['references'] != current['references']},
                'parent_media_sha256': parent['receipt']['sha256'], 'media_sha256': current['receipt']['sha256'],
                'before_scores': review_scores(parent['review']), 'after_scores': review_scores(current['review']),
                'source_run': str(self.s.run), 'causal_claim': 'none; model opinion comparison, seeds and components may differ'}
            if content.closed_loop_enabled(self.config):
                lesson.update(closed_loop.comparison_credit(parent,current,comparison,candidate_quality(current)))
                lesson['observer_protocol_changed']=protocol_changed
                if protocol_changed:
                    lesson.update(evidence='protocol_changed_comparison_not_confirmed',success_template=False)
            if not any(l['clip_id'] == cid and l['take'] == current['take'] for l in self.s.data['local_lessons']):
                self.s.data['local_lessons'].append(lesson)
                self.memory.record('local', f'{cid}/take_{current["take"]}/comparison', lesson['evidence'], lesson)
                self.s.save()
            return
        a, b = review_scores(parent['review']), review_scores(current['review'])
        from .storage import score
        both_na = set(candidate_quality(parent)['not_applicable_dimensions']).intersection(candidate_quality(current)['not_applicable_dimensions'])
        dims = [d for d in DIMENSIONS if d not in both_na]
        distinct = parent['receipt']['sha256'] != current['receipt']['sha256']
        comparable = all(score(a.get(d)) and score(b.get(d)) for d in dims)
        improved = distinct and comparable and all(b[d] >= a[d] for d in dims) and any(b[d] > a[d] for d in dims)
        duration_changed = parent['source_clip']['seconds'] != current['source_clip']['seconds']
        reference_changed = parent['references'] != current['references']
        lesson = {'clip_id': cid, 'parent_take': parent['take'], 'take': current['take'],
                  'lesson': current['plan'].get('lesson') or current['plan'].get('reason'),
                  'evidence': 'observed_local_nonregression_improvement' if improved else 'unconfirmed_hypothesis',
                  'components': {'caption': parent['caption'] != current['caption'], 'duration': duration_changed, 'references': reference_changed},
                  'causal_claim': 'none; seeds differ and multiple components may change',
                  'parent_media_sha256': parent['receipt']['sha256'], 'media_sha256': current['receipt']['sha256'],
                  'before_scores': a, 'after_scores': b, 'source_run': str(self.s.run)}
        if not any(l['clip_id'] == cid and l['take'] == current['take'] for l in self.s.data['local_lessons']):
            self.memory.record('local', f'{cid}/take_{current["take"]}/comparison', lesson['evidence'], lesson)
            self.s.data['local_lessons'].append(lesson)
            self.s.save()
            path = Path(self.config['lite_root']) / 'memory/lessons.jsonl'
            path.parent.mkdir(exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(lesson, ensure_ascii=False) + '\n')

    def global_checkpoint(self):
        if completion_flow.enabled(self.config):
            return completion_flow.global_checkpoint(self)
        if content.enabled(self.config):
            return content.global_checkpoint(self)
        count = len(self.s.data['selected'])
        if count == 0 or count % 3 or count in self.s.data['checkpoints']:
            return
        suffix = self.s.data['plan']['clips'][count:]
        if not suffix:
            self.s.data['checkpoints'].append(count)
            self.s.save()
            self.s.event('final_checkpoint_archived', selected_clips=count, whole_film_review='not_requested')
            return
        result = self.artifact(f'global_{count}', lambda: self.advisory(f'global/{count}', self.prompt(prompts.GLOBAL + (prompts.GLOBAL_FEEDBACK if self.feedback_enabled else '')),
            {'source': self.source, 'planned_suffix': suffix, 'actual_assets': self.catalog(),
             'unavailable_assets': self.registry.unavailable(),
             'actual_prefix': [{'clip': s['clip'], 'review': s['review'], 'quality': s['quality']} for s in self.s.data['selected']],
             'previous_global_lessons': self.s.data['global_lessons'][-5:],
             'feedback_to_address': [{'clip_id': s['clip']['id'], 'unresolved_findings': s['quality'].get('unresolved_findings', []),
                 'global_feedback': s['review'].get('feedback_decision', {}).get('global_feedback', [])}
                 for s in self.s.data['selected'] if s['quality'].get('unresolved_findings') or s['review'].get('feedback_decision', {}).get('global_feedback')],
             'retrieved_experience': self.memory.retrieve('global', {'source': self.source, 'suffix': suffix,
                 'observed_reviews': [s['review'] for s in self.s.data['selected'][-3:]]}, f'global/{count}'),
             'target_total_seconds': self.config.get('target_seconds', 300),
             'actual_prefix_seconds': sum(s['receipt']['seconds'] for s in self.s.data['selected'])}, max_tokens=24000))
        if result.get('decision') == 'revise':
            try:
                changed = clips(result.get('suffix'), start=count)
                self.s.data['plan']['clips'] = [deepcopy(s['clip']) for s in self.s.data['selected']] + changed
                self.s.data['plan_revision'] += 1
                self.asset_updates(result.get('asset_updates', []))
                write(self.s.run / 'plans' / f'revision_{self.s.data["plan_revision"]}.json', self.s.data['plan'])
                self.s.event('global_revision_applied', selected_prefix_frozen=count, suffix_clips=len(changed),
                             plan_revision=self.s.data['plan_revision'], analysis=result.get('analysis'))
            except (ValueError, TypeError) as exc:
                self.s.warning(f'global/{count}', f'Keeping executable prior suffix; proposal shape unusable: {exc}')
        lessons = result.get('lessons') if isinstance(result.get('lessons'), list) else []
        self.s.data['global_lessons'].append({'selected_count': count, 'lessons': lessons,
            'observed_gaps': result.get('observed_gaps'), 'feedback_dispositions': result.get('feedback_dispositions', []),
            'evidence': 'plan_proposal_not_yet_validated_by_future_media'})
        self.memory.record('global', f'global/{count}/proposal', 'unvalidated_global_proposal',
            {'selected_prefix': count, 'proposal': result, 'plan_revision': self.s.data['plan_revision'],
             'frozen_prefix_media': [s['receipt'].get('sha256') for s in self.s.data['selected']]})
        if count >= 6:
            self.memory.record('global', f'global/{count-3}/followup', 'observed_followup_not_causal_proof',
                {'prior_checkpoint': count-3, 'actual_followup': self.s.data['selected'][-3:],
                 'comparison_limit': 'Future clips have no observed before-version; no improvement attribution'})
        self.s.data['checkpoints'].append(count)
        self.s.save()
        self.ensure_assets()

    def run(self):
        if self.config.get('selective_repair'):
            from .repair_run import run
            return run(self)
        self.repair_retained_decisions()
        self.refresh_quality()
        self.plan()
        if completion_flow.enabled(self.config):
            chosen = {s['clip']['id'] for s in self.s.data['selected']}
            self._completion_clip = next((c for c in self.s.data['plan']['clips'] if c['id'] not in chosen), None)
        self.ensure_assets()
        while True:
            self.s.check_stop()
            if completion_flow.enabled(self.config):
                completion_plan.prepare(self)
            self.global_checkpoint()
            selected = {s['clip']['id'] for s in self.s.data['selected']}
            clip = next((c for c in self.s.data['plan']['clips'] if c['id'] not in selected), None)
            if clip is None:
                break
            if completion_flow.enabled(self.config):
                self._completion_clip = clip
                self.ensure_assets()
                # Lazy canonical voices can change speech timing. Commit any
                # lossless revision before local planning or a GPU claim, then
                # reselect the current clip from the authoritative plan.
                if completion_plan.prepare(self):
                    continue
            self.render_clip(clip)
        self.s.status('running', stage='film.assemble')
        self.s.event('final_checkpoint_archived', selected_clips=len(self.s.data['selected']), whole_film_review='not_requested')
        duration_budget.final(self.config,self.s.data['selected'])
        duration_budget.record(self,[],self.s.data['selected'])
        delivery = self.media.assemble(self.s.data['selected'])
        from . import completion_memory
        if completion_memory.enabled(self.config):
            completion_memory.followup(self,len(self.s.data['selected']))
        ok = all(s['quality']['ok'] for s in self.s.data['selected'])
        record_result={'scope':'global','type':'observed_segment_results_no_whole_film_judge','value':{'selected_media': [{'clip_id': s['clip']['id'], 'media_sha256': s['receipt']['sha256'],
                                'quality': s['quality']} for s in self.s.data['selected']],
             'actual_seconds': delivery.get('seconds'), 'whole_film_semantic_score': None}}
        if completion_flow.enabled(self.config):
            completion_flow.record_observation(self,'film/observed_result',record_result)
        else:
            self.memory.record('global','film/observed_result',record_result['type'],record_result['value'])
        report = {'delivery': delivery, 'quality_status': 'OK' if ok else 'draft/needs_review',
                  'clips': self.s.data['selected'], 'warnings': self.s.data['warnings'],
                  'counts': self.s.data['counts'], 'usage': self.s.data['usage'],
                  'local_content_rsi': self.s.data['local_lessons'], 'global_content_rsi': self.s.data['global_lessons'],
                  'content_acceptance_scope':'selected_shot_observation_contract_not_independent_whole_story_judgment',
                  'narrative_acceptance_passed':None,
                  'memory_scope':self.config.get('memory_policy',{}).get('scope','within_story'),
                  'r3_execution':{
                      'local_interventions':[{k:c['intervention'].get(k) for k in ('intervention_id','kind','status','actual_request_sha256','media_sha256')}
                          for items in self.s.data['candidates'].values() for c in items if c.get('intervention')],
                      'global_patch_receipts':self.s.data.get('replan_receipts',{}),
                      'observed_outcome_is_not_causal_proof':True},
                  'source_sha256': sha(self.s.run / 'source.txt'), 'whole_film_review': 'not_requested',
                  'whole_film_semantic_score': None, 'source_code_rsi': False,
                  'cost_note': 'Original native usage and worker counts retained; unknown currency costs are null, not zero.'}
        report.update(framework_version='r3', asset_registry=str(self.s.run / 'asset_registry.json'),
                      memory_consumption=self.s.data['memory_consumption'], review_system=deepcopy(self.config.get('review_system')),
                      editing_policy=deepcopy(self.config.get('editing_policy')),
                      asset_action_receipts=deepcopy(self.s.data.get('asset_action_receipts',{})),
                      protocol_migrations=deepcopy(self.config.get('protocol_migrations',[])))
        if completion_flow.enabled(self.config):
            report.update(completion_policy=self.config['completion_policy'],
                completion_asset_receipts=deepcopy(self.s.data.get('completion_asset_receipts',{})),
                completion_experience_consumption=deepcopy(self.s.data.get('completion_experience_consumption',{})), technical_completion=True,
                content_acceptance_passed=ok, completion_plan_revisions=deepcopy(self.s.data.get('completion_plan_revisions',[])),
                deferred_optional_assets=deepcopy(self.s.data.get('completion_deferred_assets',{})))
        report['attempt_budget_resets'] = deepcopy(self.s.data.get('attempt_budget_resets', {}))
        report['applicability_repair_receipts'] = deepcopy(self.s.data.get('applicability_repair_receipts', {}))
        write(self.s.run / 'report.json', report)
        self.s.data['delivery'] = delivery
        self.s.status('completed' if ok else 'completed_with_warnings', stage='film.finished', path=delivery['path'])
        return report

    def partial_result(self, reason):
        """A prefix export is explicitly incomplete and never replaces final."""
        value = {'status': 'incomplete', 'reason': str(reason),
                 'selected_clip_count': len(self.s.data['selected']),
                 'planned_clip_count': len(self.s.data.get('plan', {}).get('clips', [])),
                 'counts': deepcopy(self.s.data['counts']), 'delivery': None}
        if self.s.data['selected']:
            try:
                value['delivery'] = self.media.assemble(self.s.data['selected'], partial=True)
            except Exception as exc:
                value['export_error'] = {'type': type(exc).__name__, 'message': str(exc)}
                self.s.event('partial_export_failed', **value['export_error'])
        write(self.s.run / 'partial_report.json', value)
        self.s.event('incomplete_result_preserved', **value)
        return value
