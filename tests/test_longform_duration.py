"""Frame-accurate long-film contracts and bounded planning; no provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from lite import asset_lifecycle, duration_budget, prompts, task_parameters
from lite.storage import Pending


class LongformDurationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.auth_path = Path(folder.name) / 'authorization.json'
        self.auth = {'benchmark_id': 'fixture', 'physical_gpus': [0], 'story_indices': ['film'],
                     'duration_seconds': [570, 630]}
        self.auth_path.write_text(json.dumps(self.auth))
        self.config = {
            'authorization': str(self.auth_path), 'target_seconds': 600, 'max_final_seconds': 630,
            'duration_admission': 'short_story_frames_v1', 'image_provider': {'device': 0},
            'audio_provider': {'device': 0}, 'gpu_resource_admission': 'assigned_gpu_idle_v1',
            'execution_contract': 'asset_worker_lifecycle_v1', 'max_real_takes_per_clip': 6,
            'review_system': {'max_real_takes_per_clip': 6}, 'generation_contract': {'cast': {'host': {}}},
            'task_contract': {'version': task_parameters.LONGFORM_VERSION, 'benchmark_id': 'fixture',
                              'story_index': 'film', 'language': 'Chinese', 'physical_gpu': 0,
                              'min_seconds': 570, 'max_seconds': 630}}

    def test_entry_and_prompts_agree_on_long_duration(self):
        asset_lifecycle.admit_entry(self.config, new_story=True)
        prompt = task_parameters.prompt(prompts.DIRECTOR + prompts.CONTENT_POLICY_V3, self.config)
        self.assertIn('Target 600 seconds', prompt)
        self.assertIn('570–630', prompt)
        for stale in ['约300秒', '180至300秒', '210–240', '155–165']:
            self.assertNotIn(stale, prompt)
        self.assertIn('590-605', task_parameters.duration_correction(self.config))

    def test_plan_checks_both_limits_with_actual_prefix(self):
        selected = [{'receipt': {'fps': 24, 'frames': 60 * 24}}]
        for seconds in (510, 540, 570):
            with self.subTest(seconds=seconds):
                future = [{'seconds': 10}] * (seconds // 10)
                duration_budget.admit(self.config, future, selected)
        with self.assertRaises(Pending):
            duration_budget.admit(self.config, [{'seconds': 10}] * 23)
        for future in ([{'seconds': 10}] * 50 + [{'seconds': 10 - 1/24}],
                       [{'seconds': 10}] * 56 + [{'seconds': 10 + 1/24}]):
            with self.assertRaises(Pending):
                duration_budget.admit(self.config, future, selected)

    def test_final_uses_native_frames_and_validates_permission(self):
        for frames in (13679, 13680, 14400, 15120, 15121):
            selected = [{'receipt': {'fps': 24, 'frames': frames}}]
            with self.subTest(frames=frames):
                if 13680 <= frames <= 15120:
                    duration_budget.final(self.config, selected)
                else:
                    with self.assertRaises(Pending):
                        duration_budget.final(self.config, selected)
        self.auth['duration_seconds'] = [150, 300]
        self.auth_path.write_text(json.dumps(self.auth))
        with self.assertRaises(Pending):
            duration_budget.final(self.config, [{'receipt': {'fps': 24, 'frames': 14400}}])

    def test_long_contract_cannot_disable_admission_or_widen_budget(self):
        for changes in ({'duration_admission': None}, {'target_seconds': 590}, {'max_final_seconds': 640}):
            config = {**self.config, **changes}
            with self.subTest(changes=changes), self.assertRaises(Pending):
                duration_budget.admit(config, [{'seconds': 10}] * 60)
        config = deepcopy(self.config)
        config['task_contract']['min_seconds'] = 100
        with self.assertRaises(Pending):
            asset_lifecycle.admit_entry(config, new_story=True)

    def test_director_cap_is_optional_and_bounded(self):
        task = self.config['task_contract']
        task['director_max_output_tokens'] = 32768
        task_parameters.validate(self.config)
        for invalid in (32769, 23999, 30000.5, True):
            task['director_max_output_tokens'] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(Pending):
                task_parameters.validate(self.config)

    def test_tail_extension_reaches_actual_h3_request_and_preserves_dialogue(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from lite import completion_flow
        from lite.agent import FilmAgent
        from lite.media import Media

        agent = FilmAgent.__new__(FilmAgent)
        agent.config = self.config
        agent.config['completion_policy'] = {'version': 'completion_first_v1'}
        clip = {'id': 'clip_001', 'seconds': 7, 'asset_ids': [], 'action': 'Read the clue.',
                'dialogue': [{'speaker': 'host', 'text': 'Find the key.'}],
                'render_spec': {k: 'Show the note on the table.' for k in ('opening','action','camera','closing')}}
        remaining = [{'id': 'clip_%03d' % i, 'seconds': 10} for i in range(2, 61)]
        store = Mock()
        store.run = self.auth_path.parent
        store.data = {'candidates': {clip['id']: []}, 'assets': {}, 'selected': [],
                      'plan_revision': 1, 'plan': {'clips': [clip] + remaining}}
        agent.s = store
        agent.catalog = lambda: []
        agent.registry = Mock()
        agent.ensure_assets = Mock()
        agent.asset_updates = Mock()
        agent.references = lambda *args: []
        agent.learn_local = Mock()
        agent.feedback_enabled = False
        initial = completion_flow.compile_plan(agent, clip, {'render_spec': clip['render_spec']})
        finding = {'id': 'audiovisual:check:sentence_ending', 'category': 'sentence_ending',
                   'kind': 'observed_content_defect', 'state': 'open', 'severity': 'major',
                   'observation': 'The last word cuts off at the cut.', 'time_s': [6.5, 7]}
        parent = {'take': 1, 'source_clip': deepcopy(clip), 'caption': 'Original',
                  'references': [], 'plan': initial,
                  'receipt': {'sha256': 'a'*64, 'audio_sha256': 'f'*64, 'fps': 24, 'frames': 168},
                  'review': {'completion_review_version': 'completion_observed_v1', 'media_sha256': 'a'*64,
                             'audio_sha256': 'f'*64, 'findings': [finding], 'opinions': {}}}
        store.data['candidates'][clip['id']].append(parent)
        agent.advisory = Mock(return_value={'render_spec': clip['render_spec'], 'tail_extension_finding': finding['id']})
        media = Media.__new__(Media)
        media.store, media.config, media.root = store, self.config, store.run
        media.h3 = Mock()
        media.h3.effective_config.return_value = SimpleNamespace(diffsynth_python='python', diffsynth_project='fixture',
                fl2va_weights='fixture', ref2va_weights='fixture', export_width=1280, export_height=720,
                inference_steps=8, acceleration_profile='fixture', attention_backend='fixture',vram_reserve_gb=1,cuda_device=0)
        class StopAtWorker(Exception):
            pass
        media._worker = Mock(side_effect=StopAtWorker)
        agent.media = media
        with self.assertRaises(StopAtWorker):
            agent.render_clip(clip)
        request = media._worker.call_args.args[1]
        actual = store.data['candidates'][clip['id']][-1]
        self.assertEqual(request['target']['duration_seconds'], 8)
        self.assertEqual(media._worker.call_args.kwargs['amount'], 8)
        self.assertEqual(actual['plan']['dialogue_timeline'], initial['dialogue_timeline'])
        self.assertEqual(actual['source_clip']['dialogue'], clip['dialogue'])
        self.assertEqual(actual['plan']['tail_extension_decision'], 'applied')
        self.assertEqual(actual['intervention']['applied_patch']['seconds'], {'before': 7, 'after': 8})
        self.assertEqual(clip['seconds'], 7)

        # Restart at the same candidate before a worker acknowledgement. The
        # immutable duration is reused, without another planning call.
        agent.advisory.reset_mock()
        with self.assertRaises(StopAtWorker):
            agent.render_clip(clip)
        self.assertEqual(media._worker.call_args.args[1], request)
        agent.advisory.assert_not_called()
        original_proposal = actual['plan']['tail_extension_proposal']
        def recompile(proposal):
            return completion_flow.compile_plan(agent,clip,{'render_spec':clip['render_spec'],
                                             'tail_extension_proposal':proposal})
        for change in ({'finding_id': 'unobserved'}, {'parent_take': 99}):
            with self.subTest(change=change):
                self.assertEqual(recompile({**original_proposal, **change})['seconds'], 7)
        for submitted, native_frames, expected, decision in (
                (8, 192, 9, 'applied'), (7, 216, 10, 'applied'),
                (7, 222, 10, 'applied'), (7, 240, 10, 'at_h3_limit'), (10, 240, 10, 'at_h3_limit')):
            with self.subTest(submitted=submitted, native_frames=native_frames):
                parent['source_clip']['seconds'] = submitted
                parent['receipt']['frames'] = native_frames
                finding['time_s'] = native_frames/24
                compiled = recompile(original_proposal)
                self.assertEqual((compiled['seconds'], compiled['tail_extension_decision']), (expected, decision))
                self.assertEqual(compiled['dialogue_timeline'], initial['dialogue_timeline'])
        parent['source_clip']['seconds'], parent['receipt']['frames'] = 7, 168
        for timestamp in ([1, 2], None, [9, 10]):
            with self.subTest(timestamp=timestamp):
                finding['time_s'] = timestamp
                self.assertEqual(recompile(original_proposal)['tail_extension_decision'], 'original_duration')
        finding['time_s'] = [6.5, 7]
        for field in ('audio_sha256', 'media_sha256'):
            original = parent['review'][field]
            for altered in ('b'*64, None):
                with self.subTest(field=field, altered=altered):
                    parent['review'][field] = altered
                    self.assertEqual(recompile(original_proposal)['tail_extension_decision'], 'original_duration')
            parent['review'][field] = original
        store.data['plan']['clips'] = [clip] + remaining + [
            {'id': 'extra_%d' % i, 'seconds': seconds} for i,seconds in enumerate((10,10,8,5))]
        self.assertEqual(recompile(original_proposal)['tail_extension_decision'], 'insufficient_duration_budget')
        parent['review']['media_sha256'] = 'b'*64
        self.assertEqual(recompile(original_proposal)['tail_extension_decision'], 'original_duration')
