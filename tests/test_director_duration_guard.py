"""Run the regular screenplay -> director -> publication path using recorded-shaped replies."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from lite.agent import FilmAgent
from lite import task_parameters
from lite.storage import Pending, Stopped, Store, Unavailable


def film(name, durations):
    return {'title': name, 'assets': [], 'clips': [
        {'action': 'Walk to the desk.', 'scene': 'S1' if i < 30 else 'S2', 'seconds': seconds}
        for i, seconds in enumerate(durations)]}


class Director:
    def __init__(self, initial, rewrite, correction=None):
        self.payloads = {}
        self.initial, self.rewrite, self.correction = initial, rewrite, correction
        self.review = {'verdict': 'FAIL', 'issues': ['Estimated 542s', 'The desk scene has a prop error'],
                       'revision_notes': ['Add time', 'Check the music cut and desk prop.']}

    def json(self, operation, prompt, payload, *, validate=None, **options):
        self.payloads.setdefault(operation, []).append(deepcopy(payload))
        if operation == 'director/review':
            return deepcopy(self.review)
        if operation.startswith('screenplay/'):
            return {'verdict': 'PASS'}
        answers = {'director/initial': self.initial, 'director/rewrite': self.rewrite,
                   'director/duration_correction': self.correction}
        value = answers[operation]
        if isinstance(value, BaseException):
            raise value
        if value is None:
            raise AssertionError('Unexpected extra call: ' + operation)
        return validate(deepcopy(value))


class DirectorDurationGuardTest(unittest.TestCase):
    def agent(self, director):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        (root / 'source.txt').write_text('Two-scene example')
        auth = root / 'authorization.json'
        auth.write_text(json.dumps({'benchmark_id': 'fixture', 'physical_gpus': [0],
                                    'story_indices': ['fixture'], 'duration_seconds': [570, 630]}))
        code = root / 'code'
        (code / 'resources').mkdir(parents=True)
        (code / 'resources/ref-en.txt').write_text('Test source guide.')
        config = {'lite_root': str(code), 'authorization': str(auth),
                  'duration_admission': 'short_story_frames_v1', 'target_seconds': 600,
                  'max_final_seconds': 630, 'image_provider': {'device': 0}, 'audio_provider': {'device': 0},
                  'gpu_resource_admission': 'assigned_gpu_idle_v1', 'execution_contract': 'asset_worker_lifecycle_v1',
                  'generation_contract': {'cast': {'host': {}}}, 'max_real_takes_per_clip': 6,
                  'review_system': {'max_real_takes_per_clip': 6},
                  'task_contract': {'version': task_parameters.LONGFORM_VERSION,
                                    'story_index': 'fixture', 'benchmark_id': 'fixture', 'language': 'Chinese',
                                    'physical_gpu': 0, 'min_seconds': 570, 'max_seconds': 630}}
        return FilmAgent(Store(root), director, None, config)

    def test_valid_original_and_overlong_rewrite_keep_initial_without_new_generation(self):
        values = [7] * 55 + [8] * 27
        director = Director(film('original', values), film('over', [8] * 82))
        agent = self.agent(director)
        agent.plan()
        self.assertEqual(agent.s.data['plan']['title'], 'original')
        timing = director.payloads['director/review'][0]['host_plan_duration']
        self.assertEqual((timing['shot_count'], timing['total_frames'], timing['total_seconds']), (82, 14424, 601))
        self.assertEqual(timing['by_scene'], [{'scene': 'S1', 'shot_count': 30, 'frames': 5040, 'seconds': 210},
                                              {'scene': 'S2', 'shot_count': 52, 'frames': 9384, 'seconds': 391}])
        self.assertEqual(timing['hard_range_seconds'], [570, 630])
        self.assertEqual(timing['preferred_range_seconds'], [590, 605])
        self.assertTrue(timing['within_hard_bounds'])
        self.assertEqual(director.payloads['director/rewrite'][0]['host_plan_duration'], timing)
        self.assertEqual(director.payloads['director/rewrite'][0]['review'], director.review)
        evidence = json.loads((agent.s.run / 'artifacts/plan_revision_duration_decision.json').read_text())
        self.assertEqual(evidence['revised_duration']['total_seconds'], 656)
        self.assertFalse(evidence['revised_duration']['within_hard_bounds'])
        self.assertEqual(evidence['selected'], 'initial_plan')
        self.assertNotIn('director/duration_correction', director.payloads)
        before = deepcopy(director.payloads)
        agent.plan()
        self.assertEqual(director.payloads, before)

    def test_valid_editorial_rewrite_is_published_even_outside_preferred_band(self):
        director = Director(film('original', [10] * 60), film('keep edits', [10] * 62))
        agent = self.agent(director)
        agent.plan()
        self.assertEqual(agent.s.data['plan']['title'], 'keep edits')
        self.assertNotIn('director/duration_correction', director.payloads)

    def test_invalid_original_still_requires_correction(self):
        for correction, outcome in [(film('corrected', [10] * 60), 'corrected'),
                                    (film('still short', [10] * 55), Pending),
                                    (Unavailable('definitive failure'), Unavailable)]:
            with self.subTest(outcome=outcome):
                director = Director(film('original', [10] * 54), film('rewrite', [10] * 55), correction)
                agent = self.agent(director)
                if outcome == 'corrected':
                    agent.plan()
                    self.assertEqual(agent.s.data['plan']['title'], outcome)
                else:
                    with self.assertRaises(outcome):
                        agent.plan()
                    self.assertFalse(agent.s.data.get('plan'))
                self.assertEqual(len(director.payloads['director/duration_correction']), 1)

    def test_stop_file_prevents_publication_even_with_model_replies(self):
        director = Director(film('original', [10] * 60), film('over', [10] * 70))
        agent = self.agent(director)
        (agent.s.run / 'STOP').touch()
        with self.assertRaises(Stopped):
            agent.plan()
        self.assertFalse(agent.s.data.get('plan'))

    def test_pending_and_stop_propagate_on_optional_rewrite(self):
        for exc in (Pending('unresolved'), Stopped('explicit')):
            with self.subTest(exception=type(exc)):
                director = Director(film('original', [10] * 60), exc)
                agent = self.agent(director)
                with self.assertRaises(type(exc)):
                    agent.plan()
                self.assertFalse(agent.s.data.get('plan'))
                self.assertNotIn('director/duration_correction', director.payloads)
