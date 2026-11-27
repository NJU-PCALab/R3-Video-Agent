import json
from pathlib import Path
import tempfile
import unittest

from lite import binding_contract, director_plan, task_parameters


class LongformDirectorTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        auth = Path(self.directory.name) / 'authorization.json'
        auth.write_text(json.dumps({'benchmark_id': 'cases', 'physical_gpus': [0], 'story_indices': [5],
                                    'duration_seconds': [570, 630]}))
        self.config = {
            'task_contract': {'version': task_parameters.LONGFORM_VERSION, 'benchmark_id': 'cases',
                              'physical_gpu': 0, 'story_index': 5, 'language': 'Chinese',
                              'min_seconds': 570, 'max_seconds': 630},
            'duration_admission': 'short_story_frames_v1', 'target_seconds': 600, 'max_final_seconds': 630,
            'authorization': str(auth), 'image_provider': {'device': 0}, 'audio_provider': {'device': 0},
        }
        self.source = '\n'.join(f'| S{i + 1} | {i:02d}:00–{i + 1:02d}:00 | Scene |' for i in range(10))
        self.screenplay = {'scenes': [{'title': 'S1 Opening', 'shots': [{'dur': 8}, {'dur': 9}]}]}

    def test_source_schedule_controls_expected_scenes_when_writer_omits_them(self):
        handoff = director_plan.longform_handoff(self.screenplay, self.config, self.source)
        self.assertEqual(handoff['required_scene_order'], [f'S{i}' for i in range(1, 11)])
        self.assertEqual(handoff['source_scene_seconds']['S10'], 60)
        self.assertEqual(handoff['screenplay_scene_inventory'][0]['screenplay_seconds'], 17)
        self.assertEqual(handoff['minimum_shot_count'], 57)

    def test_timing_guide_fits_whole_shots_and_scene_totals(self):
        for seconds in range(4, 601):
            slots = director_plan.suggested_shot_durations(seconds)
            self.assertTrue(all(4 <= slot <= 10 for slot in slots))
            self.assertEqual(sum(slots), seconds)
        self.assertEqual(director_plan.suggested_shot_durations(3), [])
        handoff = director_plan.longform_handoff(self.screenplay, self.config, self.source)
        guide = handoff['suggested_shot_seconds_by_scene']
        self.assertEqual({k: sum(v) for k, v in guide.items()}, handoff['source_scene_seconds'])
        self.assertEqual(sum(map(sum, guide.values())), 600)

    def test_duration_error_reports_scene_costs_without_accepting_an_excerpt(self):
        handoff = director_plan.longform_handoff(self.screenplay, self.config, self.source)
        excerpt = [{'scene': f'S{i}', 'seconds': 8} for i in range(1, 11) for _ in range(6)]
        with self.assertRaisesRegex(ValueError, '480 seconds.*S1 48/60.*S10 48/60'):
            director_plan.validate_longform(excerpt, self.config, handoff)
        complete = [{'scene': f'S{i}', 'seconds': 10} for i in range(1, 11) for _ in range(6)]
        director_plan.validate_longform(complete, self.config, handoff)

    def test_excerpt_rejected_and_complete_ordered_plan_passes(self):
        handoff = director_plan.longform_handoff(self.screenplay, self.config, self.source)
        excerpt = [{'scene': 'S1', 'seconds': d} for d in [8] * 5 + [9] * 4]
        with self.assertRaisesRegex(ValueError, '9 shots totaling 76 seconds.*at least 57 shots'):
            director_plan.validate_longform(excerpt, self.config, handoff)
        complete = [{'scene': f'S{i}', 'seconds': 10} for i in range(1, 11) for _ in range(6)]
        director_plan.validate_longform(complete, self.config, handoff)
        complete[54]['scene'] = 'S2'
        with self.assertRaisesRegex(ValueError, 'exact ordered scene IDs'):
            director_plan.validate_longform(complete, self.config, handoff)

    def test_camera_source_shorthand_does_not_admit_routing_labels(self):
        raw = {'clips': [{'render_spec': {'camera': 'Fixed wide frontal frame matching the S1 baseline.',
                                          'action': 'Subject S1 crosses the corridor.'}},
                         {'shot_design': {'camera': 'Fixed wide matched to S1-03.'}},
                         {'render_spec': {'camera': 'Fixed wide same as the S9 baseline.'}}]}
        projected = director_plan.describe_source_camera_references(raw, {'S1'})
        self.assertIn('matching the scene 1 baseline', projected['clips'][0]['render_spec']['camera'])
        self.assertIn('scene 1, shot 3', projected['clips'][1]['shot_design']['camera'])
        self.assertIn('S1', raw['clips'][0]['render_spec']['camera'])
        for direction in (projected['clips'][0]['render_spec']['action'], projected['clips'][2]['render_spec']['camera']):
            with self.assertRaises(ValueError):
                binding_contract.english_direction(direction)

    def test_other_tasks_do_not_inherit_longform_rules(self):
        self.assertEqual(director_plan.longform_handoff(self.screenplay, {}, self.source), {})
        director_plan.validate_longform([{'seconds': 8}], {}, {})

    def test_opening_visual_comparisons_and_only_verified_adjacent_transitions(self):
        rows = [
            {'scene': 'S4', 'render_spec': {'opening': 'Frame size matching the S1 reference.',
                                          'action': 'Subject S1 steps forward.'}, 'exit_state': 'Running into S5.'},
            {'scene': 'S4', 'render_spec': {'opening': 'Identical composition to the S1 wide shot.'},
             'exit_state': 'Running into S5.'},
            {'scene': 'S5', 'action_beats': [{'action': 'She braces.', 'visible_result': 'Steps begin toward S6'}]},
            {'scene': 'S6', 'action_beats': [{'action': 'She turns.', 'visible_result': 'Steps begin toward S1'}]},
        ]
        projected = director_plan.describe_source_camera_references({'clips': rows}, {'S1', 'S4', 'S5', 'S6'})['clips']
        self.assertEqual(projected[0]['render_spec']['opening'], 'Frame size matching the scene 1 reference.')
        self.assertEqual(projected[0]['render_spec']['action'], 'Subject S1 steps forward.')
        self.assertEqual(projected[0]['exit_state'], 'Running into S5.')
        self.assertEqual(projected[1]['render_spec']['opening'], 'Identical composition to the scene 1 wide shot.')
        self.assertEqual(projected[1]['exit_state'], 'Running into scene 5.')
        self.assertEqual(projected[2]['action_beats'][0]['visible_result'], 'Steps begin toward scene 6')
        self.assertEqual(projected[3]['action_beats'][0]['visible_result'], 'Steps begin toward S1')


if __name__ == '__main__':
    unittest.main()
