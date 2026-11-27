from copy import deepcopy
import unittest

from lite import binding_contract, director_plan, prompts, shot_policy


CONFIG = {
    'generation_contract': {'version': binding_contract.VERSION,
                            'cast': {'Student': {'image_asset_id': 'student', 'voice_asset_id': None}}},
    'production_contract': {'version': shot_policy.CONTRACT_VERSION},
}


def source_shot(opening='The door stands closed.'):
    return {'seconds': 7, 'action': 'A student opens the door.', 'dialogue': [], 'asset_ids': [],
            'shot_design': {'opening': opening, 'camera': 'Keep a shoulder-level view.',
                            'closing': 'The student stands beyond the open door.'}}


class SourceDirectorPlanTests(unittest.TestCase):
    def test_legacy_projection_uses_each_shots_own_actual_direction(self):
        raw = {'assets': [], 'render_spec': {'opening': 'Unrelated film-wide note.'},
               'clips': [source_shot(), source_shot('The corridor is empty.')]}
        saved = deepcopy(raw)
        plan = director_plan.validate(raw, CONFIG)
        self.assertEqual(raw, saved)
        self.assertEqual(plan['host_render_spec_projection']['clip_positions'], [1, 2])
        self.assertEqual(plan['clips'][1]['render_spec']['opening'], 'The corridor is empty.')
        self.assertEqual(plan['clips'][0]['action_beats'][0]['visible_result'],
                         raw['clips'][0]['shot_design']['closing'])

    def test_canonical_layout_does_not_require_repeating_action(self):
        raw = source_shot()
        action = raw.pop('action')
        raw['render_spec'] = {**raw.pop('shot_design'), 'action': action}
        plan = director_plan.validate({'assets': [], 'clips': [raw]}, CONFIG)
        self.assertEqual(plan['clips'][0]['action'], action)
        self.assertNotIn('host_render_spec_projection', plan)

    def test_film_level_spec_cannot_patch_incomplete_shot(self):
        raw = source_shot()
        raw['shot_design'].pop('camera')
        with self.assertRaisesRegex(ValueError, 'clip 1: missing per-shot render_spec'):
            director_plan.validate({'assets': [], 'clips': [raw], 'render_spec': source_shot()}, CONFIG)

    def test_explicit_invalid_direction_is_not_replaced_by_legacy_design(self):
        raw = source_shot()
        raw['render_spec'] = {**raw['shot_design'], 'action': '<Subject 1> walks out.'}
        with self.assertRaisesRegex(ValueError, r'clip 1 render_spec\.action'):
            director_plan.validate({'assets': [], 'clips': [raw]}, CONFIG)

    def test_canonical_prompt_explicitly_places_shot_layout(self):
        self.assertIn('In EVERY clips[i]', prompts.BOUND_DIRECTOR)
        self.assertIn('must not replace it', prompts.BOUND_DIRECTOR)
