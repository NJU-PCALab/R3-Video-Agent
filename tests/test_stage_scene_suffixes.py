from copy import deepcopy
import unittest

from lite import binding_contract, director_plan


class SourceStageScheduleTest(unittest.TestCase):
    def setUp(self):
        self.config = {'generation_contract': {'cast': {'Sheng': {}}}}
        self.ids = {f'S{i}' for i in range(1, 7)}
        self.stage = {
            'id': 'blindfolded', 'image_asset_id': 'char_r2', 'rank': 0,
            'candidate_limit': 1, 'description': 'Adult with a coarse tunic, S1–S5',
        }

    def normalize(self, table, **kwargs):
        value = {'character_reference_stages': table, **kwargs}
        return director_plan.describe_stage_scene_suffixes(value, self.config, self.ids)['character_reference_stages']

    def test_preserves_identity_and_allocation_in_approved_family_shapes(self):
        for table in (
            {'Sheng': {'canonical_image_asset_id': 'char_r2', 'stages': [self.stage]}},
            {'Sheng': [self.stage]},
            [{'canonical_role': 'Sheng', 'stages': [self.stage]}],
        ):
            before = deepcopy(table)
            updated = self.normalize(table)
            family = updated[0] if isinstance(updated, list) else updated['Sheng']
            first = (family['stages'] if isinstance(family, dict) else family)[0]
            self.assertEqual(first, {**self.stage, 'description': 'Adult with a coarse tunic, from scene 1 through scene 5'})
            self.assertEqual(table, before)

    def test_resolves_onward_with_punctuation_for_authorized_role(self):
        table = {'Sheng': [{**self.stage, 'description': 'No blindfold, S6 onward.'}]}
        self.assertEqual(self.normalize(table)['Sheng'][0]['description'], 'No blindfold, scene 6 onward.')
        self.assertEqual(self.normalize(table, character_stage_allocation_receipt={}), table)

    def test_ambiguous_or_untrusted_labels_still_fail_english_guard(self):
        for text in ('Looks toward S1 then speaks.', 'Coarse tunic, S6–S1', 'Coarse tunic, S9 onward',
                     'Coarse tunic <Subject 1>, S1–S5', '旧衣, S1–S5'):
            modified = self.normalize({'Sheng': [{**self.stage, 'description': text}]})['Sheng'][0]
            with self.assertRaises(ValueError):
                binding_contract.english_direction(modified['description'])
        self.assertEqual(self.normalize({'Other': [self.stage]}), {'Other': [self.stage]})

    def test_director_feedback_names_the_offending_field(self):
        from lite import character_stages
        from lite.storage import Pending

        cfg = {'generation_contract': {'cast': {'Sheng': {'image_asset_id': 'char_r2'}}}}
        for update, field in (({'id': 'blindfolded'}, 'id must be'), ({'image_asset_id': 'char_r2'}, 'already assigned'),
                              ({'description': 'Turns to <Subject 2>.'}, 'description must'), ({'rank': 3}, 'rank must'),
                              ({'candidate_limit': 3}, 'candidate_limit must')):
            later = {**self.stage, 'id': 'recovered', 'image_asset_id': 'char_r2_later', 'rank': 1,
                     'description': 'Adult in a clean robe.', **update}
            first = {**self.stage, 'description': 'Adult with a blindfold.'}
            with self.subTest(update=update), self.assertRaisesRegex(Pending, 'Sheng.*index 1.*' + field):
                character_stages._table({'character_reference_stages': {'Sheng': {'stages': [first, later]}}}, cfg)


if __name__ == '__main__':
    unittest.main()
