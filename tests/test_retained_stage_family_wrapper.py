"""Exercise the retained one-object stage family wrapper without network calls."""
from copy import deepcopy
from unittest import TestCase

from lite import character_stages as stages
from lite.storage import Pending
import test_initial_stage_allocation


class RetainedStageFamilyWrapperTests(TestCase):
    def wrapped_plan(self):
        plan, config = test_initial_stage_allocation.fixture()
        family = plan['character_reference_stages']['Alice']
        family['canonical_image_asset_id'] = 'char_r1'
        plan['character_reference_stages']['Alice'] = [family]
        return plan, config

    def test_unwrap_keeps_roles_stages_dialogue_and_budget(self):
        plan, config = self.wrapped_plan()
        original = deepcopy(plan)
        result = stages.prepare_plan(plan, config)

        self.assertEqual(plan, original)
        family = result['character_reference_stages']['Alice']
        self.assertEqual(family['canonical_image_asset_id'], 'char_r1')
        self.assertEqual([(stage['id'], stage['image_asset_id'], stage['candidate_limit'])
                          for stage in family['stages']],
                         [('young', 'char_r1', 1), ('aged', 'char_r1_aged', 1)])
        self.assertEqual([clip['dialogue'] for clip in plan['clips']],
                         [clip['dialogue'] for clip in result['clips']])
        audit = result['character_stage_allocation_receipt']
        self.assertEqual(audit['proposed_table'], original['character_reference_stages'])
        self.assertEqual(audit['retained_family_shape_normalizations'][0]['from'],
                         'singleton_family_wrapper')
        self.assertEqual(stages.prepare_plan(result, config), result)

    def test_mismatched_and_extra_wrapper_fields_and_missing_families_are_rejected(self):
        for change, message in (
            ('mismatch', 'cannot change canonical person identity'),
            ('extra', 'permits only canonical_image_asset_id and stages'),
            ('missing', 'must declare every canonical cast role'),
        ):
            with self.subTest(change=change):
                plan, config = self.wrapped_plan()
                table = plan['character_reference_stages']
                if change == 'missing':
                    table.clear()
                elif change == 'mismatch':
                    table['Alice'][0]['canonical_image_asset_id'] = 'other_character'
                else:
                    table['Alice'][0]['extra'] = True
                with self.assertRaisesRegex(Pending, message):
                    stages.prepare_plan(plan, config)

    def test_children_and_chronology_still_use_original_validation(self):
        for change, message in (
            ('duplicate', 'Invalid/duplicate stage'),
            ('empty', 'finite declared stage list'),
            ('chronology', 'regressed'),
        ):
            with self.subTest(change=change):
                plan, config = self.wrapped_plan()
                rows = plan['character_reference_stages']['Alice'][0]['stages']
                if change == 'duplicate':
                    rows[1]['image_asset_id'] = rows[0]['image_asset_id']
                elif change == 'empty':
                    rows.clear()
                else:
                    earlier = deepcopy(plan['clips'][0])
                    earlier['id'] = 'clip_021'
                    plan['clips'].append(earlier)
                with self.assertRaisesRegex(Pending, message):
                    stages.prepare_plan(plan, config)
