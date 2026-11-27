from copy import deepcopy
import unittest

from lite import character_stages
from lite.storage import Pending


class StageImagePromptContractTests(unittest.TestCase):
    def test_director_sees_the_same_image_language_rule_the_host_enforces(self):
        cfg = {'character_reference_policy':{'version':character_stages.VERSION}}
        prompt = character_stages.instructions(cfg)
        self.assertIn('assets[].prompt in English', prompt)
        self.assertIn('stage.prompt is also English', prompt)

    def test_canonical_stage_error_points_at_original_image_and_keeps_it(self):
        cfg = {'character_reference_policy':{'version':character_stages.VERSION},
               'generation_contract':{'cast':{'A':{'image_asset_id':'portrait','voice_asset_id':'voice_a'}}}}
        plan = {'character_reference_stages':{'A':{'stages':[{'id':'adult','image_asset_id':'portrait',
                                                             'description':'An adult in a blue coat.'}]}},
                'assets':[{'id':'portrait','kind':'image','prompt':'蓝布便服'}], 'clips':[]}
        before = deepcopy(plan)
        with self.assertRaisesRegex(Pending, r"assets\['portrait'\]\.prompt"):
            character_stages.prepare_plan(plan,cfg)
        self.assertEqual(plan,before)
