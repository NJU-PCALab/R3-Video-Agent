import unittest

from lite import binding_contract, character_stages, prompts


def stage_instructions(modes):
    return character_stages.instructions({
        'character_reference_policy': {
            'version': character_stages.VERSION,
            'role_transition_modes': modes,
        },
        'generation_contract': {'cast': {name: {} for name in modes}},
    })


class DirectorStageInstructionsTests(unittest.TestCase):
    def test_chronological_roles_are_never_listed_as_reversible(self):
        prompt = stage_instructions({'曾亮': 'chronological_age'})
        self.assertIn('All roles progress chronologically on the main timeline.', prompt)
        self.assertNotIn('may return to an earlier declared appearance', prompt)
        self.assertIn('timeline_transition:"flashback"', prompt)

    def test_mixed_policy_names_only_roles_with_reversible_appearance(self):
        prompt = stage_instructions({'曾亮': 'chronological_age', '马骥': 'reversible_appearance'})
        self.assertIn('an explicit character_states selection: ["马骥"]', prompt)
        self.assertIn('All other roles progress chronologically.', prompt)
        self.assertNotIn('曾亮', prompt)

    def test_source_shot_example_passes_existing_h3_binding_validation(self):
        prompt = ' '.join(prompts.BOUND_DIRECTOR.split())
        example = 'match the fixed wide view from scene one, shot three'
        self.assertIn(example, prompt)
        self.assertEqual(binding_contract.english_direction(example), example)
        for direction in ('match the view from S1-03', '<Subject 1> enters'):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                binding_contract.english_direction(direction)
