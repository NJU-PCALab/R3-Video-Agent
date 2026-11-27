from copy import deepcopy
import unittest

from lite import binding_timeline, director_plan


class CompactVisibilityTests(unittest.TestCase):
    def test_both_occurrence_lists_expand_only_the_explicit_enum(self):
        lines = [{'speaker':'A','text':'on','on_screen_or_offscreen':'on'},
                 {'speaker':'B','text':'off','on_screen_or_offscreen':'off'},
                 {'speaker':'C','text':'off'}]
        plan = {'clips':[{'dialogue':lines, 'dialogue_timeline':deepcopy(lines)}]}
        before = deepcopy(plan)
        result = director_plan.project_render_specs(plan)['clips'][0]
        self.assertEqual(plan,before)
        for field in ('dialogue','dialogue_timeline'):
            self.assertEqual([x.get('on_screen_or_offscreen') for x in result[field]], ['on_screen','off_screen',None])
            self.assertEqual([x['text'] for x in result[field]], ['on','off','off'])

    def test_ambiguous_values_are_never_guessed(self):
        for value in ('voice','maybe',1,False,{}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'Unknown speaking visibility'):
                binding_timeline.visibility(value)
