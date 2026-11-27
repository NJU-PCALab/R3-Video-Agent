from copy import deepcopy
import unittest

from lite import screen_text


class RegisteredBadgeTests(unittest.TestCase):
    def test_full_literal_is_preserved_and_not_split_at_space(self):
        clip = {'render_spec': {'opening':'Engraved Chinese lettering 保镖 沈揽月 is visible on its face.'},
                'on_screen_text':['保镖', '保镖 沈揽月'], 'dialogue':[{'speaker':'沈揽月','text':'我是保镖。'}]}
        raw = deepcopy(clip)
        out = screen_text.project_plan({'clips':[clip]})['clips'][0]
        self.assertEqual(clip, raw)
        self.assertEqual(out['dialogue'], raw['dialogue'])
        self.assertEqual(out['on_screen_text'], raw['on_screen_text'])
        self.assertNotIn('沈揽月',out['render_spec']['opening'])
        self.assertIn('is visible on its face',out['render_spec']['opening'])

    def test_undeclared_or_non_display_cjk_remains_an_error(self):
        for text, artwork in [('Chinese lettering 保镖 沈揽月 is visible.', ['保镖']),
                              ('She says 保镖 沈揽月 and leaves.', ['保镖 沈揽月']),
                              ('Chinese lettering 保镖 沈揽月 rises. 她站起', ['保镖 沈揽月'])]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                screen_text.artwork_direction(text,artwork,'opening')
