import copy
import unittest

from lite import screen_text, shot_policy


class ScreenTextTests(unittest.TestCase):
    def test_true_display_numbers_spoken_amounts_and_banner(self):
        banner = '周五开放、请走北门、每人限领1.5份'
        dialogue = [{'speaker': 'host', 'text': '第一个月租金四十五万。'}]
        raw = {'clips': [
            {'render_spec': {'opening': 'Interface close-up: Chinese text panel showing 250万元 startup funds.'},
             'action_beats': [{'action': 'Chinese text panel shows 250万元 startup funds'}],
             'exit_state': '250万 figure locked on panel'},
            {'render_spec': {'action': 'He speaks the 45万 first-month figure, the assistant beside him nodding.'},
             'action_beats': [{'action': 'He states the 45万 first-month rent, the assistant nodding beside him'}],
             'dialogue': dialogue, 'asset_ids': ['receipt']},
            {'render_spec': {'opening': 'Banner close-up: Chinese text ' + banner + '.'}},
        ]}
        snapshot = copy.deepcopy(raw)
        plan = screen_text.project_plan(raw)
        fund, rent, sign = plan['clips']
        self.assertEqual(fund['on_screen_text'], ['250万元'])
        self.assertEqual(fund['render_spec']['opening'], 'Interface close-up: Chinese text panel showing 2,500,000 yuan startup funds.')
        self.assertEqual(fund['action_beats'][0]['action'], 'Chinese text panel shows 2,500,000 yuan startup funds')
        self.assertEqual(fund['exit_state'], '2,500,000 figure locked on panel')
        self.assertNotIn('on_screen_text', rent)
        self.assertEqual(rent['render_spec']['action'], 'He speaks the 450,000 first-month figure, the assistant beside him nodding.')
        self.assertEqual(rent['dialogue'], dialogue)
        self.assertEqual(rent['asset_ids'], ['receipt'])
        self.assertEqual(sign['on_screen_text'], [banner])
        self.assertEqual(sign['render_spec']['opening'], 'Banner close-up: space reserved for the specified lettering.')
        self.assertEqual(plan, screen_text.project_plan(plan))
        self.assertEqual(raw, snapshot)

    def test_written_mixed_language_and_existing_artwork(self):
        plan = screen_text.project_plan({'clips': [{
            'on_screen_text': ['示例部门'],
            'state_checkpoints': {'工牌': {'entry': 'The badge reads “样例 R&D”, then swings.',
                                           'exit': 'She lowers it.'}},
        }]})
        clip = plan['clips'][0]
        self.assertEqual(clip['on_screen_text'], ['示例部门', '样例 R&D'])
        self.assertEqual(clip['state_checkpoints']['工牌']['entry'], 'The badge reads the specified lettering, then swings.')
        for action in ['The camera tilts and 他举起手.', 'He says “明天见”.', 'The badge reads 样例A.']:
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, 'clip 1.action_beats'):
                screen_text.project_plan({'clips': [{'action_beats': [{'action': action}]}]})

    def test_decimal_and_wrong_schema_do_not_hide_errors(self):
        for raw, expected in [('He asks about 0.65亿元.', 'He asks about 65,000,000 yuan.'),
                              ('The cost is 15.25元.', 'The cost is 15.25 yuan.')]:
            artwork = []
            self.assertEqual(screen_text.artwork_direction(raw, artwork, 'clip 1'), expected)
            self.assertFalse(artwork)
        for incorrect in ['text', {}, [None], ['']]:
            with self.subTest(incorrect=incorrect), self.assertRaisesRegex(ValueError, 'on_screen_text'):
                screen_text.project_plan({'clips': [{'on_screen_text': incorrect}]})
        config = {'production_contract': {'version': shot_policy.CONTRACT_VERSION}}
        with self.assertRaisesRegex(ValueError, 'bounded action_beats'):
            shot_policy.validate_clip(screen_text.project_plan({'clips': [{}]})['clips'][0], config)
