"""Canonical IDs bind localized roles; explicitly unseen speakers need only their voice."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase

from lite import character_stages as stages, completion_flow
from lite.storage import Pending


def setup():
    cast = {'host': {'image_asset_id': 'presenter', 'voice_asset_id': 'narrator',
                     'description': 'An adult presenter with short hair.'},
            'caller': {'image_asset_id': 'unseen', 'voice_asset_id': 'phone',
                       'description': 'An adult telephone caller.'}}
    config = {'character_reference_policy': {'version': stages.VERSION, 'require_stage_declarations': True,
                'allocation_authority': stages.ALLOCATION_AUTHORITY, 'stable_single_stage_roles': list(cast)},
              'generation_contract': {'cast': cast}}
    plan = {'assets': [{'id': 'presenter', 'role': '主持人', 'kind': 'image', 'prompt': 'Presenter portrait.'},
                       {'id': 'phone', 'kind': 'audio', 'speaker': 'caller'}],
            'clips': [{'id': 'one', 'asset_ids': ['presenter'], 'dialogue': [{'speaker': 'caller', 'text': 'Ready?'}],
                       'dialogue_timeline': [{'speaker': 'caller', 'on_screen_or_offscreen': 'offscreen'}]}]}
    return plan, config


class StableCastTests(TestCase):
    def test_voice_only_emits_no_person_or_charge_and_preserves_source_image_role(self):
        plan, config = setup()
        original = deepcopy(plan)
        result = stages.prepare_plan(plan, config)
        self.assertEqual(plan, original)
        self.assertEqual(stages.prepare_plan(result, config), result)
        clip = result['clips'][0]
        self.assertEqual(clip['offscreen_voice_only_roles'], ['caller'])
        self.assertEqual(set(clip['character_stage_bindings']), {'host'})
        self.assertEqual({a['id'] for a in result['assets']}, {'presenter', 'phone'})
        self.assertEqual(result['assets'][0]['source_role'], '主持人')
        agent = SimpleNamespace(config=config, catalog=lambda: [], s=SimpleNamespace(data={'plan':result}))
        self.assertEqual(completion_flow.reference_ids(agent, clip, include_missing=True), ['presenter', 'phone'])
        clip['dialogue_timeline'][0]['on_screen_or_offscreen'] = 'onscreen'
        with self.assertRaises(Pending):
            stages.assert_clip_bindings(clip, config)

    def test_ambiguous_or_visible_caller_still_requires_a_real_image(self):
        for mutation in (
            lambda p: p['clips'][0]['dialogue_timeline'].clear(),
            lambda p: p['clips'][0]['asset_ids'].append('unseen'),
            lambda p: p['clips'][0]['dialogue_timeline'][0].update(on_screen_or_offscreen='onscreen'),
            lambda p: p['clips'][0]['dialogue'][0].update(on_screen_or_offscreen='on_screen'),
            lambda p: p['assets'][1].update(speaker='host'),
            lambda p: p['assets'][0].update(character_role='caller'),
            lambda p: p['assets'][0].update(role='prop'),
        ):
            plan, config = setup()
            mutation(plan)
            with self.subTest(mutation=mutation), self.assertRaises(Pending):
                stages.prepare_plan(plan, config)
