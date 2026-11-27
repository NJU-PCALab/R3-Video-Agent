from copy import deepcopy
import unittest

from lite import asset_lifecycle, voice_lineage


class CanonicalVoiceRolesTest(unittest.TestCase):
    def setUp(self):
        self.config = {'generation_contract': {'cast': {
            '主持人': {'voice_asset_id': 'voice_r1'}, '电话对方': {'voice_asset_id': 'voice_phone'}}},
            'execution_contract': 'asset_worker_lifecycle_v1'}
        self.asset = {'id': 'voice_phone', 'kind': 'audio', 'role': '电话画外音',
                      'speaker': '电话对方', 'language': 'zh', 'generator': 'voice_design', 'text': '待遇是真的吗？'}

    def test_canonical_ownership_enables_existing_generated_voice_controls(self):
        old = dict(self.asset, id='voice_r1', role='voice', speaker='主持人')
        prop = {'id': 'payroll_screen', 'kind': 'image', 'role': '系统面板'}
        plan = {'assets': [self.asset, old, prop]}
        frozen = deepcopy(plan)
        normalized = voice_lineage.project_canonical_roles(plan, self.config)
        phone, lin, screen = normalized['assets']
        self.assertEqual(phone['role'], 'voice')
        self.assertEqual(phone['source_role'], '电话画外音')
        self.assertEqual((lin, screen), (old, prop))
        self.assertEqual(voice_lineage.project_canonical_roles(normalized, self.config), normalized)
        calibrated = asset_lifecycle.calibration_spec(phone, self.config)
        self.assertEqual(calibrated['calibration_policy'], asset_lifecycle.CALIBRATION)
        self.assertEqual(calibrated['scene_text'], self.asset['text'])
        self.assertNotEqual(calibrated['text'], self.asset['text'])
        self.assertEqual(plan, frozen)

    def test_conflicts_do_not_gain_voice_authority(self):
        for change in ({'speaker': '主持人'}, {'speaker': None}, {'role': 'music'}, {'kind': 'image'},
                       {'generator': 'import'}, {'track_type': 'mixed_native'},
                       {'source_role': 'conflicting prior label'}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'voice_phone'):
                voice_lineage.project_canonical_roles({'assets': [{**self.asset, **change}]}, self.config)
        foreign = {'id': 'voice_r1', 'kind': 'audio', 'role': 'voice', 'speaker': '主持人'}
        clone = dict(self.asset, generator='voice_clone', reference_asset_id=foreign['id'])
        for usage in ('电话画外音', 'voice'):
            with self.subTest(usage=usage), self.assertRaisesRegex(ValueError, 'same-speaker'):
                voice_lineage.project_canonical_roles({'assets': [foreign, dict(clone, role=usage)]}, self.config)
        parent = dict(self.asset, id='phone_reference', role='voice')
        clone['reference_asset_id'] = parent['id']
        self.assertEqual(voice_lineage.project_canonical_roles({'assets': [parent, clone]}, self.config)['assets'][1]['role'], 'voice')
