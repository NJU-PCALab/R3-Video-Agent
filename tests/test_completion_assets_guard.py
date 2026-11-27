"""Exercise the optional asset path without model, media or GPU calls."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from lite import asset_lifecycle, character_stages, completion_flow, completion_memory
from lite.agent import FilmAgent
from lite.storage import Pending, Store, key, sha


class CharacterAssetGuardTests(unittest.TestCase):
    operation = 'completion/global/1/assets'

    def agent(self, directory):
        root = Path(directory)
        (root / 'resources').mkdir()
        (root / 'resources/ref-en.txt').write_text('Synthetic test guide')
        store = Store(root / 'run')
        (store.run / 'source.txt').write_text('The same person appears twice.')
        config = {
            'lite_root': str(root), 'execution_contract': asset_lifecycle.VERSION,
            'completion_policy': {'version': completion_flow.VERSION, 'experience_loop': completion_memory.VERSION},
            'generation_contract': {'cast': {'Traveler': {'image_asset_id': 'portrait', 'voice_asset_id': 'voice'}}},
            'character_reference_policy': {'version': character_stages.VERSION, 'require_stage_declarations': True},
        }
        draft = {
            'assets': [{'id': 'portrait', 'kind': 'image', 'role': 'character', 'prompt': 'A young traveler.'}],
            'character_reference_stages': {'Traveler': {'stages': [
                {'id': 'young', 'image_asset_id': 'portrait', 'description': 'A young traveler.'},
                {'id': 'older', 'image_asset_id': 'portrait_older', 'description': 'The same traveler, decades later.'},
            ]}},
            'clips': [{'id': 'clip_001', 'asset_ids': ['portrait'], 'dialogue': [],
                       'character_states': {'Traveler': {'stage': 'young'}}},
                      {'id': 'clip_002', 'asset_ids': ['portrait_older'], 'dialogue': [],
                       'character_states': {'Traveler': {'stage': 'older'}}}],
        }
        agent = FilmAgent(store, Mock(), Mock(), config)
        store.data['plan'] = character_stages.prepare_plan(draft, config)
        character_stages.freeze_plan(store, store.data['plan'], config)
        for spec in store.data['plan']['assets']:
            row = agent.registry.declare(spec['id'], spec)
            row.update(generation=1, pending_prompt=None, pending_version=None,
                       lifecycle_attempts={'1/1': {'status': 'accepted'}})
        store.save()
        return agent

    def intent(self, asset_id='portrait', action='update', role='character', **extra):
        return {'id': asset_id, 'action': action, 'kind': 'image', 'role': role,
                'prompt': 'Try different framing.', 'reason': 'Optional continuity',
                'target_clip_ids': ['clip_002'], **extra}

    def test_protected_optional_changes_are_deferred_before_reservation(self):
        cases = [
            self.intent(),
            self.intent('portrait_older'),
            self.intent('new_portrait', action='new'),
            self.intent('new_scene', action='new', role='scene', reference_asset_id='portrait_older'),
            self.intent('portrait_older', action='keep', omit_reference=True),
        ]
        for intent in cases:
            with self.subTest(intent=intent), tempfile.TemporaryDirectory() as directory:
                agent = self.agent(directory)
                assets = deepcopy(agent.s.data['assets'])
                plan = deepcopy(agent.s.data['plan'])
                with patch('lite.binding_contract.admit_gpu') as admit:
                    receipt, = agent.asset_updates([intent], self.operation)
                admit.assert_not_called()
                agent.models.assert_not_called()
                agent.media.assert_not_called()
                self.assertEqual(receipt['status'], 'deferred')
                self.assertFalse(receipt.get('reserved'))
                self.assertFalse(receipt['executed'])
                self.assertEqual(agent.s.data['assets'], assets)
                self.assertEqual(agent.s.data['plan'], plan)

    def test_protected_id_cannot_be_redeclared_as_audio_before_first_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            del agent.s.data['assets']['portrait_older']
            intent = self.intent('portrait_older', action='new', kind='audio',
                                 role='voice', speaker='Traveler')
            with patch('lite.binding_contract.admit_gpu') as admit:
                receipt, = agent.asset_updates([intent], self.operation)
            admit.assert_not_called()
            self.assertEqual(receipt['status'], 'deferred')
            self.assertIn('canonical or stage character image', receipt['reason'])
            self.assertNotIn('portrait_older', agent.s.data['assets'])

    def test_existing_reservation_stays_on_its_original_reconciliation_path(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            intent = self.intent()
            identifier = key({'operation': self.operation, 'index': 0, 'intent': intent})
            receipt = {'id': identifier, 'operation': self.operation, 'status': 'executing',
                       'reserved': True, 'proposal': intent, 'asset_action_receipt_ids': ['original']}
            agent.s.data['completion_asset_receipts'] = {identifier: receipt}
            agent.ensure_assets = Mock(side_effect=Pending('Original worker result is unknown'))

            with patch('lite.binding_contract.admit_gpu') as admit:
                with self.assertRaisesRegex(Pending, 'Original worker result is unknown'):
                    agent.asset_updates([intent], self.operation)

            admit.assert_not_called()
            agent.ensure_assets.assert_called_once_with(only_ids=['portrait'], optional=True)
            self.assertEqual(agent.s.data['completion_asset_receipts'][identifier], receipt)

    def test_inspect_only_reuses_observations_and_does_not_introduce_character(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            image = Path(directory) / 'retained_image.png'
            image.write_bytes(b'synthetic image receipt')
            agent.catalog = Mock(return_value=[{'id': 'portrait_older', 'kind': 'image',
                                                'path': str(image), 'sha256': sha(image),
                                                'review_status': 'PASS'}])
            row = agent.s.data['assets']['portrait_older']
            row['versions'] = [{'version': 1, 'retained_observation': 'Older traveler.'}]
            clip = {'id': 'clip_003', 'asset_ids': [], 'dialogue': [],
                    'character_states': {}, 'character_stage_bindings': {}}
            agent.s.data['plan']['clips'].append(clip)
            intent = self.intent('portrait_older', action='inspect', target_clip_ids=['clip_003'])

            receipt, = agent.asset_updates([intent], self.operation)

            self.assertEqual(receipt['observation'], row['versions'])
            self.assertEqual(receipt['reference_bindings'], [])
            self.assertEqual(completion_flow.reference_ids(agent, clip), [])

    def test_non_character_scene_proposal_still_reaches_resource_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            scene = self.intent('new_scene', action='new', role='scene')
            with patch('lite.binding_contract.admit_gpu',
                       side_effect=Pending('Device busy', status='awaiting_resource')) as admit:
                receipt, = agent.asset_updates([scene], self.operation)
            admit.assert_called_once()
            self.assertEqual(receipt['status'], 'deferred_resource')
            self.assertFalse(receipt.get('reserved'))
