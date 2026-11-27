"""Offline regression coverage for bounded narrative checkpoint decisions and replay."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from lite import completion_flow as flow
from lite import completion_assets
from lite import completion_memory
from lite import completion_review as review
from lite.agent import FilmAgent
from lite.storage import Pending, Store, read, sha, write


def sample_review(categories):
    return {'status': 'observed', 'observed_events': ['A hand reaches for a cup.'],
            'heard_text': '', 'checks': {category: {'status': 'clear', 'observation': 'No discrepancy visible.'}
                                          for category in categories}, 'findings': []}


def make_clip(identifier):
    return {'id': identifier, 'event_id': 'event_' + identifier, 'seconds': 4, 'action': 'Carry a cup.',
            'dialogue': [], 'asset_ids': [], 'entry_state': 'Cup in hand.', 'exit_state': 'Cup on table.'}


class BoundaryReplanTests(unittest.TestCase):
    def setup_boundary(self, tmp, status='unknown', entry='Cup on table.', count=1):
        root = Path(tmp)
        (root / 'resources').mkdir()
        (root / 'resources/ref-en.txt').write_text('Synthetic test guide')
        store = Store(root / 'run')
        (store.run / 'source.txt').write_text('Synthetic cup continuity example')
        config = {'lite_root': str(root), 'completion_policy': {'version': flow.VERSION,
                   'experience_loop': completion_memory.VERSION}}
        agent = FilmAgent(store, Mock(), Mock(), config)
        selected = [{'clip': make_clip(f'clip_{i:03}'), 'take': 1,
                     'receipt': {'sha256': str(i) * 64, 'audio_sha256': 'a' * 64,
                                 'frames': 96, 'fps': 24},
                     'review': {'observed_events': ['A hand reaches for a cup.']},
                     'quality': {'quality_status': 'draft/needs_review', 'unresolved_findings': []}}
                    for i in range(1, count + 1)]
        selected[-1]['review'].update(
            narrative_observation={'status': status, 'remaining_target': 'Cup on table.',
                                   'remaining_target_is_expectation_only': True},
            feedback_decision={'narrative_observation': {'status': status, 'remaining_target': 'Cup on table.'}},
        )
        current = make_clip('clip_007')
        current['entry_state'] = entry
        agent.s.data.update(
            selected=selected, checkpoints=[], candidates={},
            plan={'clips': [row['clip'] for row in selected] + [current]},
        )
        agent.advisory = Mock(return_value={
            'suggestions': [
                {'clip_id': selected[-1]['clip']['id'], 'entry_state': 'Wrong target.'},
                {'clip_id': current['id'], 'entry_state': 'Show whether the cup is on the table.'},
            ],
        })
        return agent, current

    def test_explicit_unknown_handoff_replans_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, current = self.setup_boundary(tmp)
            selected_before = deepcopy(agent.s.data['selected'])
            flow.global_checkpoint(agent)
            flow.global_checkpoint(agent)

            agent.advisory.assert_called_once()
            operation, _, payload = agent.advisory.call_args.args
            self.assertEqual(operation, 'completion/global/1')
            self.assertEqual(payload['checkpoint_trigger'], {
                'reason': 'carry_state_observation_unknown',
                'previous_clip_id': 'clip_001',
                'next_clip_id': current['id'],
                'carry_state_target': 'Cup on table.',
            })
            self.assertEqual(agent.s.data['selected'], selected_before)
            self.assertEqual(agent.s.data['checkpoints'], [1])
            receipt = agent.s.data['replan_receipts']['1']
            self.assertEqual(receipt['checkpoint_trigger'], payload['checkpoint_trigger'])
            self.assertEqual([row['status'] for row in receipt['receipts']], ['rejected', 'applied'])

    def test_later_reference_binding_preserves_checkpoint_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, current = self.setup_boundary(tmp, count=3)
            assets = []
            for name in ('table', 'cup'):
                path = Path(tmp) / name
                path.write_bytes(name.encode())
                assets.append({'id': name, 'kind': 'image', 'review_status': 'PASS',
                               'path': str(path), 'sha256': sha(path)})
            agent.catalog = Mock(return_value=assets)
            agent.s.data['assets'] = {asset['id']: {'spec': asset} for asset in assets}
            agent.advisory.return_value = {'suggestions': [
                {'clip_id': current['id'], 'reference_ids': ['table']}
            ]}
            flow.global_checkpoint(agent)
            checkpoint = agent.s.data['replan_receipts']['3']
            self.assertEqual(checkpoint['receipts'][0]['changes']['asset_ids'], ['table'])
            saved = deepcopy(checkpoint)

            shot = agent.s.data['plan']['clips'][-1]
            bindings = completion_assets._bind(agent, {}, [shot], assets[1])
            self.assertEqual([row['asset_id'] for row in bindings], ['cup'])
            self.assertEqual(shot['asset_ids'], ['table', 'cup'])
            self.assertEqual(checkpoint, saved)
            agent.s.save()
            self.assertEqual(Store(agent.s.run, existing=True).data['replan_receipts']['3'], saved)

    def test_off_cycle_replan_requires_link_and_before_submission(self):
        for status, entry, claimed in (
            ('observed', 'Cup on table.', False),
            ('unknown', 'Cup had been placed on table.', False),
            ('unknown', 'Cup on table.', True),
        ):
            with self.subTest(status=status, entry=entry, claimed=claimed):
                with tempfile.TemporaryDirectory() as tmp:
                    agent, current = self.setup_boundary(tmp, status, entry)
                    if claimed:
                        write(agent.s.run / f"media/videos/{current['id']}_take_1/submitted.json",
                              {'original': True})
                    flow.global_checkpoint(agent)
                    agent.advisory.assert_not_called()
                    self.assertEqual(agent.s.data['checkpoints'], [])

    def test_review_normalizer_cannot_turn_missing_or_malformed_target_into_trigger(self):
        for fields, should_replan in (
            ({}, False),
            ({'narrative_observation': ['unknown', 'Cup on table.']}, False),
            ({'narrative_observation': {'status': 'unknown', 'remaining_target': ['Cup on table.']}}, False),
            ({'narrative_observation': {'status': 'observed', 'remaining_target': 'Cup on table.'}}, False),
            ({'narrative_observation': {'status': 'unknown', 'remaining_target': 'Window is open.'}}, False),
            ({'narrative_observation': {'status': 'unknown', 'remaining_target': 'Cup on table.'}}, True),
        ):
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as tmp:
                agent, _ = self.setup_boundary(tmp)
                selected = agent.s.data['selected']
                prior = selected[-1]
                video, audio = Path(tmp) / 'video', Path(tmp) / 'audio'
                video.write_bytes(b'video')
                audio.write_bytes(b'audio')
                prior['receipt'].update(path=str(video), audio_path=str(audio), seconds=4,
                                        sha256=sha(video), audio_sha256=sha(audio))
                candidate = {'take': 1, 'source_clip': prior['clip'], 'receipt': prior['receipt']}
                replan = agent.advisory
                agent.advisory = Mock(side_effect=[
                    sample_review(review.VISUAL), sample_review(review.AV),
                    {'reason': 'Retained review', **fields},
                ])
                agent.s.data['selected'] = []
                with patch('lite.review.visual_frames', return_value={'frames': []}):
                    prior['review'] = review.review(agent, prior['clip']['id'], candidate)
                agent.s.data['selected'] = selected
                agent.advisory = replan
                self.assertEqual(prior['review']['narrative_observation']['status'], 'unknown')

                flow.global_checkpoint(agent)

                self.assertEqual(replan.call_count, int(should_replan))
                self.assertEqual(agent.s.data['checkpoints'], [1] if should_replan else [])

    def test_three_shot_checkpoint_retains_its_existing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, _ = self.setup_boundary(tmp, count=3)
            flow.global_checkpoint(agent)
            agent.advisory.assert_called_once()
            operation, _, payload = agent.advisory.call_args.args
            self.assertEqual(operation, 'completion/global/3')
            self.assertNotIn('checkpoint_trigger', payload)
            self.assertEqual(agent.s.data['checkpoints'], [3])


class SimulatedInterruption(Exception):
    pass


class ReplanReplayTests(unittest.TestCase):
    def interrupt_after_apply(self, directory, count=1):
        agent, current = BoundaryReplanTests().setup_boundary(directory, count=count)
        agent.s.data['global_lessons'] = []
        updates = [{'id': 'scene', 'action': 'inspect', 'target_clip_ids': [current['id']]}]
        agent.advisory.return_value['asset_updates'] = updates
        agent.asset_updates = Mock(side_effect=SimulatedInterruption)
        with self.assertRaises(SimulatedInterruption):
            flow.global_checkpoint(agent)
        agent.advisory.assert_called_once()
        agent.s.data = Store(agent.s.run, existing=True).data
        self.assertEqual(agent.s.data['checkpoints'], [])
        self.assertEqual(agent.s.data['global_lessons'], [])
        saved = agent.s.data['replan_receipts'][str(count)]
        self.assertTrue(saved['applied'])
        self.assertEqual(agent.s.data['plan']['clips'][-1]['entry_state'],
                         'Show whether the cup is on the table.')
        agent.advisory.reset_mock()
        agent.asset_updates = Mock(return_value=[])
        return agent, saved, updates

    def test_resumes_early_and_regular_checkpoints_without_second_request(self):
        for count in (1, 3):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                agent, saved, updates = self.interrupt_after_apply(directory, count)
                # A retained early receipt may predate the stricter trigger.
                agent.s.data['selected'][-1]['review'].pop('feedback_decision')
                prefix = deepcopy(agent.s.data['selected'])
                patched_plan = deepcopy(agent.s.data['plan'])

                flow.global_checkpoint(agent)
                flow.global_checkpoint(agent)

                agent.advisory.assert_not_called()
                agent.asset_updates.assert_called_once_with(updates, f'completion/global/{count}/assets')
                self.assertEqual(agent.s.data['checkpoints'], [count])
                self.assertEqual([lesson['selected_count'] for lesson in agent.s.data['global_lessons']], [count])
                self.assertEqual(agent.s.data['replan_receipts'][str(count)], saved)
                self.assertEqual(agent.s.data['selected'], prefix)
                self.assertEqual(agent.s.data['plan'], patched_plan)

    def test_missing_or_wrong_proposal_and_changed_prefix_never_resubmit(self):
        for changed in ('missing_proposal', 'wrong_proposal', 'wrong_prefix'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                agent, _, _ = self.interrupt_after_apply(directory)
                path = agent.s.run / 'artifacts/completion/global/1.json'
                if changed == 'missing_proposal':
                    path.unlink()
                elif changed == 'wrong_proposal':
                    proposal = read(path)
                    proposal['asset_updates'] = []
                    write(path, proposal)
                else:
                    agent.s.data['selected'][0]['receipt']['sha256'] = 'changed'
                with self.assertRaises(Pending) as raised:
                    flow.global_checkpoint(agent)
                self.assertEqual(raised.exception.status, 'needs_replan_reconciliation')
                agent.advisory.assert_not_called()
                agent.asset_updates.assert_not_called()
                self.assertEqual(agent.s.data['checkpoints'], [])
