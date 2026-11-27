"""Attach-only delivery verification, no network/model/generator invocation."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from lite import storage

spec = importlib.util.spec_from_file_location('formal_archive_guard', Path(__file__).parents[1] / 'tools/archive_guard.py')
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)


class ArchiveGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.run = self.root / 'fixture_run'; self.run.mkdir()
        self.expected = {'pid': 321, 'start_ticks': '876543', 'argv': ['python', '-B', '-m', 'lite', 'start', '--run', str(self.run)]}
        self.worker = {'pid': 432, 'start_ticks': '987654', 'argv': ['python', 'worker.py']}
        self.cfg = {'run': str(self.run), 'run_id': self.run.name, 'host': socket.gethostname(),
            'guard_dir': str(self.root / 'guard'), 'archive_dir': str(self.root / 'completions' / self.run.name),
            'expected_runner': self.expected, 'index': 12, 'story_id': 'story_fixture', 'title': 'Offline fixture',
            'classification': 'human_assisted_recovery_result'}
        media = self.run / 'media/videos/clip_001_take_1'; media.mkdir(parents=True)
        self.video = media / 'native.mp4'; self.video.write_bytes(b'actual source video fixture')
        self.pcm = media / 'native_audio.wav'; self.pcm.write_bytes(b'actual native PCM fixture')
        self.final = self.run / 'film.mp4'; self.final.write_bytes(b'actual assembled video fixture')
        self.ref = self.root / 'same_person.png'; self.ref.write_bytes(b'original image reference')
        self.receipt = {'path': str(self.video), 'sha256': storage.sha(self.video), 'audio_path': str(self.pcm),
            'audio_sha256': storage.sha(self.pcm), 'frames': 3600, 'fps': 24, 'native_audio': True}
        self.state = {'status': 'completed_with_warnings', 'plan': {'clips': [{'id': 'clip_001'}]},
            'selected': [{'clip': {'id': 'clip_001'}, 'take': 1, 'receipt': self.receipt}],
            'candidates': {'clip_001': [{'take': 1, 'source_clip': {'id': 'clip_001', 'dialogue': []},
                'review': {'unresolved': 'retained uncertain observation'}, 'caption': 'Original caption.',
                'references': [{'path': str(self.ref), 'sha256': storage.sha(self.ref)}]}]}}
        self.report = {'content_acceptance_passed': False, 'delivery': {'path': str(self.final), 'sha256': storage.sha(self.final),
            'frames': 3600, 'source_clips': [{'sha256': storage.sha(self.video)}]}}
        storage.write(self.run / 'process.json', {**self.expected, 'started_at': 1})
        storage.write(self.run / 'exits/exit.json', {'process': self.expected, 'returncode': 0})
        storage.write(self.run / 'config.json', {'media_bin': '/unused/offline-probe'})
        storage.write(self.run / 'state.json', self.state); storage.write(self.run / 'report.json', self.report)
        storage.write(media / 'process.json', self.worker); storage.write(media / 'submitted.json', {'original': True})
        self.probe = patch.object(guard, 'probe', side_effect=self.probe_fixture); self.probe.start(); self.addCleanup(self.probe.stop)
        self.identity = patch.object(guard, 'process_identity', return_value=None); self.identity.start(); self.addCleanup(self.identity.stop)
        self.groups = patch.object(guard, 'group_members', return_value={}); self.groups.start(); self.addCleanup(self.groups.stop)

    def probe_fixture(self, config, path):
        video = {'codec_type': 'video', 'avg_frame_rate': '24/1', 'nb_read_frames': '3600'}
        audio = {'codec_type': 'audio', 'codec_name': 'pcm_s16le' if Path(path) == self.pcm else 'aac'}
        return [audio] if Path(path) == self.pcm else [video, audio]

    def ack(self, receipt, **changes):
        ack = {'all_files_sha_verified': True, 'run_id': self.run.name,
            **{k: receipt[k] for k in ('archive_sha256', 'manifest_sha256', 'final_sha256')}, **changes}
        storage.write(Path(self.cfg['archive_dir']) / 'local_delivery_ack.json', ack)

    def prepare_review_replacement(self, status=500):
        from lite.explicit_recovery import replacement_claim
        from lite.providers import model_answer, visible
        from types import SimpleNamespace

        name = 'completion/review/clip_010/3/visual/format_1'
        call = self.run.resolve() / 'calls' / storage.key(name)
        original, replacement = call / 'physical_1', call / 'physical_2'
        request = call / 'request.json'
        storage.write(request, {'name': name, 'provider': 'doubao', 'body': {'input': 'immutable'}})
        for attempt in (original, replacement):
            storage.write(attempt / 'submitted.json', {'provider': 'doubao', 'name': name})
        failure = original / 'failure.json'
        storage.write(failure, {'http_status': status, 'definitive': False, 'usage': None, 'cost': None})
        permission = self.root / 'permission.json'
        receipt = {'version': 'one_unknown_request_replacement_v1', 'explicit_user_approval': True,
                   'run': str(self.run.resolve()), 'call_id': call.name, 'request_sha256': storage.sha(request),
                   'submitted_sha256': storage.sha(original / 'submitted.json'), 'failure_sha256': storage.sha(failure),
                   'replacement_attempt': 2, 'additional_attempts': 1, 'original_execution_and_cost_remain_unknown': True}
        if status != 502:
            receipt['original_http_status'] = status
        storage.write(permission, receipt)
        config = storage.read(self.run / 'config.json', required=True)
        config['unknown_request_recovery'] = {'receipt': str(permission), 'sha256': storage.sha(permission)}
        storage.write(self.run / 'config.json', config)
        owner = SimpleNamespace(run=self.run.resolve(), event=lambda *args, **kwargs: None)
        replacement_claim(owner, config, call, create=True)

        raw = {'choices': [{'message': {'content': '{"status":"observed"}', 'reasoning_content': 'retained raw only'}}],
               'model': 'fixture', 'usage': {'total_tokens': 3}}
        body = replacement / 'response.body'
        body.write_bytes(json.dumps(raw).encode())
        storage.write(replacement / 'response_metadata.json', {'http_status': 200, 'byte_count': body.stat().st_size,
            'request_sha256': storage.sha(request), 'raw_body_sha256': storage.sha(body)})
        storage.write(replacement / 'response.json', visible(raw))
        text, usage = model_answer('doubao', visible(raw))
        storage.write(call / 'answer.json', {'text': text, 'usage': usage})
        self.state.update(accounted_operations=[str(original), str(replacement)], usage={str(call): usage})
        storage.write(self.run / 'state.json', self.state)
        return call, permission

    def test_reader_is_the_one_shared_reader_and_required_missing_is_explicit(self):
        self.assertIs(guard.read, storage.read)
        (self.run / 'state.json').unlink()
        with patch.object(storage, '_read_pause'), self.assertRaises(FileNotFoundError): guard.verify_run(self.cfg)

    def test_live_missing_state_is_pending_not_a_guard_failure(self):
        (self.run / 'state.json').unlink()
        with patch.object(guard, 'process_identity', return_value=self.expected), patch.object(storage, '_read_pause'):
            result = guard.observe_once(self.cfg)
        self.assertEqual(result['status'], 'waiting_original_runner')
        self.assertEqual(result['actual_status'], 'snapshot_temporarily_unavailable')

    def test_malformed_live_state_is_not_silently_defaulted(self):
        (self.run / 'state.json').write_text('{broken')
        with patch.object(guard, 'process_identity', return_value=self.expected), self.assertRaises(json.JSONDecodeError): guard.observe_once(self.cfg)

    def test_pid_reuse_without_exact_exit_is_not_completion(self):
        (self.run / 'exits/exit.json').unlink()
        with patch.object(guard, 'process_identity', return_value={**self.expected, 'start_ticks': 'different'}):
            self.assertEqual(guard.observe_once(self.cfg)['status'], 'awaiting_exact_original_exit')
        self.assertFalse(Path(self.cfg['archive_dir']).exists())

    def test_wrong_process_or_ambiguous_exit_stops_without_redispatch(self):
        storage.write(self.run / 'process.json', {**self.expected, 'start_ticks': 'different'})
        with self.assertRaisesRegex(ValueError, 'process changed'): guard.observe_once(self.cfg)
        storage.write(self.run / 'process.json', self.expected)
        storage.write(self.run / 'exits/second.json', {'process': self.expected, 'returncode': 0})
        with self.assertRaisesRegex(ValueError, 'Ambiguous'): guard.observe_once(self.cfg)

    def test_nonzero_original_exit_never_becomes_content_failure_or_archive(self):
        storage.write(self.run / 'exits/exit.json', {'process': self.expected, 'returncode': 2})
        result = guard.observe_once(self.cfg)
        self.assertEqual(result['status'], 'needs_attention'); self.assertTrue(result['not_content_failure_by_default'])
        self.assertFalse(result['automatic_resume']); self.assertFalse(Path(self.cfg['archive_dir']).exists())

    def test_owned_worker_and_orphan_process_group_prevent_freeze(self):
        with patch.object(guard, 'process_identity', side_effect=lambda pid: self.worker if pid == 432 else None):
            with self.assertRaisesRegex(ValueError, 'worker'): guard.freeze(self.cfg)
        with patch.object(guard, 'group_members', return_value={543: {'pid': 543}}):
            with self.assertRaisesRegex(ValueError, 'group'): guard.freeze(self.cfg)

    def test_unresolved_worker_launch_and_model_submission_prevent_freeze(self):
        process = self.video.parent / 'process.json'; original = process.read_text(); process.unlink()
        with patch.object(storage, '_read_pause'), self.assertRaisesRegex(ValueError, 'launch identity'): guard.freeze(self.cfg)
        process.write_text(original)
        storage.write(self.run / 'calls/original/physical_1/submitted.json', {'original_paid_request': True})
        with patch.object(storage, '_read_pause'), self.assertRaisesRegex(ValueError, 'submission'): guard.freeze(self.cfg)

    def test_actual_frame_fps_and_native_pcm_are_required(self):
        for frames, fps, pcm in [('3599', '24/1', 'pcm_s16le'), ('3600', '25/1', 'pcm_s16le'), ('3600', '24/1', 'aac')]:
            def bad(config, path):
                result = self.probe_fixture(config, path)
                for stream in result:
                    if stream['codec_type'] == 'video': stream.update(nb_read_frames=frames, avg_frame_rate=fps)
                    if Path(path) == self.pcm: stream['codec_name'] = pcm
                return result
            with patch.object(guard, 'probe', side_effect=bad), self.assertRaises(ValueError): guard.freeze(self.cfg)

    def test_longform_archive_checks_the_same_authorized_native_frames_as_the_runner(self):
        authorization = self.root / 'authorization.json'
        storage.write(authorization, {'benchmark_id': 'fixture', 'physical_gpus': [0],
                                      'story_indices': ['film'], 'duration_seconds': [570, 630]})
        config = {'authorization': str(authorization), 'duration_admission': 'short_story_frames_v1',
                  'target_seconds': 600, 'max_final_seconds': 630, 'image_provider': {'device': 0},
                  'audio_provider': {'device': 0}, 'task_contract': {
                      'version': 'longform_task_parameters_v1', 'benchmark_id': 'fixture',
                      'story_index': 'film', 'physical_gpu': 0, 'language': 'Chinese',
                      'min_seconds': 570, 'max_seconds': 630}}
        storage.write(self.run / 'config.json', config)
        for frames, valid in [(14400, True), (3600, False)]:
            self.receipt['frames'] = self.report['delivery']['frames'] = frames
            storage.write(self.run / 'state.json', self.state)
            storage.write(self.run / 'report.json', self.report)

            def native_probe(config, path):
                streams = self.probe_fixture(config, path)
                for stream in streams:
                    if stream['codec_type'] == 'video':
                        stream['nb_read_frames'] = str(frames)
                return streams

            with self.subTest(frames=frames), patch.object(guard, 'probe', side_effect=native_probe):
                if valid:
                    self.assertEqual(guard.verify_run(self.cfg)[2]['actual_seconds'], 600)
                else:
                    with self.assertRaises(storage.Pending):
                        guard.verify_run(self.cfg)

    def test_lost_submission_marker_with_retained_response_body_is_still_unknown(self):
        folder = self.run / 'calls/original/physical_1'; folder.mkdir(parents=True)
        (folder / 'response.body').write_bytes(b'partial original paid response')
        with patch.object(storage, '_read_pause'), self.assertRaisesRegex(ValueError, 'submission'): guard.freeze(self.cfg)
        self.assertFalse((Path(self.cfg['archive_dir']) / 'remote_archive_receipt.json').exists())

    def test_authorized_500_and_legacy_502_retain_original_unknown_usage(self):
        for status in (502, 500):
            with self.subTest(status=status):
                call, permission = self.prepare_review_replacement(status)
                if status == 502:
                    self.assertEqual(guard.verify_run(self.cfg)[2]['authorized_review_replacements'][0]['original_http_status'], 502)
                    (call / 'explicit_replacement_claim.json').unlink()
                    continue
                failure = call / 'physical_1/failure.json'
                before = storage.sha(failure)
                guard.freeze(self.cfg)
                package = Path(self.cfg['archive_dir']) / 'package'
                proof = storage.read(package / 'technical_verification.json')['authorized_review_replacements'][0]
                self.assertEqual(proof['original_http_status'], 500)
                self.assertTrue(proof['original_execution_unknown'])
                self.assertIsNone(proof['original_usage'])
                self.assertIsNone(proof['original_cost'])
                self.assertEqual(storage.sha(package / failure.relative_to(self.run.resolve())), before)
                self.assertEqual(storage.sha(package / 'provenance' / ('authorized_review_replacement_' + call.name + '.json')),
                                 storage.sha(permission))
                self.assertEqual(storage.sha(failure), before)

    def test_replacement_needs_exact_claim_answer_original_ledger_and_one_slot(self):
        for name, field, value in (
            ('explicit_replacement_claim.json', 'permission_sha256', 'wrong'),
            ('physical_2/response_metadata.json', 'request_sha256', 'wrong'),
            ('physical_2/submitted.json', 'name', 'different stage'),
            ('physical_2/response.json', 'choices', []),
            ('answer.json', 'text', '{"different":true}'),
        ):
            with self.subTest(name=name):
                call, _ = self.prepare_review_replacement()
                path = call / name; saved = path.read_bytes()
                storage.write(path, {**storage.read(path), field: value})
                with self.assertRaisesRegex(ValueError, 'submission'): guard.verify_run(self.cfg)
                path.write_bytes(saved)
        self.state['usage'][str(call / 'physical_1')] = {'cost': 0}
        storage.write(self.run / 'state.json', self.state)
        with self.assertRaisesRegex(ValueError, 'submission'): guard.verify_run(self.cfg)
        self.prepare_review_replacement()
        (call / 'physical_3').mkdir()
        with self.assertRaisesRegex(ValueError, 'submission'): guard.verify_run(self.cfg)

    def test_invalid_replacement_envelope_is_read_only(self):
        call, _ = self.prepare_review_replacement()
        replacement = call / 'physical_2'
        body = replacement / 'response.body'; body.write_bytes(b'{"choices":[]}')
        storage.write(replacement / 'response.json', {'choices': []})
        path = replacement / 'response_metadata.json'
        storage.write(path, {**storage.read(path), 'raw_body_sha256': storage.sha(body), 'byte_count': body.stat().st_size})
        with self.assertRaisesRegex(ValueError, 'submission'): guard.verify_run(self.cfg)
        self.assertFalse((replacement / 'response_issue.json').exists())

    def test_actual_source_and_reference_sha_mismatch_never_publish_receipt(self):
        self.pcm.write_bytes(b'changed native PCM')
        with self.assertRaisesRegex(ValueError, 'native PCM'): guard.freeze(self.cfg)
        self.pcm.write_bytes(b'actual native PCM fixture'); self.ref.write_bytes(b'changed canonical reference')
        with self.assertRaisesRegex(ValueError, 'reference SHA'): guard.freeze(self.cfg)
        self.assertFalse((Path(self.cfg['archive_dir']) / 'remote_archive_receipt.json').exists())

    def test_full_ack_and_all_original_candidates_are_required_and_preserved(self):
        rejected = self.run / 'media/videos/clip_001_take_2/native.mp4'; rejected.parent.mkdir(); rejected.write_bytes(b'rejected original candidate')
        result = guard.observe_once(self.cfg); self.assertEqual(result['status'], 'awaiting_local_full_ack')
        folder = Path(self.cfg['archive_dir']); receipt = storage.read(folder / 'remote_archive_receipt.json')
        self.assertFalse(receipt['content_acceptance_passed']); self.assertEqual((folder / 'package' / rejected.relative_to(self.run)).read_bytes(), rejected.read_bytes())
        self.ack(receipt); self.assertEqual(guard.observe_once(self.cfg)['status'], 'completed_archived')
        self.assertFalse(guard.observe_once(self.cfg)['content_acceptance_passed'])
        self.ack(receipt, final_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'acknowledgement'): guard.observe_once(self.cfg)

    def test_interrupted_freeze_is_reusable_but_existing_different_bytes_are_preserved(self):
        original = guard.os.replace
        def fail_archive(source, target):
            if Path(target).name == 'delivery.tar.gz': raise OSError('injected archive rename interruption')
            return original(source, target)
        with patch.object(guard.os, 'replace', side_effect=fail_archive), self.assertRaises(OSError): guard.freeze(self.cfg)
        folder = Path(self.cfg['archive_dir']); self.assertFalse((folder / 'remote_archive_receipt.json').exists())
        receipt = guard.freeze(self.cfg); self.assertEqual(storage.sha(folder / 'delivery.tar.gz'), receipt['archive_sha256'])
        target = folder / 'package/final.mp4'; target.write_bytes(b'preserved conflicting evidence')
        with self.assertRaisesRegex(ValueError, 'frozen package'): guard.freeze(self.cfg)
        self.assertEqual(target.read_bytes(), b'preserved conflicting evidence')

    def test_long_ack_wait_reuses_only_receipt_verified_by_this_guard(self):
        receipt = guard.freeze(self.cfg)
        with patch.object(guard, 'freeze', side_effect=AssertionError('Repeated native decode during ACK wait')):
            self.assertEqual(guard.observe_once(self.cfg, receipt)['status'], 'awaiting_local_full_ack')
            self.ack(receipt); self.assertEqual(guard.observe_once(self.cfg, receipt)['status'], 'completed_archived')

    def test_stop_and_invalid_output_paths_do_not_create_claims(self):
        (self.run / 'STOP').write_text('retained STOP')
        self.assertEqual(guard.observe_once(self.cfg)['status'], 'stopped')
        cfg = deepcopy(self.cfg); cfg['archive_dir'] = str(self.run / 'completions' / self.run.name)
        with self.assertRaisesRegex(ValueError, 'outside'): guard.validate(cfg)
        self.assertFalse((Path(self.cfg['guard_dir']) / 'claim').exists())

    def test_cli_claim_is_consumed_once_and_contains_no_generator_dispatch(self):
        path = self.root / 'guard_config.json'; storage.write(path, self.cfg)
        process = {'pid': os.getpid(), 'start_ticks': '111', 'argv': ['python', 'archive_guard.py']}
        with patch.object(guard.sys, 'argv', ['archive_guard.py', '--config', str(path)]), patch.object(guard, 'process_identity', return_value=process), patch.object(guard, 'observe_once', return_value={'status': 'completed_archived'}), patch.object(guard.subprocess, 'Popen') as spawn:
            self.assertEqual(guard.main(), 0)
            with self.assertRaises(FileExistsError): guard.main()
            spawn.assert_not_called()
        handoff = storage.read(Path(self.cfg['guard_dir']) / 'handoff.json')
        self.assertFalse(handoff['runner_started']); self.assertFalse(handoff['runner_resumed'])


if __name__ == '__main__':
    unittest.main()
