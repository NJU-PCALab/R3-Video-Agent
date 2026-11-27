"""Exercise normal CLI recovery without a model, media backend, or GPU."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lite import __main__ as cli
from lite.media import Media
from lite.storage import Pending, Store, read, sha, write


class FinalizationReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run = self.root / 'run'
        self.run.mkdir()
        self.source = self.run / 'source.txt'
        self.source.write_text('Approved source.')
        self.config = {'lite_root': str(Path(cli.__file__).parents[1]), 'source_text_sha256': sha(self.source)}
        write(self.run / 'config.json', self.config)
        self.old = {'pid': 120, 'start_ticks': '100', 'argv': ['python', '-m', 'lite', 'start']}
        self.new = {'pid': 130, 'start_ticks': '200', 'argv': ['python', '-m', 'lite', 'resume']}
        write(self.run / 'process.json', self.old)
        store = Store(self.run, existing=False)
        media = self.run / 'media'
        media.mkdir()
        video = media / 'viewing.mp4'
        audio = media / 'native_audio.wav'
        video.write_bytes(b'original clip bytes')
        audio.write_bytes(b'original native audio bytes')
        clip = {'id': 'clip_001'}
        selected = {
            'clip': clip, 'take': 1, 'quality': {'ok': False},
            'receipt': {'path': str(video), 'sha256': sha(video), 'frames': 96, 'fps': 24,
                        'audio_path': str(audio), 'audio_sha256': sha(audio), 'native_audio': True},
        }
        store.data.update(plan={'clips': [clip]}, selected=[selected], status='running',
                          current={'stage': 'film.assemble'})
        store.save()
        self.state_before = (self.run / 'state.json').read_bytes()
        folder = self.run / 'delivery'
        folder.mkdir()
        self.final = folder / 'final.mp4'
        self.final.write_bytes(b'previously verified delivery bytes')
        self.report = {
            'delivery': {'path': str(self.final), 'sha256': sha(self.final), 'frames': 96, 'fps': 24,
                         'seconds': 4.0, 'video_streams': 1, 'audio_streams': 1, 'native_audio': True,
                         'production_complete': True,
                         'source_clips': [{'clip_id': 'clip_001', 'path': str(video), 'sha256': sha(video)}]},
            'clips': [selected], 'source_sha256': sha(self.source), 'counts': {}, 'usage': {},
            'quality_status': 'draft/needs_review', 'technical_completion': True,
            'content_acceptance_passed': False,
        }
        write(folder / 'technical.json', self.report['delivery'])
        write(self.run / 'report.json', self.report)

    def resume(self):
        with patch('sys.argv', ['lite', 'resume', '--run', str(self.run)]), \
                patch.object(cli, 'process_identity', return_value=self.new), \
                patch.object(cli, 'FilmAgent') as agent, patch.object(cli, 'Models') as models, \
                patch.object(cli, 'Media') as media, patch.object(cli, 'write', wraps=write) as writer:
            # The framework-level current pointer normally lives outside this test's run.
            writer.side_effect = lambda path, data: None if Path(path).name == 'current.json' else write(path, data)
            try:
                return cli.main()
            finally:
                agent.assert_not_called()
                models.assert_not_called()
                media.assert_not_called()

    def test_resume_after_report_preserves_delivery_and_warning_and_records_exact_new_exit(self):
        before = (self.run / 'report.json').read_bytes()
        self.assertEqual(self.resume(), 0)
        self.assertEqual((self.run / 'report.json').read_bytes(), before)
        self.assertEqual(sha(self.final), self.report['delivery']['sha256'])
        state = read(self.run / 'state.json')
        self.assertEqual(state['status'], 'completed_with_warnings')
        self.assertEqual(state['delivery'], self.report['delivery'])
        (exit_path,) = (self.run / 'exits').glob('*.json')
        exit_record = read(exit_path)
        self.assertEqual(exit_record['process'], self.new)
        self.assertEqual(exit_record['returncode'], 0)
        with self.assertRaisesRegex(Pending, 'immutable'):
            self.resume()
        self.assertEqual(list((self.run / 'exits').glob('*.json')), [exit_path])

    def test_resume_after_terminal_state_without_exit_finishes(self):
        state = read(self.run / 'state.json')
        state.update(status='completed_with_warnings', delivery=self.report['delivery'], current={'stage': 'film.finished'})
        write(self.run / 'state.json', state)
        self.assertEqual(self.resume(), 0)

    def test_changed_final_does_not_admit_any_runner_or_replace_artifacts(self):
        self.final.write_bytes(b'changed delivery bytes')
        with self.assertRaisesRegex(Pending, 'Final bytes differ'):
            self.resume()
        self.assertEqual(read(self.run / 'process.json'), self.old)
        self.assertEqual((self.run / 'state.json').read_bytes(), self.state_before)
        self.assertEqual(self.final.read_bytes(), b'changed delivery bytes')
        self.assertFalse((self.run / 'exits').exists())

    def test_changed_native_audio_blocks_replay(self):
        Path(self.report['clips'][0]['receipt']['audio_path']).write_bytes(b'changed soundtrack')
        with self.assertRaisesRegex(Pending, 'native PCM'):
            self.resume()
        self.assertEqual(read(self.run / 'process.json'), self.old)

    def test_changed_story_source_blocks_resume_without_models(self):
        self.source.write_text('Different source.')
        with self.assertRaisesRegex(Pending, 'authorized hash'):
            self.resume()
        self.assertEqual(read(self.run / 'process.json'), self.old)

    def test_saved_source_must_match_benchmark_task_even_if_recorded_hash_matches(self):
        write(self.run / 'config.json', {**self.config, 'task_contract': {'source_sha256': '0' * 64}})
        with self.assertRaisesRegex(Pending, 'authorized benchmark task hash'):
            self.resume()
        self.assertEqual(read(self.run / 'process.json'), self.old)

    def assembly_media(self):
        (self.run / 'report.json').unlink()
        media = object.__new__(Media)
        media.store = Store(self.run)
        media.config = {}
        media.ffmpeg = '/fixture/ffmpeg'
        return media

    def test_assembly_after_technical_receipt_reuses_verified_file_without_ffmpeg(self):
        media = self.assembly_media()
        technical = self.run / 'delivery/technical.json'
        before = technical.read_bytes()
        with patch.object(media, 'command') as command, patch.object(media, 'probe') as probe:
            self.assertEqual(media.assemble(self.report['clips']), self.report['delivery'])
            command.assert_not_called()
            probe.assert_not_called()
        self.assertEqual(technical.read_bytes(), before)
        self.assertEqual(sha(self.final), self.report['delivery']['sha256'])

    def test_assembly_does_not_replace_a_final_with_mismatching_source_order(self):
        media = self.assembly_media()
        first = self.report['clips'][0]
        second = {**first, 'clip': {'id': 'clip_002'}}
        technical = self.run / 'delivery/technical.json'
        original = read(technical, required=True)
        original.update(frames=192, seconds=8.0, source_clips=[
            original['source_clips'][0], {**original['source_clips'][0], 'clip_id': 'clip_002'},
        ])
        write(technical, original)
        with patch.object(media, 'command') as command:
            with self.assertRaisesRegex(Pending, 'Final source receipt order differs'):
                media.assemble([second, first])
            command.assert_not_called()
        self.assertEqual(read(technical), original)
        self.assertEqual(sha(self.final), original['sha256'])

    def test_assembly_does_not_replace_a_final_with_changed_native_pcm(self):
        media = self.assembly_media()
        Path(self.report['clips'][0]['receipt']['audio_path']).write_bytes(b'changed soundtrack')
        with patch.object(media, 'command') as command:
            with self.assertRaisesRegex(Pending, 'native PCM'):
                media.assemble(self.report['clips'])
            command.assert_not_called()
        self.assertEqual(sha(self.final), self.report['delivery']['sha256'])

    def test_preview_and_selective_repair_retain_existing_assembly_behavior(self):
        media = self.assembly_media()
        with patch.object(media, 'command', side_effect=RuntimeError('normal encoding path')) as command:
            with self.assertRaisesRegex(RuntimeError, 'normal encoding path'):
                media.assemble(self.report['clips'], partial=True)
            self.assertEqual(command.call_args.args[0][-1], self.run / 'partial_delivery/partial.mp4')
            media.config['selective_repair'] = {'parent_run': '/fixture/parent'}
            with self.assertRaisesRegex(RuntimeError, 'normal encoding path'):
                media.assemble(self.report['clips'])
            self.assertEqual(command.call_args.args[0][-1], self.final)


class InitializationLedgerTests(unittest.TestCase):
    def test_start_checks_task_hash_before_preflight_and_copies_checked_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'story.txt'
            approved = b'Approved source.\r\n'
            source.write_bytes(approved)
            config = root / 'prepared.json'
            digest = sha(source)
            write(config, {'lite_root': str(root), 'task_contract': {'source_sha256': digest}})
            run = root / 'new'
            args = ['lite', 'start', '--run', str(run), '--source', str(source), '--config', str(config)]

            def preflight_with_source_changed(_):
                source.write_bytes(b'Unapproved source after the identity check')
                return {'ready': True}

            with patch('sys.argv', args), patch.object(cli, 'preflight', side_effect=preflight_with_source_changed) as preflight, \
                    patch.object(cli.asset_lifecycle, 'admit_entry'), patch.object(cli, 'FilmAgent') as agent, \
                    patch.object(cli, 'Models'), patch.object(cli, 'Media'):
                source.write_bytes(b'Unapproved source before the identity check')
                with self.assertRaisesRegex(Pending, 'authorized benchmark task hash'):
                    cli.main()
                preflight.assert_not_called()
                agent.assert_not_called()
                self.assertFalse((run / 'source.txt').exists())

                source.write_bytes(approved)
                self.assertEqual(cli.main(), 0)
                preflight.assert_called_once()
                agent.assert_called_once()
            self.assertEqual((run / 'source.txt').read_bytes(), approved)
            self.assertEqual(read(run / 'config.json')['source_text_sha256'], digest)

    def test_start_persists_empty_state_before_writing_process_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'story.txt'
            source.write_text('Approved source.')
            config = root / 'prepared.json'
            write(config, {'lite_root': str(root)})
            run = root / 'new'

            def interrupt_process_receipt(path, value):
                if Path(path).name == 'process.json':
                    state = read(run / 'state.json', required=True)
                    self.assertEqual(state['status'], 'created')
                    self.assertEqual(state['accounted_operations'], [])
                    self.assertEqual(state['counts'], {})
                    raise OSError('interrupted process publication')
                write(path, value)

            args = ['lite', 'start', '--run', str(run), '--source', str(source), '--config', str(config)]
            with patch('sys.argv', args), patch.object(cli, 'preflight', return_value={'ready': True}), \
                    patch.object(cli.asset_lifecycle, 'admit_entry'), patch.object(cli, 'FilmAgent') as agent, \
                    patch.object(cli, 'write', side_effect=interrupt_process_receipt):
                with self.assertRaisesRegex(OSError, 'interrupted process publication'):
                    cli.main()
                agent.assert_not_called()
            self.assertEqual(read(run / 'config.json')['source_text_sha256'], sha(run / 'source.txt'))
            self.assertFalse((run / 'process.json').exists())


if __name__ == '__main__':
    unittest.main()
