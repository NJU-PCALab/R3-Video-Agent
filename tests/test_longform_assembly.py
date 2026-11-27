"""Long-film assembly preserves frame-clock boundaries and publication on retry."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from lite.media import Media
from lite.storage import Store, Unavailable, sha


class LongformAssemblyTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.media = object.__new__(Media)
        self.media.store = Store(Path(directory.name)/'run')
        self.media.config = {'task_contract': {'version': 'longform_task_parameters_v1'}}
        self.media.ffmpeg = 'ffmpeg'
        self.root = self.media.store.run
        inputs = self.root/'media'
        inputs.mkdir()
        self.rows = []
        for i in range(2):
            video = inputs/f"actor's shot {i}.mp4"
            audio = inputs/f'character {i}.wav'
            video.write_bytes(b'encoded shot')
            audio.write_bytes(b'native sound')
            self.rows.append({'clip':{'id':str(i)},'receipt':{
                'path':str(video),'sha256':sha(video),'audio_path':str(audio),'audio_sha256':sha(audio),
                'frames':1,'fps':24,'native_audio':True}})
        self.reads = []
        self.second_short_by = 0
        self.media.command = self.command
        self.media.probe = lambda path: {
            'path':str(path),'frames':2,'fps':24,'seconds':2/24,'video_streams':1,
            'audio_streams':1,'sha256':sha(path)}

    def command(self, args):
        if args[-1] == 'pipe:1':
            self.reads.append(Path(args[args.index('-i')+1]).name)
            samples = int(args[args.index('-af')+1].split('end_sample=')[1])
            if len(self.reads) == 2:
                samples -= self.second_short_by
            return bytes([len(self.reads)])*(samples*4)
        files = [args[i+1] for i, arg in enumerate(args) if arg == '-i']
        self.playlist = Path(files[0]).read_text()
        self.soundtrack = Path(files[1]).read_bytes()
        self.assertEqual(args[args.index('-c:v')+1], 'copy')
        Path(args[-1]).write_bytes(b'complete mux')

    def test_order_fractional_boundaries_and_cached_receipt_on_replay(self):
        self.second_short_by = 1
        result = self.media.assemble(self.rows)
        self.assertEqual(self.reads, ['character 0.wav','character 1.wav'])
        self.assertEqual(self.playlist.count('duration 0.041666667'), 2)
        self.assertIn("actor'\\''s shot 0.mp4", self.playlist)
        self.assertEqual(self.soundtrack, bytes([1])*(1333*4)+bytes([2])*(1333*4)+bytes(4))
        self.assertEqual(result['native_pcm_alignment']['boundary_padding'],
                         [{'clip_id':'1','samples_per_channel':1}])
        self.media.command = Mock(side_effect=AssertionError('completed receipt must be reused'))
        self.assertEqual(self.media.assemble(self.rows), result)

    def test_truncated_native_audio_cannot_publish_a_final_or_technical_receipt(self):
        self.second_short_by = 669
        with self.assertRaisesRegex(Unavailable, 'does not cover'):
            self.media.assemble(self.rows)
        self.assertFalse((self.root/'delivery/final.mp4').exists())
        self.assertFalse((self.root/'delivery/technical.json').exists())

    def test_mismatching_frame_count_does_not_publish_a_partial_mux(self):
        self.media.probe = lambda path: {'frames':1, 'fps':24, 'audio_streams':1}
        with self.assertRaisesRegex(Unavailable, 'frame/fps'):
            self.media.assemble(self.rows)
        self.assertFalse((self.root/'delivery/final.mp4').exists())
        self.assertFalse((self.root/'delivery/technical.json').exists())


if __name__ == '__main__':
    unittest.main()
