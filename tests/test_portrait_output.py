"""Verify that film orientation reaches the actual sampler and the viewing export."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from lite.media import Media, portrait_crop, sampling_size, verify_viewing_geometry
from lite.storage import Unavailable


class CapturedSubmission(Exception):
    pass


class PortraitOutputTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.media = object.__new__(Media)
        self.media.root = Path('/workers')
        self.media.store = SimpleNamespace(run=Path(temp.name))
        self.media.config = {'video_aspect_ratio':'9:16', 'image_provider':{'python':'python', 'model_path':'weights', 'model':'image'}}
        self.profile = SimpleNamespace(export_width=1344, export_height=768, cuda_device='0',
                 diffsynth_python='python', diffsynth_project='vendor', ref2va_weights='ref', fl2va_weights='fl',
                 inference_steps=8, acceleration_profile='pdd8_exact', attention_backend='flash_attn_3',
                 vram_reserve_gb=5, pdd_lora_path='lora', pdd_lora_sha256='abc', pdd_base_variant='ref2va')
        self.media.h3 = SimpleNamespace(effective_config=lambda mode: self.profile)
        def capture(folder, request, command, **kw):
            self.request, self.command = request, command
            raise CapturedSubmission
        self.media._worker = capture

    def test_sampler_and_request_match_all_three_modes(self):
        img = dict(id='actor', version=1, kind='image', path='/fixed/actor.png', sha256='hash')
        for refs, mode in (([], 't2va'), ([img], 'ref2va'), ([{**img,'frame_index':0}], 'fl2va')):
            with self.subTest(mode=mode), self.assertRaises(CapturedSubmission):
                self.media.render(mode, 'Look up.', 4, refs, 12)
            self.assertEqual(self.request['task'], mode)
            self.assertEqual(self.request['target'], {'short_edge':768, 'aspect_ratio':'9:16', 'duration_seconds':4})
            self.assertEqual((self.command[self.command.index('--width')+1],
                              self.command[self.command.index('--height')+1]), ('768','1344'))
        self.profile.export_width, self.profile.export_height = 768, 1344
        self.assertEqual(sampling_size(self.profile, '9:16'), (768,1344))
        self.profile.export_height=769
        with self.assertRaises(ValueError):
            sampling_size(self.profile, '9:16')

    def test_portrait_reference_and_legacy_landscape(self):
        with self.assertRaises(CapturedSubmission):
            self.media.image('actor','An office worker.')
        self.assertEqual((self.request['width'], self.request['height']), (768,1344))
        del self.media.config['video_aspect_ratio']
        with self.assertRaises(CapturedSubmission):
            self.media.image('actor','An office worker.')
        self.assertEqual((self.request['width'], self.request['height']), (1344,768))
        with self.assertRaises(CapturedSubmission):
            self.media.render('legacy','Looks up.',4,[],12)
        self.assertEqual(self.request['target']['aspect_ratio'],'16:9')
        self.assertEqual((self.command[self.command.index('--width')+1],
                          self.command[self.command.index('--height')+1]), ('1344','768'))

    def test_crop_preserves_native_portrait_and_refuses_rotated_or_anamorphic_source(self):
        self.assertEqual(portrait_crop(768,1344),(756,1344))
        with self.assertRaises(Unavailable):
            portrait_crop(1344,768)
        verify_viewing_geometry(dict(width=756,height=1344,sample_aspect_ratio='1:1'), self.media.config)
        for width,sar in ((768,'1:1'),(756,'4:3')):
            with self.assertRaises(Unavailable):
                verify_viewing_geometry(dict(width=width,height=1344,sample_aspect_ratio=sar), self.media.config)
