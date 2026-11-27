import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from lite.media import Media, qwen_edit_seed
from lite.storage import sha


class QwenEditSeedTest(unittest.TestCase):
    def test_request_uses_independent_seed_and_media_hash_for_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'story'
            run.mkdir()
            parent = run / 'canonical.png'
            Image.new('RGB', (16, 16), 'gray').save(parent)
            media = Media.__new__(Media)
            media.store = SimpleNamespace(run=run)
            media.root = run
            media.config = {
                'video_aspect_ratio': '9:16',
                'image_provider': {
                    'model': 'Qwen-Image-2.1',
                    'model_path': '/model/qwen',
                    'python': '/runtime/python',
                    'parameters': {'num_inference_steps': 40, 'true_cfg_scale': 1.0},
                },
            }
            requests = []

            def worker(folder, request, *args, **kwargs):
                folder.mkdir(parents=True, exist_ok=True)
                requests.append(request)
                path = folder / 'generated.png'
                Image.new('RGB', (16, 16), 'gray').save(path)
                return {'images': [{'path': str(path)}]}

            media._worker = worker
            media.image('canonical', 'A portrait.', seed=53)
            media.image('post_crash', 'Change the place and the age.', parents=[{'path':str(parent)}], seed=53)
            media.image('post_crash_retry', 'Change the place and the age.', parents=[{'path':str(parent)}], seed=53)
            original, edit, retry = requests
            digest = sha(parent)
            self.assertEqual(original['seed'], 53)
            self.assertEqual(original['parent_images'], [])
            self.assertEqual(edit['parent_images'], [{'path':str(parent), 'sha256':digest}])
            self.assertEqual(edit['seed'], qwen_edit_seed(run.name, 'post_crash', 53, [digest]))
            self.assertNotEqual(edit['seed'], original['seed'])
            self.assertNotEqual(edit['seed'], retry['seed'])
            self.assertEqual(edit['parameters'], original['parameters'])
            self.assertEqual((edit['width'], edit['height']), (original['width'], original['height']))
            media.image('post_crash', 'Change the place and the age.', parents=[{'path':str(parent)}], seed=53)
            self.assertEqual(len(requests), 3)

    def test_edit_namespace_changes_with_parent_and_operation(self):
        first = qwen_edit_seed('story', '1_1_actor', 53, ['parent'])
        self.assertEqual(first, qwen_edit_seed('story', '1_1_actor', 53, ['parent']))
        self.assertNotEqual(first, qwen_edit_seed('story', '1_1_actor', 53, ['another']))
        self.assertNotEqual(first, qwen_edit_seed('story', '1_2_actor', 54, ['parent']))
        self.assertNotEqual(first, qwen_edit_seed('other', '1_1_actor', 53, ['parent']))
        self.assertTrue(2**62 <= first < 2**63)
