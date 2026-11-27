from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

from lite import asset_lifecycle, character_stages
from lite.media import Media
from lite.storage import Unavailable, sha


class CharacterAgeEditTests(unittest.TestCase):
    def test_declared_age_edit_does_not_follow_rank_or_clothing(self):
        examples = {
            'The same adult changes into an older jacket.': None,
            'The same man in travel clothes, with mud on his sleeves.': None,
            'Older brother of the protagonist in the same stage.': None,
            'Same man about ten years younger.': 'younger',
            'The same person, visibly older after the time jump.': 'older',
            'Portrait of the same person in old age, around 75 years old.': 'elderly',
            'The same person in his elderly years, with an older coat.': 'elderly',
            'Same character, younger or older version.': None,
            '同一人物的老年阶段，保留面部特征。': 'elderly',
        }
        for description, expected in examples.items():
            with self.subTest(description=description):
                self.assertEqual(character_stages.age_edit_mode({'description': description}), expected)

    def test_stage_description_drives_prompt_and_no_numeric_age_is_invented(self):
        target = 'The same person as an elderly woman in a blue shawl.'
        prompt = character_stages.age_edit_prompt(target, 'elderly', 'Use window light.')
        self.assertIn(target, prompt)
        self.assertIn('naturally greying hair', prompt)
        self.assertTrue(prompt.endswith('Use window light.'))
        self.assertNotRegex(prompt, r'\d+ years old')

    def test_specific_visible_defect_is_not_used_as_reference(self):
        accepted = {'status': 'clear', 'evidence': 'The eyes are stable.'}
        rejected = {'status': 'defect', 'evidence': 'The skin is visibly waxy.'}
        unknown = {'status': 'unknown', 'evidence': 'The face is partially occluded.'}
        defects = character_stages.age_review_defects(
            {'character_age_review': {'identity': accepted, 'age_change': unknown,
                                      'natural_texture': rejected}})
        self.assertEqual(defects, [{'dimension': 'natural_texture', 'evidence': rejected['evidence']}])
        self.assertEqual(character_stages.age_review_defects({}), [])
        with TemporaryDirectory() as directory:
            image = Path(directory) / 'portrait.png'
            Image.new('RGB', (16, 16)).save(image)
            previous = {'path': str(image), 'sha256': sha(image), 'version': 1}
            candidate = {**previous, 'version': 2, 'character_age_defects': defects}
            row = {'spec': {'role': 'character'}, 'versions': [previous, candidate]}
            self.assertEqual(asset_lifecycle.usable_version(row), previous)

    def test_failed_age_candidate_cannot_be_promoted_by_lifecycle(self):
        row = {'spec': {'role': 'character'}, 'versions': []}
        agent = SimpleNamespace(registry=SimpleNamespace(
            publish=lambda *_: self.fail('published a known defective stage portrait'), save=lambda: None))
        candidate = {'path': 'portrait.png', 'character_age_defects': [{'dimension': 'identity'}]}
        asset_lifecycle.finish_asset(agent, 'portrait', row, [candidate], 1, optional=True)
        self.assertEqual(row['dependency_state'], 'optional_unavailable')

    def test_age_edit_rejects_multiple_sources_before_launch(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            canonical = run / 'canonical.png'
            Image.new('RGB', (16, 16)).save(canonical)
            parent = {'path': str(canonical)}
            media = Media.__new__(Media)
            media.store = SimpleNamespace(run=run)
            media.root = run
            media.config = {'image_provider': {'model': 'Qwen-Image-2.1'}}
            media._worker = lambda *_args, **_kwargs: self.fail('launched an invalid request')
            with self.assertRaisesRegex(Unavailable, 'one canonical parent'):
                media.image('stage', 'Age the same person.', parents=[parent, parent], age_edit='older')
