import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lite import prompts, task_parameters
from lite.agent import FilmAgent
from lite.providers import Models
from lite.storage import key


class SceneLevelScreenplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'task_contract': {'version': task_parameters.LONGFORM_VERSION, 'language':'Chinese'},
                    'target_seconds':600,'max_final_seconds':630,'generation_contract':{'cast':{'林':{'voice_asset_id':None}}}}

    def test_role_specific_prompts(self):
        film = FilmAgent.__new__(FilmAgent)
        film.config = self.cfg
        for role in (prompts.WRITER, prompts.EDITOR):
            prompt = film.screenplay_prompt(role)
            self.assertIn('scene level', prompt)
            self.assertIn('林', prompt)
            self.assertIn('Chinese', prompt)
            self.assertNotIn('full ordered shot list', prompt)
            self.assertNotIn('render_spec', prompt)
        self.assertIn('full ordered shot list', task_parameters.prompt(prompts.BOUND_DIRECTOR, self.cfg))

    def test_only_confirmed_truncation_in_longform_writing_or_storyboarding_skips_broken_response(self):
        for name, reason, omit in [('screenplay/initial', 'length', True), ('screenplay/rewrite_2', 'length', True),
                                   ('screenplay/initial', 'stop', False), ('director/initial', 'length', True),
                                   ('director/rewrite', 'length', True), ('director/duration_correction', 'length', True),
                                   ('director/initial', 'stop', False), ('director/review','length', False)]:
            with self.subTest(name=name,reason=reason), TemporaryDirectory() as tmp, patch('lite.memory.ExperienceMemory'):
                run = Path(tmp)
                folder = run / 'calls' / key(name + '/format_1')
                folder.mkdir(parents=True)
                (folder/'answer.json').write_text(json.dumps({'text':'{"x":','usage':{'finish_reason':reason}}))
                model = Models.__new__(Models)
                model.store = SimpleNamespace(run=run,warning=Mock())
                model.config = self.cfg
                model.text = Mock(side_effect=['{"x":','{"ok":1}'])
                original = {'source':'Retain the final scene.'}
                self.assertEqual(model.json(name,'system',original,max_tokens=20000),{'ok':1})
                args, kwargs = model.text.call_args_list[1]
                self.assertEqual(args[2]['original_task'],original)
                self.assertEqual('original_response' not in args[2],omit)
                self.assertEqual(kwargs['max_tokens'],20000)
                if omit and name.startswith('director/'):
                    self.assertIn('render specification', args[2]['instruction'])
