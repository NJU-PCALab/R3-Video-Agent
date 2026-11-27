"""Offline checks for the source release's runnable example."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from lite import asset_lifecycle, binding_contract, duration_budget, task_parameters
from lite.storage import Pending


ROOT = Path(__file__).resolve().parents[1]


class OpenSourceContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'config.example.json').read_text())
        self.auth = json.loads((ROOT / 'examples/authorization.example.json').read_text())
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.authorization = Path(folder.name) / 'authorization.json'
        self.authorization.write_text(json.dumps(self.auth))
        self.config['authorization'] = str(self.authorization)

    def test_default_task_admits_and_binds_source_and_cast(self):
        task_parameters.validate(self.config)
        asset_lifecycle.admit_entry(self.config, new_story=True)
        binding_contract.validate_cast(self.config['generation_contract']['cast'])
        task = self.config['task_contract']
        self.assertEqual(task['source_sha256'],
                         hashlib.sha256((ROOT / 'examples/the_brass_key.txt').read_bytes()).hexdigest())
        self.assertTrue(task_parameters.admission(self.config, 0))
        self.assertFalse(task_parameters.admission(self.config, 1))
        duration_budget.admit(self.config, [{'seconds': 225}])
        with self.assertRaises(Pending):
            duration_budget.admit(self.config, [{'seconds': 301}])

    def test_unbound_authorization_and_gpu_are_rejected(self):
        config = deepcopy(self.config)
        config['image_provider']['device'] = '1'
        with self.assertRaises(Pending):
            asset_lifecycle.admit_entry(config, new_story=True)
        self.auth['benchmark_id'] = 'unrelated'
        self.authorization.write_text(json.dumps(self.auth))
        with self.assertRaises(Pending):
            asset_lifecycle.admit_entry(self.config, new_story=True)

    def test_split_workspace_and_explicit_readiness_path_are_documented(self):
        workspace = Path(self.config['lite_root'])
        expected_readiness = workspace / 'verification/audio_runtime_probe/readiness.json'
        self.assertEqual(Path(self.config['audio_provider']['readiness']), expected_readiness)
        self.assertNotEqual(workspace, Path(self.config['framework_release']))
        self.assertTrue((ROOT / 'tools/probe_audio_runtime.py').is_file())


if __name__ == '__main__':
    unittest.main()
