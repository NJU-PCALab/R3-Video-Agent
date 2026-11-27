"""The workspace and the backend used by its workers must agree."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from lite import asset_lifecycle
from lite.__main__ import preflight

ROOT = Path(__file__).resolve().parents[1]


class SharedBackendTests(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.backend = Path(home.name) / 'backend'
        shutil.copytree(ROOT / 'backend', self.backend)
        self.config = json.loads((ROOT / 'config.example.json').read_text())
        self.config.update(framework_release=str(ROOT), production_root=str(self.backend))

    def test_checks_dispatched_worker_and_imported_backend_modules(self):
        self.assertTrue(asset_lifecycle.verify_runtime(self.config))
        for name in ('scripts/h3_diffsynth_worker.py', 'scripts/avh_model_worker.py',
                     'lib/h3_av/config.py'):
            with self.subTest(name=name):
                path = self.backend / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n# changed\n')
                self.assertFalse(asset_lifecycle.verify_runtime(self.config))
                path.write_bytes(original)
                self.assertTrue(asset_lifecycle.verify_runtime(self.config))

    def test_missing_shared_module_stops_preflight_before_dispatch_or_import(self):
        (self.backend / 'lib/h3_av/config.py').unlink()
        self.config['media_adapters'] = {'image': {}}
        with patch.object(asset_lifecycle, 'admit_entry'), patch('lite.media_ports.ready') as adapter_check:
            result = preflight(self.config)
        self.assertFalse(result['ready'])
        self.assertFalse(result['checks']['asset_worker_execution_contract'])
        adapter_check.assert_not_called()
