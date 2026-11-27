"""A new installation may create initial slots. Subsequent setups may only read them."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lite.providers import Models
from lite.storage import Pending
from tools import configure


class SharedRateSetupTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.install = self.root / 'installation'

    def test_fresh_configure_allows_the_first_reservation_for_each_provider(self):
        guide = self.root / 'guide.txt'
        guide.write_text('Test guide', encoding='utf-8')
        workspace = self.root / 'workspace'
        args = SimpleNamespace(install=self.install, workspace=workspace, guide=guide, cast=None,
                               source=configure.ROOT / 'examples/the_brass_key.txt', aspect='16:9',
                               language='English', gpu=0, profile='long', ffmpeg_bin=Path('/usr/bin'))
        with patch.object(configure, 'GUIDE_SHA', configure.digest(guide)), redirect_stdout(StringIO()):
            configure.configure(args)
        cfg = json.loads((workspace / 'config.json').read_text())
        models = Models(SimpleNamespace(check_stop=lambda: None), cfg)
        started = time.time()
        for provider in ('glm', 'doubao', 'gemini'):
            models._wait_rate(provider)
            state = json.loads((Path(cfg['rate_limit_root']) / (provider + '.json')).read_text())
            self.assertGreater(state['next_allowed_at'], started)

    def test_existing_provider_deadlines_are_preserved_byte_for_byte(self):
        folder = configure.shared_rate_limits(self.install)
        slot = folder / 'glm.json'
        original = b'{"next_allowed_at": 9999999999, "retained": true}\n'
        slot.write_bytes(original)
        self.assertEqual(configure.shared_rate_limits(self.install), folder)
        self.assertEqual(slot.read_bytes(), original)

    def test_partial_or_invalid_existing_state_cannot_be_reinitialized(self):
        folder = configure.shared_rate_limits(self.install)
        slot = folder / 'gemini.json'
        for value in (None, '{"next_allowed_at": false}', '{"next_allowed_at": NaN}'):
            with self.subTest(value=value):
                if value is None:
                    slot.unlink()
                else:
                    slot.write_text(value)
                with patch('time.sleep', return_value=None), self.assertRaises(Pending):
                    configure.shared_rate_limits(self.install)
                self.assertEqual(slot.read_text() if slot.exists() else None, value)

    def test_missing_directory_in_existing_installation_is_not_recreated(self):
        (self.install / 'backend').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'reconcile'):
            configure.shared_rate_limits(self.install)
        self.assertFalse((self.install / 'rate-limits').exists())

    def test_concurrent_setups_publish_one_complete_set_of_slots(self):
        start = Barrier(4)
        def setup(_):
            start.wait(timeout=10)
            return configure.shared_rate_limits(self.install)
        with patch.object(configure, 'dump', wraps=configure.dump) as writes:
            with ThreadPoolExecutor(max_workers=4) as pool:
                folders = list(pool.map(setup, range(4)))
        self.assertEqual(writes.call_count, 3)
        self.assertEqual(len(set(folders)), 1)
        self.assertEqual({p.name for p in folders[0].iterdir()},
                         {'glm.json', 'doubao.json', 'gemini.json'})
