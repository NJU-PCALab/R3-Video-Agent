"""Exercise real kernel locks with a separate owner process."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lite.storage import Pending, lock


class LocalLockDeadlineTests(unittest.TestCase):
    def test_contender_times_out_and_owner_retains_the_same_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.lock'
            program = (
                "import fcntl, sys\n"
                "with open(sys.argv[1], 'a+') as f:\n"
                "    fcntl.flock(f, fcntl.LOCK_EX)\n"
                "    print('held', flush=True)\n"
                "    sys.stdin.read(1)\n"
            )
            with subprocess.Popen([sys.executable, '-c', program, str(path)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as owner:
                try:
                    self.assertEqual(owner.stdout.readline().strip(), 'held')
                    inode = path.stat().st_ino
                    with self.assertRaises(Pending) as error, lock(path, timeout=.05):
                        self.fail('Contended lock unexpectedly acquired')
                    self.assertEqual(error.exception.status, 'awaiting_shared_lock')
                    with self.assertRaises(BlockingIOError), lock(path, nonblocking=True):
                        self.fail('Immediate acquisition must respect the live owner')
                finally:
                    owner.communicate('x', timeout=5)
            with lock(path, timeout=0):
                self.assertEqual(path.stat().st_ino, inode)

    def test_exception_releases_lock_and_invalid_timeout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rate.lock'
            with self.assertRaises(ValueError), lock(path):
                raise ValueError('worker failure')
            with lock(path, nonblocking=True):
                pass
            for invalid in (-1, True, float('inf'), float('nan')):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError), lock(path, timeout=invalid):
                    self.fail('Invalid timeout accepted')
