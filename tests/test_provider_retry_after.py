"""Offline 429 transport and replay checks, with a deterministic clock."""
import io
import tempfile
import unittest
from email.utils import formatdate
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from lite import providers
from lite.storage import Pending, Stopped, Store, Unavailable, key, read, write


class RetryAfterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / 'test-run')
        self.models = providers.Models(self.store, {'text_route': {'model': 'offline'}})
        self.folder = self.store.run / 'calls' / key('test')
        self.now = 1_800_000_000
        self.sends = []
        self.outcomes = []
        self.fake_clock = SimpleNamespace(time=lambda: self.now, monotonic=lambda: self.now,
                                          sleep=self.advance)
        patches = [
            patch.object(providers, 'time', self.fake_clock),
            patch('lite.storage._read_pause', return_value=None),
            patch.object(self.models, '_wait_rate'),
            patch.object(providers, 'resolve_openai_credentials',
                         return_value=SimpleNamespace(base_url='https://example.invalid', headers=dict)),
            patch('urllib.request.urlopen', side_effect=self.transport),
        ]
        for change in patches:
            change.start()
            self.addCleanup(change.stop)

    def advance(self, seconds):
        self.now += seconds

    def transport(self, request, timeout):
        self.sends.append((self.now, request.data))
        next_result = self.outcomes.pop(0)
        if isinstance(next_result, Exception):
            raise next_result
        return next_result

    @staticmethod
    def refusal(value, status=429):
        headers = {} if value is None else {'Retry-After': value}
        return HTTPError('https://example.invalid', status, 'error', headers, io.BytesIO(b'error'))

    @staticmethod
    def success():
        response = Mock(status=200, headers={})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"done"}}]}'
        return response

    def invoke(self):
        return self.models.text('test', 'system', 'same story')

    def test_429_seconds_delays_next_original_attempt(self):
        start = self.now
        self.outcomes = [self.refusal('75'), self.success()]
        self.assertEqual(self.invoke(), 'done')
        self.assertEqual([t for t, _ in self.sends], [start, start + 75])
        self.assertEqual(self.sends[0][1], self.sends[1][1])
        self.assertEqual(read(self.folder / 'physical_1/failure.json')['retry_not_before'], start + 75)
        self.assertEqual(self.store.data['counts']['llm_requests'], 2)
        self.assertEqual(self.invoke(), 'done')
        self.assertEqual(len(self.sends), 2)

    def test_future_date_survives_shutdown_and_long_wait_holds_no_worker(self):
        deadline = self.now + 7300
        self.outcomes = [self.refusal(formatdate(deadline, usegmt=True)), self.success()]
        for _ in range(2):
            with self.assertRaises(Pending) as caught:
                self.invoke()
            self.assertEqual(caught.exception.status, 'awaiting_rate_limit')
            self.assertEqual(len(self.sends), 1)
            self.assertFalse((self.folder / 'physical_2/submitted.json').exists())
        self.advance(7290)
        self.assertEqual(self.invoke(), 'done')
        self.assertEqual(self.sends[-1][0], deadline)

    def test_stop_then_resume_only_waits_the_remaining_cooldown(self):
        start = self.now
        self.outcomes = [self.refusal('20'), self.success()]
        def stop_during_sleep(seconds):
            self.advance(seconds)
            (self.store.run / 'STOP').touch()
        self.fake_clock.sleep = stop_during_sleep
        with self.assertRaises(Stopped):
            self.invoke()
        self.assertEqual(self.now, start + 5)
        self.assertEqual(len(self.sends), 1)
        receipt = self.folder / 'physical_1/failure.json'
        legacy = read(receipt)
        del legacy['retry_not_before']  # Pre-upgrade receipt.
        write(receipt, legacy)
        (self.store.run / 'STOP').unlink()
        self.fake_clock.sleep = self.advance
        self.assertEqual(self.invoke(), 'done')
        self.assertEqual(self.sends[-1][0], start + 20)

    def test_default_backoff_and_attempt_limit(self):
        start = self.now
        self.outcomes = [self.refusal(None), self.refusal('invalid'), self.refusal('9000')]
        with self.assertRaises(Unavailable):
            self.invoke()
        with self.assertRaises(Unavailable):
            self.invoke()
        self.assertEqual([t for t, _ in self.sends], [start, start + 62, start + 124])
        self.assertFalse((self.folder / 'physical_4').exists())

    def test_unknown_500_with_retry_after_is_never_redispatched(self):
        self.outcomes = [self.refusal('0', status=500)]
        for _ in range(2):
            with self.assertRaises(Pending):
                self.invoke()
        self.assertEqual(len(self.sends), 1)
        self.assertFalse((self.folder / 'physical_2').exists())

    def test_timeout_without_http_acknowledgement_stays_pending(self):
        self.outcomes = [TimeoutError('lost reply')]
        for _ in range(2):
            with self.assertRaises(Pending):
                self.invoke()
        self.assertEqual(len(self.sends), 1)
        self.assertFalse((self.folder / 'physical_2').exists())

    def test_definitive_429_survives_loss_of_error_body(self):
        broken_body = io.BytesIO()
        broken_body.read = Mock(side_effect=TimeoutError('lost error body'))
        error = HTTPError('https://example.invalid', 429, 'error', {'retry-after': '0'}, broken_body)
        self.outcomes = [error, self.success()]
        self.assertEqual(self.invoke(), 'done')
        self.assertEqual(len(self.sends), 2)
        self.assertEqual(read(self.folder / 'physical_1/failure.json')['http_status'], 429)

    def test_oversized_delay_yields_without_shortening(self):
        self.outcomes = [self.refusal('9' * 320)]
        with self.assertRaises(Pending) as caught:
            self.invoke()
        self.assertEqual(caught.exception.status, 'awaiting_rate_limit')
        self.assertEqual(len(self.sends), 1)


if __name__ == '__main__':
    unittest.main()
