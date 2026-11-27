"""Offline checks for the public provider and media transport boundaries."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lite.credentials import CredentialSourceError, resolve_openai_credentials
from lite.media import Media
from lite.providers import gemini_credentials, gemini_endpoint
from lite.storage import Pending, Store
from lite import media_ports


class ProviderTests(unittest.TestCase):
    def test_environment_credentials_and_public_gemini_modes(self):
        route = {'base_url_env': 'HOST', 'api_key_env': 'TOKEN'}
        environment = {'HOST': 'https://example.invalid/v1', 'TOKEN': 'fixture'}
        credentials = resolve_openai_credentials(route, environment)
        self.assertEqual(credentials.headers()['Authorization'], 'Bearer fixture')
        self.assertNotIn('fixture', repr(credentials))
        self.assertNotIn('fixture', json.dumps(credentials.metadata()))
        with self.assertRaises(CredentialSourceError):
            resolve_openai_credentials(route, {})
        with patch.dict(os.environ, {'R3_AV_URL': 'https://example.invalid/model:generateContent',
                                    'GEMINI_API_KEY': 'fixture'}, clear=True):
            self.assertTrue(gemini_endpoint({}).endswith(':generateContent'))
            self.assertEqual(gemini_credentials({})['x-goog-api-key'], 'fixture')
            self.assertEqual(gemini_credentials({'gemini_auth': 'bearer'})['Authorization'], 'Bearer fixture')
            with self.assertRaises(CredentialSourceError):
                gemini_credentials({'gemini_auth': 'other'})

    def test_independent_media_routes_keep_requests_and_replay_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / 'adapter'
            adapter.write_text('#!' + sys.executable + '\n' + """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('kind','model','request','response','output-dir'):
    p.add_argument('--' + name)
p.add_argument('--check', action='store_true')
args=p.parse_args()
if not args.check:
    job=Path(args.output_dir)
    (job/'called-once').touch(exist_ok=False)
    req=json.loads(Path(args.request).read_text())
    value={'status':'completed', 'ok':True, 'model':req['provider_model']}
    (job/'reply.tmp').write_text(json.dumps(value))
    (job/'reply.tmp').replace(args.response)
""")
            adapter.chmod(0o755)
            media = Media.__new__(Media)
            media.store = Store(root / 'story')
            media.ffmpeg = 'ffmpeg'
            media.config = {'image_provider': {'device': '0'}, 'media_adapters': {}}
            for counter, kind in media_ports.KINDS.items():
                with self.subTest(kind=kind):
                    self.assertIsNone(media_ports.command(media.config, kind))
                    media.config['media_adapters'][kind] = {'executable':str(adapter), 'model':'demo-' + kind}
                    self.assertTrue(media_ports.ready(media.config, kind))
                    folder = root / kind
                    receipt = 'worker_status.json' if kind == 'video' else 'worker_response.json'
                    def invoke():
                        return media._worker(folder, {'seed':42}, [], counter=counter, amount=1, response_name=receipt)
                    self.assertEqual(invoke()['model'], 'demo-'+kind)
                    self.assertEqual(invoke()['ok'], True)
                    self.assertTrue((folder / 'called-once').is_file())
                    record = json.loads((folder / 'request.json').read_text())
                    self.assertEqual(record, {'seed':42, 'provider_model':'demo-'+kind})
                    (folder/receipt).write_text(json.dumps({'status':'completed', 'ok':False, 'needs_reconciliation':True}))
                    with self.assertRaises(Pending):
                        invoke()
