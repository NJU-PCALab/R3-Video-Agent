import json
from pathlib import Path
import tempfile
import unittest

from lite import audio_readiness as ar
from tools.probe_audio_runtime import probe_run, probe_specs
from lite import asset_lifecycle, task_parameters


class AudioReadinessTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        def put(name, data='sample'):
            p = self.root / name
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_text(data)
            return str(p)
        for path in ('tools/probe_audio_runtime.py', 'tools/audio_worker.py', 'lite/audio_parameters.py',
                     'lite/audio_assets.py', 'python', 'adapter', 'tools/speech_worker.py', 'asr/model.safetensors'):
            put(path)
        self.cfg = {'lite_root': str(self.root), 'media_bin': str(self.root),
                    'speech_verification': {'asr_python': str(self.root/'python'),'asr_model_path': str(self.root/'asr')},
                    'audio_provider': {'device':'0','python':str(self.root/'python'),
                                       'music_source':str(self.root),'models':{}}}
        for mode in ar.MODES:
            put(mode + '/model.safetensors')
            put('model_manifests/' + mode + '.json', '{"revision":"demo"}')
            self.cfg['audio_provider']['models'][mode] = str(self.root/mode)

    def record(self):
        return {'runtime_ready': True, 'audio_fingerprint': ar.fingerprint(self.cfg),
                'runtime_evidence': {m:{'waveform_sha256':'1'*64, 'original_sha256':'2'*64,
                    'worker_receipt_sha256':'3'*64,'seconds':2.0} for m in ar.MODES}}

    def test_new_weights_at_same_path_require_probe_but_text_routing_does_not(self):
        record = self.record()
        self.cfg['text_route'] = {'model':'another-planner'}
        self.assertTrue(ar.matches(self.cfg, record))
        (self.root/'voice_clone/model.safetensors').write_text('new weights')
        self.assertFalse(ar.matches(self.cfg, record))
        record = self.record()
        (self.root/'model_manifests/music.json').write_text('{"revision":"next"}')
        self.assertFalse(ar.matches(self.cfg, record))

    def test_probe_voices_follow_the_story_language_and_its_calibration_contract(self):
        self.cfg['task_contract'] = {'version': task_parameters.REVISED_VERSION, 'language': 'English'}
        self.cfg['execution_contract'] = asset_lifecycle.VERSION
        record = self.record()
        for language in ('English','Chinese'):
            self.cfg['task_contract']['language'] = language
            for _, spec in probe_specs(self.cfg):
                if spec['role'] == 'voice':
                    self.assertEqual(asset_lifecycle.calibration_spec(spec, self.cfg)['language'], language)
        self.assertFalse(ar.matches(self.cfg, record))

    def test_worker_or_adapter_change_invalidates_the_previous_record(self):
        record = self.record()
        (self.root/'tools/audio_worker.py').write_text('worker2')
        self.assertFalse(ar.matches(self.cfg, record))
        self.cfg['media_adapters'] = {'audio':{'model':'remote-v1','executable':str(self.root/'adapter')}}
        self.assertFalse(ar.matches(self.cfg, self.record() | {'audio_fingerprint':record['audio_fingerprint']}))
        record = self.record()
        self.cfg['media_adapters']['audio']['model']='remote-v2'
        self.assertFalse(ar.matches(self.cfg, record))
        record = self.record()
        (self.root/'adapter').write_text('adapter2')
        self.assertFalse(ar.matches(self.cfg, record))

    def test_boolean_only_and_incomplete_or_invalid_evidence_are_rejected(self):
        self.assertFalse(ar.matches(self.cfg, {'runtime_ready': True}))
        for bad in (None, float('nan'), 0.9):
            record = self.record()
            record['runtime_evidence']['music']['seconds'] = bad
            self.assertFalse(ar.matches(self.cfg, record))

    def test_changed_audio_namespace_never_duplicates_an_unknown_job(self):
        a = probe_run(self.root, 'a')
        pending = a/'media/audio_assets/probe_music'
        pending.mkdir(parents=True)
        (pending/'submitted.json').write_text('{}')
        self.assertEqual(probe_run(self.root, 'a'), a)
        with self.assertRaisesRegex(RuntimeError, 'Reconcile'):
            probe_run(self.root, 'b')
        response=pending/'worker_response.json'
        response.write_text(json.dumps({'ok':False,'needs_reconciliation':True}))
        with self.assertRaisesRegex(RuntimeError, 'Reconcile'):
            probe_run(self.root, 'b')
        response.write_text(json.dumps({'ok':True}))
        self.assertNotEqual(probe_run(self.root, 'b'), a)
        self.assertTrue((pending/'submitted.json').exists())
