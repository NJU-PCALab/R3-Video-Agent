"""Host-local CPU import check; a probe from another host is insufficient."""
import json,subprocess,os

def audio_imports(config):
    audio=config['audio_provider']
    source=json.dumps(audio['music_source'])
    code=('import sys,torch,soundfile;from qwen_tts import Qwen3TTSModel;'
          'sys.path.insert(0,'+source+');from acestep.handler import AceStepHandler;'
          'print("LITE_HOST_AUDIO_IMPORTS_OK")')
    compute=config.get('runtime_import_probe')=='host_runtime_compute_v2'
    if compute:
        # The exact audio interpreter must execute cuDNN, not just import torch.
        # Use the production GPU lock and an isolated child environment.
        code += (';import fcntl;lock=open('+repr(audio['lock_file'])+',"a+");'
                 'fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);'
                 'torch.backends.cudnn.enabled=True;'
                 'y=torch.nn.functional.conv1d(torch.randn(2,64,1024,device="cuda:0"),'
                 'torch.randn(64,64,3,device="cuda:0"));torch.cuda.synchronize();'
                 'print("LITE_HOST_AUDIO_COMPUTE_OK")')
    try:
        kwargs={'env':{**os.environ,'CUDA_VISIBLE_DEVICES':str(audio['device'])}} if compute else {}
        result=subprocess.run([audio['python'],'-B','-c',code],capture_output=True,text=True,timeout=60,**kwargs)
        return (result.returncode==0 and 'LITE_HOST_AUDIO_IMPORTS_OK' in result.stdout.splitlines()
                and (not compute or 'LITE_HOST_AUDIO_COMPUTE_OK' in result.stdout.splitlines()))
    except (OSError,subprocess.TimeoutExpired):
        return False
