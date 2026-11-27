"""Opt-in cross-host rate reservation; intervals and transports stay unchanged."""
from contextlib import contextmanager
from pathlib import Path
import os,socket,time
from .storage import Pending,read,write
from .worker_wait import process_identity

def read_reservation(path,check_stop=lambda:None,attempts=51):
    """An existing shared rate slot must never become an empty/default slot.

    A shared filesystem may briefly hide a replaced name from another client. The caller
    owns the directory lock; retry only the read, then stop before any POST.
    Providers are initialized conservatively by deployment, never here.
    """
    for attempt in range(attempts):
        check_stop()
        try:
            value=read(path,required=True,attempts=1)
            deadline=value['next_allowed_at']
            if isinstance(deadline,bool) or not isinstance(deadline,(int,float)) or not __import__('math').isfinite(deadline):
                raise ValueError('Invalid reservation deadline')
            return value
        except FileNotFoundError:
            if attempt+1==attempts:
                raise Pending('Shared rate state unavailable; original request not submitted: '+str(path),status='awaiting_shared_rate_state') from None
            time.sleep(.2)
        except (KeyError,TypeError,ValueError):
            raise Pending('Invalid retained shared rate state; original slot requires reconciliation: '+str(path),
                status='awaiting_shared_rate_state') from None

@contextmanager
def directory_lock(path,check_stop=lambda:None,timeout=60):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    while True:
        check_stop()
        try:path.mkdir();break
        except FileExistsError:
            if time.monotonic()-started>=timeout:
                raise Pending('Original cross-host lock owner requires reconciliation: '+str(path),status='awaiting_shared_lock')
            time.sleep(.2)
    try:
        write(path/'owner.json',{'host':socket.gethostname(),'process':process_identity(os.getpid()),'time':time.time()})
        yield
    finally:
        (path/'owner.json').unlink(missing_ok=True)
        path.rmdir()
