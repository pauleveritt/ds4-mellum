"""Pipe-only observation of one owned native child; no artifact I/O."""
import importlib.util
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

_spec=importlib.util.spec_from_file_location('mellum_capture_identity',Path(__file__).with_name('mellum_capture_identity.py'))
identity=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(identity)
observer=identity.sibling('mellum_device_observer')
GATE_SECONDS=30
ACTIVE_SECONDS=120
IDLE_SECONDS=1
STDOUT_LIMIT=8*2**20
STDERR_LIMIT=2**20
MEMORY_LIMIT=256*2**20


class ObservationFailure(ValueError):
    def __init__(self,message: str,evidence: dict):
        super().__init__(message);self.evidence=evidence


def byte_size(value: object) -> int:
    """Count retained Python containers including bytearray capacity, once each."""
    seen=set()
    def size(item):
        if id(item) in seen:return 0
        seen.add(id(item));total=sys.getsizeof(item)
        if isinstance(item,dict):total+=sum(size(k)+size(v) for k,v in item.items())
        elif isinstance(item,(list,tuple)):total+=sum(map(size,item))
        return total
    return size(value)


def observe(argv: list[str],topology: dict) -> dict:
    result=dict(argv=argv,events=[],snapshots={},stdout=b'',stderr=b'',returncode=None)
    cr,cw=os.pipe();er,ew=os.pipe()
    child=None;failure=None
    buffers=dict(stdout=bytearray(),stderr=bytearray(),event=bytearray())
    released=False
    try:
        child=subprocess.Popen([*argv,'--control-fd',str(cr),'--event-fd',str(ew)],
                               pass_fds=(cr,ew),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        result.update(pid=child.pid,argv=child.args)
        os.close(cr);cr=-1;os.close(ew);ew=-1
        with selectors.DefaultSelector() as selector:
            for fd,name in ((child.stdout.fileno(),'stdout'),(child.stderr.fileno(),'stderr'),(er,'event')):
                os.set_blocking(fd,False);selector.register(fd,selectors.EVENT_READ,name)
            def pump(deadline):
                remaining=deadline-time.monotonic()
                if remaining<=0:raise ValueError('owned child deadline exceeded')
                for key,_ in selector.select(min(remaining,.05)):
                    name=key.data;limit={'stdout':STDOUT_LIMIT,'stderr':STDERR_LIMIT,'event':2}[name]
                    available=limit-len(buffers[name])
                    data=os.read(key.fd,min(65536,available+1))
                    if not data:selector.unregister(key.fd);continue
                    buffers[name].extend(data[:available])
                    if len(data)>available:raise ValueError('owned child output exceeds bound')
                    if name=='stdout' and not released:raise ValueError('native report before RELEASE')
            def wait_event(expected,seconds):
                deadline=time.monotonic()+seconds
                while not buffers['event']:
                    pump(deadline)
                    if not buffers['event'] and (child.poll() is not None or er not in selector.get_map()):
                        raise ValueError('owned child exited or event pipe closed')
                if bytes(buffers['event'])!=expected:raise ValueError('invalid or duplicate native event')
                buffers['event'].clear();result['events'].append(dict(token=expected.decode(),time_ns=observer.now_ns()))
            def send(token):
                result['events'].append(dict(token=token.decode(),time_ns=observer.now_ns()))
                if os.write(cw,token)!=1:raise ValueError('short native control write')
            def sample(name):
                # Reserve 128 MiB: 32 MiB capture with bytearray/copy overhead,
                # <=1 MiB / 16384-node XML decoding, buffers and orchestration headroom.
                if byte_size((topology,result,buffers))+128*2**20>MEMORY_LIMIT:
                    raise ValueError('observation capture memory bound exceeded')
                result['snapshots'][name]=observer.snapshot(topology)
            wait_event(b'R',GATE_SECONDS)
            sample('idle_before_0');time.sleep(IDLE_SECONDS);sample('idle_before_1')
            sample('opening');send(b'G');wait_event(b'D',ACTIVE_SECONDS)
            sample('closing');sample('idle_after_0');time.sleep(IDLE_SECONDS);sample('idle_after_1')
            released=True;send(b'L')
            deadline=time.monotonic()+GATE_SECONDS
            while selector.get_map():pump(deadline)
            child.wait(timeout=max(.001,deadline-time.monotonic()))
            if buffers['event'] or child.returncode:raise ValueError('native child failed after observation')
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        failure=exc
        if isinstance(exc,observer.CommandFailure):result['failed_command']=exc.evidence
    finally:
        if child is not None:
            if child.poll() is None:child.kill()
            child.wait();result['returncode']=child.returncode
            child.stdout.close();child.stderr.close()
        for fd in (cr,cw,er,ew):
            if fd>=0:os.close(fd)
        result.update(stdout=bytes(buffers['stdout']),stderr=bytes(buffers['stderr']))
    if failure is not None:raise ObservationFailure(str(failure),result) from failure
    return result
