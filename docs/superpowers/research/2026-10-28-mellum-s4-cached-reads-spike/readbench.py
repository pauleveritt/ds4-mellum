"""Spike: do unaligned F_NOCACHE preads leave pages in the unified buffer cache? Device read bytes per pass."""
import fcntl, importlib.util, json, os, sys, time
from pathlib import Path
REPO=Path('/Users/pauleveritt/.codex/worktrees/mellum-ssd-replay/ds4-mellum/tools')
def load(n):
    s=importlib.util.spec_from_file_location(n,REPO/f'{n}.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
obs=load('mellum_device_observer'); cls=load('mellum_device_classify'); meas=load('mellum_s4_measure')
D=Path(sys.argv[1]); VERIFY=sys.argv[2]=='verify'; SIZE=2<<30; SLICE=2_193_408
def make(name):
    src=D/'src.bin'
    if not src.exists():
        with open(src,'wb') as f:
            for i in range(SIZE//(64<<20)): f.write(os.urandom(64<<20))
    dst=D/name
    if VERIFY:
        meas.fresh_copy(src,dst); return dst
    # write-only: F_NOCACHE writes, fsync, never read back
    out=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); fcntl.fcntl(out,meas.F_NOCACHE,1)
    with open(src,'rb',buffering=0) as f:
        fcntl.fcntl(f.fileno(),meas.F_NOCACHE,1)
        while chunk:=f.read(64<<20): os.write(out,chunk)
    os.fsync(out); os.close(out); return dst
def device(topo,fn):
    a=obs.snapshot(topo); t=time.time(); n=fn(); dt=time.time()-t; b=obs.snapshot(topo)
    return dict(logical=n,device=cls.delta(a,b)['Bytes (Read)'],s=round(dt,2))
def reader(path,chunk,start):
    def go():
        fd=os.open(path,os.O_RDONLY); fcntl.fcntl(fd,meas.F_NOCACHE,1); fcntl.fcntl(fd,45,0)  # F_RDAHEAD 0
        n=0; off=start
        while off+chunk<=SIZE:
            n+=len(os.pread(fd,chunk,off)); off+=chunk
        os.close(fd); return n
    return go
src=D/'src.bin'
out={}
for name,chunk,start in (('aligned',4<<20,0),('unaligned',SLICE,SLICE//2)):
    path=make(name+'.bin'); f=dict(path=str(path),dev=os.stat(path).st_dev); topo=obs.discover({'model':f,'payload':dict(f)})
    out[name]=[device(topo,reader(path,chunk,start)) for _ in range(2)]
    path.unlink()
src.unlink()
print(json.dumps(out,indent=1))
