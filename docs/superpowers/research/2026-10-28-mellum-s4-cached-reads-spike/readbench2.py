"""Spike: after one F_NOCACHE pass over a write-only fresh file, does madvise(DONTNEED) or msync(MS_INVALIDATE)
on a mapping make the second pass device-true?  Device read bytes per pass."""
import ctypes, fcntl, importlib.util, json, mmap, os, sys, time
from pathlib import Path
REPO=Path('/Users/pauleveritt/.codex/worktrees/mellum-ssd-replay/ds4-mellum/tools')
def load(n):
    s=importlib.util.spec_from_file_location(n,REPO/f'{n}.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
obs=load('mellum_device_observer'); cls=load('mellum_device_classify')
D=Path(sys.argv[1]); SIZE=2<<30; SLICE=2_193_408; F_NOCACHE=48; F_RDAHEAD=45
libc=ctypes.CDLL(None, use_errno=True)
libc.mmap.restype=ctypes.c_void_p; libc.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
def write_fresh(path):
    out=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); fcntl.fcntl(out,F_NOCACHE,1)
    for _ in range(SIZE//(64<<20)): os.write(out,os.urandom(64<<20))
    os.fsync(out); os.close(out)
def read_pass(path,chunk,start):
    fd=os.open(path,os.O_RDONLY); fcntl.fcntl(fd,F_NOCACHE,1); fcntl.fcntl(fd,F_RDAHEAD,0); n=0; off=start
    while off+chunk<=SIZE: n+=len(os.pread(fd,chunk,off)); off+=chunk
    os.close(fd); return n
def drop(path,how):
    fd=os.open(path,os.O_RDONLY)
    addr=libc.mmap(None,SIZE,1,1,fd,0)  # PROT_READ, MAP_SHARED
    rc=libc.posix_madvise(ctypes.c_void_p(addr),ctypes.c_size_t(SIZE),4) if how=='dontneed' else libc.msync(ctypes.c_void_p(addr),ctypes.c_size_t(SIZE),2)  # POSIX_MADV_DONTNEED=4, MS_INVALIDATE=2
    libc.munmap(ctypes.c_void_p(addr),ctypes.c_size_t(SIZE)); os.close(fd); return rc
out={}
for how in ('dontneed','invalidate'):
    for name,chunk,start in (('aligned',4<<20,0),('unaligned',SLICE,SLICE//2)):
        path=D/f'{how}-{name}.bin'; write_fresh(path)
        f=dict(path=str(path),dev=os.stat(path).st_dev); topo=obs.discover({'model':f,'payload':dict(f)})
        res=[]
        for p in range(2):
            a=obs.snapshot(topo); n=read_pass(path,chunk,start); b=obs.snapshot(topo)
            res.append(round(cls.delta(a,b)['Bytes (Read)']/n,3))
            if p==0: res.append(('drop rc',drop(path,how)))
        out[f'{how}/{name}']=res; path.unlink()
print(json.dumps(out))
