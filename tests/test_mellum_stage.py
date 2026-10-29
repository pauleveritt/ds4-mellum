"""Tiny actual scratch writes; no model, GPU or device metadata commands."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'tools/mellum_stage_bench'
FAULT=ROOT/'tests/mellum_stage_fault'

def inputs(root):
    source=root/'source';source.write_bytes(bytes(range(64)))
    wire=root/'plan.bin'
    wire.write_bytes(struct.pack('<8sIIIIQQQQQ',b'MLREAD01',1,64,2,2,64,8,4,0,0)+
                     struct.pack('<IIQQQ',0,1,4,4,0)+struct.pack('<IIQQQ',1,7,12,4,0))
    return source,wire

def run(binary,source,wire,out,**env):
    return subprocess.run([str(binary),'--input',str(source),'--plan',str(wire),'--output',str(out)],
                          env=dict(os.environ,**env),capture_output=True)

def test_stage_sorted_slots_match_source_and_zero_padding():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);source,wire=inputs(root);out=root/'scratch'
        result=run(BINARY,source,wire,out);assert result.returncode==0,result.stderr
        receipt=json.loads(result.stdout)
        assert receipt['complete'] is True and receipt['cache_controls']['nocache'] is True
        assert receipt['fsync_ok'] and receipt['close_ok']
        assert receipt['counts']['source_bytes']==8 and receipt['counts']['write_bytes']==32768
        assert receipt['scratch']['ino']==out.stat().st_ino
        assert receipt['source']['ino']==source.stat().st_ino
        assert receipt['plan']['ino']==wire.stat().st_ino
        assert out.stat().st_mode&0o777==0o600
        assert out.read_bytes()==bytes(range(4,8))+bytes(16380)+bytes(range(12,16))+bytes(16380)
        assert source.read_bytes()==bytes(range(64))

def test_stage_faults_retain_partial_output():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);source,wire=inputs(root)
        for fault in ('partial','flag','eof','write-zero','fsync','close','stat'):
            out=root/fault
            result=run(FAULT,source,wire,out,MELLUM_STAGE_FAULT=fault)
            receipt=json.loads(result.stdout)
            if fault=='partial':
                assert result.returncode==0,result.stderr
                assert receipt['counts']['source_syscalls']>2 and receipt['counts']['write_syscalls']>2
            else:
                assert result.returncode!=0 and receipt['complete'] is False
                assert receipt['error'] and out.exists()
                if fault=='write-zero':assert receipt['counts']['write_bytes']==32 and out.stat().st_size==32
                if fault=='flag':assert out.stat().st_size==0

def test_stage_cli_rejects_existing_or_substituted_paths():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);source,wire=inputs(root);out=root/'output';out.write_bytes(b'owned')
        assert run(BINARY,source,wire,out).returncode!=0
        assert out.read_bytes()==b'owned'
        out.unlink();out.symlink_to(source)
        assert run(BINARY,source,wire,out).returncode!=0
        assert source.read_bytes()==bytes(range(64))
        out.unlink();source.unlink();source.symlink_to(wire)
        assert run(BINARY,source,wire,out).returncode!=0 and not out.exists()

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
