"""Observe actual tiny native reads with fabricated counter samples."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'tools/mellum_observe_process.py';assert path.exists(),'owned observer not implemented'
    spec=importlib.util.spec_from_file_location('process_test',path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def sampler(api):
    def sample(topology):
        start=api.observer.now_ns()
        return dict(driver_id=11,media_id=12,counters={k:0 for k in api.observer.COUNTERS},
            command=dict(start_ns=start,end_ns=api.observer.now_ns(),returncode=0,stdout=b'',stderr=b''))
    return sample

def test_ready_idle_open_go_done_close_idle_release_order():
    api=module()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'input';path.write_bytes(bytes(range(64)))
        argv=[str(ROOT/'tools/mellum_read_bench'),'--input',str(path),
            '--plan',str(ROOT/'tests/mellum_read_plan_v1.bin'),'--workers','1','--mode','buffered']
        with patch.object(api.observer,'snapshot',side_effect=sampler(api)),patch.object(api,'IDLE_SECONDS',.002):
            # All Python writes are forbidden during observe; native stdout is pipe-only.
            with patch.object(Path,'open',side_effect=AssertionError('artifact write during observation')):
                value=api.observe(argv,{})
        assert value['returncode']==0 and [e['token'] for e in value['events']]==['R','G','D','L']
        assert list(value['snapshots'])==['idle_before_0','idle_before_1','opening','closing','idle_after_0','idle_after_1']
        report=json.loads(value['stdout']);assert report['logical_bytes']==10
        s=value['snapshots']
        assert s['opening']['command']['end_ns'] <= report['summary']['start_ns']
        assert report['summary']['end_ns'] <= s['closing']['command']['start_ns']
        assert s['idle_before_1']['command']['start_ns']-s['idle_before_0']['command']['end_ns'] >= 2_000_000
        batch=api.identity.sibling('mellum_read_batch')
        jobs=[dict(job_id=0,expert_id=0,ranges=[dict(offset=0,bytes=2,dest=0),dict(offset=8,bytes=2,dest=2),dict(offset=16,bytes=2,dest=4)]),
              dict(job_id=1,expert_id=1791,ranges=[dict(offset=32,bytes=4,dest=0)])]
        metrics=batch.validate_native(report,jobs,dict(workers=1,mode='buffered'),api.identity.file_stat(path),
            api.identity.file_stat(ROOT/'tests/mellum_read_plan_v1.bin'))
        assert metrics['logical_bytes']==10

def test_child_exit_or_pipe_eof_keeps_partial_evidence():
    api=module()
    for code in ('pass', "import time;time.sleep(2)", "import os;os.write(1,b'x'*10000)"):
        with patch.object(api,'GATE_SECONDS',.03),patch.object(api,'STDOUT_LIMIT',100):
            try:api.observe([sys.executable,'-c',code],{})
            except api.ObservationFailure as exc:
                assert exc.evidence['returncode'] is not None
                assert len(exc.evidence['stdout'])<=100
                try:os.kill(exc.evidence['pid'],0)
                except ProcessLookupError:pass
                else:raise AssertionError('owned child not reaped')
            else:raise AssertionError('failed child accepted')

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
