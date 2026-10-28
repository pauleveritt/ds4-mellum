"""Generated schedule and one-shot retention checks, fabricated native outputs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def module():return load('generated_publish',ROOT/'tools/mellum_generated_publish.py')

def fixture(api,root,*,wrong_segment=False,wrong_token=False):
    f=load('spool_fixture',ROOT/'tests/test_mellum_routes_to_trace.py')
    tokens=[1,2,3,4,5,6,7]
    specs=[(0,2,1,0),(2,1,2,1),(3,2,1,2),(5,1,2,3),(6,1,2,3)]
    frames=[f.frame(i,p,rows=n,phase=phase,segment=seg,execution=0,tokens=tokens[p:p+n]) for i,(p,n,phase,seg) in enumerate(specs)]
    if wrong_segment:frames[1]=f.frame(1,2,segment=3,execution=0,tokens=[3])
    if wrong_token:frames[1]=f.frame(1,2,segment=1,execution=0,tokens=[8])
    spool=root/'owned.routes';spool.write_bytes(f.spool(*frames,cap=2))
    ops=root/'operations.txt';ops.write_text('S 0 2\nE 1 3\nS 2 5\nE 3 6\nE 3 7\n')
    return spool,ops,tokens

def test_every_frame_matches_history_operations_and_cap():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);spool,ops,tokens=fixture(api,root)
        result=api.check_schedule(spool,ops,tokens,2)
        assert result['positions']==7 and result['operations']==5
        for bad in ('segment','token','cap','operations','history'):
            spool,ops,tokens=fixture(api,root,wrong_segment=bad=='segment',wrong_token=bad=='token')
            if bad=='operations':ops.write_text('S 0 2\nE 3 3\nS 2 5\nE 3 6\nE 3 7\n')
            if bad=='history':tokens[0]=8
            try:api.check_schedule(spool,ops,tokens,1 if bad=='cap' else 2)
            except ValueError:pass
            else:raise AssertionError(bad+' accepted')

def test_failure_is_reserved_retained_and_cannot_retry():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);prepared=root/'inputs';prepared.mkdir();binary=root/'binary';binary.write_bytes(b'native');model=root/'model';model.write_bytes(b'model')
        def command(argv,**kwargs):
            assert kwargs==dict(timeout=900,limit=2**20)
            (Path(argv[-1])/'partial').write_bytes(b'keep')
            raise api.observer.CommandFailure('cancelled',dict(pid=42,returncode=-9,error_type='KeyboardInterrupt'))
        with patch.object(api,'preflight',return_value=dict(host=dict(cpu='fake'))),patch.object(api,'verify_inputs',return_value={}),patch.object(api.observer,'command',side_effect=command) as native:
            try:api.run(prepared,model,binary,root/'run')
            except ValueError:pass
            else:raise AssertionError('failure accepted')
            assert (root/'run/reservation.json').exists()
            assert (root/'run/raw/partial').read_bytes()==b'keep'
            error=api.common.load_json(root/'run/failure.json')
            assert error['command']['pid']==42 and error['error_type']=='KeyboardInterrupt'
            assert not (root/'run/result').exists()
            try:api.run(prepared,model,binary,root/'run')
            except FileExistsError:pass
            else:raise AssertionError('retry accepted')
            assert native.call_count==1

def test_busy_and_deadline_refuse_before_reservation():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        with patch.object(api,'preflight',side_effect=ValueError('busy')):
            try:api.run(root/'input',root/'model',root/'binary',root/'run')
            except ValueError:pass
            else:raise AssertionError('busy accepted')
        assert not (root/'run').exists()
        with patch.object(api,'utc_now',return_value=api.DEADLINE):
            try:api.preflight(root)
            except ValueError:pass
            else:raise AssertionError('cutoff accepted')

def load_tests(loader,tests,pattern):return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
