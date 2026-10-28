"""Generated schedule and one-shot retention checks, fabricated native outputs."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def module():
    m=load('generated_publish',ROOT/'tools/mellum_generated_publish.py')
    # Fake-boundary tests run inside the frozen window, whatever the wall clock says.
    from datetime import timedelta
    m.utc_now=lambda:m.DEADLINE-timedelta(hours=1);return m

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

def validation_fixture(api,root):
    native=root/'native';native.mkdir();prepared=root/'prepared';prepared.mkdir()
    spool,ops,tokens=fixture(api,native)
    (native/'tokens.txt').write_text(' '.join(map(str,tokens)))
    api.common.write_json(native/'result.json',dict(complete=True,parity=True,counts=[1,2],stop_classes=[1,2],host_separator_tokens=1,response0_end=3,response1_start=5,tokens=7))
    (native/'gold.f32').write_bytes(b'\0'*(5*393216))
    (native/'emitted.txt').write_text('1 3 10\n3 6 20\n3 7 30\n')
    for name,words in zip(api.inputs.TOKEN_FILES,('1 2','5','9 4','4')):
        (prepared/name).write_text(words)
        (prepared/name.replace('.tokens','.rendered')).write_bytes(b'rendered')
    recreated=native/'recreated';recreated.mkdir()
    for name in (*api.inputs.TOKEN_FILES,*api.inputs.RENDERED_FILES):(recreated/name).write_bytes((prepared/name).read_bytes())
    receipt=dict(receipt_version=1,context_tokens=8192,cap_tokens=2,capabilities=dict(tensor_tiles_available=True,pre_m5_apple=False,m5_apple=True))
    api.common.write_json(native/'runtime.json',receipt)
    f=load('profile_fixture',ROOT/'tests/test_mellum_capture_publish.py')
    (native/'owned.profiles').write_text('0 '+f.VALID_PROFILE+'\n')
    frozen=dict(source_identity=dict(revision='a'*40,source_state_id='b'*64,files={}),build=dict(gpu=dict(sha256='c'*64),log=dict(path=str(root/'build.log'))),inputs=dict(model={}),policy=dict(sha256='d'*64),source_content_id='e'*64)
    (root/'build.log').write_text('forced build')
    return prepared,native,frozen,dict(host=dict(cpu='M5 Max',model='Mac17,7',ram_bytes=128*2**30))


def test_validated_export_requires_parity_exact_native_inputs_and_complete_tape():
    api=module();validation=api.identity.sibling('mellum_generated_validation')
    with tempfile.TemporaryDirectory() as d:
        prepared,native,frozen,host=validation_fixture(api,Path(d))
        report=validation.validate(prepared,native,frozen,host)
        assert report['complete'] is True and report['analysis']['whole']['policy']=='lru'
        assert report['native']['counts']==[1,2]
        assert (native/'trace.jsonl').exists()
    for bad in ('parity','tokens','tape','emitted','count','stop','rendered','nonfinite'):
        with tempfile.TemporaryDirectory() as d:
            prepared,native,frozen,host=validation_fixture(api,Path(d))
            result=api.common.load_json(native/'result.json')
            if bad=='parity':result['parity']=False
            if bad=='count':result['counts'][0]=2
            if bad=='stop':result['stop_classes'][0]=3
            if bad=='tokens':(native/'tokens.txt').write_text('9 2 3 4 5 6 7')
            if bad=='tape':(native/'gold.f32').write_bytes(b'')
            if bad=='nonfinite':
                import struct
                with (native/'gold.f32').open('r+b') as stream:stream.write(struct.pack('<f',float('nan')))
            if bad=='emitted':(native/'emitted.txt').write_text('1 3 10\n3 7 20\n3 6 30\n')
            if bad=='rendered':(native/'recreated/initial.rendered').write_bytes(b'changed')
            (native/'result.json').unlink();api.common.write_json(native/'result.json',result)
            try:validation.validate(prepared,native,frozen,host)
            except ValueError:pass
            else:raise AssertionError(bad+' accepted')
            assert not (native/'trace.jsonl').exists()


def test_gpu_binary_and_forced_build_receipt_are_bound_before_launch():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);cpu=root/'cpu';gpu=root/'gpu';log=root/'build.log';model=root/'model'
        for path in (cpu,gpu,log,model):path.write_bytes(b'fixture')
        source=dict(revision='a'*40)
        frozen=dict(inputs=dict(model=dict(path=str(model)),binary=api.bindings.binding(cpu)),source_identity=source)
        with patch.object(api.inputs,'verify',return_value=frozen):
            try:api.verify_inputs(root,model,gpu)
            except ValueError:pass
            else:raise AssertionError('missing build accepted')
            frozen['build']=dict(forced_rebuild=True,source_identity=source,cpu=frozen['inputs']['binary'],gpu=api.bindings.binding(gpu),log=api.bindings.binding(log))
            assert api.verify_inputs(root,model,gpu)==frozen
            gpu.write_bytes(b'changed')
            try:api.verify_inputs(root,model,gpu)
            except ValueError:pass
            else:raise AssertionError('binary changed')


def test_closing_identity_gate_runs_after_validation_and_prevents_publication():
    api=module();validation=api.identity.sibling('mellum_generated_validation')
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);binary=root/'binary';model=root/'model'
        def native(argv,**kwargs):return dict(pid=1,returncode=0,stdout=b'',stderr=b'')
        with patch.object(api,'preflight',return_value={}),patch.object(api,'verify_inputs',side_effect=[{'frozen':1},ValueError('changed after validation')]) as verify,patch.object(api.observer,'command',side_effect=native),patch.object(validation,'validate',return_value=dict(complete=True)) as validate:
            try:api.run(root/'inputs',model,binary,root/'run')
            except ValueError as exc:assert 'after validation' in str(exc)
            else:raise AssertionError('closing mutation published')
            assert validate.call_count==1 and verify.call_count==2
            assert (root/'run/failure.json').exists() and not (root/'run/result').exists()


def test_report_digests_remain_valid_after_publication_links_change_ctime():
    api=module();validation=api.identity.sibling('mellum_generated_validation')
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);prepared,native,frozen,host=validation_fixture(api,root)
        report=validation.validate(prepared,native,frozen,host)
        import os
        for name,value in report['files'].items():
            assert set(value)=={'size','sha256'}, 'report binds mutable inode link metadata'
            before=api.common.file_identity(native/name)
            os.link(native/name,root/('linked-'+name.replace('/','-')))
            after=api.common.file_identity(native/name)
            assert value['sha256']==after['sha256'] and value['size']==after['size']


def test_success_publishes_nested_native_recreation_without_losing_raw():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);prepared,native,frozen,host=validation_fixture(api,root)
        def command(argv,**kwargs):
            import shutil
            shutil.copytree(native,Path(argv[-1]),dirs_exist_ok=True)
            return dict(pid=123,returncode=0,stdout=b'',stderr=b'')
        with patch.object(api,'preflight',return_value=host),patch.object(api,'verify_inputs',return_value=frozen),patch.object(api.observer,'command',side_effect=command):
            artifact=api.run(prepared,root/'model',root/'binary',root/'run')
        assert (artifact/'recreated/initial.tokens').read_bytes()==(prepared/'initial.tokens').read_bytes()
        assert (artifact/'trace.jsonl').stat().st_ino==(root/'run/raw/trace.jsonl').stat().st_ino
        report=api.common.load_json(artifact/'report.json')
        assert report['complete'] is True
        for name,value in report['files'].items():
            actual=api.common.file_identity(artifact/name)
            assert all(actual[k]==v for k,v in value.items())


def isolated(function):
    def run():
        # Make/app inject control variables; fake-boundary tests start clean.
        env={k:v for k,v in os.environ.items() if not k.startswith(('GIT_','DS4_','MELLUM_TRACE_')) and k not in ('MAKEFLAGS','MFLAGS','MAKEFILES','MAKEOVERRIDES')}
        with patch.dict(os.environ,env,clear=True):function()
    run.__name__=function.__name__
    return run


def load_tests(loader,tests,pattern):return unittest.TestSuite(unittest.FunctionTestCase(isolated(v),description=k) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
