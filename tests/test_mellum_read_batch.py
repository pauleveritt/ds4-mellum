"""Freeze order, verify real tiny native reports, and retain failed batches."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'tools/mellum_read_bench'


def module():
    name='mellum_read_batch'
    if name not in sys.modules:
        path=ROOT/'tools'/f'{name}.py'
        assert path.is_file(), 'read batch is not implemented'
        spec=importlib.util.spec_from_file_location(name,path)
        result=importlib.util.module_from_spec(spec)
        sys.modules[name]=result
        spec.loader.exec_module(result)
    return sys.modules[name]


def refuse(call):
    try:
        call()
    except (ValueError,OSError,subprocess.CalledProcessError):
        return
    raise AssertionError('invalid batch/report accepted')


@contextmanager
def tiny_batch(api):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root=Path(directory)
        model=root/'model'; model.write_bytes(bytes(range(64)))
        packed=root/'packed'; packed.mkdir()
        payload=packed/'experts.bin'; payload.write_bytes(bytes([0,1,8,9,16,17,32,33,34,35]))
        original=[dict(layer=0,expert=0,ranges=[dict(role=r,offset=o,bytes=2) for r,o in zip(('gate','up','down'),(0,8,16))]),
                  dict(layer=0,expert=1,ranges=[dict(role='gate',offset=32,bytes=4)])]
        index=dict(schema_version=1,geometry=api.index_tools.GEOMETRY,experts=original,
                   source=dict(path=str(model),**api.identity.file_identity(model)))
        index_path=root/'index.json'; api.common.write_json(index_path,index)
        manifest=dict(schema_version=1,kind='packed-experts',geometry=index['geometry'],model=index['source'],
                      index=dict(path=str(index_path),**api.identity.file_identity(index_path)),
                      payload=dict(path='experts.bin',bytes=10,sha256=api.identity.file_identity(payload)['sha256']),
                      experts=[dict(layer=0,expert=i,ranges=[dict(role='packed',offset=o,bytes=b)]) for i,o,b in ((0,0,6),(1,6,4))])
        api.common.write_json(packed/'manifest.json',manifest)
        trace=root/'trace'; trace.write_bytes(b'tiny diagnostic prerequisite')
        # Inject only full-Mellum geometry/trace/source verification prerequisites.
        for owner,name in ((api.index_tools,'verify_index_source'),(api.pack_tools,'verify_packed')):
            stack.enter_context(patch.object(owner,name))
        for owner,name in ((api.index_tools,'validate_index'),(api.pack_tools,'validate_packed')):
            stack.enter_context(patch.object(owner,name,side_effect=lambda value:value))
        stack.enter_context(patch.object(api.plans,'uniform_ids',return_value=[(0,0),(0,1)]))
        stack.enter_context(patch.object(api.plans,'diagnostic_ids',return_value=[(0,1),(0,0)]))
        batch=api.prepare_batch(index_path,packed,trace,BINARY,root/'prepared')
        yield root,batch


def test_frozen_order_has_72_runs_and_exact_byte_budget():
    api=module()
    runs=api.frozen_runs()
    assert len(runs)==72
    assert runs[:4]==[dict(plan='uniform',mode='buffered',workers=1,repetition=0,layout='original'),
                     dict(plan='uniform',mode='buffered',workers=1,repetition=0,layout='packed'),
                     dict(plan='uniform',mode='buffered',workers=1,repetition=1,layout='packed'),
                     dict(plan='uniform',mode='buffered',workers=1,repetition=1,layout='original')]
    assert len(runs)*512*6580224==242573377536 < 256*2**30
    assert runs[-1]==dict(plan='diagnostic',mode='avoid',workers=4,repetition=2,layout='packed')


def test_prepare_binds_both_layouts_and_existing_trace_digest():
    api=module()
    with tiny_batch(api) as (root,batch):
        value=api.common.load_json(batch/'manifest.json')
        assert len(value['runs'])==72 and value['timed_logical_bytes']==720
        assert set(value['plans'])=={'uniform-original','uniform-packed','diagnostic-original','diagnostic-packed'}
        assert value['inputs']['trace']['sha256']==api.identity.file_identity(root/'trace')['sha256']
        assert value['physical_io_verified'] is False
        assert value['untimed_python_read_bytes'] > 0
        assert value['diagnostic']['approved_trace_sha256']==api.plans.TRACE_SHA256
        api.verify_batch(batch)


def test_batch_rejects_input_plan_binary_or_source_mutation():
    api=module()
    for target in ('model','index.json','trace','packed/experts.bin','packed/manifest.json','plan','source','binary'):
        with tiny_batch(api) as (root,batch):
            value=api.common.load_json(batch/'manifest.json')
            if target=='source':
                value['source_identity']['source_state_id']='0'*64
                (batch/'manifest.json').unlink();api.common.write_json(batch/'manifest.json',value)
            elif target=='binary':
                value['inputs']['binary']['sha256']='0'*64
                (batch/'manifest.json').unlink();api.common.write_json(batch/'manifest.json',value)
            else:
                path=batch/next(iter(value['plans'].values()))['wire'] if target=='plan' else root/target
                with path.open('ab') as stream: stream.write(b'mutation')
            refuse(lambda:api.verify_batch(batch))


def test_batch_cannot_substitute_an_unindexed_model_path():
    api=module()
    with tiny_batch(api) as (root,batch):
        other=root/'substitute'
        other.write_bytes(b'x'*64)
        value=api.common.load_json(batch/'manifest.json')
        value['inputs']['model']=api.binding(other)
        (batch/'manifest.json').unlink();api.common.write_json(batch/'manifest.json',value)
        refuse(lambda:api.verify_batch(batch))


def native_fixture(api, root):
    path=root/'input';path.write_bytes(bytes(range(64)))
    process=subprocess.run([str(BINARY),'--input',str(path),'--plan',str(ROOT/'tests/mellum_read_plan_v1.bin'),
                            '--workers','2','--mode','buffered'],check=True,capture_output=True)
    report=json.loads(process.stdout)
    jobs=[dict(job_id=0,expert_id=0,ranges=[dict(offset=o,bytes=2,dest=d) for o,d in ((0,0),(8,2),(16,4))]),
          dict(job_id=1,expert_id=1791,ranges=[dict(offset=32,bytes=4,dest=0)])]
    run=dict(mode='buffered',workers=2)
    return report,jobs,run,api.common.identity.file_stat(path)


def test_reject_duplicate_missing_or_inconsistent_native_samples():
    api=module()
    with tempfile.TemporaryDirectory() as directory:
        report,jobs,run,stat=native_fixture(api,Path(directory))
        checked=api.validate_native(report,jobs,run,stat)
        assert checked['logical_bytes']==10
        for mutation in ('duplicate','missing','bytes','time','enqueue','controls','syscalls'):
            altered=deepcopy(report)
            if mutation=='duplicate':altered['samples'][1]=altered['samples'][0]
            elif mutation=='missing':altered['samples'].pop()
            elif mutation=='bytes':altered['samples'][0]['bytes']=5
            elif mutation=='time':altered['samples'][0]['done_ns']=0
            elif mutation=='enqueue':altered['samples'][0]['enqueue_ns']+=1
            elif mutation=='controls':altered['cache_controls']['nocache']=True
            else:altered['samples'][0]['syscalls']=0
            refuse(lambda:api.validate_native(altered,jobs,run,stat))


def test_nonfinite_duplicate_keys_and_bool_counts_refused():
    api=module()
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        path=root/'bad.json'
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
            path.write_bytes(raw);refuse(lambda:api.common.load_json(path))
        report,jobs,run,stat=native_fixture(api,root)
        report['jobs']=True
        refuse(lambda:api.validate_native(report,jobs,run,stat))


def test_partial_process_output_or_close_error_never_publishes():
    api=module()
    for failure in ('partial','close'):
        with tiny_batch(api) as (root,batch):
            real=api.subprocess.run
            def fail_native(argv,**kwargs):
                if str(BINARY)==argv[0]:
                    if failure=='close':
                        return real([str(ROOT/'tests/mellum_read_bench_fault'),*argv[1:]],
                                    **dict(kwargs,env={'MELLUM_TEST_CLOSE_FAIL':'1'}))
                    kwargs['stdout'].write(b'{"truncated":')
                    return subprocess.CompletedProcess(argv,1)
                return real(argv,**kwargs)
            with patch.object(api.subprocess,'run',side_effect=fail_native):
                refuse(lambda:api.run_batch(batch,root/'run'))
            assert not (root/'run/artifact').exists()
            assert (root/'run.failure/artifact/failure.json').is_file()
            assert list((root/'run.failure/artifact/raw').glob('*.stdout'))
            refuse(lambda:api.run_batch(batch,root/'retry'))


def test_distributions_have_known_nearest_rank_values():
    api=module()
    assert api.distribution([1,2,3,4,5,6,7,8,9,10])==dict(p50=5,p95=10,p99=10,max=10)
    refuse(lambda:api.distribution([]))


def test_flags_cannot_claim_physical_ssd_reads():
    api=module()
    with tiny_batch(api) as (root,batch):
        # Real native I/O, including macOS flags, through the complete 72 tiny runs.
        if sys.platform!='darwin':
            return
        artifact=api.run_batch(batch,root/'run')
        report=api.common.load_json(artifact/'report.json')
        assert report['physical_io_verified'] is False
        assert report['untimed_python_read_bytes'] > 0
        assert report['completed_logical_bytes']==720 and len(report['runs'])==72
        assert len(list((artifact/'raw').glob('*.stdout')))==72
        assert all('service_ms' in run['metrics'] for run in report['runs'])


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items() if name.startswith('test_'))


if __name__=='__main__':
    unittest.main()
