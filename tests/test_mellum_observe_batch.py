"""Run one independent tiny four-cell batch; never use a model or device tools."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def module():
    path=ROOT/'tools/mellum_observe_batch.py';assert path.exists(),'observed batch not implemented'
    return load('observe_batch_test',path)

@contextmanager
def tiny_run(api):
    fixture=load('manifest_tests',ROOT/'tests/test_mellum_observe_manifest.py')
    process_tests=load('process_tests',ROOT/'tests/test_mellum_observe_process.py')
    with fixture.tiny(api.manifest) as (root,prior):
        prepared=api.manifest.prepare(prior,ROOT/'tools/mellum_read_bench',root/'observed')
        with patch.object(api,'preflight',return_value=dict(processes=[])),\
             patch.object(api.observer,'discover',return_value=dict(driver_id=11,media_id=12)),\
             patch.object(api.observer,'snapshot',side_effect=process_tests.sampler(api.process)),\
             patch.object(api.process,'IDLE_SECONDS',.001):
            yield root,prepared

def test_four_cells_preserve_all_counters_reports_and_failures():
    api=module()
    with tiny_run(api) as (root,prepared):
        output=api.run(prepared,root/'run')
        value=api.common.load_json(output/'report.json')
        assert value['completed_logical_bytes']==40 and value['prewarm_bytes']==20
        assert value['physical_io_verified'] is False and len(value['runs'])==4
        for cell in value['runs']:
            assert cell['classification']['status']=='little-device-traffic'
            raw=api.common.load_json(output/cell['evidence'])
            assert len(raw['snapshots'])==6 and raw['returncode']==0
        assert (prepared.parent/'execution-reservation.json').exists()
        try:api.run(prepared,root/'repeat')
        except (ValueError,OSError):pass
        else:raise AssertionError('spent reservation reused')

def test_budget_or_active_process_refuses_before_go():
    api=module()
    with tiny_run(api) as (root,prepared),patch.object(api,'preflight',side_effect=ValueError('busy')):
        try:api.run(prepared,root/'run')
        except ValueError:pass
        else:raise AssertionError('busy process ignored')
        failure=api.common.load_json(root/'run.failure/artifact/failure.json')
        assert failure['completed_logical_bytes']==0 and failure['started_cells']==[]
        assert not (prepared.parent/'execution-reservation.json').exists()

def test_failure_never_retries_or_publishes_success():
    api=module()
    with tiny_run(api) as (root,prepared):
        real=api.process.observe;count=0
        def fail_second(argv,topology):
            nonlocal count
            count+=1
            if count==2:raise api.process.ObservationFailure('injected failure',dict(stdout=b'',stderr=b'',returncode=1))
            return real(argv,topology)
        with patch.object(api.process,'observe',side_effect=fail_second):
            try:api.run(prepared,root/'run')
            except ValueError:pass
            else:raise AssertionError('failed batch published')
        value=api.common.load_json(root/'run.failure/artifact/failure.json')
        assert count==2 and value['completed_logical_bytes']==10
        assert value['started_cells']==[0,1] and value['unstarted_cells']==[2,3]
        assert value['failed_cell_progress']=='unknown; reserved maxima retained'
        assert not (root/'run/artifact').exists()

def test_native_wire_swap_cannot_pass_restored_hash():
    api=module()
    with tiny_run(api) as (root,prepared):
        real=api.process.observe
        def swap(argv,topology):
            wire=Path(argv[argv.index('--plan')+1]);saved=wire.with_suffix('.saved')
            raw=bytearray(wire.read_bytes());raw[68:72]=(7).to_bytes(4,'little')
            wire.rename(saved);wire.write_bytes(raw)
            try:return real(argv,topology)
            finally:wire.unlink();saved.rename(wire)
        with patch.object(api.process,'observe',side_effect=swap):
            try:api.run(prepared,root/'run')
            except ValueError as exc:assert 'consumed a different plan' in str(exc),str(exc)
            else:raise AssertionError('substituted native wire accepted')

def test_input_or_topology_changes_prevent_success():
    api=module()
    with tiny_run(api) as (root,prepared):
        with patch.object(api.observer,'discover',side_effect=[dict(driver_id=11,media_id=12),dict(driver_id=99,media_id=12)]):
            try:api.run(prepared,root/'run')
            except ValueError:pass
            else:raise AssertionError('changed topology accepted')
        assert not (root/'run/artifact').exists()

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
