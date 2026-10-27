"""Run one independent tiny four-cell batch; never use a model or device tools."""
from contextlib import contextmanager
import importlib.util
import json
import subprocess
import sys
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
             patch.object(api,'host_metadata',return_value=dict(model='synthetic',ram_bytes=1024)),\
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
        assert failure['batch_path']==str(prepared)
        assert failure['cumulative_python_read_bytes']==failure['prepare_python_read_bytes']+failure['untimed_python_read_bytes']
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
            raw=bytearray(wire.read_bytes())
            for offset in (68,100,132):raw[offset:offset+4]=(7).to_bytes(4,'little')
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

def test_preflight_rejects_inference_executables():
    api=module()
    for executable in ('ds4-server','ds4-cli','test_mellum_gpu','ds4-bench','ds4-eval','test_mellum_kernels','mellum_trace','mellum_routes_gpu'):
        raw=dict(stdout=f'123 /tmp/{executable} --model model'.encode(),stderr=b'',returncode=0)
        with patch.object(api.observer,'command',return_value=raw):
            try:api.preflight()
            except ValueError:pass
            else:raise AssertionError('active inference ignored')
    raw=dict(stdout=b'123 /tmp/mellum_read_bench --input file',stderr=b'',returncode=0)
    with patch.object(api.observer,'command',return_value=raw):assert api.preflight()['processes']==[]

def test_observe_cli_exposes_commands_and_refuses_nonmac_before_io():
    cli=ROOT/'tools/mellum_storage.py'
    result=subprocess.run([sys.executable,'-I',str(cli),'observe-prepare','--help'],capture_output=True)
    assert result.returncode==0 and b'--prior-batch' in result.stdout
    code=f"import runpy,sys;sys.platform='linux';sys.argv={[str(cli),'observe-run','--batch','/missing','--output','/missing']!r};runpy.run_path({str(cli)!r},run_name='__main__')"
    result=subprocess.run([sys.executable,'-I','-c',code],capture_output=True)
    assert result.returncode!=0 and b'macOS required' in result.stderr

def test_host_identity_retains_cpu_ram_and_raw_command():
    api=module()
    raw=dict(argv=['sysctl'],stdout=b'137438953472\nMac17,7\nApple M5 Max\n',stderr=b'',returncode=0)
    with patch.object(api.observer,'command',return_value=raw):
        value=api.host_metadata()
    assert value['ram_bytes']==137438953472 and value['model']=='Mac17,7' and value['cpu']=='Apple M5 Max'
    assert value['command']['stdout']==raw['stdout']

def test_payload_verification_failure_spends_attempt_before_any_read():
    api=module()
    with tiny_run(api) as (root,prepared):
        count=0
        def fail_after_read(*args):
            nonlocal count
            count+=1
            with api.common.opened(root/'model') as (source,_):source.read(1)
            raise ValueError('verification interrupted after a payload read')
        with patch.object(api.manifest.pack_tools,'verify_packed',side_effect=fail_after_read):
            for name in ('run','repeat'):
                try:api.run(prepared,root/name)
                except ValueError:pass
                else:raise AssertionError('failed payload verification accepted')
        assert count==1, 'payload verification retried against same reservation'
        assert (prepared.parent/'execution-reservation.json').exists()

def test_mutation_inside_final_payload_check_prevents_publication():
    api=module()
    with tiny_run(api) as (root,prepared):
        count=0
        def mutate_final(*args):
            nonlocal count
            count+=1
            if count==2:
                wire=prepared/'original.bin';raw=bytearray(wire.read_bytes());raw[-1]^=1;wire.write_bytes(raw)
        with patch.object(api.manifest.pack_tools,'verify_packed',side_effect=mutate_final):
            try:api.run(prepared,root/'run')
            except ValueError:pass
            else:raise AssertionError('post-verification wire mutation published success')
        assert count==2 and not (root/'run/artifact').exists()

def test_previous_raw_cell_is_released_before_next_capture():
    import weakref
    class Evidence(dict):pass
    api=module();refs=[]
    with tiny_run(api) as (root,prepared):
        real=api.process.observe
        def checked(argv,topology):
            assert all(ref() is None for ref in refs), 'prior raw evidence retained into next capture'
            result=Evidence(real(argv,topology));result['snapshots']=Evidence(result['snapshots'])
            refs.extend((weakref.ref(result),weakref.ref(result['snapshots'])))
            return result
        with patch.object(api.process,'observe',side_effect=checked):api.run(prepared,root/'run')

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
