"""One-shot fresh-file service execution with actual tiny native children."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    assert path.exists(),f'{name} not implemented'
    spec=importlib.util.spec_from_file_location(name,path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def module():return load('service_run_test',ROOT/'tools/mellum_service_run.py')

def topology():
    registry=[dict(IORegistryEntryID=11,IORegistryEntryChildren=[dict(IORegistryEntryID=12,**{'Preferred Block Size':4096})])]
    return dict(driver_id=11,media_id=12,raw=[dict(argv=['/usr/sbin/ioreg','-a','-l','-r','-c','IOBlockStorageDriver'],
        stdout=plistlib.dumps(registry),stderr=b'',returncode=0)])

@contextmanager
def tiny_run(api):
    fixture=load('service_plan_fixture',ROOT/'tests/test_mellum_service_plan.py')
    with fixture.tiny(api.manifest) as (root,prior):
        prepared=api.manifest.prepare(prior,ROOT/'tools/mellum_read_bench',ROOT/'tools/mellum_stage_bench',root/'service')
        count=0
        def sample(t):
            nonlocal count
            reads=(count//6)*10+(10 if count%6>=3 else 0);count+=1
            start=api.observer.now_ns()
            counters={k:0 for k in api.observer.COUNTERS};counters['Bytes (Read)']=reads
            return dict(driver_id=11,media_id=12,counters=counters,
                command=dict(start_ns=start,end_ns=api.observer.now_ns(),returncode=0,stdout=b'',stderr=b''))
        with patch.object(api.host,'preflight',return_value=dict(processes=[])),\
             patch.object(api.host,'host_metadata',return_value=dict(model='synthetic',ram_bytes=1024)),\
             patch.object(api.observer,'discover',side_effect=lambda files:topology()),\
             patch.object(api.observer,'snapshot',side_effect=sample),\
             patch.object(api.process,'IDLE_SECONDS',.001),\
             patch.object(api,'free_bytes',return_value=64*2**30):
            yield root,prepared

def fail(api,prepared,root):
    try:api.run(prepared,root/'run')
    except BaseException:pass
    else:raise AssertionError('failed service published success')
    assert not (root/'run/result/artifact').exists()
    return api.common.load_json(root/'run/failure/artifact/failure.json')

def test_complete_service_uses_four_fresh_files_and_preserves_evidence():
    api=module()
    with tiny_run(api) as (root,prepared):
        output=api.run(prepared,root/'run');report=api.common.load_json(output/'report.json')
        assert output==root/'run/result/artifact'
        assert report['complete'] and report['closing_verification']=='complete'
        assert report['completed_logical_bytes']==40 and report['counts']['native_source']==40
        assert report['counts']['write']==report['counts']['scratch']==4*32768
        assert not report['physical_io_verified'] and report['prewarm_bytes']==0
        assert len({(p.stat().st_dev,p.stat().st_ino) for p in (root/'run/scratch').iterdir()})==4
        assert len(report['runs'])==4
        for c in report['runs']:
            assert c['classification']['status']=='consistent-volume'
            evidence=api.common.load_json(output/c['evidence'])
            assert len(evidence['snapshots'])==6
            assert c['verification']['sha256']==c['expected']['sha256']
            assert (output/c['staging']).exists() and (output/c['stdout']).exists()
        assert (prepared.parent/'execution-reservation.json').exists()
        try:api.run(prepared,root/'run')
        except FileExistsError:pass
        else:raise AssertionError('output overwritten')
        try:api.run(prepared,root/'repeat')
        except ValueError:pass
        else:raise AssertionError('reservation reused')

def test_calibration_and_later_failures_stop_without_replacement():
    api=module()
    for status in ('invalid','background-inconclusive','little-device-traffic','mixed-inconclusive'):
        with tiny_run(api) as (root,prepared),patch.object(api.classifier,'classify',return_value=dict(status=status)):
            report=fail(api,prepared,root)
            assert report['started_cells']==[0] and report['unstarted_cells']==[1,2,3]
            assert len(list((root/'run/scratch').iterdir()))==1
            assert report['closing_verification']=='complete'
    with tiny_run(api) as (root,prepared):
        real=api.process.observe;count=0
        def observe(argv,t):
            nonlocal count
            count+=1
            if count==2:raise api.process.ObservationFailure('cancelled',dict(error_type='KeyboardInterrupt',stdout=b'',stderr=b'',returncode=-9))
            return real(argv,t)
        with patch.object(api.process,'observe',side_effect=observe):report=fail(api,prepared,root)
        assert report['error_type']=='KeyboardInterrupt' and report['closing_verification']=='incomplete'
        assert report['started_cells']==[0,1] and report['completed_logical_bytes']==10
        assert (root/'run/failure/artifact/1.failed.json').exists()
        assert report['unknown_reserved_maxima'] and (prepared.parent/'execution-reservation.json').exists()

def test_preflight_budget_or_mutation_keeps_scratch_and_reservation():
    api=module()
    for mode in ('space','geometry','missing-geometry','stage','receipt','source','wire','scratch','verify','budget','cancel-stage','closing'):
        with tiny_run(api) as (root,prepared):
            real=api.observer.command;verify=api.manifest.base.pack_tools.verify_packed;calls=0
            def command(argv,**kwargs):
                raw=real(argv,**kwargs)
                if mode=='stage':raw['returncode']=1
                if mode=='receipt':raw['stdout']=b'{}'
                if mode=='cancel-stage':raise api.observer.CommandFailure('cancelled',raw) from KeyboardInterrupt()
                if mode in ('source','wire','scratch'):
                    path={'source':root/'packed/experts.bin','wire':prepared/'0-scratch.bin','scratch':root/'run/scratch/0.bin'}[mode]
                    with path.open('r+b') as f:f.write(b'X')
                return raw
            def closing(*args):
                nonlocal calls
                calls+=1
                if calls==2:
                    wire=prepared/'0-source.bin';wire.write_bytes(wire.read_bytes()+b'x')
                return verify(*args)
            with patch.object(api.observer,'command',side_effect=command),\
                 patch.object(api,'free_bytes',return_value=0 if mode=='space' else 64*2**30):
                if mode in ('geometry','missing-geometry'):
                    with patch.object(api,'geometry',side_effect=ValueError('unsupported geometry')):report=fail(api,prepared,root)
                elif mode=='verify':
                    with patch.object(api.verifier,'verify_scratch',side_effect=ValueError('byte mismatch')):report=fail(api,prepared,root)
                elif mode=='budget':
                    with patch.object(api.verifier,'expected',side_effect=ValueError('budget exhausted')):report=fail(api,prepared,root)
                elif mode=='closing':
                    with patch.object(api.manifest.base.pack_tools,'verify_packed',side_effect=closing):report=fail(api,prepared,root)
                else:report=fail(api,prepared,root)
            if mode in ('space','geometry','missing-geometry'):
                assert report['started_cells']==[] and not (prepared.parent/'execution-reservation.json').exists()
            else:
                assert (prepared.parent/'execution-reservation.json').exists()
                assert report['closing_verification']=='incomplete'
                if mode not in ('budget',):assert (root/'run/scratch/0.bin').exists()

def test_no_scratch_content_read_before_observed_done():
    api=module()
    with tiny_run(api) as (root,prepared):
        real_open=api.common.opened;real_observe=api.process.observe;done=set()
        def opened(path):
            if path.parent==root/'run/scratch':assert path in done,'scratch read before DONE'
            return real_open(path)
        def observe(argv,t):
            assert argv[argv.index('--mode')+1]=='avoid' and argv[argv.index('--workers')+1]=='1'
            with patch.object(Path,'open',side_effect=AssertionError('Python file I/O inside controls')):
                evidence=real_observe(argv,t)
            assert [e['token'] for e in evidence['events']]==['R','G','D','L']
            done.add(Path(argv[argv.index('--input')+1]));return evidence
        with patch.object(api.common,'opened',side_effect=opened),patch.object(api.process,'observe',side_effect=observe):
            api.run(prepared,root/'run')

def test_geometry_and_cli_reject_unsupported_hosts():
    api=module();assert api.geometry(topology())==dict(page_bytes=16384,block_bytes=4096)
    for t in (dict(driver_id=11,media_id=12,raw=[]),dict(driver_id=99,media_id=12,raw=topology()['raw'])):
        try:api.geometry(t)
        except ValueError:pass
        else:raise AssertionError('missing geometry accepted')
    cli=ROOT/'tools/mellum_storage.py'
    result=subprocess.run([sys.executable,'-I',str(cli),'service-prepare','--help'],capture_output=True)
    assert result.returncode==0 and b'--stager' in result.stdout
    code=f"import runpy,sys;sys.platform='linux';sys.argv={[str(cli),'service-run','--batch','/missing','--output','/missing']!r};runpy.run_path({str(cli)!r},run_name='__main__')"
    result=subprocess.run([sys.executable,'-I','-c',code],capture_output=True)
    assert result.returncode and b'macOS required' in result.stderr

def test_final_topology_discovery_mutation_never_publishes_success():
    api=module()
    for kind in ('source','wire','scratch'):
        with tiny_run(api) as (root,prepared):
            count=0
            def discover(files):
                nonlocal count
                count+=1
                if count==6:
                    path={'source':root/'packed/experts.bin','wire':prepared/'0-source.bin','scratch':root/'run/scratch/0.bin'}[kind]
                    with path.open('r+b') as stream:stream.write(b'X')
                return topology()
            with patch.object(api.observer,'discover',side_effect=discover):report=fail(api,prepared,root)
            assert count==6 and report['complete'] is False


def test_interrupted_staging_child_retains_partial_file_pid_and_no_go():
    api=module()
    with tiny_run(api) as (root,prepared):
        real=api.observer.command;select=api.observer.selectors.DefaultSelector.select
        target=root/'run/scratch/0.bin'
        def interrupted(selector,*args):
            if target.exists():raise KeyboardInterrupt()
            return select(selector,*args)
        def stage(argv,**kwargs):
            assert kwargs==dict(timeout=120,limit=2**20)
            code="import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_bytes(b'partial');print('staging',flush=True);time.sleep(5)"
            with patch.object(api.observer.selectors.DefaultSelector,'select',interrupted):
                return real([sys.executable,'-c',code,str(target)],**kwargs)
        with patch.object(api.observer,'command',side_effect=stage),patch.object(api.process,'observe',side_effect=AssertionError('reader launched')):
            report=fail(api,prepared,root)
        command=report['failed_command']
        assert command.get('pid') and command['returncode'] is not None
        assert command.get('error_type')==report['error_type']=='KeyboardInterrupt'
        assert target.read_bytes()==b'partial' and report['unknown_reserved_maxima']
        assert report['started_cells']==[0] and report['runs']==[]
        assert (root/'run/failure/artifact/0.staging.json').exists()
        try:os.kill(command['pid'],0)
        except ProcessLookupError:pass
        else:raise AssertionError('staging child not reaped')


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
