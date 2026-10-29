"""Frozen generated input protocols over tiny source bundles, no model/GPU."""
import hashlib
import importlib.util
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'tools/mellum_generated_inputs.py';assert path.exists(),'generated inputs missing'
    spec=importlib.util.spec_from_file_location('generated_inputs_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Fake-boundary tests run inside the frozen window, whatever the wall clock says.
    from datetime import timedelta
    m.utc_now=lambda:m.DEADLINE-timedelta(hours=1);return m

def bundle(api,root):
    files={}
    for name in api.FILES:
        data=(name+' exact\n').encode() if name not in ('templates/.gitkeep','tests/.gitkeep') else b''
        path=root/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    source=dict(commit='c'*40,tree='d'*40)
    content=dict(revision=source['commit'],tree=source['tree'],files={k:v['sha256'] for k,v in files.items()})
    value=dict(schema_version=1,kind='workload-source',repository_id='ac-demo',split='evaluation',source=source,files=files,content_id=hashlib.sha256(api.common.canonical(content)).hexdigest())
    api.common.write_json(root/'manifest.json',value);return value

def test_exact_prompt_hashes_all_files_without_lock_body_or_padding():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);v=bundle(api,root)
        with patch.object(api,'CONTENT_ID',v['content_id']),patch.object(api,'REVISION',v['source']['commit']):messages=api.messages(root)
        assert set(messages)=={'system','user','followup'}
        text=messages['user'].decode()
        assert 'Phase 1 only' in text and 'uv.lock exact' not in text
        assert v['files']['uv.lock']['sha256'] in text
        for name in api.FILES:
            if name!='uv.lock':assert (root/'source'/name).read_text() in text
        (root/'source/AGENTS.md').write_bytes(b'changed')
        with patch.object(api,'CONTENT_ID',v['content_id']),patch.object(api,'REVISION',v['source']['commit']):
            try:api.messages(root)
            except ValueError:pass
            else:raise AssertionError('source changed')

def test_prepared_inputs_bind_policy_and_preserve_native_failure():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);b=root/'bundle';b.mkdir();v=bundle(api,b)
        model=root/'model';model.write_bytes(b'model');binary=root/'binary';binary.write_bytes(b'fake')
        def native(argv,**kwargs):
            out=Path(argv[-1]);assert kwargs['timeout']==120
            for name in api.TOKEN_FILES:(out/name).write_text('1 2\n')
            for name in api.RENDERED_FILES:(out/name).write_bytes(b'exact rendered')
            return dict(argv=argv,pid=1,returncode=0,stdout=b'',stderr=b'')
        with patch.object(api,'CONTENT_ID',v['content_id']),patch.object(api,'REVISION',v['source']['commit']),patch.object(api,'MODEL_BYTES',5),patch.object(api,'MODEL_STAT',api.identity.file_stat(model),create=True),patch.object(api.observer,'command',side_effect=native):
            prepared=api.prepare(b,model,binary,root/'prepared');value=api.verify(prepared)
            assert value['split']=='evaluation' and value['settings']['caps']==[4096,1024]
            assert value['policy']['sha256']==api.common.file_identity(ROOT/'tools/mellum_ssd_replay.py')['sha256']
            (prepared/'initial.tokens').write_text('8\n')
            try:api.verify(prepared)
            except ValueError:pass
            else:raise AssertionError('tokens changed')
            with patch.object(api.observer,'command',side_effect=api.observer.CommandFailure('cancelled',dict(pid=42,returncode=-9))):
                try:api.prepare(b,model,binary,root/'failed')
                except ValueError:pass
                else:raise AssertionError('native failed')
            assert (root/'failed/raw/system.txt').exists() and (root/'failed/failure.json').exists()
            try:api.prepare(b,model,binary,root/'prepared')
            except FileExistsError:pass
            else:raise AssertionError('root reused')

def test_prepare_rejects_inherited_controls_before_creating_output():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        with patch.dict('os.environ',{'DS4_QWEN4_PREFILL_CHUNK':'1'}):
            try:api.prepare(root/'bundle',root/'model',root/'binary',root/'output')
            except ValueError as exc:assert 'inherited' in str(exc)
            else:raise AssertionError('inherited control accepted')
        assert not (root/'output').exists()


def test_forced_build_freezes_both_binaries_and_preparation_preserves_receipt():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);cpu=root/'cpu';gpu=root/'gpu'
        for p in (cpu,gpu):p.write_bytes(b'old')
        def build(argv,**kwargs):
            assert '-B' in argv and kwargs['timeout']==120
            cpu.write_bytes(b'new cpu');gpu.write_bytes(b'new gpu')
            return dict(pid=12,returncode=0,stdout=b'forced build',stderr=b'')
        with patch.object(api.observer,'command',side_effect=build):
            receipt=api.freeze_build(root/'build',cpu,gpu)
        value=api.common.load_json(receipt)
        assert value['forced_rebuild'] is True
        assert value['gpu']==api.bindings.binding(gpu)
        assert value['cpu']==api.bindings.binding(cpu)
        assert value['source_identity']==api.common.source_identity(ROOT)


def test_same_size_model_substitution_refuses_before_native_inspect():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);model=root/'model';model.write_bytes(b'model')
        binary=root/'binary';binary.write_bytes(b'fake');bundle_path=root/'bundle';bundle_path.mkdir();v=bundle(api,bundle_path)
        with patch.object(api,'MODEL_BYTES',5),patch.object(api,'CONTENT_ID',v['content_id']),patch.object(api,'REVISION',v['source']['commit']),patch.object(api.observer,'command',side_effect=AssertionError('unverified model reached native inspect')) as native:
            try:api.prepare(bundle_path,model,binary,root/'inputs')
            except ValueError as exc:assert 'model' in str(exc)
            else:raise AssertionError('unverified same-size model accepted')
            assert native.call_count==0


def test_cpu_inspect_and_build_commands_must_fit_frozen_cutoff():
    api=module()
    from datetime import timedelta
    from contextlib import ExitStack
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);b=root/'bundle';b.mkdir();v=bundle(api,b);model=root/'model';model.write_bytes(b'model');cpu=root/'cpu';gpu=root/'gpu'
        for p in (cpu,gpu):p.write_bytes(b'fake')
        with ExitStack() as stack:
            stack.enter_context(patch.object(api,'CONTENT_ID',v['content_id']));stack.enter_context(patch.object(api,'REVISION',v['source']['commit']))
            stack.enter_context(patch.object(api,'MODEL_BYTES',5));stack.enter_context(patch.object(api,'MODEL_STAT',api.identity.file_stat(model)))
            stack.enter_context(patch.object(api,'utc_now',return_value=api.DEADLINE-timedelta(seconds=30),create=True))
            native=stack.enter_context(patch.object(api.observer,'command',side_effect=AssertionError('native command crossed cutoff')))
            for action in (lambda:api.prepare(b,model,cpu,root/'input'),lambda:api.freeze_build(root/'build',cpu,gpu)):
                try:action()
                except ValueError as exc:assert 'cutoff' in str(exc)
                else:raise AssertionError('deadline allowed native command')
            assert native.call_count==0


def test_owner_window_of_2026_10_28_ends_2200z_for_cpu_and_gpu():
    # The owner reopened the window to 18:00 Eastern; both guards share it.
    api=module()
    from datetime import datetime, timezone
    spec=importlib.util.spec_from_file_location('generated_publish_window',ROOT/'tools/mellum_generated_publish.py')
    publish=importlib.util.module_from_spec(spec);spec.loader.exec_module(publish)
    cutoff=datetime(2026,10,28,22,tzinfo=timezone.utc)
    assert api.DEADLINE==cutoff and api.SETTINGS['deadline_utc']=='2026-10-28T22:00:00Z'
    assert publish.DEADLINE==api.DEADLINE
    with patch.object(api,'utc_now',return_value=datetime(2026,10,28,15,30,tzinfo=timezone.utc)):api.require_time(120)
    with patch.object(api,'utc_now',return_value=datetime(2026,10,28,21,58,1,tzinfo=timezone.utc)):
        try:api.require_time(120)
        except ValueError as exc:assert 'cutoff' in str(exc)
        else:raise AssertionError('CPU command crossed reopened cutoff')


def isolated(function):
    def run():
        # Make/app inject control variables; fake-boundary tests start clean.
        env={k:v for k,v in os.environ.items() if not k.startswith(('GIT_','DS4_','MELLUM_TRACE_')) and k not in ('MAKEFLAGS','MFLAGS','MAKEFILES','MAKEOVERRIDES')}
        with patch.dict(os.environ,env,clear=True):function()
    run.__name__=function.__name__
    return run


def load_tests(loader,tests,pattern):return unittest.TestSuite(unittest.FunctionTestCase(isolated(v),description=k) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
