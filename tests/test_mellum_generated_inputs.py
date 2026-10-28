"""Frozen generated input protocols over tiny source bundles, no model/GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'tools/mellum_generated_inputs.py';assert path.exists(),'generated inputs missing'
    spec=importlib.util.spec_from_file_location('generated_inputs_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

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
        with patch.object(api,'CONTENT_ID',v['content_id']),patch.object(api,'REVISION',v['source']['commit']),patch.object(api.observer,'command',side_effect=native):
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

def load_tests(loader,tests,pattern):return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
