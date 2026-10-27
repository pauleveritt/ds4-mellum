"""Freeze scratch plans without reading expert payloads."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    assert path.exists(),f'{name} is not implemented'
    spec=importlib.util.spec_from_file_location(name,path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def module():return load('service_plan_test',ROOT/'tools/mellum_service_plan.py')

@contextmanager
def tiny(api):
    old=load('prior_batch_fixture',ROOT/'tests/test_mellum_read_batch.py').module()
    fixture=load('prior_batch_fixture2',ROOT/'tests/test_mellum_read_batch.py')
    with fixture.tiny_batch(old) as (root,prior):
        packed_path=root/'packed/manifest.json';packed=api.common.load_json(packed_path)
        data=(root/'packed/experts.bin').read_bytes()
        for e in packed['experts']:
            r=e['ranges'][0];sha=hashlib.sha256(data[r['offset']:r['offset']+r['bytes']]).hexdigest()
            e.update(source_sha256=sha,packed_sha256=sha)
        packed_path.unlink();api.common.write_json(packed_path,packed)
        batch=api.common.load_json(prior/'manifest.json')
        batch['inputs']['packed_manifest']=api.batches.binding(packed_path)
        (prior/'manifest.json').unlink();api.common.write_json(prior/'manifest.json',batch)
        yield root,prior

def refused(call):
    try:call()
    except (ValueError,OSError):return
    raise AssertionError('invalid service input accepted')

def test_frozen_cells_have_exact_slot_order_and_totals():
    api=module()
    packed=dict(experts=[dict(layer=i//64,expert=i%64,ranges=[dict(offset=i*6580224,bytes=6580224)]) for i in range(1792)])
    with patch.object(api.base.pack_tools,'validate_packed',side_effect=lambda v:v):cells=api.cells(packed)
    assert [len(c['expert_ids']) for c in cells]==[64,512,512,512]
    assert cells[0]['expert_ids'][0]==313 and cells[0]['kind']=='calibration'
    assert [c['repetition'] for c in cells]==[None,0,1,2]
    assert sum(c['logical_bytes'] for c in cells)==10528358400
    assert sum(c['file_bytes'] for c in cells)==10538188800
    for c in cells:
        ids=[j['expert_id'] for j in c['source_jobs']]
        assert ids==sorted(c['expert_ids'])
        assert [j['expert_id'] for j in c['scratch_jobs']]==c['expert_ids']
        assert c['stride_bytes']==6586368 and c['mode']=='avoid' and c['workers']==1
    forecast=api.forecast(dict(inputs=dict(model=dict(size=12925417760),payload=dict(size=11791761408)),cells=cells))
    assert forecast['total_python_upper_bytes']==136920461888 <137438953472

def test_prepare_and_verify_are_metadata_only():
    api=module()
    with tiny(api) as (root,prior):
        before=(prior/'manifest.json').read_bytes()
        real=api.common.opened
        def guard(path):
            assert path not in (root/'model',root/'packed/experts.bin'),'payload read during metadata checks'
            return real(path)
        with patch.object(api.common,'opened',side_effect=guard):
            prepared=api.prepare(prior,ROOT/'tools/mellum_read_bench',ROOT/'tools/mellum_stage_bench',root/'service-prepared')
            value,cells=api.verify(prepared)
        assert before==(prior/'manifest.json').read_bytes()
        assert sum(c['logical_bytes'] for c in cells)==40
        assert value['physical_io_verified'] is False and value['prewarm_bytes']==0
        api.reserve(prepared,api.batches.binding(prepared/'manifest.json'))
        refused(lambda:api.reserve(prepared,api.batches.binding(prepared/'manifest.json')))
        wire=prepared/'0-source.bin';wire.write_bytes(wire.read_bytes()+b'x')
        refused(lambda:api.verify(prepared))

def test_changed_manifest_or_forecast_refuses():
    api=module()
    for field in ('caps','source','escape','binary'):
        with tiny(api) as (root,prior):
            prepared=api.prepare(prior,ROOT/'tools/mellum_read_bench',ROOT/'tools/mellum_stage_bench',root/'service-prepared')
            value=api.common.load_json(prepared/'manifest.json')
            if field=='caps':value['caps']['python']+=1
            if field=='source':value['source_identity']['source_state_id']='0'*64
            if field=='escape':value['plans']['0-source']['path']='../escape'
            if field=='binary':value['inputs']['stager']['sha256']='0'*64
            (prepared/'manifest.json').unlink();api.common.write_json(prepared/'manifest.json',value)
            refused(lambda:api.verify(prepared))
            value['inputs']['model']['size']=128*2**30
            refused(lambda:api.forecast(value))

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
