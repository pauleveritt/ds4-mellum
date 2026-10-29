"""Calibration preserves prefix cache state and the fixed native wire contract."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import runpy
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module():
    name = 'mellum_read_plan'
    if name not in sys.modules:
        path = ROOT / 'tools' / f'{name}.py'
        assert path.is_file(), 'read plan is not implemented'
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
    return sys.modules[name]


def refuse(call):
    try:
        call()
    except (ValueError, OSError):
        return
    raise AssertionError('invalid read plan accepted')


def tiny_trace(api):
    r = api.replay
    geometry = r.Geometry(layers=1, experts_per_layer=3, top_k=1, expert_bytes=10)
    header = r.Header(schema_version=1, capture_kind='synthetic', model_id='tiny', engine_revision='test',
                      numerical_mode='release', workload_id='unit', geometry=geometry)
    requests = tuple(r.Request(forward=i, segment='initial', phase='prefill' if i == 0 else 'decode', layer=0,
                              positions=(i,), experts=((expert,),), line=i+2) for i,expert in enumerate((0,0,1,0,1)))
    return r.Trace(header=header, requests=requests)


def test_seeded_plan_is_unique_and_frozen_as_explicit_ids():
    api = module()
    ids = api.uniform_ids()
    assert len(ids) == len(set(ids)) == 512
    assert ids[:8] == [(4,57),(1,38),(25,55),(4,9),(6,6),(19,9),(6,0),(13,4)]
    assert all(0 <= layer < 28 and 0 <= expert < 64 for layer,expert in ids)


def test_lru_replays_prefill_before_decode_and_keeps_repeated_misses():
    api = module()
    trace = tiny_trace(api)
    assert api.lru_decode_misses(trace, 10) == [(0,1),(0,0),(0,1)]
    assert api.lru_decode_misses(trace, 20) == [(0,1)]
    refuse(lambda: api.lru_decode_misses(trace, 0))


def test_wrong_trace_digest_split_or_short_decode_refused():
    api = module()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'trace'
        path.write_bytes(b'wrong trace')
        refuse(lambda:api.diagnostic_ids(path))
    trace = tiny_trace(api)
    refuse(lambda: api.select_diagnostic(trace))
    header = replace(trace.header, schema_version=2, capture_kind='measured', geometry=api.replay.MELLUM,
                     capture_manifest={'workload':{'split':'evaluation'}})
    refuse(lambda: api.select_diagnostic(replace(trace,header=header)))
    header = replace(header, capture_manifest={'workload':{'split':'tuning'}})
    refuse(lambda: api.select_diagnostic(replace(trace,header=header)))


def test_layouts_expand_same_jobs_to_three_or_one_ranges():
    api = module()
    fixture = runpy.run_path(str(ROOT / 'tests/test_mellum_expert_index.py'))
    index = fixture['fabricated_index']()
    ids = [(0,0),(27,63),(0,0)]
    jobs = api.make_jobs(index,ids,'original')
    assert [job['expert_id'] for job in jobs] == [0,1791,0]
    assert [r['dest'] for r in jobs[0]['ranges']] == [0,2193408,4386816]
    digest = 'a' * 64
    packed = dict(schema_version=1,kind='packed-experts',geometry=index['geometry'],
                  payload=dict(path='experts.bin',bytes=11791761408,sha256=digest),
                  index={'sha256':digest},model={'sha256':digest},
                  experts=[dict(layer=i//64,expert=i%64,ranges=[dict(role='packed',offset=i*6580224,bytes=6580224)],
                                source_sha256=digest,packed_sha256=digest) for i in range(1792)])
    other = api.make_jobs(packed,ids,'packed')
    assert [job['expert_id'] for job in other] == [0,1791,0]
    assert other[1]['ranges'] == [dict(offset=11785181184,bytes=6580224,dest=0)]
    refuse(lambda:api.make_jobs(index,ids,'unknown'))


def test_wire_literal_golden_bytes_and_all_bounds():
    api = module()
    jobs = [dict(job_id=0,expert_id=0,ranges=[dict(offset=o,bytes=2,dest=d) for o,d in ((0,0),(8,2),(16,4))]),
            dict(job_id=1,expert_id=1791,ranges=[dict(offset=32,bytes=4,dest=0)])]
    golden = (ROOT / 'tests/mellum_read_plan_v1.bin').read_bytes()
    assert len(golden) == 192
    assert api.encode_plan(jobs,64) == golden
    for key,bad in (('job_id',2),('expert_id',1792),('expert_id',True)):
        altered = deepcopy(jobs)
        altered[0][key] = bad
        refuse(lambda:api.encode_plan(altered,64))
    for key,bad in (('offset',63),('offset',2**63),('bytes',0),('bytes',6580225),('dest',1)):
        altered = deepcopy(jobs)
        altered[0]['ranges'][0][key] = bad
        refuse(lambda:api.encode_plan(altered,64))
    refuse(lambda:api.encode_plan([],64))
    refuse(lambda:api.encode_plan(jobs,2**63))
    altered = deepcopy(jobs)
    altered[0]['ranges'] *= 2
    refuse(lambda:api.encode_plan(altered,64))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
