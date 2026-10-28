"""Whole-prefix frozen windows over tiny causal traces, without model/GPU."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_windows_keep_prefill_cache_and_explicit_short_empty_overlap():
    api=load('generated_analysis',ROOT/'tools/mellum_generated_analysis.py')
    fixture=load('replay_fixture',ROOT/'tests/test_mellum_ssd_replay.py')
    records=[fixture.header(layers=1,experts=4,top_k=1)]
    for i,(phase,segment,expert) in enumerate([('prefill','initial-prefill',1),('decode','response0',1),('decode','response0',2),('prefill','append-prefill',2),('decode','response1',2)]):
        records+=fixture.forward(i,[[[expert]]],phase=phase,segment=segment)
    result=api.analyze(fixture.load(records),capacity=10)
    assert result['whole']['totals']['demand_bytes']==20
    windows=result['windows']
    assert windows['response0-first32']['demand_bytes']==10
    assert windows['response0-first32']['positions']==[1,2]
    assert windows['response0-after32']['tokens']==0
    assert windows['response0-after32']['demand_distribution'] is None
    assert windows['response1-first32']['demand_bytes']==0
    assert windows['response0-tail256']['shortened'] is True
    assert windows['response0-tail256']['overlaps_other_windows'] is True
    assert windows['response0-tail256']['demand_bytes']==10
    assert windows['initial-prefill']['hit_bytes']==0
    assert windows['response1-first32']['hit_bytes']==10
    try:api.analyze(fixture.load([fixture.header(layers=1),*fixture.forward(0,[[[0,1]]],phase='prefill',segment='initial-prefill')]),capacity=10)
    except ValueError:pass
    else:raise AssertionError('oversized atomic group accepted')

def load_tests(loader,tests,pattern):return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
