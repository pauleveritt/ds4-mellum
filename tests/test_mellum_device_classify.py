"""Literal boundary witnesses for device traffic and observation intervals."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'tools/mellum_device_classify.py';assert path.exists(),'classifier not implemented'
    spec=importlib.util.spec_from_file_location('classify_test',path)
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def sample(start,end,reads):
    return dict(driver_id=11,media_id=12,command=dict(start_ns=start,end_ns=end,returncode=0),
        counters={'Bytes (Read)':reads,'Bytes (Write)':0,'Operations (Read)':0,'Operations (Write)':0,
                  'Errors (Read)':0,'Retries (Read)':0})

def case(d,background=0):
    return dict(logical_bytes=1000,opening=sample(1030,1040,0),closing=sample(1060,1070,d),
        idle_before=(sample(0,10,0),sample(1010,1020,0)),
        idle_after=(sample(1080,1090,d),sample(2090,2100,d+background)),native_start_ns=1041,native_end_ns=1050)

def test_counter_reset_replacement_errors_and_retries_invalid():
    api=module()
    for key,value in (('Bytes (Read)',-1),('Bytes (Read)',True),('Errors (Read)',1),('Retries (Read)',1)):
        args=case(900);args['closing']['counters'][key]=value
        assert api.classify(**args)['status']=='invalid'
    for key in ('driver_id','media_id'):
        args=case(900);args['closing'][key]=99
        assert api.classify(**args)['status']=='invalid'
    args=case(900);args['idle_before'][1]['counters']['Bytes (Read)']=100
    assert api.classify(**args)['status']=='invalid'

def test_native_time_must_fit_counter_bracket():
    api=module()
    for key,value in (('native_start_ns',1039),('native_end_ns',1061)):
        args=case(900);args[key]=value
        assert api.classify(**args)['status']=='invalid'

def test_nonpositive_idle_inner_time_invalid():
    args=case(900);args['idle_after'][1]['command']['start_ns']=1090
    assert module().classify(**args)['status']=='invalid'

def test_diagnostic_bins_at_exact_boundaries():
    api=module()
    for d,want in ((50,'little-device-traffic'),(51,'mixed-inconclusive'),(899,'mixed-inconclusive'),
                   (900,'consistent-volume'),(1250,'consistent-volume'),(1251,'background-inconclusive')):
        result=api.classify(**case(d))
        assert result['status']==want,(d,result)
        assert result['device_read_bytes']==d and result['outer_ns']==40
        assert result['physical_io_verified'] is False
    assert api.classify(**case(900,250))['status']=='consistent-volume'
    assert api.classify(**case(900,251))['status']=='background-inconclusive'
    args=case(2**63);args['logical_bytes']=2**63
    assert api.classify(**args)['status']=='consistent-volume'

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
