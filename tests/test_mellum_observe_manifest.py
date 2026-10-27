"""Independent manifest/reservation over real tiny uniform range artifacts."""
from contextlib import contextmanager, ExitStack
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def module():
    path=ROOT/'tools/mellum_observe_manifest.py';assert path.exists(),'observation manifest not implemented'
    return load('observe_manifest_test',path)

@contextmanager
def tiny(api):
    old=load('old_batch_tests',ROOT/'tests/test_mellum_read_batch.py').module()
    with load('fixture_tests',ROOT/'tests/test_mellum_read_batch.py').tiny_batch(old) as (root,prior):
        # Shared sibling loader keeps geometry/trace boundaries patched by tiny_batch.
        yield root,prior

def refused(call):
    try:call()
    except (ValueError,OSError):return
    raise AssertionError('invalid manifest/reservation accepted')

def test_new_manifest_has_four_uniform_cells_and_exact_caps():
    api=module()
    with tiny(api) as (root,prior):
        before=(prior/'manifest.json').read_bytes()
        output=api.prepare(prior,ROOT/'tools/mellum_read_bench',root/'observation')
        value,jobs=api.verify(output,payloads=False)
        assert [(r['layout'],r['mode'],r['workers']) for r in value['runs']]==[
            ('original','buffered',1),('original','avoid',1),('packed','buffered',1),('packed','avoid',1)]
        assert value['timed_logical_bytes']==40 and value['prewarm_bytes']==20
        assert value['caps']==dict(timed=16*2**30,prewarm=8*2**30,python=128*2**30)
        assert (prior/'manifest.json').read_bytes()==before
        assert not (prior.parent/'execution-reservation.json').exists()
        assert set(jobs)=={'original','packed'} and value['physical_io_verified'] is False

def test_escaping_plan_or_forged_jobs_refused():
    api=module()
    with tiny(api) as (root,prior):
        value=api.common.load_json(prior/'manifest.json')
        value['plans']['uniform-original']['wire']='../escape'
        (prior/'manifest.json').unlink();api.common.write_json(prior/'manifest.json',value)
        refused(lambda:api.prepare(prior,ROOT/'tools/mellum_read_bench',root/'observation'))

def test_forecast_covers_pre_and_post_checks_before_payload_read():
    api=module()
    with tiny(api) as (root,prior):
        output=api.prepare(prior,ROOT/'tools/mellum_read_bench',root/'observation')
        value=api.common.load_json(output/'manifest.json')
        forecast=api.forecast(value)
        assert forecast['payload_base_bytes']==188  # 2 * (64 + 3*10)
        assert forecast['total_python_upper_bytes'] < 128*2**30
        api.reserve(output,api.batches.binding(output/'manifest.json'))
        refused(lambda:api.reserve(output,api.batches.binding(output/'manifest.json')))
        value['inputs']['model']['size']=128*2**30
        refused(lambda:api.forecast(value))
    source=io.BytesIO(b'0123456789')
    with api.common.read_budget(5):
        refused(lambda:api.common.read(source,10))
    assert source.tell()==0

def test_source_reads_participate_in_strict_budget():
    api=module()
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'file';path.write_bytes(b'x'*65536)
        with api.common.read_budget(65535):
            refused(lambda:api.common.file_identity(path))

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
