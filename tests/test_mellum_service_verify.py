"""Source-derived hashes and one-pass post-read scratch verification."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'tools/mellum_service_verify.py';assert path.exists(),'service verification not implemented'
    spec=importlib.util.spec_from_file_location('service_verify_test',path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);return api

def fixture(api,root):
    source=root/'source';source.write_bytes(b'abcdWXYZ')
    binding=dict(path=str(source),**api.common.file_identity(source))
    jobs=[dict(job_id=0,expert_id=0,ranges=[dict(offset=0,bytes=4,dest=0)]),
          dict(job_id=1,expert_id=1,ranges=[dict(offset=4,bytes=4,dest=0)])]
    cell=dict(source_jobs=jobs,stride_bytes=16384,file_bytes=32768,logical_bytes=8)
    packed=dict(experts=[dict(packed_sha256=hashlib.sha256(v).hexdigest()) for v in (b'abcd',b'WXYZ')])
    return binding,cell,packed

def refused(call):
    try:call()
    except (ValueError,OSError):return
    raise AssertionError('invalid scratch accepted')

def test_expected_reads_only_source_and_matches_packed_digests():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);payload,cell,packed=fixture(api,root)
        before=api.common.READ_BYTES
        value=api.expected(payload,cell,packed)
        assert api.common.READ_BYTES-before==8
        exact=b'abcd'+bytes(16380)+b'WXYZ'+bytes(16380)
        assert value['sha256']==hashlib.sha256(exact).hexdigest()
        assert len(value['experts'])==2
        packed['experts'][0]['packed_sha256']='0'*64
        refused(lambda:api.expected(payload,cell,packed))
        with api.common.read_budget(3):refused(lambda:api.expected(payload,cell,packed))

def test_post_read_verifier_rejects_bytes_padding_and_identity_changes():
    api=module()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);payload,cell,packed=fixture(api,root);expected=api.expected(payload,cell,packed)
        scratch=root/'scratch';exact=b'abcd'+bytes(16380)+b'WXYZ'+bytes(16380)
        scratch.write_bytes(exact)
        binding=dict(path=str(scratch),**api.common.identity.file_stat(scratch))
        before=api.common.READ_BYTES
        result=api.verify_scratch(payload,binding,cell,expected)
        assert result['sha256']==expected['sha256']
        assert api.common.READ_BYTES-before==32776
        for offset in (0,100):
            altered=bytearray(exact);altered[offset]^=1;scratch.write_bytes(altered)
            binding=dict(path=str(scratch),**api.common.identity.file_stat(scratch))
            refused(lambda:api.verify_scratch(payload,binding,cell,expected))
        scratch.write_bytes(exact);binding=dict(path=str(scratch),**api.common.identity.file_stat(scratch))
        scratch.unlink();scratch.write_bytes(exact)
        refused(lambda:api.verify_scratch(payload,binding,cell,expected))

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))
if __name__=='__main__':unittest.main()
