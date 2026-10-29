"""Native CLI validates identities and emits only complete successful reports."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'tools/mellum_read_bench'
FAULT = ROOT / 'tests/mellum_read_bench_fault'


def test_cli_rejects_changed_size_identity_or_bad_mode():
    assert BINARY.is_file(), 'native storage CLI is not implemented'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'input'
        path.write_bytes(bytes(range(64)))
        base = [str(BINARY),'--input',str(path),'--plan',str(ROOT/'tests/mellum_read_plan_v1.bin')]
        for workers in (1,2,4):
            result = subprocess.run([*base,'--workers',str(workers),'--mode','buffered'],capture_output=True)
            assert result.returncode == 0, result.stderr
            report = json.loads(result.stdout)
            assert report['schema_version'] == 1
            assert report['jobs'] == 2 and report['logical_bytes'] == 10
            assert report['summary']['prewarm_bytes'] == 10
            assert report['summary']['buffer_bytes'] == workers * 6
            assert report['physical_io_verified'] is False
            assert [sample['job_id'] for sample in report['samples']] == [0,1]
            assert [sample['bytes'] for sample in report['samples']] == [6,4]
            assert [sample['syscalls'] for sample in report['samples']] == [3,1]
            assert all(sample['enqueue_ns'] <= sample['start_ns'] <= sample['done_ns'] <= report['summary']['end_ns']
                       for sample in report['samples'])
        for extra in (['--workers','3','--mode','buffered'],['--workers','1','--mode','cold']):
            result = subprocess.run([*base,*extra],capture_output=True)
            assert result.returncode != 0 and not result.stdout
        path.write_bytes(b'short')
        result = subprocess.run([*base,'--workers','1','--mode','buffered'],capture_output=True)
        assert result.returncode != 0 and not result.stdout
        path.unlink()
        path.symlink_to(ROOT/'tests/mellum_read_plan_v1.bin')
        result = subprocess.run([*base,'--workers','1','--mode','buffered'],capture_output=True)
        assert result.returncode != 0 and not result.stdout


def test_no_success_report_on_late_failure():
    assert FAULT.is_file(), 'fault-boundary native CLI is not implemented'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'input'
        path.write_bytes(bytes(range(64)))
        argv = [str(FAULT),'--input',str(path),'--plan',str(ROOT/'tests/mellum_read_plan_v1.bin'),
                '--workers','2','--mode','buffered']
        for key in ('MELLUM_TEST_CLOSE_FAIL','MELLUM_TEST_STAT_FAIL'):
            result = subprocess.run(argv,env=dict(os.environ,**{key:'1'}),capture_output=True)
            assert result.returncode != 0 and not result.stdout


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items() if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
