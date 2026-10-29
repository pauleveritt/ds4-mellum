"""Observe real native processes using tiny payloads and inherited pipes."""
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'tools/mellum_read_bench'


def test_no_stdout_before_release():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'input'
        path.write_bytes(bytes(range(64)))
        base = [str(BINARY), '--input', str(path), '--plan', str(ROOT/'tests/mellum_read_plan_v1.bin'),
                '--workers', '2', '--mode', 'buffered']
        plain = json.loads(subprocess.check_output(base))
        for abort in (False, True):
            cr, cw = os.pipe(); er, ew = os.pipe()
            child = subprocess.Popen([*base, '--control-fd', str(cr), '--event-fd', str(ew)],
                                     pass_fds=(cr, ew), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.close(cr); os.close(ew)
            try:
                assert select.select([er], [], [], 3)[0], 'READY deadline'
                assert os.read(er, 1) == b'R', child.communicate(timeout=3)
                assert not select.select([child.stdout], [], [], .03)[0]
                os.write(cw, b'G')
                assert select.select([er], [], [], 3)[0]
                assert os.read(er, 1) == b'D'
                assert not select.select([child.stdout], [], [], .03)[0]
                if abort:
                    os.close(cw); cw = -1
                else:
                    os.write(cw, b'L')
                out, err = child.communicate(timeout=3)
                if abort:
                    assert child.returncode != 0 and not out
                else:
                    report = json.loads(out)
                    assert child.returncode == 0, err
                    assert [s['bytes'] for s in report['samples']] == [6, 4]
                    assert report['logical_bytes'] == plain['logical_bytes'] == 10
            finally:
                if child.poll() is None: child.kill(); child.wait()
                if cw >= 0: os.close(cw)
                os.close(er)
        for extra in (['--control-fd', '3'], ['--control-fd', '1', '--event-fd', '2'],
                      ['--control-fd', '3x', '--event-fd', '4'],
                      ['--control-fd', '3', '--control-fd', '4']):
            result = subprocess.run([*base, *extra], capture_output=True)
            assert result.returncode != 0 and not result.stdout


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in globals().items() if k.startswith('test_'))

if __name__ == '__main__': unittest.main()
