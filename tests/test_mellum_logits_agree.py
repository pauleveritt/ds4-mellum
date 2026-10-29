"""The agreement report between two traced Mellum runs (tools/mellum_logits_agree.py), on hand-made trace files,
and the llama.cpp logits driver's argument checks (tools/mellum_llama_logits, skipped when it is not built).

    python3 -I tests/test_mellum_logits_agree.py    (part of `make test-mellum`)
"""

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools' / 'mellum_logits_agree.py'
DRIVER = ROOT / 'tools' / 'mellum_llama_logits'
POSITIONS = (0, 5, 9)
BASE = [2.0, 1.0, 0.5, -1.0, 0.0, 0.25, -0.5, 1.5]


def write_rows(prefix, rows):
    """rows maps a position to a list of logits."""
    for pos, row in rows.items():
        with open('%s.p%d.logits.f32' % (prefix, pos), 'wb') as f:
            f.write(struct.pack('<%df' % len(row), *row))


def run(a_rows, b_rows, positions=POSITIONS):
    with tempfile.TemporaryDirectory() as d:
        write_rows(os.path.join(d, 'a'), a_rows)
        write_rows(os.path.join(d, 'b'), b_rows)
        return subprocess.run(
            [sys.executable, '-I', str(SCRIPT), os.path.join(d, 'a'), os.path.join(d, 'b'),
             ','.join(str(p) for p in positions)],
            capture_output=True, text=True)


def base_rows():
    return {p: list(BASE) for p in POSITIONS}


def test_identical_runs_agree_everywhere():
    result = run(base_rows(), base_rows())
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['n'] == 3 and report['top1_equal'] == 3 and report['agree'] == 3
    assert report['agree_rate'] == 1.0 and report['mean_kl'] == 0.0 and report['worst_kl'] == 0.0


def test_kl_is_kl_of_b_from_a():
    a = {0: [0.0, 0.0]}
    b = {0: [math.log(3.0), 0.0]}   # p_B = (0.75, 0.25), p_A = (0.5, 0.5)
    report = json.loads(run(a, b, positions=(0,)).stdout)
    want = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert abs(report['positions'][0]['kl'] - want) < 1e-6, report


def test_a_swapped_top1_with_a_wide_margin_disagrees():
    a = base_rows()
    a[5] = list(BASE)
    a[5][0], a[5][3] = a[5][3], a[5][0]   # A's top-1 moves far below; B's margin is 0.5 nats
    report = json.loads(run(a, base_rows()).stdout)
    row = [r for r in report['positions'] if r['position'] == 5][0]
    assert not row['top1_equal'] and row['near_tie'], 'a 0.5-nat margin is a near-tie'
    b = base_rows()
    b[5] = [5.0] + BASE[1:]                # B's top two now 3.5 nats apart
    a2 = base_rows()
    a2[5] = [-5.0] + BASE[1:]
    report = json.loads(run(a2, b).stdout)
    row = [r for r in report['positions'] if r['position'] == 5][0]
    assert not row['top1_equal'] and not row['near_tie']
    assert report['top1_equal'] == 2 and report['agree'] == 2


def test_a_swap_inside_the_near_tie_margin_agrees():
    b = base_rows()
    b[9] = [2.0, 1.9] + BASE[2:]           # B's top two 0.1 nats apart
    a = base_rows()
    a[9] = [1.9, 2.0] + BASE[2:]
    report = json.loads(run(a, b).stdout)
    row = [r for r in report['positions'] if r['position'] == 9][0]
    assert not row['top1_equal'] and row['near_tie']
    assert report['top1_equal'] == 2 and report['agree'] == 3


def test_non_finite_logits_are_refused():
    a = base_rows()
    a[0][2] = float('nan')
    result = run(a, base_rows())
    assert result.returncode == 1 and 'finite' in result.stderr, (result.returncode, result.stderr)
    assert not result.stdout


def test_driver_refuses_bad_usage():
    if not DRIVER.is_file():
        print('skip: tools/mellum_llama_logits not built')
        return
    none = subprocess.run([str(DRIVER)], capture_output=True, text=True)
    assert none.returncode == 2 and 'usage' in none.stderr.lower(), none.stderr
    with tempfile.TemporaryDirectory() as d:
        tokens = Path(d) / 't.tokens'
        tokens.write_text('1 2 3\n')
        missing = subprocess.run([str(DRIVER), str(Path(d) / 'absent.gguf'), str(tokens), str(Path(d) / 'out'), '0'],
                                 capture_output=True, text=True)
    assert missing.returncode == 2 and 'absent.gguf' in missing.stderr, (missing.returncode, missing.stderr)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    )


if __name__ == '__main__':
    unittest.main()
