"""The oracle's logits comparison (tests/mellum_logits_compare.py) on small hand-made trace files.

    python3 -I tests/test_mellum_logits_compare.py    (part of `make test-mellum`)
"""

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent / 'mellum_logits_compare.py'
POSITIONS = (0, 5, 9)
BASE = [2.0, 1.0, 0.5, -1.0, 0.0, 0.25, -0.5, 1.5]


def write_run(prefix, rows):
    """rows maps 'p0', 'p5', 'p9' and 'session' to a list of logits."""
    for name, row in rows.items():
        with open('%s.%s.logits.f32' % (prefix, name), 'wb') as f:
            f.write(struct.pack('<%df' % len(row), *row))


def base_rows():
    rows = {'p%d' % p: list(BASE) for p in POSITIONS}
    rows['session'] = list(BASE)
    return rows


class Compare(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.a = os.path.join(self.dir.name, 'a')
        self.b = os.path.join(self.dir.name, 'b')

    def tearDown(self):
        self.dir.cleanup()

    def run_compare(self, a_rows, b_rows, exact, tol=10, kltol=0.01):
        write_run(self.a, a_rows)
        write_run(self.b, b_rows)
        r = subprocess.run([sys.executable, '-I', str(SCRIPT), self.a, self.b, ','.join(map(str, POSITIONS)),
                            '1' if exact else '0', str(tol), str(kltol)], capture_output=True, text=True)
        return r.returncode, r.stdout

    def test_equal_runs_pass_both_rules(self):
        for exact in (True, False):
            rc, out = self.run_compare(base_rows(), base_rows(), exact)
            self.assertEqual(rc, 0, out)
            self.assertIn('4 of 4 logits files byte-equal', out)

    def test_a_small_difference_fails_exact_and_passes_tolerance(self):
        b = base_rows()
        b['p5'][3] += 1e-3
        self.assertEqual(self.run_compare(base_rows(), b, True)[0], 1)
        rc, out = self.run_compare(base_rows(), b, False)
        self.assertEqual(rc, 0, out)

    def test_a_different_top1_fails_tolerance(self):
        b = base_rows()
        b['p0'][7] = 2.5
        rc, out = self.run_compare(base_rows(), b, False)
        self.assertEqual(rc, 1, out)
        self.assertIn('top-1 agree 3/4', out)

    def test_a_large_logit_change_fails_tolerance(self):
        b = base_rows()
        b['p9'][3] = -20.0
        self.assertEqual(self.run_compare(base_rows(), b, False, tol=10)[0], 1)


if __name__ == '__main__':
    unittest.main()
