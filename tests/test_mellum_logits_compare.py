"""The oracle's logits comparison (tests/mellum_logits_compare.py) on small hand-made trace files.

    python3 -I tests/test_mellum_logits_compare.py    (part of `make test-mellum`)
"""

import math
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


def three_way(*p):
    """logits whose softmax is p"""
    return [math.log(x) for x in p]


def two_way(p):
    """logits whose softmax is (p, 1 - p)"""
    return [math.log(p), math.log(1 - p)]


class Compare(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.a = os.path.join(self.dir.name, 'a')
        self.b = os.path.join(self.dir.name, 'b')

    def tearDown(self):
        self.dir.cleanup()

    def run_compare(self, a_rows, b_rows, exact, tol=10, kltol=0.01, klmax=100, dlogp=100, positions=POSITIONS,
                    neartie=0):
        write_run(self.a, a_rows)
        write_run(self.b, b_rows)
        r = subprocess.run([sys.executable, '-I', str(SCRIPT), self.a, self.b, ','.join(map(str, positions)),
                            '1' if exact else '0', str(tol), str(kltol), str(klmax), str(dlogp), str(neartie)],
                           capture_output=True, text=True)
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

    def test_a_nan_in_an_earlier_reference_row_fails(self):
        b = base_rows()
        b['p0'][3] = float('nan')
        for exact in (True, False):
            rc, out = self.run_compare(base_rows(), b, exact)
            self.assertEqual(rc, 1, out)
            self.assertIn('not finite', out)

    def test_a_non_finite_value_in_either_run_fails(self):
        for side in ('a', 'b'):
            for bad in (float('nan'), float('inf'), float('-inf')):
                rows = base_rows()
                rows['p5'][6] = bad
                a, b = (rows, base_rows()) if side == 'a' else (base_rows(), rows)
                for exact in (True, False):
                    with self.subTest(side=side, value=bad, exact=exact):
                        rc, out = self.run_compare(a, b, exact)
                        self.assertEqual(rc, 1, out)
                        self.assertIn('not finite', out)

    def test_the_same_nan_in_both_runs_fails_although_byte_equal(self):
        rows = base_rows()
        rows['session'][2] = float('nan')
        for exact in (True, False):
            rc, out = self.run_compare(rows, rows, exact)
            self.assertEqual(rc, 1, out)
            self.assertIn('not finite', out)

    def test_one_position_far_off_fails_although_the_mean_kl_passes(self):
        # the review's case: 64 positions at 0.99/0.01, one of them 0.51/0.49 in the candidate (KL 0.618),
        # a mean KL of 0.0095 over 65 rows and the same top-1 everywhere
        positions = range(64)
        ref = {'p%d' % p: two_way(0.99) for p in positions}
        ref['session'] = two_way(0.99)
        cand = {k: list(v) for k, v in ref.items()}
        cand['p17'] = two_way(0.51)
        self.assertEqual(self.run_compare(cand, ref, False, positions=positions)[0], 0)
        rc, out = self.run_compare(cand, ref, False, klmax=0.05, positions=positions)
        self.assertEqual(rc, 1, out)
        self.assertIn('worst KL 6.18e-01 at p17', out)
        rc, out = self.run_compare(cand, ref, False, dlogp=0.2, positions=positions)
        self.assertEqual(rc, 1, out)
        self.assertIn('max |d log p(top-1)| 6.63e-01 at p17', out)

    def test_small_changes_at_every_position_pass_the_per_position_caps(self):
        positions = range(64)
        ref = {'p%d' % p: two_way(0.99) for p in positions}
        ref['session'] = two_way(0.99)
        cand = {k: two_way(0.985) for k in ref}
        rc, out = self.run_compare(cand, ref, False, klmax=0.05, dlogp=0.2, positions=positions)
        self.assertEqual(rc, 0, out)

    def test_the_session_row_is_not_counted_in_the_mean_kl(self):
        # the session row is the last position's row again: a change there counts once in the mean,
        # over the three positions (KL 0.0247 / 3 = 0.0082), not twice over four rows (0.0124)
        ref = {'p%d' % p: two_way(0.99) for p in POSITIONS}
        ref['session'] = two_way(0.99)
        cand = {k: list(v) for k, v in ref.items()}
        cand['p9'] = cand['session'] = two_way(0.95)
        rc, out = self.run_compare(cand, ref, False, kltol=0.01)
        self.assertEqual(rc, 0, out)
        self.assertIn('mean KL 8.25e-03', out)


    def near_tie_rows(self, ref_p, cand_p):
        ref = {'p%d' % p: three_way(0.6, 0.3, 0.1) for p in POSITIONS}
        ref['session'] = three_way(0.6, 0.3, 0.1)
        cand = {k: list(v) for k, v in ref.items()}
        ref['p5'], cand['p5'] = three_way(*ref_p), three_way(*cand_p)
        return cand, ref

    def test_a_top1_change_at_a_near_tie_passes(self):
        # the reference's top two are 0.12 nats apart (0.45 / 0.40); the candidate swaps them
        cand, ref = self.near_tie_rows((0.45, 0.40, 0.15), (0.40, 0.45, 0.15))
        rc, out = self.run_compare(cand, ref, False, neartie=0.75)
        self.assertEqual(rc, 0, out)
        self.assertIn('top-1 agree 3/4 (1 change at a near-tie)', out)

    def test_the_same_change_fails_under_a_narrower_margin(self):
        cand, ref = self.near_tie_rows((0.45, 0.40, 0.15), (0.40, 0.45, 0.15))
        rc, out = self.run_compare(cand, ref, False, neartie=0.05)
        self.assertEqual(rc, 1, out)

    def test_a_top1_change_at_a_wide_margin_fails(self):
        # the reference's top two are 1.25 nats apart (0.7 / 0.2): a change there is not a near-tie, whatever the caps
        cand, ref = self.near_tie_rows((0.7, 0.2, 0.1), (0.2, 0.7, 0.1))
        rc, out = self.run_compare(cand, ref, False, neartie=0.75)
        self.assertEqual(rc, 1, out)
        self.assertIn('top-1 agree 3/4 (0 changes at a near-tie)', out)

    def test_a_near_tie_change_still_meets_the_caps(self):
        cand, ref = self.near_tie_rows((0.45, 0.40, 0.15), (0.40, 0.45, 0.15))
        self.assertEqual(self.run_compare(cand, ref, False, neartie=0.75, klmax=1e-4)[0], 1)
        self.assertEqual(self.run_compare(cand, ref, False, neartie=0.75, dlogp=0.05)[0], 1)


if __name__ == '__main__':
    unittest.main()
