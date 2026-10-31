"""CPU tests of the real-use report (tools/mellum_pi_report.py): bands of requests by context, and request marks."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / f'tools/{name}.py'
    assert path.exists(), f'tools/{name}.py missing'
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def req(index, cached, new, prefill_s=1.0, generated=100, total_s=2.0):
    Request = load('mellum_server_requests').Request
    return Request(index=index, cached=cached, prompt=cached + new, new=new, prefill_s=prefill_s,
                   generated=generated, total_s=total_s, finish='stop', error=None)


class Bands(unittest.TestCase):
    def test_small_prefills_leave_the_median(self):
        m = load('mellum_pi_report')
        first = m.bands([[req(0, 0, 100), req(1, 100, 3000)]])[0]
        self.assertEqual((first.low, first.high, first.prefill_n, first.decode_n), (0, 4096, 1, 2))
        self.assertAlmostEqual(first.prefill_median, 3000.0)

    def test_band_by_context_at_start(self):
        m = load('mellum_pi_report')
        found = m.bands([[req(0, 4095, 3000), req(1, 4096, 3000)]])
        self.assertEqual([b.prefill_n for b in found], [1, 1, 0, 0])
        self.assertEqual([b.runs for b in found], [1, 1, 0, 0])

    def test_empty_band_has_none_not_zero(self):
        m = load('mellum_pi_report')
        last = m.bands([[req(0, 0, 3000)]])[-1]
        self.assertEqual((last.low, last.high), (16384, 32769))
        self.assertIsNone(last.prefill_median)
        self.assertIsNone(last.decode_median)
        self.assertEqual((last.prefill_n, last.decode_n, last.runs), (0, 0, 0))

    def test_runs_counts_each_run_once(self):
        m = load('mellum_pi_report')
        first = m.bands([[req(0, 0, 3000), req(1, 10, 3000)], [req(0, 0, 3000)]])[0]
        self.assertEqual((first.prefill_n, first.runs), (3, 2))

    def test_first_request_is_marked_cold(self):
        m = load('mellum_pi_report')
        self.assertEqual(m.marks(0, []), ['cold'])
        self.assertEqual(m.marks(3, [3]), ['compaction'])
        self.assertEqual(m.marks(2, [3]), [])


if __name__ == '__main__':
    unittest.main()
