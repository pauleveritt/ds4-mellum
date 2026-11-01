"""CPU tests of tools/mellum_bk_anatomy.py: per-kernel GPU time of --mellum-bk's forwards from an encoder timeline."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / 'tools/mellum_bk_anatomy.py'
    assert path.exists(), 'tools/mellum_bk_anatomy.py missing'
    spec = importlib.util.spec_from_file_location('mellum_bk_anatomy', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['mellum_bk_anatomy'] = m
    spec.loader.exec_module(m)
    return m


def buf(seq, kernels, span_us):
    lines = [f'B {seq} {len(kernels)} 1000 {1000 + span_us * 1000:.0f}']
    lines += [f'E {seq} {i} 0 0 {d:.2f} 0.00 0x0 1 1x1x1 1x1x1 {name}' for i, (name, d) in enumerate(kernels)]
    return lines


def decode(us): return [('kernel_mellum_moe_reduce_add', 10.0), ('kernel_qwen4_moe_mid', us)]
def chunk(us): return [('kernel_qwen4_attn_decode_rows_npt4', 5.0), ('kernel_qwen4_moe_mid', us)]
RESTORE = [('kernel_cpy', 1.0)]
KS = (1, 2, 4)


class Anatomy(unittest.TestCase):
    def timeline(self, rounds=2):
        """As --mellum-bk runs: per k, rounds of D C C D (D: k one-row forwards, C: one k-row forward; at k = 1 a C
        is itself a one-row forward), a restore after each arm.  Each k's decode steps are slower than the last
        k's, and round 0 differs, so pooling or keeping round 0 shows in the result.  Kernels overlap: each
        buffer's span is below the sum of its dispatches."""
        lines, seq = ['# ds4 encoder timeline'], 0
        for k in KS:
            for r in range(rounds):
                for arm in ('D', 'C', 'C', 'D'):
                    one_row = arm == 'D' or k == 1
                    for _ in range(k if arm == 'D' else 1):
                        seq += 1
                        us = 100.0 + 10 * k + (50 if r == 0 else 0)
                        lines += (buf(seq, decode(us), span_us=us) if one_row
                                  else buf(seq, chunk(100.0 * k + r), span_us=100.0 * k))
                    seq += 1
                    lines += buf(seq, RESTORE, span_us=1.0)
        return lines

    def test_each_k_has_its_own_decode_steps_with_round_0_dropped(self):
        m = load()
        found = m.anatomy(m.parse(self.timeline()), rounds=2, ks=KS)
        self.assertEqual(sorted(found), [1, 2, 4])
        self.assertAlmostEqual(found[1].decode['kernel_qwen4_moe_mid'], 110.0)
        self.assertAlmostEqual(found[2].decode['kernel_qwen4_moe_mid'], 120.0)
        self.assertAlmostEqual(found[4].decode['kernel_qwen4_moe_mid'], 140.0)
        self.assertNotIn('kernel_cpy', found[1].decode)

    def test_chunks_by_k_with_round_0_dropped(self):
        m = load()
        found = m.anatomy(m.parse(self.timeline()), rounds=2, ks=KS)
        self.assertIsNone(found[1].chunk)
        self.assertAlmostEqual(found[2].chunk['kernel_qwen4_moe_mid'], 201.0)
        self.assertAlmostEqual(found[4].chunk['kernel_qwen4_moe_mid'], 401.0)

    def test_totals_are_gpu_spans_not_summed_dispatches(self):
        m = load()
        found = m.anatomy(m.parse(self.timeline()), rounds=2, ks=KS)
        self.assertAlmostEqual(found[4].decode_span_us, 140.0)   # the dispatches sum to 150
        self.assertAlmostEqual(found[4].chunk_span_us, 400.0)    # the dispatches sum to 406

    def test_wrong_counts_are_refused(self):
        m = load()
        with self.assertRaisesRegex(ValueError, 'chunk forwards'):
            m.anatomy(m.parse(self.timeline()), rounds=2, ks=(1, 2, 4, 8))
        with self.assertRaisesRegex(ValueError, 'decode steps'):
            lines = self.timeline()
            m.anatomy(m.parse(lines[:1] + lines[4:]), rounds=2, ks=KS)   # the first decode step's B and 2 E lines


if __name__ == '__main__':
    unittest.main()
