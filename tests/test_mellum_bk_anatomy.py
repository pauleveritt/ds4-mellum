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


def buf(seq, kernels):
    lines = [f'B {seq} {len(kernels)} 0 0']
    lines += [f'E {seq} {i} 0 0 {d:.2f} 0.00 0x0 1 1x1x1 1x1x1 {name}' for i, (name, d) in enumerate(kernels)]
    return lines


DECODE = [('kernel_mellum_moe_reduce_add', 10.0), ('kernel_qwen4_moe_mid', 100.0)]
def chunk(t): return [('kernel_qwen4_attn_decode_rows_npt4', 5.0), ('kernel_qwen4_moe_mid', t)]
RESTORE = [('kernel_cpy', 1.0)]


class Anatomy(unittest.TestCase):
    def timeline(self):
        # one round per k is dropped; rounds = 2, ks = (2, 4): per k, D C C D twice, with restores between
        lines, seq = ['# ds4 encoder timeline'], 0
        for k, t in ((2, 150.0), (4, 250.0)):
            for r in range(2):
                for arm in ('D', 'C', 'C', 'D'):
                    for _ in (range(k) if arm == 'D' else [0]):
                        seq += 1
                        lines += buf(seq, DECODE if arm == 'D' else chunk(t + r))
                    seq += 1
                    lines += buf(seq, RESTORE)
        return lines

    def test_chunks_by_k_with_the_first_round_dropped(self):
        m = load()
        found = m.anatomy(m.parse(self.timeline()), rounds=2, ks=(2, 4))
        self.assertEqual(sorted(found), [1, 2, 4])
        self.assertAlmostEqual(found[2]['kernel_qwen4_moe_mid'], 151.0)   # round 1 only
        self.assertAlmostEqual(found[4]['kernel_qwen4_moe_mid'], 251.0)
        self.assertAlmostEqual(found[1]['kernel_qwen4_moe_mid'], 100.0)
        self.assertNotIn('kernel_cpy', found[1])

    def test_wrong_chunk_count_is_refused(self):
        m = load()
        with self.assertRaisesRegex(ValueError, 'chunk forwards'):
            m.anatomy(m.parse(self.timeline()), rounds=2, ks=(2, 4, 8))


if __name__ == '__main__':
    unittest.main()
