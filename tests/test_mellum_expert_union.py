"""CPU tests of tools/mellum_expert_union.py: top-8 routing from MoE inputs, and distinct experts over windows."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / 'tools/mellum_expert_union.py'
    assert path.exists(), 'tools/mellum_expert_union.py missing'
    spec = importlib.util.spec_from_file_location('mellum_expert_union', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['mellum_expert_union'] = m
    spec.loader.exec_module(m)
    return m


class Routing(unittest.TestCase):
    def test_top_k_of_router_logits(self):
        m = load()
        router = np.eye(4, dtype=np.float32)          # 4 experts over a 4-wide row: logit e = x[e]
        x = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
        self.assertEqual(m.route(router, x, 2), frozenset({1, 3}))


class Windows(unittest.TestCase):
    def test_distinct_experts_per_layer_and_window(self):
        m = load()
        # 3 positions, 2 layers
        sel = [[frozenset({0, 1}), frozenset({5, 6})],
               [frozenset({1, 2}), frozenset({5, 6})],
               [frozenset({3, 4}), frozenset({5, 7})]]
        self.assertEqual(m.mean_union(sel, 1), 2.0)
        # windows (0,1): layer0 3, layer1 2; (1,2): layer0 4, layer1 3 -> mean 3.0
        self.assertEqual(m.mean_union(sel, 2), 3.0)
        self.assertEqual(m.mean_union(sel, 3), (5 + 3) / 2)

    def test_window_longer_than_the_run_is_none(self):
        m = load()
        self.assertIsNone(m.mean_union([[frozenset({0})]], 2))


if __name__ == '__main__':
    unittest.main()
