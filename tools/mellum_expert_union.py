"""How many distinct routed experts k consecutive tokens use, per layer: what a k-row verify's expert kernels read.

    uv run --no-project --with gguf python -I tools/mellum_expert_union.py MODEL.gguf PREFIX POS0 N [K...]

Reads tests/mellum_trace's <PREFIX>.p<pos>.ffn_in.f32 (28 x 2304 f32, each layer's MoE input) for positions
POS0 .. POS0+N-1 and each layer's router (blk.<l>.ffn_gate_inp.weight, f32) from the GGUF, routes every position
on the CPU (Mellum's selection is the top 8 router logits, no bias), and prints, for each window length K
(default 1 2 3 4 8), the mean over windows and layers of the number of distinct experts.  CPU float order can
differ from the GPU's at a near tie, so a rare selection may differ from the engine's.
"""
from pathlib import Path
import sys

import numpy as np

N_LAYER, N_EMBD, N_EXPERT, N_USED = 28, 2304, 64, 8


def route(router: np.ndarray, x: np.ndarray, n_used: int = N_USED) -> frozenset[int]:
    """router: n_expert x n_embd; x: n_embd.  The n_used experts with the largest logits."""
    logits = router @ x
    return frozenset(int(e) for e in np.argsort(-logits, kind='stable')[:n_used])


def mean_union(selected: list[list[frozenset[int]]], k: int) -> float | None:
    """selected[position][layer]; the mean over windows of k consecutive positions and over layers."""
    if k > len(selected):
        return None
    counts = [len(frozenset().union(*(selected[p][l] for p in range(start, start + k))))
              for start in range(len(selected) - k + 1) for l in range(len(selected[0]))]
    return sum(counts) / len(counts)


def routers(model: Path) -> list[np.ndarray]:
    from gguf import GGUFReader
    by_name = {t.name: t for t in GGUFReader(str(model)).tensors}
    return [np.asarray(by_name[f'blk.{l}.ffn_gate_inp.weight'].data, dtype=np.float32).reshape(N_EXPERT, N_EMBD)
            for l in range(N_LAYER)]


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    model, prefix, pos0, n = Path(argv[1]), argv[2], int(argv[3]), int(argv[4])
    ks = [int(k) for k in argv[5:]] or [1, 2, 3, 4, 8]
    rs = routers(model)
    selected = []
    for p in range(pos0, pos0 + n):
        ffn = np.fromfile(f'{prefix}.p{p}.ffn_in.f32', dtype=np.float32).reshape(N_LAYER, N_EMBD)
        selected.append([route(rs[l], ffn[l]) for l in range(N_LAYER)])
    for k in ks:
        u = mean_union(selected, k)
        print(f'k {k}: distinct experts per layer {"-" if u is None else f"{u:.2f}"} '
              f'({"-" if u is None else f"{u / N_USED:.3f}"} x one token), windows {max(n - k + 1, 0)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
