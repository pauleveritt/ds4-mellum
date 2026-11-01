"""Per-kernel GPU time of one decode step and of each k-row forward, from a --mellum-bk encoder timeline.

    python3 tools/mellum_bk_anatomy.py TIMELINE ROUNDS

TIMELINE is DS4_METAL_ENCODER_TIMELINE's file from one `./ds4_test --mellum-bk` process (tests/mellum_bk.sh with
TIMELINE=1), ROUNDS its DS4_TEST_MELLUM_BK_N.  A decode step's command buffer is the one with the fused one-row
MoE reduce (kernel_mellum_moe_reduce_add); a forward of 2 or more rows is the one with the rows attention
(kernel_qwen4_attn_decode_rows_npt4).  The test runs k = 1, 2, 3, 4, 6, 8 in that order, 2 chunk forwards per
round, so the chunk forwards fall into groups of 2 x ROUNDS per k >= 2; the first round of each group is dropped,
as the test drops it.  Decode steps are averaged over all but the first round of k = 1 (4 steps).  Prints one row
per kernel (ms per forward, the mean of the kept forwards) and the total.  The timeline's own pass boundaries add
a little time to every dispatch: read the anatomy, not the absolute times.
"""
from collections import defaultdict
import sys
from typing import Iterable

KS = (2, 3, 4, 6, 8)
DECODE_MARK = 'kernel_mellum_moe_reduce_add'
CHUNK_MARK = 'kernel_qwen4_attn_decode_rows_npt4'


def parse(lines: Iterable[str]) -> list[list[tuple[str, float]]]:
    """Command buffers in order, each a list of (kernel, GPU microseconds)."""
    order, by_seq = [], {}
    for line in lines:
        f = line.split()
        if not f or f[0] == '#':
            continue
        if f[0] == 'B':
            by_seq[f[1]] = []
            order.append(f[1])
        elif f[0] == 'E':
            by_seq.setdefault(f[1], []).append((f[11], float(f[5])))
    return [by_seq[s] for s in order]


def _mean(buffers: list[list[tuple[str, float]]]) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for b in buffers:
        for name, us in b:
            total[name] += us / len(buffers)
    return dict(total)


def anatomy(buffers, rounds: int, ks=KS) -> dict[int, dict[str, float]]:
    """{1: one decode step, k: one k-row forward}, each {kernel: mean microseconds}."""
    decode = [b for b in buffers if any(n == DECODE_MARK for n, _ in b)]
    chunks = [b for b in buffers if any(n == CHUNK_MARK for n, _ in b)]
    if len(chunks) != 2 * rounds * len(ks):
        raise ValueError(f'{len(chunks)} chunk forwards, expected {2 * rounds * len(ks)} (2 x {rounds} rounds x '
                         f'{len(ks)} values of k)')
    found = {1: _mean(decode[4:])}
    for i, k in enumerate(ks):
        found[k] = _mean(chunks[i * 2 * rounds + 2:(i + 1) * 2 * rounds])
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        found = anatomy(parse(f), int(argv[2]))
    ks = sorted(found)
    names = sorted({n for v in found.values() for n in v}, key=lambda n: -found[ks[-1]].get(n, 0.0))
    print('| kernel | ' + ' | '.join('1 row (decode)' if k == 1 else f'{k} rows' for k in ks) + ' |')
    print('|---' * (len(ks) + 1) + '|')
    for n in names:
        print(f'| {n} | ' + ' | '.join(f'{found[k].get(n, 0.0) / 1e3:.3f}' for k in ks) + ' |')
    print('| total | ' + ' | '.join(f'{sum(found[k].values()) / 1e3:.3f}' for k in ks) + ' |')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
