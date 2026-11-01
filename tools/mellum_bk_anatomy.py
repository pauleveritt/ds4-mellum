"""Per-kernel GPU time of the decode steps and the k-row forward of each k, from a --mellum-bk encoder timeline.

    python3 tools/mellum_bk_anatomy.py TIMELINE ROUNDS

TIMELINE is DS4_METAL_ENCODER_TIMELINE's file from one `./ds4_test --mellum-bk` process (tests/mellum_bk.sh with
TIMELINE=1), ROUNDS its DS4_TEST_MELLUM_BK_N.  The test runs k = 1, 2, 3, 4, 6, 8 in that order, ROUNDS rounds of
D C C D each (D: k one-row forwards; C: one k-row forward, itself a one-row forward at k = 1), so in timeline order
each k contributes 4 x ROUNDS one-row command buffers at k = 1 and 2k x ROUNDS at larger k, plus 2 x ROUNDS k-row
buffers.  A one-row forward's buffer is the one with the fused one-row MoE reduce (kernel_mellum_moe_reduce_add); a
k-row forward's, the one with the rows attention (kernel_qwen4_attn_decode_rows_npt4).  Round 0 of every k is
dropped, as the test drops it, and each k's ratio uses that k's own decode steps.

Per kernel: the mean summed dispatch time per buffer.  Dispatches can overlap, so these sum to more than the
buffer's GPU span; totals and ratios use the span (the B line's GPU end minus start).  The timeline's pass
boundaries make every dispatch a little slower than in an untimed run.
"""
from collections import defaultdict
from dataclasses import dataclass
import sys
from typing import Iterable

KS = (1, 2, 3, 4, 6, 8)
DECODE_MARK = 'kernel_mellum_moe_reduce_add'
CHUNK_MARK = 'kernel_qwen4_attn_decode_rows_npt4'


@dataclass(frozen=True)
class Buffer:
    span_us: float
    kernels: tuple[tuple[str, float], ...]   # (kernel, GPU microseconds) per dispatch


@dataclass(frozen=True)
class K:
    decode: dict[str, float]          # one decode step, mean microseconds per kernel
    decode_span_us: float
    chunk: dict[str, float] | None    # one k-row forward (None at k = 1)
    chunk_span_us: float | None


def parse(lines: Iterable[str]) -> list[Buffer]:
    """Command buffers in timeline order."""
    order, spans, kernels = [], {}, defaultdict(list)
    for line in lines:
        f = line.split()
        if not f or f[0] == '#':
            continue
        if f[0] == 'B':
            order.append(f[1])
            spans[f[1]] = (float(f[4]) - float(f[3])) / 1e3
        elif f[0] == 'E':
            kernels[f[1]].append((f[11], float(f[5])))
    return [Buffer(spans[s], tuple(kernels[s])) for s in order]


def _mean(buffers: list[Buffer]) -> tuple[dict[str, float], float]:
    total: dict[str, float] = defaultdict(float)
    for b in buffers:
        for name, us in b.kernels:
            total[name] += us / len(buffers)
    return dict(total), sum(b.span_us for b in buffers) / len(buffers)


def anatomy(buffers: list[Buffer], rounds: int, ks=KS) -> dict[int, K]:
    decode = [b for b in buffers if any(n == DECODE_MARK for n, _ in b.kernels)]
    chunks = [b for b in buffers if any(n == CHUNK_MARK for n, _ in b.kernels)]
    big = [k for k in ks if k > 1]
    per_round = {k: 4 if k == 1 else 2 * k for k in ks}
    if len(chunks) != 2 * rounds * len(big):
        raise ValueError(f'{len(chunks)} chunk forwards, expected {2 * rounds * len(big)} (2 x {rounds} rounds x '
                         f'{len(big)} values of k above 1)')
    if len(decode) != rounds * sum(per_round.values()):
        raise ValueError(f'{len(decode)} decode steps, expected {rounds * sum(per_round.values())}')
    found, d0, c0 = {}, 0, 0
    for k in ks:
        group = decode[d0:d0 + rounds * per_round[k]]
        d0 += rounds * per_round[k]
        dec, dspan = _mean(group[per_round[k]:])
        chunk, cspan = None, None
        if k > 1:
            chunk, cspan = _mean(chunks[c0 + 2:c0 + 2 * rounds])
            c0 += 2 * rounds
        found[k] = K(dec, dspan, chunk, cspan)
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        found = anatomy(parse(f), int(argv[2]))
    big = [k for k in sorted(found) if k > 1]
    one = found[1].decode
    names = sorted(set(one) | {n for k in big for n in found[k].chunk},
                   key=lambda n: -found[big[-1]].chunk.get(n, 0.0))
    print('| kernel (ms, summed dispatches) | 1 row | ' + ' | '.join(f'{k} rows' for k in big) + ' |')
    print('|---' * (len(big) + 2) + '|')
    for n in names:
        print(f'| {n} | {one.get(n, 0.0) / 1e3:.3f} | ' +
              ' | '.join(f'{found[k].chunk.get(n, 0.0) / 1e3:.3f}' for k in big) + ' |')
    print(f'| GPU span, the forward | {found[1].decode_span_us / 1e3:.3f} | ' +
          ' | '.join(f'{found[k].chunk_span_us / 1e3:.3f}' for k in big) + ' |')
    print('| GPU span, one decode step in the same rounds | - | ' +
          ' | '.join(f'{found[k].decode_span_us / 1e3:.3f}' for k in big) + ' |')
    print('| span ratio, forward / decode step | 1.000 | ' +
          ' | '.join(f'{found[k].chunk_span_us / found[k].decode_span_us:.3f}' for k in big) + ' |')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
