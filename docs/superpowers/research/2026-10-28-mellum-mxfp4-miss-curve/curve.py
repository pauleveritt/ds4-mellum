"""Descriptive miss curve over the -02 held-out trace (n=1) at a given expert size; chooses nothing.

Usage: python3 -I curve.py TRACE_JSONL [EXPERT_BYTES]
Prints decode loads and MB per token.  Without EXPERT_BYTES the trace's own (Q8_0) size is used; with it,
the same routes are costed at that size, routing assumed unchanged by quantization.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tools'))
import mellum_ssd_replay as R

with open(sys.argv[1]) as stream:
    trace = R.load_trace(stream)
override = int(sys.argv[2]) if len(sys.argv) > 2 else None
expert = override or trace.header.geometry.expert_bytes
tokens = len({r.forward for r in trace.requests if r.phase == 'decode'})
print(f'expert_bytes {expert} ({"override" if override else "trace"}), decode tokens {tokens}')
for policy in ('lru', 'relaxed-optimal'):
    for gib in (2, 2.5, 3, 4, 5, 6):
        result = R.replay(trace, int(gib * 2**30), policy, expert_bytes=override)
        loads = result['by_phase']['decode']['demand_loads'] / tokens
        print(f'{policy:15s} {gib:3g} GiB slots {result["cache"]["slots"]:4d} '
              f'loads/token {loads:6.2f} MB/token {loads * expert / 1e6:6.1f}')
