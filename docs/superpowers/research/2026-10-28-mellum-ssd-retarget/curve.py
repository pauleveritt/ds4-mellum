"""Descriptive miss curve over the -02 held-out trace (n=1); chooses nothing.

Usage: python3 -I curve.py TRACE_JSONL  (prints decode loads and MB per token)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tools'))
import mellum_ssd_replay as R

with open(sys.argv[1]) as stream:
    trace = R.load_trace(stream)
expert = trace.header.geometry.expert_bytes
tokens = len({r.forward for r in trace.requests if r.phase == 'decode'})
for policy in ('lru', 'relaxed-optimal'):
    for gib in (2, 3, 4, 5, 6, 7, 8, 10):
        decode = R.replay(trace, gib * 2**30, policy)['by_phase']['decode']
        loads = decode['demand_loads'] / tokens
        print(f'{policy:15s} {gib:2d} GiB loads/token {loads:6.2f} MB/token {loads * expert / 1e6:6.1f}')
