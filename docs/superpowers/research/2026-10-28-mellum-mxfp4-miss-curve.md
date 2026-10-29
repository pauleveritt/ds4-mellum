# The -02 miss curve at MXFP4 expert size (S6 phase A)

Date: 2026-10-28. Branch `research/mellum-8gb-q4`. Tool: `tools/mellum_ssd_replay.py` with the
`expert_bytes` override (`d55d718`).

**Honesty header.** This is a replay, not a measurement of MXFP4. It takes one thinking-conversation
routing trace, captured from the **Q8_0** model (-02, n=1, 5,120 decode tokens), and costs the same routes
at MXFP4's expert size, 3,290,112 B. It assumes MXFP4 routes as Q8_0 does. That is unmeasured, and
JetBrains' 85.6% top-token match against BF16 says the logits do move. No SSD was read, no model ran,
and nothing here predicts throughput. Every figure is descriptive and chooses nothing.

## Method

`docs/superpowers/research/2026-10-28-mellum-mxfp4-miss-curve/curve.py` replays the trace under LRU
and under the relaxed-optimal lower bound. It reports decode demand loads per token and MB per token
(loads × expert bytes / 10^6) at expert caches of 2, 2.5, 3, 4, 5 and 6 GiB:

```
python3 -I curve.py TRACE              # Q8_0, the trace's own size
python3 -I curve.py TRACE 3290112      # MXFP4
```

The trace is `~/.codex/visualizations/2026/10/28/mellum-generated-phase1-02/validated/raw/trace.jsonl`,
sha256 `26b05a0c287ec7701b7fd52a051e6babb077393ea251cb4a430432835524d906` (22,334,547 B, read only).
The full output is in `curve-output.txt`. The Q8_0 rows at 2, 3, 4, 5 and 6 GiB equal
`2026-10-28-mellum-ssd-retarget/curve-output.txt`, which checks the override path against the earlier
run.

## Result (n=1 trace)

| Cache | Q8_0 slots | Q8_0 LRU MB/token | MXFP4 slots | MXFP4 LRU loads/token | MXFP4 LRU MB/token | MXFP4 bound MB/token |
|---:|---:|---:|---:|---:|---:|---:|
| 2 GiB | 326 | 651.2 | 652 | 34.72 | 114.2 | 47.8 |
| 2.5 GiB | 407 | 468.8 | 815 | 21.02 | 69.2 | 27.9 |
| 3 GiB | 489 | 372.1 | 979 | 12.62 | 41.5 | 15.8 |
| 4 GiB | 652 | 228.4 | 1,305 | 3.50 | 11.5 | 3.8 |
| 5 GiB | 815 | 138.3 | 1,631 | 0.31 | 1.0 | 0.3 |
| 6 GiB | 979 | 83.0 | 1,958 | 0.00 | 0.0 | 0.0 |

"Bound" is the relaxed-optimal policy, a whole-trace lower bound for atomic group residency, not a
schedule the engine runs.

Halving the expert doubles the slots in the same bytes. So an MXFP4 cache of B GiB serves exactly the
loads/token of a Q8_0 cache of 2B GiB (for example, 2.5 GiB MXFP4 is 21.02 loads/token, Q8_0's 5 GiB
row), at half the bytes per load. At 2.5 GiB, the S6 planning point, LRU demand falls from 468.8 to
69.2 MB/token, a factor of 6.8. All 1,792 routed experts (5.49 GiB in MXFP4) fit in the 6 GiB cache.

Arithmetic, not a prediction: at 2.5 GiB, 10 decode tokens/s would need about 0.69 GB/s of demand
reads, and 20 tokens/s about 1.38 GB/s. The iMac's SSD rate for this access pattern is unmeasured; plan
C's read bench measures it.

## What would falsify this

- MXFP4 routing differs enough from Q8_0 that a trace captured on the MXFP4 model gives LRU demand at
  2.5 GiB outside 35–140 MB/token (2× either way). Plan B can capture that trace on the M5 Max.
- A non-thinking or Pi-shaped trace has much worse locality than -02.
- The override path costs something other than loads × expert bytes. The Q8_0 rows matching the
  retarget run, and the unit tests in `tests/test_mellum_ssd_replay.py`, argue against this.

## Risks

- One trace, one workload, n=1. Every demand figure in the S6 memory plan comes from it.
- The replay does not model prefill tiling, prefetch, staging buffers or the page cache. On an 8 GB
  machine under memory pressure, the page cache may hold less than on the 128 GB laptop that captured
  the trace.
- The quantization changes the logits (JetBrains: KLD 0.115 against BF16), so routing may move too.
