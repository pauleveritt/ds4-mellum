# S6 phase D: the merged build on the 8 GB iMac (Task 1, cache stage only)

Date: 2026-10-29. Build `754d419` (`mellum` after the S6 merge), the 8 GB M1 iMac over SSH, phase C's kit
(`run_imac.sh bringup`, then `cache`). Captures: [`2026-10-29-mellum-s6-d-evidence/ref/`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-29-mellum-s6-d-evidence/ref).
The owner concluded the run after this stage, so Task 1's three-run footprint was not taken.

**Honesty header.** Measured: one machine, n=1 per cache size, the phase C workload (512 tokens, apps closed).
Not measured: the footprint at n=3, anything after this stage.

**Bringup:** `make`, `make tools`, `test-mellum` and `test-mellum-kernels` rc 0. The three `ds4_test` streaming
checks failed for memory, as in phase C (recorded, not gated). They cost about 35 minutes and left the swap
nearly full (1.76 of 2 GiB), so the 3 GiB run below swapped.

| Cache | Metal peak (B) | Against the C2 limit (4,652,892,160 B) | Decode t/s | First token s | Experts read/token | Swap-outs |
|---|---|---|---|---|---|---|
| 2 GiB | 4,185,079,808 | 467,812,352 under | 5.77 | 16.79 | 35.35 | 0 |
| 2.5 GiB | 4,721,868,800 | **68,976,640 over** | 7.14 | 16.74 | 23.00 | 0 |
| 3 GiB | 5,261,950,976 | 609,058,816 over | 8.82 | 16.87 | 14.46 | 11,792 |

- The 2.5 GiB overage matches the estimate from the M5: phase C's 4,677,697,536 plus the merge's 44,171,264 B
  (the rewind frontier) is exactly 4,721,868,800. Phase C missed by 24.8 MB.
- First token is about 2.6 s faster than phase C's 19.4 s; experts read per token rose slightly (35.35 against
  34.71 at 2 GiB). Both are n=1 and not explained here; the merge changed decode's fused kernels.

Recompute: `python3 -I` over `ref/cache/*.measure.json` with phase C's metric
([`imac/analyze.py`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-s6-c-evidence/imac/analyze.py) reads the same fields).
