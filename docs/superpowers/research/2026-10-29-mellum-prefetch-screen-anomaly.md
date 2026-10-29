# The M5 prefetch-screen anomaly: an ungated dispatch after the S6 merge

Date: 2026-10-29. A screen (`docs/sdd.md`, "Evals"), closing the anomaly that S6 phase D's pause left open.

**Honesty header.** Measured on the M5 Max (128 GB), MXFP4 streamed at 2.5 GiB (815 slots), phase C's workload,
128 tokens, n=2 per arm unless stated. The cause is shown by a test that fails before the fix and passes after.

## The anomaly

The synchronous next-layer prefetch screen avoided 3.68 stops per token on the earlier build (`42f51b7`, 16.52 →
12.84; [stream decode](2026-10-29-mellum-stream-decode.md)) but only 1.94 on the merged build (16.55 → 14.61;
[sync screen](2026-10-29-mellum-s6-d-sync-screen/m5-dryrun/analysis.txt)), with the same cache, prompt and
prefetch-off baseline.

## What it was

- **Not the order of prefetch and encode.** Prefetching before the resumed pass is encoded (a screen switch,
  `anomaly/early-*`) gave 14.27 stops per token against 14.61 after it (n=2 each), and lost the read overlap
  (39-44 ms/token).
- **The prediction's input was overwritten.** A diagnostic compared scratch `FFN_NORM` at each stop with the copy
  the check kernel saved for the stopped layer: 0 of 500 matched (n=1, 64 tokens).
- **The cause:** `ds4_gpu_add_rms_norm_weight_tensor` dispatched directly, not through `ds4_gpu_dispatch_gated`.
  `mellum`'s decode fused the attention residual and FFN norm into it (`0bb422a`), and the S6 merge (`c04037c`)
  made streamed decode use it too, for word identity with resident decode. After a stop, every later layer's
  fused norm still ran and overwrote `ATTN_RES` and `FFN_NORM`. Resume restores both from the save slot, so logits
  stayed exact and every identity test passed; the cost was GPU time and a wrong prediction input.
- **The test** `--mellum-stream-stop-scratch` (`73406bb`, red: 189 of 196 stops overwritten; `4e6450a`, green:
  196 of 196 intact, 0 differing logit words). The full Mellum model suite passes on Q8_0 and MXFP4 (32 of 32).

## After the fix ([captures](2026-10-29-mellum-s6-d-sync-screen/m5-after-gate-fix/analysis.txt))

| Arm | Decode ms/token | Stops/token | Demand experts/token | Prefetched experts/token |
|---|---|---|---|---|
| prefetch off | 26.3, 26.4 | 16.55 | 29.01 | 0 |
| prefetch on | 27.5, 27.8 | 12.84 | 19.08 | 13.81 |

- Stops avoided: **3.72 per token**, matching the earlier build's 3.68. Prefetched 13.81 (earlier 13.9).
- Prefetch off is about 0.7 ms/token faster than before the fix (26.3-26.4 against 26.9-27.0), the suppressed
  layers' norms no longer running. n=2 per arm; not preregistered.
- On the iMac, 3.72 avoided stops × about 5.1 ms is about 19 ms/token gross, before the extra reads: still at or
  under phase D's 20 ms/token bar. The sync screen there remains unrun.

## What would falsify this

- A build with the fix whose stop-scratch test shows any overwritten stop.
- An iMac sync screen whose stops avoided differ from 3.72 by more than the routing difference between machines.

## Risks

- n=2 per arm, one prompt, one machine. Other kernels were checked by reading for direct dispatches on the decode
  layer path (`mellum_dispatch` and the Q8_0 projections are gated); the stop-scratch test watches only
  `ATTN_RES` and `FFN_NORM`.
