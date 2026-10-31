# S5 measurement 02: grouped prefill cuts the cold first token from 60.6 to 37.7 s, still exact

Date: 2026-10-28, 21:11–21:26 EDT, one run. [Preregistration](2026-10-28-mellum-s5-measurement-02-prereg.md)
(`b46087d`, amended before timing in `397f355`), measured at `397f355` with the default K = 8. Raw outputs:
`~/.codex/visualizations/2026/10/28/mellum-s5-measure-02`. Analysis from the raw outputs
only: `.superpowers/sdd/2026-10-28-mellum-s5-layer-major-prefill/analyze-s5-02.py`.

**Honesty header.**
- **Measured:** this 128 GiB M5 Max's SSD and compute only. The same frozen prompt (30,620 tokens) and
  appended tool result (1,905 tokens) as measurement 01. A/B n=6 per arm on the CLI's session path, R n=1,
  C n=3.
- **Passes:** no regression, identity, suffix reuse, device-true reads.
- **One preregistered prediction failed:** prefill loads came in above the ceiling (below).
- **Not claimed:** speed on any other machine, or why the loads exceeded the ceiling.

## Rules

| Rule | Result |
|---|---|
| No regression (session path) | **Pass.** B/A medians 0.996 for prefill (516.0 vs 518.3 t/s) and 1.000 for generation (138.8 vs 138.8 t/s), n=6 each. |
| Identity | **Pass, 3 of 3.** Every C run gives R's 2,048 ids and its 256 ids after the append. |
| Suffix reuse | **Pass, all four runs.** The append starts at 32,668, ends at 34,573, and the common prefix is 32,668. |
| Read class | **Device-observed, 3 of 3.** 228.9 GB logical against 228.57–228.61 GB device (0.999), under the 600 GB cap. Each copy verified after its run. |
| Prefill loads ≤ 7,168 | **Contradicted.** 7,416 in all three runs, 3.5% over. |

**The load ceiling.** The ceiling assumed each layer pass loads at most a layer's 64 experts per group:
4 groups × 1,792. The figure counts the first decode token's loads too, and those cannot be separated from
the raw data; the second token alone loaded 53. Even so, 7,416 is 248 over the ceiling, more than one
token's loads plausibly add. So some experts were loaded more than once within a group's layer pass.
- **Hypothesis, not tested:** eviction is frequency-based (`prune_global`), so a freshly loaded expert has a
  low count. When a later chunk of the same layer needs a slot, the expert the previous chunk just loaded
  may be the one evicted, since the protect set covers only the current chunk.

The prediction is recorded as contradicted. The saving it was meant to bound did happen: 50,806 prefill
loads became 7,416, 85% fewer.

## Results

| Arm | Cold first token | Prefill loads | Decode t/s, all / first 256 / last 256 | Append: first token, loads | Decode after append | Peak footprint / Metal |
|---|---:|---:|---|---|---:|---|
| R | 28.9 s | 0 | 96.0 / 102.4 / 94.8 | 3.73 s, 0 | 99.0 | 0.99 / 12.84 GiB |
| C, n=3 | 36.1–45.6 s, median 37.7 | 7,416 | 38.9–39.4 / 36.0–36.4 / 38.6–39.1, median 39.3 | 3.80–4.97 s, median 4.16; 869 | 30.1 | 7.05 / 7.94 GiB |
| 01's C, n=3 | 60.4–66.9 s, median 60.6 | 50,806 | median 39.0 | median 4.29; 1,642 | 29.5 | 6.98 / 7.87 GiB |

- **Cold first token:** 60.6 to 37.7 s median, 8.8 s above resident. The prediction was near 34 s. C1's
  45.6 s is the outlier of the three. Logical reads per process fell from 519.5 to 228.9 GB; most of what
  remains is decode's.
- **The append** read 869 experts instead of 1,642, as predicted, but its first token only moved from 4.29 to
  4.16 s: at this size, the append is not read-bound.
- **Decode is unchanged,** as expected: 8.72 stops and 10.79 loads per token, 39.3 t/s. After the append it
  is 30.1 t/s, with 12.69 stops per token, the same rise as in 01.
- **Memory:** the startup plan line counts the group buffer. Peak Metal allocation rose by exactly
  75,497,472 B, the predicted group buffer, to 7.94 GiB: still under the about 8 GiB budget.

## Against S5's targets, for orientation only (this laptop, not the M1 Pro)

Decode 39.3 t/s (target 15); cold first token 37.7 s (target 60); a turn's first token 4.2 s (target 10).
All three are now inside the targets here.

## What would falsify this

- **A rerun with prefill loads well below 7,416, or varying between runs,** would contradict the
  determinism seen here (identical in all three runs and in the load test).
- **A per-group load trace** showing no repeated expert within a layer pass would undo the eviction
  hypothesis.

## Risks

- One prompt and one appended text, thinking-phase decode, n=3, one machine.
- The cold first token varies by 9.5 s across three runs; the median, not the range, carries the claim.
- K = 8 only; other K values are unmeasured.

## Next

1. **The M1 Pro bring-up and its 32K run,** where S5's targets are judged.
2. **Fewer stops per token** (S5 step 3). The eviction hypothesis above is a candidate input: within a
   group, protecting the whole layer pass rather than one chunk would test it.
