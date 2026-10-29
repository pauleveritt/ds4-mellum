# S5 measurement 01: streamed decode at 32K is 49% faster after gating; a turn's first token in about 4 s

Date: 2026-10-28, 18:55–19:11 EDT, one run. [Preregistration](2026-10-28-mellum-s5-measurement-01-prereg.md)
(`71fc046`), measured at `4d0349c`; the engine is unchanged since `8ecb058`. Raw outputs:
`/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-s5-measure-01`. Analysis from the raw outputs
only: `.superpowers/sdd/2026-10-28-mellum-s4-cached-execution/analyze-s5-01.py`.

**Honesty header.**
- **Measured:** this 128 GiB M5 Max's SSD and compute only. One frozen prompt (30,620 tokens), one frozen
  appended tool result (1,905 tokens rendered), 2,048 + 256 greedy tokens. A/B n=6 per arm, R n=1, C n=3.
- **Passes:** no regression, identity, device-true reads.
- **One recorded failure:** R's suffix-reuse time check (below). Under the preregistration, R's per-turn
  case is invalid. The per-turn figures are reported, not claimed.
- **Not claimed:** speed on any other machine. S5's targets are judged on the M1 Pro.

## Rules

| Rule | Result |
|---|---|
| No regression | **Pass.** B/A medians 1.006 for prefill (514.3 vs 511.2 t/s) and 0.993 for generation (137.0 vs 138.0 t/s), n=6 each. |
| Identity | **Pass, 3 of 3.** Every C run gives R's 2,048 ids and R's 256 ids after the append. |
| Read class | **Device-observed, 3 of 3.** 519.5 GB logical against 517.2, 517.3 and 517.8 GB device (0.996–0.997), under the 600 GB cap. Each copy verified after its run. |
| Suffix reuse, positions | **Pass, all four runs.** The append sync starts at 32,668, ends at 34,573 (+1,905), and the checkpoint's common prefix with the turn is 32,668. |
| Suffix reuse, R's time | **Recorded failure.** R's append took 3.89 s against a bound of 2.96 s (a tenth of its cold 29.56 s). |

**Why the time check failed (arithmetic, not measured).** The bound assumed the appended rows cost about
what the average prompt row costs. They come at positions 32.7K–34.6K, where every full-attention layer
attends over the whole context, so they cost about twice the prompt's average (2.0 ms per row against
0.97). A rebuild would take at least 29.6 s, so 3.89 s is consistent with reuse, and the position and
common-prefix checks pass. Still, the preregistered rule failed, and it stands. A later preregistration
should use the common-prefix check instead.

## Results

| Arm | Cold first token | Decode t/s, all / first 256 / last 256 | Stops / loads per token | Append: first token, loads | Decode t/s after append | Peak footprint / Metal |
|---|---:|---|---|---|---:|---|
| R | 29.6 s | 95.6 / 102.7 / 90.9 | 0 / 0 | 3.89 s, 0 | 96.8 | 0.97 / 12.84 GiB |
| C, n=3 | 60.4–66.9 s, median 60.6 | 37.9–39.4 / 34.9–36.1 / 37.6–38.6, median 39.0 | 8.72 / 10.79 | 4.11–5.21 s, median 4.29; 1,642 | 29.0–29.7, median 29.5 | 6.98 / 7.87 GiB |
| 02's C, n=3 | 58.6–60.9 s | 26.1–27.2 / 24.0–25.7 / 24.3–26.0 | 8.72 / 10.79 | not measured | — | 6.93 / 7.81 GiB |

- **Gating paid off at 32K.** Decode rose from a median of 26.2 (02, recomputed from its raw outputs) to 39.0 t/s (+49%), with stops and
  loads per token unchanged. `9e9757f` removed re-run work, not reads.
- **Cold first token is unchanged and read-bound:** prefill still loaded 50,806 experts. C0's 66.9 s is
  the outlier of the three.
- **The append read 1,642 experts (10.8 GB)** for 1,905 tokens: about 0.9 of one pass over the experts,
  because two chunks touch nearly all 1,792.
- **Decode after the append is slower:** 29.5 t/s, with 12.69 stops and 17.1 loads per token, against
  8.69 stops over the last 256 tokens before it. The first token after the append stopped 26 times. The
  append's loads evicted decode's working set, and the new turn routes differently. This is a policy
  input for S5 step 3, not yet explained further.
- **Memory:** the larger context added about 59 MB, as preregistered: 6.98 GiB footprint, 7.87 GiB Metal.

## Against S5's targets, for orientation only (this laptop, not the M1 Pro)

- Decode 39.0 t/s at 32K, and 29.5 after a turn: both above 15.
- Cold first token, median 60.6 s: just over 60.
- A turn's first token, median 4.3 s, under 10, with the caveat that R's preregistered reuse check failed.

## What would falsify this

- **A rerun giving a decode median outside about 37–40 t/s** at the same settings would make the gating gain
  less certain than three runs suggest.
- **An append time that does not scale with appended tokens** in a run designed to vary them would undo
  the reuse inference. Only one appended length was measured.

## Risks

- One prompt, one appended text, thinking-phase decode, n=3, one machine.
- The post-append slowdown is one turn's behavior; other turns may route differently.
- The appended turn follows generated text cut mid-thinking, not a finished assistant turn.

## Next

1. **Layer-major prefill** (S5 step 2, [options](2026-10-28-mellum-s5-layer-major-prefill-options.md)): the
   cold first token is still about 31 s of reads above resident.
2. **M1 Pro bring-up**, then its preregistered 32K run.
3. **Fewer stops**, informed by the post-append rise from 8.7 to 12.7 stops per token.
