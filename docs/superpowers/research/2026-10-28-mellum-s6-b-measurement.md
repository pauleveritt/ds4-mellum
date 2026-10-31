# S6 phase B (c), (d), (e): Q8_0 no regression, Q8_0 against MXFP4, streamed MXFP4

Date: 2026-10-28, 22:38–23:11 EDT. Build `7759005` (arm B). Arm A is `1500320`, built in a detached worktree.
Machine: the M5 Max, 128 GiB, macOS 27.2. Judged by [the preregistration](2026-10-28-mellum-s6-b-preregistration.md),
committed (`ff22dcc`) before any S6 GPU run.

**Honesty header.**
- **Measured:** this machine only, one pass of the preregistered processes. Each figure carries its `n`.
- **Not measured:**
  - **8 GB behavior.** The streamed runs had 128 GiB around them, so the page cache can serve expert reads.
    Their decode speed is not an 8 GB figure, and device reads are not classified.
  - **Energy.** `powermetrics` needs root.
- **No reruns:** nothing was rerun, and no process failed.

## Method

[`run_cde.py`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-s6-b-evidence/measure/run_cde.py) ran the preregistered orders with 30 s rests.
Each arm ran in its own checkout, because the engine compiles `metal/*.metal` relative to its working
directory (`ds4_metal.m:4768`). Each process's stdout, stderr and argv are in `measure/raw/` (git-ignored,
kept). The figures and verdicts are in `measure/summary-{c,d,e}.json`, computed by the script.

**Deviations from the preregistration's wording** (the review's minor 10):
- (c) compares the 12 runs' generated text, not token ids.
- The collapse check looks anywhere in the 2,048 tokens, not only at the end, which is stricter.

## (c) Q8_0 no regression: **pass**

`ds4 -c 8192 --temp 0 -n 256` on the short prompt, A/B/B/A, n=6 per arm.

| | Prefill t/s, median (range) | Generation t/s, median (range) |
|---|---|---|
| A `1500320` | 499.7 (493.4–506.7) | 139.54 (139.0–139.7) |
| B `7759005` | 495.2 (492.3–502.2) | 139.57 (139.1–139.7) |
| B / A | 0.991 (≥ 0.98) | 1.000 (≥ 0.98) |

- **Generated text:** identical in all 12 runs.
- **Traced logits:** `tests/mellum_trace` prefill logits on the three oracle token files are byte-equal
  between A and B, 160 of 160 files (30 + 65 + 65).

## (d) Resident Q8_0 against resident MXFP4 (reported, not gated)

**Long prompt:** 30,620 tokens, 2,048 greedy tokens, context 32,768, in Q8/MX/MX/Q8/Q8/MX order. Medians,
with ranges in parentheses:

| | n | Time to first token, s | Prefill t/s | Decode t/s, tokens 0–255 | Decode t/s, last 256 |
|---|---:|---|---|---|---|
| Q8_0 | 3 | 28.77 (28.16–32.12) | 1,064 (953–1,088) | 102.7 (100.9–103.8) | 88.1 (86.1–91.4) |
| MXFP4 | 3 | 29.50 (28.56–31.59) | 1,038 (969–1,072) | 115.9 (115.2–118.1) | 100.4 (99.6–101.9) |

**Short prompt:** A/B/B/A, n=6 per arm.

| | Prefill t/s | Generation t/s |
|---|---|---|
| Q8_0 | 497.8 (496.0–507.8) | 139.5 (137.7–139.7) |
| MXFP4 | 580.2 (575.0–601.5) | 166.8 (166.4–167.2) |

On this machine:
- **Decode:** MXFP4 is about 1.13× (long prompt, both windows) and 1.20× (short prompt). That is inside the
  design's 1.1–1.3× expectation and well under the 1.45× byte-count ceiling.
- **Long-prompt prefill:** unchanged, within the ranges.
- **Energy:** not measured.

The figures are n=3 and n=6 on one machine. They feed the later bigger-Mac decision and claim nothing for
it.

## (e) Streamed MXFP4 against resident MXFP4: **pass**

Long prompt, 2,048 greedy tokens. The resident run is n=1. The streamed runs are two at 2.5 GiB and one at
3 GiB.

| Run | Slots | Ids equal to resident | Decode loads/token | Time to first token, s | Decode t/s (0–255 / last 256) | Peak `phys_footprint` | Metal allocated peak |
|---|---:|---|---:|---:|---|---:|---:|
| resident | 0 | (reference) | 0 | 27.9 | 117.6 / 103.7 | 1.01 GB | 7.83 GB |
| 2.5 GiB #1 | 815 | yes, 2,048/2,048 | 19.98 | 44.5 | 26.3 / 22.6 | 3.67 GB | 4.62 GB |
| 2.5 GiB #2 | 815 | yes, 2,048/2,048 | 19.98 | 42.6 | 25.6 / 22.4 | 3.67 GB | 4.62 GB |
| 3 GiB | 979 | yes, 2,048/2,048 | 11.47 | 42.9 | 33.1 / 34.2 | 4.21 GB | 5.16 GB |

- **Collapse:** none in the resident run's 2,048 ids, under the check described above.
- **Loads per token against the replay:** 19.98 at 2.5 GiB, against the replay's 21.02 from the Q8_0
  trace ([miss curve](2026-10-28-mellum-mxfp4-miss-curve.md)). That is inside the preregistered 10.5–42
  band. At 3 GiB it is 11.47, against the replay's 12.62. This is a different prompt (code, not the -02
  thinking trace), n=1 per size.
- **Memory plan:** the Metal allocation at 2.5 GiB is 4.62 GB (4.30 GiB). The S6 memory plan put the whole
  process near 4.59 GiB at 32K, so it is in range. The iMac measures its own.
- **Decode speed:** 22–26 t/s streamed at 2.5 GiB, on this machine with its page cache. It is a ceiling
  for the iMac, not an estimate of it.

## What would falsify this

- On an 8 GB machine, loads per token at 2.5 GiB fall outside 10–40 on a Pi-shaped prompt, so the
  locality here does not carry over.
- A rerun of (c) after a reboot shows B more than 2% slower than A, so (c)'s pass was noise.
- Streamed ids diverge from resident ones on a prompt or a context these runs did not cover.

## Risks

- One prompt pair, one machine, n=3 to 6.
- The streamed speeds include warm page-cache reads. The iMac's SSD and memory pressure are the real
  test (phase C).
- MXFP4 does not get Mellum's E1 Q8_0 tile tuning. It runs the qwen4 default, and `DS4_QWEN4_MOE_MM_NAX`
  steers its prefill. Prefill speed may have room left.
