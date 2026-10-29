# S5 measurement 02 (grouped layer-major prefill at K = 8): preregistration

Date: 2026-10-28, committed before any timing. Branch `codex/mellum-ssd-replay`. S5 step 2 in the
[roadmap](../../../ROADMAP.md); design: [grouped layer-major prefill](../specs/2026-10-28-mellum-s5-layer-major-prefill-design.md).
- **Why:** [S5 measurement 01](2026-10-28-mellum-s5-measurement-01.md) left the cold first token at 60.6 s
  (median, n=3) against resident's 29.6 s, with 50,806 prefill loads (334 GB). Grouped prefill should
  load each layer's experts about once per group.
- **The measured build** is this file's amendment commit, which follows the final review's fix
  (`21848ca`).
- **Amended before any timing,** after the final review: the A/B arms now run the CLI's session path
  (`DS4_CLI_FORCE_SESSION=1`). The default CLI path (`generate_qwen4_metal_argmax`) never enters the session
  sync this change touches. The prefill-loads definition is also stated more precisely. Its
  engine changes since S5 measurement 01's build (`4d0349c`) are the grouped prefill and its plan
  (`git diff 4d0349c..<build> -- ds4.c`), plus the `ds4_engine_close(NULL)` fix (`3a63836`).

**Honesty header.**
- **Measured:** this 128 GiB M5 Max's SSD and compute only, the same frozen prompt and appended text as
  measurement 01, n as stated per arm.
- **One run:** no rerun or retry follows any outcome.
- **Not claimed:** speed on any other machine. S5's targets are judged on the owner's M1 Pro; this run
  reports against them for orientation only.

## Frozen inputs (as measurement 01)

- **Model:** `~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` (Q8_0, 12,925,417,760 B).
- **Prompt:** [`prompt.txt`](2026-10-28-mellum-s4-measurement-prereg/prompt.txt), SHA-256 `2164763c…6344da`,
  30,620 tokens rendered.
- **Appended text:** [`append.txt`](2026-10-28-mellum-s5-measurement-01-prereg/append.txt), SHA-256
  `ed93c59f…09f23ce`, 1,905 tokens rendered.
- **Context, tokens and cache:** 36,864; 2,048 greedy tokens, the append, then 256; `--stream-bytes 6442450944`
  (979 slots).
- **K:** `DS4_MELLUM_STREAM_PREFILL_GROUP` unset, so the default 8. The driver run's stderr records the
  startup plan line.

## Arms

| Arm | Build | Tool | Runs |
|---|---|---|---|
| A | pre-S4 baseline `b1994c6` | `DS4_CLI_FORCE_SESSION=1 ds4 -c 8192 --temp 0 -n 256 -p "Explain how a hash map works, with an example in C."` | 6 |
| B | measured build, streaming off | the same command | 6 |
| R | measured build, resident | `tools/mellum_s4_measure` with `--append` | 1 |
| C | measured build, streamed at 6 GiB, K = 8 | the driver with `--append` and `--stream-bytes`, each from a fresh write-only `F_NOCACHE` copy | 3 |

- **A/B runs** because the session sync's chunk loop, shared with resident Mellum, changed. Both builds'
  session path prints `ds4: prefill: X t/s, generation: Y t/s`, and the runner parses that line (smoke run,
  n=1 per build, 517.6 and 519.1 t/s prefill). The rates are not comparable with measurement 01's A/B, which
  used the default path.
- **Order:** A/B/B/A blocks (`regression_order(6)`) with 30 s rests, then R, then C, with 30 s rests.
- **Each C process:** bracketed by block-driver read counters; its copy is verified after the run and
  then deleted.
- **The shared model lock:** every run respects it. A held lock before the first run postpones the whole
  measurement.
- **Raw outputs:** `/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-s5-measure-02`, an exclusive
  root.

## Decisions fixed now

- **No regression:** B's median prefill and generation t/s are each at least 0.98 of A's (one-sided).
- **Identity:** every C run's 2,048 ids and 256 post-append ids equal R's.
- **Suffix reuse:** for every run, `append_pos_before` and `append_common_prefix` equal prompt + 2,048
  (32,668), and `append_pos_after` equals that plus `append_tokens`. Unlike measurement 01, there is no time
  rule.
- **Read class per C run:** device-observed when device bytes are at least 0.99 × (loads × 6,580,224);
  otherwise cache-contaminated. Over 600 GB of expert bytes is a recorded failure.
- **Predictions, reported against:**
  - prefill loads at most 7,168 (4 groups × 1,792). That is a ceiling, not a point: a count above it
    contradicts the design. The driver records cumulative loads after each decode token, so the figure is
    the first decode token's cumulative count: prefill plus that one token's loads (about 11 in 01);
  - cold first token near 34 s;
  - the append's loads below measurement 01's 1,642.
- **Reported, not gated:**
  - cold time to first token;
  - decode t/s over all 2,048 tokens, the first 256 and the last 256;
  - stops and loads per decode token;
  - the append's tokens, time to first token and loads;
  - post-append decode t/s, stops and loads per token;
  - logical and device bytes;
  - peak `phys_footprint`, RSS and Metal allocation, against the about 8 GiB budget. The group buffer is
    predicted to add 75.5 MB.

  C is shown as a range and a median, against R, against measurement 01's C, and against S5's targets for
  orientation only.
- **Speed claims** come only from runs classed device-observed, and only for this SSD.

## What would change the plan

Nothing in the run. Any result outside these rules is reported as it is. A second measurement needs its own
preregistration.
