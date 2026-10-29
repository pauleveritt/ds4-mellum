# S6 phase B: preregistration

Date: 2026-10-28, committed before any S6 GPU run. Branch `research/mellum-8gb-q4`, rebased onto
`codex/mellum-ssd-replay` at `1500320`. Design: [S6 spec](../specs/2026-10-28-mellum-s6-mxfp4-8gb-design.md),
goals 1, 2, 3 and 6. Plan: phase B's plan, written after this file.

**Honesty header.** This fixes, in advance, what phase B measures and how each result is judged.
- **Measured:** this 128 GiB M5 Max (macOS 27.2) only. Nothing here is an 8 GB result.
- **Not claimed:** answer quality, speed on any other Mac, or any rate without its `n`.
- **One run each:** no result is rerun or retried to change a verdict. A crashed run is recorded as
  crashed. A rerun after a fixed bug is a new run, reported beside the old one.

## Frozen inputs

- **Q8_0:** `~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf`, 12,925,417,760 B.
- **MXFP4:** `Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf` in the Hugging Face cache, snapshot `4159d684`,
  7,029,537,056 B. Its SHA-256 is recorded in the first run's capture.
- **Token files:** `tests/mellum_oracle_{short,1100,2200}.tokens` (29, 1,100 and 2,200 templated tokens),
  traced at up to 64 positions each by `tests/mellum_trace` in prefill mode, as the oracle does.
- **Long prompt:** the S4 frozen prompt (`2026-10-28-mellum-s4-measurement-prereg/prompt.txt`, SHA-256
  `2164763c…`, 30,620 tokens rendered), context 32,768, 2,048 greedy tokens.
- **Short prompt:** S4's regression command, `ds4 -c 8192 --temp 0 -n 256 -p "Explain how a hash map works,
  with an example in C."`.
- **llama.cpp:** Homebrew `llama.cpp` 0.7.0 (build 11669, `620d2225c`), driven through `libllama` by a small
  driver that feeds the same token ids and writes logits at the same positions in `tests/mellum_trace`'s
  file layout. No chat template is applied by llama.cpp; the ids are already templated.

## (a) Model-free Metal kernel tests

`make test-mellum-kernels` gains MXFP4 cases: the expert mid (gate, up, SwiGLU) and down, for decode rows
(T = 1) and tiled rows (T = 64 and 1,024), resident and through the four `_addr` entry points.
- **Pass:** every existing Q8_0 case passes unchanged. Every MXFP4 case selects the reference's experts,
  with weights within `MOE_W_TOL` (2.5e-6) and output within `MOE_TOL` (1e-5, max|d| / max|ref|). The
  reference is `mref_moe_row` at type 39 for decode rows, and a staged MXFP4 reference (the tile
  kernels' staging, read from `metal/qwen4.metal`) for tiled rows.
- **Repeated** with `DS4_METAL_DISABLE_METAL4=1`, under the same bounds.
- **Decides:** a failure stops phase B at the failing kernel. Nothing after this runs on MXFP4 until the
  failure is fixed.

## (b) Our logits against llama.cpp's, with Q8_0 as the control

Both engines run on both files, at all traced positions of the three token files. The metrics are
`tests/mellum_logits_compare.py`'s: per-position KL(llama.cpp ‖ ours), its mean, and top-1 agreement.
Near-ties count as agreeing, as in the oracle: where llama.cpp's top two are within 0.75 nats.
- **Control (Q8_0):** reported first. If the control fails the absolute bound below, llama.cpp and this
  engine disagree about Mellum itself. Then (b) is recorded as **uninformative** about MXFP4, and the
  MXFP4 verdict rests on (a) and the finite-logit and collapse checks.
- **Pass (MXFP4)**, all of:
  - top-1 agreement on at least 97% of traced positions;
  - mean KL ≤ 0.01 on each file;
  - worst position KL ≤ 1.0;
  - mean KL ≤ max(3 × the control's, the control's + 0.005) on each file.
- **Finite and not collapsed:** every traced logit is finite (the comparison refuses NaN and Inf). A
  resident MXFP4 run of 2,048 greedy tokens on the long prompt does not end in collapse. Collapse means
  a span of at least 512 tokens made of one block of period ≤ 64, repeated back to back.
- **Reported, not gated:** our MXFP4 against our Q8_0 on the same positions (mean and worst KL, top-1),
  beside JetBrains' 0.115 against BF16, which is a different reference and a different text.
- **Decides:** a pass is the S6 "not broken" verdict for resident MXFP4 on this machine. A fail stops
  S6 at diagnosis, before phase C.

## (c) Q8_0 no regression

S4's procedure, unchanged. Arm A is `1500320` built in a detached worktree; arm B is phase B's build,
streaming off. Six processes per arm on the short prompt, in A/B/B/A blocks (`regression_order(6)`), with
30 s rests between processes.
- **Pass:** B's median prefill t/s and median generation t/s are each at least 0.98 of A's
  (`regression_verdict`). Both arms' 256 generated tokens are identical. The three token files' traced
  logits from `tests/mellum_trace` are byte-equal between A and B (`cmp`).
- **Decides:** a fail is an anti-goal breach. It is fixed before anything is reported as S6 progress.

## (d) Goal 6: resident Q8_0 against resident MXFP4 on the M5 Max

Reported, not gated; it feeds a later product decision, and S6 claims nothing from it.
- **Long prompt:** three processes per file, in Q8/MX/MX/Q8/Q8/MX order, with 30 s rests. Reported:
  time to first token, prefill t/s, decode t/s over tokens 0–255 and over the last 256, and peak
  `phys_footprint`.
- **Short prompt:** six processes per file in A/B/B/A blocks. Reported: prefill and generation t/s.
- **Energy:** `powermetrics` needs root, which an unattended run does not have. Energy is **not measured**
  in this run; the report says so and leaves the method for the owner.
- Each figure is shown as a median and a range, with its `n`.

## (e) Streamed MXFP4 against resident MXFP4

Long prompt, 2,048 greedy tokens. `tools/mellum_s4_measure` resident (one run) against
`--stream-bytes 2684354560` (2.5 GiB, 815 slots; two runs) and `--stream-bytes 3221225472` (3 GiB, 979
slots; one run).
- **Pass:** every streamed run's 2,048 token ids equal the resident run's.
- **Reported, not gated:** decode loads per token at 2.5 GiB, beside the replay's estimate of 21.02 from
  the Q8_0 trace ([MXFP4 miss curve](2026-10-28-mellum-mxfp4-miss-curve.md)), and stops, loads and
  logical bytes. Device reads are not classified here: the page cache may serve them on 128 GiB, and the
  8 GB read behavior is phase C's.
- **Decides:** a pass lets phase C plan the iMac run on this code. A fail is a streamed-path defect, fixed
  before phase C.

## What would falsify this

- (b) passes, yet phase C's iMac logits disagree with this machine's resident MXFP4 beyond phase C's own
  preregistered bound. That would mean (b) tested a kernel path the M1 does not take.
- Loads per token at 2.5 GiB in (e) fall outside 10.5–42 (2× either way of the replay's 21.02). That is
  the spec's routing falsifier, here on a different prompt, so it is read as a warning, not a verdict.
- The Q8_0 control in (b) fails while MXFP4 passes. That would say the bound measures llama.cpp, not us.

## Risks

- llama.cpp's Mellum graph may differ from this engine's in ways unrelated to MXFP4: YaRN, sliding
  attention, KV precision. The Q8_0 control is there to show that, and it may make (b) uninformative.
- The 97% top-1 and 0.01 mean-KL bounds are judgment, set from this engine's oracle results on Q8_0
  (mean KL 1.3e-3 to 2.1e-3, top-1 64/65 to 65/65, MELLUM.md). They were not derived for a
  cross-implementation comparison.
- One machine, one OS. The Metal-4-disabled repeat in (a) covers only part of what an M1 changes.
- The long prompt is synthetic code text; the 8 GB user's Pi-shaped requests are phase C's.
