# S6 phase B (b): MXFP4 logits against llama.cpp, with Q8_0 as the control

Date: 2026-10-28, 22:34 EDT. Build `e930f6d` on `research/mellum-8gb-q4`. Machine: the M5 Max, 128 GiB,
macOS 27.2. Judged by [the preregistration](2026-10-28-mellum-s6-b-preregistration.md), item (b), which was
committed (`ff22dcc`) before any S6 GPU run.

**Honesty header.**
- **Measured:** this engine's logits and Homebrew llama.cpp 0.7.0's (build 11669, `620d2225c`) for the same
  templated token ids. That covers 157 traced positions over three files, on both the Q8_0 and the MXFP4
  artifact, one run each.
- **Not claimed:**
  - Answer quality.
  - Agreement with BF16. JetBrains' 0.115 / 85.6% is theirs, on other text.
  - Any machine but this one.
- **Comparison point:** llama.cpp is a second implementation, not ground truth. Q8_0 is there to measure
  how far the two engines differ on Mellum before MXFP4 enters.

## Method

`evidence/logits/run_b.sh` (in [`2026-10-28-mellum-s6-b-evidence/`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-s6-b-evidence))
runs, for each file and artifact:
- `tests/mellum_trace MODEL TOKENS prefill`, with `DS4_MELLUM_TRACE_AT` set to the oracle's positions (64
  spread over the file, or all of them below 64);
- `tools/mellum_llama_logits` on the same ids, as one `llama_decode`, all layers on Metal, flash attention
  `auto`, f16 KV cache, `n_ubatch` 512;
- `tools/mellum_logits_agree.py ours llama POSITIONS`: KL(llama ‖ ours), top-1, and the 0.75-nat near-tie rule.

Model SHA-256s are in `logits/provenance.txt`:
- Q8_0 `ea1e4ec7…`;
- MXFP4 `cf09584a…`, HF snapshot `4159d684`.

The raw logits (549 MB) are kept in `logits/raw/`, which git ignores. The JSON for every comparison and
every run's log are committed. Every logit was finite; the report refuses NaN and Inf.

## Results (n = traced positions)

| File | n | Q8_0 vs llama.cpp: mean KL, worst KL, top-1 equal | MXFP4 vs llama.cpp: mean KL, worst KL, top-1 equal (+ near-ties) |
|---|---:|---|---|
| `mellum_oracle_short` (29 ids) | 29 | 1.28e-3, 1.05e-2, 29/29 | 6.21e-4, 8.30e-3, 28/29 (+1) |
| `mellum_oracle_1100` | 64 | 8.59e-4, 1.66e-2, 62/64 (+2) | 2.04e-3, 1.76e-2, 62/64 (+2) |
| `mellum_oracle_2200` | 64 | 5.89e-4, 8.86e-3, 64/64 | 1.55e-3, 1.76e-2, 62/64 (+2) |

Reported, not gated: our MXFP4 against our Q8_0 at the same positions.
- **Mean KL:** 0.263, 0.148 and 0.131.
- **Worst KL:** 1.54, 0.741 and 0.489.
- **Top-1 equal:** 24/29, 51/64 and 47/64.
- **Agreement with near-ties counted:** 26/29, 61/64 and 62/64.

## Verdict by the preregistration

- **Control:** passes the absolute bound on all three files. So (b) is informative about MXFP4.
- **MXFP4:** **pass.** Agreement, with near-ties counted as the preregistration says, is 29/29, 64/64 and
  64/64 (100%, at least 97% required). Mean KL is at most 2.04e-3 (at most 0.01). Worst KL is 1.76e-2 (at
  most 1.0). Each file's mean KL is within max(3× control, control + 0.005): 6.2e-4 ≤ 6.3e-3,
  2.0e-3 ≤ 5.9e-3 and 1.6e-3 ≤ 5.6e-3.
- **Strict top-1, stated plainly:** MXFP4's strict top-1 equality is 96.6–96.9%, below 97%. Each of the
  five differing positions is a near-tie under the preregistered rule.
- **Finite logits:** held. The 2,048-token collapse check is run (e).

Under the preregistration, a pass is S6's "not broken" verdict for resident MXFP4 on this machine. The
kernel-level evidence agrees:
- the decode driver is within 6.4e-7 of the double reference on a model-free MXFP4 arena;
- the tiled experts are no farther from their staged reference than Q8_0's (1.74e-4 against 1.87e-4);
- `ds4_test --mellum-moe` on the real MXFP4 file passes, with exact rows within 7.7e-7 and tiled rows
  within 3.7e-5.

Our MXFP4-to-Q8_0 distance (mean KL 0.13–0.26) is about what JetBrains report for MXFP4 against BF16 (0.115).
Theirs is a different reference and different text, so this is a sanity check, not a comparison.

## What would falsify this

- An M1, or Metal 4 off at full model scale, gives MXFP4 logits outside phase C's bound against this
  machine's. Then this pass covered only the M5 kernel path.
- A longer context (32K, S6's plan) shows MXFP4 drifting from llama.cpp where Q8_0 does not. The longest
  file here is 2,200 tokens.
- llama.cpp's Mellum graph shares a defect with ours. Then agreement would hide it. The double-reference
  kernel tests are the independent check on the experts; nothing here independently checks attention.

## Risks

- 157 positions over three prompts, one run each. The near-tie rule turns five strict top-1 differences
  into agreements, and the 97% bound sits just above the strict rate.
- Both engines run on the same GPU and Metal compiler, so shared toolchain behavior would cancel out.
- The bounds were judgment set from this engine's Q8_0 oracle, as the preregistration says, not derived
  for a cross-implementation comparison.
