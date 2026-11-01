# What a k-row forward costs Mellum 2.1, and why

**Honesty header.** Measured, on one M5 Max (128 GB, AC, High Power, no thermal warning before or after any
process): the wall time of one k-row forward against k decode steps (k = 1, 2, 3, 4, 6, 8) at contexts 2,048, 8,192
and 32,768, both published quants, n = 30 per arm per point; the same at 32,768 with each k in its own process and
with k descending; the per-kernel GPU time and GPU span of the same forwards from the engine's encoder timeline
(n = 6 forwards per k, separate runs); and the number of distinct routed experts that k consecutive tokens select,
from traced MoE inputs of five token files (64 positions each, Q8_0). **Not measured:** a full verify. The forward
measured here computes the LM head for its last row only; a verify needs logits for all k rows (below, "What a
verify adds"). Inference: that the extra experts explain the 2-row cost (kernel time and expert counts were measured
on different texts). Arithmetic: the estimates in finding 4. Conjecture: what this means for prompt lookup and an
MTP head; no speculative decoder was built or run.

The question (README, decode "Next"): speculation needs a cheap multi-row verify, and phase 8b measured a two-row
forward plus a second head at 1.29-1.44 decode steps (`MELLUM.md`, B0). Why is it not close to 1, and how does it
grow with k?

## Commands

```
# the screen: per quant, one process per context, 45 s cooling, then a timeline run (N=4) at 2K and 32K
docs/superpowers/research/2026-10-31-mellum-verify-cost/screen.sh            # calls tests/mellum_bk.sh
# 32K: each k in its own process, then one process with k descending (DS4_TEST_MELLUM_BK_K)
docs/superpowers/research/2026-10-31-mellum-verify-cost/32k-diagnosis/screen.sh
# per-kernel anatomy and GPU spans
python3 tools/mellum_bk_anatomy.py <(xz -dc docs/superpowers/research/2026-10-31-mellum-verify-cost/Q8_0-timeline/timeline_2048.txt.xz) 4
# distinct experts (each file's header line has its trace command)
uv run --no-project --with gguf python -I tools/mellum_expert_union.py MODEL PREFIX POS0 64
```

Evidence: `2026-10-31-mellum-verify-cost/` (`Q8_0/`, `MXFP4_MOE/`: `summary.txt`, the test's output, power and
thermal state; `*-timeline/`: the timelines, xz; `32k-diagnosis/`; `expert_union/`; `failed-1/`, a first attempt
that stopped on a runner bug before measuring). In every run the k-row forward's last-row logits equal the k-th
decode step's bytes (0 words differ, k ≤ 8).

## The cost of a k-row forward, in decode steps

Median k-row forward / median one decode step (the same process, alternated D C C D), n = 30 per arm:

| quant | context | k = 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| Q8_0 | 2,048 | 1.256 | 1.589 | 1.910 | 2.571 | 3.173 |
| Q8_0 | 8,192 | 1.284 | 1.635 | 1.991 | 2.661 | 3.153 |
| Q8_0 | 32,768 | 1.414 | 1.728 | 2.132 | 2.888 | 3.754 |
| MXFP4 | 2,048 | 1.313 | 1.675 | 2.026 | 2.758 | 3.446 |
| MXFP4 | 8,192 | 1.352 | 1.734 | 2.108 | 2.866 | 3.507 |
| MXFP4 | 32,768 | 1.453 | 1.851 | 2.260 | 3.099 | 3.889 |

k = 1 is 0.998-1.001 everywhere. The timeline runs' GPU spans give the same ratios (Q8_0 2K: 1.257, 1.568, 1.899,
2.551, 3.158; 32K: 1.429, 1.827, 2.235, 3.051, 3.757), each against the decode steps of its own k's rounds.

**32K, unexplained.** At 32,768 the decode step's wall time depends on what the process ran before it. With every
k in one process (ascending), Q8_0's step is 9.27 ms at k = 1-2 and 10.2-10.7 ms from k = 3. With each k in its own
process it still rises with k (9.28 ms at k = 1, 10.34 at k = 8), and with k descending it is 10.8-11.0 ms from the
start. The GPU span of a decode step does not show the rise (Q8_0 8.98-9.12 ms through k = 6). So the 32K ratios
vary with the arrangement: Q8_0 at 8 rows 3.754 / 3.522 / 3.510 (ascending, own process, descending), MXFP4
3.889 / 3.751 / 3.854; at 4 rows 2.132 / 2.171 / 2.114 and 2.260 / 2.306 / 2.240. Against a decode step from a
process that ran nothing larger (9.27 ms), Q8_0's 8-row forward (36.4 ms, own process) is 3.9 steps.

## Where the time goes

Q8_0 at 2,048 (`tools/mellum_bk_anatomy.py`, n = 6 forwards per k; "1 row" is the decode steps of the k = 1 rounds).
Per kernel: summed dispatch time per forward, in ms. Dispatches can overlap, so the GPU span (last rows) is the
total:

| | 1 row | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| experts, `moe_mid` | 2.175 | 3.174 | 4.359 | 5.476 | 7.713 | 9.734 |
| experts, `moe_down` | 1.094 | 1.621 | 2.250 | 2.840 | 4.024 | 5.156 |
| Q8_0 projections and the LM head (`mul_mv_q8_0`) | 2.555 | 2.886 | 3.575 | 4.276 | 5.641 | 7.018 |
| sliding attention (`attn_swa`) | 0.828 | 0.859 | 1.169 | 1.378 | 1.787 | 1.949 |
| GPU span, the forward | 7.402 | 9.566 | 12.257 | 14.776 | 19.827 | 24.355 |
| GPU span, a decode step in the same rounds | | 7.612 | 7.814 | 7.779 | 7.774 | 7.713 |

The LM head is one `mul_mv_q8_0` dispatch of 0.417 ms per forward (the last dispatch of every decode step, n = 197).
At 32,768 one more kernel dominates: the multi-row full attention (`attn_decode_rows_npt4`) takes 3.275 ms at 2 rows
and 11.223 at 8, against 1.806 ms for decode's one-row full attention (`attn_decode_npt4` + `attn_merge_wide`).
MXFP4 shows the same shape (the same command on `MXFP4_MOE-timeline/`).

Distinct experts per layer that k consecutive tokens select (mean over windows and the 28 layers; 8 for one token):

| file | k = 2 | 3 | 4 | 8 |
|---|---|---|---|---|
| held-out code | 11.84 | 14.74 | 17.16 | 23.80 |
| held-out long C | 11.74 | 14.40 | 16.72 | 22.82 |
| held-out tool | 11.79 | 14.37 | 16.50 | 22.80 |
| held-out Unicode | 11.55 | 14.26 | 16.43 | 22.59 |
| oracle prose, 2,200 | 11.10 | 13.44 | 15.28 | 20.49 |

## Findings

1. **At 2 rows most of the extra time is the experts, in proportion to the extra experts (inference).** The expert
   kernels take 1.47x one row's (Q8_0 2K: 4.795 against 3.269 ms), and two consecutive tokens select 1.39-1.48x the
   experts of one (11.1-11.8 of 8). The projections without the head go from 2.138 to 2.469 ms (1.15x). The rest is
   small kernels the one-row path fuses. The timings ran on one text (`speed-bench/promessi_sposi.txt`), the routes
   on five others, so the match is a correlation.
2. **From 3 rows the expert and projection kernels do more than the extra bytes.** At 4 rows the experts take 2.54x
   one row's while the tokens select 1.91-2.15x the experts; at 8 rows 4.56x against 2.56-2.98x. The projections
   without the head take 1.81x at 4 rows and 3.09x at 8. These are the exact-rows paths, kept below 32 rows so a
   chunk's bits equal its decode steps' (`mellum_proj`, `ds4_mellum_graph.inc`).
3. **At long context the multi-row full attention reads the KV cache per row**: 1.81x decode's at 2 rows, 6.21x at
   8 (Q8_0, 32K, GPU time). This is why the 32K ratios are the highest.
4. **Arithmetic, not measured:** if the expert time grew from its 2-row value only with the distinct experts
   (11.7 / 16.5 / 22.8 at k = 2 / 4 / 8, the middle files) and everything else stayed at its 2-row cost, the
   forward's GPU span would be 11.53 ms at 4 rows and 14.12 at 8: 1.48 and 1.83 decode steps, against the measured
   1.90 and 3.16 (Q8_0 2K, spans).

## What a verify adds

A verify needs logits for every row. The forward measured here computes the LM head for its last row only (the
engine's chunk forward refuses all-rows logits, `mellum_graph_forward_tokens`). Heads run one at a time would add
0.417 ms of GPU time each, k - 1 of them: about 0.05 decode steps per extra row at 2K, so an 8-row verify would be
about 3.5 steps rather than 3.16 (spans). A batched head would read the head's weights (98,304 x 2,304 Q8_0, about
0.24 GB) once. The byte-equality above covers the last row; that the other rows' logits would equal decode's is
expected from the same exact-rows kernels and was measured at 2 rows only (B0, phase 8b).

## What it means (conjecture)

- Prompt lookup with one-id drafts walked 1.07-1.43 ids per step on these files (`MELLUM.md`, D3); a 2-row verify
  (B0, with its second head) costs 1.29-1.44 steps, so it would gain only on long, repetitive C source.
- An MTP head drafting one token breaks even when its acceptance exceeds the verify's extra cost plus the draft's:
  about 0.29 at 2K (B0) plus the head's own cost. Drafting more than one token needs the batched, bit-identical
  kernels of finding 2 and, at long context, a multi-row full attention that reads the cache once.

## What would falsify this

- A rerun of `tests/mellum_bk.sh` on the same machine and head giving 2K or 8K ratios outside these by more than a
  few percent, or logits that differ.
- Expert-kernel time at 2 rows not tracking the distinct-expert count on the timing text itself.
- A k-row forward through batched, bit-identical kernels costing more than the measured exact-rows path.

## Risks

- One machine (M5 Max, Metal 4 tensor tiles); the exact-rows kernels may behave differently on an M1.
- The 32K decode-step rise is unexplained; the 32K ratios carry its spread (above).
- Routing overlap depends on the text (the oracle prose overlaps most); the timing text's own was not traced.
- The routes are computed on the CPU from traced MoE inputs (the prefill path's); a near tie could select
  differently from the GPU's decode.
- The timeline's pass boundaries change dispatch timing; its span ratios agree with the untimed ones within 0.021
  at 2K, and within 0.10 at 32K (where the untimed decode step rose).
- The `ds4_test` binary is not hashed in `setup.txt`, only the repository head.
