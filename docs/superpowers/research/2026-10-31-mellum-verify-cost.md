# What a k-row verify costs Mellum 2.1, and why

**Honesty header.** Measured: the wall time of one k-row forward against k decode steps (k = 1, 2, 3, 4, 6, 8) at
contexts 2,048, 8,192 and 32,768, both published quants, n = 30 per arm per point, on one M5 Max (128 GB, AC,
High Power, no thermal warning before or after any process); the per-kernel GPU time of the same forwards from
the engine's encoder timeline (n = 6 per k, separate runs); and the number of distinct routed experts that k
consecutive tokens select, from traced MoE inputs of five token files (64 positions each, Q8_0). Arithmetic, not
measured: the "bytes-bound" estimates below. Conjecture: what this means for prompt lookup and an MTP head; no
speculative decoder was built or run.

The question (README, decode "Next"): speculation needs a cheap multi-row verify, and phase 8b measured a two-row
verify at 1.29-1.44 decode steps (`MELLUM.md`, B0). Why is it not close to 1, and how does it grow with k?

## Commands

```
# the screen: per quant, one process per context, 45 s cooling, then a timeline run (N=4) at 2K and 32K
docs/superpowers/research/2026-10-31-mellum-verify-cost/screen.sh        # calls tests/mellum_bk.sh
# per-kernel anatomy (ms per forward)
python3 tools/mellum_bk_anatomy.py <(xz -dc docs/superpowers/research/2026-10-31-mellum-verify-cost/Q8_0-timeline/timeline_2048.txt.xz) 4
# distinct experts (each file's header line has its trace command)
uv run --no-project --with gguf python -I tools/mellum_expert_union.py MODEL PREFIX POS0 64
```

Evidence: `2026-10-31-mellum-verify-cost/` (`Q8_0/`, `MXFP4_MOE/`: `summary.txt`, the test's output, power and
thermal state; `*-timeline/`: the timelines, xz; `expert_union/`; `failed-1/`, a first attempt that stopped on a
runner bug before measuring). Engine: `research/verify-cost` at the screen's `setup.txt` head. Every k-row
forward's last-row logits equal its k decode steps' bytes (0 words differ, every point): a verify below 32 rows
can accept exactly what decode would.

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

k = 1 is 0.998-1.001 everywhere. The 2-row figures agree with phase 8b's B0 (1.29-1.44, which also counts a second
LM head). At 32K the decode step itself drifted within the process (Q8_0 9.27 ms at k = 1-2, 10.2-10.7 ms at
k = 3-8; MXFP4 8.0 to 9.4 ms); each ratio uses its own k's alternated decode steps.

## Where the time goes

Per-kernel GPU ms per forward, Q8_0 at 2,048 (`tools/mellum_bk_anatomy.py`, n = 6 forwards per k; the timeline
adds a little to every dispatch):

| kernel | 1 row | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| experts, `moe_mid` | 2.247 | 3.174 | 4.359 | 5.476 | 7.713 | 9.734 |
| experts, `moe_down` | 1.105 | 1.621 | 2.250 | 2.840 | 4.024 | 5.156 |
| Q8_0 projections (`mul_mv_q8_0`) | 2.840 | 2.886 | 3.575 | 4.276 | 5.641 | 7.018 |
| sliding attention (`attn_swa`) | 0.851 | 0.859 | 1.169 | 1.378 | 1.787 | 1.949 |
| total | 8.708 | 10.639 | 13.619 | 16.379 | 21.784 | 26.677 |

At 32,768 one more kernel dominates: the multi-row full attention (`attn_decode_rows_npt4`) takes 3.275 ms at 2 rows
and 11.223 at 8, against 1.82 ms for decode's one-row full attention (`attn_decode_npt4` + `attn_merge_wide`).
MXFP4 shows the same shape (its tables: the same command on `MXFP4_MOE-timeline/`).

Distinct experts per layer that k consecutive tokens select (mean over windows and the 28 layers; 8 for one token):

| file | k = 2 | 3 | 4 | 8 |
|---|---|---|---|---|
| held-out code | 11.84 | 14.74 | 17.16 | 23.80 |
| held-out long C | 11.74 | 14.40 | 16.72 | 22.82 |
| held-out tool | 11.79 | 14.37 | 16.50 | 22.80 |
| held-out Unicode | 11.55 | 14.26 | 16.43 | 22.59 |
| oracle prose, 2,200 | 11.10 | 13.44 | 15.28 | 20.49 |

## Findings

1. **At 2 rows the cost is the MoE's.** Two tokens select 1.39-1.48x the experts of one (11.1-11.8 of 8), and the
   expert kernels take 1.43x (Q8_0, 2K: 4.795 against 3.352 ms). The projections batch (1.02x). The other 2-row
   cost is small kernels the one-row path fuses (about 0.4 ms).
2. **From 3 rows the expert and projection kernels do per-row work.** At 4 rows the experts take 2.48x one row's
   while the tokens select 1.91-2.15x the experts; at 8 rows 4.44x against 2.56-2.98x. The projections go from
   1.02x at 2 rows to 2.47x at 8. These are the exact-rows paths, kept below 32 rows so a chunk's bits equal its
   decode steps' (`mellum_proj`, `ds4_mellum_graph.inc`).
3. **At long context the multi-row full attention reads the KV cache per row**: 1.8x decode's at 2 rows, 6.2x at
   8 (Q8_0, 32K). This is why the 32K ratios are highest.
4. **Arithmetic, not measured:** if the expert time grew only with the distinct experts and everything else stayed
   at its 2-row cost, a 4-row verify would cost about 1.45 decode steps and an 8-row one about 1.75 (Q8_0, 2K,
   middle union values 11.7 / 16.5 / 22.8, timeline totals), against 1.88 and 3.06 in the same timeline (1.91 and
   3.17 untimed).

## What it means (conjecture)

- Prompt lookup with one-id drafts walked 1.07-1.43 ids per step on these files (`MELLUM.md`, D3); a 2-row verify
  costs 1.26-1.45 steps, so it would gain only on long, repetitive C source.
- An MTP head drafting one token needs an acceptance above about 0.3 to break even at 2K (1.26 plus the head's own
  cost), less than MTP heads usually reach; drafting more than one needs the batched, bit-identical kernels of
  finding 2 (and, at long context, a multi-row full attention that reads the cache once).

## What would falsify this

- A rerun of `tests/mellum_bk.sh` on the same machine and head giving ratios outside these by more than a few
  percent, or logits that differ.
- Expert-kernel time at 2 rows not tracking the distinct-expert count in another set of token files.
- A k-row forward through batched, bit-identical kernels costing more than the measured exact-rows path.

## Risks

- One machine (M5 Max, Metal 4 tensor tiles); the exact-rows kernels may behave differently on an M1.
- One text (`speed-bench/promessi_sposi.txt`) for the timings; routing overlap depends on the text (the oracle prose
  overlaps most).
- The routes are computed on the CPU from traced MoE inputs; a near tie could select differently from the GPU.
- The timeline's pass boundaries inflate every dispatch; its totals (8.7 ms) are above the untimed step (7.75 ms).
