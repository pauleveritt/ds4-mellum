# Mellum 2.1 on MXFP4 experts: what the Metal path needs (code read)

Date: 2026-10-28. Branch `research/mellum-8gb-q4` at `01521b3` (= `mellum`) for the resident path;
`codex/mellum-ssd-replay` read at `3addce0` (it moved past `9e9757f` during the read) for the S4
streamed path. Line numbers are for those commits; the SSD branch's `ds4.c` is offset by about +24
to +70 lines from `mellum`'s.

**Honesty header.** This is a code read. Nothing was built, run or tested; no GPU was used. No
speed, quality or routing agreement is measured. "Reusable" means the code reads as generic in the
weight type; it is not shown to produce correct Mellum output. The artifact is JetBrains'
`Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf`. Its layout comes from parsing the GGUF header, not
from running it. Non-expert tensors are Q8_0 and F32, as in the Q8_0 artifact. The 84 routed-expert
tensors are MXFP4: gate/up [2304, 896, 64] and down [896, 2304, 64]. Each gate, up or down slice of
one expert is 1,096,704 B, so a whole expert is 3,290,112 B (Q8_0: 6,580,224).

## Summary

Mellum already runs its routed experts through the **Qwen4 MoE kernels**, and those take a weight
type and already handle MXFP4 (type 39). Qwen3.8 Flash Next uses them in production for MXFP4
*down* experts. Mellum's own code pins Q8_0 at about eight host-side gates. The resident path looks
like small host changes and no new Metal kernel. The S4 streamed path has four more `!= 8u` gates
and a hard-coded Q8_0 expert size used for its budget. The DeepSeek MXFP4 kernels in `moe.metal`
are not the right base.

## Resident path changes

Gates that refuse a non-Q8_0 expert (on `mellum` `01521b3`):

| Where | What |
|---|---|
| `ds4.c:5706-5707` | The layout comment says experts are Q8_0, "the only quantization the Mellum kernels exist for". |
| `ds4.c:5751-5753` | `tensor_expect_layout(ffn_{gate,up,down}_exps, Q8, ...)`: the load-time refusal. |
| `ds4.c:7339-7341` | `n_ff_exp % 32`, worded "(Q8_0 blocks)". 896 is also a whole number of MXFP4 blocks (32 values), so only the wording changes. |
| `ds4.c:60547-60549` | `mellum_graph_moe` returns false unless all three expert tensors are Q8_0. |
| `ds4.c:60556` | `p.expert_type = DS4_TENSOR_Q8_0` is hard-coded; it should come from the tensor's type. |
| `ds4_gpu.h:3841-3854` | The contract says experts are Q8_0 (`expert_type 8`). |
| `ds4_metal.m:50982` | `mellum_moe` refuses `p->expert_type != 8u`. |

Compute paths, all through the Qwen4 kernels:

- **Decode and short prefill (T < `MELLUM_TILED_MIN_ROWS`):** `mellum_moe` (`ds4_metal.m:50978`).
  It calls `ds4_gpu_qwen4_moe_mid_tensor` (`:49618`) and `ds4_gpu_qwen4_moe_down_tensor` (`:49677`),
  with no shared expert (`shared_type = UINT32_MAX`).
  - The kernels are `kernel_qwen4_moe_mid` (`metal/qwen4.metal:2766`) and `kernel_qwen4_moe_down`
    (`:3047`). Both branch on the type in `qwen4_row_dot` (`:2600`); its MXFP4 branch is at `:2614`.
  - The activation is `qwen4_silu(g) * u` (`:2800`): plain SwiGLU with no clamp. That matches the
    Q8_0 path Mellum runs today, because it is the same kernel.
  - On M5, MXFP4 down already selects `kernel_qwen4_moe_down_mxfp4_pf` (`ds4_metal.m:49708-49715`).
    On M5 and M3 Ultra, function-constant specialization turns on for type 39 (`:48340-48347`).
- **Prefill (T >= 32, tiled):** `ds4_gpu_qwen4_moe_mm_mid_tensor` and `_mm_down_tensor`
  (`ds4_metal.m:49971`, `:50053`).
  - Both accept type 39 (`:49989`, `:50072`) with `in_dim % 64` and `ff_dim % 64`; 2304 and 896
    pass.
  - MXFP4 defaults to the level-2 tensor tiles (`:49895-49903`). `ds4_gpu_qwen4_moe_q8_tiles`
    (`:49890`, set by Mellum at `ds4.c:60568`) affects type 8 only, so for MXFP4 it is inert.
- **CPU path:** none. Mellum refuses any backend but Metal (`ds4.c:73211`). The CPU dot
  `ds4_vec_dot_mxfp4_f32` exists (`ds4.c:4301`) but is not on Mellum's path.
- **MTP:** not on the engine path for Mellum (`ds4.c:1076` excludes it), so nothing changes there.

Byte layout: `qwen4_expert_row_bytes` (`ds4_metal.m:48483`) gives MXFP4 rows of `in_dim/32*17`
bytes: 1,224 for gate/up and 476 for down. Times the rows, that is 1,096,704 B each, matching the
header. E8M0 decoding (`metal/moe.metal:24-27`, 2^(e-127)) together with the E2M1 table
(`:19-22`) equals ggml's half-scaled int table, so the same GGUF bytes decode alike. This is
checked by algebra only.

## Streamed path changes (S4, `codex/mellum-ssd-replay` at `3addce0`)

- **Q8_0 gates.**
  - The same graph gate as resident, at `ds4.c:60600-60602`, and `p.expert_type` at `:60611`.
  - The four address-table entry points refuse `weight_type != 8u`:
    - `ds4_gpu_qwen4_moe_mid_addr_tensor` (`ds4_metal.m:49890`, check at about `:49902`);
    - `_down_addr_tensor` (`:49921`, about `:49932`);
    - `_mm_mid_addr_tensor` (`:50474`);
    - `_mm_down_addr_tensor` (`:50487`).
  - Their kernels (`kernel_qwen4_moe_{mid,down}_addr` and the `_addr` template instances of the mm
    and tensor-tile kernels, from the `metal/qwen4.metal` diff) reuse `qwen4_row_dot` and the
    type-driven dequantizers. They read as type-generic; the gates are host policy.
- **Expert size constant.**
  - `DS4_MELLUM_EXPERT_BYTES 6580224` (`ds4_mellum_stream.h:11`) turns a byte budget into slots
    (`ds4_mellum_stream.c:5-8`) and sets the 64-slot minimum (`:14-21`).
  - The engine uses it at `ds4.c:74160` and `:74168`.
  - With MXFP4 it would give half the true slots for a byte budget. It must come from the
    artifact (gate×2 + down from the tensor types).
- **Already generic.**
  - The per-load sizes come from `streaming_layer_gate_down_expert_bytes` (`ds4.c:5143`) through
    `routed_expert_row_bytes`, which knows MXFP4 (`mellum` `ds4.c:5049-5063`).
  - The slab slot is `gate*2 + down` rounded to a page (`ds4_metal.m:14199-14227`). That is
    3,293,184 B for MXFP4 (201 × 16 KiB) against 6,586,368 B for Q8_0.
  - Up and down sit at 32-byte-aligned offsets inside a slot (1,096,704 and 2,193,408).
- **Tools and tests that pin the Q8_0 geometry** (the replay and read service, not the engine):
  - `tools/mellum_ssd_replay.py:31,210` (refuses non-6,580,224 traces);
  - `tools/mellum_expert_index.py:11`, `mellum_expert_pack.py:20-22`;
  - `tools/mellum_read_{plan,io,run}.c`, `mellum_stage_io.c` (`max_job_bytes > 6580224`);
  - `tests/test_mellum_stream.c:26,50,52` and the Python replay/read tests.
  - Re-running the miss curve at MXFP4 needs a second geometry, or an `expert_bytes` parameter.

## Reusable kernels vs new kernels

- **Reused, no new Metal:**
  - decode mid/down and the M5 MXFP4 prefetch down;
  - the tiled mid/down, simdgroup and tensor tiles;
  - the `_addr` streamed variants, if their gates are opened.
  - MXFP4 **down** rows have production use on Qwen3.8 Flash Next (`docs/QWEN38_FLASH_NEXT.md:31`:
    "Q4_K gate/up and MXFP4 down").
  - MXFP4 **gate/up** through these Qwen4 kernels has, as far as this read finds, no production
    user: the `qwen4_row_dot` and mm dequant branches for type 39 run there for the first time.
- **Not reusable:**
  - `metal/moe.metal`'s `kernel_mul_mv_id_mxfp4_*` (`:4589-4900`, `:6403`) are DeepSeek V4's
    routed-MoE kernels, with `ds4_metal_args_mul_mv_id`.
  - They fuse the route weight into mid, apply DeepSeek's SwiGLU clamp (`act.clamp_value`,
    `:4577-4580`), and include fixed 6-slot `slots6`/`sum6` variants.
  - `kernel_test_mxfp4_down_half_lut` (`:2512`) is a test-only raw-bit oracle for the half-scale LUT
    (`metal/generate_mxfp4_half_lut.py`).
- **Rough size:** about 8 resident and about 7 streamed host edits, mostly lifting `== Q8_0` into
  "Q8_0 or MXFP4, uniform across the three". Then test changes: the refusal tests flip to accepting
  MXFP4, and the double reference gains an MXFP4 dequantizer. No new kernel, unless MXFP4 gate/up
  turns out to need one for speed.

## Other type-sensitive pieces

- **Refusal tests that would flip:**
  - `tests/test_mellum_config.c:1127-1128` (`MF_DOWN_Q4K`, `MF_UP_Q4K` "not Q8_0");
  - `tests/test_mellum_kernels.c:1650-1654` (Q4_0 and Q4_K refused);
  - per binding rule 4, keep the refusals for non-MXFP4 types next to a new MXFP4 success test.
- **Q8_0-only references:**
  - `tests/mellum_moe_ref.inc:4-15` (`MREF_Q8_BLOCK_BYTES`, Q8 row sizes);
  - the random-Q8_0 fixtures in `tests/test_mellum_kernels.c:1284-1321`.
  - `tests/mellum_fixture.h:22` already has `MF_Q4_K` and needs an MXFP4 (39) type.
- **The real-weights test:** `tests/ds4_test.c --mellum-moe` (`:996-1244`) compares against the
  Q8_0 double reference.
- **E1:** Mellum's Q8_0 tensor-tile opt-in (`MELLUM.md:746-787`) does not apply to MXFP4. MXFP4
  uses the default Qwen4 levels, which were tuned for Flash Next's shapes, not Mellum's.
- **JetBrains' Q4_K_M is a much bigger port** (not this note's target):
  - its down experts are Q8_0 or Q5_0 per layer, because 896 is not a multiple of 256, so Q4_K
    cannot hold a down row;
  - Q5_0 (type 6) is not in `qwen4_expert_row_bytes`;
  - its attention tensors are Q4_K/Q6_K, where Mellum's projections are Q8_0-only.

## Open questions

1. Does the type-39 gate/up path in `kernel_qwen4_moe_mid`, and the mm mid tiles, give correct
   Mellum output? It is untested here and unused in production.
2. Should the artifact's expert type be uniform across the three tensors and all layers (simplest;
   true of this file), or per tensor?
3. Does `DS4_MELLUM_EXPERT_BYTES` become a per-engine value, and does the 64-slot minimum stay in
   slots (0.20 GiB at MXFP4) rather than bytes?
4. Are the Qwen4 MXFP4 tuning defaults (tails, specialization, prefetch, NAX level) right for
   Mellum's 2304×896 shapes? That is a measurement question, and waits for the GPU.
5. Routing agreement and quality against Q8_0: JetBrains reports KLD 0.115 and 85.6% top-token match
   on Wikitext-2. That is their figure, not ours.

## What would falsify this

- A Mellum code path that dispatches experts outside `mellum_moe` or `mellum_graph_moe`, so it is
  missed by this read and stays Q8_0-only.
- The type-39 branches in `qwen4_row_dot` or the mm dequantizer turning out to assume Flash Next's
  dims (for example, a hidden-size constant).
- An E8M0/E2M1 decode mismatch with ggml's MXFP4. Algebra says they agree; a model-free test on a
  known block would settle it.
- Any `_addr` kernel instance specialized only for Q8_0 behind the host gate.

## Risks

- First production use of MXFP4 gate/up rows in the Qwen4 kernels; numerical bugs would show as
  quality loss and be hard to separate from quantization loss.
- MXFP4's published quality is the weakest of JetBrains' four quants.
- The SSD branch is moving (another session committed during this read); its line numbers will drift.
- The replay and read tools are Q8_0-only, so the MXFP4 miss-curve figures are arithmetic on the
  Q8_0 trace until a tool takes an expert size.
