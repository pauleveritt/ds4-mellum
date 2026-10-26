# Mellum 2.1 on ds4

Branch `mellum` ports JetBrains' Mellum 2.1 (12B-A2.5B, thinking) onto
antirez/ds4, starting at `5379c1c`, in small commits. This file records what
works, what was measured (each number with its command and n), what diverges
from the model's training, and what is not covered.

## Status

Phases 1 (recognize and load), 2 (text) and 3a up to the oracle (steps 15-20)
are done: a `mellum` GGUF is detected, validated, bound and planned; the
`mellum2` pre-tokenizer, the stop classes, the chat prefix and the sampling
defaults match the model's own tokenizer and template; the Metal forward pass
gives logits byte-identical to the old port's on three prompts in prefill and
decode, and 64 equal greedy tokens (Checkpoint 2). The forward pass is reachable
only from the trace driver (`tests/mellum_trace`, while `DS4_MELLUM_TRACE` is
set): every other forward entry still refuses by name until step 22. Session
save/load refuses by name, and ds4-server and ds4-agent refuse a Mellum engine
at startup (no Hermes tool syntax yet).

Model-free tests: `make test-mellum`. With `DS4_TEST_MELLUM_MODEL` set,
`./tests/test_mellum_config` also validates, binds and renders against the real
GGUF, `./tests/test_mellum_tokenizer` checks token ids through its vocabulary,
and `./ds4_test --mellum-session` checks session allocation against the plan.

Generated test data (each generator's command is in its docstring; they ran on
the local Mellum2.1-12B-A2.5B-Thinking MLX snapshot, whose `tokenizer.json`
and `chat_template.jinja` are byte-identical to the Hugging Face snapshot
`6f239041` the old port used, by the sha256 its fixtures recorded):
`ds4_mellum_unicode.inc` (`tests/gen_mellum_unicode.py`),
`tests/mellum_tokenizer_cases.inc` (`tests/gen_mellum_tokenizer_cases.py`),
`tests/mellum_chat_ids.inc` (`tests/gen_mellum_chat_ids.py`).

## Changes to upstream code other models run

- The Qwen3.8 attention decode kernels (`metal/qwen4.metal`) gained an
  `ungated` word in place of `pad0` and a branch in the decode tile and the two
  merge kernels, for Mellum's ungated full layers (step 17). The gated path is
  unchanged: a probe hashing the raw output of `ds4_gpu_qwen4_attn_decode_tensor`
  and `ds4_gpu_qwen4_attn_decode_rows_tensor` on fixed inputs (head dims 128
  and 256, 26 to 2,100 keys, one and two rows, both merge kernels, with and
  without partials: 60 cases) gave identical bytes before and after, and
  `make test-qwen4-kernels` passes. Flash Next's decode speed was not
  re-measured: there is no Flash Next GGUF on this machine.

## Commits

Phase 0: `15824be` baseline, `57278db` dependency audit.

Phase 1:
- `e8ba0ac` recognize `general.architecture = mellum`; refuse it by name.
- `5109a6d` validate the hyperparameters, each refusal naming its key.
- `34b5d44` bind the tensors by name and check their Q8_0 layout.
- `f4d1ef1` memory plan and session allocation; the forward entries refuse.

Phase 2:
- `e398212` generated Unicode classes, with the generator (step 7).
- `5e4884d`, `a7f0b4f` tokenizer cases from the model's own tokenizer; the
  `mellum2` pre-tokenizer (step 8). The old port's batch drivers
  (`dump_segments_batch`, `dump_tokens_batch`) are not ported: a generated
  fixture and `tests/test_mellum_tokenizer` replace them, so step 10 (the
  drivers refusing a signed record count) has nothing to apply to; upstream's
  tokenizer entry points take C strings.
- `79922e5` special tokens classed, corpus markers are content (step 9; it
  also covers step 12: `<|im_end|>` is EOS, id 0 EOS-alt, `</tool_call>`
  stops nothing).
- `8021751`, `20a696c` chat ids from the model's own template; Mellum's own
  system turn and assistant prefix (step 11).
- `dd0d997` sampling defaults from the engine, read by the CLI, agent and
  server (step 13).
- `a74bafe` session save and load refuse Mellum by name (step 14).

Phase 3a (forward pass on Metal, identical to the old engine):
- `f58b776` q/k norm and RoPE kernel, YaRN on full layers only (step 15).
- `bb2a2f3` ring commit and sliding-window attention kernels (step 16).
- `f0332e3` ungated full attention on upstream's qwen4 decode kernels (step 17).
- `2712e56` MoE, top-8 of 64 routed experts, no shared expert (step 18).
- `b3f9897` the forward pass behind `DS4_MELLUM_TRACE`, and its trace driver (step 19).
- `f86c6b4`, this commit: the oracle's token files and comparison script (step 20).

## Measured

| what | command | result | n |
|---|---|---|---|
| tokenizer ids vs the model's own tokenizer, 30 strings | `DS4_TEST_MELLUM_MODEL=... ./tests/test_mellum_tokenizer` | 30/30 equal | 1 |
| chat renders vs the model's own template, 3 conversations + 2 prefixes | `DS4_TEST_MELLUM_MODEL=... ./tests/test_mellum_config` | all equal | 1 |
| forward pass vs the old port (`refs/old/mellum21-stage1`), session logits | `tests/mellum_oracle_compare.sh ORACLE_DIR $MODEL 64` | byte-equal (`cmp`) on 29, 1,100 and 2,200 templated tokens, prefill and decode; prefill equals decode in this engine on all three; 64 of 64 greedy tokens equal, with equal logits at each step | 1 |
| Metal kernels vs double references | `make test-mellum-kernels` | 379 checks, 0 failures | 1 |
| MoE on real layers 0/13/27 vs double | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-moe` | same experts selected; weights 1.7e-7, output 3.4e-7 | 1 |
| session bytes allocated vs planned, Q8_0 GGUF | `DS4_TEST_MELLUM_MODEL=... ./ds4_test --mellum-session` | equal at ctx 20,000 / 65,536 / 131,072 for chunks 1,024 and 512 (601,513,984 / 1,254,318,080 / 2,193,842,176 bytes at chunk 1,024) | 1 |

## Backends covered

Metal only, on Apple Silicon. CUDA, ROCm, the CPU reference path and
tensor-parallel/distributed inference are not built or tested for Mellum here,
and Mellum is not expected to run on them until someone ports and checks it.

## Baseline at 5379c1c (2026-10-26)

Machine: Apple M5 Max, 128 GiB (`sysctl -n hw.memsize machdep.cpu.brand_string`),
macOS SDK 27.0 (`xcrun --show-sdk-version`).

`make -j8` builds `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval` and `ds4-agent`
with no warnings (10.9 s wall). The model-free checks, each run once (n = 1),
all exit 0:

| command | result |
|---|---|
| `./ds4-eval --validate-cases` | 92 core and 50 hard cases validated |
| `./ds4-eval --self-test-extractors` | passed |
| `./ds4_agent_test` | ok |
| `./ds4_test --server` | ok |
| `./ds4_test --metal-kernels` | ok |
| `./tests/test_layer_pack` | 97/97 |
| `./tests/test_engine_mgpu_placement` | 114/114 |
| `./tests/test_gpu_args` | passed |
| `./tests/test_gpu_args_cli.sh` | 90 passed, 0 failed |
| `./tests/test_prompt_prefix` | passed |
| `./tests/test_sampling` | OK |
| `./tests/test_deepseek4_vision_image` | exit 0 |
| `./tests/test_image_decode` | exit 0 |

The rest of `make test` (`./ds4_test` with no arguments) needs a DeepSeek V4
Flash model and was not run.

## Dependency audit (step 2, 2026-10-26)

The old port lives on three local-only branches, fetched by path into
`refs/old/*` (no remote added):

```sh
git fetch --no-tags <ds4-engine>/.git/worktrees/p42-2d-mellum-fit/modules/external/ds4 \
  'refs/heads/mellum21-*:refs/old/mellum21-*'
```

`refs/old/mellum21-on-upstream` (`7e8d9e6`) is antirez `0aaea5a` + 40 fork
commits (Qwen3.5 dense, DFlash, verify, host extensions, ending at `ebdcb0f`)
+ 21 Mellum engine commits; `refs/old/mellum21-stage1` (`f5cc405`) adds 22
agent/host commits; `refs/old/mellum21-minimal` (`e7b77ed`) is the engine
squashed onto `ebdcb0f`. So the Mellum work is `git diff ebdcb0f
refs/old/mellum21-stage1`, and everything it needs is either in that diff, in
upstream `5379c1c`, or in the fork commits `0aaea5a..ebdcb0f`. The audit lists
the third group. Method: every identifier in the added lines of that diff,
checked against trees of `5379c1c` and `ebdcb0f` (found in the fork, absent
upstream), then each hit read in context.

Addendum (found at step 4): the old port also relied on the fork raising
`DS4_MAX_HEAD_KV` from 2 to 4 (`6a015c6`, for Qwen3.5's 27B). Mellum has 4 KV
heads; upstream reads the limit nowhere else, so `5109a6d` raises it.

### Fork-only Metal code to re-home under Mellum names

| old name (where, in `refs/old/mellum21-stage1`) | from fork commit | Mellum use | step |
|---|---|---|---|
| `kernel_dflash_ring_write` (`metal/dflash.metal:51-82`), host `ds4_gpu_dflash_ring_write_tensor` (`ds4_metal.m:51309`) | `8f3af76` | commits a sliding layer's chunk to its f16 K/V ring and tags each slot with its position, after attention has read the ring (`ds4.c:60726`) | 16 |
| `kernel_dflash_head_norm_rope` (`metal/dflash.metal:86-145`), host `dflash_head_norm_rope_dispatch` (`ds4_metal.m:51333`) | `e2d05e2`; Mellum's `ba07f9e` changed `pad0` to `mscale` | per-head q/k RMS norm + NeoX RoPE; `ds4_gpu_mellum_qk_norm_rope_tensor` (`ds4_metal.m:51381`) fills the 64 per-pair frequencies (YaRN blend on full layers) and the mscale in double | 15 |
| `dflash_dispatch` (`ds4_metal.m:51269`), a named-pipeline dispatch helper | `e2d05e2` | dispatches `kernel_mellum_attn_swa` (`ds4_metal.m:51476`) | 16 |
| the `DS4_METAL_DFLASH_SOURCE` entry in `ds4_gpu_full_source` (`ds4_metal.m:4804`) | `e2d05e2` | `metal/mellum.metal` is appended after it; upstream's list is at `ds4_metal.m:4754-4800` | 15 |

`kernel_mellum_attn_swa` itself (`metal/mellum.metal`) is already Mellum's
own and self-contained: it uses no identifier from `dflash.metal` or
`qwen4.metal`.

Upstream cannot serve the q/k norm + RoPE: its `ds4_gpu_qwen4_attn_prep_tensor`
(`ds4_gpu.h:3556`) takes rotary frequencies from one process-wide table of at
most 32 pairs (`ds4_gpu_qwen4_set_rope`, `ds4_metal.m:49051-49063`), while
Mellum rotates all 128 dims (64 pairs) with two bases, `freq_base_swa` on
sliding layers and `freq_base` + YaRN on full ones.

### Host extensions to replace

- `ds4_engine_sampling_defaults` (fork `8b70f74`, Mellum values added in
  `3e31173`): the old CLI (`ds4_cli.c:545`) and agent read it. Upstream has no
  per-engine table: `cli_apply_model_sampling_defaults` (`ds4_cli.c:528-536`)
  and `agent_apply_model_sampling_defaults` (`ds4_agent.c:1006-1014`) set
  T 1.0 / top_p 0.95 / min_p 0 for GLM-DSA only; every other model, Mellum
  included, gets `DS4_DEFAULT_*` (`ds4.h:59-61`: T 1.0, top_p 1.0, min_p
  0.05). The server starts from the same defaults (`ds4_server.c:992-994`)
  and forces them again in thinking mode for knobs the request left out
  (`ds4_server.c:13950-13958`). Step 13 adds the narrow engine call; the
  server's thinking-mode override has to read it too.
- `ds4_session_supports_payload` (fork `8b70f74`): in the old branches only
  tests call it (`tests/test_mellum_config.c:928,957,1217`); no host does.
  Upstream's contract already covers it: `ds4_session_payload_bytes` returns 0
  when a session cannot be saved (`ds4.c:63511`), and stage/save/load/snapshot
  return an error. Step 14 refuses Mellum by name in those functions. It must:
  a Mellum session sits in the QWEN4_EXP family, so as is it would take the
  `ds4_session_is_qwen4` branch (`ds4.c:63512`) and size a Qwen3.8 payload.

### Other fork-isms in the old code (dropped, nothing to replace)

- `ds4_model_is_qwen35()` appears only in guards (`!ds4_model_is_qwen35() &&
  !ds4_model_is_mellum()`); upstream has no Qwen3.5, so the clause goes.
  `DS4_SHAPE_QWEN35` is the "sibling model" in `tests/test_mellum_config.c`
  and `tests/test_mellum_frontends.c`; `DS4_SHAPE_QWEN4_EXP` takes its place.
- `g_ds4_qwen_reasoning_effort` (fork `591c6a0`): the old Mellum config turns
  it off because 2.1's template has no reasoning effort. Upstream's
  `qwen4_chat_system` (`ds4.c:44450`) always prepends
  `ds4_qwen4_reasoning_effort_text`, and `qwen4_chat_assistant_prefix`
  (`ds4.c:44438`) opens `<think>\n` when thinking is on, so Mellum needs its
  own system turn and assistant prefix (steps 11 and 33), not a flag.
- The Metal 4 language-version fix `ebdcb0f` is not carried (CPython only).

### Upstream code the old port changed

These are not fork dependencies, but they touch other models' paths:

- The Qwen3.8 attention decode kernels gain an `ungated` word in place of
  `pad0` in `ds4_metal_args_qwen4_attn_decode` (`metal/qwen4.metal:2090`) and
  a branch in `qwen4_attn_decode_tile` (2115), `kernel_qwen4_attn_merge`
  (2254) and `qwen4_attn_merge_wide_head` (2294); the host wrappers
  `ds4_gpu_qwen4_attn_decode_tensor` and `ds4_gpu_qwen4_attn_decode_rows_tensor`
  become thin calls into `_impl` versions. This is step 17. `make
  test-qwen4-kernels` covers the gated path numerically; there is no Qwen3.8
  Flash Next GGUF on this machine, so its decode speed cannot be re-measured
  here.
- `ds4_qwen4_layer_is_linear/_ple/_nextn` need a Mellum guard (a Mellum shape
  has `n_full_attn_interval` 0 and arm64 `x % 0 == x`), and session creation
  must not give Mellum a shared prefill workspace (`ds4.c:74337`).

### Upstream functions Mellum will use

Each exists at `5379c1c` with the same prototype as at the fork base `ebdcb0f`
(and at `0aaea5a`). The kernels behind them are unchanged too: the fork's
edits to existing Metal code are Qwen3.5 GDN only (`git diff 0aaea5a ebdcb0f
-- metal/` deletes 6 lines, all in `kernel_qwen4_gdn_scan`/`_front`), and
upstream's only Metal change since `0aaea5a` is a new DSV4 kernel in
`metal/dsv4_misc.metal` (`git diff --stat 0aaea5a 5379c1c -- ds4_metal.m
metal/`: 2 files, 49 lines). So the oracle's byte-equality premise holds for
the reused kernels.

| function | `ds4_gpu.h` | `ds4_metal.m` | Mellum use |
|---|---|---|---|
| `ds4_gpu_embed_tokens_q8_0_tensor` | 588 | 12244 | token embedding |
| `ds4_gpu_rms_norm_weight_rows_tensor` | 1187 | 22301 | attn, ffn and output norms |
| `ds4_gpu_matmul_q8_0_tensor` | 727 | 19611 | one-row projections and the LM head |
| `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` | 847 | 19637 | T-row projections, bit-equal to T one-row calls |
| `ds4_gpu_tensor_copy_f32_to_f16` | 61 | 9379 | K/V to f16 (full cache rows, sliding staging) |
| `ds4_gpu_add_tensor` | 2549 | 31163 | residuals |
| `ds4_gpu_qwen4_attn_decode_tensor` | 3584 | 49229 | full-layer decode attention (with `ungated`) |
| `ds4_gpu_qwen4_attn_decode_rows_tensor`, `ds4_gpu_qwen4_attn_rows_stage` | 387, 360 | 49491, 49323 | full-layer prefill rows, row by row |
| `ds4_gpu_qwen4_attn_part_floats` | 3583 | 49225 | split partials sizing |
| `ds4_gpu_qwen4_multi_gemv_tensor` | 3542 | 50183 | F32 router logits |
| `ds4_gpu_qwen4_router_topk_tensor` | 3548 | 49016 | top-8 of 64 |
| `ds4_gpu_qwen4_moe_mid_tensor`, `_down_tensor`, `_reduce_tensor` | 3593, 3599, 3620 | 49530, 49589, 49692 | Q8_0 experts, per token, no shared expert (`shared_type = UINT32_MAX`) |
| `ds4_gpu_qwen4_moe_mm_mid_tensor`, `_down_tensor` | 3629, 3634 | 49875, 49957 | step 25 only: Q8_0 (type 8) is accepted on simdgroup tiles; tensor tiles are types 12, 39, 16, 10 only (`qwen4_moe_mm_nax`, `ds4_metal.m:49809`) |
| `glm_graph_begin_commands_if_needed`, `ds4_gpu_end_commands`, `ds4_gpu_tensor_view/read/write/alloc` | | | one command buffer per chunk (`ds4.c:56437`) |

Upstream's speculative loop is where the brief says: `ds4.h:549-558`
(`ds4_sessions_eval_batch_speculative_argmax`) and `ds4.h:604`
(`ds4_session_rewind_speculative`). The model id and syntax selection that
step 3 tests are `ds4_server.c:1215-1225` and `ds4_agent.c:412-417`.

### Cherry-pick probe

`git diff ebdcb0f refs/old/mellum21-stage1 -- <file>` applied to a `5379c1c`
tree with `patch -p1 --dry-run`: `ds4_agent.c` 52 of 53 hunks apply (the
failing one is `agent_apply_model_sampling_defaults`, the host-extension
caller); `ds4.h` 2 of 3; `Makefile` 7 of 9; `ds4_cli.c` 0 of 1 (the sampling
call again); `ds4.c` 30 of 54. The agent cherry-picks; the engine is re-ported.

### Where the brief and the code disagree (for Checkpoint 1)

1. "Three fork-only kernels": the fork code Mellum uses is two kernels
   (ring write, head norm + RoPE) and one host dispatch helper. The sliding
   attention kernel is already Mellum's own.
2. Step 13, "upstream's agent defaults top_p to 0.95": only for GLM-DSA.
   Mellum would get T 1.0 / top_p 1.0 / min_p 0.05, so the divergence
   Mellum's table replaces is min_p 0.05 (top_p already matches).
3. Steps 25-27 and prefill-equals-decode: `ds4_gpu_matmul_q8_0_tensor` does
   take the Metal 4 tile (rows a multiple of 32, or the multiple-of-32 prefix
   when n_tok >= 192 outside quality mode, `ds4_metal.m:19422-19440`), but the
   old forward never calls it for more than one row. `mellum_proj`
   (`refs/old/mellum21-stage1:ds4.c:60688-60700`) uses the decode-rows-exact
   kernel so a prefill chunk equals its decode steps bit for bit, and
   `ds4_gpu_mellum_moe_tensor` avoids `moe_mm_*` because they stage half
   operands. So step 27 has nothing to confirm in the oracle-matched forward:
   the tile, like the `moe_mm` and `attn_mm` paths of steps 25-26, is a
   kernel swap with a tolerance, and it gives up the exact prefill = decode
   that step 21 builds and the backlog's B5 relies on.
4. Step 6, ring of window + 1 "so a speculative write never overwrites an
   entry a later token still reads": the old engine attends first and commits
   the chunk to the ring afterwards. After a forward ending at position P-1 a
   W-slot ring holds P-W..P-1, and after a rewind to P-k the next query needs
   P-k-W+1..P-k-1, which is in the ring when k <= 1. A depth-1 cycle only ever
   drops the draft row (k = 1), so W slots already suffice; W + 1 buys a
   rewind of 2. The kernel also uses one number for both (`slot = pos %
   args.window`, `metal/mellum.metal:62`), so W + 1 means a ring-size argument
   separate from the window: still the same logits, but not a verbatim port.
5. The brief lists 11 commits of `mellum21-on-upstream`; the branch has 21
   Mellum commits (on 40 fork commits). The list reads as a selection.
6. Not in the brief's list: `g_ds4_qwen_reasoning_effort` and the qwen4
   system/assistant renders above, and the old port's change to upstream's
   Qwen3.8 attention kernels (`ungated`), whose speed this machine can only
   check model-free.
