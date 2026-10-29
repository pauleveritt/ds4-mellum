# S6-C: streamed MXFP4 Mellum on the 8 GB M1 iMac — Implementation Plan and Runbook

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure S6's Goals 4 and 5 on the target machine: streamed MXFP4 Mellum runs on an 8 GB M1 iMac,
with an editor and a browser open, without swapping that slows decode, at a reported speed. Before that, show
its logits are not broken on the M1's kernel path.

**Architecture:** Two parts.
- **Tasks 1–4 (agent, on the M5 Max, CPU only or model-free):** make the tools MXFP4-aware and freeze the
  workload and the bounds.
- **Tasks 5–8 (owner, by hand on the iMac):** a runbook. Every step writes its capture to one evidence
  directory, and Task 9 (agent) analyzes the captures without re-running the model (binding rule 5).

**Spec:** [`2026-10-28-mellum-s6-mxfp4-8gb-design.md`](../specs/2026-10-28-mellum-s6-mxfp4-8gb-design.md),
"Order", section C, and Goals 4 and 5. Phase B's results:
[logits](../research/2026-10-28-mellum-s6-b-logits.md), [measurement](../research/2026-10-28-mellum-s6-b-measurement.md).
Template: [M1 Pro bring-up](2026-10-28-mellum-m1pro-bringup.md).

## Global Constraints

- **Target (owner):** M1 iMac, 8 GB, 1 TB SSD, macOS 27.1. No `iogpu.wired_limit_mb` change, ever.
- The owner runs every iMac step. Nothing is run over a network connection the owner did not set up.
- Preregistration: [`2026-10-28-mellum-s6-c-preregistration.md`](../research/2026-10-28-mellum-s6-c-preregistration.md),
  committed before any iMac run. One run per preregistered item; failures are recorded, not retried.
- Agent tasks follow phase A/B's rules: the red test is committed before its fix, one concern per commit,
  and Q8_0 behavior is unchanged.
- Never `--raw-prompt`.

## Review Focus

1. **Q8_0 unchanged** in every tool touched: a Q8_0 model or index gives today's output byte for byte.
2. **An expert type the tools do not know** (Q4_K) is refused with a message naming the tensor.
3. **M1 without `MTLGPUFamilyMetal4`:** the runbook records it and continues on the non-Metal-4 path.
   Phase B's tests covered that path on the M5 (`DS4_METAL_DISABLE_METAL4=1`).
4. **An iMac step that fails halfway** leaves its partial capture in place; the runbook never deletes it.
5. **The M5 reference logits** the iMac is compared against are the ones phase B's (b) wrote, copied byte
   for byte. Their SHA-256 is checked on both machines.

---

### Task 1 (agent): the S4 measurement tool sizes experts from the model

**Files:** `tools/mellum_s4_measure.py`, `tests/test_mellum_s4_measure.py`, `tests/test_mellum_s4_device_reads.py`.

- [ ] Red test: `expert_bytes_for(model_path)` reads the type of `blk.0.ffn_gate_exps` through
  `tools/mellum_gguf.py`. Type 8 gives 6,580,224, type 39 gives 3,290,112, and anything else raises a
  ValueError naming the tensor. Test it on the sparse fixtures the GGUF tests already build. Commit.
- [ ] Implement it. `EXPERT_BYTES` stays as the Q8_0 constant. The device-read test uses
  `expert_bytes_for`. Verify `make test-mellum`. Commit.

### Task 2 (agent): the expert index and pack accept MXFP4

**Files:** `tools/mellum_expert_index.py`, `tools/mellum_expert_pack.py`, and their tests.

- [ ] Red tests:
  - an all-MXFP4 directory indexes at PART 1,096,704 and expert 3,290,112;
  - a mixed directory (layer 2 MXFP4) is refused, naming `blk.2`;
  - the pack's span check takes the index's expert size;
  - a Q8_0 index is unchanged.

  Commit.
- [ ] Implement: the geometry comes from the first gate tensor's type (8 or 39), and every tensor must
  match it. Verify, then commit.

### Task 3 (agent): the Metal working-set probe and the background load

**Files:** create `tools/mellum_metal_limit.m` (prints `MTLDevice` name, `recommendedMaxWorkingSetSize`,
`hasUnifiedMemory`, `supportsFamily:MTLGPUFamilyMetal4` when the SDK has it, and `MTLGPUFamilyApple7..10`;
no command buffers) and `tools/mellum_background_load.sh` (opens PyCharm on a named small project and
Safari with the ten URLs fixed in the preregistration, then waits for the owner's Enter).
`tools/mellum_machine_facts.sh` calls the probe when it is built.

- [ ] Build the probe and run it on the M5 Max: record its output in the C evidence directory as the
  reference. The load script is checked with `sh -n`, since it opens apps and only the owner runs it.
  Commit.

### Task 3b (agent): the trace driver runs streamed

**Files:** `tests/mellum_trace_context.h` (`mellum_trace_stream_bytes`), `tests/mellum_trace.c`, a new CPU test
`tests/test_mellum_trace_stream.c` in `test-mellum`.

- [ ] Red test of the parser:
  - NULL means resident (0 bytes);
  - a decimal at least 1 is that many bytes;
  - empty, non-decimal, 0 or an overflow is a usage error.

  Commit.
- [ ] Implement it. A non-zero value sets `ssd_streaming` and `ssd_streaming_cache_bytes`. On the M5, a
  streamed (2.5 GiB) MXFP4 trace of `mellum_oracle_2200` must be byte-equal (`cmp`) to phase B's
  resident trace files. Commit.

### Task 4 (agent): the Pi-shaped workload and the iMac kit

- [ ] Extract one mellum-worker first request (system prompt, tools, a few files) from
  `../mellum-worker-raw-research-2026-10-13/ablation-a21-control.tar.gz`, into a scratch directory, never
  run. Render it with the engine's template the way the worker sent it. Freeze it as
  `docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/workload.txt`, with its SHA-256 and its token
  count from a resident M5 run (`--gen 1`).
- [ ] Assemble the kit the owner copies to the iMac:
  - the branch at a named commit;
  - phase B's `logits/raw/ours.mx.*.logits.f32` reference files, with a SHA-256 list;
  - the workload, the prompt files and this runbook.

  Commit everything except the raw logits, which stay git-ignored and are listed by hash.

### Task 5 (owner, iMac): bring-up

- [ ] Copy the MXFP4 GGUF and check its SHA-256 (`cf09584a…`). Run `tools/mellum_machine_facts.sh` and
  `tools/mellum_metal_limit`.
- [ ] `make`, then `make test-mellum`, then `make test-mellum-kernels`. Record the counts. The MXFP4 lines
  must pass. Then the three streamed `ds4_test` checks on `$MX`: `--mellum-stream-identity`, `-prefill`,
  `-faults`. A failure stops here, and the next step is a debugging session, not a timing run.

### Task 6 (owner, iMac): correctness against the M5 Max

- [ ] Run `tests/mellum_trace $MX FILE prefill` with `MELLUM_TRACE_STREAM_BYTES` set to the cache the preregistration's rule
  picks, on the three oracle token files at the oracle positions. Then run `tools/mellum_logits_agree.py`
  against the M5 reference files. The bound is the preregistration's.

### Task 7 (owner, iMac): footprint with apps open (Goal 4)

- [ ] Apps closed, then the background load: three runs each of `tools/mellum_s4_measure` on the workload,
  streamed at the chosen cache, with `vm_stat` sampled before and after the timed decode window. The order
  and `n` are the preregistration's.

### Task 8 (owner, iMac or M4 Max): the falsifier's 27B figure

- [ ] On the 36 GB M4 Max with `ds4-engine`, run the 27B's decode t/s on the same workload, n=3. This is
  the spec's falsifier comparison.

### Task 9 (agent): analysis

- [ ] From the captures alone, write `2026-10-28-mellum-s6-c-imac.md`: verdicts by the preregistration, with
  each figure's `n`, "What would falsify this" and "Risks". Update ROADMAP S6.
