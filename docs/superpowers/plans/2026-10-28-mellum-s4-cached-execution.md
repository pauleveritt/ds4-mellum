# S4 Mellum Cached Expert Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Mellum opt-in from a bounded 6 GiB slot cache of Q8_0 experts,
byte-identical to resident output, with no resident regression. Then measure
it at 32K context on this laptop.

**Architecture:** Reuse upstream's Metal expert cache (slabs, entry table,
address tables, `pread` pool, eviction) and add address-table variants of
Mellum's MoE kernels. A decode token is one command buffer that stops at its
first missed layer and resumes at that layer's expert compute; prefill syncs
per layer after the router. A host-pure C state machine owns the protocol.

**Tech Stack:** C99, Objective-C/Metal, the Python 3.14 stdlib for
measurement tooling; no new dependencies.

**Spec:** [S4 design](../specs/2026-10-28-mellum-s4-cached-execution-design.md),
with evidence in the [layer-sync spike](../research/2026-10-28-mellum-layer-sync-spike.md)
and [retarget](../research/2026-10-28-mellum-ssd-retarget.md).

## Global Constraints

- **Flags.** Streaming is opt-in: `--ssd-streaming --ssd-streaming-cache-experts 6GB`.
  For Mellum, `6GB` = 6 GiB of slots = **979** experts of **6,580,224** B,
  with no carve-out for prefill layers.
- **No resident change.** Weights, routing, Q8_0 arithmetic and expert
  reduction order are unchanged. Resident behavior is unchanged: identical
  logits and output, and speed within **2%** of baseline (medians, six
  processes per arm, A/B/B/A, 30 s rests).
- **Decode waits.** The host adds no waits on an all-hit token. Decode stops
  only at a missed layer.
- **Reads.** Streamed reads use an `F_NOCACHE` model descriptor, with no
  `F_RDADVISE` on that path.
- **Process.** Superpowers SDD: a separate observed RED commit before each
  GREEN, about 150 product lines per commit, one concern per commit. Every new
  test is shown to fail when its behavior is broken.
- **Test tiers:** CPU-only tests run freely. Metal kernel tests (GPU, no
  model) and live tests (real weights) each need an owner-authorized window.
  Never bypass `/tmp/ds4.lock`, and never kill another owner.
- **Out of scope:** CUDA/ROCm, server/agent, `../ds4-engine`, hotlists,
  prefetch, Q4, and layer-major prefill.
- **Gates.** `make -f Makefile test-mellum` (the full serial model-free gate)
  passes before every GREEN commit. Add each new CPU test to it.

## Review Focus

1. **A miss in layer 0 or layer 27** stops and resumes correctly (Task 4).
2. **An expert loaded at one stop** is not evicted by a later stop in the same token (Task 7).
3. **Prefill chunks under 32 rows** use the exact path's address variant and stay identical (Task 6).
4. **A short read or `EIO`** during a load gives a clean error, never a hang (Task 7).
5. **Streaming off** allocates no slot, gate or table, and a `0` cache is refused (Tasks 1, 2).

---

## File Structure

| File | Responsibility |
|---|---|
| `ds4_mellum_stream.h/.c` (new) | Host-pure: slot budget, decode stop/resume state machine, prefill unique-expert planner. No GPU or model calls. |
| `tests/test_mellum_stream.c` (new) | CPU tests of the above with fake ops. |
| `metal/qwen4.metal` | Address-table variants of Mellum MoE kernels; `kernel_mellum_stream_check`. |
| `ds4_metal.m`, `ds4_gpu.h` | Gate buffer and gated dispatch; address-variant encoders; Mellum stream table; the `F_NOCACHE` fd; wait counter. |
| `tests/test_mellum_kernels.c` | Resident-vs-address kernel equality, gate suppression (Metal, no model). |
| `ds4.c` | Admission of Mellum streaming; streaming branch in `mellum_graph_moe` and `mellum_graph_forward_tokens`. |
| `tools/mellum_s4_prompt.py`, `tools/mellum_s4_measure.py` (new) | Frozen synthetic 32K workload; measurement runner. |
| `tests/test_mellum_s4_measure.py` (new) | CPU tests of the runner with fake commands. |

## Shared interfaces (exact)

```c
/* ds4_mellum_stream.h */
#define DS4_MELLUM_EXPERT_BYTES 6580224ull
uint32_t ds4_mellum_stream_slots(uint64_t cache_bytes);          /* floor; 0 if < 1 expert */
int ds4_mellum_stream_config(bool streaming, uint64_t cache_bytes, uint32_t *slots, char *err, size_t n);
typedef struct { uint32_t layer; uint32_t n_missing; int32_t missing[8]; } ds4_mellum_stream_stop;
typedef struct ds4_mellum_stream_ops {
    void *ctx;
    bool (*encode)(void *ctx, uint32_t from_layer, bool resume_at_experts);
    bool (*finish)(void *ctx, bool *stopped, ds4_mellum_stream_stop *stop);  /* commit+wait, read gate */
    bool (*load)(void *ctx, uint32_t layer, const int32_t *ids, uint32_t n);  /* blocks until resident */
    bool (*cancelled)(void *ctx);
} ds4_mellum_stream_ops;
enum { DS4_MELLUM_STREAM_OK = 0, DS4_MELLUM_STREAM_LOAD = -1,
       DS4_MELLUM_STREAM_PROTOCOL = -2, DS4_MELLUM_STREAM_CANCELLED = -3, DS4_MELLUM_STREAM_GPU = -4 };
int ds4_mellum_stream_decode(const ds4_mellum_stream_ops *ops, uint32_t n_layer, uint32_t *stops);
uint32_t ds4_mellum_stream_unique(const int32_t *selected, uint32_t rows, uint32_t top_k, int32_t out[64]);
```

There are three protocol rules:
- Each stop's layer must be at or after the previous stop's layer; it may
  equal it only if the second miss set is disjoint after a load. Otherwise
  return `PROTOCOL`.
- At most `n_layer × 8` stops per token, or `PROTOCOL`.
- `encode(from_layer, true)` is used only after a stop.

---

### Task 1: Gate suppression primitive (the riskiest piece, first)

**Files:**
- Modify `ds4_gpu.h`, `ds4_metal.m` and `metal/qwen4.metal`.
- Test in `tests/test_mellum_kernels.c`.

**Interfaces:**
- Produces:
  - `ds4_gpu_tensor *ds4_gpu_stream_gate_create(void)`;
  - `int ds4_gpu_stream_gate_reset(ds4_gpu_tensor *gate)`;
  - `int ds4_gpu_stream_gate_bind(ds4_gpu_tensor *gate_or_null)`: while a gate
    is bound, every Mellum-layer encoder dispatch is suppressed on the GPU
    once the gate holds a stop;
  - `int ds4_gpu_stream_gate_trip_if_zero_tensor(ds4_gpu_tensor *gate, const ds4_gpu_tensor *probe, uint32_t layer)`,
    a test helper that trips the gate when `probe[0] == 0`;
  - `int ds4_gpu_stream_gate_read(const ds4_gpu_tensor *gate, uint32_t *layer, bool *stopped)`;
  - `uint64_t ds4_gpu_wait_count(void)`, which counts host waits in
    `ds4_gpu_end_commands`.
- Mechanism: the builder chooses between indirect dispatch with gate-zeroed
  threadgroup counts and guarded streaming copies. Record the choice in the
  commit message. Blits on the gated path must become gated kernels.

- [ ] **Step 1: Write the failing tests.** Each runs a chain of 28 "layers",
  each `ds4_gpu_add_tensor(acc, acc, one, n)`:
  - `check_gate_suppresses_after_trip`: trip at layer 9 with a zero probe;
    assert `stopped && layer == 9` and `acc == 10` everywhere.
  - `check_gate_all_hit_one_wait`: no trip; assert `acc == 28` and exactly
    one `ds4_gpu_wait_count()` increment.
  - `check_unbound_gate_inert`: bind `NULL` and trip a gate; assert `acc == 28`.
- [ ] **Step 2: Build and run it (owner window).** Run `make test-mellum-kernels`.
  Expected: it fails on the missing symbols. Commit the RED.
- [ ] **Step 3: Implement the gate, bind, trip, read and wait counter.**
  Resident encoders must keep their exact dispatch when no gate is bound.
- [ ] **Step 4: Run the tests.** Run `make test-mellum-kernels`. Expected:
  the three new checks pass and the existing 379 still pass.
- [ ] **Step 5: Run the full gate, then commit the GREEN** (`feat: gate suppression for streamed Mellum decode`).

### Task 2: Admission, budget, `F_NOCACHE` reads and the Mellum stream table

**Files:**
- Create `ds4_mellum_stream.h/.c` (budget only here) and
  `tests/test_mellum_stream.c`.
- Modify `ds4.c` (engine open near `ds4.c:73879` and the cache plan near
  `ds4.c:70430-70810`), `ds4_metal.m` and the `Makefile`.

**Interfaces:**
- Produces:
  - `ds4_mellum_stream_slots` and `ds4_mellum_stream_config`. With streaming
    off, `config` sets `*slots = 0` and returns 0. With streaming on and a
    budget under one expert, it returns -1 and names
    `--ssd-streaming-cache-experts` in `err`. The engine open calls it;
  - `int ds4_gpu_mellum_stream_open(int model_fd_nocache, const void *map, uint64_t size, uint32_t slots)`;
  - `ds4_gpu_stream_expert_table` filled per Mellum layer, with the gate,
    up and down offsets from the layer weights;
  - the gates that admit Mellum: `n_expert == 8`, `n_total_expert == 64`, Q8_0.

- [ ] **Step 1: Write the failing CPU tests.**
  - `slots(6ull<<30) == 979`, `slots(6580223) == 0`, `slots(0) == 0`.
  - `test_config_refuses_zero_cache`: `config(true, 0, …)` returns -1, and
    `err` contains `--ssd-streaming-cache-experts`.
  - `test_config_off_has_no_slots`: `config(false, 6ull<<30, …)` returns 0
    with `slots == 0`.
  - `test_config_6gb`: `config(true, 6ull<<30, …)` gives `slots == 979`.
- [ ] **Step 2: Run the tests and observe them fail.** Run
  `make -f Makefile tests/test_mellum_stream && ./tests/test_mellum_stream`.
  Commit the RED.
- [ ] **Step 3: Implement.** Accept `--ssd-streaming` for Mellum on Metal
  only, refusing it elsewhere with a message. Parse `6GB` to slots with no
  prefill-layer carve-out. Open a second model descriptor with
  `fcntl(fd, F_NOCACHE, 1)` and `F_RDAHEAD 0` for streamed reads. When
  Mellum streams, skip `F_RDADVISE` and the model residency set.
- [ ] **Step 4: Run the CPU tests and the full gate.** Expected: they pass.
  Commit the GREEN.

### Task 3: Address-table kernel variants (decode, tiled, E1)

**Files:**
- Modify `metal/qwen4.metal`: `kernel_qwen4_moe_mid` at 2766,
  `kernel_qwen4_moe_down` at 3047, the `kernel_qwen4_moe_mm_mid/down` NT
  1/2/4/8 instances at 3650-3880, and the `kernel_qwen4_moe_mm_*_nax_t`
  instances at 3942-4234.
- Modify `ds4_metal.m` and `ds4_gpu.h`.
- Test in `tests/test_mellum_kernels.c`.

**Interfaces:**
- Produces `ds4_gpu_qwen4_moe_{mid,down,mm_mid,mm_down}_addr_tensor`: each
  takes its resident encoder's arguments plus `const ds4_gpu_tensor
  *gate_addrs, *up_addrs, *down_addrs`, a `uint64[64]` of GPU addresses each.
- When an address is 0, the kernel reads nothing for that expert and leaves
  its output rows untouched.
- Resident instantiations keep their `host_name`s and arithmetic. A template
  parameter selecting the base source is allowed.

Split into three commit pairs, one per family: **3a** decode, **3b** tiled,
**3c** E1 NAX/NAXF/NAXC.

- [ ] **Step 1 (per family): Write the failing test.**
  - `check_addr_<family>_equals_resident`: build 64 distinct random Q8_0
    experts in one buffer and a shuffled copy in slot buffers, with the
    address tables pointing at the shuffled slots. Over 4 random selections
    (T = 1 for decode; T = 31, 32, 64 and 100 for tiled and E1), assert the
    resident and `_addr` outputs are equal under `memcmp`.
  - `check_addr_<family>_zero_address_reads_nothing`: set one table entry to
    0 and fill the output with a sentinel. Assert that expert's rows keep the
    sentinel and no GPU fault is reported.
- [ ] **Step 2: Run the test (owner window).** Run `make test-mellum-kernels`.
  Expected: FAIL on the missing encoder. Commit the RED.
- [ ] **Step 3: Implement the `_addr` pipelines** for that family.
- [ ] **Step 4: Run the tests.** Expected: the new checks pass and all
  existing checks are unchanged. Commit the GREEN.

### Task 4: Decode stop/resume state machine (CPU-pure)

**Files:**
- Modify `ds4_mellum_stream.c/.h` (`ds4_mellum_stream_decode`).
- Test in `tests/test_mellum_stream.c`.

**Interfaces:**
- Consumes the interfaces above.
- Produces `ds4_mellum_stream_decode`, called by Task 5 with real ops.

- [ ] **Step 1: Write the failing tests.** The fake ops record the call
  sequence. A scripted `finish` returns stops.

  | Test | Script | Assertions |
  |---|---|---|
  | `test_all_hit_one_finish` | no stops | sequence `encode(0,false), finish`; `stops == 0` |
  | `test_stop_resume_at_experts` | stop at layer 9, missing {3, 41} | sequence `encode(0,false), finish, load(9,{3,41}), encode(9,true), finish`; `stops == 1` |
  | `test_stop_layer0_and_layer27` | both ends | both resume with `encode(L,true)` |
  | `test_stop_regression_is_protocol` | layer 9, then layer 4 | returns `PROTOCOL` |
  | `test_same_layer_after_reload_is_protocol` | layer 9 twice, same ids | returns `PROTOCOL` |
  | `test_load_failure` | `load` returns false | returns `LOAD`, no further `encode` |
  | `test_cancel_between_stops` | `cancelled` true after the first load | returns `CANCELLED` |
  | `test_stop_cap` | 225 stops | returns `PROTOCOL` at `n_layer*8 + 1` |
- [ ] **Step 2: Run the tests and observe them fail.** Commit the RED.
- [ ] **Step 3: Implement** the loop as specified by the rules above.
- [ ] **Step 4: Run the tests and the full gate.** Expected: they pass.
  Commit the GREEN.

### Task 5: Streaming decode wiring in `ds4.c`

**Files:**
- Modify `ds4.c`: `mellum_graph_moe` at 60521-60581 and
  `mellum_graph_forward_tokens` at 60770+.
- Modify `ds4_metal.m` and `metal/qwen4.metal`.

**Interfaces:**
- Consumes Tasks 1–4.
- Produces `int ds4_gpu_mellum_stream_check_tensor(ds4_gpu_tensor *gate, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *gate_addrs, uint32_t layer)`.
  It is a GPU kernel, `kernel_mellum_stream_check`. When any of the row's 8
  addresses is 0, it trips the gate and writes the missing IDs into the gate
  record.

Streaming decode splits the resident fused `ds4_gpu_mellum_moe_tensor` into:
1. router and top-8, saved in `SELECTED` and `WEIGHTS`;
2. the check;
3. `_addr` mid and down;
4. reduce.

On resume from layer L's experts, `encode(L,true)` starts at step 3 with
the saved `SELECTED` and `WEIGHTS`, then the residual add, then layers L+1
onward and the head. Because suppression leaves every scratch buffer
untouched after the stop, `ATTN_RES`, `FFN_NORM` and the KV writes of layer
L are not recomputed. The `load` op calls the reused cache to make the
missing IDs resident, waits for their reads, protects them for this token,
and writes the address tables.

- [ ] **Step 1: Write the failing live test (owner window).** It is a new
  `ds4_test --mellum-stream-identity` with a resident session and a streamed
  session.
  - Streamed at a 6 GiB cache: 64 greedy tokens of the MELLUM.md
    hash-map prompt; assert identical token IDs and an identical
    last-row logits `memcmp` at every step.
  - Streamed at a 64-slot cache: the same assertions, plus `stops > 0` on
    at least half of the tokens.
- [ ] **Step 2: Run** with
  `DS4_TEST_MELLUM_MODEL=$MODEL ./ds4_test --mellum-stream-identity`.
  Expected: FAIL (streaming refused or not wired). Commit the RED.
- [ ] **Step 3: Implement the split, the check kernel, the ops adapter and
  the resume encode.** Decode only for now; prefill stays resident-only until
  Task 6. A streaming prefill attempt returns a clear "not yet" error, and
  the test feeds the prompt through decode-sized chunks of 1 row.
- [ ] **Step 4: Run the identity test.** Expected: it passes both
  configurations. Then run `make test-mellum-kernels` and the full gate.
  Commit the GREEN.

### Task 6: Streaming prefill (per-layer sync)

**Files:**
- Modify `ds4.c` (the prefill branch of `mellum_graph_moe`) and
  `ds4_mellum_stream.c` (`ds4_mellum_stream_unique`).
- Test in `tests/test_mellum_stream.c` and `ds4_test --mellum-stream-identity`.

For each layer of a streamed chunk:
1. run the router and top-8;
2. `end_commands`, then read `SELECTED`;
3. find the unique set;
4. load the missing experts and protect all of them;
5. begin a new command buffer and run the `_addr` E1, tiled or exact
   variant. T < 32 uses the exact path; this follows `ds4_gpu_qwen4_moe_q8_tiles`.

- [ ] **Step 1: Write the failing tests.**
  - `test_unique_sorted`: rows `{1,2,…}` with repeats → sorted unique; at most
    64; the count is correct.
  - `test_unique_rejects_out_of_range`: an id of 64 → returns `UINT32_MAX`.
  - Extend the live identity test: a streamed prefill of a 2,200-token prompt
    (`tests/mellum_oracle_2200.tokens`) at a 6 GiB cache and at a 64-slot
    cache, plus a 31-token prompt, gives logits equal under `memcmp` to
    resident prefill.
- [ ] **Step 2: Run the tests and observe them fail.** Commit the RED.
- [ ] **Step 3: Implement.** The 64-slot case must still hold one layer's
  needs: a chunk can need up to 64 experts, and the cache must refuse rather
  than thrash if it holds fewer.
- [ ] **Step 4: Run the CPU tests, the live identity test and the full gate.**
  Expected: they pass. Commit the GREEN.

### Task 7: Read failure, eviction safety and cancellation

**Files:**
- Modify `ds4_metal.m` (the load path) and `ds4.c` (error propagation).
- Test in `tests/test_metal_ssd_experts.c` (Mellum shape mode) and
  `tests/test_mellum_stream.c`.

- [ ] **Step 1: Write the failing tests.**
  - `test_metal_ssd_experts --mellum-shape` (Metal, no model), using the
    eviction oracle at N=8 over 28×64 entries with 979 and 16 slots:
    - no slot in use by the current token is evicted, including an expert
      loaded at the first of two stops in one token, with 16 slots;
    - a refill never publishes an address before its read completes.
  - `--mellum-read-failure`: an injected short `pread` makes `load` return
    false, the session reports `DS4_MELLUM_STREAM_LOAD`, and a following
    `ds4_session_*` call returns an error rather than hanging. Use a fault
    hook compiled only into the test.
- [ ] **Step 2: Run the tests and observe them fail.** Commit the RED.
- [ ] **Step 3: Implement the protection set, the publish-after-read
  ordering and the error path.** Wire `cancelled` to the engine's existing
  interrupt flag.
- [ ] **Step 4: Run the tests and the full gate.** Expected: they pass.
  Commit the GREEN.

### Task 8: Measurement tooling (CPU)

**Files:**
- Create `tools/mellum_s4_prompt.py`, `tools/mellum_s4_measure.py` and
  `tests/test_mellum_s4_measure.py`.
- Modify the `Makefile` gate.

**Interfaces:**
- `build_prompt(root: Path, budget_tokens=30000) -> dict`:
  - takes sorted tracked `*.c`/`*.md` files from `git ls-files`, excluding
    `docs/superpowers/research/**` and anything under ac-demo paths;
  - concatenates path-delimited blocks until the native token count is
    reached, never splitting a file;
  - returns `{bytes, sha256, files}`.
- `fresh_copy(model: Path, dest: Path) -> dict`: an exclusive create, written
  with `F_NOCACHE` in 64 MiB blocks, then size and `cmp` verification.
- `run(arms, n, rest_s=30, read_cap=400e9) -> Path`:
  - takes `arms` from {A, B, C}, in A/B/B/A order for A and B (n=6) and
    with C interleaved (n=3);
  - brackets each C process with the device-counter observer
    (`tools/mellum_device_observer.py`);
  - classifies C as `device-observed` when device/logical ≥ 0.99, else
    `cache-contaminated`;
  - stops a run that exceeds `read_cap` as a recorded failure;
  - never retries.

- [ ] **Step 1: Write the failing tests** with fake commands and fake
  counters.
  - The prompt is deterministic and excludes research and ac-demo.
  - The arm order matches.
  - A 0.985 ratio is classed `cache-contaminated`.
  - A read of 400e9+1 is a recorded failure with no second process.
  - A lock-held preflight postpones and consumes no root.
  - All raw outputs are retained.
- [ ] **Step 2: Run the tests and observe them fail.** Commit the RED.
- [ ] **Step 3: Implement.** The stdlib only.
- [ ] **Step 4: Run the tests and the full gate.** Expected: they pass.
  Commit the GREEN.

### Task 9: Review, the preregistered measurement, evidence

- [ ] **Step 1: Independent whole-branch review** of `b1994c6..HEAD` against
  the spec and Review Focus. Fix Important findings with separate RED and
  GREEN commits.
- [ ] **Step 2: Preregister, and commit before any timing,** in
  `docs/superpowers/research/2026-10-2x-mellum-s4-measurement-prereg.md`:
  the prompt sha, the arms and n, the 2% rule, the 0.99 classification, the
  400 GB cap and the metrics. The metrics are time to first token, decode
  t/s over the first and last 256 tokens, misses and stops per token,
  logical and device bytes, and peak `phys_footprint` and Metal allocation.
- [ ] **Step 3: Run once (owner window, lock free).** The pre-S4 baseline is
  `b1994c6`, built in a detached worktree.
- [ ] **Step 4: Write up.** The evidence note carries an honesty header, n,
  limits (this M5 Max only), a falsifier and risks. Update the ROADMAP S4
  status, run `git diff --check`, and commit.

## Self-review notes

Coverage: goal 1 is Task 2; goal 2 is Tasks 1, 3 and 9; goal 3 is Tasks 3,
5 and 6; goal 4 is Tasks 1, 4 and 5; goal 5 is Tasks 4 and 7; goal 6 is
Tasks 2, 8 and 9. Task 5 feeds the prompt in 1-row chunks so decode identity
is provable before Task 6.
