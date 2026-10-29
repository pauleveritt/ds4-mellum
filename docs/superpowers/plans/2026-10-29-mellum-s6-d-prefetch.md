# S6 phase D: the 2.5 GiB fit and asynchronous expert prefetch, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise streamed MXFP4 decode on the 8 GB M1 iMac toward 9 t/s: first fit the 2.5 GiB expert cache under
phase C's rule, then, only if an iMac screen clears its bar, prefetch the next layer's predicted experts behind a GPU
event so that the stop at that layer does not happen.

**Architecture:** The prefetch protocol is one optional op in the model-free stop/resume driver
(`ds4_mellum_stream.c`), pinned by the fake-ops tests. The engine side (`ds4.c`) predicts layer L+1's top-8 on the CPU
from layer L's FFN input and starts the reads with the current layer protected. The Metal side (`ds4_metal.m`) adds a
CPU-signaled `MTLSharedEvent` that the resumed pass waits on before layer L+1's check. Decisions are made on the iMac
over SSH; the M5 is a no-regression guard.

**Tech Stack:** C99, Objective-C/Metal, POSIX shell for the iMac runners, Python 3.10+ (`python3 -I`) for analysis.

**Spec:** `docs/superpowers/specs/2026-10-29-mellum-stream-prefetch-design.md` (approved 2026-10-29).

## Global Constraints

- Branch `phase-d/prefetch` from `mellum` at `c8fc0bb`, worktree `.worktrees/phase-d`. Never `main`; never push.
- One concern per commit; the failing behavioral test committed before its fix; about 150 product lines per commit.
- Every rate carries its n. Screens run freely; Task 6's decision run is preregistered first.
- Exact: prefetch chooses only what is read early. Streamed ids equal resident ids (S4's rule), word for word in
  `ds4_test --mellum-stream-identity`.
- Prefetch runs only with `--ssd-streaming` and only when switched on; with streaming off nothing changes.
- No `iogpu.wired_limit_mb` change. No new Metal compute kernels; no change to the router or the gate's arithmetic.
- C2 rule, unchanged: Metal peak ≤ recommendedMaxWorkingSetSize − 1 GiB = 4,652,892,160 B on the iMac, and no
  swap-out growth. Time to first token on the iMac no more than 10% worse than Task 1's reference.
- The iMac is `pauleveritt@herveline.local`, SSH only (the sandbox must be off for local network). Clone
  `~/ds4-mellum-s6`, model `~/models/Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf`, Python 3.14 first on `PATH`
  (`/opt/homebrew/opt/python@3.14/libexec/bin`). Apps closed: check `top -l 1 -o mem` before each stage.
- A smaller default context, or dropping `mellum`'s rewind frontier under streaming, needs the owner's approval.

## Review Focus

1. **A prefetch read that fails or is cancelled must still signal the event**, or the GPU waits forever: Task 5's
   fault test injects a failed prefetch read inside the read task and expects resident logits, and an `alarm()`
   timeout in the test turns a hang into a failure (`ds4_test` has no timeout of its own).
6. **An install while the GPU runs layer L** must never evict or overwrite a slot L is reading: Task 5's forced-
   eviction test (Fable review, 2026-10-29, finding 3).
2. **Tiny caches:** at 64 slots a prefetch could evict the stopped layer's experts. Task 5 runs identity at 64
   slots with prefetch on.
3. **The last layer:** a stop at layer 27 has no layer 28. Task 4's test expects no prefetch call there.
4. **Streaming off:** no event is created or waited on. Task 5 checks that resident decode encodes no event wait.
5. **A cancelled token with a prefetch in flight** discards the pass and leaves the slots consistent: Task 4's
   cancel test, and Task 5's `--mellum-stream-faults` with prefetch on.

---

### Task 1: A fresh iMac reference on the merged build (no code)

The merge added `mellum`'s rewind frontier (about 44 MB on the M5: the 2.5 GiB Metal peak went from 4,697,161,728 to
4,741,332,992 B). Phase C's references are from `69492de`; Goal 4 compares against this build instead.

**Files:**
- Create: `docs/superpowers/research/2026-10-29-mellum-s6-d-evidence/ref/` (captures),
  `docs/superpowers/research/2026-10-29-mellum-s6-d-reference.md`

- [ ] **Step 1:** Create the worktree and branch (Global Constraints). Build: `make && make tools/mellum_s4_measure
  tools/mellum_metal_limit tests/mellum_trace ds4_test`. Expected: rc 0.
- [ ] **Step 2:** Ship the branch to the iMac: `git bundle create /tmp/d.bundle 69492de..phase-d/prefetch` (the iMac has `69492de`),
  `scp` it, then on the iMac `git fetch /tmp/d.bundle phase-d/prefetch:phase-d/prefetch && git checkout phase-d/prefetch`.
  Expected: `git rev-parse --short HEAD` prints the branch tip.
- [ ] **Step 3:** On the iMac, from the clone: `MX=… OUT=~/s6-d-ref sh docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/run_imac.sh bringup`,
  then `cache`, then `footprint closed BYTES` at the largest qualifying cache. Expected: bringup's gated lines rc 0;
  three Metal peaks recorded.
- [ ] **Step 4:** `rsync` `~/s6-d-ref` into `ref/`, keep the per-position logits git-ignored, and write the short note:
  Metal peaks at 2, 2.5 and 3 GiB against the limit, the reference decode t/s and first token (n=3), and the bytes
  the 2.5 GiB cache is over (the M5 estimate is about 69 MB).
- [ ] **Step 5:** Commit the captures, then the note (two commits).

### Task 2: Fit the 2.5 GiB cache

**Files:** decided in Step 1 (likely `ds4.c` `mellum_requests` / the streamed scratch sizing); test in
`tests/test_mellum_config.c`.

**Interfaces:** Produces the cache the rest of the plan runs at: `CHOSEN_BYTES`, 2684354560 if this task passes,
else 2147483648.

- [ ] **Step 1: Inventory.** On the iMac at 2.5 GiB, run the workload once with `DS4_METAL_TRACE_ALLOCS=1`
  (`tools/mellum_s4_measure … --gen 16`), and list allocations by size. Add the streaming buffers that sit outside
  `mellum_requests` (the gate, the save slots, the 84 address-table views, `ds4.c:59562-59567`; `ds4_test.c:600-601`
  exempts them) and the slabs. Pick the smallest change that frees Task 1's overage plus 64 MB of margin (the measure
  tool samples `currentAllocatedSize` once per token, so its peak is a lower bound), in this order: buffers that streamed decode sizes for prefill but never uses; the S5
  group buffer; `prefill_cap` 1024 → 512 (only if Step 5's first token stays within 10%). The frontier and the
  context are the owner's (Global Constraints). Ledger the choice as a ruling.
- [ ] **Step 2: Write the failing test** in `tests/test_mellum_config.c`, `test_streamed_plan_fits_2_5_gib`: the
  planned bytes of the chosen allocation class at ctx 32768 with streaming on are at most the reference minus the
  overage plus margin from Step 1 (exact numbers from Step 1), and with streaming off are unchanged.
- [ ] **Step 3:** `make test-mellum`. Expected: the new check FAILS, naming the planned bytes.
- [ ] **Step 4: Implement** the reduction behind the streaming flag only.
- [ ] **Step 5:** `make test-mellum` and, on the M5, `DS4_TEST_MELLUM_MODEL=… ./ds4_test` for every `--mellum-*` test
  on Q8_0 and MXFP4. Expected: all rc 0, stream identity 0 differing words.
- [ ] **Step 6:** On the iMac, `run_imac.sh cache` into a new `$OUT`. Expected: 2.5 GiB's Metal peak ≤
  4,652,892,160 B, no swap-out growth; first token within 10% of Task 1. If not, rule and stop at 2 GiB.
- [ ] **Step 7:** Commit test, fix and evidence separately.

### Task 2b: Two streaming fixes the review found (Fable, 2026-10-29, findings 5 and 6)

**Files:** `ds4_metal.m` (the failed-read path: `pending_load_finish`, `prepare_load_buffers`), `ds4.c`
(`mellum_stream_finish`), tests in `tests/ds4_test.c`.

- [ ] **Step 1: Failing test** `--mellum-stream-read-fault`: a test hook fails one pread task (`tasks[i].ok = 0`)
  inside a decode load; afterwards the cache's free slab-slot count equals its count before, and a re-sync gives
  resident logits. Sibling: the same run without the fault, same counts. Expected now: FAIL, slots lost (the comment
  at `ds4_metal.m:51525` admits it).
- [ ] **Step 2:** Return the slab slots taken in `prepare_load_buffers` when a read fails. Run the test: PASS.
- [ ] **Step 3: Failing test** for `mellum_stream_finish`: in the kernel fixture, a dispatch after the stop that
  rewrites `SELECTED`; the stop must still report the stopped layer's ids. Then read the ids from the save slot (the
  check kernel saved them) instead of scratch `SELECTED`. PASS; all `--mellum-stream-*` tests on both quants rc 0.
- [ ] **Step 4:** Commit each red test and each fix separately.

### Task 3: Screen the synchronous prefetch on the iMac (screen-only patch, no product code)

**Files:** Create `docs/superpowers/research/2026-10-29-mellum-s6-d-sync-screen/` with `prefetch-sync.patch`
(ported from `…/2026-10-29-mellum-stream-decode-screen/prefetch/prefetch-hack.patch`), `run_imac_sync.sh`, captures,
and the note `…-sync-screen.md`.

- [ ] **Step 1:** Port the hack onto the branch: at each stop, after the stop's own load, predict layer L+1's top-8
  from `MELLUM_S_FFN_NORM` through L+1's F32 router, and load the missing ones before the commit with L's experts
  protected. Count `prefetched`, `prefetch_used` (prefetched experts that L+1 then selected) and stops per token.
- [ ] **Step 2:** The runner applies the patch, builds, runs the workload at `CHOSEN_BYTES` with `SCREEN_PREFETCH`
  off/on/off/on (128 tokens, 20 s apart), with the stop profile's `DS4_MELLUM_STREAM_PROFILE=1`, reverts, rebuilds
  clean. Dry-run on the M5 first.
- [ ] **Step 3:** Run on the iMac over SSH and pull the captures back.
- [ ] **Step 4: The bar.** Report the measured ms/token change (off minus on), stops avoided per token, prefetch
  accuracy and wasted experts per token. Project the asynchronous saving as
  `avoided × (5.5 ms − w) − wasted × r`, where 5.5 ms is an avoided stop's cost from the four-term refit (the stop
  profile's corrected note), `w` the projected event wait per prefetch (median read span of the prefetch's size minus
  about 2.0 ms of GPU work before L+1's check, floored at 0) and `r` the measured read time per wasted expert. Pass if
  the measured change or the projection is at least 20 ms/token. Stops avoided alone decides nothing.
- [ ] **Step 5:** Commit the screen and its note. **If it fails, skip to Task 7** with a verdict.

### Task 4: The prefetch op in the driver (model-free)

**Files:** Modify `ds4_mellum_stream.h`, `ds4_mellum_stream.c`; test `tests/test_mellum_stream.c`.

**Interfaces:**
- Produces: a new optional field at the end of `ds4_mellum_stream_ops`:
  `bool (*prefetch)(void *ctx, uint32_t stop_layer);` — best effort, called at a stop after `load_begin` and before
  the resumed pass's `encode`, only when `load_begin` is set and `stop_layer + 1 < n_layer`. Its return value is
  ignored by the driver (a failed prefetch leaves the next layer to stop and load on demand); the engine records
  whether it armed a wait.

- [ ] **Step 1: Write the failing tests** (fake ops log `P<layer> `):
  - `test_prefetch_called_between_begin_and_encode`: stops at 9 and 20 → log
    `"E0 F B9:3,41 P9 E9x L9:3,41 F B20:7 P20 E20x L20:7 F "`.
  - `test_prefetch_not_at_last_layer`: a stop at 27 → no `P27`.
  - `test_prefetch_failure_is_not_fatal`: prefetch returns false → rc OK, same log otherwise.
  - `test_prefetch_cancel_discards`: cancel after load → log ends `L9:3 D `, rc CANCELLED.
  - `test_prefetch_absent_is_unchanged`: `prefetch == NULL` → the existing overlap log exactly.
- [ ] **Step 2:** `make test-mellum-stream`. Expected: FAIL (the field does not exist; then the log lacks `P9`).
- [ ] **Step 3: Implement** the call in `ds4_mellum_stream_decode` and the header comment. Existing initializers
  gain a trailing `NULL`.
- [ ] **Step 4:** `make test-mellum`. Expected: PASS, all prior protocol tests unchanged.
- [ ] **Step 5:** Commit the tests, then the implementation.

### Task 5: The engine and Metal sides of the asynchronous prefetch

**Files:** Modify `ds4_metal.m`, `ds4_gpu.h`, `ds4.c` (`mellum_stream_*` callbacks, `mellum_graph_moe`'s streamed
check); tests in `tests/ds4_test.c` and `tests/test_mellum_kernels.c`.

**Prerequisites in the read machinery** (Fable review, finding 1, confirmed in code): the pread pool refuses a
second batch while one runs (`ds4_metal.m:13699-13703`), the pending load is one global (`g_stream_expert_pending_load`),
`begin_selected_load` finishes a non-matching pending load synchronously, and the Mellum protect list holds one
layer (`ds4_metal.m:15485-15487`). This task first adds a second in-flight record for the prefetch, a pool that
queues one batch behind the running one and calls a completion function (which signals the event), and a protect
list for two layers.

**Invariants** (finding 3; today every cache change happens with the GPU idle):
- every eviction for the prefetch happens at prefetch begin, while the GPU is idle at the stop;
- L's and L+1's ids stay protected from begin through install; the install never prunes;
- every address-table and residency write happens before the event is signaled; failure and cancel signal too;
- `prefetch_drain` runs before `ds4_gpu_discard_commands` on every exit of the driver.

**Interfaces:**
- Consumes: Task 4's `prefetch` op.
- Produces, in `ds4_gpu.h`:
  - `int ds4_gpu_mellum_prefetch_begin(const ds4_gpu_stream_expert_table *t, const int32_t *ids, uint32_t n,
    uint32_t protect_layer, const int32_t *protect, uint32_t protect_n, uint64_t *event_value);` — starts the reads
    on the pread pool; when the last read completes (or fails) it marks the loaded slots resident and signals the
    event with `*event_value`. Returns 0 if nothing was issued.
  - `int ds4_gpu_mellum_prefetch_wait(uint64_t event_value);` — encodes `encodeWaitForEvent` into the open batch,
    splitting the encoder (prior art `ds4_metal.m:11163`).
  - `int ds4_gpu_mellum_prefetch_drain(void);` — waits for every prefetch in flight (token end, cancel, failure).
- Switch: `DS4_MELLUM_STREAM_PREFETCH=1` while screening; read at graph allocation into `g->mellum_prefetch`.

- [ ] **Step 0: Read machinery, test first.** Failing model-free or kernel-fixture tests: a second batch queued
  while one runs both complete and call their completion function once; a two-layer protect list keeps both layers'
  entries through a prune. Then implement (its own commits, about 150 product lines each).
- [ ] **Step 1: Write the failing kernel test** `check_prefetch_event_orders_residency` in
  `tests/test_mellum_kernels.c` (the gate chain fixture): a chain whose add at "layer" 5 waits on an event the host
  signals after writing the address table from a second thread; the chain must not trip at 5. Sibling
  `check_prefetch_event_unsignaled_trips`: the table is not written, the event is signaled → the gate trips at 5.
- [ ] **Step 2: Write the failing model tests** in `tests/ds4_test.c`, `--mellum-stream-prefetch`
  (`DS4_TEST_MELLUM_MODEL`): with `DS4_MELLUM_STREAM_PREFETCH=1`, decode at 6 GiB and at 64 slots equals resident,
  0 differing words, and `prefetch_used > 0`; with a failed prefetch read injected (a hook like
  `ds4_test_mellum_stream_fail_load`, but failing the read task itself), still 0 differing words, no hang (an
  `alarm()` in the test), and `--mellum-stream-faults` passes with prefetch on. Forced eviction: at 64 slots, a
  prefetch whose install must evict while layer L runs still gives 0 differing words.
- [ ] **Step 3:** Run both. Expected: FAIL (functions missing; then `prefetch_used == 0`).
- [ ] **Step 4: Implement** in three commits of about 150 product lines: the Metal event and prefetch begin/drain;
  the engine's CPU prediction (layer L+1's F32 router over `MELLUM_S_FFN_NORM`, top-8, as the hack does) and the
  `prefetch` op; the encode-side wait before L+1's check, only when a wait was armed.
- [ ] **Step 5:** `make test-mellum`, `make test-mellum-kernels`, and every `--mellum-*` model test on the M5 for
  Q8_0 and MXFP4, with the switch off and on. Expected: all rc 0; identity 0 differing words in both.
- [ ] **Step 6:** iMac screen: the stop profile kit (`2026-10-29-mellum-imac-stop-profile/`) with the switch off/on,
  n=2 each. Ledger the per-token split and stops per token before Task 6.

### Task 6: Preregister and run the decision

**Files:** Create `docs/superpowers/research/2026-10-29-mellum-s6-d-preregistration.md`, a runner beside it, and
`…-s6-d-evidence/decision/`.

- [ ] **Step 1:** Commit the preregistration before any run: arms prefetch off/on at `CHOSEN_BYTES`, n=3 each,
  interleaved off/on/off/on/off/on, 512 tokens, phase C's workload, apps closed, 20 s rests; Goal 2's pass rule (the
  on median saves ≥ 20 ms/token against the off median); Goal 1 reported; exactness (all six runs' ids equal); the
  M5 guard afterwards (S4's A/B, n=6 per arm, streamed Q8_0 at 6 GiB and MXFP4 at `CHOSEN_BYTES`, resident
  unchanged).
- [ ] **Step 2:** Run on the iMac; pull back; analyze from the captures alone.
- [ ] **Step 3:** Run the M5 guard.
- [ ] **Step 4:** Apply the default rule from the spec: on by default under `--ssd-streaming` only if Goal 2 and the
  M5 guard both pass; else opt-in (`--ssd-streaming-prefetch`, plus the README's 8 GB note) — that flag gets its own
  failing CLI test first.
- [ ] **Step 5:** Commit captures, analysis, and any default change separately.

### Task 7: Close phase D

- [ ] Result note `docs/superpowers/research/2026-10-29-mellum-s6-d.md` with honesty header, every figure's n,
  "What would falsify this" and "Risks"; ROADMAP S6 phase D status; README/MELLUM.md only for what changed. Commit.
