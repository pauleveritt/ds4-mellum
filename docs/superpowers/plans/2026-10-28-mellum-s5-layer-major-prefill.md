# S5 Grouped Layer-Major Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamed Mellum prefill runs groups of K chunks per layer pass (K = 8 by default), so each layer's
experts load about once per group, with logits identical to resident.

**Architecture:** A host-pure helper sizes each group, honouring the one-row-tail rule. The streamed graph
gains a group hidden buffer, counted in Mellum's memory plan. A new grouped forward embeds the group's
chunks into that buffer. For each layer it copies each chunk's rows into `MELLUM_S_HIDDEN`, runs the
unchanged `mellum_graph_layer`, and copies them back. The session sync calls it for streamed Mellum when
K > 1. Resident Mellum and K = 1 keep today's per-chunk loop.

**Tech Stack:** C99 and Objective-C/Metal (no new kernels), the engine's test binary `ds4_test`, the
Python 3.14 stdlib for the measurement; no new dependencies.

**Spec:** [grouped layer-major prefill design](../specs/2026-10-28-mellum-s5-layer-major-prefill-design.md)
(owner-approved 2026-10-28), with the [options note](../research/2026-10-28-mellum-s5-layer-major-prefill-options.md).

## Global Constraints

- **K:** `DS4_MELLUM_STREAM_PREFILL_GROUP`, an integer 1–64, **default 8**. Any other value refuses the
  streamed session with a message. K = 1 runs today's per-chunk loop unchanged.
- **Streaming only.** Resident Mellum code paths do not change behavior, and their memory plan is
  byte-for-byte as today.
- **Identical output.** Grouped streamed logits equal resident logits word for word (`memcmp` per float).
- **No change to weights, routing, Q8_0 arithmetic, expert reduction order or any Metal kernel.**
- **Group buffer:** K × `cap_tokens` × n_embd × 4 B (K = 8, cap 1,024, n_embd 2,304: **75,497,472 B**). It
  is counted in the plan, and a session's printed allocated bytes equal its planned bytes.
- **Keep the GPU wait after each streamed (layer, chunk) router** (`ds4.c:60663`): it makes the CPU-written
  full-attention rows table safe to reuse within a layer.
- **One-row tail:** only a suffix's last chunk can have one row. The group ends before it, and that row
  runs through today's T = 1 path (`mellum_stream_decode`).
- **Checkpoints:** a finished group is the checkpoint unit. Cancellation is checked between groups only. A
  failure inside a group invalidates the checkpoint, as a failed chunk does today (`ds4.c:78428-78433`).
- **Progress:** `prefill_chunk` fires per finished group. `prefill_display` (display callback, not a
  checkpoint) fires after each layer pass inside a group.
- **Process:** Superpowers SDD; a separate observed RED commit before each GREEN; about 150 product lines
  per commit; one concern per commit. Every new test is shown to fail when its behavior is broken.
- **Runs:** CPU tests freely. Live tests (`DS4_TEST_MELLUM_MODEL`) and the measurement each need an
  owner-authorized window; never bypass `/tmp/ds4.lock` or kill another owner.
- **Gate before every commit:** `make -f Makefile -f .superpowers/sdd/2026-10-28-mellum-generated-cohort/fixture-lock.mk test-mellum`
  exits 0, plus the task's live tests. Commit only if all pass (`if … && …; then git commit`).

## Review Focus

1. **A later session on the same engine** reuses lent scratch (`qwen4_graph_transfer_scratch`,
   `ds4.c:59736`). A second streamed session must still match resident and own a valid group buffer
   (Task 3 test).
2. **A prompt ending at the context limit.** A group must not run past `ctx_cap`; a sync filling context
   2,048 to 2,047 tokens matches resident (Task 3 test).
3. **K larger than the prompt's chunk count.** One partial group at the default K = 8 matches resident
   (Task 3 test).
4. **A bad K setting** (`0`, `65`, `abc`, empty) refuses the streamed session cleanly, and a good one opens
   (Tasks 1 and 2).
5. **Rewind after a grouped prefill,** then decode, keeps the sliding rings consistent: the existing
   `--mellum-rewind` live test, run with streaming and K = 2 (Task 3 verification).

---

### Task 1: Group sizing and the K setting (host-pure)

**Files:**
- Modify: `ds4_mellum_stream.h`, `ds4_mellum_stream.c`
- Test: `tests/test_mellum_stream.c` (CPU, run by `make test-mellum-stream`)

**Interfaces:**
- Produces:
  - `#define DS4_MELLUM_STREAM_GROUP_DEFAULT 8u`
  - `#define DS4_MELLUM_STREAM_GROUP_MAX 64u`
  - `uint32_t ds4_mellum_stream_prefill_group(const char *env)`: NULL → 8; a decimal 1–64 → itself; anything
    else (including empty, `0`, `65`, `8x`) → 0, meaning refuse.
  - `uint32_t ds4_mellum_stream_group_rows(uint32_t remaining, uint32_t cap, uint32_t k)`: rows for the next
    group, `min(k × cap, remaining)`, minus one when that would take the whole suffix, is more than one row,
    and ends in a one-row chunk (`rows % cap == 1`); 0 when `remaining`, `cap` or `k` is 0.

- [ ] **Step 1: Write the failing tests** `check_prefill_group_setting` and `check_group_rows`:

```c
CHECK(ds4_mellum_stream_prefill_group(NULL) == 8u, ...);
CHECK(ds4_mellum_stream_prefill_group("2") == 2u && ds4_mellum_stream_prefill_group("64") == 64u, ...);
CHECK(ds4_mellum_stream_prefill_group("0") == 0u && ds4_mellum_stream_prefill_group("65") == 0u &&
      ds4_mellum_stream_prefill_group("") == 0u && ds4_mellum_stream_prefill_group("8x") == 0u, ...);
CHECK(ds4_mellum_stream_group_rows(30620, 1024, 8) == 8192u, ...);
CHECK(ds4_mellum_stream_group_rows(2200, 1024, 2) == 2048u && ds4_mellum_stream_group_rows(152, 1024, 2) == 152u, ...);
CHECK(ds4_mellum_stream_group_rows(1281, 256, 2) == 512u && ds4_mellum_stream_group_rows(257, 256, 2) == 256u &&
      ds4_mellum_stream_group_rows(1, 256, 2) == 1u, ...);
CHECK(ds4_mellum_stream_group_rows(1025, 1024, 8) == 1024u, ...);
CHECK(ds4_mellum_stream_group_rows(2200, 1024, 1) == 1024u && ds4_mellum_stream_group_rows(0, 1024, 8) == 0u, ...);
```

- [ ] **Step 2: Run** `make -f Makefile test-mellum-stream`. Expected: link failure on the two new
  symbols. Commit the RED test alone.
- [ ] **Step 3: Implement both functions** in `ds4_mellum_stream.c`, using `strtoul` with an end-pointer
  check.
- [ ] **Step 4: Run** `make -f Makefile test-mellum-stream` (all pass). Mutation proof: drop the one-row
  rule, and the `257` and `1025` checks fail. Restore, rebuild (delete the binary first: same-second
  mtimes), and run the gate.
- [ ] **Step 5: Commit** `feat: size streamed Mellum prefill groups, K from DS4_MELLUM_STREAM_PREFILL_GROUP`.

### Task 2: The group hidden buffer in the plan and the graph

**Files:**
- Modify: `ds4.c`: `mellum_requests` (`:17582`), `mellum_context_memory` (`:17637`), `mellum_graph_alloc`
  (`:59495-59530`), `qwen4_graph_free`, the graph struct (`mellum_stream` fields)
- Test: `tests/ds4_test.c`, new `--mellum-stream-group-memory`

**Interfaces:**
- Consumes: `ds4_mellum_stream_prefill_group` (Task 1).
- Produces:
  - Graph fields `uint32_t mellum_group_k;` and `ds4_gpu_tensor *mellum_group_hidden;` (NULL unless
    streaming with K > 1).
  - `mellum_requests(uint32_t ctx, uint32_t cap_tokens, uint32_t group_rows, mellum_req *out)` adds
    `MELLUM_REQ_GROUP_HIDDEN` (class `MELLUM_CLASS_PREFILL`, `group_rows × E × 4` bytes) when
    `group_rows > cap_tokens`.
  - Every caller passes `group_rows = 0` unless the engine is a streamed Mellum engine with K > 1; then it
    passes `K × cap`. The engine-level plan callers (`:40943`, `:61973`) use the same rule, so the startup
    plan line counts it.
- A streamed open with K = 0 from Task 1 fails `ds4_engine_open` with
  `ds4: Mellum SSD streaming: DS4_MELLUM_STREAM_PREFILL_GROUP must be 1-64`.

- [ ] **Step 1: Write the failing test** `test_mellum_stream_group_memory`. With `DS4_TEST_MELLUM_MODEL`,
  `prefill_chunk` 1,024 and context 8,192, open a streamed engine (6 GiB) at K = 1 and at K = 8, and
  capture each session line (`test_capture_session_line`). Assert:
  - allocated == planned for both;
  - allocated(K = 8) − allocated(K = 1) == 75,497,472;
  - `DS4_MELLUM_STREAM_PREFILL_GROUP=0` makes `ds4_engine_open` fail;
  - `=2` opens.
  Restore the env with `test_save_env`/`test_restore_env`.
- [ ] **Step 2: Run** `./ds4_test --mellum-stream-group-memory` (live). Expected: the difference is 0, and
  K = 0 opens. Commit the RED test.
- [ ] **Step 3: Implement** the request, the allocation and free, the plan rule and the refusal.
- [ ] **Step 4: Run** the new test plus `--mellum-stream-identity`, `--mellum-stream-prefill` and the
  existing session-memory test (`ds4_test.c:547-570`, resident). All pass, and resident planned bytes are
  unchanged. Then run the gate.
- [ ] **Step 5: Commit** `feat: plan and allocate the streamed Mellum prefill group buffer`.

### Task 3: The grouped forward and its use in the session sync

**Files:**
- Modify: `ds4.c`: a new `mellum_graph_prefill_group` beside `mellum_graph_forward_tokens` (`:61015`);
  the session sync's chunk loop (`:78416-78437`)
- Test: `tests/ds4_test.c`, new `--mellum-stream-group`

**Interfaces:**
- Consumes: `ds4_mellum_stream_group_rows` (Task 1); `g->mellum_group_k`, `g->mellum_group_hidden`
  (Task 2); `ds4_gpu_tensor_copy(dst, dst_off, src, src_off, bytes)` (an ordered blit; needs open commands).
- Produces: `static bool mellum_graph_prefill_group(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w, const int *tokens, uint32_t rows, float *logits_out, ds4_session_progress_fn display, void *display_ud, int display_base, int display_total)`.
  - It runs `rows` (> 1, ≤ K × cap) as ⌈rows / cap⌉ chunks, at `g->pos` and onward.
  - Per chunk: write its tokens, embed them into `MELLUM_S_HIDDEN`, and copy the rows to the chunk's slot.
    End commands before each token write; `MELLUM_S_TOKENS` is shared.
  - Per layer, then per chunk in order: copy the slot into `MELLUM_S_HIDDEN`, call `mellum_graph_layer`
    with that chunk's `pos0` and row count, and copy the rows back.
  - After each layer pass, call `display(display_ud, "prefill_display", display_base + rows done, display_total)` when set.
  - The head runs on the last chunk's last row; `g->pos += rows` on success only.
- Sync loop rule: for a streamed Mellum graph with `mellum_group_k > 1`, take
  `n = ds4_mellum_stream_group_rows(remaining, cap, K)`. `n > 1` calls the grouped forward; `n == 1` calls
  `qwen4_graph_forward_tokens` as today. Push the n tokens, mark the checkpoint valid, and fire
  `prefill_chunk` once per group. Otherwise, the loop is unchanged.

- [ ] **Step 1: Write the failing test** `test_mellum_stream_group`. Use a helper that, like
  `mellum_stream_prefill` (`ds4_test.c:~650`), opens an engine with a given `prefill_chunk`, context,
  streaming arm and K, syncs `test_mellum_token(i)` prompts, and returns the last-row logits' word diff
  against a resident reference, plus the session's loads (`ds4_test_mellum_stream_loads`). Assert diff == 0
  for:
  - chunk 256, K = 2, lengths 1,281 (groups 512, 512, 256, then the one-row tail) and 1,100, at 6 GiB and at
    64 slots;
  - chunk 1,024 at the default K = 8, length 2,200 (one partial group);
  - context 2,048, chunk 256, K = 2, length 2,047 (ends at the limit);
  - a warm append: sync 700 tokens, then the same session to 1,281;
  - a second session on the same engine after the first is freed, length 1,100.

  Also assert loads(K = 2) < loads(K = 1) at chunk 256, 64 slots, length 1,100, and that K = 1's diff is 0.
- [ ] **Step 2: Run** `./ds4_test --mellum-stream-group`. Expected: the loads assertion fails, because
  K = 2 still runs per chunk and the loads are equal. Commit the RED test.
- [ ] **Step 3: Implement** `mellum_graph_prefill_group` and the sync-loop rule. Keep
  `mellum_graph_layer`, `mellum_graph_moe` and every kernel unchanged.
- [ ] **Step 4: Run** the new test, then `--mellum-stream-identity`, `--mellum-stream-prefill`,
  `--mellum-stream-faults`, `--mellum-stream-close`, and `--mellum-rewind` with
  `DS4_MELLUM_STREAM_PREFILL_GROUP=2`. All pass.
  - Mutation proof: run layer l over the group's chunks in reverse order. Identity must fail, which shows
    the order is pinned. Restore and rebuild.
  - Run the gate.
- [ ] **Step 5: Commit** `feat: grouped layer-major prefill for streamed Mellum`.

### Task 4: Progress, cancellation and failure inside groups

**Files:**
- Test: `tests/ds4_test.c`, new `--mellum-stream-group-events`
- Modify: `ds4.c` only if a test exposes a gap

**Interfaces:**
- Consumes: Task 3's loop; `ds4_session_set_progress`, `ds4_session_set_display_progress`,
  `ds4_session_set_cancel`, `ds4_test_mellum_stream_fail_load(s, nth)`.

- [ ] **Step 1: Write the tests** (chunk 256, K = 2, 64 slots, length 1,100, against a resident reference):
  - **Display:** the display callback sees 28 `prefill_display` events per group, ascending. The progress
    callback sees `prefill_chunk` at 512, 1,024 and 1,100.
  - **Cancellation:** the progress callback sets a flag on its first `prefill_chunk`, and the cancel
    function returns it. Assert:
    - the sync returns `DS4_SESSION_SYNC_INTERRUPTED`;
    - `ds4_session_pos` is 512 and `ds4_session_checkpoint_valid` is true;
    - after clearing the cancel, a re-sync completes, and its logits match the reference.
  - **Failure:** `ds4_test_mellum_stream_fail_load(s, 30)` (inside the first group's layers). Assert:
    - the sync fails, and `ds4_session_checkpoint_valid` is false;
    - `ds4_test_mellum_stream_fail_load(s, 0)`, then a re-sync from scratch, matches the reference.
- [ ] **Step 2: Run** them. These pin Task 3's behavior and may pass at once. Show each fails when broken:
  - drop the display call: Display fails;
  - check cancellation inside the layer loop: Cancellation fails;
  - leave the checkpoint valid on failure: Failure fails.
  Record each mutant's failure in the ledger, then restore. If a test fails for real, fix `ds4.c` in a
  separate GREEN commit after a RED commit.
- [ ] **Step 3: Run the gate and commit** `test: progress, cancellation and failure inside streamed prefill groups`.

### Task 5: Preregister, measure, record

**Files:**
- Create: `docs/superpowers/research/2026-10-28-mellum-s5-measurement-02-prereg.md`, then `…-02.md`
- Create (git-ignored ledger): `.superpowers/sdd/<this plan>/measure-s5-02.py`, adapted from
  `measure-s5-01.py`
- Modify: `ROADMAP.md`, S5 step 2

- [ ] **Step 1: Preregister** before any timing, from S5 measurement 01's prereg.
  - **Inputs:** the same prompt (SHA-256 `2164763c…`) and appended text (`ed93c59f…`), context 36,864,
    2,048 + 256 tokens, 6 GiB.
  - **Arms:** A/B n=6 against `b1994c6` (the sync loop is shared code), R n=1, C with the default K = 8,
    n=3, each from a fresh copy with device counters.
  - **Rules:** identity in both phases; device-observed at a ratio of 0.99 or more; a read cap of 600 GB.
  - **Suffix reuse:** judged by `append_common_prefix` equal to prompt + 2,048 and the positions, not by
    time.
  - **Predictions:** cold first token near 34 s; prefill loads at most 7,168 (a ceiling, as the spec says).
- [ ] **Step 2: Run once** in the owner's window, the way measurement 01 ran.
- [ ] **Step 3: Record** the note with its n, against R, against measurement 01's C (60.6 s cold, 4.29 s
  append, 39.0 t/s decode) and against S5's targets, for orientation. Update ROADMAP S5 step 2. Commit.

### Task 6: Independent review and finish

- [ ] **Step 1:** A fresh Opus reviewer reads the spec, this plan and `git diff 9b8515d..HEAD`.
  - Critical and Important findings are fixed RED/GREEN.
  - Minors are recorded in the ledger.
- [ ] **Step 2:** Update ROADMAP's S5 status and `## Now`. Confirm every claim carries its n. Do not push
  or merge; the branch is the owner's.
