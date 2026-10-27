# Mellum owned capture adapter implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Execute inline with separate RED/GREEN commits and one fresh whole-change review.

**Goal:** Connect the approved recorder to an owned Mellum session and actual Metal encoding sites, with CPU/build verification now and explicit live gates later.
**Architecture:** A private handle in `ds4.c` owns an ordinary session, CPU controller and two cap-strided ID buffers. A scoped C observer binds successful Metal math encodings to an immutable catalog and the controller's current profile. Unrecorded graphs retain the ordinary numerical path.
**Tech stack:** C99, Objective-C/Metal, existing Python 3.14+ stdlib exporter.
**Spec:** `docs/superpowers/specs/2026-10-27-mellum-session-capture-design.md`, already approved. Continuation authorized by user “Proceed”; preserve inline execution and existing isolated branch.

## Global constraints

- Worktree `codex/mellum-ssd-replay`, baseline `c9ac887`; no push/merge.
- No GPU initialization, model inference or GPU experiments. Compile/link Metal only. Live tests are authored but not executed without a newly authorized window.
- Preserve routes, Q8 bytes, precision, kernel arguments/order and reductions.
- Use the existing 128-profile/64-descriptor/six-value limits and 2,200,000 logical recorder-byte bound at cap=1024. No per-layer allocation or new per-layer waits.
- Any failed active capture is terminal; FINISHED errors preserve bytes. Never manufacture successful provenance or completion.
- Unsupported builds provide linkable error-returning public APIs and never allocate a session or write a header.
- Small commits, approximately <=150 product additions each; behavioral RED tests committed separately. CPU tests establish bookkeeping, not real device observation or ordering.
- Live numerical/cleanup/overhead tests and qualified source-provenance harness remain open; this stage cannot close S2b.

## Files and interfaces

- `ds4_mellum_capture.h`: opaque `ds4_mellum_capture`, options `{spool_sink, spool_context, cancel, cancel_context}`, public create/sync/eval/copy_logits/finish/free, using existing engine/tokens/cancel/sink types.
- `ds4.c`: private handle, supported-engine validation, graph recorder pointer, snapshots/readback/cleanup, operation wrappers. Existing test-only reference hooks remain available.
- `ds4_mellum_observer.h/.c` and `ds4_mellum_kernels.inc`: fixed production catalog, one serialized active scope and projection/head tag. No GPU dependency.
- `ds4_metal.m`: record actual successful dispatch identity and actual chosen specialization constants, including fallback/tail branches and auxiliary kernels.
- `tests/test_mellum_capture_api.c`, `tests/test_mellum_observer.c`: model-free real APIs; no fake GPU abstraction.
- `tests/mellum_routes_gpu.c`: later live harness runs owned handle alongside ordinary/reference paths; compile now, execute only later.
- Makefile links new C modules for normal/core, CPU and test consumers. Add model-free targets to `test-mellum`.

Public API exact signatures (0 success except logits count):

```c
int ds4_mellum_capture_create(ds4_mellum_capture **out, ds4_engine *, int context_tokens,
    const ds4_mellum_capture_options *, char *err, size_t errlen);
int ds4_mellum_capture_sync(ds4_mellum_capture *, const ds4_tokens *, uint32_t segment, char *, size_t);
int ds4_mellum_capture_eval(ds4_mellum_capture *, int token, uint32_t segment, char *, size_t);
int ds4_mellum_capture_copy_logits(ds4_mellum_capture *, float *, int capacity);
int ds4_mellum_capture_finish(ds4_mellum_capture *, mellum_profile_sink, void *, char *, size_t);
void ds4_mellum_capture_free(ds4_mellum_capture *);
```

Observer interfaces:
`const mellum_kernel_catalog *mellum_observer_catalog(void)`;
`bool mellum_observer_enter(mellum_profile_scope *)`;
`bool mellum_observer_leave(mellum_profile_scope *)`;
`void mellum_observer_projection_role(bool head)`;
`bool mellum_observer_encode(const char *name, size_t count, const uint32_t *values)`.
Enter borrows an already-active scope. Refuse nested/foreign attempts without damaging the existing observer. Leave clears only a matching scope even when its profile failed; any unknown/malformed observation poisons the current profile. Encode while disabled returns true before consulting arguments. Projection role defaults to PROJECTION, resets on enter/leave and affects only projection kernels. Auxiliary kernels retain OTHER; attention/router/expert/reduce/embed have fixed catalog roles. The table includes all audited reachable identities and exact named integer pipeline constants, not runtime dimensions that are already shape metadata.

## Review focus

- Encoding failures/observer overflow must reach ordinary command cleanup, clear collector scope and never emit a footer; add CPU observer poisoned-leave test and require live failure tests later.
- Generic/exact/tiled/NAX/tail/fallback branches must describe the kernel that encoded, including actual function constants; pin constant differences and head tags in CPU observer tests, audit actual Metal sites, defer physical proof.
- Unsupported/invalid engine creation must write no bytes and leave `*out=NULL`; test real CPU API including bounded/null error buffers.
- Prefix/token/context errors and cancellation must fail before subsequent reuse, while no-op sync preserves logits; existing CPU controller tests plus adapter validation tests and later live matrix.
- The recorder-off path must allocate nothing and preserve inference semantics; disabled observer accepts even unknown/null observations; compile ordinary/reference harness and require live baseline overhead/parity.

### Task 1: Public unsupported-platform contract

**Files:** create public header and `tests/test_mellum_capture_api.c`; modify `ds4.c`, Makefile and `.gitignore`.
**Consumes:** existing `ds4.h`, routes/profile sink definitions.
**Produces:** public API above; CPU/non-Apple definitions with safe unsupported/null-handle behavior.

- [x] Write real linked API tests: create with a sentinel out pointer returns nonzero, sets NULL, writes no sink bytes, reports unsupported; null out/options/engine/context and null/zero/tiny error buffers remain safe; null sync/eval/finish fail, null logits returns 0, free(NULL) succeeds. Tests do not construct/load a model.
- [x] Add declarations/minimal false-return stubs; `make test-mellum-capture-api` must fail an assertion rather than linking. Commit RED.
- [x] Implement safe bounded errors/platform stubs, run same target PASS, commit GREEN.

### Task 2: Bounded production encoding observer

**Files:** observer header/source/catalog, `tests/test_mellum_observer.c`, Makefile.
**Consumes:** existing scope/registry APIs, types above.
**Produces:** observer interfaces above; immutable audited production catalog.

- [x] Tests with real registry: validated catalog; disabled null/unknown observations are silent; nested/foreign enter and leave preserve owner; unknown name/wrong value count/null values poison active profile; poisoned leave clears global scope so a fresh registry can enter. Missing required roles remain invalid.
- [x] Record a complete exact-path fixture with literal real kernel names, projection/head tags, required roles and auxiliary conversions. Reverse ordering reuses ID; change NSG/specialization constants creates distinct ID. Canonical strings expose named values and absent head. Add minimal stubs; run assertion RED with `make test-mellum-observer`, commit.
- [x] Implement fixed table plus observer, run PASS. Keep table/logic in separate small GREEN commits if needed.

### Task 3: Actual Metal encoding observations

**Files:** `ds4_metal.m`, catalog if audit requires additional real identities; encoding audit in usage.
**Consumes:** Task 2 observer, successful encode branches.
**Produces:** actual math descriptor events while scope active, no observation of blits or requested-only controls.

- [x] Audit reachable calls from Mellum graph: embedding; weighted norm; Q8 legacy/decode/exact/MV-ext/NAX/generic/unpack plus tails; f32/f16 conversion; residual add; Mellum norm/RoPE/ring/SWA; qwen4 router/top-k/lists/MoE mid/down/reduce/staging; full attention decode/rows/merge variants. Record every actual encoding branch, excluding observer copies.
- [x] Add live assertions to diagnostic harness for nonempty profiles, required roles and real head/expert path distinctions. Live RED/run is unavailable: record this limitation; CPU supplied-event tests are not a substitute for device proof.
- [x] Hook successful dispatch sites. Record constants from the same selected values used to create that actual pipeline; unspecialized values use the actual defaults, not merely requested settings. Preserve command-buffer finalization even on observer failure; sticky profile failure is checked at graph completion.
- [x] `make ds4_metal.o` and `make test-mellum-observer`: expect compile/PASS without running Metal.

### Task 4: Owned session and graph adapter

**Files:** `ds4.c`, API tests and Makefile.
**Consumes:** CPU controller, Task 2 observer/catalog and Task 3 encoding events.
**Produces:** supported Metal create and real graph snapshots, public operation wrappers.

- [x] Expand API tests for invalid tokens/contexts and terminal/FINISHED controller gating using real internal controller state where no session operation is reached. Write live owned-versus-baseline/reference tests for sync/eval/append/logits, allocation/copy/read errors and cancellation; defer actual runs explicitly.
- [x] Create validates a ready resident Mellum Q8 Metal engine, non-distributed, no TP/multi-tier/speculation; fresh private ordinary session, actual cap layout, host+GPU IDs allocated before header. On failure free all owned resources and leave out NULL. Callback error/status handling is null-buffer safe.
- [x] Graph pointer borrows handle. Open forward/scope before math; acknowledge each successful ID copy before selected scratch reuse; ordinary cleanup always runs. After existing completion read IDs once, clear observer, finalize frame only on all-success. On any error clear observer/owned scope and fail terminally. Tag only output projection as HEAD; no-logits forwards must explicitly omit head.
- [x] Sync validates actual private checkpoint prefix/context/token IDs before inference; eval validates token/context/position; preserve cancellation status 2. End checks actual graph/checkpoint frontier through the controller. No-op sync emits no frames. Logits only with controller availability; free detaches graph, frees private session/buffers, never invokes sinks/footer.
- [x] Run model-free targets, `make ds4.o ds4_metal.o ds4_cpu.o` and build diagnostic harness; expect pass/compile only. Split implementation into focused <=150-product-line commits.

### Task 5: CPU/build gate and final review

- [x] Run `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum`, explicit C99 CPU build and observer ASan/UBSan. No broad `make test`, GPU/model process or model lock bypass.
- [x] Update roadmap/usage/spec with compiled adapter status and exact unrun live/qualified-provenance gates; keep S2b open. Persist encoding audit and decisions.
- [x] One fresh read-only independent review of `c9ac887..HEAD`, with this plan/spec/rulings and Review Focus. Fix Critical/Important in one RED/GREEN pass, ledger deferred minors, no second reviewer.
- [x] Preserve branch/worktree and commit final evidence. No push/merge.

## Self-review

The complete approved design remains authoritative. This plan covers the native
adapter/observer wiring; actual device proof and the qualified source-identity
publisher are expressly separate later gates. No new GPU scheduling abstraction
is introduced. Existing core tests own lifecycle rules; adapter tests own native
boundary validation; actual backend observation requires the deferred live runs.

## Completion record

The CPU/build stage and one independent review are complete. Both Important
findings were fixed with separate RED/GREEN commits. Live checks remain unrun.
See the [retained review and execution ledger](../research/2026-10-27-mellum-owned-capture-review.md)
for verification, all rulings and their costs, and remaining device/publisher gates.
