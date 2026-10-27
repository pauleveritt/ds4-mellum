# Mellum capture CPU core implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Execute inline, with separate red-test commits and one fresh final review.

**Goal:** Implement the GPU-free profile registry and capture lifecycle core of S2b.
**Architecture:** Fixed-size canonical kernel profiles feed the existing native spool writer through an append-only operation/forward controller. The later owned-session adapter supplies actual GPU events and the immutable production kernel catalog.
**Tech stack:** C99, existing C writer and Python 3.14+ stdlib exporter/tests.
**Spec:** `docs/superpowers/specs/2026-10-27-mellum-session-capture-design.md` (approved by user “Proceed”).

## Global constraints

- Existing isolated `codex/mellum-ssd-replay` worktree; baseline `a97a1a2`. No push/merge.
- No GPU initialization, Metal experiments or large CPU model inference.
- Preserve Q8 bytes and routing; this plan does not alter inference or default executable link inputs.
- At most 128 profiles, 64 descriptors/profile, six integer values/descriptor, descriptor <=32 bytes.
- ID buffers at cap=1024 total 1,835,008 bytes. Recorder plus registry/current profile and 16 KiB formatting <=2,200,000 logical bytes.
- Any invalid active capture operation is terminal; FINISHED errors leave completed bytes/state intact.
- One concern per commit, behavioral tests committed red before implementation; ~150 product additions per implementation commit.
- The CPU core cannot establish actual Metal observation, synchronization or session ownership. Those remain later S2b work.

## Files and interfaces

- `ds4_mellum_profile.h/.c`: immutable borrowed kernel catalog, canonical sets, scoped registry and catalog formatting. No GPU/global state or heap allocation.
- `ds4_mellum_capture_state.h/.c`: caller-owned lifecycle, operation labels, forward sequencing and real native writer. No session/GPU ownership yet.
- `tests/test_mellum_profile.c`, `tests/test_mellum_capture_state.c`: real modules with bounded sinks and declared synthetic kernel definitions, no mocked GPU.
- Makefile: model-free dedicated targets included in `test-mellum`; use explicit C99 flags.
- `tests/test_mellum_capture_state_export.py`: emitted C fixture -> existing reader/exporter/replay checks.

Profiles use `mellum_kernel_def {name, value_count, value_names[6]}` and a validated borrowed `mellum_kernel_catalog`. Catalog names are unique ASCII identifiers, bounded to 95 bytes, value labels unique within a kernel and <=31 bytes. Lifetime/immutability is a caller contract; the real backend supplies a static table later.

`mellum_kernel_descriptor` holds u16 function ID, u8 role, u8 value_count, six u32 values. Roles: EMBED, PROJECTION, ROUTER, MID, DOWN, REDUCE, ATTENTION, HEAD, OTHER. Require the first seven roles and HEAD iff requested. Ignore unused value words, never compare padding. Sort by role/function/value_count/active values.

## Review focus

- Descriptor padding/inactive values cannot split profiles; specialization differences cannot merge them (Task 1/2).
- A foreign/nested scope must not corrupt the active owner; every owned error exit clears its scope (Task 2/5).
- Catalog/output overflow never produces truncated successful metadata or a valid footer (Task 3/5).
- Failed/cancelled operations and invalid prefixes cannot produce a completed artifact, even after earlier complete frames (Task 4/5).
- FINISHED state rejects writes; independent controllers and exact cap-strided selection bounds remain isolated (Task 5).

### Task 1: Canonical profile sets

**Files:** create profile header/source and `tests/test_mellum_profile.c`; add C99 model-free target.
**Interfaces:** `bool mellum_kernel_catalog_init(catalog*, const mellum_kernel_def*, size_t)`; `bool mellum_profile_init(profile*, const catalog*, bool with_head)`; `bool mellum_profile_record(profile*, mellum_kernel_descriptor)`; `bool mellum_profile_valid(const profile*)`. Profile holds <=64 sorted descriptors and sticky failure.

- [x] Write tests: valid definitions; invalid/null/duplicate/bounded names and labels; all mandatory roles; missing/extra head; reversed insert order; duplicate at capacity; inactive words ignored; distinct active constants; 64 admitted and 65th rejected; bad function/role/value count sticky.
- [x] Add contract declarations and minimal failing stubs only after tests; run `make test-mellum-profile`, expect assertion failure on initialization. Commit RED separately.
- [x] Implement catalog validation and sorted descriptor insertion, normalize unused words. Run target, expect all checks pass. Commit GREEN within product-line budget.

### Task 2: Scoped profile registry

**Files:** extend profile module/tests.
**Interfaces:** `bool mellum_profiles_init(registry*, const catalog*)`; `bool mellum_profile_scope_enter(scope*, registry*, bool head)`; `bool mellum_profile_scope_record(scope*, descriptor)`; `bool mellum_profile_scope_exit(scope*, registry* owner, bool success, uint32_t *id)`.
Registry owns current set plus 128 profiles; scope borrows its active registry. Equal complete profiles reuse IDs; new IDs follow observation order. Failed owner cannot restart. Wrong/nested owner operations return false without altering a different active owner. A matching exit always clears scope; error exit invalidates owner. Constructors only on fresh/unowned storage.

- [x] Tests: canonical reuse across order/padding; distinct constants/head profiles; 128 stored and duplicate reuse at capacity; 129th fails; failed record/exit; nested and wrong-owner exit preserve the original; two sequential registries isolate IDs. Add stubs, run RED and commit.
- [x] Implement equality/interning and scope guards; run `make test-mellum-profile`, expect pass; commit GREEN.

### Task 3: Bounded profile descriptions

**Files:** extend profile module/tests.
**Interfaces:** `bool mellum_profile_describe(const profile*, char *dst, size_t cap)` and `bool mellum_profiles_emit(registry*, mellum_profile_sink, void*)`; sink bool signature `(void*, uint32_t id, const char*, size_t length)`.
Canonical ASCII: `profile-v1;head=encoded|not-encoded` followed by `;ROLE:kernel(label=value,...)` in descriptor order. Length excludes NUL. On formatting failure clear dst[0] when writable. Emit uses <=16 KiB scratch, refuses active/failed/empty registry, marks failed on rendering/callback failure; no further callbacks once failed.

- [x] Literal string/order and absent-head assertions; tiny/zero/null output and canary; long valid catalog exceeding output bound; callback failure stops delivery; complete used IDs only. Add stub, observe RED, commit.
- [x] Implement checked formatting and emit; run profile tests, expect pass; commit GREEN.

### Task 4: Capture operation lifecycle

**Files:** create capture-state header/source/tests and C99 Make target.
**Interfaces:** `bool mellum_capture_state_init(state*, uint32_t cap, sink, void*, const catalog*)`; `bool mellum_capture_begin_sync(state*, const uint32_t *committed, size_t count, const uint32_t *prompt, size_t length, uint32_t segment)`; `bool mellum_capture_begin_eval(state*, uint64_t position, uint32_t segment)`; `int mellum_capture_end_operation(state*, scope*, int status)`; `void mellum_capture_abort(state*, scope*)`; `bool mellum_capture_has_logits(const state*)`.
States FAILED=0, READY, BUSY, FINISHED. Successful sync/eval enters BUSY with explicit phase 1/2 and segment. Sync validates actual complete token prefix against committed count=writer.positions, nonempty prompt, no shortening/change. Eval requires position=writer.positions. Operation end returns original nonzero status (including cancellation=2), otherwise zero only if BUSY with no pending forward and healthy writer/registry. End/abort clear only an owned scope. No-op sync preserves previous logits availability. No reset/retry after failure; abort makes native writer unfinishable, never emits bytes. Errors in FINISHED leave it unchanged.

- [x] Tests: exact header and invalid initialization; initial/identical/append prefix versus shrink/change/null; wrong position; explicit one-token prefill label; BUSY misuse; error/cancellation before frames; terminal callback silence; controller/storage bound. Add minimal stubs, observe RED, commit.
- [x] Implement initialization/prefix/phase/terminal state transitions; run both CPU targets, expect pass; commit GREEN.

### Task 5: Completed forwards and safe finish

**Files:** extend capture-state module/tests; add Python round-trip check.
**Interfaces:** `bool mellum_capture_forward_begin(state*, scope*, uint64_t position, uint32_t rows, bool head)`; `bool mellum_capture_layer(state*, uint32_t layer)`; `bool mellum_capture_forward_end(state*, scope*, bool gpu_success, const uint32_t *tokens, size_t token_count, const uint32_t *selected, size_t selected_count)`; `bool mellum_capture_finish(state*, profile_sink, void*)`.
Begin requires BUSY, no pending forward, contiguous position and valid S2a layout; decode rows=1. Begin opens the owned profile scope. Layer calls acknowledge successful adapter copies, strictly 0..27. End occurs only after the adapter's completed GPU wait (not CPU-provable): validate 28 layers/profile, intern ID and call real writer with stored phase/segment/position/rows. On every error clear owned scope and mark FAILED; no frame from an incomplete forward. Last head flag controls logits availability after operation end. Finish emits profiles before footer, READY/nonempty only; success FINISHED, all errors terminal except already FINISHED.

- [x] Tests: two chunks under one prefill label; decode and one-token append; missing/out-of-order layers; decode multirow; discontinuous/overflowed position; GPU/copy/read error notification; missing roles; short selected/token buffers; active scope cancellation; byte-preserving double finish/mutation; sequential controllers; sink failure in frame/footer; catalog failure prevents footer.
- [x] C `--emit-fixture PATH` emits complete cap=2 prefill/decode/append spool with declared fixture profiles. Python checks actual reader frames, labels/token digest, export round-trip using existing test metadata and exact profile strings; incomplete captures rejected. Observe RED and commit before implementation.
- [x] Implement forward lifecycle and finalization. Run `make test-mellum-capture-state test-mellum-profile`, expect pass; split GREEN implementation into <=150 product-line commits if needed.

### Task 6: Final CPU gate and record

- [x] Run `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum` and explicit C99 builds for new modules: expect all model-free checks pass. No GPU target/build required.
- [x] Record the approved design status, CPU-core completion and remaining owned-session adapter/Metal observation/live gates in roadmap/usage. Do not claim S2b complete.
- [ ] Independent final review of `a97a1a2..HEAD`, including Review Focus and ledger decisions; fix important findings with RED/GREEN, defer minors.
- [ ] Keep branch/worktree; no integration. General `make test` remains unavailable without default DeepSeek model and GPU authorization; report it rather than launch it.

## Self-review

This deliberately plans the independently testable CPU subsystem first, not the whole approved S2b adapter. Remaining requirements (private session ownership, actual encoding sites/source provenance, adapter cleanup and live parity) have no CPU-only substitute and remain visible in the roadmap. Registry and controller share one validated catalog, profile-sink signature and scope-owner contract; no unspecified helper is consumed. Constructors are fresh-storage-only; low-level CPU fixtures do not claim GPU observations.
