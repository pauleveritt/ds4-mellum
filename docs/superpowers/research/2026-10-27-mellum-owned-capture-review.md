# Mellum owned capture adapter: CPU completion and review

Date: 2026-10-27. Branch: `codex/mellum-ssd-replay`.
Native-stage baseline: `c9ac887`. Reviewed head: `855a6bf`.
Final code head: `2186c35`. CPU/build portion complete; S2b remains open.
No GPU/model process ran during this stage. No SSD performance result follows.

## Independent review and corrections

One fresh read-only review by gpt-6-astra found two Important issues,
no Critical issues and no deferred minors. Both findings were accepted and
fixed in one pass, with separate failing-test and implementation commits.
No second reviewer was dispatched.

1. Standalone Metal consumers omitted the observer/profile link dependencies.
   The real make dry-run regression failed before the fix (`1523e9c`), then
   passed with all five affected executables compiled and linked (`c6b4909`).
   Mellum kernels, Qwen kernels, GLM KDA, GLM attention and MXFP4 Metal were
   built without executing them.
2. Ordinary token-vector growth could terminate the host on checkpoint OOM
   after inference. The API regression demonstrated missing fallible allocation
   before both sync and eval (`cfebc28`). The adapter now reserves checkpoint
   capacity before delegation and fails terminally on allocation error
   (`2186c35`), so ordinary Mellum checkpoint appends need no allocation.

## Verification

All commands below passed; build-only targets do not establish device behavior.

```sh
env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum ds4.o ds4_metal.o ds4_cpu.o tests/mellum_routes_gpu
make -j4 test-mellum-capture-build tests/test_mellum_kernels tests/test_qwen4_kernels tests/test_glm53_kda tests/test_glm_attention tests/test_mxfp4_metal
```

Explicit C99 CPU engine compilation also passed. The real observer and the
linked CPU API boundary tests each passed AddressSanitizer/UndefinedBehaviorSanitizer.
The final CPU API sanitizer run includes the checkpoint OOM regressions.
No broad model-dependent `make test`, model lock bypass or GPU initialization
was performed. The native harness was compiled/linked only.

## Next gates

In an authorized GPU window, run owned-session versus ordinary/reference
route and logit comparisons, intermediate/no-head/head checks, lifecycle/error
and cancellation/isolation cases, actual kernel/fallback profile coverage,
allocation accounting and observer/default-path overhead measurements.
Device allocation failures and cancellation between chunks need additional
live injection coverage. Neither compilation nor supplied-event observer tests
prove these properties.

The qualified source-identity and observed-profile publisher remains a later
CPU stage under the approved design. Current raw spools remain unqualified;
no source identity or requested environment controls substitute for observed
encoding metadata. No streaming backend or host integration is added here.
Branch and worktree remain in place; nothing is pushed or merged.

## Execution ledger and decisions

This retained ledger records implementation scope, rulings and their costs,
verification and the independent review's explicitly deferred behaviors.
The plan's temporary scratch can be removed after this record is committed.

```text
# SDD ledger — plan: docs/superpowers/plans/2026-10-27-mellum-owned-capture-adapter.md
Authorization: user “Proceed” continues the approved owned-session/Metal observer design and prior inline execution. Existing isolated branch verified clean at c9ac887; model-free gate passed immediately before continuation. No additional setup/dependencies needed.
Pre-flight: Task 1 public types -> Task 4 handle: existing opaque ds4_engine and ds4_tokens, sink/cancel types agree.
Pre-flight: Task 2 observer -> Tasks 3/4 graph: controller owns profile scope, observer only borrows it; leave must run before controller destroys scope.
Pre-flight: Task 4 graph -> native writer: actual cap-strided host IDs, committed token prefix/frontier, successful ordinary completion agree with CPU core.
Ruling: continue implementation under the already-approved design and explicit “Proceed,” without repeating design approval — new plan refines that same authorized work and preserves inline execution — cost if wrong: review may require revising implementation details.
Ruling: author live behavioral gates but do not execute their RED/GREEN cycle without GPU authorization — user authorized CPU-only progress and approved design explicitly requires reporting unavailable device gates — cost if wrong: compilation/CPU evidence cannot detect live ordering or numerical bugs.
Task 1: complete (commits f96f7d4..27562ea, tests: make test-mellum-capture-api → Mellum unsupported capture API checks: PASS)
Task 2: complete (commits 27562ea..8da98fa, tests: make test-mellum-observer → Mellum encoding observer checks: PASS)
Task 3: live owned-handle profile/operation-logit assertions authored before encoding hooks. They currently reach the unsupported create stub; no live RED was executed because GPU is not authorized. GPU cleanup/parity remain unproved until actual runs.
Task 2 audit detail: catalog uses the actual embedding function spelling kernel_get_rows_q8_0_f32. Decode MoE records a specialization-defined flag plus effective type/shared/dim/row values, distinguishing undefined constants and their shader fallbacks.
Task 3: complete (commits 8da98fa..ded8a79, tests: make ds4_metal.o test-mellum-observer → Mellum encoding observer checks: PASS)
Task 4 build correction: added object dependency rules had changed default make selection, and command-line CORE_OBJS overrides dropped capture dependencies. Two real make dry-run tests reproduced both failures, RED committed separately, then GREEN by moving rules below product targets and appending/deduplicating mandatory helpers with override. Neither test executes a build or GPU.
Task 4 CPU checks: real ordinary sync cancellation returns status 2 through the adapter without model work; real CPU identical-prefix sync performs no inference, writes no frames and preserves logits availability; changed prefix becomes terminal. Tests use private fixture session state only for boundaries, no GPU abstraction.
Task 4 live checks authored: owned header failure, cancellation-before-first-frame, frame short write, actual NULL-copy and NULL-read rejection followed by normal cleanup; direct no-head then head forwards versus matching ordinary direct forwards. Still not executed.
Task 4: Ruling: make Mellum session/graph host allocations fallible to honor the approved create-failure contract — the reused ordinary path used fatal xmalloc/xcalloc — cost if wrong: failure cleanup must be exercised in the live allocation matrix.
Task 4: complete (commits ded8a79..ce24ff3, tests: make ds4.o ds4_metal.o ds4_cpu.o tests/mellum_routes_gpu test-mellum-capture-api test-mellum-observer test-mellum-capture-build → OK)
Task 5 pre-review gates: full model-free test-mellum exited 0; explicit cc -std=c99 -D_GNU_SOURCE -DDS4_NO_GPU engine compilation passed; observer ASan/UBSan passed. GPU/model code was not executed.
Task 5: complete (commits ce24ff3..855a6bf, tests: env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum ds4.o ds4_metal.o ds4_cpu.o tests/mellum_routes_gpu → OK)
Final review: fresh independent gpt-6-astra review of c9ac887..855a6bf; two Important findings, no Critical or Minor findings.
Final: fixed standalone Metal links — test_standalone_metal_consumers_link_observer_dependencies RED→GREEN (1523e9c, c6b4909), all five binaries compile/link without execution, full model-free suite green.
Final: fixed fatal checkpoint growth — checkpoint_allocation_failure sync/eval RED→GREEN (cfebc28 and current fix), full model-free suite green; C99 API ASan/UBSan green.
Final: Ruling: Physical route/logit parity, intermediate logits and no-head/head behavior — retain the explicit live gate; CPU tests cannot establish device equality — cost if wrong: device parity may fail.
Final: Ruling: Device encoding failures and observer-overflow cleanup — retain the live cleanup gate; static control flow is insufficient device proof — cost if wrong: failure cleanup may retain work or resources.
Final: Ruling: Real GPU allocation/copy/read failure injection — retain the live failure matrix; host checkpoint OOM is fixed separately — cost if wrong: device failures may violate terminal cleanup.
Final: Ruling: Cancellation between chunks and sequential two-handle isolation — retain live lifecycle checks before qualifying capture — cost if wrong: partial operations or stale state may escape.
Final: Ruling: Exhaustive generic/exact/tiled/NAX/tail/fallback coverage — retain the physical path matrix; static encoding audit is preparatory — cost if wrong: some reachable kernels may lack accurate profiles.
Final: Ruling: Recorder-off runtime overhead and actual allocation peaks — require measured default overhead and physical memory accounting — cost if wrong: capture or observer costs may exceed budget.
Final: Ruling: Qualified source identity, runtime overrides and observed-profile v2 publication — leave the approved publisher as a later stage; raw spools remain unqualified — cost if wrong: premature publication would yield unreliable provenance.
Final: Ruling: Concurrent/reentrant and distributed/CUDA capture — retain explicit unsupported contracts; this diagnostic stage supplies serialized Metal capture and CPU stubs — cost if wrong: unsupported callers require a new design and platform validation.
Final: Ruling: Broad model-dependent make test — run the approved model-free gate and compile/link only; no GPU/model authorization — cost if wrong: model-dependent regressions remain undetected until live gates.
Final: deferred minors: none. Branch/worktree preserved; no push, merge or GPU/model execution. Durable review/evidence record will be linked from this plan before scratch cleanup.
Task 5: complete (commits ce24ff3..2186c35, tests: env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum ds4.o ds4_metal.o ds4_cpu.o tests/mellum_routes_gpu → OK)
```
