# Mellum SSD Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the reviewed CPU-only routing trace replay tool with synthetic tests.

**Architecture:** A standalone standard-library Python tool validates complete JSONL traces before replaying grouped LRU and a relaxed optimal bound. Structured reports distinguish references, unique group demand and decode-token statistics.

**Tech Stack:** Python 3.14+ standard library, unittest function tests, Make.

**Spec:** `docs/superpowers/specs/2026-10-27-mellum-ssd-replay-design.md` (approved in chat).

## Global Constraints

- No GPU, model loading, SSD benchmark, inference-code changes or upstream rebase.
- Preserve ordered routes. Use cold caches, complete-expert identities and byte capacities.
- Default sweep: 2, 2.5, 3 and 3.5 GiB; Mellum geometry: 28/64/8, 6,580,224 bytes/expert.
- Another agent owns upstream integration; rebase this branch only after its integrated Mellum revision is ready.

## Review Focus

- Partial/truncated forwards and malformed scalar types must fail with line numbers.
- Group pinning must prevent false hits and evictions due to router order.
- Bypass must participate in optimal eviction; compare with exhaustive search.
- Prefill union demand and decode-token statistics must remain distinct.
- CLI errors must not publish partial successful reports.

### Task 1: Validated trace contract

**Files:** Create `tools/mellum_ssd_replay.py`, `tests/test_mellum_ssd_replay.py`.

**Interfaces:** `load_trace(lines: Iterable[str]) -> Trace`; immutable Trace/Header/Request values; `TraceError(ValueError)` with line diagnostics. Header fields: schema_version, capture_kind, model_id, engine_revision, numerical_mode, workload_id, geometry. Geometry fields: layers, experts_per_layer, top_k, expert_bytes. Requests use the spec's exact field names.

- [x] Write function tests for valid ordered routes, full forward consistency, provenance/geometry and line-specific malformed/truncated input errors.
- [x] Run `python3 -I tests/test_mellum_ssd_replay.py`; expected missing-feature failure.
- [x] Implement strict complete-trace validation, rejecting duplicate JSON keys and nonstandard non-finite JSON numbers too.
- [x] Run the dedicated suite; expected all task-1 tests pass. Commit.

### Task 2: Cache replay and statistics

**Files:** Extend the tool and its tests.

**Interfaces:** `replay(trace: Trace, capacity_bytes: int, policy: str) -> dict`, policies `lru` and `relaxed-optimal`. `run_sweep(trace: Trace, capacities: Sequence[int]) -> dict` returns provenance/assumptions and all capacity/policy reports.

- [x] Write hand-derived tests for cold/warm loads, layer identity, byte-flooring, group protection, oversized groups, prefill unions, phase retention, decode percentiles and ordered series. Add exhaustive tiny-state reference tests for relaxed optimal and bypass.
- [x] Run suite; expected failures for absent replay functions.
- [x] Implement deterministic grouped LRU, optional-bypass farthest-next-use relaxed replay, and common reports. Clarify group hits as all-hit groups; row hits are relative to pre-group cache. Bound metrics are explicitly relaxed.
- [x] Run dedicated suite and `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum`; expected pass. Commit.

### Task 3: CLI, documentation and integration

**Files:** Extend tool/tests and `Makefile`; create `docs/MELLUM_SSD_REPLAY.md`.

**Interfaces:** `python3 tools/mellum_ssd_replay.py TRACE [--cache-gib GIB ...]` emits one JSON document only after every replay succeeds. Decimal GiB parsing is exact and positive; failures use argparse-style stderr and nonzero exit without traceback.

- [x] Write subprocess tests for default sweep, fractional capacity, malformed/truncated trace, unsupported groups, bad capacity and missing files; require empty stdout on every failure.
- [x] Run suite; expected CLI failures.
- [x] Implement CLI and dedicated Make target; add target to model-free Mellum suite. Document executable synthetic example, schema, metric semantics and limitations.
- [x] Run dedicated Make target and model-free Mellum suite; expected pass. Run `git diff --check`. Commit.
- [x] Obtain one independent whole-branch review, address important findings with failing regressions, and rerun relevant CPU checks.

## Execution record

User explicitly authorized proceeding with the reviewed design; execute inline without another approval gate. After the user reported the other agent's completed Checkpoint 3.6, replay commits were rebased onto `59932c9e44802b7e869c1c76a7a02e72aa19c8b3`, which includes upstream `ee7e840` and the Mellum port/E1. This is local `mellum`, while local `main` remains raw upstream. Keep the branch/worktree available for the next GPU phase. No steps 34–40 or model measurements were performed here.

Post-rebase verification: `git range-diff` confirms all five replay-only
commits applied unchanged. `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum`
passes on the new base, including 21 replay tests and 14 logits-comparison
tests. The working tree is clean after the baseline-record update.

Task commits after the rebase: `5ac4062` (validation), `a978a2f` (replay), `ab58f75` (CLI/docs/Make).
Each task's tests failed on missing functionality before implementation and
passed afterward. Final verification: 21 replay tests; model-free
`make test-mellum` passed, including 14 logits-comparison tests. The documented
synthetic example was executed and all eight reports matched 224 cold loads,
then zero warm loads. These are synthetic accounting checks only.

Independent code review: no Critical or Important findings. The status above
is updated as completion bookkeeping. Deferred minor: an explicit oversized
prefill-union test where each individual row fits (e.g. `[0,1]`, `[1,2]` with
two slots); the current implementation already rejects the union correctly.

The Python floor was raised to 3.14+ during subsequent plan review; the
standard-library-only constraint remains. The original
decision to preserve the Mellum base until upstream integration completed
has now been fulfilled by the clean rebase onto `59932c9`. Reject oversized groups
under the approved atomic model (tiling remains future work); cold start means
the first full forward; provenance is producer-declared and other allocations
remain unmodeled. GPU, storage-performance and native integration claims are
deferred to their actual experiments, so this work establishes no throughput
or whole-machine memory result.
