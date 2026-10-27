# Mellum SSD Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Qualify cache replay claims and preserve reproducible, complete capture evidence before collecting real Mellum routes.

**Architecture:** Extend the existing standalone replay tool with additive result interpretation, a version 2 manifest and report identities. Keep both cache algorithms unchanged. Use synthetic traces and the existing stdlib test harness; native capture and streaming follow separate milestones.

**Tech Stack:** Python 3.14+, standard library, JSON Lines, unittest function tests, Make.

**Spec:** [S1 evidence design](../specs/2026-10-27-mellum-ssd-evidence-design.md), including its manifest table and linked evaluation protocol.

## Global Constraints

- Python 3.14+, standard library only; preserve standalone `python3 -I` execution.
- Use synthetic fixtures only. No GPU use, model loading, inference, storage benchmarks or inference-code changes in S1.
- Preserve Q8 bytes and trained routing.
- Small commits: one concern each, failing behavioral tests committed red separately, at most about 150 lines of product code per implementation commit.
- Do not push or merge as part of S1.
- Work from this isolated branch based on Mellum `59932c9` (upstream `ee7e840` plus E1).
- Another agent owns port integration; steps 34–40 stay parked.
- Keep v1 parsing/report shape compatible, except explicitly reject a v1 `capture_manifest` instead of discarding it.
- 10 tok/s remains a proposed checkpoint; no feasibility verdict or latency conversion.

## Review Focus

- Complete-forward truncation: reject a syntactically complete prefix whose declared extent describes more capture (Task 3).
- Nested extensions and JSON exponent overflow: preserve extensions and reject non-finite numbers anywhere (Tasks 2–3).
- Version 1 metadata compatibility: retain existing report header shape and label legacy measured provenance (Tasks 3–4).
- Line endings/encoding: digest exact file bytes, decode once and publish no report on invalid UTF-8 (Task 4).
- Context/chunk configuration mismatch: reject captures whose positions exceed context or whose prefill rows exceed the declared chunk (Task 3).

---

## File responsibilities and execution

Modify `tools/mellum_ssd_replay.py` in place: validation, replay and CLI remain
one standalone file; no import/package restructuring. Modify
`tests/test_mellum_ssd_replay.py`: synthetic builders, independent witness
policies/oracle and CLI checks. Update `docs/MELLUM_SSD_REPLAY.md` only when
behavior lands; mark the design, plan and `ROADMAP.md` complete only after
verification. No Makefile change is needed.

Run `python3 -I tests/test_mellum_ssd_replay.py` for every red/green cycle.
The harness wraps all `test_*` functions using `load_tests`; do not assume
pytest or direct unittest selection by function name. A red run must fail
on the new behavior, not imports, malformed fixtures or unrelated failures.
Commit only the tests before implementing the behavior. If a task would
exceed the product-code limit, split its failing test/fix pair by behavior
before editing; do not silently combine concerns.

### Task 1: Scope lower-bound interpretation and pin counterexamples

**Files:** Modify `tools/mellum_ssd_replay.py` (`replay`, `run_sweep` assumptions); test `tests/test_mellum_ssd_replay.py`.

**Interfaces:** Consumes `load_trace`, `replay(trace, capacity_bytes, policy)`,
`RelaxedOptimal(slots, groups).serve(required)`, test `optimal_misses(sequence, capacity)`.
Produces additive `result['interpretation']`, with exact fields/values from
the design's table. No policy or accounting signature changes.

- [x] Add `test_bound_interpretation_and_phase_counterexample`: build a one-layer, top-1 trace with prefill A,B,C,B,C then 19 decode A groups; expert size 10, capacity 20. Assert relaxed total bytes 40, prefill 30, decode 10, decode p95 bytes 10. Add a test-only causal pin-A/rotate-other simulation; assert total bytes 50 and all decode bytes/p95 zero. Assert the complete interpretation objects for both policies equal the design table, including `diagnostics_are_bounds == False`.
- [x] Add `test_sorted_bound_excludes_flexible_group_order`: groups `{0,2},{1,2},{0,1}`, 2 slots. Assert relaxed replay loads 4, `optimal_misses((0,2,1,2,0,1), 2) == 4`, and `optimal_misses((0,2,2,1,0,1), 2) == 3`. Assert `order_flexible_tiling_covered == False`. Use one layer/top-2 and 10-byte experts.
- [x] Add `test_prefill_union_exceeds_capacity_even_when_each_row_fits`: one prefill with rows `[0,1]`, `[1,2]`, top-2, two slots; both policies raise `TraceError` mentioning `group requires 3` and `unsupported by atomic group residency`.
- [x] Run the CPU script; expect new metadata assertion failures. Commit tests: `test: expose limits of whole-trace SSD cache bounds`.
- [x] Add the interpretation object to `replay` and qualify `run_sweep` assumptions. Leave eviction, sorting, admission and distribution calculations untouched.
- [x] Run the CPU script; expect all tests pass. Run `git diff --check`. Commit product change: `fix: scope replay bounds to total demand and declared schedules`.

### Task 2: Validate the capture manifest as a standalone contract

**Files:** Modify `tools/mellum_ssd_replay.py` (new validation helpers); test `tests/test_mellum_ssd_replay.py`.

**Interfaces:** Consumes existing `fail`, `integer`, `string_field`, `object_value`.
Produces `validate_capture_manifest(raw: object, header_fields: dict) -> dict`.
`header_fields` contains `model_id`, `engine_revision`, `numerical_mode`.
Return the full validated manifest; raise line-1 `TraceError` naming the path
on error. This task does not connect the helper to trace loading yet.

- [x] Add test helper `manifest_fixture(metadata: dict, *, forwards=1, positions=1, first=0, last=0) -> dict`. Populate every required field from the design table, matching the three header identities; use `manifest_version=1`, `complete=True`, context 8192, chunk 1024, generated mode, tuning split, zero-filled 64-character digests and nonempty `fixture-*` strings. Settings include `seed: None`; capabilities may be empty. Extent uses the arguments.
- [x] Add `test_manifest_preserves_extensions`: insert nested extension objects/lists/nulls and assert `validate_capture_manifest(manifest, metadata) == manifest`. Deep-copy the expected fixture so mutation/loss cannot pass unnoticed.
- [x] Add `test_manifest_rejects_missing_wrong_and_inconsistent_fields`: table-drive every required field's removal, wrong section/scalar types, blank identity strings, invalid enums, version 2, `complete=False`, count booleans/zero/negative values, bad/uppercase digests and each identity mismatch. Assert line 1 plus the offending path; reject non-finite floats in required objects and nested extensions. Unknown finite extension values remain accepted.
- [x] Run the CPU script; expect missing-helper failures. Commit tests: `test: define complete versioned SSD capture manifests`.
- [x] Implement the helper and small shared recursive finite-JSON validation using stdlib `math.isfinite`; accept only JSON value types and string object keys. Enforce the design table without verifying declared artifact/hardware truth. Do not add dependencies or access model files.
- [x] Run the CPU script and `git diff --check`. Commit: `feat: validate preserved SSD capture manifests`.

### Task 3: Load v2 traces and verify capture extents

**Files:** Modify `tools/mellum_ssd_replay.py` (`Header`, `parse_header`, `parse_object`, `load_trace`, report header construction); test `tests/test_mellum_ssd_replay.py`.

**Interfaces:** Consumes Task 2's validator. Produces
`Header.capture_manifest: dict | None = None`; unchanged
`load_trace(lines: Iterable[str]) -> Trace`. `run_sweep` omits the null
manifest key for v1 and includes the full object for v2.

- [x] Change the existing unsupported-schema fixture from version 2 to 3; version 2 without a manifest now fails specifically for `capture_manifest`.
- [x] Add `test_v2_manifest_round_trip_and_v1_compatibility`: load synthetic and Mellum-shaped measured v2 fixtures; assert routes unchanged and `run_sweep(... )['trace']['capture_manifest'] == manifest` including extensions. Assert an existing v1 report's `trace` equals its original header exactly. A v1 header containing a manifest must fail on line 1 with a version-2 requirement. Reject version 3 and boolean versions.
- [x] Add `test_v2_capture_extent_and_execution_checks`: valid multiphase trace beginning at position 42, one two-row prefill then one decode (forwards 2, positions 3, first 42, last 44). Check accepted extent. Remove the final full forward without changing the manifest: reject at next expected line. Independently alter each extent field; reject at EOF with the field name. Check context 44 rejects last position 44, context 45 passes; chunk 1 rejects the two-row prefill and chunk 2 passes. Check booleans do not pass integer fields. Preserve existing incomplete-layer diagnostics.
- [x] Add `test_nested_json_exponent_overflow_is_rejected`: parse raw JSON containing `1e999` in manifest extensions and an otherwise ignored v1 header extension; assert line-1 failure. Pin nested duplicate-key rejection too. This closes the decoder's finite-exponent gap without losing finite extension values.
- [x] Run the CPU script; expect version/extent/preservation failures. Commit: `test: reject incomplete and inconsistent version 2 routing captures`.
- [x] Integrate v1/v2 selection and manifest preservation; apply recursive JSON checking after decode. After normal structural validation, derive forwards/positions once per layer-zero request, compare all extent fields and context/chunk constraints as designed. Preserve v1 report shape by omitting its null manifest. Header errors use line 1; EOF extent mismatches use next expected line; context/chunk mismatches name the relevant request line and manifest path.
- [x] Run the CPU script and `git diff --check`. Commit: `feat: preserve version 2 provenance and verify routing capture extents`.

### Task 4: Identify the input snapshot and replay implementation

**Files:** Modify `tools/mellum_ssd_replay.py` (`run_sweep`, `main`, new identity helper); test `tests/test_mellum_ssd_replay.py`.

**Interfaces:** Extend
`run_sweep(trace: Trace, capacities: Sequence[int] = DEFAULT_CAPACITIES, *, trace_sha256: str | None = None) -> dict`.
Add `report_identity(trace: Trace, trace_sha256: str | None) -> dict` returning
the exact `evidence` keys from the design. Uses SHA-256 of `Path(__file__)`
contents and `platform.python_version()`; errors use the existing CLI path.
Null trace identity means original bytes unavailable. Supplied identities are
syntax-validated declarations, never reconstructed hashes.

- [x] Add `test_report_identity_without_raw_input`: programmatic sweep has null `trace_sha256`, source hash equals `hashlib.sha256(SCRIPT.read_bytes()).hexdigest()`, Python version equals `platform.python_version()`. Check statuses: all synthetic versions `synthetic-declared`; measured v1 `legacy-declared`; measured v2 `manifest-declared`. Invalid non-null digests (including bool, empty, uppercase and short) raise `ValueError`.
- [x] Add `test_cli_hashes_exact_bytes_and_preserves_manifest`: write equivalent UTF-8 JSONL fixtures with LF and CRLF bytes using `write_bytes`, run the existing isolated CLI, assert different expected exact-byte hashes but identical parsed trace headers and policy results. Assert full manifest/extensions and source identity survive the CLI path; v1 header shape remains unchanged. Do not hash text-mode normalized reads in the test.
- [x] Add `test_cli_invalid_utf8_emits_no_report`: write invalid UTF-8 bytes, assert nonzero exit, empty stdout, nonempty stderr and no traceback. Retain the existing no-partial-output checks for a later failing sweep capacity.
- [x] Run the CPU script; expect absent evidence/keyword failures. Commit: `test: require exact trace and replay identities in SSD reports`.
- [x] Implement report identity using `hashlib` and `platform`. In `main`, read bytes once, hash that snapshot, decode UTF-8 and pass `io.StringIO(decoded, newline=None)` to `load_trace` (corrected by final review below); pass the digest into `run_sweep`. Update CLI help to versions 1/2. Keep report serialization after all capacities succeed. Do not reread the input or infer a digest from the parsed dataclasses.
- [x] Run the CPU script and `git diff --check`. Commit: `feat: identify exact routing inputs and replay source in reports`.

### Task 5: Document the shipped contract and record the gate

**Files:** Modify `docs/MELLUM_SSD_REPLAY.md`, `ROADMAP.md`, the S1 spec and this plan. No product changes.

**Interfaces:** Documents Tasks 1–4's actual JSON, referencing the manifest
table and `mellum-ssd-evaluation-v1`; no new behavior.

- [x] Update usage for versions 1/2, required manifest, report `interpretation`/`evidence`, declared provenance, atomic/fixed-order limits and exact-byte digest semantics. Keep the v1 synthetic example valid; add a compact v2 construction using the spec table rather than implying a native recorder exists. Link evaluation rules and explicitly retain append-only lifecycle limitations.
- [x] Run `make test-mellum-ssd-replay`; require all synthetic tests pass. Then `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum`; require the full model-free target exits zero. Do not enable model or GPU targets. Stop for unexpected failures and diagnose before claiming completion.
- [x] Review the product diff: no changes to cache algorithms or inference files; no extra dependencies; each implementation commit respects the product-line limit. Check every changed local Markdown link exists and run `git diff --check`.
- [x] Record actual commands, test count and commit IDs in this plan's execution record. Update S1 status only if every gate passes. State that no model, GPU or storage measurement occurred. Commit: `docs: record qualified SSD evidence contract and CPU verification`.

## Plan self-review

Spec mapping: bound objective/schedule and union regression → Task 1;
manifest structure, preservation and declarations → Tasks 2–3;
completeness/context/chunk → Task 3; trace/source identity and CLI failure
handling → Task 4; current usage/evaluation references and acceptance → Task 5.
All five Review Focus items have explicit regression owners. The only new
cross-task product interface is the manifest validator, consumed by header
parsing; existing replay algorithms and signature remain unchanged. Later
native scheduling, memory peaks, lifecycle events and workload selection are
explicitly deferred, not missing implementation tasks.

## Execution record

Implemented natively after user review; Python floor updated to 3.14+ before
execution. Python 3.14.7 used throughout. Design builder check found no material
unanswered questions. No deviations or rulings in Tasks 1–5.

| Task | Red test commit | Implementation commit | Green replay count |
| --- | --- | --- | --- |
| 1: bound scope/witnesses | `fbf36e5` | `42c909c` | 24 |
| 2: manifest contract | `15d73bd` | `95c8ab8` | 26 |
| 3: v2/completeness | `3367900` | `e6cf375` | 29 |
| 4: input/source identity | `2765766` | `6b4d92e` | 32 |

Each red run failed on the missing behavior; each green run passed the complete
replay script. Implementation commits add 11, 76, 29 and 25 product lines,
respectively, with no cache-algorithm or inference-code changes.

Verification: `make test-mellum-ssd-replay` passes 32 tests;
`env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum` exits zero, including the
32 replay tests and 14 logits-comparison tests. Both documented synthetic
examples replay successfully; changed local Markdown links and whitespace
are checked. No model, GPU, storage or native streaming experiment occurred.
Independent whole-branch review (`59932c9..1593748`) found one Important
issue: `str.splitlines()` splits U+2028/U+2029/U+0085 inside valid JSON strings.
No Critical/Minor findings or declined judgments. Corrected the plan's Task 4
mechanism to `io.StringIO(decoded, newline=None)`, preserving the same byte
snapshot/hash and JSONL record boundaries. Compatibility risk is record
splitting; existing LF/CRLF and new v1/v2 Unicode regressions cover it.

Separate red commit `5cbbcfa` reproduces the CLI failure; fix `1288b0e` passes
all 33 replay tests. The final `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum`
gate exits zero, including 33 replay and 14 logits-comparison tests. The single
review fix pass is complete with no unresolved findings. S1 is complete;
branch/worktree retained for S2. Nothing pushed or merged.
