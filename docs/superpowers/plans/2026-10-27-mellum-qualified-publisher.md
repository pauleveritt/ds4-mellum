# Mellum qualified capture publisher implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Prepare a CPU-tested publisher that brackets the existing owned Metal proof harness with source/binary consistency checks and exports observed-profile v2 evidence only after successful collection.

**Architecture:** Python owns build/capture orchestration and metadata; the existing native diagnostic harness owns inference, comparison and profile collection. A small receipt writer emits actual session context/cap and existing backend capability queries. The publisher stages everything privately, verifies unchanged inputs and ordered reference routes, then invokes the existing atomic no-overwrite exporter.

**Tech Stack:** Python 3.14+ stdlib, C99/Objective-C, existing Makefile.

**Spec:** `docs/superpowers/specs/2026-10-27-mellum-session-capture-design.md` (approved; particularly Observe encoding, Metadata and harness, Verification).

## Global constraints

- Preserve trained routes, Q8 bytes and matching resident numerical paths.
- No GPU initialization, model process or GPU experiment during implementation; native compilation/linking only.
- Keep `codex/mellum-ssd-replay` worktree; no push, merge or changes to main.
- Python 3.14+ stdlib; C99/Objective-C. No C++, streaming backend or host integration.
- Small focused RED/GREEN commits; approximately 150 product additions per implementation commit.
- This publisher qualifies recorder/source consistency, not representative locality, quality or 8 GB feasibility. Live S2b remains open.

## Implementation decisions within the approved design

Use the existing seven-session proof harness rather than inventing a second inference driver. This initial publisher supports only its teacher-forced initial-prefill/decode/final-append schedule. Generated cohorts need a later driver.

`tools/mellum_capture_identity.py` owns bounded file hashing and source snapshots. Snapshot all Git-listed tracked/untracked nonignored C/H/M/INC/Metal/Python/Makefile inputs, plus every default `metal/**/*.metal`, the Makefile and existing harness token helper. Reject symlink source inputs. Persist per-file digests, HEAD and a canonical aggregate digest; this describes the source state, not a claim that a preexisting binary matches it. Force rebuild with `make -B -j4 tests/mellum_routes_gpu` and retain the command log. Hash the resulting binary and recheck source/binary/token/model-file identity after the subprocess; model bytes are not hashed (record stat identity and caller model ID). Snapshot before build too, so changes during compilation fail. This is consistency evidence, not adversarial attestation; changes restored between checks cannot be detected.

`tools/mellum_capture_publish.py` owns the CLI and pipeline. Arguments: `--model PATH --tokens PATH --first-prefill N --context N --prefill-chunk N --metadata PATH --output PATH [--exact-prefill]`. Repository root is derived from the script. Reject inherited `DS4_*` controls except `DS4_TEST_MELLUM_MODEL`, and inherited `MELLUM_TRACE_*`; no lock bypass. Set only explicit context/chunk/exact controls in the child. Metadata JSON contains a `header` only, no caller execution-profile catalog. Preserve caller artifact/workload labels, but reject conflicts with derived header/build/execution/generation fields. Cap/capabilities come from the native receipt, OS from Python, device label remains caller-declared and identified as such. Generation is teacher-forced with empty sampling settings. Segment labels are fixed IDs 0/1/2; record the concrete schedule separately. Source/binary identities and checked-close artifact hashes are added to the manifest. Read observed `owned.profiles` with strict decimal contiguous IDs, ASCII bounded canonical descriptions and no duplicate IDs; JSON serialization supplies escaping. Derive context/chunk from the receipt and check against the request/spool.

Only the existing native harness executable may be launched by the CLI. Build and capture subprocesses run with repository cwd and private temporary output. A nonzero child status, malformed/missing receipt/profile/footer, source mutation, ordered route mismatch or conflicting metadata leaves no public output. The exporter remains responsible for atomic no-overwrite publication. CPU tests may replace only the subprocess boundary with a synthetic producer; they do not establish GPU correctness. No arbitrary successful old spool can be upgraded by a publish-only command.

## Review Focus

- Source/binary changes during build or capture, deleted/new source inputs and symlinks must reject publication; test snapshots and controlled subprocess mutations.
- Untrusted inherited routing/source/trace controls must fail before building or launching inference; test even empty overrides and replay controls.
- Child failure, incomplete files, duplicate/malformed profiles and close errors must not publish; test synthetic subprocess outputs through the real exporter.
- Caller metadata must not invent observed profiles, capabilities, generation mode or source identity; test conflicts and successful derived-field round trip.
- Existing output paths, including symlinks, must remain untouched; test no-overwrite and temporary cleanup while keeping GPU execution deferred.

### Task 1: Source and binary identity

**Files:** create `tools/mellum_capture_identity.py`, `tests/test_mellum_capture_publish.py`.
**Interfaces:** `file_identity(path: Path) -> dict`, `source_identity(root: Path) -> dict`, `checked_environment(env: dict[str, str]) -> dict[str, str]`. Source result contains revision/files/source_state_id; file result contains sha256/stat identity. Raise ValueError/OSError on invalid inputs. Environment returns a copy, without changing caller state.

- [x] Add real temporary Git fixture tests with literal content SHA-256 checks: stable digest, changed/deleted/new source changes identity; symlink source rejected; modified bytes change binary digest. Add inherited-control rejection tests for source/replay/empty trace settings and allowed unrelated environment.
- [x] Run `python3 -I tests/test_mellum_capture_publish.py`, expect assertion RED (missing module); commit tests separately.
- [x] Implement streaming 64 KiB hashes, canonical JSON source map, deterministic Git listing and environment checks. Run same command PASS; commit GREEN.

### Task 2: Native actual-session receipt

**Files:** create `tests/mellum_capture_receipt.h`, `tests/test_mellum_capture_receipt.c`; modify native harness and Makefile.
**Interfaces:** `bool mellum_capture_receipt_write(FILE *, unsigned context, unsigned cap, bool tensor_tiles, bool pre_m5, bool m5)` writes JSON receipt version 1; caller owns/checks close. No backend initialization in writer. Harness supplies actual session cap and existing backend query values, writes `owned.runtime.json` with exclusive creation and checked close.

- [x] Add C test that writes a real receipt, checks literal JSON and reports output-stream failure; run a minimal false-return writer RED, commit.
- [x] Implement writer and wire actual values in `run_owned` after successful finish/checked spool/profile close. Run receipt test PASS and build harness without execution; commit GREEN.

### Task 3: Checked capture and atomic publication

**Files:** create `tools/mellum_capture_publish.py`; extend Python tests.
**Interfaces:** `publish(root: Path, *, model: Path, tokens: Path, first_prefill: int, context: int, prefill_chunk: int, metadata: Path, output: Path, exact_prefill: bool = False) -> None`; CLI derives root. Existing exporter and sibling modules loaded explicitly for isolated execution. Private `_run` wraps subprocess.run only, permitting synthetic CPU producer tests at that external boundary.

- [x] Add synthetic subprocess integration tests: successful source-checked capture passes real SpoolReader/exporter/replay; observed profiles/identity/capabilities/teacher-forced schedule preserved. Failure matrix: build/capture failure; source/binary/token mutation; missing/malformed receipt; profile duplicate/unknown/missing IDs; truncated spool; reference mismatch; metadata/profile/capability/generation conflicts; existing output/symlink. Assert no public output/temporary residue and early invalid-input rejection. Commit RED after expected failures.
- [x] Implement bounded parsers (metadata <=1 MiB, profiles <=128 lines of <=16 KiB, receipt <=16 KiB), requested controls, build/capture bracketing, observed profile metadata and real export. Split GREEN into focused commits <=150 product additions; rerun tests as interfaces become complete.
- [x] Run `python3 -I tests/test_mellum_capture_publish.py`, expect PASS, and real native compile/link only. Commit GREEN.

### Task 4: Gate, usage and independent review

**Files:** Makefile, ROADMAP.md, docs/MELLUM_ROUTE_CAPTURE.md and retained review record.

- [x] Include Python publisher and native receipt checks in model-free `test-mellum`. Run `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum` and compile/link the native harness; expect PASS with no model/GPU execution.
- [x] Document invocation, caller-declared versus observed/derived fields, limits of source consistency, teacher-forced cohort restriction and outstanding live gates.
- [ ] One fresh read-only review of this stage starting at `01603cc`, with spec/plan/ledger and Review Focus. Fix Important/Critical in one RED/GREEN pass, ledger minors and all excluded behaviors. No second reviewer.
- [ ] Persist evidence/rulings, commit small changes, remove only this plan's scratch and preserve branch/worktree.

## Self-review

The approved publisher requirements map to identity (Task 1), actual runtime facts (Task 2), checked collection/observed profiles/export (Task 3), and verification/documentation (Task 4). No generated workload support, hardware timing or SSD inference is claimed. Python subprocess substitution tests publication behavior only. Device labels and model/tokenizer/template labels are explicit caller declarations; effective functions and backend capability flags are supplied by native collection. Source hashes plus a forced build improve consistency without promising binary reproducibility or hostile-change detection.
