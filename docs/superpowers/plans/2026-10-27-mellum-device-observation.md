# Mellum device observation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Preserve the established inline execution method.

**Goal:** Observe block-driver traffic around four existing CPU expert-read cases, retaining negative/inconclusive outcomes without claiming SSD bandwidth.

**Architecture:** Optional native pipe gates pause after worker preparation and after timed completion. A bounded Python macOS observer maps payload files to a unique physical disk/driver, samples counters and idle controls, and classifies raw deltas. A separate frozen manifest and one-shot runner reuse only S3a's uniform jobs and verified payloads; the original batch is never executed or modified.

**Tech Stack:** Python 3.14+ standard library, isolated `python3 -I`; C99/POSIX, pthreads, existing Makefile. No new packages or GPU objects.

**Spec:** [Approved design](../specs/2026-10-27-mellum-device-observation-design.md), approved by the user's “proceed” after `9110b9c`.

## Global Constraints

- Keep `codex/mellum-ssd-replay` and its current isolated worktree; no push, merge, rebase or main integration.
- Preserve exact Q8 bytes, routes, plans and original model/packed payload. No inference, GPU, server, dependency installation, ds4-engine or ac-demo changes.
- No privileged access, global vnode flags, cache purge, memory pressure, payload rewrite, new layout or automatic fallback/repetition.
- One worker; original-buffered, original-avoid, packed-buffered, packed-avoid, in that order, 512 uniform jobs each. No diagnostic/held-out routes.
- Total timed logical bytes 13,476,298,752; cap 16 GiB. Native prewarm 6,738,149,376; cap 8 GiB. Cumulative preparation/verification cap 128 GiB, separate from native reads.
- Metadata commands: fixed Apple paths/arguments, 32 MiB output cap and 5-second timeout each. Gates: 30 seconds each. Active native completion: 120 seconds. Adjacent idle controls: one second each.
- Keep `physical_io_verified=false`; never convert observed traffic to SSD bandwidth, miss stall, inference tok/s or physical 8 GB suitability.
- Small focused commits; commit observed failing behavioral tests separately before fixes; roughly 150 product lines per GREEN commit. A task may contain several such pairs.
- Full serial model-free gate and one independent final implementation review precede real execution. Stop after this reservation, including failure/inconclusive evidence.

## Files and responsibility

- `tools/mellum_read_gate.c` (new): bounded native pipe protocol; `mellum_read.h`, `mellum_read_run.c`, `mellum_read_bench.c` (modify): optional lifecycle hooks/CLI integration. Ordinary native invocation/report remain compatible.
- `tools/mellum_device_observer.py` (new): bounded metadata subprocesses, topology resolution and timestamped snapshots. No payload reads.
- `tools/mellum_device_classify.py` (new): pure interval/counter validation and diagnostic bins.
- `tools/mellum_observe_manifest.py` (new): immutable four-cell metadata/wire manifest, integrity checks, read forecast and independent reservation.
- `tools/mellum_observe_process.py` (new): owned native child, pipes/deadlines, idle/active observation sequence, bounded in-memory outputs.
- `tools/mellum_observe_batch.py` (new): preflight, before/after verification, orchestration and checked success/failure publication.
- `tools/mellum_storage.py`, `Makefile`, `.gitignore` as needed; focused matching `tests/test_mellum_*` files. Reuse sibling loading, canonical JSON, checked bindings, native report validator and artifact publisher; do not refactor unrelated code.
- Extend `docs/MELLUM_STORAGE_EVIDENCE.md`, ROADMAP.md; retain a dated observation evidence record and execution ledger.

## Review Focus

1. APFS identifiers can change or have several stores/driver candidates: refuse ambiguity and identity changes (Task 2).
2. Snapshot commands can be slow, malformed, oversized or read the same device: retain command intervals, bound memory/time and preserve inconclusive outcomes (Tasks 2–3).
3. Pipe EOF, inherited endpoint leaks or child failure can deadlock or publish incomplete evidence: close unused ends, enforce deadlines and reap only the owned child (Tasks 1, 5).
4. Old manifest provenance or a restored substituted wire can conceal changed consumed inputs: reconstruct jobs from checked indexes and compare native consumed wire identity (Tasks 4, 6).
5. An interrupted run or undercounted verification can exceed the approved experiment: reserve once, forecast all stages before reads, retain failure bytes and refuse reuse (Tasks 4, 6).

## Task 1: Native observation gates without changing read execution

**Files:** create `tools/mellum_read_gate.c`, `tests/test_mellum_read_gate.c`, `tests/test_mellum_read_observed.py`; modify native header/run/bench, Makefile and existing native tests as necessary.

**Interfaces:** Add `mellum_read_observers {int (*ready)(void *,char *,size_t); int (*done)(void *,char *,size_t); void *context;}`. Add `int mellum_read_execute_observed(int fd,const mellum_read_plan *,unsigned,mellum_read_sample *,mellum_read_summary *,const mellum_read_observers *,char *,size_t)`; retain `mellum_read_execute` as a wrapper with NULL observers. Add `mellum_read_gate {int control_fd,event_fd;}` and `int mellum_read_gate_ready(void *,char *,size_t)`, `int mellum_read_gate_done(void *,char *,size_t)`.

- [ ] Add lifecycle regression `test_observed_workers_wait_before_first_read_and_after_join`: with tiny real file and read/completion witness, READY sees prepared workers and zero timed reads, GO permits jobs, DONE sees joined workers, no close/report before RELEASE. Ordinary execute still reads the same bytes/timestamps. Run `make -f Makefile test-mellum-read-run`; observe the intended missing interface/behavior RED and commit `test: pin observation boundaries around native reads`.
- [ ] Implement hooks: READY after buffers are allocated/touched and all workers ready, before setting start/releasing workers; DONE after end timestamp and successful join, before freeing buffers/closing payload. Callback failure stops/releases/joins all workers and refuses success. Run native run tests, PASS; commit `feat: bracket prepared native reads with observer hooks`.
- [ ] Add native protocol tests for R→G and D→L single-byte events/commands; wrong byte, EOF, closed event reader, EINTR and deadline refusal. Fake only clock/poll/syscalls at the existing compile-time test boundary; default production deadline 30 seconds. Observe/commit RED.
- [ ] Implement gate functions with monotonic absolute deadlines and poll, exact one-byte tokens, no partial-success fallback; ignore SIGPIPE only in opt-in observed CLI mode and report failure. Validate distinct inherited pipe descriptors >2, access directions and optional flags as a pair. CLI adds `--control-fd N --event-fd N`; no flags preserves current arguments/report. Run `make -f Makefile test-mellum-read-gate test-mellum-read-bench`, PASS; split protocol/CLI GREEN commits if needed.
- [ ] Add actual subprocess integration `test_no_stdout_before_release`, `test_parent_eof_reaps_without_success`, `test_observed_equals_plain_job_bytes`, malformed/duplicate flags. Use tiny encoded plans, never model files. Run `python3 -I tests/test_mellum_read_observed.py`, RED then GREEN as required. Add gate source to both normal/fault builds; preserve build provenance recipe updates wherever the native source list is declared.

## Task 2: Bounded topology and counter collection

**Files:** create `tools/mellum_device_observer.py`, `tests/test_mellum_device_observer.py`.

**Interfaces:** `command(argv: list[str]) -> dict` returns `{argv,start_ns,end_ns,returncode,stdout:bytes,stderr:bytes}`; `parse_topology(files: dict[str,dict], volume_info: dict[str,dict], apfs: dict, registry: list[dict], disk_info: dict) -> dict`; `discover(files: dict[str,dict]) -> dict`; `snapshot(topology: dict) -> dict`. `files` holds model/payload path and expected stat identity. Use `time.clock_gettime_ns(time.CLOCK_MONOTONIC)` for recorded Python event/snapshot timestamps, matching the native clock; do not assume Python `monotonic_ns` has the same macOS epoch. Snapshot returns raw command result plus `{driver_id,media_id,counters}`; topology retains UUIDs, disk/model/size and raw discovery commands.

- [ ] Write `test_unique_apfs_store_maps_by_media_not_counter_size`, `test_multistore_duplicate_driver_wrong_size_and_st_dev_refused`, `test_changed_driver_or_media_cannot_be_sampled`. Use literal plist fixtures with one nonmatching high-counter driver and correct parent/child tree; require both payloads on the same disk. Assert selected BSD name, UUIDs and driver/media IDs; registry order does not choose the winner. Run `python3 -I tests/test_mellum_device_observer.py`, observe/commit RED.
- [ ] Implement discovery from `/usr/sbin/diskutil info -plist PATH_OR_DEVICE`, `/usr/sbin/diskutil apfs list -plist`, `/usr/sbin/ioreg -a -l -r -c IOBlockStorageDriver`. Require one physical store, whole disk and unique driver child matching BSD Name, Whole and size. Snapshot uses `/usr/sbin/ioreg -a -r -c IOBlockStorageDriver`, preserving selected boot-local ID and raw Statistics. Check fstat/path-derived st_dev against resolved volume; reject unsupported OS/topology. PASS named mapping tests; commit focused GREEN.
- [ ] Add `test_command_bounds_both_streams_and_reaps_timeout`, `test_malformed_plist_missing_bool_or_negative_counter_refused`, `test_snapshot_records_uncertainty_and_raw_bytes`, `test_python_native_clock_intervals_are_comparable`. Use a tiny owned helper process for overflow/timeout; inject command runner for plist fixtures. Require 32 MiB combined captured streams, no unbounded communicate/capture_output, 5-second production deadline, native unsigned 64-bit counters with bool excluded. Observe/commit RED; implement selector-based bounded capture/kill/reap and parsers; full module PASS, commit GREEN.

## Task 3: Honest interval arithmetic and diagnostic classification

**Files:** create `tools/mellum_device_classify.py`, `tests/test_mellum_device_classify.py`.

**Interfaces:** `delta(before: dict,after: dict) -> dict`; `classify(logical_bytes: int,opening: dict,closing: dict,idle_before: tuple[dict,dict],idle_after: tuple[dict,dict],native_start_ns: int,native_end_ns: int) -> dict`. Returns raw D, outer T, both idle inner intervals/rates, B, threshold operands, status/reasons and `physical_io_verified=False`. Inputs are Task 2 snapshots. Missing/malformed evidence returns invalid observation with reasons; never wrapped counter arithmetic.

- [ ] Write `test_counter_reset_replacement_errors_and_retries_invalid`, `test_native_time_must_fit_counter_bracket`, `test_nonpositive_idle_inner_time_invalid`. Require opening.end ≤ native.start ≤ native.end ≤ closing.start; timestamps within each command ordered, idle windows temporally adjacent and correctly ordered. Observe/commit RED using `python3 -I tests/test_mellum_device_classify.py`.
- [ ] Implement validation/deltas and classifications in focused GREEN commits. T = closing.end − opening.start; each idle denominator = second.start − first.end; B = max(idle read rates) × T. Use exact integer/rational comparisons to avoid bin-boundary float drift.
- [ ] Pin `test_diagnostic_bins_at_exact_boundaries`: with L=1000, quiet controls yield D=50 little traffic, D=51 mixed, D=899 mixed, D=900 consistent, D=1250 consistent, D=1251 inconclusive. B=10 is allowed, B>10 background flagged. Invalid outranks background; background outranks low/consistent bins. Retain raw D rather than subtracting B. Test large uint64 values and all classifications retaining false physical verification. Separate RED/GREEN if classification has not yet been implemented; full module PASS before commit.

## Task 4: New frozen manifest, cumulative read forecast and reservation

**Files:** create `tools/mellum_observe_manifest.py`, `tests/test_mellum_observe_manifest.py`; modify `tools/mellum_storage.py` for preparation dispatch.

**Interfaces:** `prepare(prior_batch: Path,binary: Path,container: Path) -> Path`; `verify(manifest_dir: Path, *, payloads: bool) -> tuple[dict,dict]`; `forecast(manifest: dict) -> dict`; `reserve(manifest_dir: Path,manifest_binding: dict) -> None`. `verify` returns validated manifest and jobs keyed original/packed. Preparation CLI: `observe-prepare --prior-batch DIR --binary FILE --output CONTAINER`.

- [ ] Add `test_new_manifest_has_four_uniform_cells_and_exact_caps`, `test_escaping_plan_or_forged_jobs_refused`, `test_prior_batch_never_run_or_modified`. Assert literal run order, workers=1, timed=13476298752, prewarm=6738149376 and 16/8/128 GiB caps. Reject changed prior manifest/wire digests; no diagnostic trace read. Test fixtures use tiny real files and inject only Mellum-size prerequisites. Run `python3 -I tests/test_mellum_observe_manifest.py`, RED; commit tests.
- [ ] Implement metadata-only preparation: bind prior batch/index/packed manifests and uniform job descriptions; structurally validate geometry and old uniform IDs, reconstruct jobs with `make_jobs` and require old wire bytes match `encode_plan`. Copy/re-encode into this new artifact, recording its new wire stat/digest for comparison to native consumption. Bind current binary/source/build recipe independently of the historical batch source. Do not call historical `verify_batch` (requires old source and reads diagnostic data). No payload reads in prepare; mark payload verification pending execution. Commit GREEN after focused PASS.
- [ ] Add `test_forecast_covers_pre_and_post_checks_before_payload_read`, `test_changed_metadata_cannot_expand_allowance`, `test_existing_or_interrupted_reservation_refuses_go`. Reserve exclusively at new manifest container's `execution-reservation.json`, not in the old batch container. Keep retry_permitted=false after failure. Observe/commit RED.
- [ ] Implement forecast using actual bounded verification call graph. Use `verify_packed` exactly once before and once after all four cells; it already verifies source index, full model hash, all expert source/packed bytes and full packed hash. Do not add redundant full input bindings/hashes. Payload base per pass is M+3P; two passes total 96,601,403,968 bytes for approved M/P, plus actual GGUF directory/prefix reads and bounded metadata/tool/source/report reads. Derive/check the precise total from current parser/check paths before any payload read; reserve conservative upper bounds for bounded metadata stages. Forecast and actual cumulative bytes across prepare+run must stay ≤128 GiB. Run tiny-file spies proving stage forecasts dominate actual counted reads and no over-budget stage starts. Commit GREEN with exact forecast explanation in manifest.
- [ ] `verify(payloads=True)` calls strict original/packed byte verification and checks every source/binary/metadata/wire binding before/after; `payloads=False` is for per-cell metadata/stat checks. Never use a correct payload hash to authorize forged offsets. Keep schema_version=1, kind='device-observation-batch', split='tuning', `physical_io_verified=false`, exact four cells and explicit inherited source identities. Full manifest tests PASS.

## Task 5: Observe an owned native child and adjacent idle windows

**Files:** create `tools/mellum_observe_process.py`, `tests/test_mellum_observe_process.py`.

**Interfaces:** `observe(argv: list[str],topology: dict) -> dict`; consumes Task 2 snapshot API and Task 1 pipe protocol; returns native stdout/stderr/exit, argv/events/timestamps, four idle snapshots and two active snapshots. `ObservationFailure` carries the same partial evidence shape and actual progress. Keep stdout ≤8 MiB, stderr ≤1 MiB, event bytes bounded; topology/discovery and six retained snapshot command outputs fit a declared 256 MiB total observation memory cap (refuse overflow).

- [ ] Add real tiny-helper tests `test_ready_idle_open_go_done_close_idle_release_order`, `test_no_artifact_writes_inside_active_or_idle_windows`, `test_child_exit_or_pipe_eof_keeps_partial_evidence`, `test_timeout_kills_only_owned_child_and_reaps`, `test_inherited_unused_end_cannot_hide_eof`. Inject counter sampler and short test deadlines; production constants remain unchanged. Assert sequence R → two pre-idle samples spaced ≥1s → fresh opening → G → D → closing → two post-idle samples spaced ≥1s → L → final JSON. Run `python3 -I tests/test_mellum_observe_process.py`; RED and commit.
- [ ] Implement pipes with explicit `pass_fds`, close unused parent/child ends, bounded selector loop that drains stdout/stderr/events while enforcing absolute deadlines. No blocking unbounded line read or disk-backed capture. Prewarm happens before R; DONE keeps prepared resources until L. Snapshot metadata commands run while native waits; no hashing/logging/publishing between first idle sample and last idle sample. GO only after opening snapshot succeeds. Failed control/snapshot/protocol aborts and reaps this child; never signal another PID. Commit focused GREEN pieces after their tests pass.
- [ ] Add `test_actual_observed_native_report_matches_frozen_wire_and_ranges` using Task 1 binary and literal tiny golden jobs with fabricated device samples. Reuse `mellum_read_batch.validate_native` with expected plan identity and task configuration. PASS process and native observed suites. No test invokes diskutil/ioreg or reads real model data.

## Task 6: Checked batch publication, CLI and model-free gate

**Files:** create `tools/mellum_observe_batch.py`, `tests/test_mellum_observe_batch.py`; modify thin storage CLI, Makefile, usage and ROADMAP.md.

**Interfaces:** `run(manifest_dir: Path,container: Path) -> Path`; CLI `observe-run --batch DIR --output CONTAINER`. Reuse Tasks 2–5; no alternate read path. `preflight() -> dict` records fixed `/bin/ps -axo pid=,comm=,args=` output through bounded Task 2 command handling, identifies ds4 inference/server/CLI/GPU test processes using existing project naming conventions, refuses if present; inspection itself never initializes GPU.

- [ ] Add `test_four_cells_preserve_all_counters_reports_and_failures`, `test_native_wire_swap_cannot_pass_restored_hash`, `test_input_or_topology_changes_prevent_success`, `test_budget_or_active_process_refuses_before_go`, `test_failure_never_retries_or_publishes_success`. Use tiny actual native integration with injected device/process metadata, real checked files and real publication. Assert failed/unexecuted cell IDs, reserved allowance and actual known bytes remain visible; no repeated GO or overwritten artifacts. Run `python3 -I tests/test_mellum_observe_batch.py`, RED then commit.
- [ ] Implement preflight, metadata verification/forecast, unique mapping and full pre-verification outside controls; exclusively reserve four-cell budget before launching the first native child (and therefore before prewarm or GO). Validate each native report/consumed wire identity using existing validator; classify with Task 3. Persist each cell's RAM evidence only after controls finish. Perform full post-verification and topology recheck before successful publication. Recheck source/binary/wire/input metadata before/after every cell. Unexpected failures stop remaining cells and publish a separate `.failure/artifact` via existing conventions; reservation survives. Never claim unknown partial native timed/prewarm bytes as zero: retain reserved maxima plus known progress and explicit unknown counts on early child failure. Commit GREEN in small pieces.
- [ ] Add all Python/native suites to `test-mellum-storage`; update clean/ignore and fixed compiler recipe for new gate source. CLI rejects observe commands below Python 3.14 or unsupported platform before metadata execution. Document exact opt-in commands, budgets, cumulative accounting exclusions, thresholds, failure handling and distinction between block traffic/file attribution/NAND service. No tests use actual metadata tools, GPU or payloads.
- [ ] Run `make -f Makefile test-mellum` serially and `git diff --check`; PASS required. Inspect native link command: only read/gate C sources and pthreads, no engine/GPU. Commit gate/docs integration. Retain tests/build outputs for final review.

## Task 7: Independent review, four-cell execution and evidence handoff

**Files:** dated `docs/superpowers/research/2026-10-27-mellum-device-observation-evidence.md` and matching `-ledger.md`; usage/roadmap/spec status.

- [ ] One fresh independent final review of implementation, approved spec, this plan and all five Review Focus items. Resolve Critical/Important findings through observed separate RED/GREEN commits, then rerun the full serial model-free gate; retain rulings and deferred minors. Do not opportunistically fix prior unrelated deferred diagnostics.
- [ ] Freeze current source/binary after tests/review. Prepare one new manifest under the task visualization root with a new, unused `mellum-device-observation-...` container. Prior batch is `mellum-storage-s3a-batch/artifact` under `/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4`. Require original model SHA `ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b`, size12925417760; packed SHA `bcce2e85ad8740b9102bf196e4ff2193e2a6db147af23112267bb1774d2bb5a8`, size11791761408. No copying/repacking payloads, new trace or historical batch run.
- [ ] Execute that manifest once via observe-run after read-only preflight and forecast pass; this is the approved four-run CPU diagnostic, not another approval question. If preflight refuses or execution fails, retain the refusal/failure and stop. No waiting loop, automatic retry, replacement cell or expanded budget.
- [ ] Independently check published artifact completeness/digests and arithmetic from raw evidence. Retain exact commands, build/source identities, host/mapping/raw snapshots, all bins, native timings, read accounting, errors/unknowns and idle/snapshot uncertainty. Report raw device deltas without attributing them to NAND or dividing them by native runtime. Do not infer layout performance from single repetitions.
- [ ] Commit evidence/status docs separately, preserving old S3a artifacts and clean branch/worktree. All bins end this experiment. State what the results permit proposing next; do not implement an aligned/cold-cache/streaming variant or begin ac-demo collection.

## Self-review and execution handoff

Spec mapping: native gates → Task 1; discovery/snapshots → Task 2; controls/interpretation → Tasks 3/5; independent manifest/budgets → Task 4; checked orchestration/failure → Task 6; review/execution/report → Task 7. Review Focus has owning regression tests. Schema names, pipe tokens, snapshot intervals and function signatures agree across tasks. Ordinary read API is retained, current report validation is reused, and the spent reservation cannot be consumed. Preparation remains metadata-only; two full byte verification passes fit the cap without weakening existing source/payload checks. The exact parser/metadata allowance is an execution prerequisite, not permission to increase the cap.

This plan is ready for user review. Inline execution is preserved from the preceding milestone. No implementation, new payload reads or new timed experiment has started.
