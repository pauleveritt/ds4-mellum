# Mellum Storage Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure one-worker Q8 expert-read costs in storage-observed batches, or retain a specific failed cache-state method and stop.

**Architecture:** Reuse the native reader, wire format, owned observer, topology mapper and exact classifier. Add a small native aligned scratch writer and Python source-derived manifests, post-read verifier and one-shot service runner. Scratch lives outside publication staging so failure cleanup cannot delete it; no scratch input reads occur before timing.

**Tech Stack:** Python 3.14+ standard library, C99/POSIX, existing macOS descriptor flags and Makefile; no new dependencies or GPU objects.

**Spec:** [Approved design](../specs/2026-10-27-mellum-storage-service-design.md), approved by the user's “proceed” after `5953705`.

## Global Constraints

- Existing `codex/mellum-ssd-replay` branch/worktree; execution baseline is the committed plan. No main integration, rebase, push, GPU, inference, routing/quantization, ac-demo or ds4-engine changes.
- Preserve original model/packed payload and all previous artifacts. Exact expert bytes 6,580,224; page 16,384; block 4,096; stride 6,586,368; padding 6,144. Refuse other real-host geometry.
- Four cells: 64 calibration experts then three 512-expert repetitions; seed 20261027 uniform list, one worker, avoid mode, zero prewarm. Slots sort selected global expert IDs; read order preserves frozen uniform order.
- Timed/staging source reads 10,528,358,400 B each; writes/retained scratch 10,538,188,800 B. Each native read/write class has 16 GiB cap; Python inputs 128 GiB. Require 32 GiB free before staging and each new file.
- Only `consistent-volume` plus successful post-read byte verification unlocks the next cell. Any other bin/failure stops; no retries, substitutes or expanded budget. Reserve before any full payload verification.
- Fresh exclusive scratch inode/cell; uncached aligned writes, fsync, checked close. No scratch read/hash/mmap/prewarm before timing. Keep files on failure, never overwrite/delete/reuse them automatically.
- No sudo, purge, pressure, raw devices, global vnode flags, F_FULLFSYNC, reboot, APFS clone/copy shortcuts, compression request or dependency installation.
- Keep `physical_io_verified=false`. No individual miss/NAND attribution, inference speed, layout comparison, original-GGUF/concurrency extrapolation or physical 8 GB conclusion. 10 tok/s remains a proposed checkpoint.
- Small focused commits; observed behavioral RED committed separately before GREEN; roughly 150 product lines/implementation commit. All real execution waits for full model-free gate and independent review.

## Review Focus

1. Cancellation after READY or during staging preserves child PID/exit, collected evidence, exception type and owned scratch; no further work (Tasks1/2/4).
2. A source/wire/scratch substitution during a check never publishes qualified success (Tasks2/3/4).
3. Normal identity helpers must not read scratch before its timed use; even correct prehashing would ruin this experiment (Tasks3/4).
4. Partial native writes, fsync/close failures or malformed receipts retain known counts and unknown maxima without counting a shorter run as complete (Tasks2/4).
5. Calibration failure, budget exhaustion, low free space or an existing output stops before the next file/GO, preserving reservations/files (Tasks3/4).

## File map and shared contracts

- Modify `tools/mellum_observe_process.py`, `tools/mellum_observe_batch.py` and their tests: evidence-carrying cancellation only; preserve ordinary observed runs.
- Create `tools/mellum_stage.h`, `tools/mellum_stage_io.c`, `tools/mellum_stage_bench.c`: bounded fresh-file writer; link existing `mellum_read_plan.c` and `mellum_read_io.c`, never core/GPU. Tests `tests/test_mellum_stage.c`, `tests/test_mellum_stage.py` and fault-injection headers/helpers as needed.
- Create `tools/mellum_service_plan.py`: metadata-only preparation, deterministic plans, manifests/bindings/forecast/reservation. Tests `tests/test_mellum_service_plan.py`.
- Create `tools/mellum_service_verify.py`: expected-source streaming hashes and post-read source/scratch comparison; no native control or general publisher. Tests `tests/test_mellum_service_verify.py`.
- Create `tools/mellum_service_run.py`: one fixed four-cell state machine using the previous contracts. Tests `tests/test_mellum_service_run.py`.
- Modify `tools/mellum_storage.py`, `Makefile`, `.gitignore`, `docs/MELLUM_STORAGE_EVIDENCE.md`, `ROADMAP.md`; add dated service evidence/ledger in Task5.

A cell definition has `cell`, `kind` (`calibration` or `measurement`), `repetition` (null or 0..2), `expert_ids`, `source_jobs`, `scratch_jobs`, `file_bytes`, `logical_bytes`, `mode='avoid'`, `workers=1`. Both job lists use existing `{job_id,expert_id,ranges:[{offset,bytes,dest}]}` and MLREAD01 encoding. Source jobs sort IDs for sequential scratch writes; scratch jobs retain uniform read order. Slot offset is sorted-ID ordinal times stride; no new wire format.

## Task 1: Retain cancellation evidence

**Files:** observer/process and batch files/tests above.

**Interfaces:** retain `observe(argv: list[str],topology: dict) -> dict` and `ObservationFailure(message: str,evidence: dict)`. Failed evidence gains `error_type` and nonempty `error`, with original exception as cause; batch records that identity too.

- [ ] Add `test_idle_cancellation_keeps_ready_snapshot_and_reaps_child` using actual tiny native fixture and injected `KeyboardInterrupt` in the first idle sleep; assert READY, snapshot, PID, returncode, `error_type='KeyboardInterrupt'`, nonempty error and reaped child. Add batch cancellation test asserting `0.failed.json`, spent reservation, no GO/retry/success.
- [ ] Run `python3 -I tests/test_mellum_observe_process.py` and `python3 -I tests/test_mellum_observe_batch.py`; observe the new failures and commit RED.
- [ ] Route cancellation/unexpected exceptions through evidence capture after guaranteed cleanup. Preserve original type in evidence and cause; do not continue work after interruption. Do not opportunistically change the deferred late-stderr issue.
- [ ] Rerun both complete suites; require PASS and commit GREEN in focused pieces.

## Task 2: Native aligned scratch preparation

**Files:** create `tools/mellum_stage.h`, `tools/mellum_stage_io.c`, `tools/mellum_stage_bench.c`, `tests/test_mellum_stage.c`, `tests/test_mellum_stage.py`; modify `Makefile` and `.gitignore`.

**Interfaces:** in `mellum_stage.h`, define `mellum_stage_counts` with `source_bytes`, `source_syscalls`, `write_bytes`, `write_syscalls`, `slots_completed`, `buffer_bytes`; define `int mellum_stage_copy(int source_fd,int scratch_fd,const mellum_read_plan *source_plan,size_t page,mellum_stage_counts *counts,char *err,size_t n)`. Caller owns descriptors; function zeroes counters, allocates one aligned rounded-slot buffer, reads packed source jobs, zero pads and writes sequential slots. Requires sorted unique experts and one source range/job. Reuse existing plan parser, read job and file-identity helpers.

CLI: `tools/mellum_stage_bench --input PATH --plan PATH --output NEW_PATH`. Production pins 16 KiB pages; output size is `plan.count*round_up(plan.max_job_bytes,page)`. Tiny tests may use smaller job bytes on the same page size. No configurable alternate real geometry or buffered production fallback.

Receipt JSON has `complete`, `source`, `plan`, `scratch` (stat identities), `counts`, `page_bytes`, `stride_bytes`, `cache_controls.nocache`, `fsync_ok`, `close_ok`, start/end monotonic timestamps, `error_type`/`error` on failure. No native hashing library: expected hashes are Task3's Python contract. Receipt is provenance/counts, not proof of written bytes.

- [ ] Add `test_stage_sorted_slots_match_source_and_zero_padding` and `test_stage_rejects_invalid_plan_before_writing`: sorted source experts map to consecutive padded slots; every padding byte is zero; buffer address/length is page aligned; source bytes/calls and write counts match. Reject duplicate/unsorted experts, compound source ranges, overflow and wrong source size before writes.
- [ ] Add `test_stage_faults_retain_partial_output` and `test_stage_cli_rejects_existing_or_substituted_paths`: exclusive creation/symlink refusal, partial reads/writes and EINTR, EOF, cache flag/fsync/close failure, changed source/plan/path, cancellation and partial-file retention. Assert no complete receipt on failure, known partial counters survive where available; never invent zero for unknown work. Use function/syscall injection as in existing read tests, not model/device tools.
- [ ] Add minimal Makefile compile/test targets and ignores for these test deliverables; run `make -f Makefile test-mellum-stage` to observed RED, then commit tests separately.
- [ ] Implement core and CLI in small GREEN commits. Open source without following symlinks, check opened/path identities before/after; create scratch with `O_CREAT|O_EXCL|O_NOFOLLOW`, mode 0600, set F_NOCACHE before first write. Use `posix_memalign`, explicit pread/pwrite loops, zero padding, fsync and checked closes; never truncate, clone or read scratch. Leave partial output owned by this attempt on every failure.
- [ ] Run full `test-mellum-stage` and existing read-plan/read-io tests; require PASS. Capture exact compiler command with only stage/read sources, no engine objects.

## Task 3: Frozen service plans and verification without premature reads

**Files:** create `tools/mellum_service_plan.py`, `tools/mellum_service_verify.py` and matching `tests/test_mellum_service_plan.py`, `tests/test_mellum_service_verify.py`.

**Interfaces:**
- `mellum_service_plan.cells(packed: dict) -> list[dict]` derives exact 64/512 counts and jobs from existing `uniform_ids()`/packed ranges.
- `prepare(prior_batch: Path,reader: Path,stager: Path,container: Path) -> Path` publishes metadata-only `container/artifact`, source/scratch wire plans and bindings; no payload or scratch reads.
- `verify(manifest_dir: Path) -> tuple[dict,list[dict]]` rederives metadata/ranges/caps/binary/source/wire bindings, never full payload bytes. Reuse `mellum_observe_manifest.definitions`, `common.source_identity` and existing binding checks; do not call its old four-cell verifier/reservation.
- `forecast(value: dict) -> dict`; `reserve(manifest_dir: Path,binding: dict) -> None` creates exclusive parent `execution-reservation.json` with every class's reserved maximum and `retry_permitted=false`.
- `mellum_service_verify.expected(payload: dict,cell: dict,packed: dict) -> dict` reads selected source bytes only, in sorted slot order, and computes expected per-expert SHA-256 and full padded-file SHA-256; compare expert hashes to immutable packed manifest. Return expected hashes/bytes/geometry and source binding. This is an explicit budgeted source pass before staging, not scratch prehashing.
- `verify_scratch(payload: dict,scratch: dict,cell: dict,expected: dict) -> dict` performs one bounded post-timing streaming comparison/hash pass over scratch and independent selected source ranges, validates zero padding, digests and before/after descriptor/path identities. Returns verified digests/counts/bindings. No second scratch hashing pass.

Expected-hash prepass adds 10,528,358,400 Python source bytes; it avoids new C cryptography or piping payloads through subprocess stdout. Full forecast is `2*(M+3P) + 2*selected_expert_bytes + scratch_file_bytes + 134217728 + 8589934592`: on declared inputs 136,920,461,888 B < 137,438,953,472 B cap. The two selected-source passes are expected-before-stage and independent comparison-after-timing. Count preparation/source/metadata reads via existing strict nested read budgets; exclude native counters, imports/Git/metadata subprocess internals explicitly. Do not silently increase allowances.

- [ ] Test `test_frozen_cells_have_exact_slot_order_and_totals`: calibration uses first 64 uniform IDs; three repetitions use all 512, sorted slot assignment and unchanged read order. Assert production arithmetic 10,528,358,400/10,538,188,800, forecast 136,920,461,888 and cap 137,438,953,472. Tiny injected fixtures exercise actual wires without model reads.
- [ ] Add `test_prepare_and_verify_are_metadata_only` with payload reads forbidden; escaping paths, altered binaries/source/plans, wrong cells/caps/geometry and changed bindings refuse. Test reservation exclusive and forecast overflow refuses before any payload read; expired reservation stays spent.
- [ ] Add `test_expected_reads_only_source_and_matches_packed_digests` and `test_post_read_verifier_rejects_bytes_padding_and_identity_changes` against source fixtures and existing expert digests. For the expected-hash phase forbid every scratch read. Post-read verification accepts exact content, rejects byte/padding/digest/substitution errors, and consumes exactly one scratch pass plus one source-comparison pass. Check counts and refusal before an over-budget read.
- [ ] Run `python3 -I tests/test_mellum_service_plan.py` and `python3 -I tests/test_mellum_service_verify.py` to RED and commit. Implement these contracts in small GREEN commits, then rerun both complete suites and existing observation-manifest/packed tests.

## Task 4: One-shot service orchestration and CLI

**Files:** create `tools/mellum_service_run.py`, `tests/test_mellum_service_run.py`; modify `tools/mellum_storage.py`, `Makefile`, `.gitignore`, `docs/MELLUM_STORAGE_EVIDENCE.md`, `ROADMAP.md`.

**Interfaces:** `mellum_service_run.run(manifest_dir: Path,container: Path) -> Path`. Exclusively create output root, then persistent `container/scratch`; publish success under `container/result/artifact` or failure/stopped evidence under `container/failure/artifact`. Never put scratch beneath a directory `publication.publish` cleans up. An interrupted output root is spent even if no artifact child exists. A manifest reservation is separately exclusive and survives all outcomes.

Geometry helper: `geometry(topology: dict) -> dict` in `mellum_service_run.py` returns `page_bytes` from `os.sysconf('SC_PAGE_SIZE')` and `block_bytes` from selected driver/media's `Preferred Block Size` in discovery's retained `ioreg -l` plist. Use existing decoder and discovered registry IDs; no new command, default or boot-specific ID. Require 16,384/4,096 before first staging, and validate the eventual scratch mapping too. Add wrong/missing geometry to the preflight-refusal test.

CLI: `service-prepare --prior-batch DIR --reader PATH --stager PATH --output DIR`; `service-run --batch DIR --output DIR` in existing `mellum_storage.py`. Reject non-macOS/Python<3.14 before metadata commands. Nonqualifying completed calibration or later cell publishes failure evidence, exits nonzero, and never launches a replacement.

- [ ] Add `test_complete_service_uses_four_fresh_files_and_preserves_evidence`, a tiny actual writer/reader integration with injected host/topology/counters: calibration and all three cells qualify, exact byte verification passes, every later file is a distinct inode, counts/source hashes/snapshots/raw receipts/native samples/digests are retained. Tiny fixtures may scale IDs/bytes via injected plan contracts; production constants cannot be overridden through CLI.
- [ ] Add `test_calibration_and_later_failures_stop_without_replacement` and `test_preflight_budget_or_mutation_keeps_scratch_and_reservation`: calibration's every nonqualifying bin and verification failure stop before the next file/GO; later failure retains earlier cells and unstarted IDs. Test low free space, malformed/failed writer receipt, source/wire/scratch mutation, budget exhaustion, cancelled writer/observer, existing root/reservation and closing full-verifier mutation. Assert no success, no retry, persistent scratch/partial evidence and known counts plus unknown reserved maxima.
- [ ] Add `test_no_scratch_content_read_before_observed_done`: fail any Python scratch read/file-identity hash before owned reader DONE; staging only writes, discovery/stat checks stay metadata-only, and observe receives `--mode avoid --workers 1`. Check no hashing/staging/artifact writes inside controls.
- [ ] Run `python3 -I tests/test_mellum_service_run.py` to RED and commit. Implement the fixed state machine below in small GREEN commits, preserving shared helpers' existing behavior.

State sequence: inspect processes/host/free space, source topology/geometry and verify metadata; reserve; full existing `verify_packed` then close metadata/bindings; for each allowed cell compute expected source hashes, recheck space/bindings, invoke bounded stager, validate receipt/counts/opened identities without reading scratch, map actual source/scratch device geometry, observe native read, validate consumed source/scratch plans and native samples, classify, post-read verify scratch, retain evidence/cumulative counters, then decide next cell. Stager deadline 120 s, combined captured output1 MiB; use existing bounded command capture, retain failed command and partial scratch, bound receipt JSON before parsing. No stdout payload stream.

Normally completed final or nonqualifying cell: run closing full `verify_packed`, close metadata/source/reader/stager/wire/scratch stat checks, recheck topology and publish qualified success only for four passing cells. On cancellation or I/O/budget failure, stop immediately with no extra payload verification; explicitly record closing verification incomplete. Do not hash scratch during generic binding checks; only Task3's post-read pass may hash it. Release prior raw evidence references before next capture; reuse observer's captured-memory bounds.

- [ ] Add `test-mellum-service` target covering stage and three service Python suites; connect to `test-mellum-storage`; update clean/ignore, usage and roadmap status. Run `make -f Makefile test-mellum` serially, inspect native link commands and `git diff --check`; require PASS before committing integration.

## Task 5: Independent review, bounded execution and evidence

**Files:** `docs/superpowers/research/2026-10-27-mellum-storage-service-evidence.md`, matching `-ledger.md`, usage/roadmap/spec/plan status.

- [ ] Fresh independent review of implementation/spec/plan and five Review Focus items; no real experiment until Critical/Important findings are resolved through observed separate RED/GREEN commits and full model-free gate rerun. Record reviewed range, exclusions and deferred minors. No second reviewer unless a substantive new question requires one.
- [ ] Freeze current reader/stager/source identities after tests/review. Prepare a new service manifest under `/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-storage-service-batch`, using prior `mellum-storage-s3a-batch/artifact`, existing unchanged Q8 payload and current binaries. Bind model SHA `ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b`, size 12925417760, packed SHA `bcce2e85ad8740b9102bf196e4ff2193e2a6db147af23112267bb1774d2bb5a8`, size 11791761408. If those new destinations already exist, stop; never reuse them.
- [ ] Execute that manifest once to new `mellum-storage-service-run`. Approval of this plan authorizes only its declared budget and gates after tests/review. Active inference, insufficient space or unsupported geometry may refuse despite no GPU requirement; retain refusal and stop. Never wait/retry, terminate another process or expand the experiment automatically.
- [ ] Independently audit saved artifacts only: manifest/source/binary/wire/receipt/report/evidence digests, source-derived expected/verified digests, native samples, raw plists, exact counter/idle/classification arithmetic, job order, inode separation, native/Python counts/caps, source/geometry identities, closing-verification status, failed/unstarted cells and reservation. Do not rehash model/payload/scratch as a gratuitous extra pass; recorded bytes were verified within budget. Retain audit script/output and hash index.
- [ ] Commit evidence/status separately. All outcomes end this attempt. If all cells qualify, report native service distributions for this one layout/concurrency in storage-observed batches, descriptive repetition spread and limits. Otherwise report the failed method/remaining question and stop. S3 remains open: original-range/concurrent read costs, representative ac-demo locality, S4/S5 and physical 8 GB validation remain separate. No GPU/ac-demo/engine work in this plan.

## Self-review and handoff

Spec coverage: cancellation→Task1; aligned writer/padding/counters→Task2; source-derived hashes/frozen cells/read forecast→Task3; persistent scratch, exact comparison, gating/budgets/publication→Task4; review/one reserved attempt/audit/limits→Task5. Five Review Focus items have owning tests. Interfaces and wire/job field names agree; existing native reader and classifier stay unchanged. Source expected-hash prepass and persistent output hierarchy are implementation calls within approved layout/budgets, not new experiments.

No product code or experiment was run while planning. Continue with the established inline/native execution method after the user reviews this plan. Use `superpowers:executing-plans`; do not create another worktree, switch branches or start a background job.
