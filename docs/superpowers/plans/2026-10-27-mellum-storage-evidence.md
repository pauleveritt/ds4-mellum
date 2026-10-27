# Mellum storage evidence implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare and measure exact Mellum Q8 expert reads in original and contiguous layouts, with reproducible calibration plans and explicit cache-control limits.

**Architecture:** Bounded Python helpers inspect GGUF metadata, validate expert ranges, pack unchanged bytes and freeze read batches. A separate C99/POSIX executable parses a fixed binary plan and owns timed pread/worker scheduling. Python verifies inputs and complete native reports before publishing evidence; the existing replay supplies LRU semantics without modification.

**Tech Stack:** Python 3.14+ standard library; C99/POSIX, pthreads and existing Makefile. No GPU objects or external packages.

**Spec:** `docs/superpowers/specs/2026-10-27-mellum-storage-evidence-design.md` (approved by the user's subsequent “Proceed”).

## Global Constraints

- Preserve exact Q8 bytes and original model file. No routing, arithmetic, inference, GPU, server, dependency installation or `../ds4-engine` changes.
- Stay on `codex/mellum-ssd-replay`; no push/merge/rebase. Small focused commits, separate behavioral RED commits, roughly 150 new product lines per GREEN commit.
- Python 3.14+ stdlib and isolated `python3 -I`; C99/POSIX for timed reads, no C++/Metal dependency.
- Requires the preceding workload-source plan's `tools/mellum_artifact.py:publish(container, build) -> container/artifact`. Each output uses a fresh exclusive container; consumers require the complete `artifact` child.
- Do not inspect or change ac-demo or open its routing results. All calibration here uses existing tuning/diagnostic evidence; S3b owns generated evaluation.
- No global cache purge, pressure generator or privileged operation. Cache avoidance requested is not proven physical SSD traffic. No tok/s conversion or feasibility verdict.
- Fixed experiment: 512 expert jobs per plan; 2 plans × 2 layouts × 2 cache modes × 3 worker counts × 3 repetitions = 72 runs. Timed logical reads below 256 GiB; all preparation/prewarm/verification bytes separate.
- Real model path: `/Users/pauleveritt/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf`. Real preparation/measurement is explicit CLI work after model-free verification, never part of `make test-mellum`.
- One fresh independent final review covers this plan, followed by at most one Critical/Important fix pass. Record deferred minors and rulings; retain branch/worktree.

## Files and responsibility

Python modules in `tools/`: `mellum_gguf_reader.py` (bounded metadata reads), `mellum_gguf.py` (directory/spans), `mellum_expert_index.py` (Mellum geometry/identity), `mellum_expert_pack.py` (exact-byte payload), `mellum_read_plan.py` (jobs/binary format), `mellum_read_batch.py` (frozen batch/orchestration), and `mellum_storage.py` (thin CLI dispatch).

Use the existing isolated identity/sibling loader rather than modifying sys.path or packaging this repository. All durable JSON is canonical UTF-8 with sorted keys, finite numbers, no duplicate-key acceptance, bounded loading (8 MiB per index/manifest/report) and SHA-256 identities. Reuse existing identity/replay helpers where their contracts match; do not change the checked capture publisher.

Native files in `tools/`: `mellum_read.h`, `mellum_read_plan.c` (wire validation), `mellum_read_io.c` (complete reads/cache controls), `mellum_read_run.c` (workers/timing), `mellum_read_bench.c` (CLI/complete JSON). Corresponding focused tests live under `tests/test_mellum_*`. Build executable `tools/mellum_read_bench` with C99, POSIX/Darwin feature definitions and pthreads only. Add its binary/test binaries to .gitignore and `clean`; do not link CORE_OBJS or initialize an engine.

Extend `docs/MELLUM_STORAGE_EVIDENCE.md`, ROADMAP.md and one retained dated research record. Makefile adds `test-mellum-storage` to the model-free Mellum gate; real commands remain opt-in.

## Review Focus

1. Oversized/overflowing GGUF fields and overlapping expert/non-expert spans must fail before allocation or file access outside bounds (Tasks 1–2).
2. Source/index/payload/plan mutation or path substitution must not produce a successful artifact using mismatched bytes (Tasks 2–4, 8).
3. Partial reads, EINTR, early EOF and errors after some jobs complete must not yield a shorter successful run or unjoined workers (Tasks 6–8).
4. Compound jobs under concurrency must retain distinct buffers, correct offset/byte accounting and honest enqueue/service/end times (Tasks 5–7).
5. Cache-control failure and buffered prewarming must be visible; frozen order, all repetitions and diagnostic-only provenance must survive reports (Tasks 6, 8–9).

## Task 1: Bounded GGUF metadata reader

**Files:** create `tools/mellum_gguf_reader.py`, `tests/test_mellum_gguf_reader.py`.
**Consumes:** seekable BinaryIO plus explicit file size.
**Produces:** `Reader(source: BinaryIO, file_size: int)`, methods `read(n: int) -> bytes`, `u32() -> int`, `u64() -> int`, `string() -> str`, `skip_value(kind: int, depth: int = 0) -> None`, and checked `position`.

Caps: header/directory access within 32 MiB and file size; string length <=1 MiB, arrays <=1,048,576 elements, nesting <=4. Scalar GGUF types 0–12 use their specified widths; string/array are 8/9. Checked seeks may skip bytes but never exceed either bound. Reject malformed UTF-8 and unknown types; do not materialize skipped arrays. Integer products/offsets must fit signed 64-bit POSIX offsets. Caps reject unsupported input, not silently truncate it.

- [ ] Add `test_scalar_and_string_arrays_skip_exactly`, `test_truncated_string_array_and_unknown_type`, `test_limits_apply_to_skips_before_read_or_allocation`, `test_nested_array_and_utf8_limits`. Use tiny real BytesIO fixtures and a seek/read spy that raises on an out-of-bound request. Assert positions/literal bytes and early rejection of huge declared fields.
  Literal reader witness:
  ```python
  reader = Reader(io.BytesIO(struct.pack('<Q', 3) + b'abc'), 11)
  assert reader.string() == 'abc'
  assert reader.position == 11
  ```
- [ ] Run `python3 -I tests/test_mellum_gguf_reader.py`; expect missing-reader RED. Commit `test: bound GGUF metadata reads and skips`.
- [ ] Implement reader/metadata skipping in small GREEN commits if needed, verifying completed test groups explicitly before each commit; full command must PASS before Task 2. Commit `feat: read GGUF metadata within explicit bounds`.

## Task 2: Directory spans and Q8 expert index

**Files:** create `tools/mellum_gguf.py`, `tools/mellum_expert_index.py`, `tests/test_mellum_expert_index.py`.
**Consumes:** Task 1 Reader, existing file/source identity helpers, prior plan `publish`.
**Produces:** immutable `Tensor(name: str, dims: tuple[int,...], type: int, offset: int, bytes: int)` and `Directory(data_start: int, alignment: int, tensors: tuple[Tensor,...], architecture: str)`; `read_directory(source: BinaryIO, file_size: int) -> Directory`; `expert_entries(directory: Directory) -> list[dict]`; `build_index(model: Path) -> dict`; `validate_index(value: dict) -> dict`; `verify_index_source(index: dict) -> None`.

Require GGUF v3, `general.architecture='mellum'`, <=65,536 metadata keys, <=4,096 tensors, 1–4 positive dimensions, power-of-two alignment <=1 MiB (default 32). Reject duplicate metadata/tensor names, split GGUF metadata indicating >1 shard, invalid aligned offsets, header overlap, out-of-file spans and overlapping tensors. Initially support tensor types F32=0, F16=1 and Q8_0=8; Q8 row dim must divide by 32. Compute all tensor spans, not only experts.

Index `schema_version=1`, source path/stat/full SHA-256, directory-prefix SHA-256 through `data_start`, tool/source identity, geometry `{layers:28, experts_per_layer:64, expert_bytes:6580224}`, and ordered `experts` entries `{layer, expert, ranges:[{role,offset,bytes},...]}`. Exact roles gate/up/down; 84 tensors with the approved shapes, three 2,193,408-byte parts, 1,792 entries. Validate every structural/numeric/range/geometry invariant on reloading; input JSON is not trusted because this tool once wrote it. Hash using the same opened regular file; check its fstat/path identity before/after parsing/hashing. Full hash is untimed. `verify_index_source` reopens the recorded source, checks
its identity/full hash, recomputes the actual directory/entries and compares them
with the supplied index. A correct model hash alone must not authorize forged
range offsets. Packing and batch preparation use this verification, not schema
validation alone.

- [ ] Add `test_directory_padding_types_and_nonexpert_overlap`, `test_duplicate_names_bad_alignment_shards_and_overflow`, `test_expert_first_last_offsets_and_role_order`, `test_wrong_shape_type_missing_or_duplicate_role`, `test_index_reload_rejects_forged_geometry_or_spans`, `test_correct_model_hash_cannot_authorize_forged_offsets`, `test_source_changes_during_inspection_fail`. Small GGUF fixtures test actual directory I/O; fabricated Directory objects test all 84 expert tensors without allocating/hashing a 12 GB fixture. Assert entry 0/1791 and independently calculated role offsets/lengths.
  With the fabricated directory's final tensor being layer-27 down:
  ```python
  entries = expert_entries(directory)
  assert len(entries) == 1792
  assert (entries[-1]['layer'], entries[-1]['expert']) == (27, 63)
  assert entries[-1]['ranges'][2]['offset'] == directory.tensors[-1].offset + 63 * 2193408
  ```
- [ ] Commit observed RED tests separately. Implement directory parsing first, then expert/index validation and checked file identity in focused GREEN commits; run `python3 -I tests/test_mellum_gguf_reader.py` and `python3 -I tests/test_mellum_expert_index.py`, expect PASS. Test invalid real file wrappers without faking a successful huge-model hash.
- [ ] Expose CLI `python3 -I tools/mellum_storage.py index --model MODEL --output CONTAINER`, producing `artifact/index.json` through `publish`; add a CLI test on an invalid tiny GGUF asserting no complete output. Real model success belongs to Task 9.

## Task 3: Exact-byte packed artifact

**Files:** create `tools/mellum_expert_pack.py`, `tests/test_mellum_expert_pack.py`; extend CLI.
**Consumes:** Task 2 validated index and original model; Task 1 plan's `publish`.
**Produces:** `copy_entries(source: BinaryIO, destination: BinaryIO, entries: list[dict]) -> list[dict]` (bounded generic byte-copy primitive), `pack(index_path: Path, container: Path) -> Path`; `validate_packed(value: dict) -> dict`;
`verify_copy(source: BinaryIO, destination: BinaryIO, original_entries: list[dict], packed_entries: list[dict]) -> None`;
`verify_packed(index: dict, manifest: dict, packed_dir: Path) -> None`.
Output `artifact/experts.bin`, `artifact/manifest.json` with `schema_version=1`,
`kind='packed-experts'`, original geometry, source index-file/model identities,
`payload={path:'experts.bin',bytes,sha256}`, and ordered `experts` entries
`{layer,expert,ranges:[{role:'packed',offset,bytes}],source_sha256,packed_sha256}`.
Offsets are the cumulative exact expert lengths. Include preparation/verification
byte counts and tool/source identity. `validate_packed` checks complete geometry,
entry uniqueness/order, offsets, equal per-expert source/packed digests and
payload totals; file-content verification remains a separate required step.
`verify_copy` rereads actual ranges with bounded buffers and compares source and
packed bytes/digests. `verify_packed` binds the supplied original index to its
source, checks manifest/index linkage, restricts the payload to `experts.bin`
inside the artifact, then performs full copy/hash/identity verification. Task 8
calls this API rather than treating per-expert claimed digests as proof.

Public `pack` requires the strict real index, calls `verify_index_source` and requires free space >=payload+2 GiB, then streams gate/up/down in ascending (layer,expert), no padding. Buffer limit one expert plus 64 KiB hashing scratch. Verify every written expert against its source concatenation in an untimed full reread, and verify final file size/hash. Checked flush/close precedes publication; retain relative payload paths within the complete artifact and reject escaping paths on consumption.

- [ ] Add `test_copy_noncontiguous_roles_preserves_exact_bytes`, `test_source_and_packed_hashes_match_per_expert`, `test_short_read_write_close_or_verify_failure_never_publishes`, `test_source_index_mutation_and_insufficient_space_fail`, `test_existing_payload_and_escaping_path_refused`. Generic copy/verify tests use real tiny byte streams; public-wrapper boundaries inject only large-model identity/space prerequisites. Assert actual output bytes, no partial valid manifest and bounded read chunk sizes.
  For a tiny source with selected parts `ab`, `ij`, `qr`:
  ```python
  assert destination.getvalue() == b'abijqr'
  assert packed_entry['source_sha256'] == hashlib.sha256(b'abijqr').hexdigest()
  assert packed_entry['packed_sha256'] == packed_entry['source_sha256']
  ```
- [ ] Run `python3 -I tests/test_mellum_expert_pack.py`, expect missing implementation RED; commit tests. Implement copy/verify, then public packing separately, validating the relevant groups and the full suite before completion. Commit messages name exact copying and checked publication separately.
- [ ] Expose `pack --index INDEX --output CONTAINER`. Real payload creation remains Task 9, outside all test gates.

## Task 4: Logical calibration jobs and wire encoding

**Files:** create `tools/mellum_read_plan.py`, `tests/test_mellum_read_plan.py`, `tests/mellum_read_plan_v1.bin`. Task 8 supplies its CLI orchestration.
**Consumes:** validated original/packed indexes; existing replay `load_trace`, `group_ids`, `LRU.serve`.
**Produces:** `uniform_ids() -> list[tuple[int,int]]` (512 distinct, seed 20261027), `lru_decode_misses(trace, capacity_bytes: int) -> list[tuple[int,int]]` (complete prefix replay), `diagnostic_ids(trace_path: Path) -> list[tuple[int,int]]` (first 512 misses), `make_jobs(index: dict, ids: list[tuple[int,int]], layout: str) -> list[dict]`, `encode_plan(jobs: list[dict], file_bytes: int) -> bytes`.
Each job is `{job_id,expert_id,ranges:[{offset,bytes,dest}]}`. `make_jobs` accepts
only layouts `original`/`packed` and invokes the matching strict index/manifest
validator. Packed relative paths resolve within the supplied complete artifact;
no unchecked manifest path selects a different payload.

`diagnostic_ids` requires v2 measured/tuning Mellum geometry, the exact approved trace SHA-256 `75318d1374f8780beb3fb7c5dc880b70afb32786bd976c170d9b7aac0203b2de`, 3 GiB cache/effective 489 slots, and >=512 decode demand misses. Never reset at decode; derive missing IDs from real serve hits, sorted per group. Preserve duplicate expert jobs. Pure `lru_decode_misses` supports tiny synthetic traces for tests; it is not an alternative measured-input validator.

Wire v1: 64-byte little-endian header `struct '<8sIIIIQQQQQ'`: magic `MLREAD01`, version 1, header_bytes 64, job_count, range_count, file_bytes, logical_bytes, max_job_bytes, reserved0=0, reserved1=0. Then 32-byte records `'<IIQQQ'`: job_id, expert_id (`layer*64+expert`), file_offset, length, destination_offset. Consecutive job IDs 0..N-1, 1–3 consecutive records per job, same expert ID per job, destination offsets start zero and concatenate. N in 1..4096, expert IDs 0..1791, positive range lengths, summed job bytes <=max_job_bytes<=6,580,224, all source spans within declared file size and signed 64-bit offsets. Header totals must equal decoded records; exact EOF, no extra bytes. Repeated expert IDs across jobs are legal. This generic byte format supports tiny-file native tests; production plans retain strict Mellum sizes through index validation.

- [ ] Add `test_seeded_plan_is_unique_and_frozen_as_explicit_ids`, `test_lru_replays_prefill_before_decode_and_keeps_repeated_misses`, `test_wrong_trace_digest_split_or_short_decode_refused`, `test_layouts_expand_same_jobs_to_three_or_one_ranges`, `test_wire_literal_golden_bytes_and_all_bounds`. Assert independently known tiny cache misses, not a second LRU implementation; compare encoded bytes against a literal golden fixture and reject mutated jobs/totals.
  Golden binary fixture: 192 bytes; header 2 jobs/4 records/file_bytes=64/
  logical_bytes=10/max_job_bytes=6. Records `(job,expert,offset,len,dest)` are
  `(0,0,0,2,0)`, `(0,0,8,2,2)`, `(0,0,16,2,4)`, `(1,1791,32,4,0)`.
  ```python
  assert encode_plan(jobs, 64) == golden_path.read_bytes()
  assert len(golden_path.read_bytes()) == 192
  assert len(set(uniform_ids())) == 512
  ```
- [ ] Observe/commit RED, implement IDs/LRU job preparation and wire encoding in separate focused GREEN commits. Run `python3 -I tests/test_mellum_read_plan.py`, expect PASS. Do not change existing replay policy or format.

## Task 5: Native wire reader

**Files:** create `tools/mellum_read.h`, `tools/mellum_read_plan.c`, `tests/test_mellum_read_plan.c`; add CPU-only Makefile build/test targets.
**Consumes:** Task 4 wire contract exactly, no Python runtime.
**Produces:** `mellum_read_range {uint64_t offset,bytes,dest;}`, `mellum_read_job {uint32_t expert,nranges; uint64_t bytes; mellum_read_range ranges[3];}`, `mellum_read_plan {uint32_t count; uint64_t file_bytes,total_bytes,max_job_bytes; mellum_read_job *jobs;}`; `int mellum_read_plan_load(const char *path, mellum_read_plan *out, char *err, size_t n)` (0 success); null-safe `void mellum_read_plan_free(mellum_read_plan*)`.

- [ ] Add literal golden input shared with Python and adversarial cases for all Task 4 invariants, byte order, short header/records, >4096 jobs, overlapping destination ranges, inconsistent totals, integer overflow, trailing bytes and allocation/close failure. Assert decoded job/range values and reset/null output on failure. Header/prototype declarations may land with RED tests; missing definitions must cause the expected linker failure before implementation.
  After loading the shared literal golden fixture:
  ```c
  assert(plan.count == 2 && plan.total_bytes == 10);
  assert(plan.jobs[0].nranges == 3 && plan.jobs[0].ranges[2].offset == 16);
  assert(plan.jobs[1].expert == 1791 && plan.jobs[1].bytes == 4);
  ```
- [ ] Commit RED; implement checked header/record parsing and publication of the complete plan in a focused GREEN commit (split supporting validation into its own tested unit only if needed for size). Run `./tests/test_mellum_read_plan`, expect PASS before Task 6. Recompile with C99, warnings and pthreads only; verify no CORE_OBJS in the link command.

## Task 6: Complete range I/O and explicit cache controls

**Files:** create `tools/mellum_read_io.c`, `tests/test_mellum_read_io.c`; extend native header.
**Consumes:** Task 5 job/range types and caller-owned buffer sized >=job.bytes.
**Produces:** `mellum_read_counts {uint64_t bytes,syscalls;}`; `int mellum_read_job_run(int fd, const mellum_read_job*, unsigned char *buffer, mellum_read_counts*, ssize_t (*reader)(int,void*,size_t,off_t), char *err,size_t n)`; `int mellum_read_cache_mode(int fd, bool avoid, char *err,size_t n)`. Production passes `pread`; fault callback is the narrow I/O boundary, not a fake model/backend.

Read each job's parts sequentially into their prescribed buffer offsets. Retry EINTR/partial reads, count every syscall attempt and positive returned bytes, reject zero/oversized/impossible results and EOF. Buffered mode reports explicit prewarming, implemented by running all jobs through this helper outside timing. Avoid mode sets F_NOCACHE=1 and F_RDAHEAD=0 on the actual read fd under __APPLE__; other platforms fail that mode clearly. Failed controls never become buffered success.

- [ ] Add `test_pread_assembles_actual_noncontiguous_bytes`, `test_partial_and_eintr_preserve_offset_and_counters`, `test_eof_and_error_after_progress_fail`, `test_cache_control_failure_is_explicit`, `test_same_descriptor_receives_controls`. Actual temp-file byte assertions plus injected syscall faults; test cache control at a compile-time syscall boundary, not by requiring real SSD behavior.
  For a real 64-byte file filled with bytes 0..63, first golden job:
  ```c
  const unsigned char expected[] = {0, 1, 8, 9, 16, 17};
  assert(memcmp(buffer, expected, sizeof(expected)) == 0);
  assert(counts.bytes == 6 && counts.syscalls == 3);
  ```
- [ ] Observe/commit RED, implement bounded complete reads and platform controls in focused GREEN commits. Run native I/O and plan tests; expect PASS, no GPU use.

## Task 7: Worker execution and complete native report

**Files:** create `tools/mellum_read_run.c`, `tools/mellum_read_bench.c`, `tests/test_mellum_read_run.c`; extend header/Makefile.
**Consumes:** Tasks 5–6; caller-owned read fd, validated plan and worker count 1,2,4.
**Produces:** `mellum_read_sample {uint64_t enqueue_ns,start_ns,done_ns,bytes,syscalls;}` and `mellum_read_summary {uint64_t start_ns,end_ns,prewarm_bytes,buffer_bytes;}`; `int mellum_read_execute(int fd,const mellum_read_plan*,unsigned workers,mellum_read_sample*,mellum_read_summary*,char*,size_t)`.

Allocate/touch one max_job_bytes buffer per worker and create ready/waiting threads before timing; samples bounded by job_count. Release them through a shared start gate; thread-creation failure must release/stop and join those already created. Queue all jobs at batch start (same enqueue_ns, explicitly open-loop), dispatch consecutive job IDs under a mutex, record start after acquisition and completion after all part reads. On any error stop dispatch, join every started worker and return failure; no successful partial report. Threads do not print/hash full buffers. Report samples by job ID with timestamps permitting observed completion order; derive order by completion time then job ID for equal timestamps.

CLI `tools/mellum_read_bench --input FILE --plan PLAN --workers {1,2,4} --mode {buffered,avoid}`. Open regular input read-only, refuse symlink/mismatched size, validate signed-offset compatibility. Capture fd dev/ino/size/mtime_ns/ctime_ns before and after; require equality and check close before emitting JSON. Apply cache mode on that fd. Buffered prewarming is separate from execute/timed totals. JSON version 1 contains only fixed string enums/numeric data: mode/control results, file identity, workers, plan/job/byte counts, summary timestamps, buffer/prewarm bytes and every sample. Check stdout flush; error paths exit nonzero without a complete success document.

- [ ] Add `test_workers_read_each_compound_job_once_with_distinct_buffers`, `test_open_loop_times_are_ordered_and_match_job_bytes`, `test_worker_io_or_thread_start_failure_joins_every_worker`, `test_cli_rejects_changed_size_identity_or_bad_mode`, `test_no_success_report_on_late_failure`. Test 1/2/4 workers on tiny files; validate actual buffer content through a test-only completion boundary, complete set of job IDs, real timestamps and unchanged input. Inject delayed reads to exercise out-of-order completion without asserting brittle exact runtimes.
  Success assertions include every golden job exactly once and:
  ```c
  assert(samples[i].enqueue_ns <= samples[i].start_ns);
  assert(samples[i].start_ns <= samples[i].done_ns);
  assert(samples[i].bytes == plan.jobs[i].bytes);
  ```
- [ ] Commit observed RED. Implement worker allocation/dispatch, failure/join handling, then CLI/report in separate focused GREEN commits, verifying their named groups. Run all native tests and CLI tiny-file integration; expect PASS with no model/GPU objects. Native input tests must also execute under strict C99/warnings.

## Task 8: Frozen batches and checked report aggregation

**Files:** create `tools/mellum_read_batch.py`, `tests/test_mellum_read_batch.py`; finish thin storage CLI and usage.
**Consumes:** Tasks 2–7 and artifact helper.
**Produces:** `prepare_batch(index_path: Path, packed_dir: Path, trace_path: Path, binary: Path, container: Path) -> Path`; `run_batch(batch_dir: Path, container: Path) -> Path`. CLI subcommands `prepare --index --packed --trace --binary --output` and `run --batch --output`.

Prepare records original/packed index-file and model/payload identities (after `verify_index_source` and `verify_packed`), exact logical IDs, human-readable jobs, binary plan SHA-256s, binary/source/compiler-command identity, protocol/cache labels and the 72 explicit invocations. Canonical loop order: plan [uniform,diagnostic], mode [buffered,avoid], workers [1,2,4], repetition [0,1,2]; layouts [original,packed] for even repetitions and reverse for odd. Check total timed bytes <256 GiB before publication. No result-dependent order or repetition changes.

Run verifies prepared identities/binary/plan hashes, full source/packed hashes before batch, fd/stat identity around every run and full hashes after batch. Hashing, prewarming and verification are disclosed untimed reads. Retain every invocation's complete private stdout/stderr/exit status. Native errors abort public success; a separate failure record may retain diagnostics without claiming completion. Parse bounded native JSON strictly, validate exact job IDs, mode/controls/file identity, sample bytes/syscalls, monotonic timestamps and summary counts against each frozen plan. Require total completed bytes equal planned bytes. No success if the binary/plan/input changes during execution.

Report all raw runs, per-job queue/service/end-to-end times, per-run makespan and wall throughput, nearest-rank p50/p95/p99/max with decimal bytes/s and milliseconds. Include role/range lengths, file-offset alignment (filesystem block and 16 KiB), syscalls per job and returned-read-byte/logical-byte ratio, explicitly not physical device amplification. Keep repetitions/cases separate before any aggregates. Host metadata: platform/Python/CPU/RAM, os.statvfs and st_dev for both layouts; optional read-only macOS diskutil info plist when available, recorded as unavailable otherwise. `physical_io_verified=false` unless separate actual device evidence exists; never infer it from cache flags. Record estimated worker buffers and excluded runtime/thread-stack/OS memory. Aggregate output retains source/replay/binary/plan identities and all exclusions.

- [ ] Add `test_frozen_order_has_72_runs_and_exact_byte_budget`, `test_prepare_binds_both_layouts_and_existing_trace_digest`, `test_batch_rejects_input_plan_binary_or_source_mutation`, `test_reject_duplicate_missing_or_inconsistent_native_samples`, `test_nonfinite_duplicate_keys_and_bool_counts_refused`, `test_partial_process_output_or_close_error_never_publishes`, `test_distributions_have_known_nearest_rank_values`, `test_flags_cannot_claim_physical_ssd_reads`. Use actual tiny native integration for byte/timing schema plus a subprocess-boundary fixture for late failures, all CPU-only. Assert complete raw sample retention and JSON identities, not subprocess mocks alone.
  Pin the batch and interpretation with literal assertions:
  ```python
  assert len(batch['runs']) == 72
  assert batch['timed_logical_bytes'] == 242573377536
  assert batch['timed_logical_bytes'] < 256 * 2**30
  assert report['physical_io_verified'] is False
  ```
- [ ] Commit observed RED. Implement preparation, strict result validation, then execution/aggregation in focused GREEN commits; run `python3 -I tests/test_mellum_read_batch.py` and native suites, expect PASS. Document full CLI, cache semantics, memory/read budgets and absence of inference throughput claims.

## Task 9: Model-free gate, final review and real evidence

**Files:** Makefile, docs/MELLUM_STORAGE_EVIDENCE.md, ROADMAP.md and dated research evidence.
**Consumes:** complete prior tasks, original model and approved diagnostic trace; never uses ac-demo routes.
**Produces:** checked original index, exact-byte packed artifact, frozen batch and complete 72-run report, or a specific retained real-run failure without a false S3 verdict.

- [ ] Add `test-mellum-storage` running Python suites/native tests against fabricated files only; include it in `test-mellum`. Run `make -f Makefile test-mellum` serially, expect PASS; run `git diff --check`; verify native build contains no GPU/core objects.
- [ ] One fresh independent final review of this plan's implementation/spec/ledger/Review Focus. Fix Critical/Important once through separate RED/GREEN commits and a green full model-free suite. Record deferred minors and every ruling; no second review.
- [ ] With code frozen, run explicit real index/pack/prepare commands in fresh visualization-directory containers. Require model/trace digests, 1,792 experts, exact payload verification, free-space margin and 72 frozen runs. If real inspection contradicts supported layout, retain the refusal and fix a demonstrated defect through TDD; do not silently broaden types/geometry or reduce workload counts.
- [ ] Execute the frozen batch on the development laptop with no inference process started by this work. Count/report all preparation/prewarming/timed bytes and device/cache controls. Retain failures/repetitions without substituting successful samples. Do not
   automatically repeat a failed batch or exceed the approved cumulative timed
   256 GiB budget; a further batch needs a new identified experiment and budget.
   No privileged cache controls and no extra GPU permission request.
- [ ] Retain commands, artifact paths/digests, host/storage identities, all limits/review rulings and outcomes in a dated evidence record; commit docs separately. Mark S3a preparation/measurements accurately, keep S3 open for representative collection/windows/physical-storage validation. Preserve branch/worktree and artifacts, remove only this plan's completed scratch after retaining ledger.

## Self-review

The spec maps bounded GGUF/index to Tasks 1–2, exact-byte layout to Task 3, fixed jobs/wire to Tasks 4–5, complete native reads/timing to Tasks 6–7, fixed experiment/provenance/statistics to Task 8 and real evidence to Task 9. All five Review Focus risks have named owning tests. Wire struct field order/size, POSIX offset bounds, public signatures, publication
paths and CLI arguments agree across tasks. Index schema validation is distinct
from binding its offsets to the actual source directory; packed schema checks
are distinct from full byte verification. Both bindings are required before
native measurements. No raw-trace slicing, policy tuning, generated collection, physical-cache assertion or streamed-inference prediction is introduced. The workload-source plan handles ac-demo independently. This plan is ready for user plan review; implementation has not started.
