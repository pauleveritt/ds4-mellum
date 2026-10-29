# Mellum Router Capture Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the reviewed native execution approach, or superpowers:subagent-driven-development if the user selects it. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a CPU-testable native spool writer and bounded exporter that produces complete v2 Mellum routing traces from synthetic selections.

**Architecture:** Small C helpers own layout and binary framing without engine/GPU dependencies. A Python exporter validates one forward at a time, derives provenance/completeness fields, and publishes JSONL only after complete validation. The Metal hook and live parity are a separate S2b milestone.

**Tech Stack:** C99, existing Make, Python 3.14+ standard library, unittest function tests; macOS/Linux local filesystems.

**Spec:** [S2a capture design](../specs/2026-10-27-mellum-route-capture-design.md), including its exact binary layout and S1 evidence references.

## Global Constraints

- Use C99 for native helpers, Python 3.14+ and its standard library for export.
- No C++, third-party dependencies, model loading, GPU initialization, inference, Metal builds or storage benchmarks.
- No changes to inference files, `ds4.h`, GPU headers, kernels or default executable link inputs in S2a.
- One concern per commit, failing tests committed red separately, at most about 150 product lines per implementation commit.
- No push or merge. Keep this worktree/branch based on Mellum `59932c9`; no integration of the other agent's frontend work.
- Fixed geometry 28/64/8; cap_tokens in 1..1024; preserve ordered router slots.
- Explicit little-endian integers; no raw native struct serialization.
- Metadata input <=1 MiB; one bounded forward in memory; never buffer an entire capture.
- No overwrite or partial publication; native/footer failure is terminal.
- S2 stays open until S2b GPU gates pass. Synthetic round trips are not locality, parity or throughput evidence.

## Review Focus

- Padded layer buffers: serialize active rows only and preserve slot order even when padding holds invalid IDs (Tasks 1–2).
- Producer/output failure: a short sink write cannot later acquire a successful footer, and a preexisting output/symlink is never replaced (Tasks 3 and 6).
- Hostile lengths/truncation: reject oversized rows before reading/allocating payload, and reject a missing whole-frame/footer tail (Task 4).
- Provenance conflict: supplied computed metadata must match derived fields; conflicting extensions cannot silently overwrite evidence (Task 5).
- Unicode and multi-phase capture: preserve labels/extensions and distinguish one-row prefill from decode; no Unicode record splitting (Tasks 4–6).

---

## Files and task interfaces

Create `ds4_mellum_routes.h` and `.c`: fixed geometry/layout plus sink-driven
writer. Create `tools/mellum_routes_to_trace.py`: reader, metadata completion,
publication/CLI. Create `tests/test_mellum_routes.c` and
`tests/test_mellum_routes_to_trace.py`. Modify Makefile only for CPU test rules,
help and `test-mellum` dependency; add generated test binary to `.gitignore`
using its existing explicit-test-output style. Add usage documentation in
`docs/MELLUM_ROUTE_CAPTURE.md` when the behavior ships.

C tests use standalone functions and a small CHECK macro, matching existing
model-free tests. Python tests use the existing stdlib `load_tests` function
pattern, not pytest or classes. Use independently encoded byte fixtures and
literal expected fields. Each task runs its full owning test target at RED
and GREEN and commits the red tests before product code. If implementation
would exceed the product-line limit, split that task's behavioral red/fix
pairs before editing; record the split in the execution ledger.

### Task 1: Checked recorder-buffer layout

**Files:** Create native header/source and `tests/test_mellum_routes.c`; modify Makefile and `.gitignore` after RED.

**Interfaces:** Produce `mellum_routes_layout_info` with u64 `buffer_bytes`,
`layer_offset`, `copy_bytes`; and the exact `mellum_routes_layout(cap, rows,
layer, out)` signature from the spec.

- [x] Write C tests: `(cap=1024, rows=1024, layer=27)` returns bytes 917504, offset 884736, copy 32768, and offset+copy equals total. At rows=1 copy is 32 with the same total/offset. Cap=2, rows=1, layer=1 gives bytes 1792, offset 64, copy 32. Reject cap 0/1025/UINT32_MAX, rows 0/cap+1, layer 28/UINT32_MAX and null output.
- [x] Compile/run via `cc -std=c99 -Wall -Wextra -I. tests/test_mellum_routes.c -o /tmp/mellum-routes-layout-test`; expect the absent native interface to prevent compilation. Commit only tests: `test: define bounded Mellum router capture layout`.
- [x] Implement checked layout, constants and include guard; no allocations or GPU calls. Add `tests/test_mellum_routes` build rule linking only test and `ds4_mellum_routes.c` with `$(CC) $(QUALITY_CFLAGS) -I.`, plus `test-mellum-routes-c` to run it. Ignore the binary; do not alter default executable link inputs.
- [x] Run `make test-mellum-routes-c`; require all layout cases pass. Run `git diff --check`; commit: `feat: calculate bounded Mellum route capture offsets`.

### Task 2: Header and validated forward framing

**Files:** Modify native header/source and C tests.

**Interfaces:** Produce sink typedef, writer state, `mellum_routes_begin` and
`mellum_routes_forward` with the spec's signatures. Writer owns no heap/file;
only an active initialized writer may emit. Counters advance only after full
frame write success. Finish is added in Task 3.

- [x] Add a bounded in-memory sink in tests and a 2-row-cap padded ID fixture. Set active rows to ordered IDs 7,6,5,4,3,2,1,0 at every layer and padding to UINT32_MAX. Begin cap=2 and emit forward 0 at position 42, rows=1, phase=2, segment=7, execution=11, token 1234. Assert literal 24-byte header; 36-byte prefix fields; token bytes D2 04 00 00; all 28 ordered rows; total bytes before footer 960. No padding emitted.
- [x] Add failure cases: null sink/writer/buffers, undersized input lengths, wrong phase, zero/oversized rows, decode rows=2, forward gap, position gap/overlap, u64 next-position overflow, duplicate/out-of-range IDs. Assert invalid forward adds no bytes and no later frame succeeds. Accept consecutive one-row prefill and decode frames with changed labels, nonzero start and larger input buffer lengths.
- [x] Run `make test-mellum-routes-c`; expect missing writer declarations/symbols. Commit tests: `test: specify ordered native route spool frames`.
- [x] Implement begin/header and validate-then-emit forward. Encode integer scalars explicitly little-endian; serialize active rows from cap-strided input. Check all frame fields and IDs before its prefix. Enforce sticky failure and arithmetic guards. No inference integration.
- [x] Run `make test-mellum-routes-c` and `git diff --check`; commit: `feat: write validated native Mellum routing frames`.

### Task 3: Complete-or-failed native spool lifecycle

**Files:** Modify native header/source and C tests.

**Interfaces:** Produce `mellum_routes_finish(w)`; finished/failed state rejects
all further frame/finish calls without callbacks. Extend C test executable with
`--emit-fixture PATH` using the real writer and fwrite sink; this is a synthetic
fixture mode, not an inference harness. It emits the Task 2 valid one-forward
fixture, calls finish and checks flush/close errors.

- [x] Test literal footer `END1` with forward_count=1, position_count=1, making total spool size 980. Two consecutive frames yield counts 2/2. Empty finish fails. Frame after finish and double finish leave bytes unchanged.
- [x] Use a capacity-limited sink that returns a short count while writing header, frame or footer. Assert terminal failure, no later writes/footer, and no successful frame count for a short frame. Do not assert only mock call counts; inspect the real emitted bytes and final writer result.
- [x] Run `make test-mellum-routes-c`; expect missing finish behavior. Commit tests: `test: require complete-or-failed native route spools`.
- [x] Implement footer/state guards and fixture emission. Begin requires a fresh caller-owned writer; do not add implicit reset/recovery. Add the fixture mode's file handling to test utilities only.
- [x] Run `make test-mellum-routes-c`; emit a temporary fixture and assert its size is 980 bytes. Run `git diff --check`; commit: `feat: finalize route spools and make write failures terminal`.

### Task 4: Bounded Python spool reader

**Files:** Create exporter module and Python tests; modify Makefile.

**Interfaces:** Produce `RouteCaptureError(ValueError)`; frozen `RouteForward`
with `forward`, `first_position`, `rows`, `phase`, `segment_id`, `execution_id`
integers and `tokens`, `selected` packed bytes. Produce `SpoolReader(source:
BinaryIO)`, `.cap_tokens`, `.read_forward() -> RouteForward | None` and
`.summary() -> dict`. Summary is available only after valid footer/EOF and
contains cap_tokens, forwards, positions, first_position, last_position,
spool_sha256 and token_stream_sha256. No catalogs/counters keyed by arbitrary
input IDs accumulate in this reader.

- [x] Add stdlib Python tests with independently `struct.pack`-encoded valid one/two-frame spools. Assert explicit positions/phases/slot order, token digest over LE token bytes once per position, spool digest over exact full bytes, and summary rejection before footer. Test prefill rows=1 is retained as prefill.
- [x] Test every truncated prefix of the 980-byte fixture, absent footer after a complete frame, count mismatch, trailing bytes, bad magic/tag/geometry, cap 1025, invalid phase, rows=0/1025/UINT32_MAX, decode rows=2, forward/position gaps, overflowing next-position, duplicate/out-of-range experts. Errors must include byte offset and reason. Use a test stream that refuses reads larger than the maximum valid payload; corrupt lengths must fail before such a read.
- [x] Run `python3 -I tests/test_mellum_routes_to_trace.py`; expect missing reader module/interface. Commit tests: `test: define bounded native route spool decoding`.
- [x] Implement checked incremental reads/digests, payload validation, ordering, footer/count/EOF validation. Read only the validated bounded payload; do not load the file or all frames at once. Add `test-mellum-routes` depending on the C binary/C test target and running the Python test script.
- [x] Run `make test-mellum-routes` and `git diff --check`; commit: `feat: decode and validate native Mellum route spools`.

### Task 5: Complete v2 evidence metadata

**Files:** Modify exporter and Python tests.

**Interfaces:** Produce `read_metadata(path: Path) -> tuple[dict, dict[int,str],
dict[int,str]]` returning template header, segments and execution profiles;
`complete_metadata(header: dict, summary: dict, profiles: dict[int,str],
profile_counts: dict[int,int]) -> dict`. Return a new completed header,
without mutating caller input. Import S1 validation by absolute sibling file
location; no sys.path edits or dependency installation.

- [x] Test metadata >1 MiB, invalid UTF-8, duplicate keys, exponent overflow, missing/non-object catalogs, noncanonical/out-of-u32 keys and blank labels/profile descriptions. A file exactly at the cap with valid JSON padding is accepted. Accept Unicode and nested finite header extensions.
- [x] Given summary cap=2, one forward/position at 42, known token/spool digests, and used profile 11 with count 1, assert derived geometry 28/64/8/6580224, complete=true, exact extent, token digest and `effective_kernel='native-profile-catalog'`. Assert route_capture includes cap=2, spool digest, only profile '11' and count 1. Preserve all other fields/extensions and input immutability.
- [x] Table-drive each supplied computed field matching (accept) or conflicting (reject), including nested route_capture, geometry and token digest. Reject missing S1 fields, non-v2 templates, identity/numerical-mode mismatches. Check both synthetic and declared-measured header validation; all fixtures remain fabricated.
- [x] Run `make test-mellum-routes`; expect missing metadata behavior. Commit tests: `test: bind native route spools to complete v2 evidence`.
- [x] Implement bounded strict metadata loading, catalogs and computed-field conflict handling; complete and validate with S1 `parse_header`. Include only used profiles and counts; the export loop must reject unknown IDs before adding counters. Do not infer actual hardware paths from requested settings.
- [x] Run `make test-mellum-routes` and `git diff --check`; commit: `feat: derive complete routing evidence from native spools`.

### Task 6: Stream and atomically publish the trace

**Files:** Modify exporter, Python tests and Makefile help.

**Interfaces:** Produce `export_trace(spool: Path, metadata: Path, output:
Path) -> None` and `main()` with the CLI in the spec. Retain exact source
spool; no input files are modified. Errors return nonzero through concise CLI
stderr, no traceback or stdout artifact.

- [x] Test exporter success with multiple prefills/decode/segments, nonzero start, Unicode U+2028/U+2029/U+0085 and extensions. Feed output to S1 `load_trace`/`run_sweep`: routes, phase and provenance preserved; no invented cache seeding. Use a tiny-row-cap but full Mellum-geometry fixture.
- [x] Test context overflow, chunk mismatch, unknown segment/execution IDs, late corruption/footer failure, missing files, output-open/publication failure and existing output (regular file or symlink): no new output, no changed existing bytes, no leftover exporter temporary files. Use real filesystem failures where practical; a narrow failure injection at the publication boundary is acceptable to exercise the race without depending on timing.
- [x] Test existing-name race: create the output immediately before the actual os.link call; require EEXIST, preserve raced-in bytes and clean temporary files. Test enough consecutive frames through a stream that denies seek/unbounded reads; assert no need to reread the spool. Do not use timing/RSS thresholds as unit tests.
- [x] Add the C-to-Python round trip: run C fixture mode from Task 3, compare its 980 bytes to the independent Python fixture, export with matching synthetic metadata, replay output and assert exact ordered selected rows and cold demand 224 experts = 1,473,970,176 bytes. Invoke CLI via `python3 -I` from another working directory to prove sibling loading.
- [x] Run `make test-mellum-routes`; expect missing export/CLI behavior. Commit tests: `test: require atomic native routing export and replay round trips`.
- [x] Implement one-pass reader to private body JSONL, checking every frame's catalog/context/chunk before counter growth; finalize metadata only after valid footer/EOF. Prefix the completed header into a second sibling temporary file with bounded copy. Close/flush, publish with no-overwrite os.link and clean both private files in finally blocks. Keep valid stdout empty.
- [x] Run `make test-mellum-routes` and `git diff --check`; commit: `feat: publish complete native routing traces without overwriting evidence`.

### Task 7: Document and verify CPU preparation

**Files:** Create `docs/MELLUM_ROUTE_CAPTURE.md`; modify `ROADMAP.md`, this plan, design status, Makefile `test-mellum` dependency.

**Interfaces:** User documentation for the shipped helpers/exporter; S2b remains unimplemented.

- [x] Add `test-mellum-routes` to model-free `test-mellum`. Document a fully synthetic fixture/metadata/export/replay recipe, binary version, metadata ownership, file publication guarantees, diagnostic memory formula and S2b's remaining source/parity gates. Do not imply a native recorder is attached or an actual kernel path has been measured.
- [x] Run `make test-mellum-routes`, `make test-mellum-ssd-replay`, then `env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum`; require all pass and record actual test counts/commands. No Metal/default make target or model environment variables.
- [x] Check changed local Markdown links, `git diff --check`, source diff and per-commit product-line counts. Confirm no inference/default link changes, GPU calls or new dependencies. Mark only S2a complete after gates; S2b/S2 remain open.
- [x] Commit docs/gate: `docs: record CPU route capture preparation and remaining GPU gates`. Execute one independent whole-branch review, with one red/green fix pass for Important/Critical findings per executing-plans; record findings and disposition before handoff. Keep the branch/worktree.

## Plan self-review

Spec coverage: layout/buffer bounds -> Task 1; header/frames -> Task 2;
footer/failure -> Task 3; binary validation/identities -> Task 4;
metadata/capture profiles -> Task 5; streaming/no-overwrite/round trip -> Task 6;
documentation/CPU acceptance -> Task 7. Each Review Focus item has behavioral
tests in its owning task. Reader summary excludes unconstrained catalogs;
exporter counters are bounded by the prevalidated metadata catalog. All
cross-task interfaces and byte counts above match the design. No S2b runtime
attachment is hidden in a CPU preparation task.

## Execution record

Implemented natively following user approval. Python 3.14.7; C99 compatibility
also compiled/ran explicitly with `cc -std=c99 -Wall -Wextra`. No deviations or
rulings in Tasks 1–7. The design builder check found no material unknowns.

| Task | Separate red commit | Implementation commit |
| --- | --- | --- |
| 1: layout | `39cdb23` | `96582ba` |
| 2: frames | `4985a1d` | `04e7814` |
| 3: footer/failure | `bf7f957` | `fa429fe` |
| 4: reader | `2799a20` | `694a658` |
| 5: metadata | `b6d7de7` | `fea52dc` |
| 6: publication/CLI | `59b407d` | `5911ebc` |

Every red run failed on the missing interface/behavior. Green runs execute all
owning tests, with Python counts 3 -> 7 -> 12. Native tests cover layout,
framing and finish/short writes (every byte boundary of the golden spool).
Implementation commits add 36, 77, 10, 122, 78 and 67 product lines respectively;
no inference/default executable link changes or added dependencies.

Gate: `make test-mellum-routes` passes native checks and 12 Python tests;
`make test-mellum-ssd-replay` passes 33; and
`env -u DS4_TEST_MELLUM_MODEL make -j4 test-mellum` exits zero, including those
checks and 14 logits-comparison tests. The documented synthetic example exports
and replays successfully. Local Markdown links and whitespace are checked.
No model, GPU initialization, inference, Metal build or storage benchmark ran.
Independent review of `59932c9..46c0a24` found no Critical/Important findings.
It independently passed C99/native checks, 12 capture/export tests, 33 replay
tests and an additional native-to-Python fixture with distinct layers/rows/slots,
invalid padding and boundary token IDs. No S2a fix pass was needed.

Deferred minor: `tools/mellum_ssd_replay.py:107` does not catch RecursionError
for deeply nested JSON. The CLI emits a traceback while still returning nonzero
with empty stdout. This is a diagnostics issue in the existing replay tool;
it does not publish invalid evidence. It is deferred rather than bundled into
the completed capture milestone.

Rulings on declined review areas: GPU synchronization/parity, capture overhead,
physical 8 GB behavior and SSD performance remain S2b/later gates under the
user's no-GPU constraint. Cost if assumptions fail: hardware correctness or
feasibility may fail; this CPU milestone makes no measured claim. Publication
remains scoped to local macOS/Linux filesystems, excluding power-loss durability
and network-filesystem semantics. Cost outside that contract: an artifact may
not survive a crash or unsupported filesystem behavior.

S2a is complete. Branch/worktree retained, nothing pushed or merged. S2b remains
open; no GPU experiment or native inference attachment occurred.
