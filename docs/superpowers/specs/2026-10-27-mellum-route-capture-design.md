# Mellum router capture preparation

Date: 2026-10-27. Status: S2a complete and reviewed; S2b GPU gates remain open.
S2 of the [roadmap](../../../ROADMAP.md), following the completed
[S1 evidence contract](2026-10-27-mellum-ssd-evidence-design.md).

## Scope and decisions

Prepare a bounded native capture handoff that can be tested on the CPU. Split
S2 into **S2a: native spool and exporter** and **S2b: Metal attachment and live
validation**. This design and its implementation plan cover S2a only. A writer
and exporter that round-trip synthetic selections are a useful deliverable;
they do not establish GPU safety, parity, locality or throughput.

The user authorized proceeding without GPU access. Stay on this isolated
`codex/mellum-ssd-replay` worktree, currently `c0eb8a5`, based on Mellum
`59932c9` and upstream `ee7e840`. The separate `mellum` checkout has since
advanced to `f635616`; its extra changes concern the agent/frontends and their
tests, not `ds4.c` or `ds4_metal.m`. Do not rebase or integrate that work here.
The older statement that steps 34–40 are parked describes this branch's
planning baseline; it is not a claim about the other agent's current state.

Use C99 for the native helpers, Python 3.14+ and its standard library for
export. No C++, third-party dependencies, model loading, GPU initialization,
inference, Metal builds or storage benchmarks. No changes to inference files,
`ds4.h`, GPU headers, kernels or default executable link inputs in S2a.
One concern per commit, failing tests committed red separately, at most about
150 product lines per implementation commit. No push or merge.

My implementation choices: a small native binary spool, a separate Python
exporter, one bounded forward at a time, and publish a v2 JSONL artifact only
after complete validation. This avoids JSON/provenance policy in GPU scheduling
and avoids buffering a whole session in RAM. These engineering choices were reviewed before implementation.

## Source findings that constrain S2b

These are source observations on this branch, not experiments:

- `ds4.c:mellum_graph_moe` writes ordered top-8 IDs into
  `mellum_scratch[MELLUM_S_SELECTED]` on both exact and tiled expert paths.
  That scratch is reused by the next layer. Copy immediately after a successful
  `mellum_graph_layer` and before the next layer starts; no router rerun.
- `ds4.c:mellum_graph_forward_tokens` normally ends/waits for its command
  buffer before return, including the final cleanup when no logits are requested.
  Read captured selections only after that existing successful completion.
- `ds4_metal.m:ds4_gpu_tensor_copy` encodes a blit in the active command buffer.
  It closes the current compute encoder but does not commit/wait. It is a
  candidate primitive, not evidence of negligible overhead or proven ordering.
- Existing `DS4_MELLUM_TRACE` waits and reads per layer. Keep it off during the
  recorder's parity/timing gate; do not extend it into the new recorder.
- Phase comes from session operation: a one-token prefill tail is still prefill.
  Segment/phase cannot be inferred from chunk size. Capture callers must label
  initial prompt, generated continuation and incremental tool appends explicitly.
- Effective E1/Metal paths cannot be inferred solely from requested environment
  variables. The wrappers choose hardware paths/fallbacks. S2b must report what
  actually encoded, including projections, expert paths/tails and output head.
- `ds4_session_sync_internal` can reset/replay after prefix mismatch; rewind can
  rebuild. The append-only recorder must abort that capture instead of silently
  stitching duplicate positions. Held-session/compaction capture is later work.

S2b will own a per-session GPU ID buffer and host readback, each
`28 * cap_tokens * 8 * 4` bytes (917,504 bytes at cap=1024). Layer `il` writes
at `il * cap_tokens * 8 * 4`; copy only `T * 8 * 4` active bytes per layer.
The host writer consumes the padded layout but serializes only active rows.
Two ID buffers total 1,835,008 logical bytes (1.75 MiB), plus token data and
metadata; this is additional diagnostic memory, not an 8 GB peak measurement.

Before S2b coding, settle narrow diagnostic session attachment/labels,
resource cleanup and actual-path reporting at encoding branches. Its harness
must use session sync/eval directly; it must not depend on agent steps 34–40.
Traced/untraced logits must match byte-for-byte for the same tokens and actual
numerical path. Release prefill is compared to release prefill, not to exact
prefill. Validate no additional host wait per layer, ordered selection snapshots,
error/cancellation cleanup, and measured capture overhead with the GPU later.

## S2a module boundaries

- `ds4_mellum_routes.h` / `.c`: fixed Mellum geometry, checked layout, native
  spool writer and sticky failure state; no engine/GPU dependency or heap buffer.
- `tools/mellum_routes_to_trace.py`: bounded binary reader, metadata completion,
  strict v2 validation and atomic publication. Load the adjacent replay module
  by absolute file location with `importlib.util`, registering the module before
  execution so dataclasses work under `python3 -I`. Do not change `sys.path`.
- `tests/test_mellum_routes.c`: layout/writer tests and `--emit-fixture PATH`
  mode producing one deterministic synthetic spool using the actual C writer.
- `tests/test_mellum_routes_to_trace.py`: stdlib function tests, independently
  packed fixtures, publication failure checks and a C-to-Python round trip.
- Makefile: dedicated CPU test targets and inclusion in `test-mellum`; use
  `$(CC) $(QUALITY_CFLAGS) -I.` and link only the native helper/test, no Metal.

## Binary spool version 1

All integers are unsigned, little-endian, encoded explicitly. No raw C struct
writes, native padding, native-endian casts or seek-back updates. Only Mellum's
28 layers, 64 experts/layer and top-8 are supported. Validate cap in 1..1024
before size arithmetic; reject larger caps instead of allocating from them.

| Part | Fields in byte order |
| --- | --- |
| Header, 24 bytes | ASCII `MLRT0001` (8 bytes); u32 layers=28, experts=64, top_k=8, cap_tokens |
| Forward prefix, 36 bytes | ASCII `FWD1`; u64 forward; u64 first_position; u32 rows; u32 phase; u32 segment_id; u32 execution_id |
| Forward body | `rows` u32 token IDs, then u32 expert IDs in layer/row/router-slot order, exactly `28 * rows * 8` IDs |
| Footer, 20 bytes | ASCII `END1`; u64 forward_count; u64 position_count; then EOF |

Phase 1=prefill, 2=decode. Prefill rows in 1..cap; decode exactly one.
Forward numbers start at 0 and advance by 1. The first position may be
nonzero; subsequent positions equal previous first+rows. Reject arithmetic
that would overflow the u64 next-position or cumulative counters. Segment and
execution IDs are arbitrary u32 keys resolved by export metadata.

Token IDs are u32, matching the S1 digest encoding. Experts are 0..63 and
distinct within each ordered row. Validate each full forward before writing
its first byte; invalid calls make the writer failed. Out-of-range IDs, bad
phase, inadequate input buffers and duplicates fail before emitting the frame.
A writer failure may leave a private partial spool, never a valid footer.

Footer requires at least one successful forward. Reject missing/truncated
footer, mismatched counts, trailing bytes, unknown tags/version/geometry,
truncated frame fields/payload and oversized rows before allocating payload.
All parser errors name byte offset and reason. No recovery/resume is supported.
The footer is consistency evidence, not authentication against a lying producer.

## Native interfaces and lifecycle

Use prefix `mellum_routes_`, no public inference API. Put declarations in the
new header; define an ordinary value layout struct and a writer struct whose
fields the implementation owns.

- `bool mellum_routes_layout(uint32_t cap, uint32_t rows, uint32_t layer,
  mellum_routes_layout_info *out)` returns `buffer_bytes` (all 28 padded layers),
  `layer_offset` and `copy_bytes` as u64. Reject null out, cap outside 1..1024,
  rows outside 1..cap and layer >=28. It does not allocate or access a GPU.
- `typedef size_t (*mellum_routes_sink)(void *ctx, const void *data, size_t bytes)`;
  exact returned byte count means success. Any short return is terminal; no
  retry. A future file sink must additionally check flush/close before export.
- `bool mellum_routes_begin(mellum_routes_writer *w, uint32_t cap,
  mellum_routes_sink sink, void *ctx)` initializes caller-provided state and
  emits the header. Only begin on a fresh/unowned writer; no implicit restart.
- `bool mellum_routes_forward(mellum_routes_writer *w, uint64_t forward,
  uint64_t position, uint32_t rows, uint32_t phase, uint32_t segment_id,
  uint32_t execution_id, const uint32_t *tokens, size_t token_count,
  const uint32_t *selected, size_t selected_count)` validates then emits one
  frame. Selected input has `28 * cap * 8` elements, with cap-strided layers;
  validate only active rows, ignore padding. Reject null/short buffers; allow
  larger buffers. Caller guarantees readable storage of declared length.
- `bool mellum_routes_finish(mellum_routes_writer *w)` emits the footer once.
  No frames/finish after finished or failed; no writes on those calls.

State: zero/uninitialized -> active -> finished; any failed operation after
begin -> failed. Begin must receive nonnull writer/sink and valid cap; failure
with a nonnull writer leaves it failed. No allocations or file ownership;
callers own buffers/sink lifetime. No callbacks once failed/finished. Sink
failure during header/frame/footer is sticky. Never count a partially written
frame as successful. The future GPU hook only calls forward after completion;
C tests cannot prove that temporal precondition.

## Export metadata and publication

CLI: `python3 -I tools/mellum_routes_to_trace.py SPOOL --metadata HEADER.json
--output TRACE.jsonl`. Use explicit files; no stdout artifact or overwrite.
Output must not already exist. Errors: nonzero, concise stderr, no traceback,
no new successful output; private temporary files are cleaned on success/failure.

HEADER.json contains `header` (the S1 v2 header template), `segments` and
`execution_profiles`. Both catalogs are nonempty objects with canonical decimal
u32 string keys. Segments map to nonempty labels. Execution profiles map to
nonempty actual-path descriptions; e.g. projection/expert/head families and
fallbacks. Catalog descriptions are declared evidence; CPU export cannot
verify that those kernels ran. Every frame's IDs must resolve.

Limit metadata input to 1 MiB before JSON decoding. Use S1 strict object parsing
for duplicate keys, finite JSON and UTF-8 handling. The exporter owns these
computed header fields: geometry, `capture_manifest.complete`, `extent`,
`artifacts.token_stream_sha256`, `execution.effective_kernel`, and the
`execution.route_capture` extension. A template may omit computed fields;
if supplied, they must equal the derived values, otherwise reject. All other
required S1 fields must be supplied, validated and preserved with extensions.
Require template schema_version=2; context/chunk must cover every frame.

`execution.effective_kernel` is the fixed string `native-profile-catalog`;
`execution.route_capture` holds `spool_sha256`, `cap_tokens`, `profiles` (only
used profile-ID/description pairs) and `profile_forward_counts`. This preserves
actual-path descriptions and counts without accumulating a per-forward catalog
in RAM. Retain the binary spool to resolve each forward's execution ID later.
Measured producers must populate descriptions from actual encoding branches;
S2a synthetic descriptions are explicitly declared fixtures.

Read/hash the spool once, sequentially; derive token digest directly from its
u32 little-endian token bytes once per position. Use bounded `read_exact` calls
(maximum frame payload at cap=1024 is 921,600 bytes) and one frame in memory.
Write layer JSONL records to a private body file, increment extent/counters and
at EOF validate footer, catalogs, context/chunk and the completed S1 header.
Hash includes header, frames and footer exactly. Build a second temporary file
with header+body using bounded copies. Flush/close before publication.

Create temporary files in output's directory. Publish with `os.link` from the
completed temporary file to output, which atomically refuses an existing name;
then unlink the temporary name. Target macOS/Linux local filesystems. Do not
use check-then-replace or delete a preexisting output, including a symlink.
No promise of power-loss durability or network-filesystem semantics.
Bound working memory by metadata cap plus one forward and serialization buffer;
spool/body/final artifacts grow on disk. Streaming export is not an SSD benchmark.

## Builder check

A fresh independent C/Python builder read this design and reported “nothing
important left” on 2026-10-27. The user approved the companion plan; its CPU
implementation passes the gate. Final independent whole-branch review found
no blockers; its deferred diagnostics minor is recorded in the plan.

## Acceptance and deferred evidence

CPU acceptance: exact layout boundaries, golden binary bytes, sticky failure,
truncation/corruption rejection, capped input allocation, preserved slot order,
computed identities/extents, Unicode labels/extensions, no overwrite/partial
publication, synthetic C-writer -> exporter -> S1 replay round trip, and the
full model-free Mellum target. Tests must derive expected bytes independently.

S2a does not attach the writer to inference, supply the final diagnostic session
API, freeze real workload contents or approve a GPU path. Keep roadmap S2 open
until S2b parity, synchronization, bounded allocation and live harness gates
pass. Physical 8 GB behavior, useful warm throughput and SSD locality remain
unmeasured. Host integration follows correct bounded native streaming.
