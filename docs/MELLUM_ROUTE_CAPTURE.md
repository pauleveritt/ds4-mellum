# Native Mellum route spool preparation

S2a supplies CPU-only capture helpers and an exporter. A separate native
diagnostic adapter now connects them to inference, but its live gates are
still unrun. The synthetic example below fabricates selections; no GPU, locality,
model quality or performance result follows from them.

Use Python 3.14+ and a C99-capable compiler. No third-party dependencies.
Run the new test target with `make test-mellum-routes`; it runs native layout,
framing and short-write checks plus the Python reader/export tests. These also
run under the full model-free `make test-mellum` target.

## Synthetic example

From the repository root, build the test helper and create a fresh directory:

```sh
make test-mellum-routes
capture_dir=$(mktemp -d)
./tests/test_mellum_routes --emit-fixture "$capture_dir/routes.bin"
cat > "$capture_dir/metadata.json" <<'JSON'
{
  "header": {
    "schema_version": 2, "capture_kind": "synthetic",
    "model_id": "fixture-model", "engine_revision": "fixture-engine",
    "numerical_mode": "release", "workload_id": "synthetic-one-token",
    "capture_manifest": {
      "manifest_version": 1, "capture_id": "synthetic-capture",
      "artifacts": {"model_id": "fixture-model", "tokenizer_id": "fixture-tokenizer", "template_id": "fixture-template"},
      "build": {
        "engine_revision": "fixture-engine", "source_state_id": "fixture-source",
        "binary_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "recorder_revision": "fixture-recorder"
      },
      "execution": {
        "context_tokens": 8192, "prefill_chunk_tokens": 1024,
        "numerical_mode": "release", "requested_kernel": "synthetic",
        "device": "synthetic", "os": "synthetic", "capabilities": {}
      },
      "generation": {"mode": "teacher-forced", "settings": {"seed": null}},
      "workload": {
        "protocol_id": "synthetic-example", "conversation_id": "synthetic",
        "repository_id": "synthetic", "split": "tuning", "continuation_kind": "synthetic"
      }
    }
  },
  "segments": {"7": "synthetic-decode"},
  "execution_profiles": {"11": "synthetic-exact-projections-experts-head"}
}
JSON
python3 -I tools/mellum_routes_to_trace.py "$capture_dir/routes.bin" \
  --metadata "$capture_dir/metadata.json" --output "$capture_dir/trace.jsonl"
python3 -I tools/mellum_ssd_replay.py "$capture_dir/trace.jsonl"
```

The fixture has one decode position (42), token ID 1234, 28 layers and ordered
expert slots 7 through 0 at each layer. A cap of 2 is recorded, but inactive
padding is omitted from its 980-byte spool. A replay starts cold and loads
224 experts, or 1,473,970,176 bytes. This is constructed demand, not measured
Mellum locality. The dummy identities and zero build hash are declarations.

## Writer and spool contract

`ds4_mellum_routes.h` exposes layout, begin, forward and finish helpers. The
caller owns the writer, sink, tokens and padded selected-ID storage. Initialize
a fresh writer through begin; pass readable arrays of the declared lengths.
The writer validates every active row before its frame, preserves slot order,
ignores padding, allocates no heap memory and never initializes a GPU. A short
sink write or invalid forward is terminal; no later frame/footer can succeed.
Future file sinks must check flush and close before invoking the exporter.

The [design](superpowers/specs/2026-10-27-mellum-route-capture-design.md#binary-spool-version-1)
specifies byte layout and native signatures. Version 1 is explicit little-endian:
24-byte `MLRT0001` header; each `FWD1` prefix is 36 bytes followed by tokens and
layer/row/slot IDs; a 20-byte `END1` footer supplies totals and must end the file.
Geometry is fixed at 28/64/8, cap is 1..1024, forwards/positions are consecutive,
and one-row prefill remains prefill. This format cannot represent reset/rewind.

At cap=1024 a future GPU capture buffer and its host readback would each need
917,504 logical bytes, totaling 1.75 MiB, plus tokens and metadata. No such GPU
allocation is made by these helpers, and this is not a measured process peak.

## Metadata and publication

The exporter reads metadata up to 1 MiB and validates the spool one bounded
forward at a time. Catalog IDs are canonical decimal u32 strings, with nonempty
segment labels and execution-path descriptions. Every frame must resolve both
IDs. Context and chunk declarations must cover the actual captured positions.

The exporter derives geometry, capture completeness/extent and token digest,
using each token's little-endian u32 bytes once per position. It derives the
exact spool digest and stores used profile descriptions/counts under
`execution.route_capture`, with `effective_kernel="native-profile-catalog"`.
Retain the spool to associate each forward with its execution profile later.
Supplied computed values must match; conflicts are errors. Other manifest
fields/extensions are preserved. All other required [S1 fields](MELLUM_SSD_REPLAY.md#version-2-capture-manifest)
come from the producer; the exporter does not verify their truth or infer
actual hardware paths. Real measured producers must observe encoding branches.

Private body/final files grow on disk; RAM use is bounded by metadata and a
forward plus serialization/copy buffers. Publication uses a hard link from a
closed, complete sibling temporary file. An existing destination, including a
symlink or a raced-in name, is never replaced. Failures leave no new published
artifact and clean exporter temporary files. Target local macOS/Linux filesystems;
power-loss durability and network filesystems are outside this guarantee.
The CLI writes no stdout artifact and reports errors concisely on stderr.

## CPU session-controller core

The approved [session-recorder design](superpowers/specs/2026-10-27-mellum-session-capture-design.md)
has a GPU-free core in `ds4_mellum_capture_state.h` and `ds4_mellum_profile.h`.
Run `make test-mellum-profile test-mellum-capture-state`; both also run under
`make test-mellum`. The latter checks a synthetic controller capture through
the existing reader, exporter and replay, and rejects an aborted capture.

Construct the controller on fresh caller-owned storage with an immutable
borrowed kernel catalog. Keep the catalog/definitions and sink context alive
through finish or abort. Calls are serialized, same-thread and non-reentrant;
sinks must not call back into the controller. There is no reset/reuse after
failure or finish. The later adapter owns the actual session and supplies its
committed prefix; this core does not verify that prefix against session KV.

Begin sync with the full unchanged/extended prompt, or eval at the next
position. Each forward begins an initially empty profile scope, borrowed until
forward end/abort. Record actual selected kernel descriptors in that scope and
acknowledge successful layer copies in order 0..27. Only after successful GPU
completion/readback may the adapter end the forward with its actual tokens and
cap-strided expert IDs. CPU fixture events cannot prove those GPU events.
Operation end requires every promised position to have a complete frame.
No-op sync preserves logits availability; a final forward without a head clears
it. Finish emits the used profile catalog before the native completion footer.
Cancellation, incomplete profiles/layers and sink failures are terminal and
prevent a successful artifact. Abort never writes a footer. FINISHED rejects
further mutations while preserving its bytes and logits availability.

Profiles contain at most 64 canonical descriptors with six integer constants
each, with at most 128 distinct profiles per controller. The descriptions use
`profile-v1;head=encoded|not-encoded;ROLE:kernel(label=value,...)` and distinguish
every observed specialization. Definitions are supplied by the caller; the
synthetic test catalog does not establish production kernel provenance.
The registry/controller, 16 KiB formatting scratch and proposed pair of ID
buffers at cap=1024 fit within the 2,200,000 logical-byte design bound, excluding
session, token and sink storage. This is not a measured process peak.

## Owned diagnostic adapter (compiled, live gates pending)

`ds4_mellum_capture.h` declares a separate opaque handle. Create requires a
ready resident Mellum Q8 Metal engine without distributed/TP/multi-tier or
speculative support. It owns a fresh private session, controller and bounded
host/GPU route buffers; the engine and sinks remain caller-owned. CPU and
non-Apple builds return an unsupported error without creating a session or
writing spool bytes. Hidden-state trace controls are incompatible with capture.

Use create, sync of a full unchanged/extended token prefix, one-token eval,
copy_logits, finish and free. Sync reserves the same generation room as the
ordinary session API. An identical prompt emits no new frames; cancellation
remains terminal even before the first frame and preserves status 2. Public
operations check token IDs, context and the actual private committed prefix.
A failed handle cannot continue. Finish emits observed profiles before the
footer; free never finalizes or closes caller resources. Calls and callbacks
must obey the serialized, non-reentrant lifetime contract in the header.

Run `make test-mellum-capture-api test-mellum-observer test-mellum-capture-build`
for model-free boundary, observer and build checks. These are included in
`make test-mellum`. The API tests supply private CPU fixture state to reach
validation and real ordinary cancellation/no-op behavior without model work;
no GPU temporal claims follow. The following compiles/links the native paths
without executing them:

```sh
make ds4.o ds4_metal.o ds4_cpu.o tests/mellum_routes_gpu
```

The immutable `ds4_mellum_kernels.inc` catalog identifies actual math functions
and named specialization values. One scoped observer borrows the current
profile; nested/foreign scope operations preserve the existing owner. Unknown
functions, malformed constants and overflow poison the profile. Notifications
do not return early through GPU cleanup; the forward is rejected after ordinary
completion. No table walk, heap allocation or GPU synchronization occurs in a
disabled observer. Default-path overhead still requires measurement.

Actual-encoding audit:

| Reachable math | Observation site | Recorded constants |
| --- | --- | --- |
| Q8 embedding | `ds4_gpu_encode_get_rows_q8_0` after dispatch | Actual `kernel_get_rows_q8_0_f32` identity |
| Weighted norm/residual add | Weighted rows norm and add dispatches | Fixed actual functions |
| KV f32-to-f16 | `ds4_gpu_encode_cpy_f32_f16_1d` | Actual contiguous or generic function |
| Q8 decode/exact rows | Actual MV dispatch | NSG used by the pipeline |
| Q8 multi-row MV extension | Actual extension dispatch | NSG and NXPSG |
| Q8 NAX aligned prefix and generic tail/fallback | Each actual legacy matmul dispatch | Boundary constants; function identifies tile variant |
| Optional Q8 unpack | Actual unpack dispatch and following f16 matmul | Actual unpack/f16 identity and boundary constants |
| Router/top-k, list building, staging, reduce | `qwen4_dispatch_resident` | Actual selected function |
| Exact MoE mid/down | Same successful dispatch site | Specialization flag plus effective type/shared/dim/rows, including shader defaults |
| Tiled MoE main/tails/NAX/SIMD-group/fallback | Same successful dispatch site | Weight-type and tail-base pipeline constants |
| Full attention decode/rows/merge | Same successful dispatch site | Actual selected variant |
| Mellum QK norm/RoPE, ring commit and SWA | `mellum_dispatch` after dispatch | Actual selected function |
| Output projection | Actual Q8 site under graph HEAD tag | Projection constants, role HEAD |

Observer blits are excluded. Shape, position, runtime source identity and
capabilities remain separate metadata. Profiles are canonical specialization
sets, not ordered dispatch timelines. Future newly reachable unlisted math
must extend this audit/catalog; an observed unknown function fails capture.

The live harness retains its ordinary/buffered/synchronous-reference matrix
and adds owned-handle operation-logit checks, observed-profile checks, ordered
route comparison (ignoring independently assigned profile IDs), a direct
no-head/head sequence and cancellation/header/frame/copy/read failure checks.
It compiles, but none of these new GPU checks has run. Real allocation-failure
injection, cancellation between chunks, actual profile/fallback coverage and
measured allocation/overhead remain GPU gates. Ordinary Mellum session/graph
host allocations now return errors rather than terminating the process.

## Source-checked publisher (CPU-tested, live collection pending)

`tools/mellum_capture_publish.py` is the orchestration entry point for a future
authorized GPU window. It force-builds the diagnostic harness, runs its ordinary,
buffered, synchronous-reference and owned-session comparisons, then exports the
owned spool only if the entire harness succeeds. This command runs GPU inference;
it has not been executed during CPU-only development. `--help` and
`make test-mellum-capture-publish test-mellum-capture-receipt` are model-free.

Prepare a metadata file containing only a caller-declared header. For example:

```json
{
  "header": {
    "model_id": "your-pinned-mellum-q8-model",
    "workload_id": "your-frozen-workload",
    "capture_manifest": {
      "capture_id": "your-unique-capture-id",
      "artifacts": {
        "model_id": "your-pinned-mellum-q8-model",
        "tokenizer_id": "your-pinned-tokenizer",
        "template_id": "your-pinned-template"
      },
      "execution": {"device": "your-physical-test-device-label"},
      "workload": {
        "protocol_id": "your-workload-protocol",
        "conversation_id": "your-conversation",
        "repository_id": "your-repository",
        "split": "tuning",
        "continuation_kind": "teacher-forced-continuation"
      }
    }
  }
}
```

During an authorized device window, from any directory:

```sh
python3 -I /absolute/path/to/ds4-mellum/tools/mellum_capture_publish.py \
  --model /absolute/path/to/mellum-q8.gguf \
  --tokens /absolute/path/to/frozen.tokens --first-prefill 32 \
  --context 8192 --prefill-chunk 1024 \
  --metadata /absolute/path/to/cohort.json --output /absolute/path/to/new-trace.jsonl
```

Add `--exact-prefill` for that numerical cohort. Tokens use the existing native
decimal-ID/comment grammar, capped at a 16 MiB input file. The protocol requires
at least two tokens after the initial prompt: teacher-forced decode, then a final
one-token prompt append. It reserves 64 context positions as the proof harness
does. It does not generate a continuation. Actual token bytes, positions, phase,
segments and chunk boundaries are checked against the input and schedule, beyond
reference-route agreement. Publication never overwrites an existing output.

The publisher rejects inherited `DS4_*` controls (except the unused
`DS4_TEST_MELLUM_MODEL`), `MELLUM_TRACE_*` settings, `GIT_*` settings and make override controls;
it sets only its explicit context/chunk/exact controls. This includes empty
source overrides and routing replay controls. The normal model lock still applies.

The build manifest retains source file hashes, revision, aggregate source-state
identity, binary hash and the forced-build command log. Sources include Git-listed
C/header/Objective-C/include/Metal/Python/Makefile inputs and default Metal files,
including `GNUmakefile` and lowercase `makefile`. The build explicitly selects
the repository's `Makefile`; Git identity commands ignore Git environment overrides.
Source inputs cannot be symlinks. The publisher checks source state before/after
the build and after collection, requires a newly built executable, and checks
binary/token/model-file identity after collection. Model bytes are not hashed;
model/tokenizer/template/workload IDs and the device label remain caller-declared.
Model file stat identity is recorded. Restored changes between checks and hostile
compiler behavior are outside this consistency check; it is not attestation or
a reproducible-build guarantee.

The native finish callback supplies the bounded profile catalog. Its descriptions
must match the complete native grammar, canonical descriptor order, kernel/constant
definitions, required roles and head state in `ds4_mellum_kernels.inc`. The checked-close
runtime receipt supplies actual context/cap and backend capability flags, recorded
separately from requested controls. Caller conflicts with derived fields are errors;
caller-supplied profile catalogs cannot be used. JSON serialization escapes catalog
strings. Private temporary files are removed on success or failure; the existing
exporter validates footer/catalog/extents and publishes atomically without overwrite.
No live qualified artifact, workload locality or SSD performance follows from the
CPU subprocess-boundary tests. Generated workloads and the remaining live failure,
path, memory and overhead matrix are still separate gates.

## Remaining GPU work

The adapter now supplies per-session storage, buffered copies, completion
readback, observed-kernel notifications and sync/eval labels in code. It needs
byte-equal traced/untraced logits, no added per-layer host wait, ordered GPU
snapshots and cancellation/error cleanup checks. The current blit primitive
closes the compute encoder; overhead must be measured later. The existing
hidden-state trace's per-layer waits do not provide that evidence.

See the [roadmap](../ROADMAP.md) and [evaluation protocol](MELLUM_SSD_EVALUATION.md).
S2 remains open. The CPU-core work adds no GPU experiments or SSD inference.

## GPU primitive proof

On macOS, `make test-mellum-routes-gpu` builds a dedicated diagnostic executable.
Set `DS4_TEST_MELLUM_MODEL` to the Mellum Q8 GGUF to run short, tiled and exact
cases; without it the execution step skips. One model process at a time.

The harness compares logits after every sync/eval and compares buffered route
spools with a per-layer synchronous reference. It includes a final one-token
prompt append. Its graph hooks compile only into the test executable; shipping
builds omit these test-only hooks; the diagnostic API and native writer are
linked into the ordinary engine, with no recorder allocation until create. The smoke
wrapper clears inherited Mellum trace/path and prefill-chunk controls.

The [GPU proof](superpowers/research/2026-10-27-mellum-route-gpu-proof.md)
records the earlier primitive matrix and exploratory replay. The new owned-session
live lifecycle and actual encoded-kernel observations must be validated before
S2b is complete. Do not turn experimental raw spools into qualified v2 evidence
using guessed profile descriptions.
