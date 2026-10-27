# Native Mellum route spool preparation

S2a supplies CPU-only capture helpers and an exporter. They are not attached
to inference. All examples below fabricate selections; no GPU, locality,
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

## Remaining GPU work

S2b must attach per-session capture storage, copy selections before scratch
reuse, read after existing command completion, report actual kernel/fallback
paths and supply phase/segment labels from sync/eval operations. It still needs
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
builds have no observer allocation or native-writer dependency. The smoke
wrapper clears inherited Mellum trace/path and prefill-chunk controls.

The [GPU proof](superpowers/research/2026-10-27-mellum-route-gpu-proof.md)
records the first matrix and exploratory replay. Owned-session integration,
live lifecycle failure coverage and actual encoded-kernel metadata are still required before
S2b is complete. Do not turn experimental raw spools into qualified v2 evidence
using guessed profile descriptions.
