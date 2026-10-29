# Mellum SSD evidence qualification

Date: 2026-10-27. Status: implemented and verified; review finding fixed with a red/green regression.
Implements S1 of the [roadmap](../../../ROADMAP.md), building on the
[completed replay design](2026-10-27-mellum-ssd-replay-design.md).

## Goal and boundary

Make offline replay evidence interpretable and reproducible before real
routing capture. Keep the existing policies; add precise claim metadata,
a versioned capture manifest, completeness checks and report identities.
Freeze the [evaluation protocol](../../MELLUM_SSD_EVALUATION.md).

Python 3.14+, standard library only; preserve standalone `python3 -I`
execution. Use synthetic fixtures only. No GPU use, model loading, inference,
storage benchmarks or inference-code changes in S1. Preserve Q8 bytes and
trained routing. No implementation of prediction, native recording, tiled
schedules, window analysis or streaming in this milestone.

Work from this isolated branch based on Mellum `59932c9` (upstream `ee7e840`
plus E1). Another agent owns port integration; steps 34–40 stay parked.
Small commits: one concern each, failing behavioral tests committed red
separately, at most about 150 lines of product code per implementation commit.
Do not push or merge as part of S1.

## Correct the lower-bound claim

Keep `relaxed-optimal` and its existing sorted, distinct `(layer, expert)`
access sequence and optional bypass. Its whole-trace demand loads/bytes are
lower bounds for atomic full-group residency with the same equal-size cache,
and exact minima for its specified sequential order. They are not minima
for every possible within-group execution order. No eviction algorithm change.

Add `interpretation` to every result with these exact values:

| Field | `relaxed-optimal` | `lru` |
| --- | --- | --- |
| `kind` | `lower-bound` | `causal-policy` |
| `objective` | `whole-trace-demand-loads-and-bytes` | `whole-trace-demand-loads-and-bytes` |
| `schedule_scope` | `atomic-full-group-residency-or-fixed-sorted-sequence` | `atomic-full-group-residency` |
| `diagnostics_are_bounds` | `false` | `false` |
| `order_flexible_tiling_covered` | `false` | `false` |

`by_phase`, `by_segment`, `forwards`, `decode_tokens`, `cold_start`,
`decode_distribution` and blocked-layer counts are policy outcomes, never
phase/tail/latency bounds. State this beside the metadata and in user docs.
No automatic feasibility verdict or tok/s conversion is added.

Two regression witnesses pin the reason:

1. Capacity 2, singleton prefill groups A,B,C,B,C, then 19 singleton decode
   groups A. Total-optimal loads 4: prefill 3, decode 1; decode nearest-rank
   p95 is one expert. A causal policy pinning A and rotating its other slot
   loads 5 entirely in prefill, with zero decode loads/p95. Total optimum
   does not imply a decode optimum. Implement the pin-A witness only in tests.
2. Capacity 2, groups {0,2},{1,2},{0,1}. Sorted sequence 0,2,1,2,0,1 needs
   4 loads; sequence 0,2,2,1,0,1 needs 3. The latter can serve resident 2
   first, then replace it with 1 while retaining 0. A bound for atomic
   residency cannot rule out such tiled execution. Test the second sequence
   through the existing sequential oracle, not a new production scheduler.

Also pin oversized prefill union: rows [0,1] and [1,2] require three slots
under the atomic model even though each row fits two. Both policies reject
at two slots with the existing explicit unsupported-model diagnostic.

## Version 2 capture manifest

Continue accepting version 1 with its existing semantics. Version 2 uses the
same requests and required header fields, plus a required `capture_manifest`
object for both synthetic and measured traces. No new real captures exist;
no migration or native recorder is included. Version 1 measured evidence is
explicitly legacy and unsuitable for a decisive feasibility verdict.

Add `capture_manifest: dict | None = None` to `Header`. Omit that field from
version 1 report headers so their existing shape stays compatible. Preserve the entire
version 2 manifest, including extension keys at any depth, as JSON values in
`trace.capture_manifest`. Do not silently discard recognized provenance.
Version 1 containing `capture_manifest` fails at line 1 with a version-2
requirement rather than silently dropping it. Other unknown header fields
retain legacy behavior; producers put new evidence inside the manifest.

Manifest version 1 requires the fields below. A string means nonempty string;
SHA-256 means exactly 64 lowercase hexadecimal characters. Object means a JSON
object, including an empty object where permitted. Counts are integers,
excluding booleans. All JSON numbers must be finite, including overflowed
exponents such as `1e999`, at every nesting depth. Existing duplicate-key
rejection applies recursively. Missing/wrong fields name their manifest path
and line 1. Extra fields are preserved and validated as finite JSON.

| Section | Required fields and types |
| --- | --- |
| root | `manifest_version`: integer 1; `capture_id`: string; `complete`: boolean true |
| `artifacts` | `model_id`, `tokenizer_id`, `template_id`: strings; `token_stream_sha256`: SHA-256 |
| `build` | `engine_revision`, `source_state_id`, `binary_sha256`, `recorder_revision`: strings, except `binary_sha256` is SHA-256 |
| `execution` | `context_tokens`, `prefill_chunk_tokens`: positive integers; `numerical_mode`, `requested_kernel`, `effective_kernel`, `device`, `os`: strings; `capabilities`: object |
| `generation` | `mode`: `generated` or `teacher-forced`; `settings`: object |
| `workload` | `protocol_id`, `conversation_id`, `repository_id`, `split`, `continuation_kind`: strings; `split`: `tuning` or `evaluation` |
| `extent` | `forwards`, `positions`: positive integers; `first_position`, `last_position`: nonnegative integers |

The manifest's `artifacts.model_id`, `build.engine_revision`, and
`execution.numerical_mode` must equal the corresponding header fields.
`source_state_id` identifies clean source or dirty patch contents; a bare
`dirty=true` does not distinguish two builds. Kernel fields name the actual
selected path, including fallback, rather than merely an environment setting.
Capabilities record relevant E1/Metal support. Generation settings record all
effective sampling options and seed (or explicitly null if unavailable).
Token-stream digest refers to SHA-256 of the full captured input/generated
IDs encoded as consecutive unsigned 32-bit little-endian integers, once per
position, not once per layer. IDs outside that range require a new contract.
Artifact/tokenizer/template identifiers must resolve to immutable content
identities in measured captures; synthetic fixtures use declared fixture IDs.

The parser validates structure and consistency, not the truth of hardware,
artifact or build claims. It does not open model/tokenizer files or rehash
captured tokens that are absent from the routing trace. Report provenance
status as `synthetic-declared`, `legacy-declared` (v1 measured), or
`manifest-declared` (v2 measured); never `verified`.

At EOF compare extent with validated requests: count forwards once per
layer-zero request, count its positions once, compare first/last absolute
positions and require last-first+1 == positions. This detects a trace cut
at a complete-forward boundary against its declared manifest. Require
`context_tokens > last_position` and every prefill forward's row count <=
`prefill_chunk_tokens`. Decode has its existing one-position rule.
Use next-expected-line diagnostics for extent mismatch at EOF. `complete`
false is a line-1 error. This is consistency checking, not tamper evidence:
a producer can falsely declare shorter extents.

Future recorder publication: write body privately, finalize counts and token
digest, then publish header plus body as one complete artifact. Record gaps
or failures as failed captures, never as a successful shorter run. S1 does
not implement the recorder or an in-progress trace format.

## Identify reports

Add report `evidence` with `trace_sha256`, `replay_source_sha256`,
`python_version` and `provenance_status`. Hash exact input file bytes before
UTF-8 decoding in the CLI (including CRLF/whitespace); parse that same byte
snapshot using JSONL newline handling that preserves Unicode characters
inside strings. No reread of a changing trace. Hash the executing standalone script
contents for replay identity, rather than assuming a clean Git checkout.
The digest identifies bytes, not authenticity or validated model quality.

Keep `load_trace(lines: Iterable[str]) -> Trace`. Extend
`run_sweep(trace, capacities=DEFAULT_CAPACITIES, *, trace_sha256=None) -> dict`.
Programmatic replay cannot recover original bytes: null means unavailable,
never a reconstructed JSON hash. Supplied trace digests must match the SHA-256
syntax; the caller declares their association. `replay` remains unchanged in
signature; the new interpretation metadata is additive. Report source identity
and Python version for programmatic use too. File-read/decode errors fail
before stdout publication, using the existing concise CLI error path.

## Evaluation and later native gates

The evaluation protocol is frozen provisionally as version 1 before tuning;
actual workload content and deployment budget must be recorded before decisive
capture. S1 does not invent performance measurements or an agreed threshold.
Append-only traces cannot model rewind, reset, compaction or held-session
cache reuse. Extend lifecycle semantics before measuring those scenarios.

Later S4 must prove cached Q8 decode and E1 prefill with original reduction
order, resident-path byte equality, GPU-readable addresses, all-hit execution
without additional host waits, forced-miss readiness and exact resume boundary,
slot lifetimes, cancellation and read-error recovery. The existing Metal
validator's top-6 and host wait are not a Mellum top-8 solution.

Refresh native peak-memory accounting: 1.05 GiB fixed, roughly 3 GiB experts,
about 0.40 GiB 8K session state are planning inputs, not measured process peak.
E1 global half buffers add 19,398,656 logical bytes (18.5 MiB) at T=1024:
1024*2304*2 + 1024*8*896*2. These are outside the session request table.
Include global/backend scratch, staging, recorder, in-flight slots, allocation
rounding and whole-machine pressure simultaneously. This laptop is the first
functional test machine; physical 8 GB validation remains required later.

## Builder check

A fresh independent builder read this design and evaluation protocol on
2026-10-27 and reported “nothing important left.” The user authorized the implementation plan; its CPU tasks are now implemented.

## Acceptance

The two bound witnesses, union rejection, v1 compatibility, v2 preservation,
manifest rejection/completeness and CLI byte-identity regressions pass, followed
by `make test-mellum-ssd-replay` and the full model-free Mellum suite.
Documentation describes both supported versions and the S1 additions.
No GPU, inference or storage result is claimed by completing S1.
