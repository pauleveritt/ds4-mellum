# Mellum diagnostic session recorder

Date: 2026-10-27. Status: proposed S2b design, awaiting approval.
Baseline: `8d9c53c`, isolated `codex/mellum-ssd-replay` worktree.

## Intent and evidence

Turn the [GPU primitive proof](../research/2026-10-27-mellum-route-gpu-proof.md)
into a bounded diagnostic recorder with trustworthy operation labels, failure
semantics and observed kernel choices. Preserve trained routes, Q8 bytes and
the matching resident numerical path. Remain on this branch; no push or merge.
GPU access has been released. CPU implementation can proceed after design approval and
implementation planning; live Metal validation waits for another authorized window.

S2a supplies the native spool/layout/writer and bounded v2 exporter. The test-only
S2b proof established ordered buffered/reference snapshots and matching final
operation logits on eight cases. It did not establish production ownership,
cancellation, intermediate/no-logits behavior or actual kernel provenance.
The exploratory 3 GiB LRU results are not qualified decision evidence.

These are my implementation choices, not additional user requirements:
use an opaque diagnostic handle that owns a fresh ordinary session; retain the
existing spool; collect bounded canonical kernel descriptors; fail the capture
on any error. Keep inference-critical work in C99/Objective-C, Python 3.14+
stdlib for metadata/export. No C++, Python scheduling in the forward path,
cache-biased routing, streaming backend or host integration.

## Ownership boundary

Prefer an owned diagnostic session over attaching to an arbitrary public
`ds4_session`. The general API exposes checkpoint loading, rewrite, rewind,
speculative and batch operations. Guarding all of them would spread recorder
policy through unrelated code. The diagnostic handle never exposes its inner
session; its only inference operations are text sync and one-token eval.

Add declarations in a separate `ds4_mellum_capture.h`, using opaque
`ds4_engine` and `ds4_tokens` from `ds4.h` and the existing native sink type.
Do not enlarge the general session API. Define the adapter in `ds4.c`, where
private session/graph details already live, with a separate GPU-free controller
and profile registry module. The engine remains caller-owned and must outlive
the handle. On CPU/non-Apple builds create returns an unsupported error without
initializing a GPU; declarations and stubs remain linkable.

Proposed interface (0 success, nonzero error; sync preserves cancellation=2):

- `ds4_mellum_capture_create(out, engine, context_tokens, options, err, errlen)`:
  sets `*out=NULL` on failure; validates a resident, non-distributed Mellum Q8
  Metal engine without speculative support, creates a fresh session at position
  zero, allocates recorder storage, then starts the spool. Options contain the
  spool sink/context and optional existing cancellation callback/context.
- `ds4_mellum_capture_sync(handle, full_prompt, segment_id, err, errlen)`:
  initial prompt or an exact append to the committed token prefix. Phase is
  prefill, including one-token tails/appends. An identical nonempty prompt is a
  successful no-op with no frame. Reject shortened/changed prefixes before GPU
  work; never reset and silently concatenate a replay.
- `ds4_mellum_capture_eval(handle, token, segment_id, err, errlen)`:
  one ordinary session eval, phase=decode. Segment IDs are explicit u32 values
  on every operation, constant across its prefill chunks. Tokens/segment names
  and sampling policy remain the caller's responsibility.
- `ds4_mellum_capture_copy_logits(handle, out, capacity)`:
  mirror ordinary copy semantics: vocabulary count on success, zero otherwise.
  Available after a successful inference operation and after successful finish;
  unavailable before any logits, while busy, or after failure.
- `ds4_mellum_capture_finish(handle, profile_sink, profile_context, err, errlen)`:
  emits the used profile catalog, then the existing native footer, exactly once.
  Profile callback receives `(context, u32 id, const char *description, size_t
  length)` and returns bool success. Description is NUL-terminated ASCII, length
  excludes NUL, valid during that callback only. Callback must be nonnull.
- `ds4_mellum_capture_free(handle)`: null-safe; destroys the inner session and
  recorder in every state, never emits a footer or calls either sink.

All operations, including free, are same-thread, serialized and non-reentrant;
callbacks cannot call capture/inference APIs or free the handle. Concurrent
inference on the engine/backend is unsupported during capture. Multiple handles
may be operated sequentially, with distinct storage and catalogs. No thread
safety, simultaneous-session scheduler, attachment to a held session or resume
from a failed capture is promised. Do not bypass the model instance lock.

Sinks and their I/O resources remain caller-owned. The adapter never opens,
flushes or closes files. Caller must check flush/close on spool and metadata,
then use the existing exporter for atomic no-overwrite publication. Successful
finish is necessary, not sufficient, for a publishable artifact.

## Lifecycle and failure

Create returns a READY handle, or no handle with partial resources freed. Allocate
all recorder/session resources before writing the native header. Header short
write fails create; private partial bytes may remain in the caller's sink.

READY -> BUSY -> READY for successful sync/eval. Any operation error, cancelled
sync, allocation/copy/read failure, missing/overflowed profile, short sink write,
invalid prefix or footer/catalog failure is terminal FAILED. FAILED rejects
further inference, logits and finish without callbacks. Caller may only free.
An interrupted sync may already have complete frames, but gets no valid footer.
A first-call cancellation also fails this capture; no automatic retry.

Finish requires READY and at least one complete forward. Empty/double finish,
mutation after finish and missing profile sink return errors without new bytes.
Catalog callbacks run before footer writes; catalog failure leaves no footer.
Successful finish makes the handle FINISHED/read-only. Failure while attempting
an invalid operation on a FINISHED handle leaves its already-complete artifact
unchanged. Never restart a handle or reset a failed writer.

The actual GPU adapter always executes ordinary command-buffer cleanup before
returning an inference error and before freeing ID storage. A profiler overflow
mid-encoding must drain existing work and reject the forward, not return early
past cleanup. Free occurs outside operations; there is no background readback.
A sink can fail after GPU computation changed KV; the failed handle is never
reused. This stricter diagnostic contract leaves ordinary sessions unchanged.

## Graph recording and bounded storage

Each owned graph borrows its handle's recorder pointer while that session lives.
The handle owns one cap-strided GPU ID tensor and one host ID buffer. Allocate
using the checked S2a layout and actual session cap, never requested env alone.
Each is `28 * cap * 8 * 4` bytes; cap must be 1..1024. Copy `T * 8 * 4` active
bytes after each successful layer and before selected scratch reuse. Read only
after the existing successful forward completion, then serialize active rows.
No new per-layer CPU read, command-buffer commit or wait in buffered mode.

Controller records operation phase/segment before ordinary sync/eval, verifies
positions remain contiguous from zero, and admits a frame only after all 28
layer snapshots, successful completion and a complete profile. Ordinary initial
session reset is permitted before the first forward. Later resets cannot occur
through this append-only interface. No-logits forwards use the same completion
boundary, with an explicit absent-head profile entry.

Use a bounded profile registry: at most 128 profiles, each a canonical set of
at most 64 descriptors. A descriptor occupies at most 32 bytes, with a static
function ID, stage/role and up to six named integer specialization values.
Names/constant labels live in a versioned static table, not unbounded strings.
Descriptors deduplicate and sort by their complete value; profiles are equal
only when all descriptor values and expected/absent head state match. Assign
profile IDs in first-observation order. Hashes, if used, never replace equality.
Overflow or unknown descriptors fail capture rather than truncate metadata.

Two ID buffers plus registry/current profile and a bounded 16 KiB temporary
catalog-description buffer must fit within 2,200,000 logical recorder bytes at
cap=1024. Tests compute the real sizeof-based bound. Inner session/model state,
Metal page rounding/global caches, caller buffers and stdio remain separately
accounted. No per-layer heap allocation; stream frames instead of retaining
session history. No extra whole-token-stream buffer is needed.

## Observe encoding, not requested settings

A scoped backend collector borrows the current forward's profile while encoding.
Reject nested scopes before starting work; clear it on every exit, including
failure, so the next handle cannot inherit it. No collector work/allocation or
new synchronization when recording is disabled. Default GPU kernel order,
arguments, precision and reductions stay unchanged; measure default-path cost
in the next GPU window.

Audit every math-kernel encoding reachable from `mellum_graph_forward_tokens`.
Record actual function identity and chosen specialization constants at successful
encoding branches. `qwen4_dispatch_resident` covers many Mellum/MoE/attention
kernels; Q8 projection generic/exact paths, embedding, norm, f32->f16 conversion
and residual add require their non-qwen4 encoding sites too. Pipeline lookup or
requested environment values alone are insufficient evidence.

Use graph scope tags to distinguish projections and the output head when they
share a wrapper. Record exact/tiled MoE, selected NAX/SIMD-group kernels, main
and tail dispatches and any fallback that actually encoded. Include auxiliary
math kernels, not just the experts. Exclude observer blits. Every successful
frame must have observed embedding, projection, router/top-k, expert mid/down,
reduce and attention roles, plus head when requested. No-logits frames explicitly
say head=not-encoded. Missing roles fail instead of fabricating a profile.

The profile describes the set of actual math-kernel choices for a forward,
not dispatch order/counts, GPU elapsed times or per-layer attribution. Shape,
position, phase and source identity remain separate metadata. Deduplication
must not erase specialization choices that change arithmetic.

The harness records requested controls and actual capabilities separately.
For qualified runs require immutable source inputs: identify/hash the binary,
C/build state and all runtime-loaded Metal source files; reject source overrides
in the qualified harness and check default source files unchanged before/after
collection. Capture identity is consistency evidence, not adversarial attestation.
Never treat `engine_revision` alone as runtime Metal source identity.

## Metadata and harness

Reuse native MLRT0001 spools and the v2 exporter unchanged unless a demonstrated
interface gap requires a separately reviewed change. Each frame's execution ID
resolves to its observed canonical description from finish. The caller writes
catalog strings with real JSON escaping into `execution_profiles`, supplies
segment labels and the other S1 manifest fields, then exports only after checked
file closure. Recorder/exporter may not substitute requested NAX for observations.

Adapt the live capture harness to the opaque handle and compare it against a
fresh ordinary session and the existing deliberately synchronous reference.
Keep matching numerical configurations and teacher-forced/generated cohorts
separate. New captures may qualify recorder provenance; they still do not
establish representative workloads, warm locality or physical 8 GB feasibility.
Freeze representative conversations/generation/windows before decisive S3 data.

## Verification and sequencing

CPU-first: real controller/writer/registry tests for transitions, explicit phase
and segments, identical/invalid prefix, cancellation/error propagation, exact
buffer boundaries, full descriptor equality, dedup/ordering, overflow, nested
collector refusal, clear-on-error, sink/profile/footer failures and no footer on
free. Validate resulting complete/partial bytes with the existing reader/exporter.
CPU tests of supplied descriptors do not prove actual Metal observation sites.
Do not add a production GPU abstraction solely to fake temporal correctness.

Then wire the graph/Metal observer in small commits. Each behavior gets a failing
test committed separately before implementation; about 150 product additions per
implementation commit. Run model-free Mellum and explicit C99/CPU builds. Report
unavailable GPU gates; do not substitute huge CPU inference or run a GPU process.

Next authorized GPU window: per-forward routes and intermediate/final logits
against matching baseline/reference; direct no-logits forward then later logits;
short/tiled/multiple chunks and one-token append; observed path/tails/fallback;
allocation/copy/read/sink failures with safe cleanup; cancellation before/between
chunks; sequential two-handle isolation; recorder-off behavior and bounded actual
allocation. Re-measure capture/default overhead and run resident/kernel/fast-path
checks with one model process. Distributed/CUDA execution is outside this Metal
feature; all new symbols must still compile/link safely on their guarded paths.

S2b closes only after CPU and live gates pass and a valid observed-profile v2
artifact round-trips through replay. SSD storage/scheduler feasibility remains
S3 onward. General `make test` also needs the absent default DeepSeek GGUF;
record that constraint instead of silently substituting the Mellum model.

## Builder check

A fresh independent C99/Metal builder reviewed this design on 2026-10-27 and
reported “nothing important left.” No unresolved user-only question remains.
The ownership boundary, terminal failure behavior and bounded profile registry
are implementation choices presented here for approval, not prior agreements.
