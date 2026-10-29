# Mellum SSD evaluation protocol

Protocol `mellum-ssd-evaluation-v1`, 2026-10-27. This freezes evaluation rules,
not a measured workload or a performance verdict. No decisive captures yet.
See the [roadmap](../ROADMAP.md) and [S1 design](superpowers/specs/2026-10-27-mellum-ssd-evidence-design.md).

## Freeze before capture and tuning

Create an immutable workload manifest containing exact prompt/tool transcripts,
repository/content identities, tokenizer/template identities, context lengths,
generation configuration and all window boundaries. Cover initial code prompts,
actual generated code, incremental tool-result appends and topic changes.
Include short, medium and intended maximum context within the eventual memory
budget. Name concrete token lengths and case counts before decisive collection;
S1 does not choose them without target use cases or memory measurements.

Assign whole conversations and repositories to tuning or evaluation before
policy selection. Do not split random tokens from one conversation across both.
Teacher-forced and self-generated continuations are separate cohorts; do not
pool them into one locality claim. Record the token stream identity and actual
execution path for every capture, including E1 fallback, chunk size and source
state. A changed configuration or workload manifest creates a new cohort.

Choose policies and their parameters on tuning data. Freeze policy code/digest,
windows and the decision budget before opening evaluation results. If evaluation
informs a new policy, obtain a new held-out set and disclose the previous use.
Predictors see only causally available information, with prefetch deadlines and
in-flight memory accounted for. Future-aware replay is a comparison bound.

## Cache and windows

Each independent trace starts cold. Replay its entire prefix before summarizing
a warm window; slicing the raw trace and starting a fresh cache is a different
experiment. Keep cold start, prefill, post-append recovery and sustained decode
separate. Declare boundaries by forward/position and event in the frozen
workload manifest; publish whole-trace demand alongside window summaries.
No window-analysis implementation is included in S1.

The current format is append-only. Reset/rewind/compaction and cache reuse
across held sessions need explicit lifecycle events and cache semantics before
capture; do not concatenate independent traces to imply persistent cache.
Idle-prefetch claims additionally need measured idle durations and deadlines.

## Decision evidence

Report per-case and aggregate distributions, complete token series, cache
capacity, effective slots, demanded bytes, capture/replay identities and all
exclusions. Total-optimal demand is a whole-trace bound under its scheduling
scope; phase/window/p95 outcomes are not independent lower bounds. Atomic-model
failure does not reject order-flexible tiling. Neither converts directly to
latency or model quality.

Before a feasibility verdict, record an agreed performance/latency checkpoint,
its workload windows and host-memory reservation. 10 tok/s is only a proposed
checkpoint. State what schedule/objective a negative conclusion rules out.
Do not treat synthetic routes or legacy v1 provenance as decisive Mellum data.

Measure storage separately with actual expert ranges (gate/up/down), candidate
contiguous layouts, intended concurrency, staging and latency distributions.
Distinguish OS-cache-warm reads from physical SSD reads and report cache controls.
Then measure native streamed inference; overlap and stalls cannot be inferred
from bandwidth alone. Capture overhead is not product throughput.

Use the development laptop first, recording actual device/RAM/OS/storage.
An M3 Pro starts at 18 GB; it cannot establish an 8 GB whole-machine result.
Validate physical 8 GB memory pressure, swap/compression and useful steady-state
latency later. Compare streamed arithmetic with the matching resident path;
unchanged routing/Q8 bytes and byte equality are the baseline. Cache-biased
routing is a separate behavior/quality experiment, outside this protocol.
