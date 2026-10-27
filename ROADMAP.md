# Mellum roadmap

Updated 2026-10-27. [MELLUM.md](MELLUM.md) records the port, measurements and
numbered checkpoints. This roadmap tracks the separate SSD investigation.

## Current baseline

- Mellum Checkpoint 3.6: `59932c9`, rebased onto upstream `ee7e840`, with E1.
- CPU replay preparation: implemented on `codex/mellum-ssd-replay`; 33
  synthetic replay tests and the model-free Mellum suite pass.
- No Mellum SSD inference, real routing capture or storage benchmark has been
  performed by this investigation. Physical 8 GB feasibility is unresolved.
- The other agent's port steps 34–40 remain parked pending Paul's go. SSD
  milestones do not authorize those steps or depend on adopting their numbering.

## SSD streaming milestones

### S0 — Offline replay foundation: complete

Version 1 trace validation, grouped LRU, fixed-order relaxed optimal replay,
byte-demand statistics and synthetic tests. See the [completed plan](docs/superpowers/plans/2026-10-27-mellum-ssd-replay.md).

Completion establishes accounting under the declared model. The later review
identified interpretation and capture-provenance gaps, addressed by S1.

### S1 — Qualify replay evidence and prepare captures: complete, CPU only

- Restrict the lower-bound claim to whole-trace total loads/bytes under the
  atomic residency model or the specified fixed access order.
- Identify phase/window/tail statistics as policy outcomes, and prevent a
  future feasibility decision from treating them as phase-specific bounds.
- Preserve an explicit measured-capture manifest and identify trace/replay
  inputs. Keep version 1 evidence labeled as legacy declared provenance.
- Freeze a collection/evaluation protocol before tuning policies or capturing
  the evidence used for a decision.

Exit met: 33 synthetic replay tests and the model-free Mellum suite pass.
Independent whole-branch review found one Unicode JSONL parsing issue; a
separate red-test/fix pair resolves it, followed by the full CPU gate.
See the [S1 design](docs/superpowers/specs/2026-10-27-mellum-ssd-evidence-design.md)
and [implementation plan](docs/superpowers/plans/2026-10-27-mellum-ssd-evidence.md).

### S2 — Native router recorder: GPU phase, not started

Buffer selected IDs on the GPU before per-layer scratch is reused; export at
existing synchronization points. Bound recorder memory and reject incomplete
captures. Preserve routes and logits against the same resident execution path.
Report the actual kernel configuration and produce the S1 manifest.

Exit: traced/untraced parity; no added per-layer host wait; bounded recording
storage; a capture harness that exercises appends without requiring steps 34–40.
Instrumented timing is not product throughput evidence.

### S3 — Locality and storage evidence: not started

Collect the frozen workload matrix on this development laptop. Replay warm,
post-append and whole-session demand separately, with held-out evaluation.
Measure the actual gate/up/down read ranges and candidate contiguous layout,
including concurrency and latency, with OS-cache effects identified.

Exit: measured miss curves and read performance with explicit assumptions.
Any rejection names the scheduling class and objective it actually rules out.
10 tok/s remains a proposed checkpoint, not a requirement or prediction.

### S4 — Cached execution and miss recovery proof: not started

Prove GPU-readable Q8 slots through decode and E1 prefill. Specify address
translation, slot ownership/readiness and the resume boundary before building
a full scheduler. Exercise all-hit, forced-miss, eviction, cancellation and
read-failure paths. Preserve arithmetic and original expert reduction order.

Exit: equality with the matching resident path; no additional host wait on
all-hit layers; safe miss recovery and slot reuse; refreshed peak-memory plan.
The existing top-6 host-waiting validator is not Mellum's top-8 solution.

### S5 — Bounded streaming and deployment validation: not started

Implement the measured cache/read policy, preserving Q8 weights and routing.
Account for fixed tensors, session state, E1/global backend scratch, cache,
staging, recorder state and in-flight references at the same peak. Compare
resident and streamed behavior under the same numerical configuration.

Exit: correct bounded engine on the development laptop, then whole-machine
memory pressure and useful latency on physical 8 GB hardware. Native host
integration in `../ds4-engine` follows engine correctness and measurements;
Python owns workflow policy, while C/Rust owns inference-critical scheduling.

## Working rules

Use Superpowers design/planning per milestone; plan later implementation only
when preceding measurements settle its choices. Keep one concern per commit,
commit a failing behavioral test separately before its fix, and keep product
changes to roughly 150 lines per commit. Use messages that name the concrete
behavior changed. Documentation-only changes do not need executable tests.

The worktree and branch remain available for subsequent phases. Record any
new measurement baseline explicitly; leave main/port integration to its owner.
