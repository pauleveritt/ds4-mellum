# Mellum roadmap

Updated 2026-10-27. [MELLUM.md](MELLUM.md) records the port, measurements and
numbered checkpoints. This roadmap tracks the separate SSD investigation.

## Current baseline

- Mellum Checkpoint 3.6: `59932c9`, rebased onto upstream `ee7e840`, with E1.
- CPU replay preparation: implemented on `codex/mellum-ssd-replay`; 33
  synthetic replay tests and the model-free Mellum suite pass.
- First diagnostic resident routing captures now pass on an M5 Max / 128 GiB
  laptop. No Mellum SSD inference or storage benchmark has been performed.
  Physical 8 GB feasibility is unresolved.
- This branch retains the Checkpoint 3.6 baseline. The separate `mellum`
  checkout has advanced to `f635616` with frontend work; this SSD branch does
  not integrate or change it. SSD phases keep their own numbering.

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

### S2 — Native router recorder: complete on the development laptop

**S2a, CPU only:** native binary spool/layout helpers and bounded exporter to
v2 traces, tested with fabricated selections. Includes completeness, write-error
handling, exact-byte/token identities and atomic no-overwrite publication.
The [capture design](docs/superpowers/specs/2026-10-27-mellum-route-capture-design.md)
and [implementation plan](docs/superpowers/plans/2026-10-27-mellum-route-capture.md)
are implemented, with native checks and 12 Python export tests passing.
See [usage](docs/MELLUM_ROUTE_CAPTURE.md). S2a is complete: independent
whole-branch review found no blockers. One existing replay-CLI nesting
error-message issue is recorded as a deferred minor in the plan.

**S2b, live validation complete:** the earlier
[primitive proof](docs/superpowers/research/2026-10-27-mellum-route-gpu-proof.md)
passes eight cases with byte-identical routes/logits and no added buffered
per-layer host wait. The approved
[session design](docs/superpowers/specs/2026-10-27-mellum-session-capture-design.md),
[CPU-core plan](docs/superpowers/plans/2026-10-27-mellum-capture-cpu-core.md),
[owned adapter](docs/superpowers/plans/2026-10-27-mellum-owned-capture-adapter.md)
and [qualified publisher](docs/superpowers/plans/2026-10-27-mellum-qualified-publisher.md)
are now implemented and live validated. Release/exact/fallback paths,
short/tiled/multi-chunk/append schedules, no-head profiles, per-operation logits,
cancellation, copy/read/sink failures and sequential handles pass.

The optional failure target covers 107 creation allocation boundaries and
matching frees. A shared Metal read/write/contents lifetime defect found during
measurement was reproduced in `b84a3fa` and repaired in `e567a5f`. Recorder-owned
storage measures 2097200 bytes at cap 1024, within the 2.2 MB bound. After repair,
all eight ordinary/owned sessions return to the same device allocation baseline,
and post-work device allocations remain stable per arm. The resident kernel,
fast-path, failure matrix and final serialized model-free suite pass.

Four source-checked v2 artifacts at `fa3d188` and a post-repair artifact at
`e567a5f` round-trip through replay at 3 GiB. The post-repair ordered trace rows
and replay results exactly match the earlier tiled capture. Recorder-disabled
comparison found no resolved generation regression before the lifetime repair;
post-repair diagnostic recording adds 2.53–8.47% decode time in three adjacent
pairs, with timing drift. These fixtures are diagnostic tuning data, not S3's
representative cohort. See the
[owned live evidence](docs/superpowers/research/2026-10-27-mellum-owned-live-proof.md)
for source identities, measurements, review findings and limits.

Exit met on M5 Max / 128 GiB: matching-path parity, no added per-layer host
wait, bounded recorder storage and append-capable qualified collection. This
establishes neither physical 8 GB peak memory nor SSD inference performance.

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
