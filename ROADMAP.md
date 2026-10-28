# Mellum roadmap

Updated 2026-10-28. [MELLUM.md](MELLUM.md) records the port, measurements and
numbered checkpoints. This roadmap tracks the separate SSD investigation.

## Current baseline

- Mellum Checkpoint 3.6: `59932c9`, rebased onto upstream `ee7e840`, with E1.
- CPU replay preparation: implemented on `codex/mellum-ssd-replay`; 33
  synthetic replay tests and the model-free Mellum suite pass.
- First diagnostic resident routing captures now pass on an M5 Max / 128 GiB
  laptop. The bounded CPU warm-read batch is complete; physical SSD traffic
  remains unverified. No Mellum SSD inference has run, and physical 8 GB
  feasibility is unresolved.
- This branch retains the Checkpoint 3.6 baseline. The separate `mellum`
  checkout has advanced to `f635616` with frontend work; this SSD branch does
  not integrate or change it. SSD phases keep their own numbering.

## SSD streaming milestones

Target order (owner decision 2026-10-28, [retarget](docs/superpowers/research/2026-10-28-mellum-ssd-retarget.md)):
16 GB first, with Q8 weights unchanged, then 8 GB (S6). Streaming stays opt-in
(`--ssd-streaming`); with it off, resident speed must not regress.

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

### S3 — Locality and read service assessed: complete

The user selected `../ac-demo` and its roadmap as the workload source. The
[S3a proposal](docs/superpowers/specs/2026-10-27-mellum-storage-evidence-design.md)
freezes that clean starter's inputs, reserves it for evaluation and designs
CPU-only expert-range/contiguous-layout read tools. Its independent builder
check found no remaining material question and the user approved the design.
The [source-freeze plan](docs/superpowers/plans/2026-10-27-mellum-workload-source.md)
and [storage plan](docs/superpowers/plans/2026-10-27-mellum-storage-evidence.md)
were approved for inline execution. The [source freeze](docs/superpowers/research/2026-10-27-mellum-workload-source-evidence.md)
preserves all nine tracked files at the pinned commit. The
[CPU storage evidence](docs/superpowers/research/2026-10-27-mellum-storage-evidence.md)
now includes checked original ranges, an exact 11.79 GB packed payload and all
72 bounded native runs. Packed warm reads were 14–30% faster, but both cache
labels produced 29–78 GB/s; physical SSD traffic remains unverified. No generated
representative cohort or streamed inference has run. The next
[device-observation proposal](docs/superpowers/specs/2026-10-27-mellum-device-observation-design.md)
brackets four CPU read runs with mapped block-driver counters and idle controls;
the design is approved, its [implementation plan](docs/superpowers/plans/2026-10-27-mellum-device-observation.md)
is implemented and CPU validated. Its first live attempt refused during
device discovery; the saved-metadata WholeDisk repair passed the CPU gate.
A separately authorized fresh attempt completed all four cells, with zero
block-driver read bytes in every timed bracket and adjacent idle control,
including both cache-avoidance cells. These warm-file timings leave SSD miss
cost unresolved; `physical_io_verified=false` remains. See
[execution evidence](docs/superpowers/research/2026-10-27-mellum-device-observation-evidence.md).

S3 now has its first candidate-layout storage-service measurement. The next
work must add representative locality and remaining read costs so the results
can inform scheduling. Tool qualification alone does not establish feasibility.

| Next result | Evidence needed | GPU needed? |
|---|---|---|
| SSD read cost | Aligned packed subset at one worker measured: service p50 0.655–0.658 ms/expert in three storage-observed repetitions. Original GGUF gate/up/down ranges and concurrency remain open; retain cache/counter uncertainty | No; CPU and SSD only |
| Representative cache locality | Frozen Phase 1 ac-demo workload captures and held-out replay at roughly 3 GiB expert cache; miss bytes and demand bursts for warm, post-append and whole-session windows | Yes for native routing captures; preparation and offline replay use CPU only |

The [storage-service design](docs/superpowers/specs/2026-10-27-mellum-storage-service-design.md)
for controlled reads on this laptop is approved; its [implementation plan](docs/superpowers/plans/2026-10-27-mellum-storage-service.md)
is complete for its one bounded attempt. [Evidence](docs/superpowers/research/2026-10-27-mellum-storage-service-evidence.md)
records qualifying calibration plus all three repetitions, exact byte checks,
closing verification, passing CPU gates and independent review/audit. On this
128 GiB M5 Max, native logical read throughput was 9.875–9.954 GB/s for the
three repetitions; this is not inference stall cost or 8 GB feasibility.
Keep the completed attempt and its reservation; no automatic rerun follows.
The first ac-demo task remains Phase 1 only; later phases
require their own requests and prior-phase checkpoints.

The [first generated Phase 1 plan](docs/superpowers/plans/2026-10-28-mellum-generated-cohort.md)
has its first locality result, [attempt -02](docs/superpowers/research/2026-10-28-mellum-generated-phase1-02-evidence.md)
(the [first attempt](docs/superpowers/research/2026-10-28-mellum-generated-phase1-evidence.md)
failed CPU preparation and is retained). One generated two-turn conversation
(n=1) passed byte-exact resident parity on all 5122 operations. Both turns hit
their 4096/1024 caps inside thinking. Cold global grouped LRU at 3 GiB: warm
decode about 352–365 MiB/token of miss demand, p95 about 634–640 MiB, worst
1117 MiB (1142 MiB over all decode). Row misses are about 25%. The post-append and tail windows do not
reset. The analysis is an artifact-only validation after a publisher
serializer fix, with an independent audit. Tool-result, answer-phase and
medium/8K coverage remain open. Demand is not stall time. S4 remains separate
until this result is reviewed.

The [S3 scheduling assessment](docs/superpowers/research/2026-10-28-mellum-s3-scheduling-assessment.md)
combines the two results under stated assumptions. Those are atomic group
residency, on-demand reads, one serial worker at 0.655 ms per expert, and
reads added to compute. Mean serial reads are 37.0 ms/token, p95 66.8 and
worst 119.2. No token was all-hit, which rules out an all-hit warm regime
for this trace. Every token under 100 ms is ruled out by reads alone. The
recorded resident decode compute (7.5–8.6 ms/token, n=3, MELLUM.md) puts the
serial mean at about 22 tok/s, with reads 81–83% of token time. Overlapping
reads with compute can recover at most that compute.

Closed 2026-10-28 by the [deep review and retarget](docs/superpowers/research/2026-10-28-mellum-ssd-retarget.md).
The read term is bandwidth-bound: about 10 GB/s, so bytes per token decide
the ceiling. A descriptive miss curve on the held-out trace (n=1) gives LRU
demand of 651 MB/token at 2 GiB, 372 at 3, 47 at 7 and 23 at 8. A 10 tok/s
mean at 3 GiB would need sustained miss-order reads of about 3.7 GB/s or
more. Carried open: physical cold reads, concurrency, and answer-phase or
tool-result locality.

Exit met: measured miss curves and storage-read performance, followed by an explicit
assessment of what they permit or rule out under stated scheduling assumptions.
These measurements inform S4/S5; they do not establish streamed generation speed
or physical 8 GB feasibility. Any rejection names the scheduling class and
objective it actually rules out. 10 tok/s remains a proposed checkpoint, not a
requirement or prediction.

### S3b — 16 GB fit, measured on a 16 GB Mac: paper only

On paper, a 16 GB Mac's Metal `recommendedMaxWorkingSetSize` is about 10.67
GiB (secondary sources, unverified). That is below resident Q8's 12.44 GiB
plan at 8K, so Q8 needs streaming there. The engine's 7/8 budget leaves
about 7 GiB of expert cache. Measure on a 16 GB Mac before S5 relies on it:
the real working-set limit, whether resident Q8 runs without swapping, and
optionally that machine's miss-order read service. No GPU work on this
laptop is needed.

Exit: measured limit and resident behavior on 16 GB, and the expert-cache
budget derived from them. Requires borrowed hardware.

### S4 — Cached execution and miss recovery proof: complete on the development laptop

The [design](docs/superpowers/specs/2026-10-28-mellum-s4-cached-execution-design.md) and
[plan](docs/superpowers/plans/2026-10-28-mellum-s4-cached-execution.md) are implemented:
`--ssd-streaming` for Mellum. Decode stops only at missed layers; prefill syncs per layer.

[Measurement 02](docs/superpowers/research/2026-10-28-mellum-s4-measurement-02.md), at 32K and a 6 GiB
cache, with device-true reads (0.996–1.002, n=3):
- **Exact:** 3 of 3 runs give resident's 2,048 token ids.
- **Memory:** peak footprint 6.93 GiB, Metal allocation 7.81 GiB.
- **Speed:** decode 26–27 t/s against 92 resident; time to first token about 60 s against 29 s.
- **No resident regression,** from [measurement 01](docs/superpowers/research/2026-10-28-mellum-s4-measurement.md)
  (n=6 per arm).

01's streamed reads were cache-served; the [spike](docs/superpowers/research/2026-10-28-mellum-s4-cached-reads-spike.md)
found why and fixed it. By arithmetic, prefill is read-bound at about 10.8 GB/s. Decode has about
18 ms/token of overhead beyond its reads and syncs.

Next: profile the stop/resume path, then S5.

### S5 — Bounded streaming and deployment validation: not started

Implement the measured cache/read policy, preserving Q8 weights and routing.
Account for fixed tensors, session state, E1/global backend scratch, cache,
staging, recorder state and in-flight references at the same peak. Compare
resident and streamed behavior under the same numerical configuration.

Exit: correct bounded engine on the development laptop, then whole-machine
memory pressure and useful latency on physical 16 GB hardware. Native host
integration in `../ds4-engine` follows engine correctness and measurements;
Python owns workflow policy, while C/Rust owns inference-critical scheduling.

### S6 — 8 GB target: deferred

Reopen when one of these holds:
- a target-class SSD read bench measures sustained miss-order reads of about
  3.7 GB/s or more at 6.6 MB;
- two or more non-thinking conversations show demand at 3 GiB well below the
  -02 trace's 372 MB/token;
- the owner accepts lower-bit streamed experts (about a quarter of the bytes
  at 3 GiB, estimated).

Also needed: an 8 GB memory plan with the measured Metal limit.

## Working rules

Use Superpowers design/planning per milestone; plan later implementation only
when preceding measurements settle its choices. Keep one concern per commit,
commit a failing behavioral test separately before its fix, and keep product
changes to roughly 150 lines per commit. Use messages that name the concrete
behavior changed. Documentation-only changes do not need executable tests.

The worktree and branch remain available for subsequent phases. Record any
new measurement baseline explicitly; leave main/port integration to its owner.
