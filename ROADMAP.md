# Mellum roadmap

Updated 2026-10-29. [MELLUM.md](MELLUM.md) records the port, measurements and
numbered checkpoints. This roadmap tracks the separate SSD investigation.

## Current baseline

- Mellum Checkpoint 3.6: `59932c9`, rebased onto upstream `ee7e840`, with E1.
- S0–S4 complete on `codex/mellum-ssd-replay`, unpushed. Opt-in
  `--ssd-streaming` runs Mellum on Metal from a bounded expert cache. On this
  M5 Max / 128 GiB laptop it matches resident output at 32K, with reads that
  reach the SSD (S4 below).
- Not established: streamed speed or memory on any other machine, including
  16 GB or 8 GB hardware. S5's targets are judged on the owner's M1 Pro.
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
about 7 GiB of expert cache. Measure on a 16 GB Mac before S5b relies on it:
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
- **Speed:** decode 26–27 t/s against 92 resident at 32K (resident is about 133–139 on a short prompt,
  MELLUM.md); time to first token about 60 s against 29 s.
- **No resident regression,** from [measurement 01](docs/superpowers/research/2026-10-28-mellum-s4-measurement.md)
  (n=6 per arm).

01's streamed reads were cache-served; the [spike](docs/superpowers/research/2026-10-28-mellum-s4-cached-reads-spike.md)
found why and fixed it. By arithmetic, prefill is read-bound at about 10.8 GB/s. Decode has about
18 ms/token of overhead beyond its reads and syncs.

The [decode profile](docs/superpowers/research/2026-10-28-mellum-s4-decode-profile-spike.md) followed
(n=1 per arm, short prompt). Gating the projections and copies after a stop (`9e9757f`) raised streamed
decode from 30.6 to 39.6 t/s, against about 136 resident. The rest scales with stops per token, not SSD
time: about 10.7 GPU passes per token, adding about 5.4 ms of GPU execution and 2.6 ms of per-buffer cost.
Loads take about 0.95 ms per stop, serially. On 2026-10-28 the owner chose not to pursue load overlap,
expecting a high-complexity change for a small gain (the spike's "Not pursued"); S5 step 2b did it on
2026-10-29. A check-kernel trim was tried and not kept. Measurement 02's 32K figures predate the gating.

### S5 — Streamed speed, on the development laptop and an M1 Pro: in progress

**Targets (owner, 2026-10-28),** judged on the owner's M1 Pro / 32 GB, not on this M5 Max, at a 6 GiB
cache and 32K:
- decode at least 15 t/s; below 10, lower-bit experts reopen (S6);
- time to first token at most 60 s for a cold 30K prompt, and at most 10 s after appending about 2K
  tokens to a warm session.

The M1 Pro is closer to a base-chip 16 GB laptop but still optimistic for decode, so a miss there
predicts a miss on 16 GB and a pass does not prove one. Its
[bring-up plan](docs/superpowers/plans/2026-10-28-mellum-m1pro-bringup.md) runs alongside steps 1–2.

Steps, in order. Each is measured before and after with its n, keeps Q8 weights and routing, and stays
exact against resident:
1. **Re-measure 32K here at the current build: done.** [Measurement 01](docs/superpowers/research/2026-10-28-mellum-s5-measurement-01.md)
   (C n=3, device-true, exact, no regression at n=6): decode 39.0 t/s median, up 49% from 02; cold first
   token 60.6 s; 29.5 t/s after a 1,905-token append, whose first token took 4.3 s. R's preregistered
   reuse-time check failed, so the per-turn figures are reported, not claimed.
2. **Layer-major prefill: done.** [Grouped prefill](docs/superpowers/specs/2026-10-28-mellum-s5-layer-major-prefill-design.md),
   K = 8 chunks per layer pass, streaming only. [Measurement 02](docs/superpowers/research/2026-10-28-mellum-s5-measurement-02.md)
   (C n=3, device-true, exact, no regression at n=6): cold first token 60.6 → 37.7 s median, prefill loads
   50,806 → 7,416, Metal 7.94 GiB. The preregistered load ceiling (7,168) was exceeded by 3.5%.
2b. **Cheaper stops: done (2026-10-29, on the owner's overnight instruction to act on a fresh review).**
   A stop's reads now run while the resumed pass is encoded, and the cache's victim scans stop at the
   installed area. [Result](docs/superpowers/research/2026-10-29-mellum-stream-decode.md) (preregistered,
   MXFP4 at 2.5 GiB on the M5, n=3): 44.8 → 24.7 ms/token (22 → 40 t/s), identical ids; 3 GiB 32.0 → 18.8 ms (n=1).
   The load overlap S4 had not pursued added 92 product lines and gave most of the gain (screen, n=2 per
   arm: 39.0 → 26.9 ms/token).
   Chunked reads, windowed passes and synchronous next-layer prefetch were screened and lost; pre-gating
   predicts 7.0 of the next layer's 8 experts, so an asynchronous prefetch stays a candidate for step 4.
3. **Fewer stops per token:** hotness for layers that hit, and the eviction policy. Choose on a fresh
   trace, never the held-out -02 trace or ac-demo. After the append, stops rose from 8.7 to 12.7 per
   token. Wait for the M1 Pro's baseline, whose cost per stop
   decides how much this is worth.
4. **Predictive prefetch,** only under its own design, if 2 and 3 leave too little.

Exit: the targets met on the M1 Pro, or the steps exhausted with a recorded verdict; exact output; peak
footprint and Metal allocation within about 8 GiB at 32K.

### S5b — Deployment validation on a 16 GB Mac: waiting on hardware

After S3b and S5; no 16 GB Mac is available yet. It checks what the M1 Pro cannot: a 16 GB Metal
working-set limit and a current base chip. Run S5's build on physical 16 GB hardware with the cache budget S3b derives. Account
for fixed tensors, session state, E1/global backend scratch, cache, staging, recorder state and in-flight
references at the same peak. Compare resident and streamed behavior under the same numerical
configuration.

Exit: whole-machine memory pressure, with an editor, a browser and a couple of apps open, and the S5
target latency on that machine. Native host integration in `../ds4-engine` follows engine correctness and
measurements; Python owns workflow policy, while C/Rust owns inference-critical scheduling.

### S6 — 8 GB target with MXFP4 experts: phases A-C done and merged, phase D paused

Deferred until 2026-10-28, behind three reopen conditions: a target-class SSD
read bench, better non-thinking locality, or lower-bit streamed experts.
**Owner decision, 2026-10-28:** lower-bit streamed experts are accepted for
8 GB. This breaks the "Q8 weights unchanged" rule for S6 only; 16 GB (S3b-S5)
keeps Q8. Work runs on `research/mellum-8gb-q4`, CPU-only and model-free
until S4/S5 merge and the GPU is free.

Target file: JetBrains' `Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf`
(7,029,537,056 B, HF snapshot `4159d684`). Its header, read without running
it, matches the Q8_0 file except for the routed experts:
- the 1.037 GiB of non-expert tensors are Q8_0, as in the Q8_0 file;
- all 84 expert tensors are MXFP4, 3,290,112 B per expert, half of Q8_0.

JetBrains' table gives it KLD 0.115 and 85.6% top-token match against BF16
(Q4_K_M 0.075 / 88.0%, Q8_0 0.008 / 96.1%; Wikitext-2, `n_ctx=512`). These
are their figures, not ours.

Re-estimated from the retarget's LRU curve (descriptive, n=1 thinking trace,
same routing assumed): a 3 GiB cache holds as many slots as 6 GiB at Q8, so
about 41.5 MB/token, not 372. At 2 GiB, about 114 MB/token.

The [code read](docs/superpowers/research/2026-10-28-mellum-mxfp4-metal-path.md)
at `01521b3` and `3addce0` found no new Metal kernels needed. Mellum's
expert kernels already branch on MXFP4. The work is about 8 resident Q8_0
gates, about 7 streamed ones, and a slot count derived from the file. Risk:
no production model runs MXFP4 gate/up rows through those kernels yet.
JetBrains' Q4_K_M is a larger port: there is no Q5_0 down kernel, and its
attention tensors are Q4_K/Q6_K.

Still needed: our own quality and routing-agreement checks against Q8_0, a
target-class SSD read bench, and an 8 GB memory plan with the measured Metal
limit. Target machine: an M1 iMac, 8 GB, 1 TB, macOS 27.1. The
[design](docs/superpowers/specs/2026-10-28-mellum-s6-mxfp4-8gb-design.md)
was approved on 2026-10-28.

Phase A (CPU only, `7160525..ff22dcc`): the loader opens all-MXFP4 experts
and refuses mixtures, streamed slots follow the file's expert size, and the
double reference reads MXFP4. Replay at MXFP4 size (n=1 trace): LRU 69.2
MB/token at 2.5 GiB ([curve](docs/superpowers/research/2026-10-28-mellum-mxfp4-miss-curve.md)).
Phase B ([plan](docs/superpowers/plans/2026-10-28-mellum-s6-b-m5.md), on the SSD tip by the owner's
call) passes its [preregistration](docs/superpowers/research/2026-10-28-mellum-s6-b-preregistration.md) on
the M5 Max: MXFP4 kernels within tolerance of the double reference; logits agree with llama.cpp on 157/157
positions (near-ties counted), mean KL <= 2.0e-3 ([logits](docs/superpowers/research/2026-10-28-mellum-s6-b-logits.md));
Q8_0 unchanged (B/A 0.991/1.000, n=6); streamed MXFP4 at 2.5/3 GiB gives resident ids (n=3), 19.98
loads/token at 2.5 GiB; resident MXFP4 decodes 1.13-1.20x Q8_0 ([measurement](docs/superpowers/research/2026-10-28-mellum-s6-b-measurement.md)).
Phase C's [plan](docs/superpowers/plans/2026-10-28-mellum-s6-c-imac.md) and [preregistration](docs/superpowers/research/2026-10-28-mellum-s6-c-preregistration.md)
are committed with the iMac kit (`s6-c-prereg/KIT.md`). The kit's build is now the merge `633fd23`, which
brings in the SSD tip and S5 step 2b (preregistration amended 2026-10-29; references rechecked). The M5's
reference for the Pi-shaped workload is streamed 2.5 GiB at 40.7 t/s, up from 23-24 (n=3,
[speed run](docs/superpowers/research/2026-10-29-mellum-speed-run/summary.md)). `ds4-server` does not serve Mellum, so the workload runs flattened.
Phase C [result](docs/superpowers/research/2026-10-28-mellum-s6-c-imac.md) (2026-10-29): C1 and C3 pass; C2's rule
picks 2 GiB, since 2.5 GiB's Metal peak is 24.8 MB over the limit, which fires the preregistration's design falsifier.
Apps closed, decode 5.85 t/s (n=3), first token 19.4 s, no swap growth at 2 GiB. C4's apps-open runs (no desktop
access) and C6 (no M4 Max) were not run, so both are undecided. Merged into `mellum` on 2026-10-29 (`c04037c`).
Phase D ([design](docs/superpowers/specs/2026-10-29-mellum-stream-prefetch-design.md),
[plan](docs/superpowers/plans/2026-10-29-mellum-s6-d-prefetch.md)) **paused 2026-10-29 by the owner** after Task 1's cache
stage and Task 2b:
- On the merged build 2.5 GiB is 68,976,640 B over the C2 limit (n=1; [reference](docs/superpowers/research/2026-10-29-mellum-s6-d-reference.md)).
- A failed expert read no longer loses slab slots, on the early-load path, the synchronous path or a prefill
  (`--mellum-stream-read-fault`, `cb9cefd`, `be265cd`). Not changed: the GLM/DeepSeek batch path and seeding.
- The M5 prefetch-screen anomaly (1.94 stops avoided against about 3.7) was a dispatch the S6 merge left ungated:
  the fused add + norm ran past every stop (`--mellum-stream-stop-scratch`, fix `4e6450a`). After it the screen
  avoids 3.72 stops per token, prefetch off is about 0.7 ms/token faster, and prefetch on is still 1.3 ms/token
  slower (n=2 per arm, [note](docs/superpowers/research/2026-10-29-mellum-prefetch-screen-anomaly.md)).
Reopen: free about 69 MB for 2.5 GiB (+25% measured), then the iMac screen; 3.72 avoided stops are worth about
19 ms/token gross there, at or under the 20 ms/token bar before the extra reads.

## Working rules

Use Superpowers design/planning per milestone; plan later implementation only
when preceding measurements settle its choices. Keep one concern per commit,
commit a failing behavioral test separately before its fix, and keep product
changes to roughly 150 lines per commit. Use messages that name the concrete
behavior changed. Documentation-only changes do not need executable tests.

The worktree and branch remain available for subsequent phases. Record any
new measurement baseline explicitly; leave main/port integration to its owner.
