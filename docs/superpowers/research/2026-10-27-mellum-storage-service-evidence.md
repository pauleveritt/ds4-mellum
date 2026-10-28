# Mellum storage-service evidence

**Date:** 2026-10-27 (execution 2026-10-28 00:22:14–00:23:16 UTC).

**Question:** Can fresh aligned files yield storage-observed native expert-read costs on this laptop?

**Scope:** One reserved CPU-only attempt, exact Q8 bytes, one worker, one aligned packed-subset layout. No inference or GPU experiment.

## Findings

1. **Claim:** The approved fresh-file method produced four qualifying storage-observed cells.
   - **Evidence:** Calibration and all three repetitions are `consistent-volume`; device reads were 100.086–100.092% of completed logical expert bytes. Every cell passed its post-timing source comparison, padding and digest checks. Opening and closing full source verification completed; no cell is missing. No replacement or retry ran.
   - **Source:** [Retained report](/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-storage-service-run/result/artifact/report.json), this study, accessed 2026-10-27; `runs`, `closing_verification`, `counts`.
   - **Assessment:** Fact about the declared batch-level classifier and verification gates; no per-file, per-job or NAND attribution.

2. **Claim:** The three 512-expert repetitions measured median native service of 0.655–0.658 ms per 6,580,224-byte expert.
   - **Evidence:** Raw native samples and independently recomputed nearest-rank distributions agree with the report. Native logical throughput was 9.875–9.954 GB/s (decimal); this divides logical bytes by native makespan, never device bytes by that shorter interval.
   - **Source:** Same report, `runs[1:4].metrics`, and each retained `N.stdout`; artifact-only [audit](/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-storage-service-qualification/audit-output-final.json), this study, accessed 2026-10-27.
   - **Assessment:** Fact about native read service in storage-observed batches on this host. These are descriptive repetitions, with no confidence interval or layout ranking.

| Cell | Experts | Device read bytes | Service p50 / p95 / p99 / max (ms) | Native makespan (ms) | Logical GB/s |
| --- | ---: | ---: | --- | ---: | ---: |
| Calibration | 64 | 421,494,784 | 0.670 / 0.761 / 3.244 / 3.244 | 46.239 | 9.108 |
| Repetition 0 | 512 | 3,372,187,648 | 0.658 / 0.695 / 0.721 / 2.970 | 341.167 | 9.875 |
| Repetition 1 | 512 | 3,372,187,648 | 0.655 / 0.700 / 0.744 / 3.274 | 340.340 | 9.899 |
| Repetition 2 | 512 | 3,372,187,648 | 0.655 / 0.680 / 0.699 / 3.230 | 338.467 | 9.954 |

Queue and end-to-end distributions remain in the report: all jobs enqueue at
batch start. For example repetition 0 has queue p50 171.840 ms and end-to-end
p50 172.541 ms. Its service p50 0.658 ms excludes that queue. None of these
open-loop batch quantities is an inference miss stall or generation rate.

## Method, provenance and uncertainty

Actual host: Mac17,7, Apple M5 Max, **128 GiB RAM**, macOS 27.0.1,
APPLE SSD AP2048Z (2,001,111,162,880 bytes). Geometry: 16,384-byte pages and
4,096-byte media blocks. Source and all four distinct scratch inodes mapped to
the same uniquely discovered APFS volume/store/disk; closing topology matched.
This is not physical target-hardware validation for an 8 GB laptop.

Implementation freeze: `010e30c64c283f572b6ab4bdefdac84f2a10d22c`.
The metadata-only manifest binds unchanged original model SHA
`ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b`
and packed payload SHA
`bcce2e85ad8740b9102bf196e4ff2193e2a6db147af23112267bb1774d2bb5a8`.
The frozen uniform seed is 20261027. Calibration uses its first 64 experts;
three repetitions use all 512 in the same order. Slots sort global expert IDs,
with 6,586,368-byte stride: exact expert bytes plus 6,144 zero bytes.

Each private file was exclusively created, written through explicit bounded
userspace I/O with F_NOCACHE, fsynced and closed. Scratch was never read or
hashed before the timed reader. Expected hashes came from source bytes; the
one post-timing scratch pass compared independently reread source bytes and
padding. Scratch and all raw evidence remain preserved. Source hashes, staging,
verification, topology and artifact writes are outside the six counter/idle
snapshots and native R/G/D/L interval. Full source verification warms sources;
this method separates them from fresh measured scratch.

The counter span exceeds native makespan (for example repetition 0: 387.167 ms
versus 341.167 ms). Idle-before reads were 16,384 bytes for calibration and
5,111,808 bytes for repetition 1; other adjacent idle windows had zero reads.
The larger adjacent idle rate projected over each outer span stayed below the
existing 1% threshold. Device writes and all raw command intervals/plists are
retained as contention and uncertainty context. An unrelated coincident burst,
filesystem/controller effects and cached pages cannot be excluded by these
controls. Freshness and cache flags alone do not prove a miss. The classifier
and thresholds are unchanged; **`physical_io_verified=false` throughout**.

## Budgets and audit

Timed reads and native staging-source reads each completed **10,528,358,400 B**.
Native writes and retained scratch each total **10,538,188,800 B**. Every class
is below its 16 GiB cap; explicit prewarm is zero. Runner cumulative Python
preparation/verification/metadata reads were **128,674,319,931 B**, below its
136,920,461,888 B forecast and 128 GiB ceiling. Two artifact-only audits read
22,259,248 and 26,607,908 B. A conservative additional 16 MiB covers small
wrapper/report/qualification metadata reads and copies: cumulative upper bound
**128,739,964,303 B < 137,438,953,472 B**. Runtime imports, Git internals and
metadata subprocess internals remain excluded as declared; native classes are
separate logical counts, not physical-device caps.

The final independent artifact audit passed: frozen source-state canonical ID,
model/payload declarations, all wire bytes/order, every retained evidence digest,
raw plist counters and media geometry, actual native consumed identities/samples,
classifier arithmetic, native metrics, distinct scratch inodes, expected/verified
digests, counts/caps, closing verification and spent reservation agree. Neither
audit reread model, payload or scratch. The audit script/output, exact commands,
review, build recipes, gate logs and hash index are retained in
[qualification](/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-storage-service-qualification).
See the [execution ledger](2026-10-27-mellum-storage-service-ledger.md) for
review fixes, test isolation and rulings.

## Open questions and next action

This supplies the first declared candidate-layout storage-service result on
this host. It does not establish original GGUF gate/up/down costs, concurrency,
representative route locality, GPU recovery overhead, useful generation speed or
8 GB feasibility. Preserve the completed attempt; no rerun or variant follows
automatically from success.

Next decision-making measurement: frozen ac-demo Phase 1 routing captures when
GPU access is available, followed by held-out replay at roughly 3 GiB expert
cache to measure warm/post-append miss bytes and demand bursts. S4 separately
needs the GPU-readable all-hit cache path and miss recovery proof. Ten tok/s
remains a proposed checkpoint. Keep S3 open until its remaining measurements
support an explicit scheduling assessment.
