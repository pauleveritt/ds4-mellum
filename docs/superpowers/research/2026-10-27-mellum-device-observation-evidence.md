# Device-observation implementation and execution evidence

Date: 2026-10-27. Implements the approved
[design](../specs/2026-10-27-mellum-device-observation-design.md) and
[plan](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/plans/2026-10-27-mellum-device-observation.md).

## First-attempt outcome

CPU tooling is implemented. The one real attempt stopped during device discovery,
before payload verification, reservation, prewarming or native timed work. There
are no new per-device experiment deltas, SSD bandwidth measurements, miss stalls,
inference results or physical 8 GB conclusions. The earlier
[S3a warm-read evidence](2026-10-27-mellum-storage-evidence.md) remains unchanged.

The refusal exposed a compatibility defect: diskutil info uses `WholeDisk`,
whereas the IOKit media property is `Whole`. The code had used `Whole` for both.
An observed failing regression (`e996153`) and fix (`e6e3066`) repair that
schema distinction using saved metadata only. The offline reconstruction now
maps the captured APFS metadata to disk0/driver 4294971011/media 4294971012.
This validates the schema repair, not a new live discovery or device-read test.
No automatic retry or replacement experiment was performed. The old manifest
binds the pre-repair source revision and must not be reused with repaired code.

## Implementation and verification

Native optional READY/GO/DONE/RELEASE gates preserve the ordinary read interface
and existing expert reads. Python performs unique topology mapping, bounded
metadata capture/decoding, exact interval/bin arithmetic, one-shot reservation,
strict before/after input checking and failure-only publication. No engine,
quantization, routing, GPU, ac-demo or ds4-engine behavior changes.

The final serial `make -f Makefile test-mellum` passes after the compatibility
repair; `git diff --check` passes. Observation coverage comprises 33 Python
function tests plus native gate/lifecycle checks, including real tiny native
processes. Existing Mellum CPU coverage also passes. No model or device tools
are invoked by these tests. The native link uses only the five storage/gate
C files and pthreads, with no GPU/core objects.

The independent reviewer inspected `72da50b..15c7952`: no Critical findings,
six Important findings and one Minor. All Important findings received observed
RED/GREEN regressions followed by the full model-free gate:

- Reserve before any payload verification, preserving a failed attempt's allowance.
- Recheck all metadata/source/wire bindings after each full payload verification.
- Detect actual repository compute executables from their Makefile link rules.
- Normalize malformed XML/schema/discovery errors into evidence-carrying failures.
- Kill/reap metadata helpers on cancellation and unexpected failure.
- Account for retained containers and transient capture/decoding memory, bound
  XML structure, and release the previous cell's raw data before the next capture.

There was no second independent review. The subsequent WholeDisk repair has its
own failing regression, saved-metadata check and fresh full-suite verification.

Deferred Minor: native stderr emitted while a snapshot fails can remain undrained
at cleanup; the refusal and already captured evidence survive, but late native
diagnostics can be absent. The older S3a detailed-worker-error minor also remains
outside this change.

## Real attempt and accounting

Execution source: `17ee04b`, before the WholeDisk repair. The unchanged original
model and packed payload declarations match the approved S3a size/SHA identities;
this attempt did not perform their full byte verification. Preparation succeeded,
and observe-run exited 1 during the five-command discovery sequence. Saved raw
plists include both volume info commands, APFS container list, whole-disk info
and IOKit driver tree. The preflight found no matching active compute process.

| Class | Planned allowance | Actual this attempt |
|---|---:|---:|
| Native timed logical reads | 13,476,298,752 B; 16 GiB cap | 0 B |
| Native prewarming | 6,738,149,376 B; 8 GiB cap | 0 B |
| CLI Python preparation | included in cumulative cap | 18,094,888 B |
| CLI Python run checks | included in cumulative cap | 19,494,975 B |
| Cumulative instrumented CLI Python reads | 128 GiB cap | 37,589,863 B |
| Model/packed byte-verification passes | two planned | zero |

The frozen forecast was 105,325,556,288 B: payload base 96,601,403,968 B,
directory/prefix allowance 134,217,728 B, metadata/source allowance 8,589,934,592 B.
The captured host reports Mac17,7, Apple M5 Max, 137,438,953,472 B RAM.
Whole disk is APPLE SSD AP2048Z, 2,001,111,162,880 B, APFS data volume disk3s5.
No execution-reservation.json was created and no cell started. Despite unused
native allowance, this attempt ended; resumption needs a fresh explicit execution
decision and source-bound manifest. CLI accounting excludes Git internals, runtime
imports, metadata subprocess internals and subsequent artifact-only inspection.
There were no uninstrumented expert payload reads in that inspection.

## Retained artifacts

Root:
`/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4`

| Artifact below root | SHA-256 |
|---|---|
| `mellum-device-observation-batch/artifact/manifest.json` | `4c790b9ef4b3e007ae0a60f2cbdcbf80fd89f23db7948b96c203fb049d5d2b12` |
| `mellum-device-observation-run.failure/artifact/failure.json` | `8708959ebcec0f47cbda13b27a9b15099b73e6af56c31e301ad7cebc8619fbd7` |

`mellum-device-observation-execution` retains prepare/run logs, exact command
arguments/exits, baseline/final CPU gates, the review diff/summary, execution
ledger and offline topology check. The failure and original batch artifacts
remain immutable; the successful-run artifact does not exist.

## Next step after the first attempt

The repaired tooling is ready for a newly authorized CPU-only attempt using the
same four-cell design and explicit budgets. Until it runs, device-observed read
service remains open. Generated ac-demo S3b collection, streamed inference and
physical 8 GB validation remain later milestones. Ten tok/s is still only a
proposed checkpoint.

## 2026-10-27 — Authorized second attempt completed

**Question:** Do the four warm-file native read cells produce observable reads
at the mapped block driver? **Scope:** CPU-only diagnostic on the development
laptop, using the same frozen 512 distinct tuning experts, one worker and one
repetition per cell. After the user authorized proceeding, a fresh manifest
bound repaired source `3edad694a4f7985e161c538191c7cab932f1ac92`. The first
attempt and its artifacts remain unchanged. This was a separately authorized
attempt, with no automatic retry or replacement cell.

**Finding (fact):** Preparation and execution both exited 0. All four cells
completed 3,369,074,688 logical bytes each. Every timed bracket and all eight
adjacent idle windows recorded zero block-driver read-byte deltas. All four
classifications are `little-device-traffic`; the conservative background
estimate is exactly 0/1 bytes in each cell. Native cache controls report
F_NOCACHE enabled and F_RDAHEAD disabled for both `avoid` cells.

| Cell | Layout | Mode | Native makespan | Outer counter interval | Device read bytes |
|---|---|---|---:|---:|---:|
| 0 | original | buffered | 109.635 ms | 161,854,000 ns | 0 |
| 1 | original | avoid | 117.202 ms | 169,496,000 ns | 0 |
| 2 | packed | buffered | 89.268 ms | 138,552,000 ns | 0 |
| 3 | packed | avoid | 90.089 ms | 136,536,000 ns | 0 |

Adjacent idle inner intervals, before/after respectively, were
1,006,935,000/1,007,337,000 ns (cell 0),
1,007,231,000/1,004,227,000 ns (cell 1),
1,002,422,000/1,005,668,000 ns (cell 2), and
1,007,171,000/1,006,136,000 ns (cell 3).
The native interval is contained within each counter bracket; these outer
intervals include snapshot uncertainty and are not native service times.

**Source:** Local native reports, raw IOKit plists and report JSON under
`mellum-device-observation-02-run/artifact`, produced by this investigation and
audited 2026-10-27. Exact commands, exits and the artifact-only audit are in
`mellum-device-observation-02-execution` below the retained root above. Commands:

```sh
python3 -I tools/mellum_storage.py observe-prepare --prior-batch "$ROOT/mellum-storage-s3a-batch/artifact" --binary "$WORKTREE/tools/mellum_read_bench" --output "$ROOT/mellum-device-observation-02-batch"
python3 -I tools/mellum_storage.py observe-run --batch "$ROOT/mellum-device-observation-02-batch/artifact" --output "$ROOT/mellum-device-observation-02-run"
```

Here ROOT is the retained visualization root above and WORKTREE is
`/Users/pauleveritt/.codex/worktrees/mellum-ssd-replay/ds4-mellum`.
The execution log records 38.524 seconds wall time. Source, native binary,
plans, original model and packed payload identities were checked before/after
execution. Successful publication follows both full model/packed byte-verifier
calls. The later audit checked source/metadata hashes, consumed-wire identities,
all 512 native samples per cell, stable topology and every raw snapshot's
SHA/plist counters, then rederived classifications and byte arithmetic. It
read saved artifacts only and did not rehash either large payload.

| Class | Actual second attempt | Cap |
|---|---:|---:|
| Native timed logical reads | 13,476,298,752 B | 16 GiB |
| Native prewarming | 6,738,149,376 B | 8 GiB |
| CLI Python preparation | 18,095,197 B | cumulative below |
| CLI Python execution | 96,902,362,193 B | cumulative below |
| Cumulative instrumented CLI Python reads | 96,920,457,390 B | 128 GiB |
| Full model/packed byte-verification passes | 2 | 2 |

Cumulative Python reads also fit the frozen 105,325,556,288 B forecast. The
same accounting exclusions described for the first attempt apply. The new
reservation is spent, with `retry_permitted=false`.

The host is Mac17,7, Apple M5 Max, 128 GiB RAM, macOS 27.0.1 arm64, with
APPLE SSD AP2048Z (2,001,111,162,880 B). Before/after topology agrees:
volume disk3s5 (`BF58637C-3FAD-437B-8E01-90E30EC8D019`), container disk3
(`8C90DA7F-D18C-44E8-86D1-0F854C355C46`), physical store disk0s2
(`7E8A1F90-90CD-47F3-8CF8-E3D767907249`), whole disk disk0,
driver ID 4294971011, media ID 4294971012.

| Artifact below root | SHA-256 |
|---|---|
| `mellum-device-observation-02-batch/artifact/manifest.json` | `057be462f0f3a13d4d2314708c97c384c0f9f5a99413f490312c494dbdcc764b` |
| `mellum-device-observation-02-batch/execution-reservation.json` | `df799b50aa4d3c94f135de03d84d06cc175d6cc16a9bde0016695b0152908bf9` |
| `mellum-device-observation-02-run/artifact/report.json` | `f2d7cd6205f7d6727029a101fb6ea28c53389c6cbe138325b0577f018d7429b9` |
| `mellum-device-observation-02-execution/artifact-hashes.json` | `1d99c041feaa2e4677eb479719f7a57f9248c26f2a72e0fec0ee3d4443eb05a8` |

The 24-file hash index includes both immutable artifact containers, reservation,
commands/logs/exits, `audit_saved.py` and its passing `audit-saved.log`.

**Assessment (inference):** These invocations cannot supply SSD-service costs.
The cache flags did not produce observed physical-device reads in this warm-file
setup. Device-wide counters do not prove file attribution or NAND service;
`physical_io_verified=false` remains unchanged. Single repetitions do not
establish layout superiority. No miss-stall, inference tok/s or physical 8 GB
feasibility result follows from these timings.

**Recommendation:** Design a separate controlled physical-read experiment before
using read latency in streaming feasibility estimates. Confidence is high that
this diagnostic leaves SSD miss cost unresolved. No extra cells, cache purges,
pressure, privileged operations, aligned/cold-read variants or inference were
started. S3b representative collection, GPU execution and ds4-engine integration
remain later work; 10 tok/s remains only a proposed checkpoint.
