# Device-observation implementation and refused first attempt

Date: 2026-10-27. Implements the approved
[design](../specs/2026-10-27-mellum-device-observation-design.md) and
[plan](../plans/2026-10-27-mellum-device-observation.md).

## Outcome

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

## Next step

The repaired tooling is ready for a newly authorized CPU-only attempt using the
same four-cell design and explicit budgets. Until it runs, device-observed read
service remains open. Generated ac-demo S3b collection, streamed inference and
physical 8 GB validation remain later milestones. Ten tok/s is still only a
proposed checkpoint.
