# Mellum S3a follow-up: observe block-device reads

Date: 2026-10-27. Status: approved by the user’s “proceed” after `9110b9c`; independent builder
check complete. The [implementation plan](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/plans/2026-10-27-mellum-device-observation.md)
was approved by the user’s next “proceed” and implemented. Independent review
and the full model-free gate pass after fixes. The first live attempt refused
before any native reads; an offline-tested compatibility repair is retained
without retry. See [evidence](../research/2026-10-27-mellum-device-observation-evidence.md).
Baseline: `5f59171`, branch `codex/mellum-ssd-replay`, existing worktree.
[Research](../research/2026-10-27-mellum-device-observation.md) supplies primary
sources; [S3a evidence](../research/2026-10-27-mellum-storage-evidence.md)
retains the completed warm-read batch.

## Purpose and boundary

Determine whether the existing read path produces substantial block-driver read
traffic on this M5 Max / 128 GiB laptop. Successful F_NOCACHE/F_RDAHEAD calls
and fast pread timings did not answer that question. This is an observational
diagnostic, with an acceptable negative or inconclusive result, before deciding
whether a different storage-access experiment is necessary.

Preserve exact Q8 bytes, routes, plans and the original model/packed payload.
Use Python 3.14+ standard library for orchestration and C99/POSIX for the native
read loop. No GPU, inference, dependencies, privileged access, global vnode
flags, cache purge, memory-pressure generator, payload rewrite, new expert
layout or ds4-engine/ac-demo changes. No physical 8 GB or inference-speed claim.
The old 72-run reservation is spent and its artifacts remain immutable.

My calls below narrow this phase to four runs with current buffers and reads.
An aligned bounce-buffer variant is deferred: reference kernel source shows
that aligned direct reads may still consume cached pages. No automated fallback
or escalation is included. This design does not complete S3.

## Device mapping and snapshots

Resolve both payload descriptors' filesystem device identities through diskutil
plist volume/container metadata to one physical store and whole disk. Require
both inputs to resolve to the same unique whole disk for this diagnostic.
Reject unsupported/multiple-store, missing, changed or ambiguous topology.
Retain volume/container/store UUIDs, disk identifiers, size and model. Never
select a driver by largest/nonzero counters or hard-code this boot's entry IDs.

Use `ioreg -a -l -r -c IOBlockStorageDriver` outside measurement to associate
that whole disk's IOMedia BSD Name/Whole/size with exactly one driver parent.
Retain the raw plist and selected registry IDs. Read root Statistics for that
same driver at every snapshot; require unsigned monotonic byte and operation
counters, unchanged driver/media identity, and no read error/retry increment.
Recheck topology after the batch. Reinstantiation, counter regression or missing
fields invalidate observation rather than producing wrapped deltas.

Run only fixed Apple executable paths/arguments, with bounded output (32 MiB)
and a 5-second timeout per metadata command; parse plists without external
libraries. Snapshot collection records monotonic timestamps immediately before
and after each command, raw output/hash, selected counters and command status.
Counter samples are non-atomic observations within those timestamp intervals.
Retain writes/errors/retries as context, not as read bytes. No iostat-derived
read attribution and no assumed counter time units.

## Native phase gates

Add an optional observation protocol, leaving ordinary CLI behavior unchanged.
Use dedicated inherited control/event pipes; native stdout remains the final
JSON report. Validate inputs, open payload, apply controls, allocate/touch the
worker buffer, and perform any explicit prewarm before emitting READY. Native
then blocks before starting any timed work. Python records the opening counters
and sends GO only after successful collection. Native records its existing
monotonic timing, reads all jobs, joins workers and emits DONE, then blocks for
RELEASE before checked close/report publication. Python collects closing
counters while native is quiescent, then releases it.

No payload hashing, prewarm, log/report disk writes or Python input verification
occurs between the two counter snapshots. Keep results/snapshots in bounded RAM
until closing observation and adjacent idle control finish. Native timing stays
inside the device observation interval; report both intervals and snapshot
uncertainty, never divide device bytes by the shorter native time as if the
intervals matched. Gates have a 30-second deadline each; I/O process completion
has a 120-second deadline. Protocol errors, EOF, missing DONE or failure kill
only this owned child, reap it, and retain a failure artifact. A pre-existing
inference process is reported at read-only preflight, never terminated.

This protocol is observation-only: it does not alter native read ranges,
worker scheduling or cache policy. Tests must prove no read before GO and no
publication before RELEASE, including failures and parent-pipe closure.

## One new diagnostic reservation

Create a new content-addressed manifest/reservation in a distinct artifact
container; never invoke the old batch's run operation. Reuse only the frozen
uniform 512 distinct expert IDs and checked original/packed range descriptions.
Bind the prior manifests/digests and actual newly consumed native wire plans;
retain existing opened-fd/path identity and exact-byte checks. No diagnostic
trace sampling or held-out ac-demo data is needed.

Freeze this order, one worker and one invocation per cell:

1. Original GGUF, buffered with explicit prewarm.
2. Original GGUF, avoidance requested, no explicit prewarm.
3. Packed payload, buffered with explicit prewarm.
4. Packed payload, avoidance requested, no explicit prewarm.

Each run has 512 jobs, 3,369,074,688 logical bytes. Total timed logical bytes:
13,476,298,752, cap **16 GiB**. Explicit native prewarm totals 6,738,149,376,
cap **8 GiB**. Preparation/verification, counted separately by the existing
Python accounting, has a **128 GiB** logical-read cap; refuse any stage whose
planned reads would exceed it. Do not create a second payload. These counts
are logical file reads, not physical-device budgets. Controls naturally warm
the following avoidance arm; this deliberately tests the current hot-file path,
not a cold-cache condition or an unbiased layout-speed comparison.

Reserve the full four-run budget before first GO, using no-overwrite publication
and the existing incomplete/failure artifact conventions. No retries, substitute
runs, repeated batches or automatic wait-until-quiet loop. Any completed cell,
failure and unused reserved allowance remain recorded. If preflight cannot map
the device or active inference is observed, issue no GO and retain the refusal;
resumption requires a fresh explicit execution decision, not an automatic retry.

Before/after source verification and explicit prewarm are outside observed
intervals; disclose their warming effects. Builder must calculate the exact
preparation-read forecast from the implemented checks before reservation; the
128 GiB cap cannot silently increase. Track actual bytes on failure as well.

## Idle controls and interpretation

For each cell, after READY/prewarm collect a one-second quiescent idle window,
then fresh opening counters and GO. After DONE/closing counters, collect a
one-second quiescent idle window before RELEASE. The child holds its prepared
resources but performs no reads or writes during these controls. Include the
snapshot-command overhead in each control's reported interval and uncertainty.
No experiment artifact writes during controls. Keep all raw samples; do not
subtract an estimated background rate and relabel the remainder as file I/O.

Let L be successfully completed logical bytes and D the raw observed device
read-byte delta enclosing the run. Let T be the conservative outer observation
span (opening command start through closing command end). For each idle window,
compute read rate using the conservative inner duration between the first
command's end and second command's start. Nonpositive inner duration invalidates
the control. B is the larger idle rate times T. My diagnostic thresholds:

- Observation invalid: mapping/counter/protocol/identity failure, incomplete
  jobs, new read errors/retries, or unusable controls. Keep timing as failed or
  unqualified evidence, never a valid shorter benchmark.
- Background flagged/inconclusive: B exceeds 1% of L, or device bytes exceed
  125% of L. Keep raw counters; neither case proves background is the cause.
- Little observed device traffic: otherwise D is at most 5% of L.
- Device traffic consistent with logical volume: otherwise D is between 90%
  and 125% of L, inclusive.
- Everything else: mixed/inconclusive.

Thresholds are conservative agent-selected diagnostic bins, not statistical
confidence intervals or a proof of file attribution. A background burst during
the run can escape both controls. Even the consistent-volume bin does not
certify SSD bandwidth, NAND reads, a per-expert stall, or file-specific service.
Keep `physical_io_verified=false`; add explicit device-observation status and
raw evidence rather than repurposing that legacy boolean. Do not average bins
across cells or compare layout speed from one repetition.

A little-traffic result settles that these invocations cannot measure storage
service cost. A consistent-volume result supports designing a later controlled
SSD-service study. Mixed/invalid results identify the missing control. All
outcomes end this budget and return to written assessment; none unlocks an
aligned variant, cold-cache test, extra repetition or streamed inference.

## Evidence, validation and handoff

Retain frozen manifest, source/native/build/plan/file identities, host/topology,
controls and gates, native samples, stdout/stderr/exit, raw snapshots, all time
intervals, bytes and classification rationale in a newly published artifact.
Retain a dated report/ledger linking it; never alter the old S3a report.

Use tiny fabricated files and injected plist/counter/protocol events for
model-free tests: unique/ambiguous/multi-store mapping, resets/identity changes,
bounded malformed output/timeouts, contamination/bin boundaries, failed gates,
partial reads and no-overwrite reservations/publication. Full serial
`make -f Makefile test-mellum` must pass; real disk/GPU access stays outside it.
Use small commits, separately committed behavioral RED then GREEN, roughly
150 product lines per implementation commit, and independent review before
real diagnostic execution. Documentation-only approval work needs diff/link
checks, not rerunning the CPU suite.

After design approval, write an implementation plan with explicit execution
and stop points. This approval would cover the declared four-run CPU-only
reservation after tests/review, but never an expanded experiment. Generated
ac-demo S3b collection and S4/S5 remain separate work.

## Builder check

A fresh independent macOS C99/Python storage-measurement builder read this
design and reported “nothing important left” on 2026-10-27. No human-only
question remains. Agent-selected calls for review: observe existing reads before changing
alignment; four one-worker cells; 16/8/128 GiB read caps; native pipe gates;
one-second adjacent controls; diagnostic thresholds; no automatic repetitions.
