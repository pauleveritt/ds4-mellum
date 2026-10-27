# Mellum S3: first storage-observed expert read costs

Date: 2026-10-27. Status: proposed; implementation and execution await design
approval. Baseline `73c75ca`, existing `codex/mellum-ssd-replay` worktree.
This is the next CPU-only work in [S3](../../../ROADMAP.md), before GPU time
returns. [Previous evidence](../research/2026-10-27-mellum-device-observation-evidence.md)
contains four successful warm-file runs with zero timed-window device reads.

## Intent and outcome

Obtain the first native expert-read latency distributions in batches accompanied
by substantial observed storage reads. Existing full payload verification warms
files that fit this 128 GiB laptop. Repeating those files with F_NOCACHE has not
provided SSD-service costs. The next work must deliver measurements or a specific
failed cache-state method, without expanding into another general benchmark
framework. It needs no GPU, inference or representative routes.

This first study measures one declared candidate layout and one-worker demand.
It does not complete S3: original GGUF gate/up/down service, concurrency, and
representative ac-demo locality remain open. Do not extrapolate its results to
those access patterns, individual proven SSD misses, inference tok/s, NAND
service, or physical 8 GB feasibility. Ten tok/s remains a proposed checkpoint.

## Approach and alternatives

My proposed approach is a fresh private scratch file for every cell. Copy exact
Q8 experts through explicit bounded userspace reads/writes, request uncached
writes before writing any scratch bytes, and avoid reading the scratch file
until its one measured use. This separates verification-warmed sources from the
file being measured. Neither those flags nor freshness prove a cold cache; raw
read counters decide whether the attempt produced enough observed traffic.

Alternatives considered: repeating the unchanged file repeats the demonstrated
warm-cache condition; alignment of reads alone does not bypass existing cached
pages in the reference kernel; a file larger than RAM or artificial memory
pressure would substantially disturb this laptop. Defer those alternatives.
No sudo, purge, pressure generation, raw device reads, global vnode flags,
F_FULLFSYNC, reboot, APFS cloning/reflinks, downloads or new dependencies.

## Sources and limits of the cache-state method

Apple's installed SDK `usr/include/sys/fcntl.h`, lines 253–270, describes
F_NOCACHE as descriptor-scoped and distinguishes F_GLOBAL_NOCACHE. Installed
`usr/share/man/man2/fcntl.2`, lines 198–204, documents caching off/on without a
physical-read guarantee. SDK root:
`/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.
Read on 2026-10-27; these are interface facts, not evidence this method works.

Apple reference [vfs_cluster.c](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/vfs/vfs_cluster.c#L3209)
sets CL_NOCACHE for direct writes at 3209–3210, checks buffer/block alignment
at 3272–3278, waits for issued direct writes at 3579–3582, and sends an
unhandled tail through the copy path at 3590–3603. The read path checks existing
cached pages at 5576–5587. File SHA-256:
`3d210d3c725d63f11f85b774ec9ea77b80125a39333ed96c561a99157b930289`.
Accessed 2026-10-27. This supports trying aligned uncached writes to fresh files;
it does not attest the running APFS/kernel implementation or successful eviction.

## Frozen data and physical layout

Reuse the existing index, packed manifest and packed payload, their immutable
source/model digests, and the frozen uniform 512 distinct experts (seed20261027).
No diagnostic trace or held-out evaluation data is used. Original model and
packed payload stay unchanged. Validate them with the existing full verifier
before staging and after the experiment; retain its warming effects on sources.

Each expert has 6,580,224 exact Q8 bytes. In scratch, sort selected expert IDs
and assign consecutive slots with 16,384-byte alignment and stride 6,586,368.
Write all expert bytes plus 6,144 zero padding bytes in every slot. No holes,
truncation-based allocation, kernel copy shortcut or write-time compression
request. This is an aligned-start packed subset, distinct from both original
GGUF and the existing unpadded full packed layout. Record its geometry explicitly.
The measured read is only the exact 6,580,224 expert bytes, not its padding.

Require this host's page size 16,384 and device block size 4,096; refuse other
geometry for this bounded study. Use one page-aligned staging buffer of one
padded slot, set F_NOCACHE=1 on the exclusively created scratch write descriptor
before its first write, handle partial writes/EINTR, fsync and checked close.
Alignment is a write-preparation contract, not a claim of guaranteed direct reads.
Record all flag results, write counts, timestamps, digest and opened/path identity.
Any flag/write/close/identity failure ends the attempt.

Compute expected file/per-expert digests from the bytes being staged, then
verify actual scratch bytes only after its timed read and adjacent controls.
Do not read, mmap, hash or prewarm scratch before timing. After timing, compare
scratch bytes/padding and digests with independently derived source bytes in
one streaming comparison/hash pass;
any mismatch invalidates that cell and stops remaining work. Preserve immutable
source provenance; a successful write call alone is not the byte-equality gate.
Keep scratch files privately under the experiment container for this attempt;
no automatic deletion, reuse or overwrite.

## One reserved execution with an early calibration gate

Freeze all IDs, slot/read plans, cell order and declared budgets before reading
payloads or writing scratch. First fix the reviewed cancellation-evidence gap:
interruption must reap the child and retain events/snapshots/PID and an explicit
exception type, with no retry. Reuse existing bounded gates, topology mapping,
classifications, native reader and checked publication rather than duplicating
those systems. Ordinary Mellum/storage CLI behavior stays unchanged.

One worker, F_NOCACHE=1 and F_RDAHEAD=0 on the actual read descriptor, no native
prewarm. Preserve existing malloc reader buffers and pread/job timing semantics;
this design changes preparation and file geometry, not the native read loop.

1. Calibration: first 64 experts of the frozen uniform list, staged into a fresh
   private file, read once in that frozen list order.
2. If calibration is valid and `consistent-volume`, passes post-read byte
   verification and stays within all budgets, run three 512-expert repetitions.
   Every repetition has a new scratch inode, the same fixed slot assignment and
   the same frozen uniform read order. Never reuse a measured file as input.
3. Any other calibration bin or any failure ends the batch. Any later cell
   outside `consistent-volume` ends remaining work and qualifies the attempt as
   incomplete for the proposed three-repetition result. Keep all prior cells,
   unknown partial counts and reserved maxima. No substitute, repeat or fallback.

Calibration is a prospective positive control for the preparation/observation
path: a nonzero lifetime counter is not its success criterion. It may fail;
that is a retained failed method, not permission for another attempt. Successful
calibration alone does not prove later files' traffic; qualify every repetition.

Use the existing six snapshots and READY/GO/DONE/RELEASE per cell, including
one-second adjacent idle controls. No staging, verification or artifact writes
inside controls or timed brackets. Revalidate scratch/source/binary/plans before
and after cells. Require source and scratch on the same uniquely mapped APFS
volume/disk and stable topology. Retain writes as contention context; no claim
of exclusive or uncontended SSD access. Preflight reports other inference
processes and stops; never terminate someone else's process.

## Budgets, measurements and acceptance

My proposed bounds, all logical byte counts rather than physical-device caps:

| Class | Planned maximum | Hard cap |
|---|---:|---:|
| Native timed expert reads:64+3*512 jobs |10,528,358,400 B |16 GiB |
| Native staging source reads |10,528,358,400 B |16 GiB |
| Native scratch writes, including padding |10,538,188,800 B |16 GiB |
| Native explicit prewarm |0 B |0 B |
| All Python verification/metadata input reads |implementation forecast required |128 GiB |
| Retained scratch file bytes |10,538,188,800 B |16 GiB |

Before implementation execution, calculate the complete Python-read forecast:
two existing full verification passes have base 96,601,403,968 B, plus actual
scratch rereads, independent selected-source comparisons, and bounded metadata.
Count every input read by its owner; no unaccounted second scratch or source
verification pass. Refuse if the concrete forecast cannot fit 128 GiB. Native
staging counters are separate and cumulative across failed cells. Reserve the
whole attempt once before its first full payload verification; unused allowance
is not permission to retry. Require at least 32 GiB free before staging and
recheck before each new file. No measured whole-process RSS claim; staging has
one padded-slot buffer and the reader retains its existing one-expert buffer.

Retain raw per-expert enqueue/start/done times, bytes/syscalls, worker/buffer
sizes, all three repetitions, nearest-rank p50/p95/p99/max service/queue/total
latencies, native makespan and logical throughput. Retain outer counter spans,
raw device deltas, idle windows, bins, staging/fsync and verification times,
source/native/plan/file/host identities, exact commands and cumulative counts.
Never divide device bytes by the shorter native interval or turn open-loop
queued reads into inference throughput. Three repetitions support descriptive
repeatability only; no layout ranking or confidence interval follows.

Success: all four storage-observed cells pass exact-byte/identity checks and
`consistent-volume`, with budgets and artifacts independently audited. Report
native expert-read latency in storage-observed batches, with file-attribution,
cache/controller and snapshot uncertainty. Keep `physical_io_verified=false`;
no individual job or NAND attribution is established by the device-wide bin.
Failure: preserve the method, traffic/bin/failure and limitations, then stop.
No follow-on variant is authorized by this design's execution result.

Use Python 3.14+ stdlib, C99/POSIX, existing build and model-free test workflows.
Tests use tiny files and injected metadata, including actual writer/reader
children, cancellation, post-read byte mismatch, premature scratch reads,
partial I/O, flag/fsync/close failures, no-overwrite reservation, calibration
stop and cumulative budgets. Keep small separately committed behavioral RED
then GREEN changes; full serial model-free Mellum gate and independent review
precede this one bounded execution. New code should be limited to missing
scratch preparation/orchestration contracts, not a new measurement framework.

## Handoff

A written implementation plan follows approval. This work uses no GPU. ac-demo
Phase 1 schedule/input preparation can proceed independently afterward; native
routing captures need tomorrow's GPU, and offline cache replay does not.
Even successful storage results leave representative miss bytes/bursts and
S4's all-hit GPU cache path unresolved. The resulting report must name those
next measurements rather than treating tooling completion as feasibility.

## Builder check

A fresh independent macOS C99/Python storage-measurement builder read this
design and existing verifier/budget contracts on 2026-10-27 and reported
“nothing important left.” No implementation or new experiment has run.

My calls for user review: fresh aligned-write scratch files, one-worker packed
subset only, 64-expert calibration followed by three 512-expert repetitions,
16 GiB native read/write caps and 128 GiB Python cap, and stop rather than retry
when cache-state preparation fails. The user supplied the roadmap objective and
GPU availability, not those experimental choices.
