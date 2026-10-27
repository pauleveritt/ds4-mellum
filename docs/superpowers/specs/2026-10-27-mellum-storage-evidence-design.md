# Mellum S3a: expert ranges, storage reads and workload source freeze

Date: 2026-10-27. Status: proposed; independent builder check complete,
awaiting user review before implementation planning.
Baseline: `a44532d`, `codex/mellum-ssd-replay`. Implements the first CPU-only
slice of S3 in the [roadmap](../../../ROADMAP.md), following the
[owned-recorder proof](../research/2026-10-27-mellum-owned-live-proof.md).

## Intent and limits

Measure the storage work implied by unchanged Mellum Q8 experts before building
a streaming scheduler. Produce a checked expert-range index, an exact-byte
contiguous comparison file, reproducible read plans and native read measurements.
Separately freeze the repository inputs for the user's chosen coding workload:
`../ac-demo` and its roadmap. Keep these two evidence tracks distinct.

This slice needs no GPU, inference, dependency installation or server. It does
not implement streaming, prediction, new routing, generated capture, a general
agent harness, or `../ds4-engine` integration. It does not complete S3, establish
physical 8 GB suitability, or convert read bandwidth into inference tok/s.
No performance threshold is assumed; 10 tok/s remains a proposal.

Use Python 3.14+ standard library for preparation, identities and orchestration;
C99/POSIX for timed reads and worker scheduling. No C++ or Metal dependency.
Stay on this branch/worktree; no push, merge or port integration. Use focused
commits, separate failing behavioral tests before fixes, and roughly 150 new
product lines per implementation commit.

## Workload source: ac-demo

The user chose `../ac-demo` and its roadmap. The actual checkout is
`/Users/pauleveritt/projects/pauleveritt/ac-demo`, clean at
`bfb6e1f6e2ef141a1ac88f5c25466d3a076caab3` when inspected. It has nine tracked
files and no `app.py` or `models.py`; templates/tests contain only `.gitkeep`.

Make a content-addressed source bundle from exactly that commit, with original
path, commit, tracked-file list, exact file bytes and individual SHA-256 values.
Include AGENTS.md, all three specs, pyproject.toml, uv.lock and placeholders.
Do not inspect `.venv`, copy credentials/untracked files, change the checkout,
start a server or claim to implement a phase. Fail on a source revision/content
change during bundling; never silently update a frozen bundle.

My proposed first evaluation task is: **implement Phase 1 only, including its
home-page tests, without starting a server**. The input bundle preserves the
full roadmap, but the request explicitly stops at Phase 1, as its roadmap
requires. Preserve its single-quote and Request-import instructions. Do not
change its Python constraint as part of this investigation; the 3.14+ constraint
here applies to these measurement tools, and its stack document already names
Python 3.14. Future execution of its tests must use its required
`uv run --frozen python -m pytest tests` in an isolated experiment checkout.

My call: reserve **all ac-demo conversations/phases as evaluation**, with the
existing Mellum fixtures remaining tuning/diagnostic data. Do not open ac-demo
routing results while choosing policies or read-plan sampling. Initial policy
remains the existing global LRU; no policy tuning is part of S3a.

This is a workload-*source* freeze, not yet a complete collection manifest.
Before GPU collection, a separate S3b design must freeze exact rendered
messages/template and token identities, generation settings/stops, operation
schedule, append transcripts, case counts and warm/recovery windows. The current
publisher's teacher-forced, final-single-append schedule cannot stand in for
generated multi-turn work. Do not pad this small project with repeated text to
claim medium/8K coverage. Later phases need actual prior-phase checkpoints or
an explicitly labeled reference fixture; the empty starter cannot supply them.

## Checked Q8 range index

Read the GGUF header/tensor directory without initializing an engine, GPU or
whole-model resident mapping. A bounded streaming reader may reuse existing
GGUF parsing primitives, but must reject malformed lengths/counts, duplicates,
overflow, out-of-file/overlapping spans and unsupported layouts. Do not inherit
the mixed-DeepSeek tool's unchecked allocation limits as an input contract.
Support this monolithic GGUF v3 Mellum Q8 variant first; reject shards/other
quantizations with a clear diagnostic. Choose documented metadata/count caps
that bound parser memory independently of file size.

Require every `blk.{0..27}.ffn_{gate,up,down}_exps.weight`, exactly once, Q8_0:
gate/up dimensions `[2304,896,64]`, down `[896,2304,64]`. Derive row bytes using
32 elements / 34 bytes per Q8_0 block, preserving row boundaries. Each part is
2,193,408 bytes per expert; all three total 6,580,224 bytes. Validate against
the file directory rather than invent offsets from these constants. Retain
all directory spans needed to prove that expert ranges do not overlap another
tensor. Unsupported non-expert types may be rejected rather than guessed.

Output a versioned JSON index of 1,792 `(layer,expert)` entries, each containing
the three named absolute offset/length ranges in gate/up/down order. Record
geometry, GGUF directory/alignment identity, full model SHA-256, size/stat
identity, tool/source identities and exact source paths. Full hashing is an
untimed preparation read and may warm the OS cache; disclose it. Check opened
file identity before/after inspection and use atomic no-overwrite publication.

Synthetic tests must cover nonzero data offsets/padding, role order, last expert,
duplicate/missing roles, wrong shapes/type, truncation, overlap and arithmetic
overflow. A real-file index is an explicit opt-in CPU model inspection, separate
from model-free tests. No benchmark result follows from successful inspection.

## Exact-byte contiguous comparison

Create an optional experiment directory containing a raw payload and sidecar
index. Logical order is ascending `(layer,expert)`, each payload gate then up
then down, with no per-expert padding or Q8 transformation. The first payload
starts at offset zero. This tests a simple contiguous candidate, not a proposed
final GPU slot layout. The original GGUF remains read-only.

Copy with bounded staging (at most one expert plus a small streaming hash
buffer). Hash and compare every expert's source concatenation with the packed
payload in an untimed full verification pass. Retain both identities and total
bytes. Detect short reads/writes, interruption, changed source or output, and
insufficient space. Private incomplete files never become valid artifacts.
Publish the entire directory atomically without replacement after checked
flush/close; do not leave a valid manifest beside an incomplete payload.

My operational budget: permit one full expert payload (about 11 GiB) plus 2 GiB
free margin, refusing before copying if unavailable. No global cache purge,
memory-pressure generator, source-file rewrite or privileged operation.

## Frozen read plans

Freeze these two **tuning/calibration**, non-evaluation plans before measuring:

1. Uniform layout calibration: 512 distinct expert identities sampled without
   replacement from all 1,792, seed `20261027`; store the resulting ordered list,
   rather than relying on future PRNG implementation behavior.
2. Diagnostic demand calibration: replay the complete existing qualified tiled
   trace at 3 GiB with the existing LRU, then take the first 512 demand-load
   identities in its decode segment, preserving repeats. Start LRU cold at the
   trace beginning; do not start a new cache at decode. Obtain misses from
   `required - hits` using the real `LRU.serve` result, sorted within a group.
   Refuse a short/inconsistent input rather than repeat/pad it. Record the trace
   digest, replay source digest, capacity/effective slots and selection rule.

My initial diagnostic input is the post-repair source-checked tiled trace
documented in the owned proof (`after-fix.trace.jsonl`, SHA-256
`75318d1374f8780beb3fb7c5dc880b70afb32786bd976c170d9b7aac0203b2de`).
Require the input to match that digest; do not silently select a newer trace.

Each plan expands the same logical expert jobs to three original-GGUF ranges
or one contiguous range. Preserve job order/identity; concurrency may change
completion order, which is recorded. These are open-loop storage jobs, without
inference, cache deadlines or dependency stalls. They are not streamed tokens.
Report role read sizes, offset alignment and syscall/read amplification.

The native input is a versioned, bounded binary range plan, not ad-hoc JSON
parsing in C. Specify magic/version, little-endian fixed-width fields, exact
EOF, logical job IDs, range counts and offsets/lengths before implementation.
Reject invalid counts/offsets/lengths, overflow and mismatched file size before
issuing reads. Python retains the human-readable plan and all digests.

## Native storage experiment

Use `pread` with one reusable complete-expert buffer per worker, concurrency
1, 2 and 4. Each worker handles all parts of one expert sequentially; do not
share a file position or overwrite an in-flight buffer. Loop on partial reads
and EINTR, refuse EOF, and stop the entire run on an I/O error. Timed code does
not initialize the model, hash full payloads, print or flush per request.
Keep complete bounded samples in memory and publish only after worker join and
checked close. Never report a failed/truncated run as a shorter success.

Run both layouts, both calibration plans, both cache modes and all three worker
counts, three measured repetitions each: 72 runs of 512 expert jobs. Alternate
layout order per repetition and freeze the complete run order in the batch
manifest before execution. Timed logical reads remain below 256 GiB total;
preparation, verification and prewarming bytes are counted separately.

Cache modes are **buffered, explicitly prewarmed** and **cache avoidance
requested**. On macOS the latter requires successful `F_NOCACHE=1` and
`F_RDAHEAD=0` on the actual read descriptor. No silent fallback. Other POSIX
platforms may support buffered mode first and reject the avoidance mode.
Do not call the second mode physically cold or proven SSD traffic. Record
control results, prewarming method, preparation order and any available device
I/O observations; report inability to verify physical reads as a limit.

Record host/OS, volume/device/filesystem identities, source and packed file
identities, executable/source/plan digests, worker/buffer memory, planned versus
completed jobs, logical bytes and actual syscall bytes/count. Record each job's
enqueue/start/completion times with a monotonic clock. Derive queue wait,
service and end-to-end latency distributions (nearest-rank p50/p95/p99/max),
wall-clock throughput and total makespan, with explicit units. Include all raw
samples and all repetitions; do not replace them with only an aggregate mean.
Check file identity again after reads. Keep preparation/verification untimed.

CPU tests use tiny fabricated files/plans and verify actual bytes, offsets,
compound-job timing aggregation, deterministic job accounting, EOF/errors,
invalid plans and no-overwrite publication. Inject partial reads/EINTR/errors
only at a test boundary; do not write tests that merely mirror the parser.
The model-free gate never inspects the real GGUF or performs a timed SSD run.

## Acceptance and next handoff

S3a is complete when model-free tests and the full Mellum CPU gate pass, the
native tool builds without GPU objects, an independent review has no unresolved
Important findings, and optional real-file preparation/measurements have an
explicit evidence record (or are clearly reported as not run). Preserve small
RED/GREEN commits and the existing checked publisher contract.

Do not mark S3 complete from S3a. S3b still needs the approved ac-demo collection
schedule, generated/teacher-forced cohorts, prefix-preserving window analysis
and actual GPU collection. A feasibility decision additionally needs agreed
performance/host-memory objectives and stronger physical-storage evidence;
native streamed inference remains S4/S5. Python owns workflow policy and native
code owns timed/inference-critical scheduling.

## Builder check

A fresh independent C99/Python storage-measurement builder reviewed this
design, the evaluation protocol and existing LRU on 2026-10-27 and reported
“nothing important left.” No additional user-only question was identified.
The proposed case counts, sampling seed, read budget, contiguous file layout,
Phase 1 first task and whole-ac-demo evaluation reservation are this agent's
calls for review; the user supplied the repository and roadmap, not those details.
