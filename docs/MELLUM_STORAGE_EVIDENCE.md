# Mellum storage evidence tools

These Python 3.14+ standard-library tools prepare CPU-only evidence. They do
not initialize the model or GPU, generate conversations, or implement streaming.

## Freeze the evaluation source

Run from this worktree with a fresh output container whose parent exists:

```sh
python3 -I tools/mellum_workload_bundle.py \
  --repo /Users/pauleveritt/projects/pauleveritt/ac-demo \
  --revision bfb6e1f6e2ef141a1ac88f5c25466d3a076caab3 \
  --output /absolute/existing-parent/ac-demo-source
```

The result is `ac-demo-source/artifact`, containing `manifest.json` and
`source/` with exact committed regular-file bytes and executable modes. The
manifest records the commit, tree, per-file hashes, tool/source identity and
machine-independent content ID. The checkout must remain at the exact clean
tracked revision before and after copying. Untracked and ignored files are
excluded; `.venv` is never enumerated or read.

The entire repository is reserved for evaluation. The proposed first task is
Phase 1 only, including home-page tests, without starting a server. The bundle
preserves AGENTS.md, roadmap and stack instructions. `collection_complete=false`
means this is source material, not rendered/tokenized prompts, generated
conversations or a completed phase. Those need the later S3b design. No ac-demo
tests run during freezing because no phase is implemented.

Publication reserves a new container exclusively. Only its complete `artifact`
child is valid; the container can exist earlier. Existing files, directories
and symlinks are refused. A failed build exposes no complete artifact. An
interrupted reservation must not be reused. Builders check writes/flush/close
before promotion; this is complete-directory visibility, not power-loss durability.

Run synthetic tests without inspecting the real repository or model:

```sh
make -f Makefile test-mellum-workload-source
```

## Prepare exact Q8 reads

All outputs below are fresh exclusive containers. Consumers use their complete
`artifact` child. Set each uppercase path to the intended local file/directory;
these commands are explicit opt-in work outside every model-free test gate.

```sh
make -f Makefile tools/mellum_read_bench
python3 -I tools/mellum_storage.py index --model "$MODEL" --output "$INDEX_OUTPUT"
python3 -I tools/mellum_storage.py pack --index "$INDEX_OUTPUT/artifact/index.json" --output "$PACK_OUTPUT"
python3 -I tools/mellum_storage.py prepare \
  --index "$INDEX_OUTPUT/artifact/index.json" --packed "$PACK_OUTPUT/artifact" \
  --trace "$DIAGNOSTIC_TRACE" --binary "$PWD/tools/mellum_read_bench" --output "$BATCH_OUTPUT"
python3 -I tools/mellum_storage.py run --batch "$BATCH_OUTPUT/artifact" --output "$RUN_OUTPUT"
```

Indexing supports monolithic GGUF v3 Mellum with F32, F16 and Q8_0 tensors,
bounded metadata/directory access (32 MiB), and the strict 28×64 expert geometry.
The 1,792 entries retain actual gate/up/down offsets. Source verification
recomputes the actual directory; a model hash alone cannot authorize supplied
offsets. Full hashing is untimed and may warm the OS cache.

Packing copies unchanged Q8 bytes in ascending layer/expert order, with
contiguous gate/up/down parts and no padding. Each expert is 6,580,224 bytes;
the payload is 11,791,761,408 bytes. Preparation refuses without that space plus
2 GiB free margin. Copying and independent full rereading verify actual source
and payload bytes; manifest digests alone are insufficient. Python staging is
bounded to small read chunks; native workers each hold one complete expert.

## Fixed storage measurement

The two calibration plans use tuning data: 512 uniformly sampled distinct
experts (seed 20261027), and the first 512 decode demand misses from the approved
post-repair diagnostic trace. Replay starts cold at the full trace beginning,
uses unchanged LRU at 3 GiB (489 complete experts), and preserves repeats. No
ac-demo routing data is used. The trace must match the approved SHA-256.

Each plan expands into either three original ranges or one contiguous range.
The native plan has a 64-byte little-endian `MLREAD01` header and 32-byte range
records; both Python and C validate bounds and complete-job accounting. The C99
executable links only its five storage/gate source files with pthreads, without engine
or GPU objects. It uses pread loops, a ready-worker gate and one reusable buffer
per worker. Samples include every job's enqueue/start/completion timestamps.

The manifest freezes 72 invocations: plans uniform/diagnostic, modes
buffered/avoid, workers 1/2/4, repetitions 0/1/2, with layout order reversed in
odd repetitions. Total timed logical bytes are 242,573,377,536 (below 256 GiB).
The execution reservation charges the full planned budget once and prohibits
retrying that batch, including after failure. Failures retain raw diagnostics
under a separate `.failure/artifact`, never a successful partial report.

Buffered runs explicitly prewarm all planned jobs outside timing. Avoidance on
macOS requires successful F_NOCACHE=1 and F_RDAHEAD=0 on the actual read fd.
This requests cache avoidance; it does not prove physical SSD reads. No global
cache purge, memory-pressure generator or privileged operation is used.
`physical_io_verified=false` is retained throughout.

`report.json` retains the frozen protocol, identities, host/volume metadata,
every repetition's metrics and references/hashes to complete raw stdout/stderr
and per-job metrics. Each JSON input/report is bounded to 8 MiB and rejects
nonfinite numbers and duplicate keys. Separate raw files preserve bounded
loading. Distributions use nearest-rank p50/p95/p99/max in milliseconds;
throughput uses decimal bytes/second. Returned-byte and syscall amplification
are logical read measures, not physical-device amplification. Worker-buffer
memory excludes runtime, thread stacks and OS cache memory.

Untimed Python read accounting includes instrumented model/payload, metadata,
source and report reads, and excludes Git internals/runtime imports. Native
prewarming bytes and timed bytes are reported separately. Source identities and
binary hashes are frozen; compiler provenance identifies the fixed Makefile
recipe and the actual executor build log, without claiming signed attestation.
This experiment reports storage jobs, not streamed tokens or inference tok/s.
S3 remains open for generated ac-demo collection, window/locality analysis and
physical-storage/8 GB validation.

## Device-observed follow-up

The [approved follow-up](superpowers/specs/2026-10-27-mellum-device-observation-design.md)
uses a distinct uniform-only manifest and reservation. It does not rerun or
modify the spent S3a batch. No model copy or GPU is needed.

```sh
python3 -I tools/mellum_storage.py observe-prepare \
  --prior-batch /absolute/prior-batch/artifact \
  --binary /absolute/checkout/tools/mellum_read_bench \
  --output /absolute/new-observation-container
python3 -I tools/mellum_storage.py observe-run \
  --batch /absolute/new-observation-container/artifact \
  --output /absolute/new-observation-run
```

Prepare reads only metadata, current source/binary and uniform plans. It freezes
four one-worker cases in order: original buffered/avoid, then packed buffered/avoid.
Run independently verifies actual model/packed bytes before and after all cells,
using exactly two full packed-verifier passes. It discovers a unique APFS
physical store, whole disk and IOBlockStorageDriver by topology, then retains
raw read/write/operation/error/retry counters. Unsupported or ambiguous mapping,
active inference, changed identities and consumed-wire substitution are refusals.

The optional native R/G/D/L pipe gates pause after prewarm/buffer preparation and
after timed worker join. Python collects one-second idle controls on both sides
and counters enclosing the active work, holding snapshots/output in memory until
controls finish. Retained container sizes plus 128 MiB of transient capture/decoder
headroom must fit a 256 MiB accounting ceiling; this is not a measured whole-process
RSS limit. Previous cell raw data is released before the next capture. Snapshot
start/end uncertainty is retained. Metadata commands
have 5-second/32 MiB capture bounds; XML decoding additionally requires at most
1 MiB, 16,384 nodes and depth 64, with internal entities refused. Native gates
have 30-second bounds, and active work 120 seconds.
No hashing, prewarm or experiment report writes occur inside these windows.

The new reservation permits 13,476,298,752 timed logical bytes (16 GiB cap),
6,738,149,376 native prewarm bytes (8 GiB cap) and at most 128 GiB cumulative
Python preparation/verification reads. The forecast includes two M+3P payload
verification passes, 128 MiB directory/prefix allowance and 8 GiB metadata/source
allowance; actual budgeted reads are refused before exceeding their ceiling.
Git internals, runtime imports and metadata subprocess internals are excluded
from Python logical file-read accounting. Command outputs and native reads are
retained separately. Failed children can have unknown actual native progress;
reports retain reserved maxima and never label that progress zero.

Each cell retains native samples, raw stdout/stderr, six timestamped counter
snapshots (raw command bytes encoded as base64), events and identities. A second
execution is refused, including after failure. Partial evidence is published only
under a separate `.failure/artifact`; missing/unstarted cells are explicit.

Interpret D as raw device-read bytes, L as completed logical bytes, and B as the
larger adjacent idle rate times the conservative outer observation span. B>1% L
or D>125% L flags inconclusive background; otherwise D≤5% L is little traffic,
90% L≤D≤125% L is consistent volume, and other cases are mixed/inconclusive.
Counter/identity/interval failures invalidate observation. These bins provide no
statistical attribution guarantee: an unrelated burst can escape both controls.
Neither consistent volume nor successful cache flags establish file-specific
service, SSD bandwidth, NAND reads, expert miss stall or inference tok/s.
`physical_io_verified=false` remains set. All outcomes end this four-run budget.

### Fresh-file storage-service attempt

The approved CPU-only service study uses a new aligned scratch file for each
cell: 64-expert calibration, then three 512-expert repetitions if each previous
cell is `consistent-volume` and passes exact byte verification. It preserves Q8
bytes and routing. This measures one-worker reads of an aligned packed subset.

```sh
make -f Makefile tools/mellum_stage_bench tools/mellum_read_bench
python3 -I tools/mellum_storage.py service-prepare \
  --prior-batch /absolute/prior-batch/artifact \
  --reader "$PWD/tools/mellum_read_bench" \
  --stager "$PWD/tools/mellum_stage_bench" \
  --output /absolute/new-service-batch
python3 -I tools/mellum_storage.py service-run \
  --batch /absolute/new-service-batch/artifact \
  --output /absolute/new-service-run
```

Preparation freezes source, binaries, sorted slots, read order and budgets using
metadata only. Run reserves once, fully verifies unchanged sources, computes
expected hashes from selected source bytes, stages through bounded explicit
uncached writes, then observes the reader. Scratch is first read by the timed
native reader; its exact content, padding and hash are checked afterward in one
streaming comparison with independent source reads. Source verification also
runs at normal termination, including a completed nonqualifying calibration.
Cancellation or I/O/budget failure stops immediately with closing verification
marked incomplete. No retry or replacement is automatic or authorized.

The root is exclusively created and never reused. Scratch and partial scratch
remain under `scratch/`; raw evidence remains under `evidence/`. Publication uses
`result/artifact` for all four qualifying cells and `failure/artifact` otherwise.
The batch reservation stays spent after failure. Preparation and execution are
bound to the same committed source state; changes require a separately approved
attempt, not reuse of an old manifest.

The planned native logical reads are 10,528,358,400 B in each of the timed and
staging-source classes; writes and retained scratch are 10,538,188,800 B each.
Each class has a 16 GiB cap, with zero explicit prewarm. The cumulative Python
forecast is 136,920,461,888 B under 128 GiB, including expected source hashes and
post-timing comparison. Partial native progress may remain unknown; reserved
maxima are retained. Require 32 GiB free, 16 KiB host pages and 4 KiB media blocks.
The stager has a 120-second deadline and 1 MiB combined output bound.

All device classifiers and their thresholds remain unchanged. Even qualifying
volume traffic cannot attribute individual expert service to SSD or NAND;
`physical_io_verified=false` always. No inference throughput or 8 GB feasibility
conclusion follows. Original-range costs, concurrency and representative locality
remain separate S3 work; the all-hit GPU cache path remains S4 work.
