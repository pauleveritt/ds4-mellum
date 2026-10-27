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
executable links only its four storage source files with pthreads, without engine
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
