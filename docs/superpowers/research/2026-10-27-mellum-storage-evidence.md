# S3a exact-Q8 storage evidence

Implementation and measurement follow the approved
[storage plan](../plans/2026-10-27-mellum-storage-evidence.md) and
[design](../specs/2026-10-27-mellum-storage-evidence-design.md).
The [evaluation source freeze](2026-10-27-mellum-workload-source-evidence.md)
is separate: no ac-demo routing data is used here.

## Scope and host

Development laptop: Mac17,7, Apple M5 Max, 137,438,953,472 bytes RAM (128 GiB).
APFS data volume `/dev/disk3s5`, approximately 898 GiB free before preparation.
No matching ds4-server/CLI/agent or Mellum trace/GPU test process was observed
at preflight. This work does not initialize the GPU or run inference.

The native binary's actual build command is:

```sh
cc -std=c99 -O2 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE \
  -Wall -Wextra -Werror -pthread tools/mellum_read_bench.c \
  tools/mellum_read_plan.c tools/mellum_read_io.c tools/mellum_read_run.c \
  -o tools/mellum_read_bench
```

No core, engine or GPU objects are linked. Python is 3.14.7, standard library
only. Source/binary hashes in artifacts bind the actual inputs; the compiler
recipe is recorded as a declaration plus executor build evidence, not signed
build attestation.

## Model-free verification

Full serial `make -f Makefile test-mellum` passed after the storage gate was
added at `a20dd09`; `git diff --check` passed. Focused storage verification
includes malformed/truncated GGUF inputs, all expert roles/offsets, exact byte
copy/reread, literal shared wire bytes, native parser failure injection,
partial reads/EINTR/EOF, cache-control failures, worker content/timing/join
checks and CLI identity/checked-close failures. The complete orchestration also
passed 72 tiny native invocations on synthetic files; these are tests, not
Mellum SSD performance measurements.

## Read accounting and interpretation

The fixed real batch has 72 invocations of 512 complete expert jobs, totalling
242,573,377,536 timed logical bytes, below the one-batch 256 GiB budget.
Native prewarming bytes and instrumented untimed Python preparation/verification
reads are separate. The latter include metadata, source and raw-report reads;
Git internals and runtime imports are excluded. None of these logical byte
counts proves physical-device traffic. Hashing and full verification may warm
the OS cache.

The two modes are explicitly prewarmed buffered I/O and cache avoidance
requested via successful macOS F_NOCACHE/F_RDAHEAD controls. Every artifact
retains `physical_io_verified=false`. Per-job latency, makespan and bandwidth
are storage-job measurements; there is no conversion to inference tok/s or
claim about useful physical 8 GB performance. S3 remains open for generated
ac-demo collection, held-out windows/locality and stronger device evidence.

Independent review, real commands/artifacts and outcomes are retained below
before this plan is closed.

## Independent review and fix pass

Fresh gpt-6-astra review of `f05bbab..a20dd09` found two Important issues, no
Critical issue, and one deferred minor. No second review was dispatched.

- Packed output descriptor/path substitution: the regression failed before
  `5e15e6c`. Additional same-file content-mutation coverage failed before
  `9d0a185`. The output identity is now unchanged from the completed write,
  through verification/hashing and checked close, and must still name the
  published payload path.
- Temporary wire substitution during the actual native invocation: the
  regression failed before `5d6e156`. The native parser now checks its opened
  plan descriptor/path identity before/after parsing, emits that consumed
  identity, and Python compares it to the frozen wire identity. A restored
  content hash alone cannot authorize different consumed offsets.

Deferred minor: worker stderr loses the detailed first failed job/read error
and retains a generic execution refusal. Failure detection, cleanup and refusal
remain correct; richer diagnostics can follow separately.

Reviewer exclusions were assessed: source-freeze work has its own retained
review (cost: storage review does not re-review that scope); real-file and
full-gate evidence remain the executor's task below (cost: synthetic/review
results alone prove neither); physical/cold-cache/inference/8 GB claims remain
unverified (cost: no target feasibility verdict); build provenance is not signed
attestation (cost: no cryptographic compiler-to-binary guarantee); publication
has no new power-loss durability guarantee (cost: interrupted preparation can
need a fresh output); privileged metadata falsification/kernel corruption are
outside ordinary filesystem identity checks (cost: a privileged adversary can
evade them). The full gate includes the subprocess test omitted by the reviewer.
