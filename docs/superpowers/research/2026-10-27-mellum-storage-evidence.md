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

## Real preparation and bounded batch
Code/source revision: `89369ea`; source state ID: `aee6dd423c156d76798b3844d9ffc491310a7705e8474f689f663e7b37f6a5ed`.
All four explicit CLI stages succeeded once, in order: index, pack, prepare, run.
The source model and diagnostic trace are the exact approved inputs. No repeat
batch or extra timed reads were performed.

Artifact root:

`/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4`

| Stage | Complete artifact below root | JSON SHA-256 |
|---|---|---|
| index | `mellum-storage-s3a-index/artifact/index.json` | `ac642c513a7256e74d04c08c4705d59489b8069c6d119f9ecd411536510be296` |
| packed | `mellum-storage-s3a-packed/artifact/manifest.json` | `03188eb7d2f839b74b170bc4399539bfda6c7ed976b254bb8d3328620ee2d728` |
| batch | `mellum-storage-s3a-batch/artifact/manifest.json` | `78595662c0e6e3c651846386cd11f2c90892230c2bc0e807fb12b331f1a1c6c4` |
| run | `mellum-storage-s3a-run/artifact/report.json` | `45133f3a2e3ac9448c71fd05e3b880a52474a5ed802d64e1b4bdfe5a3c2773dd` |

Model SHA-256: `ea1e4ec737bb259ac2ffe1f20d412f458fd52dd230f42dce9e944f111093c29b`.
Packed payload: 11,791,761,408 bytes, SHA-256 `bcce2e85ad8740b9102bf196e4ff2193e2a6db147af23112267bb1774d2bb5a8`.
Native binary SHA-256: `2d477f1de55ef40a4e893b2acdaaaf392d30c60f4af8bb3c4fbb67fd276b8088`.
Trace SHA-256: `75318d1374f8780beb3fb7c5dc880b70afb32786bd976c170d9b7aac0203b2de`.

The GGUF data section starts at byte 3,589,408, alignment 32. The complete
model is 12,925,417,760 bytes. All 1,792 expert entries passed strict geometry,
actual directory binding, exact copying and independent full byte verification.
The source and packed inputs were fully checked before and after the batch.
All 72 raw stdout hashes were independently checked after publication; every
raw report contains all 512 samples and the required cache controls.

| Read class | Logical bytes |
|---|---:|
| index untimed Python reads | 12,947,638,784 |
| packed untimed Python reads | 60,131,359,795 |
| batch untimed Python reads | 86,008,503,929 |
| run untimed Python reads | 172,042,256,721 |
| native prewarming, outside timing | 121,286,688,768 |
| native timed jobs | 242,573,377,536 |

The full timed budget was reserved once. Instrumented Python reads total
331,129,759,229 bytes; Git internals/runtime imports are excluded.
Native buffers peaked at 26,320,896 bytes (four complete experts), excluding
thread stacks/runtime/OS caches. These are preparation/read-tool memory figures,
not an inference or whole-machine 8 GB budget.

## Observed warm read timings

Each cell is the median of all three repetitions. Throughput is decimal GB/s;
service p95 is the median of the three per-run nearest-rank p95 values. Raw
per-job queue/service/end-to-end timings, completion order and every repetition
remain in the artifact; the table does not replace them.

| Plan | Mode | Workers | Original GB/s | Packed GB/s | Original service p95 ms | Packed service p95 ms |
|---|---|---:|---:|---:|---:|---:|
| uniform | buffered | 1 | 31.078 | 37.909 | 0.275 | 0.204 |
| uniform | buffered | 2 | 50.893 | 63.407 | 0.343 | 0.246 |
| uniform | buffered | 4 | 66.350 | 75.447 | 0.496 | 0.401 |
| uniform | avoid | 1 | 29.245 | 36.204 | 0.293 | 0.211 |
| uniform | avoid | 2 | 50.350 | 61.748 | 0.334 | 0.253 |
| uniform | avoid | 4 | 64.711 | 77.738 | 0.502 | 0.407 |
| diagnostic | buffered | 1 | 29.724 | 37.805 | 0.284 | 0.205 |
| diagnostic | buffered | 2 | 48.929 | 63.831 | 0.345 | 0.245 |
| diagnostic | buffered | 4 | 65.298 | 74.192 | 0.489 | 0.408 |
| diagnostic | avoid | 1 | 28.525 | 36.612 | 0.295 | 0.205 |
| diagnostic | avoid | 2 | 48.435 | 60.784 | 0.344 | 0.261 |
| diagnostic | avoid | 4 | 61.632 | 72.165 | 0.525 | 0.437 |

Packed reads use one syscall per expert versus three for original gate/up/down
reads, with returned/logical byte ratio 1.0 in successful runs. Median throughput
improved 13.6–30.5% across these cases. Total native timed makespan across all
72 runs was 5.102188 seconds. All repetitions are retained.

Both cache labels delivered roughly 29–78 GB/s. The avoidance flags succeeded,
but these rates are consistent with hot OS-cache service and do not establish
physical SSD throughput. No per-device read deltas were captured. There is no
measured SSD miss stall, streamed-inference tok/s or feasibility decision.

The native host collector recorded device observations as unavailable. A
separate post-run read-only `diskutil info -plist` inspection of `/dev/disk3s5`
and `/dev/disk0` identified APFS on the data volume, Apple Fabric, solid-state
storage and `APPLE SSD AP2048Z` (2,001,111,162,880 bytes). It is retained at
`/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-storage-s3a-device.json`. This metadata
is not physical-I/O evidence and does not alter the frozen report.

## Next work

S3a tooling, source freeze and this bounded warm-read experiment are complete.
S3 remains open. Device-observed storage traffic must be established before
using these rates as an SSD constraint. S3b still needs the generated Phase 1
ac-demo schedule, exact rendered/tokenized inputs, appends and representative
held-out windows. Native streamed inference and physical 8 GB memory/latency
validation remain S4/S5. Ten tok/s remains only a proposed checkpoint.

Execution commands, native build output, CLI stage logs and the final gate log
are retained in `/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-storage-s3a-execution`.
The [execution ledger](2026-10-27-mellum-storage-evidence-ledger.md) retains all
task completion checks, rulings and the deferred minor.
