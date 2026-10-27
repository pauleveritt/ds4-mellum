# Mellum owned recorder live validation

**Date:** 2026-10-27  
**Question:** Does the owned diagnostic recorder preserve resident execution,
fail safely and publish source-checked v2 evidence?  
**Scope:** Resident Q8 routing collection on the development laptop. No SSD
inference, storage throughput measurement or physical 8 GB test.

## Evidence locations and source identity

The primary evidence is this session's native test/publisher output, retained in
[the live artifact directory](/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/mellum-owned-live-2054).
Sources below are local experiment records produced by this project and accessed
2026-10-27. `artifact-index.json` records file digests; `replay-summary.json`
records trace digests and replay statistics. Full manifests live in each trace.

Hardware: Mac17,7, Apple M5 Max, 128 GiB (137438953472 bytes), macOS 27.0.1
(build 26A434). This is not the physical 8 GB target. One model process at a time.
The model is `/Users/pauleveritt/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf`.
The publisher records model stat identity, not a fresh model-content hash.
Model/tokenizer/template/workload IDs and device label remain caller-declared.

All four qualified captures were force-built at `fa3d18804d59472e4be10b3f358fd7e5fbe41dec`,
with clean recorded source state `1fb482d092d612096196853627259391a34405be177ba929adcbf5e1d9d6bebf`.
The later failure test/build changes do not alter inference sources. The normal
shell sets `GIT_PAGER`; publication used `env -u GIT_PAGER python3 -I` because
inherited Git overrides are deliberately refused. No main/rebase/push work here.

## Findings

1. **Claim:** The owned recorder preserves routes and per-operation logits on
   the exercised release, exact, short, tiled and multi-chunk schedules.
   - **Evidence:** All four publisher invocations completed successfully. Each
     forced the parity harness, checked ordered reference/owned selections and
     requested schedule, validated observed profiles and runtime receipt, and
     atomically published a complete v2 trace. The harness checks logits after
     every sync/eval, final one-token prompt append, terminal state, immediate
     cancellation, copy/read/sink failures, sequential handles, and a direct
     no-head forward followed by logits. All corresponding native checks passed.
   - **Source:** `publish-{short,tiled,exact,multichunk}.log`, the embedded build
     manifests and four `*.trace.jsonl` files. Publisher logs are empty on success;
     private parity logs are removed by the publisher. `short.log` and `short/`
     retain an additional direct harness run with raw profiles and receipt.
   - **Assessment:** Fact about tested paths; no claim about untested hardware.

   | Artifact | Fixture tokens | First prefill | Chunk cap | Mode | Forwards | Profiles |
   | --- | ---: | ---: | ---: | --- | ---: | ---: |
   | short | 29 | 20 | 1024 | release | 10 | 2 |
   | tiled | 164 | 64 | 1024 | release | 101 | 2 |
   | exact | 164 | 64 | 1024 | exact-prefill | 101 | 2 |
   | multichunk | 6000 | 5998 | 1024 | release | 8 | 3 |

   The long prefill supplies absent-head intermediate-chunk profiles. Actual
   capabilities report tensor tiles available, M5 true and pre-M5 false. These
   are pretokenized tuning fixtures with teacher-forced continuation, not
   representative generated or held-out evaluation cohorts.

2. **Claim:** A forced non-NAX/non-Metal4 path also preserves matching-path
   logits/routes and supplies valid actual-encoding profiles.
   - **Evidence:** Native harness passed with `DS4_MELLUM_MOE_NAX=0`,
     `DS4_METAL_DISABLE_METAL4=1`, `DS4_QWEN4_PREFILL_CHUNK=100`, code fixture,
     first prefill 64. Ordered spool comparison passed; both owned profiles
     passed the complete Python profile validator. The prefill profile names
     `kernel_mul_mm_q8_0_f32`, `kernel_qwen4_moe_mm_mid` and
     `kernel_qwen4_moe_mm_down`, rather than assuming requested flags identify
     actual execution. Native hooks report zero added buffered per-layer waits;
     the owned implementation uses the same copy primitive without a host wait.
   - **Source:** `fallback.log`, `fallback/{reference,owned}.routes`,
     `fallback/owned.profiles`; `tests/mellum_routes_compare.py` and
     `tools/mellum_capture_metadata.py` checks.
   - **Assessment:** Fact about matching-path parity. Raw fallback spools have
     not been published as qualified v2 evidence; the publisher exposes only
     default/exact controls. Cross-path numerical equality is not asserted.

3. **Claim:** Injected creation failures and between-chunk cancellation leave
   no tracked owned resources or usable failed capture.
   - **Evidence:** The model-gated `test-mellum-capture-failures-gpu` passes all
     107 allocation boundaries reached during successful creation. It tracks
     intercepted C allocations and Metal tensor handles through matching frees;
     each failure returns null, writes no spool header and leaves no active
     commands. Cancellation follows a successful logits-producing sync, then
     interrupts a later multi-chunk sync after one 32-token chunk. Exactly two
     forwards / 33 positions exist; state is FAILED, collector scope is cleared,
     valid-buffer logits and finish are refused, and free adds no output. A fresh
     handle succeeds with profile ID zero and readable logits.
   - **Source:** `failures-reviewed-{build,run}.log`,
     `failure-mutation-check.json`, `tests/mellum_capture_failures_gpu.c`.
   - **Assessment:** Fact at intercepted C/Metal tensor API boundaries. This does
     not simulate physical driver OOM or account for all Metal-internal resources.

   Fresh independent review found two P2 test gaps: uncounted frees and a
   vacuous NULL/zero-capacity logits check. Both were strengthened. Deliberately
   omitting `free(c->host)` fails the live-count assertion; deliberately bypassing
   the failed-capture logits guard fails the valid-buffer assertion. Both mutated
   engines exit 1; the real engine exits 0. Source copies, build/run logs are
   retained as `leak-mutant-*` / `logits-mutant-*`. No production change was needed.
   Build review also prompted a Darwin-only target and dependencies matching the
   included engine. The Linux dry-run prints a skip and does not link Metal.

4. **Claim:** Recorder-disabled resident generation shows no resolved regression
   in the small alternating comparison.
   - **Evidence:** Before-adapter `c9ac887` versus current `fa3d188`, one warmup
     each and three measured samples per arm, alternating order. Context 8192,
     temperature zero, 33 prompt tokens, 512 generated tokens. Baseline rates:
     126.41, 125.21, 122.40 tok/s; current: 124.07, 126.35, 126.01 tok/s.
     Medians 125.21 versus 126.01 tok/s (+0.64%). Every stdout SHA-256 is
     `74b81bf9a6d126b9b8ddbbe6c8eb2d0b27b3fa8dfb94f630f8140d1f0c0a7e91`.
   - **Source:** `decode-comparison.json`, `decode-rates.json`, eight
     `decode-*.{stdout,stderr}` files and both build logs; baseline source archive
     is retained in `baseline-source/`.
   - **Assessment:** Measurement plus limited inference. The sample spread is
     larger than the median difference; +0.64% is not evidence of an improvement.

5. **Claim:** The fixed recorder's directly owned allocation fits the declared
   2.2 MB bound at cap 1024; whole-engine memory is a separate measurement.
   - **Evidence:** Actual allocator sizes: handle 245760 bytes (requested
     233384), host IDs 917504, Metal backing allocation 917504, tensor wrapper
     heap 48, plus the bounded 16384-byte catalog stack buffer: **2097200 bytes**.
     Four ordinary/four owned create/free cycles return tracked tensor bytes to
     zero; device counters can retain small backend allocations independently.
     Ordinary session request is 432496896 bytes at context 8192/cap 1024 in
     the timing experiment, separate from resident weights and recorder storage.
   - **Source:** `allocation-lifecycle.{stdout,stderr}`, `allocation_lifecycle.c`,
     `allocation_probe-v2.m`, `owned-cost-v2.stdout` (actual allocator/Metal queries).
   - **Assessment:** Fact about directly owned buffers/heap and a specified
     stack bound. It excludes caller/stdio buffers, backend/global allocations,
     driver/page effects and model/session state; it is not whole-machine peak.

6. **Claim:** The initial enabled-recording measurements indicate overhead,
   with substantial timing drift across local runs.
   - **Evidence:** Eight sequential sessions: warmup pair followed by three
     samples per arm with alternating order. Tool fixture, 64-token initial
     prefill, 901 teacher-forced decode tokens, one final prompt append.
     Every operation's full logits match the first ordinary run. With the
     second probe, median decode time is 9.465769 s ordinary versus 9.731613 s
     owned (+2.81%); rates 95.19 versus 92.58 tok/s. First-probe medians were
     8.032840 versus 8.780495 s (+9.31%), during concurrent CPU compilation.
     These intervals include route spool writes for the owned arm, exclude
     logits comparison, session creation and final catalog/close, and use
     warm OS caches/private temporary files. Short prefill/append timings are
     retained but too small/noisy to characterize product performance.
   - **Source:** `owned-cost{,-v2}.{stdout,stderr}`, both summary JSON files,
     `owned_cost.c`, probe sources and build logs.
   - **Assessment:** Diagnostic measurements, not stable throughput predictions
     or SSD overhead. No enabled-overhead pass/fail threshold was agreed.

   After the lifetime repair at `e567a5f`, the same unpooled eight-session
   experiment passed all per-operation logit comparisons. Every session returns
   to exactly **12924813312 device bytes** after free, including all four owned
   sessions. Device bytes after work are stable per arm (13357219840 ordinary,
   13358137344 owned); their difference is the 917504-byte ID buffer. The fixed
   recorder bound remains 2097200 bytes. This closes the observed allocation
   retention issue without requiring a caller-supplied Objective-C pool.

   Post-fix adjacent measured pairs add **2.53%, 8.47%, 3.79%** decode time
   (median 3.79%). Aggregate-arm medians are 8.176677 s versus 8.869533 s
   (+8.47%; 110.19 versus 101.58 teacher-forced tokens/s). Pair and aggregate
   summaries differ because ordinary times drift upward through the run. Retain
   both rather than claiming a stable recorder cost. Source:
   `owned-cost-after-fix.{stdout,stderr}`, `owned-cost-after-fix-summary.json`,
   `owned-cost-after-fix-build.log`; no CPU compilation overlapped this run.

7. **Claim:** A shared Metal C API lifetime defect explains the observed
   post-session buffer retention.
   - **Evidence:** On the original backend, four alloc/free cycles of 1 MiB
     have zero device delta. Adding tensor read, write or contents access leaves
     4194304 bytes per four buffers despite zero tracked tensor bytes. Wrapping
     whole inference sessions in a diagnostic autorelease pool returns every
     ordinary/owned arm to exactly 12924813312 device bytes after pool drain.
     Merely draining the allocation probe's getter was insufficient. The
     buffer-property temporaries need a pool within the C helper boundary.
     The native regression fails at read (1769472 to 5963776 device bytes) and
     passes all four access modes after local pools in contents/read/write.
   - **Source:** `tensor-lifetime.stdout`, `owned-cost-pool.stdout`,
     `tensor-lifetime-{red,green}.log`, `tests/test_metal_tensor_lifetime.m`.
   - **Assessment:** Reproduced defect and targeted repair. Lifetime handling
     changes; pointer ownership, access bounds, copies and arithmetic are retained.

   Ruling: repair the three shared Metal helpers within the reviewed memory gate,
   since all three independently reproduce the same defect and C callers cannot
   be required to provide an Objective-C pool. Cost if wrong: a lifetime regression
   in other shared-buffer callers; covered by the native regression and resident
   parity/kernel checks. RED commit `b84a3fa`; no changes to routing or Q8 bytes.
   GREEN repair commit `e567a5f`; resident kernel/fast-path and all 107 creation
   allocation failures pass again after the repair (`resident-after-lifetime-fix.log`).
   Test/build instrumentation commit `c29540a` retains the reviewed failure matrix.
   Fresh independent review of the lifetime fix found no Critical/Important
   issues. Two minors are deferred: use a separate readback to strengthen the
   returned-pointer content check (its immediate equality may constant-fold),
   and include the new lifetime executable in `make clean`. Exact device-counter
   equality is verified on this host; other driver accounting remains untested.

8. **Claim:** All four published v2 artifacts round-trip through the existing
   offline replay; the results are diagnostic policy outcomes.
   - **Evidence:** A 3 GiB request fits 489 complete experts (3217729536 bytes).
     LRU decode demand including the transition from the initial prompt:

     | Capture | Decode tokens | MB/token (decimal) | Decode expert hit fraction |
     | --- | ---: | ---: | ---: |
     | short | 8 | 620.186 | 57.92% |
     | tiled | 99 | 416.216 | 71.76% |
     | exact | 99 | 415.751 | 71.79% |
     | multichunk | 1 | 1197.601 | 18.75% |

   - **Source:** `{short,tiled,exact,multichunk}.replay.json` and
     `replay-summary.json`, using `tools/mellum_ssd_replay.py --cache-gib 3`.
   - **Assessment:** Fact under the grouped atomic residency model. The long
     multi-chunk case tests prefill completeness and contains only one decode
     token; it cannot establish warm locality. Even the 99-token continuations
     are diagnostic teacher-forced fixtures, not the frozen S3 cohort. Replay
     also retains relaxed-optimal results; only its whole-trace total demand
     bound has the documented fixed-order interpretation. Per-phase outcomes
     must not be promoted to independent bounds or converted to SSD tok/s.

## Final validation and disposition

The repaired source published `after-fix.trace.jsonl` at engine revision
`e567a5fe92b29a612d0b092d096373bcac42ee26`, source-state ID
`82f45c457d521422643aa25688a8a3b1376e468d6a0220bc09a52e74b1429c77`.
Trace SHA-256:
`75318d1374f8780beb3fb7c5dc880b70afb32786bd976c170d9b7aac0203b2de`.
Every ordered trace row after the provenance header is byte-identical to the
original tiled capture, and both policies' full replay results match exactly.
Sources: `publish-after-fix.log`, `after-fix.{metadata.json,trace.jsonl,replay.json}`.

The native lifetime regression, resident kernel/fast-path gates and creation
failure matrix passed after the repair. The final serialized model-free gate is
retained in `model-free-after-lifetime-fix.log`. The earlier CPU gate attempt
was mistakenly overlapped with a live test and hit the shared process lock;
`model-free-serialized.log` and the final gate are the successful serialized runs.
No inference source changed after these checks. S2b's recorder correctness,
provenance, bounded allocation and overhead measurement gates are met on this
hardware. The branch/worktree remain for S3; no push/merge is part of this work.

## Open questions and limits

No streamed inference penalty, SSD bandwidth, predictive-prefetch effectiveness
or achievable physical 8 GB memory budget follows from these tests. The model
is resident (~12.03 GiB of weights); native context/session plans are separate
from recorder storage. RSS/`time -l` figures do not measure all mapped/shared
GPU/model memory and must not be used to claim 8 GB suitability.

The create matrix does not inject every later session/history reservation failure
or Metal-internal driver failure. CUDA/distributed execution and pre-M5 hardware
remain untested. General `make test` still requires an absent DeepSeek GGUF.
The completed publisher review retains its minor deferred subprocess-after-output
and checked-close injection coverage; this live stage does not resolve it.

## Recommendation

Use these captures as recorder correctness/provenance evidence. Freeze the S3
workload cohorts before decisive locality data, then measure actual expert read
ranges and storage behavior. A 10 tok/s checkpoint is still a proposal; there is
no SSD performance prediction or feasibility verdict from this stage.
