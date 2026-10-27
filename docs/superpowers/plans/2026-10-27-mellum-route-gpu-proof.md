# Mellum routing GPU proof

Date: 2026-10-27. Status: completed diagnostic proof; S2b remains open.
GPU window ended early after the resident gates passed; GPU released at 18:41 UTC.
Baseline: e382350; isolated codex/mellum-ssd-replay worktree.

## Scope

Validate the S2b capture primitive before selecting a public session API.
Use a dedicated test translation unit containing ds4.c, with compile-time
layer/completion hooks absent from shipping builds. One session/process at a
time. The harness owns one cap-strided GPU buffer, one host buffer and spool.
Copy ordered router IDs after each successful layer; drain only after the
normal forward completion. Caller labels prefill/decode and append segments.
A deliberately synchronous reference drains each layer for ordering comparison;
never use its timings as performance evidence. Compare all final logits bytes
against uninstrumented execution using the same configuration and tokens.

This is a bounded diagnostic proof, not completion of S2b. Public attachment,
cancellation/failure lifecycle and actual encoded kernel provenance remain open.
Raw spools remain experimental; do not publish measured v2 traces with invented
kernel descriptions. No SSD inference or physical 8 GB conclusion.

## Tasks

### Task 1: live behavioral test (red commit)

Write the dedicated harness and Makefile target. Run baseline, buffered and
synchronous modes on fresh sessions in one model process. Compare final logits
and complete native spools byte-for-byte. Include prefill, one-token prefill
append and decode, explicit phase/segment labels. Allocate by checked native
layout. Without engine hooks the spool must be empty and fail the test.

Expected: build succeeds; live test fails because no forwards were captured.
Commit failing behavioral test separately.

### Task 2: test-only graph hooks (green commit)

After each successful layer invoke the test hook before scratch reuse. Invoke
completion hook only after the existing successful command-buffer completion.
Compile hooks only when DS4_MELLUM_ROUTES_TEST is defined. No new public API,
default link inputs, router calculations or production allocation changes.
Expected: live parity/ordering checks pass. Product additions <150 lines.

### Task 3: GPU evidence

Run exact and release configurations; cover decode, short/tiled prefill and
multiple chunks. Measure warmed baseline/buffered forward times separately,
record allocation bounds and test limitations. Run applicable CPU regressions,
Metal smoke and independent whole-change review; commit evidence and roadmap.
No push/merge. Keep S2b open for its remaining gates.

## Review focus

Check copy placement before selected scratch reuse, offset/stride arithmetic,
normal wait before CPU read, resource lifetime, phase labels, parity assertions,
reference independence and default-build isolation. Do not grade test-only proof
as a completed production recorder or locality/throughput evidence.

## Completion record

Eight matrix cases / 56 sessions pass, with complete matching route spools
and matching per-operation logits. The controlled smoke passes 21/21 sessions
under inherited chunk=1/exact=1. Resident trace, Metal kernels, fast-path and
model-free Mellum gates pass. General `make test` cannot finish without the
absent default `ds4flash.gguf`; a preceding attempt encountered a separate
server's instance lock. GPU use has ended; no push, merge or worktree cleanup.

Independent review accepted the core capture. Its smoke coverage findings
were resolved by using >=32 rows for exact and clearing inherited controls.
See the [evidence and limits](../research/2026-10-27-mellum-route-gpu-proof.md).

Execution decisions and costs:

- Deliver a test-only primitive first; replacing that harness may be necessary.
- Extend evidence with raw-spool replay at the existing 3 GiB budget; discard
  these exploratory counts if provenance is inadequate, with no feasibility verdict.
- Defer public ownership/cancellation/rewind; this does not prove production safety.
- Defer encoded-kernel provenance/v2 publication; new instrumentation may require recapture.
- Defer representative locality, throughput, portability and physical 8 GB claims;
  these fixtures cannot justify deployment decisions.
- Rely on the existing port for absolute model/kernel correctness; parity can
  share a baseline defect even though the existing kernel/fast-path gates pass.
- Source-review the no-logits completion boundary, without separately observing
  live no-logits/intermediate logits; add dedicated tests before S2b closure.
- Parent validates retained matrix/smoke results; a provenance error would need
  a log/source audit despite the independent implementation review.
