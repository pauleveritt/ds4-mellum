# Mellum routing GPU proof

Date: 2026-10-27. GPU window ends 19:13:49 UTC.
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
