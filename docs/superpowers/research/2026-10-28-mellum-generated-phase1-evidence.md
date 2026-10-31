# First generated Phase 1 preparation: retained failure, CPU fix qualified

Date: 2026-10-28, overnight window. Branch `codex/mellum-ssd-replay`, existing
isolated worktree. [Design](../specs/2026-10-28-mellum-generated-cohort-design.md),
[plan](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/plans/2026-10-28-mellum-generated-cohort.md). S3 locality remains open.
**No GPU inference, generated routing capture, cache replay result or SSD
inference measurement was obtained.** The GPU experiment was never reserved.
The earlier CPU storage-service experiment remains complete and spent.

## Implemented and reviewed

The new research collector preserves native greedy token IDs through two turns,
including terminal stops and capped first-turn host termination. It uses owned
resident capture and one ordinary resident replay, comparing every sync/eval
logit vector byte-for-byte. Python freezes source, model provenance, text,
native token inputs, forced CPU/Metal builds and the unchanged initial LRU.
Publication checks native receipts, every consumed spool frame, history, profiles,
nonfinite/tape failures, closing identities and whole-prefix LRU windows at
3 GiB. Raw failure evidence survives; successful artifact publication is separate.
No production engine source, routing, weights or quantization changed.

Fresh gpt-6-astra reviewed `19c7231..3ec435f`: no Critical, three Important,
two Minor, no declined judgments. The Important fixes are verified by native
fault-injection and CPU tests: success appears only after all evidence and the
completion receipt close; CPU commands must fit the frozen cutoff; protocol
stop stage/segment/terminal token and backend errors survive failure. Separate
RED `9c21722`/`9c02e38`, GREEN `da20684`/`6ec8849`; integration `2343a4c`.
No second reviewer was substituted for covering fix tests.

Full serial `make -f Makefile test-mellum` passed before the real preparation
at `2343a4c`, without the synthetic-lock overlay. Both binaries then passed a
fresh forced build. This gate runs no live GPU experiment.

## One real CPU preparation

Retained root:
`/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-generated-phase1`.

The sole native CPU inspection, PID **11994**, exited **1** after
**0.410472 seconds**. Its stderr was
`generated native operation failed; partial output retained`.
Exact frozen system/user/followup files remain (161/6661/316 bytes), plus command
and failure receipts. No native token files were written; no prepared artifact
or run directory/reservation exists. No retry or replacement ran. The source
freeze was `2343a4c67a00d42b0a3514cdbeda6ab9a750b28f`.

Source inspection found the cause: the Qwen/Mellum `inspect_only` branch in
`ds4_engine_open` returns before `vocab_load`. The collector's main checks for
98304 vocabulary entries, sees zero, and refuses before tokenization. Earlier
mocked preparation tests and the review missed this API detail. The failure
was unrelated to GPU availability, cache locality or the 8 GB memory budget.

All 379 frozen source inputs and both original binaries were copied and
hash-verified under `qualification/` before repair edits. The saved-evidence
failure audit passes: command/exit/duration, source and binary copies, missing
success/token/route outputs, and absent GPU reservation. It rereads no model,
old storage payload or scratch file. Original model provenance remains the
previous fully hashed index and unchanged inode/timestamps; no full model
rehash was performed.

## CPU-only correction and verification

RED `53656ca`, test expectation correction `22be1a6`, GREEN **`7170525`** add a
research-only CPU adapter compiled with the unchanged core. It opens inspect
mode and explicitly loads that model's existing vocabulary, without creating an
inference graph. Production `ds4.c` remains unchanged. A sparse-GGUF integration
test now exercises ordinary inspect returning vocabulary zero, the adapter
loading the native vocabulary, actual chat-token preparation/rendering and the
native end-marker tokens. Mellum thinking-on leaves the assistant prefix open;
the model emits its own think tag. The test initially expected Qwen's injected
think tag; its separate correction follows Mellum's existing template contract.

The focused native preparation test and rebuilt CPU collector pass. The complete
serial model-free gate passes again after `7170525`; `git diff --check` passes.
This validates the correction on the synthetic native fixture. The repaired
collector has **not** been reopened on the real model and has not been live
GPU validated. The plan workspace remains retained because Task 4 is incomplete.

## Decisions and remaining limits

Executor rulings, in order:

1. Use one bounded two-turn Phase 1 coding proposal with a scripted user review,
   without executing output or inventing tool results. Cost if unrepresentative:
   n=1 does not close real tool-result, medium/8K or broader repository coverage.
2. Retain a bounded native gold-logit tape and compare every operation during
   one ordinary parity replay. Cost: about 2 GiB of bounded disk writes/reads
   and the shared 900-second child budget, separate from SSD read measurements.

Deferred review minors: operation start/end timing is not implemented (emitted
end timestamps cannot recover initial/append sync or first-token durations);
explicit EOS_ALT and driver capped-thinking-separator tests remain absent.
No timings or live outcomes are invented to fill these gaps.

The frozen preparation is a one-shot step and its original output root stays
reserved. Honor the plan's no-retry boundary: a fresh bounded CPU preparation
needs an owner decision before any real-model retry, followed by the still
unspent single GPU capture/parity attempt. The overnight automation is paused
at that boundary after useful independent repair/testing and evidence retention.
The next decision-making measurement remains generated ac-demo locality at
roughly 3 GiB; S4 all-hit GPU-readable Q8 slots and miss recovery remain separate.
No useful streamed generation speed or physical 8 GB feasibility is demonstrated.
