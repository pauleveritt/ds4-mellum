# Mellum storage-service execution ledger

Date: 2026-10-27. Branch `codex/mellum-ssd-replay`, existing isolated worktree.
Baseline `7263eff`; implementation freeze `010e30c`. All work local. No GPU,
inference, routing/quantization change, ac-demo collection, engine integration,
main change or push occurred during this CPU plan.

## Commits and verification

- Task 1: `7371533` RED → `22b0893` GREEN; cancellation retains captured cell evidence.
- Task 2: `a9bb860` RED → `7eac6c4`, `c71ed92` GREEN; aligned explicit writer, faults and CLI.
- Task 3: `748c1f5` RED → `da6154c`, `1b4cd28` GREEN; frozen plans, expected source hashes, post-read comparison. Tiny fixture output-name collision corrected in GREEN.
- Task 4: `8ffa495` RED → `39896bd`, `19e83d0` GREEN; bounded stages/observations and one-shot orchestration. `a6d5ce3` adds CLI docs/full-gate integration.
- Final review: fresh gpt-6-astra reviewed **7263eff..a6d5ce3**; three Important findings, no Critical/new Minor findings. Full report retained in qualification.
- Final fix pass: `a97d78f` separately commits four failing regressions; `010e30c` closes final bindings after discovery/evidence preparation and preserves command PID/original cancellation type. Four regressions RED→GREEN; complete observer **10/10**, process **6/6**, service runner **7/7** suites pass.
- Final full serial model-free `test-mellum` gate passes with the verification-only synthetic-fixture lock overlay; `git diff --check` passes. Reader/stager link only standalone C I/O sources, no inference/GPU objects. No second review or experiment was substituted for the fix tests.

The default full gate twice reached `test_mellum_config`'s synthetic backend
refusal and stopped because unrelated live inference held the shared lock
(PIDs 70334 then 73710). Its successful counterpart runs identical prerequisites
and recipe checks, applying the existing DS4_LOCK_FILE control only to this
DS4_NO_GPU synthetic fixture. No production lock or experiment preflight was
changed. The successful final command was:

```sh
make -f Makefile \
  -f .superpowers/sdd/2026-10-27-mellum-storage-service/fixture-lock.mk \
  test-mellum
```

## Rulings, in order

1. Python computes expected hashes from a selected-source prepass rather than new native cryptography. This preserves stdlib-only hashing and avoids scratch reads before timing. **Cost if wrong:** an extra 10,528,358,400 source bytes, included in the frozen forecast.
2. Isolate only the synthetic configuration fixture's lock for verification, because unrelated inference blocked the default gate. The real attempt still performs unchanged busy-process preflight and shared-lock rules are untouched. **Cost if wrong:** the fixture's lock coverage differs from the global production lock; no production inference-lock behavior changed.
3. Leave the preexisting late-stderr draining issue outside this change, explicitly deferred by Task 1. Primary events/counters and original failure identity remain the acceptance evidence. **Cost if wrong:** stderr emitted late during snapshot failure can be absent from diagnostics.

Deferred minor: preexisting late-stderr draining after a failing snapshot. The
review explicitly declined to reopen it; no other behavior was set aside.

## Reserved real execution and audit

The one approved prepare/run executed at `010e30c`, both commands exit 0.
Calibration and three repetitions qualified; all source/scratch byte checks,
closing verification and topology passed. The root, reservation, four scratch
files, raw receipts, native samples and counter snapshots remain preserved.
No retry or replacement ran. Sources stayed unchanged; all declared caps held.
See [results](2026-10-27-mellum-storage-service-evidence.md) for measurements and
scientific limits, including the 128 GiB actual host and unverified physical
attribution.

Retained root:
`/Users/pauleveritt/.codex/visualizations/2026/10/27/01a17e35-bf7c-7df2-9d99-b9bc0e0658f4/`.

- `mellum-storage-service-batch/artifact`: frozen manifest and eight wires; parent holds spent execution reservation.
- `mellum-storage-service-run/scratch`: four original private files; never hash or reread for routine status/audit.
- `mellum-storage-service-run/result/artifact`: qualified report, hardlinked raw evidence, expected/verified records and per-cell results.
- `mellum-storage-service-qualification`: exact commands, both gate refusal logs, passing final gate, review, RED/GREEN logs, saved progress, artifact-only audits/scripts and qualification hash index. Hash indices exclude model/payload/scratch.

Both saved-artifact audits pass; the final audit also checks raw media geometry
and full native validator output. The first audit/script remains retained;
`audit-final.py` and `audit-output-final.json` record the expanded audit. Audit
rechecks concern saved evidence only, not another storage execution.

## Handoff

The CPU plan is complete. Keep branch/worktree for Paul's separately authorized
overnight GPU work. The heartbeat `mellum-overnight-gpu-work` checks from
2026-10-28 02:00 Eastern every 15 minutes, stops GPU work at 08:00, reports and
pauses. It may continue routine SSD roadmap work without waiting for Paul, while
respecting existing per-plan limits. No repeat of this completed storage attempt
is authorized. Next is representative locality; S4's all-hit GPU cache path and
miss recovery remain separate. S3 stays open.
