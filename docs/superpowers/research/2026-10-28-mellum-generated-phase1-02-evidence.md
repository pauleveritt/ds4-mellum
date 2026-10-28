# Generated Phase 1 conversation, attempt -02: capture, parity and 3 GiB LRU

Date: 2026-10-28, owner-reopened window to 22:00Z. Branch `codex/mellum-ssd-replay`.
[Design](../specs/2026-10-28-mellum-generated-cohort-design.md),
[plan](../plans/2026-10-28-mellum-generated-cohort.md),
[first attempt](2026-10-28-mellum-generated-phase1-evidence.md).

**Honesty header.** Measured: one generated two-turn ac-demo conversation
(**n=1**) on a 128 GiB M5 Max, its native routing, byte-exact ordinary resident
logit parity, and one cold-start global grouped LRU replay at 3 GiB over the
saved trace. Not measured: SSD reads, stall time, streamed generation speed or
8 GB feasibility. Both turns hit their caps inside an unclosed think block, so
every decode window is thinking-phase routing. The publisher failed after the
native run; the analysis below comes from an artifact-only validation of the
saved raw files with a one-line serializer fix, not from the publisher's own run.

## What ran

Root: `/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-generated-phase1-02`
(`build`, `inputs`, `run`, `validated`, `qualification`). The first root
stays untouched.

- **Window.** The 08:00 Eastern cutoff passed unspent. The owner reopened it to
  18:00 Eastern. Test-clock correction `8a0028c`, RED `69665ac`, GREEN `cda49b0`.
  The publisher now shares the inputs module's deadline.
- **Freeze.** Full serial `make -f Makefile test-mellum` passed at `cda49b0`.
  Forced CPU/Metal build PID 63735, exit 0, 11.07 s. Source `cda49b0`, state
  `08f8b27c…`, 381 files. LRU policy `fb298de…`, unchanged since `19c7231`.
  Copies of the source and both binaries are hash-verified under `qualification/`.
- **CPU preparation, once.** The repaired collector, PID 63984, exited 0 in 0.496 s.
  Native tokens: initial 1950, append 72, close 2, newline 1.
- **GPU, once.** Lock free and no inference process before reservation. Native
  collect PID 64092, exit 0, 107.7 s of the 900 s cap: `complete` and `parity`
  true, counts 4096/1024, both stop class 0 (cap), 7144 tokens, 5122 S/E
  operations, 2,014,052,352-byte gold tape (5122 × 98304 floats).
- **Publisher failure.** `validate` copied `host_metadata()`'s sysctl command,
  bytes and all, into the trace header, and `write_json` raised `TypeError`.
  `run/failure.json` and every raw file are kept, and nothing was retried. The
  fixture host had no `command` field. RED `354f7b6`, GREEN `d641df3`.
- **Artifact-only validation.** The driver hardlinks `run/raw` into a new
  exclusive `validated/raw`. It asserts the tree differs from `cda49b0` only by
  that test and the one-line `host=inputs.host.serial(host)`, then runs the
  publisher's post-collect gates against the frozen identity:
  - native receipts, operation ledger, recreated inputs, finite tape;
  - spool schedule, profiles, export;
  - the one LRU replay, closing input identity.

  `provenance.json` records both revisions. The first driver attempt refused
  at its opening source gate, before creating anything, and is retained.
- **Audit.** `audit.py`, retargeted to `-02`, passes. It independently rebuilds
  the S/E frames and token IDs, the tape length, and the source and binary
  copies. Its separate grouped LRU matches the totals and every window.
  Trace `26b05a0c…`.
- **Post-fix gate.** Full serial `test-mellum` passes at `d641df3`. Another
  session's `ds4-server` (PID 64754, started after this run's GPU child
  exited) held `/tmp/ds4.lock`. A verification-only overlay in the SDD ledger
  (`fixture-lock.mk`) points `DS4_LOCK_FILE` at a private lock file for the two
  `DS4_NO_GPU` synthetic refusal fixtures only (`test_mellum_config`,
  `test_mellum_generate_prepare`). This follows the 10-27 storage-service
  precedent. Production lock and preflight code are unchanged, and the lock
  holder was left alone.

## Locality at 3 GiB (489 expert slots), one replay from cold

Demand is LRU miss bytes, with each layer's full routed group made resident
before use. Windows overlap and are not summed. No window was shortened.

| Window | Tokens | Demand MiB/token | p50 / p95 / worst MiB | Row miss |
|---|---:|---:|---|---:|
| initial-prefill (2 chunks) | 1950 | 11.2 | 10856 / 11007 / 11007 per chunk | 1.000 |
| response0 first 32 | 32 | 404.2 | 358 / 747 / 1142 | 0.288 |
| response0 after 32 (warm) | 4064 | 351.6 | 326 / 634 / 1117 | 0.250 |
| response0 tail 256 | 256 | 400.0 | 364 / 684 / 960 | 0.285 |
| append-prefill (1 chunk) | 74 | 106.8 | 7901 | 0.937 |
| response1 first 32 (post-append) | 32 | 400.4 | 377 / 703 / 1098 | 0.285 |
| response1 after 32 | 992 | 365.0 | 345 / 640 / 1073 | 0.260 |
| response1 tail 256 | 256 | 385.1 | 358 / 715 / 1067 | 0.274 |

Whole decode, 5120 tokens: 1774 GiB demand, 354.8 MiB/token; per token p50
333, p95 640, worst 1142 MiB. A quarter of expert references miss (0.252),
and only 14.5% of layer groups are fully resident (20,744 of 143,360). Per
token, p50 25 of 28 layers wait on a miss. The trace touched 1782 distinct
experts (10.9 GiB), 3.6 times the cache. Neither the append nor the turn
boundary visibly resets locality: post-append first-32 sits with the other
early and tail windows.

Recompute from saved artifacts only (the audit script lives in the retained,
untracked SDD ledger):

```bash
python3 -I .superpowers/sdd/2026-10-28-mellum-generated-cohort/audit-02.py
```

## Limits

- n=1 scripted review; no tool results, no medium/8K context, no final-answer
  routing (thinking never closed before either cap). Host closed both turns
  with `<|im_end|>\n` after the cap.
- Timing is recorder-resident and includes logit-snapshot overhead. Per-operation
  start/end timing remains absent (deferred minor); no durations are claimed.
- EOS_ALT and capped-thinking-separator branches still lack explicit tests
  (deferred minor); this run exercised only the cap path, on both turns.
- Analysis code at `d641df3` differs from the frozen capture source `cda49b0`
  by the serializer fix alone, checked by the driver and recorded in provenance.
- Demand bytes are not stall time. Nothing here gives SSD inference speed or
  says whether 8 GB is physically feasible; S4 cached execution and miss
  recovery remain separate.

## What would falsify this

A conversation that leaves thinking, uses tool results, or runs at medium/8K
context and shows materially lower warm-window demand would show that this
thinking-only trace overstates misses. A different grouped policy or tiling
that beats LRU by a wide margin at 3 GiB would make LRU the wrong baseline.
A failure when re-running the audit on these artifacts would void the numbers.

## Risks

Thinking-phase routing may be unrepresentative of answer and tool turns. One
conversation cannot separate content from position effects. The artifact-only
validation re-ran publisher gates outside `run()`. Its delta check and the
independent audit are the controls on that, and no reviewer has read this
driver yet.
