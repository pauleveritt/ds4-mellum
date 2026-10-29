# Streamed Mellum decode, faster stops: preregistration

Date: 2026-10-29. Committed before any of the changes it judges. The screen behind it is
[`2026-10-29-mellum-stream-decode-screen.md`](2026-10-29-mellum-stream-decode-screen.md).

**Honesty header.**
- **Fixed in advance:** the arms, the workload, `n`, the metric and the pass rules.
- **Measured:** one M5 Max (128 GB). Nothing here speaks for the 8 GB iMac, whose SSD and GPU differ.
- **Not claimed:** speed at other cache sizes or with Q8_0 experts. Those are reported if run, and are not
  judged.

## Arms

- **A:** `research/mellum-stream-decode` at its merge commit `42f51b7`, which is phase C's kit (`227d9cb`)
  plus the SSD branch at `9fb27ad`.
- **B:** the same branch at the head the night ends on, after the changes this note judges. Each change is
  its own commit and keeps decode's arithmetic. The candidates, from the screen:
  - overlapping the resumed pass's encode with the reads;
  - bounded victim scans;
  - chunked reads;
  - windowed command buffers.
  A candidate that a screen shows to lose is not landed, and the research note records it.

Both arms are built in their own worktree, from clean trees, with `make tools/mellum_s4_measure`.

## Workload

- **Model:** the MXFP4 GGUF, SHA-256 `cf09584a…` (phase C's).
- **Prompt and command:** phase C's frozen `workload-prompt.txt` (2,686 tokens), run as
  `tools/mellum_s4_measure --prompt … --gen 512 --context 32768 --stream-bytes 2684354560` (815 slots).
- **Order:** A B B A A B. Each run is a fresh process with a 20 s rest before it. Nothing else runs on the
  GPU; the machine is on AC power.
- **n:** 3 per arm.

## Metric

- **Per run:** the median of the per-token `ns` over decode tokens 1–511 (token 0 follows the prefill).
- **Per arm:** the median of its three run medians, with the range.
- **Also reported, not judged:**
  - time to first token;
  - loads and stops per token;
  - a resident run of B, as a floor;
  - one run of each arm at 3 GiB.

## Pass rules

1. **Identity (gate):** every run of B generates the same 512 token ids as every run of A. Decode's
   arithmetic must not change. A mismatch fails the whole night's change, whatever its speed.
2. **Speed:** B's arm median is at most 0.85 × A's. The bar is set where the screen's run-to-run spread
   (37–44 ms per token on one command) cannot produce it alone.
3. **Tests:** `make test-mellum`, `make test-mellum-kernels`, and `ds4_test --mellum-stream-identity`,
   `--mellum-stream-prefill` and `--mellum-stream-faults` on the MXFP4 file and the Q8_0 file all pass at B.

If rule 1 holds and rule 2 fails, the changes stay only where their own screens showed a gain. The note
reports that as "no decided gain".

## What would falsify the screen's reading

- B passes rule 1 but improves by less than the sum of the screen's estimates for the landed changes, by more
  than half. Then the estimates double-counted: the costs overlap rather than add.

## Risks

- Thermal state drifts across six processes. The ABBAAB order spreads the drift; it does not remove it.
- One prompt. The stop rate depends on the routing, which depends on the text.
- The 0.85 bar is judgment.
