# S6 phase C: preregistration (the 8 GB M1 iMac)

Date: 2026-10-28. Committed before any iMac run. Plan:
[`2026-10-28-mellum-s6-c-imac.md`](../plans/2026-10-28-mellum-s6-c-imac.md). Design: Goals 4 and 5 and section C.
The owner runs every step by hand. The agent analyzes the captures afterwards, and only them.

**Honesty header.**
- **Fixed in advance:** the inputs, the cache choice rule, and the pass rules.
- **Measured:** one 8 GB M1 iMac (macOS 27.1), n stated per item.
- **Not claimed:** other 8 GB Macs, answer quality, battery.
- **Each item runs once.** A failed or crashed run is recorded as it is, and a rerun after a fix is a new
  run, reported beside the old one.

## Frozen inputs

- **Model:** the MXFP4 GGUF, SHA-256 `cf09584a30423f6f80b0c9fe2f42e152520dabf08fecd1cad641ce21a7588166`.
- **Build:** the kit's named commit, the same on the iMac and in the M5 reference runs it is compared with.
- **Correctness reference:** the M5 Max's resident MXFP4 decode-mode logits at the oracle positions of
  `tests/mellum_oracle_{short,1100,2200}.tokens`. Their SHA-256s are in
  `2026-10-28-mellum-s6-c-evidence/m5-reference-sha256.txt`. On the M5, streamed equals resident byte for
  byte at all 157 positions.
- **Workload:** `2026-10-28-mellum-s6-c-prereg/workload-request.json` (SHA-256 `73cd5f77…`).
  - It is request #4 of a mellum-worker Pi run: system prompt, user task, and four tool rounds of file
    reads, with 7 tools.
  - It is replayed by `tools/mellum_replay_request.py` through `ds4-server -c 32768`: greedy,
    `max_tokens` 512.
  - The M5 reference for it is in `2026-10-28-mellum-s6-c-evidence/workload-m5/`.
- **Background load:** `tools/mellum_background_load.sh PROJECT urls.txt`, with PyCharm on a small
  Python project (`../mellum-demo` unless the owner names another before the run) and Safari with the
  ten URLs in `urls.txt`.

## C1. Bring-up (gate)

- **Pass:** `make test-mellum` and `make test-mellum-kernels` report 0 failures; every MXFP4 line runs.
  `ds4_test --mellum-stream-identity`, `--mellum-stream-prefill` and `--mellum-stream-faults` on the
  MXFP4 file pass.
- **Recorded:** `tools/mellum_metal_limit` (recommendedMaxWorkingSetSize, families, Metal 4), the machine
  facts, and the engine's startup lines.
- **A failure stops phase C.** The next step is diagnosis, not timing.

## C2. Choosing the cache (before any timed run)

Candidates in order: 3 GiB (979 slots), 2.5 GiB (815), 2 GiB (652). With the apps closed, start
`ds4-server --ssd-streaming --ssd-streaming-cache-experts N -c 32768` at each candidate in turn and replay
the workload once.

The chosen cache is the first candidate where both hold:
- the server's planned memory line is at most recommendedMaxWorkingSetSize − 1 GiB;
- `vm_stat` swap-outs do not grow during the replay.

If no candidate qualifies, record that. It is the spec's falsifier "under about 2 GiB for the cache".

## C3. Correctness against the M5 (gate)

`tests/mellum_trace MX FILE decode`, with `MELLUM_TRACE_STREAM_BYTES` set to the chosen cache and
`MELLUM_TRACE_SESSION_AT` set to the oracle positions, on all three files. Compared with
`tools/mellum_logits_agree.py` against the M5 reference; B is the M5.

**Pass, on every file:**
- agreement (near-ties within 0.75 nats counted) on at least 97% of positions;
- mean KL ≤ 0.01;
- worst KL ≤ 1.0.

A fail stops phase C at diagnosis.

## C4. Footprint with apps open (Goal 4)

At the chosen cache, three replays with the apps closed, then the background load, then three replays with
it running. `vm_stat` is sampled before and after each replay. The server restarts between the two
conditions.

**Pass, both:**
- swap-outs grow by 0 during every apps-open replay;
- the median apps-open decode t/s is at least 0.85 of the median apps-closed decode t/s.

## C5. Speed (Goal 5, reported only)

- From C4: time to first token and decode t/s, as medians with ranges, n=3 per condition.
- The M5's figures for the same workload, resident and streamed at 815 slots, are reported beside them,
  not compared with them.

## C6. The falsifier's comparison

The 27B on the owner's 36 GB M4 Max, with `ds4-engine`, on the same request (n=3, greedy, `max_tokens`
512). The design is falsified if the iMac's median streamed MXFP4 decode t/s from C4, apps open, is not
above the 27B's median.

## What would falsify this preregistration's design

- C3 passes, but C4's apps-open runs swap. Then the memory plan was wrong, not the kernels.
- C2 picks 2 GiB or less. Decode on the iMac then depends on the SSD, which C4's single request may hide.

## Risks

- One request, n=3. A Pi session sends 10–20 requests, growing to about 33 KB.
- Safari's and PyCharm's memory varies with their state. The load is scripted, but not byte-identical
  between runs.
- macOS 27.1 on the iMac against 27.2 on the M5. The Metal compiler differs.
- The 0.85 margin is judgment, not derived.
