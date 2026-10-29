# S6 phase C: preregistration (the 8 GB M1 iMac)

Date: 2026-10-28. Committed before any iMac run. **Amended 2026-10-28 23:56 (the time first written as 23:58 and
corrected in `e44dff2` to the commit's), still before any iMac run:**
`ds4-server` refuses Mellum ("Mellum 2.1 is not served yet: ds4-server has no Hermes tool-call syntax";
`2026-10-28-mellum-s6-c-evidence/workload-m5/server-*.log`). So the workload runs through
`tools/mellum_s4_measure` as one flattened user turn, not as a chat request replayed through the server. The
items that changed are marked (amended). Plan:
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
- **Build:** the kit's named commit. The M5 references were made at `471b902` (decode logits) and `614eb33`
  (workload); no engine, Metal or driver source changed after `471b902` (`git diff --stat 471b902 HEAD --
  '*.c' '*.m' '*.h' '*.metal'` lists only the tool and test sources of phase C's agent tasks).
- **Correctness reference:** the M5 Max's resident MXFP4 decode-mode logits at the oracle positions of
  `tests/mellum_oracle_{short,1100,2200}.tokens`. Their SHA-256s are in
  `2026-10-28-mellum-s6-c-evidence/m5-reference-sha256.txt`. On the M5, streamed equals resident byte for
  byte at all 157 positions.
- **Workload (amended):**
  - The source is `2026-10-28-mellum-s6-c-prereg/workload-request.json` (SHA-256 `73cd5f77…`), request #4
    of a mellum-worker Pi run: system prompt, user task, four tool rounds of file reads, and 7 tools.
  - `flatten.py` (`19134e5d…`) turns it into `workload-prompt.txt` (`2f34a817…`, 8,770 B, 2,686 tokens
    rendered).
  - It runs as `tools/mellum_s4_measure --prompt workload-prompt.txt --gen 512 --context 32768`, greedy,
    with `--stream-bytes` set to the cache.
  - The M5 reference, n=3 per arm, is in `2026-10-28-mellum-s6-c-evidence/workload-m5/`:
    - resident: time to first token 0.95 s, decode 143 t/s;
    - streamed at 2.5 GiB: time to first token 2.2–2.6 s, decode 23–24 t/s, 22.6 loads/token.
- **Background load:** `tools/mellum_background_load.sh PROJECT urls.txt`, with PyCharm on a small
  Python project (`../mellum-demo` unless the owner names another before the run) and Safari with the
  ten URLs in `urls.txt`.

## C1. Bring-up (gate)

- **Pass:** `make test-mellum` and `make test-mellum-kernels` report 0 failures; every MXFP4 line runs.
- **Recorded, not gated (amended 2026-10-29, before any iMac run):** `ds4_test --mellum-stream-identity`,
  `--mellum-stream-prefill` and `--mellum-stream-faults` on the MXFP4 file, each bracketed by `vm_stat`.
  - They build a resident session (about 6.5 GiB of weights), and identity also builds a 6 GiB streamed
    cache. So on 8 GB they can fail or swap for memory, not correctness.
  - A pass is reported. A failure is read with its `vm_stat` and log. C3 is the correctness gate on this
    machine.
- **Recorded:** `tools/mellum_metal_limit` (recommendedMaxWorkingSetSize, families, Metal 4), the machine
  facts, and the engine's startup lines.
- **A failure stops phase C.** The next step is diagnosis, not timing.

## C2. Choosing the cache (before any timed run)

Candidates in order: 3 GiB (979 slots), 2.5 GiB (815), 2 GiB (652). With the apps closed, run the workload
once at each candidate with `tools/mellum_s4_measure --stream-bytes` (amended: the measure tool, not the
server).

The chosen cache is the first candidate where both hold:
- the run's `metal_allocated_peak` is at most recommendedMaxWorkingSetSize − 1 GiB (amended: it was the
  server's planned-memory line);
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

At the chosen cache, three workload runs with the apps closed, then the background load, then three runs
with it running. `vm_stat` is sampled before and after each run. The order and the 20 s rests are as in
`workload-m5/run.sh` (amended: process runs, not server replays).

**Pass, both:**
- swap-outs grow by 0 during every apps-open run;
- the median apps-open decode t/s is at least 0.85 of the median apps-closed decode t/s.

## C5. Speed (Goal 5, reported only)

- From C4: time to first token and decode t/s, as medians with ranges, n=3 per condition.
- The M5's figures for the same workload, resident and streamed at 815 slots, are reported beside them,
  not compared with them.

## C6. The falsifier's comparison

The 27B on the owner's 36 GB M4 Max, with `ds4-engine`, on the same flattened prompt as one user turn (n=3,
greedy, 512 tokens; amended).
- **Metric, both sides:** decode t/s = (generated tokens − 1) / (seconds from the first generated token to
  the last). On the iMac it comes from `measure.json`'s per-token `ns`, tokens 1 to 511.
- **The 27B's exact invocation is not fixed yet (amended 2026-10-29).** The owner fixes it in a dated
  addendum before C6 runs, with this metric and settings. C6 is not run without it. The design is falsified if the iMac's median streamed MXFP4 decode t/s from C4, apps open, is not
above the 27B's median.

## What would falsify this preregistration's design

- C3 passes, but C4's apps-open runs swap. Then the memory plan was wrong, not the kernels.
- C2 picks 2 GiB or less. Decode on the iMac then depends on the SSD, which C4's single request may hide.

## Risks

- One request, flattened (amended).
  - The chat template's tool section and the tool-call syntax differ from the worker's real request.
  - `flatten.py` drops the four assistant turns' `reasoning_content`: 4,077 characters, where the request
    sets `preserve_thinking`. It also drops `tool_call_id` and `chat_template_kwargs`.
  - So the workload is about a third shorter in conversation text than the real request (2,686 tokens
    rendered).
  - n=3. A Pi session sends 10–20 requests, growing to about 33 KB.
- Safari's and PyCharm's memory varies with their state. The load is scripted, but not byte-identical
  between runs.
- macOS 27.1 on the iMac against 27.2 on the M5. The Metal compiler differs.
- The 0.85 margin is judgment, not derived.
