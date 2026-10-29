# Streamed Mellum decode: cheaper stops (result)

Date: 2026-10-29. Branch `research/mellum-stream-decode`. Preregistration:
[`2026-10-29-mellum-stream-decode-prereg.md`](2026-10-29-mellum-stream-decode-prereg.md). Screen:
[`2026-10-29-mellum-stream-decode-screen.md`](2026-10-29-mellum-stream-decode-screen.md).

**Honesty header.**
- **Measured:** one M5 Max (128 GB, macOS 27.2, on AC power). MXFP4 Mellum 2.1, phase C's flattened workload
  (2,686 tokens, then 512 greedy). The verdict is the preregistered A/B, n=3 per arm. The candidates were
  screened at n=1–2 on 128 tokens.
- **Not measured:**
  - the 8 GB iMac, the M1 Pro, Q8_0 speed, and long sessions;
  - why the overlap saves more GPU-wait time than the encode it hides (below; a conjecture).
- **Decode's arithmetic is unchanged:** every streamed run gave the resident run's 512 token ids, and the
  stream tests give 0 differing logit words on both expert files.

## Verdict (preregistered)

The A/B ran twice:
- **First:** B at `e31a071`, before the review.
- **Final:** B at `dd0b6f4`, after the review's two fixes. The preregistration's B is the head the night ends
  on, so the final run decides. The first is kept beside it.

| Arm | ms/token, median of 3 (range) | t/s | Loads/token | Stops/token |
|---|---|---|---|---|
| A: `42f51b7` (phase C kit + SSD tip), final run | 44.76 (41.84–45.98) | 22.3 | 22.62 | 15.05 |
| **B: `dd0b6f4`, final run** | **24.69 (24.61–24.73)** | **40.5** | 22.62 | 15.05 |
| A, first run | 43.52 (40.66–45.59) | 23.0 | 22.62 | 15.05 |
| B: `e31a071`, first run | 25.27 (25.19–25.36) | 39.6 | 22.62 | 15.05 |

- **Rule 1, identity:** pass in both runs. Every run's 512 ids are equal: six streamed runs at 2.5 GiB, the
  resident run, and both 3 GiB runs.
- **Rule 2, speed:** pass. B/A = 0.552 in the final run, 0.581 in the first; the bar is 0.85.
- **Rule 3, tests:** pass at both heads. `make test-mellum`, `make test-mellum-kernels`, and
  `ds4_test --mellum-stream-identity/-prefill/-faults` on the MXFP4 and the Q8_0 files; at the final head,
  also the review's new `--mellum-stream-slabs`.
- **Not judged (final run):**
  - at 3 GiB (979 slots), A 32.00 and B 18.82 ms/token (31.2 and 53.1 t/s), n=1 each;
  - B resident, 7.00 ms (142.9 t/s);
  - time to first token, 1.59–1.64 s streamed (one A run 1.40);
  - Metal peak unchanged (4.37 GiB at 2.5 GiB).
- B's three runs span 0.12 ms; A's span 4.1 ms.

Recompute: `python3 -I 2026-10-29-mellum-stream-decode-ab/analyze.py 2026-10-29-mellum-stream-decode-ab/final`
(drop `/final` for the first run). To rerun, see `run_ab.sh`.

## What landed

1. **The resumed pass is encoded while its stop's reads run** (`dfca52a` test, `ed43bc2` loop, `cda51b6`
   engine).
   - `ds4_mellum_stream_ops` gains optional `load_begin` and `discard` ops. A stop begins its reads through
     the existing early-load path, encodes the resumed pass, and finishes the reads before the commit.
   - The layer's selected experts are protected from that path's evictions.
   - A failed load or a cancel after the encode drops the pass uncommitted (`ds4_gpu_discard_commands`).
   - `DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD` turns the overlap off. It gives load-then-encode
     again only since the review fix (`a946e4d`). Before it, it gave encode-then-load, a new order, and that is
     what the screen's "off" arm ran. The A/B's arm A is the old code itself, so the verdict is unaffected.
   - **Screen** (128 tokens, n=2 per arm, alternated): 39.0 → 26.9 ms/token.
   - **Profiled** (64 tokens, n=1 per arm), the gain splits into load wait −4.9 ms (the hidden encode) and GPU
     wait −3.6 ms. The GPU wait falls although the GPU work is identical. Conjecture, not measured: with
     shorter idle gaps between passes the GPU keeps its clocks up. B's narrow spread fits that reading.
2. **The cache's victim scans stop at the highest installed layer and expert** (`e31a071`).
   - Mellum's live area is 28 × 64 of the 80 × 384 table: 1,759 entries scanned instead of 30,720.
   - Each scan drops from 0.056 to 0.016 ms. The victims are the same: loads, stops and ids are identical.
   - 26.8 → 26.3 ms/token (n=2 per arm).
   - No new unit test, because nothing observable changed. The evidence is the identical victim choice.

## Review

A fresh Opus review of `42f51b7..e31a071` (code read, nothing run) found no Critical issues and two Important
ones. Both are fixed, each with a test that failed first.
- **Slab residency.** When `load_begin` started no reads, the stop's experts were loaded after the resumed
  pass was encoded. A slab that load allocated was never made resident for the pass. This happens when early
  load is off, there is no capacity, or a stale pending load matched.
  - Reproduced: `--mellum-stream-slabs` (early load off, 16 MiB slabs) gave all 8,650,752 logit words
    different at `cda51b6`, against 0 at `42f51b7`.
  - Fixed in `a946e4d`: when no reads were started, `load_begin` makes the whole load before the encode, and
    it finishes a stale pending load first.
- **A failed first-pass encode was committed** by the caller's `end_commands`. This predates the night's
  work. Fixed in `dd0b6f4`: with `discard` set, a failed encode is dropped. The review's missing test for an
  encode failing after `load_begin` was added with it.
- **Minor, deferred:**
  - `wait_inflight` can flush the open batch (latent: Mellum never marks entries in flight);
  - victims are now chosen before `load` updates hotness (performance only; the stopped layer's experts are
    protected either way);
  - `discard` leaves a timeline record and transient buffers to the next finish.

## Screened and not landed

Each is kept, with its patch and captures, under the screen's evidence directory.
- **Chunked reads.** Each tensor's read is split into 2 or 4 page-aligned tasks for queue depth. 4 chunks:
  30.1–30.5 ms/token; 2 chunks: 27.4; whole tensors: 26.7–26.8 (n=2, 1, 2). Deeper queues lose on this
  machine. `chunks/`.
- **Windowed passes** (`ddeb386`, `e8af156`, reverted in `b49261e`). At most W layers per pass, so a stop
  suppresses fewer layers:
  - W=3: 31.0–32.8 ms/token;
  - W=4: 29.5–31.0;
  - W=6: 25.5–27.3;
  - W=28 (whole token): 26.8–26.9;
  - n=2 each.
  The screen's model (`window_model.py`) predicted W=3–4 best by about 4.9 ms. A window boundary without a
  stop costs more than the fitted round cost. `window/`.
- **Synchronous next-layer prefetch.**
  - **Accuracy screen** (`pregate_screen.py`, 64 positions of the 2,200-token oracle prompt, 1,728
    layer pairs): layer L's FFN input through layer L+1's router predicts 7.01 of L+1's 8 experts on
    average. All 8 are right in 45.7% of pairs; two layers ahead, 6.65.
  - **Speed screen:** loading the predicted misses at each stop, before the commit, cut stops per token from
    16.5 to 12.8 and demand loads from 28.1 to 18.8. But it added 13.9 prefetched experts per token, serially:
    28.6–28.9 ms/token against 26.8–27.1 (n=2).
  - **An asynchronous form** might pay: the GPU would wait on a shared event before layer L+1's check while
    the reads run. It was not tried. `prefetch/`.

## Where the time goes now (B, inferred)

About 25 ms per token at 2.5 GiB:
- about 9 ms of real layer work;
- about 15 stops, each still paying a fixed round trip (about 0.26 ms), the suppressed tail after it, and
  whatever part of its reads the encode does not hide.

The floor with every expert resident is 7.0 ms resident and 7.95 streamed at 6 GiB (n=1). Fewer stops,
through the cache policy (S5 step 3) or an asynchronous prefetch, are what is left.

## What would falsify this

- On another machine, especially the M1 iMac with its slower SSD and GPU, B is no faster than A. Then the
  gain is specific to how fast the M5's encode and reads are relative to each other.
- A run where B's token ids differ from A's: the overlap would then have changed what a pass reads.
- A GPU timestamp trace showing that the GPU-wait drop came from something other than shorter idle gaps. That
  would correct the conjecture, not the measured gain.

## Risks

- One prompt, one machine, n=3 per arm. The A arm's spread (40.7–45.6 ms) is wide.
- The overlap leans on the early-load path, shared with GLM and DeepSeek. Only Mellum calls it this way, under
  a Mellum protect list.
- An encode failure after `load_begin` leaves the early load pending. The next `load_begin` finishes it
  first. The protocol tests cover a failed encode; the real-model fault tests cover a failed load and a cancel.
- The owner had set load overlap aside during S4 (ROADMAP, S4). This work was done on the owner's
  instruction of 2026-10-29 to act on a fresh review's ideas overnight.
