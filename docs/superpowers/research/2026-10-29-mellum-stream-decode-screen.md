# Mellum streamed decode: where a token's time goes (screen)

Date: 2026-10-29. A screen, not a preregistered run (`docs/sdd.md`, "Evals").

**Honesty header.**
- **Measured:** two 128-token runs on one M5 Max (128 GB, macOS 27.2). MXFP4 Mellum 2.1, streamed at
  2.5 GiB (815 slots), on S6 phase C's flattened workload prompt (2,686 tokens). The build is
  `research/mellum-stream-decode` at `42f51b7` plus the uncommitted timing patch kept here (`profile.patch`).
- **Conjecture:** the split of the GPU wait into fixed, executed-layer and suppressed-layer costs comes from a
  linear fit, not from GPU timestamps.
- **Not measured:** an 8 GB Mac, Q8_0 experts, other cache sizes.

## Why

On the M5 the phase C workload decodes at 143 t/s resident and at 23–24 t/s streamed at 2.5 GiB. That is
about 22.6 expert loads, or 74 MB, per token (`2026-10-28-mellum-s6-c-evidence/workload-m5/`, n=3). At SSD
speed 74 MB should cost far less than the 35 ms per token the streamed path adds.

## Method

`profile.patch` times the three callbacks of `ds4_mellum_stream_decode`:
- `encode`: CPU encoding from a layer to the head;
- `finish`: commit, wait for the GPU, read the gate;
- `load`: `mellum_stream_load_ids`, the synchronous expert reads.

With `DS4_MELLUM_STREAM_PROFILE_ROUNDS` it also logs every round's first layer, its stop layer and its
`finish` time. `regress.py` fits `finish = fixed + a·(layers executed) + b·(layers suppressed after the stop)`.
`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY` adds the engine's own read timings.

```
tools/mellum_s4_measure --model MX --prompt …/workload-prompt.txt --gen 128 --context 32768 --stream-bytes 2684354560
uv run --with numpy python -I regress.py b.rounds.txt
```

## Results

| Per decode token (ms) | Run a | Run b |
|---|---|---|
| Total | 43.7 | 37.8 |
| Encode | 5.6 | 4.8 |
| Finish (GPU wait) | 23.9 | 19.5 |
| Load (reads) | 14.2 | 13.2 |

- **Rounds:** 2,243 encodes for 128 tokens, about 17.5 per token, for 28 layers.
- **Reads:** the reads go through a parallel `pread` pool and average 0.91 ms per load call (run a).
- **Run b's fit** (2,226 rounds after the first token, r² 0.78):
  - fixed cost per round 0.26 ms;
  - 0.31 ms per executed layer;
  - 0.025 ms per suppressed layer;
  - each round executes 1.6 layers on average, and a stop leaves 14.7 suppressed layers behind it.
- **Run-to-run spread:** a third run, of a build that had failed to rebuild and so was run a's binary, gave
  36.9 ms per token. Its files were deleted by mistake (only its summary line survives, in the session log).
  The spread of 37–44 ms on one command means an A/B must interleave arms and carry its `n`.

A 128-token run's median is 41.1 ms (run a's `measure.json`); the phase C reference's 512-token runs
average about 42.

### Second screen (after Fable's review)

- **The floor, run c.** At 6 GiB, 1,958 slots, every expert fits. A 128-token run took 129 encodes and one
  load call, and decoded at 7.95 ms per token: encode 0.39, finish 7.54. Resident is 7.0–7.2. So the cost
  that does not depend on stops is about 0.9 ms per token.
- **Reads and command buffers, run d.** A 64-token run with `DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE` and
  `DS4_METAL_CB_TIMES`; `summarize_d.py d.pread-cb.txt`:

| Experts per read | n | Median wall (ms) | GB/s | Threads |
|---|---|---|---|---|
| 1 | 595 | 0.545 | 6.0 | 3 |
| 2 | 298 | 0.789 | 8.3 | 6 |
| 3 | 125 | 1.037 | 9.5 | 9 |
| 8 | 221 | 2.208 | 11.9 | 9 |

  The prefill's loads are in these counts. A one- or two-expert read, the usual decode stop, gets three to six
  threads, one per tensor, and half the device's rate.
- **Command buffers:** 1,248 of them, with a median gap of 0.2 ms between commit-to-done and GPU span.
- **Load time beyond `pread`:** a decode load call averaged about 0.88 ms against a `pread` wall of about 0.6
  ms (run d), so cache bookkeeping, slot preparation and invalidation cost about 0.3 ms per stop. Run a's
  timing summary puts the table scans alone at about 1.1 ms per token.

### Review

A fresh Fable review (no code run) reached the same split, from the phase C evidence and the S4 decode
profile spike (`2026-10-28-mellum-s4-decode-profile-spike.md`), and ranked these levers:
1. encode the resumed pass while the reads are in flight;
2. bound the cache's victim scans to Mellum's 28 × 64 entries, not the table's 80 × 384;
3. split each tensor's read into chunks, for queue depth;
4. windowed command buffers, so a stop suppresses at most a few layers;
5. prefetching the previous token's experts, screened first on the routing-trace replay.

Its regression intercept of 5–6 ms per token of stop-independent cost is contradicted by run c's 0.9 ms.

## Reading

Per token:
- the 28 real layers cost about 9 ms;
- on top of them: reads about 13 ms, the suppressed tail about 6.5 ms (17.5 × 14.7 × 0.025), fixed
  round cost about 4.5 ms, and re-encoding about 4 ms.

Every one of these scales with the number of stops, and none overlaps with GPU work. So the levers are:
- **fewer stops:** load more per stop, or prefetch ahead;
- **cheaper stops:** encode less past the point where a stop is likely;
- **hidden reads:** overlap them with GPU work.

## What would falsify this

- Timestamps from inside the GPU showing that the suppressed dispatches cost nothing. Then the 0.025 ms per
  suppressed layer is a fitting artefact absorbed from the fixed term.
- A run where removing the reads (an all-resident cache with streaming on) leaves the per-token time near 38
  ms.

## Risks

- n=2 runs of 128 tokens on one machine.
- The fit's terms are correlated: the rounds that execute more layers suppress fewer.
- The timing patch adds a clock read per callback (negligible against milliseconds, but not measured).
