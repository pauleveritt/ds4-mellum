# Where a streamed decode stop's time goes on the 8 GB iMac (screen)

Date: 2026-10-29. A screen, not a preregistered run (`docs/sdd.md`, "Evals"). Kit, captures and the M5 dry
run: [`2026-10-29-mellum-imac-stop-profile/`](2026-10-29-mellum-imac-stop-profile/README.md). Figures from
`python3 -I docs/superpowers/research/2026-10-29-mellum-imac-stop-profile/analyze.py docs/superpowers/research/2026-10-29-mellum-imac-stop-profile/imac`
(saved as `imac/analysis.txt`).

**Honesty header.**
- **Measured:** the 8 GB M1 iMac, build `69492de` plus the screen-only `profile.patch`, phase C's workload,
  MXFP4 streamed at 2 GiB (652 slots), apps closed, run over SSH 13:45-13:50. Two unprofiled and two profiled
  128-token runs, and one 64-token run with command-buffer timings.
- **Conjecture, marked where it appears:** what the per-round fixed GPU cost is made of, and what the levers
  would save.
- **Not measured:** other cache sizes, longer contexts, any change to the code.

## Results

The patch costs about 1%: unprofiled 178.0 and 178.1 ms/token, profiled 180.7 and 179.6. The first unprofiled
run grew swap-outs by 8,456; the other four by 0.

**Per decode token** (profiled runs, n=2; the reads run agrees):

| Part | ms/token | Share |
|---|---|---|
| GPU wait (`finish`: commit, execution, gate read) | 102.2, 101.3 | 56% |
| Encoding passes on the CPU | 50.1, 48.5 | 27% |
| Read wait left after the resumed pass is encoded | 25.2, 26.0 | 14% |
| Starting reads, and the rest | 5.0, 4.8 | 3% |

The routing is the M5's: 20.8 stops and 21.8 GPU rounds per token, a median of 2 experts per stop.

**One stop, about 6.4 ms**, which agrees with phase C's fit of 6.71 ms per stop:
- **GPU round, 3.7 ms median.** Fit (reads run, 1,405 rounds): 2.1 ms fixed per round plus 2.1 ms per
  executed layer; a round executes 1.28 layers on average. A decode command buffer spends a median 2.9 ms on
  the GPU of 3.7 ms from commit to done (M5: 0.6 of 0.8).
- **Reads and the resumed pass, about 2.7 ms.** The read start (0.1 ms), the resumed pass's encode (1.9-2.0
  ms, which overlaps the reads and is needed either way) and the remaining read wait (0.7-0.8 ms median). A read of 1 expert (3.3 MB)
  spans 2.3 ms, about 1.4 GB/s; of 2, 3.4 ms; of 8, 8.5 ms, about 3.1 GB/s. These are the SSD's: 8 GB cannot
  cache the 7 GB file. (The M5's 6-11 GB/s came from its page cache.)

**Why encoding costs so much.** Each pass is encoded from its layer to the head, and the gate suppresses
what follows the next stop. That is 335 layers encoded per token for 28 executed. On the M5 that costs
0.24 ms per pass; here 2.0 ms. But after a stop the encode runs while the reads are in flight, and it is
shorter than they are, so cutting it would mostly expose read wait.

## What this says about the levers (conjecture)

1. **Prefetch the next layer's experts, to avoid stops.** Pre-gating already predicts 7.0 of the next layer's
   8 experts, all 8 in 45.7% of layer pairs (the [M5 screen](2026-10-29-mellum-stream-decode.md)). Hiding a
   stop's reads alone saves only the exposed wait, about 1.2 ms per stop on average (25-26 ms per token),
   because the resumed pass must be encoded either way. A prefetch that lands before the GPU reaches the next
   layer's check avoids that stop entirely, about 6.4 ms. That is S5 step 4 in the backlog; see the
   [prefetch design](../specs/2026-10-29-mellum-stream-prefetch-design.md).

   **Corrected 2026-10-29, after writing that design.** This item first read: hiding the reads "could remove
   most of the ~2.7 ms per stop, up to about 55 ms of 180 per token". That counted the resumed pass's encode
   (about 2 ms per stop) as read time, though it is on the critical path with or without reads.
2. **The fixed cost of a GPU round, 2.1 ms × 21.8 rounds, about 45 ms per token.** Its parts are not split:
   suppressed-layer dispatches, command-buffer latency (0.8 ms between commit-to-done and GPU span) and the
   gate read. Shorter passes cut the suppressed dispatches; windowed passes lost on the M5, whose costs are
   balanced differently.
3. **Fewer stops** (S5 step 3) save about 6.4 ms each, on both of the above.
4. **Faster encoding** pays only once reads stop being the critical path.

Executed layers cost 2.1 ms each, about 59 ms per token, roughly twice the bandwidth estimate of 27 ms in the
phase C answer. So a fully stop-free iMac would still not reach that estimate; that gap is not explained.

## What would falsify this

- A rerun at 2 GiB whose per-token split moves by more than run-to-run spread (here about 1%).
- Command-buffer GPU spans that do not grow with executed layers, which would make the round fit wrong.
- A prefetch screen on the iMac that hides reads yet saves far less than the read wait it removed.

## Risks

- One machine, one prompt, n=2 per arm; 128 tokens, not phase C's 512 (178 against 171 ms/token).
- The profile times callbacks with the CPU clock; the GPU split comes only from `DS4_METAL_CB_TIMES`.
- The engine's pread profile prints nothing on the asynchronous read path, so a read's span is inferred as
  start + encode + wait.
