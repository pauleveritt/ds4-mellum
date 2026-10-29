# Mellum streamed decode: asynchronous expert prefetch and the 2.5 GiB fit (S5 step 4, S6 phase D)

Date: 2026-10-29. Status: design, written on the owner's go-ahead; a builder check's three questions are settled as designer's calls, marked below.

**Amended 2026-10-29, after Fable's review and before any phase D work.**
- **An avoided stop is worth about 5.1 ms, not 6.4.** The stop profile's two-term fit put the 7.8 ms output head
  into each round's fixed cost; a four-term refit gives 1.0-1.2 ms fixed per round, 2.0-2.1 ms per executed
  layer, 0.04 ms per suppressed layer (n=2 runs plus the reads run; r² 0.39-0.74). An avoided stop removes one
  round's fixed cost (1.1), its 14.7 suppressed dispatches (0.6), the resumed pass's encode (2.1), the mean read
  wait (1.2) and the read start (0.1). Below, "6.4 ms" reads as 5.1, and the screen's bar is the plan's: the
  measured ms/token change, or a projection that subtracts the event wait and wasted reads.
- **The "existing protect mechanism" does not suffice.** It holds one layer, the read pool takes one batch at a
  time, and the pending load is one global. The plan's Task 5 adds a second in-flight read, a queueing pool whose
  completion signals the event, and a two-layer protect list, under stated install invariants.

## Why

Streamed MXFP4 on the 8 GB M1 iMac decodes at 5.85 t/s (n=3, 2 GiB cache;
[phase C](../research/2026-10-28-mellum-s6-c-imac.md)). The owner asked for the low-hanging fruit only: what
can be aimed for "without chasing weird ideas". The [stop profile](../research/2026-10-29-mellum-imac-stop-profile.md)
(n=2) splits one stop on the iMac into about 6.4 ms:

| Part of a stop | ms | Notes |
|---|---|---|
| GPU round (`finish`) | 3.7 | 2.1 ms fixed per round plus 2.1 ms per executed layer (fit) |
| Encoding the resumed pass | 1.9-2.0 | runs while the stop's reads are in flight |
| Read wait after that encode | 0.7-0.8 median, 1.2 mean | the only part of the reads still exposed |
| Starting the reads | 0.1 | |

There are 20.8 stops per token. A stop costs about 6.4 ms whether or not its reads are slow, so on this
machine **avoiding a stop is worth far more than hiding its reads**. (The stop profile's note named hiding
reads as the larger lever, up to about 55 ms per token; that over-counted. The encode is on the critical path
either way, so hiding reads alone saves at most the exposed wait, about 25 ms per token.)

Two levers are low-hanging:
1. **Prefetch the next layer's experts.** Layer L's FFN input through layer L+1's router predicts 7.01 of
   L+1's 8 experts, and all 8 in 45.7% of layer pairs (M5 screen, 1,728 pairs; [stream decode](../research/2026-10-29-mellum-stream-decode.md)).
   On the M5, a synchronous form (read the predicted misses before committing the resumed pass) cut stops per
   token from 16.5 to 12.8 but lost overall, because there a stop costs well under 1 ms and the extra 13.9
   reads per token ran serially. On the iMac a stop costs 6.4 ms. An asynchronous form, where the GPU
   waits on an event before layer L+1's check while the reads run behind layer L's work, was named there and
   not tried.
2. **Fit the 2.5 GiB cache.** Its Metal peak, 4,677,697,536 B, was 24.8 MB over phase C's rule
   (recommendedMaxWorkingSetSize − 1 GiB = 4,652,892,160 B). At 2.5 GiB the iMac ran 7.31 t/s (n=1 in each of
   two stages) with 15.07 stops per token instead of 19.98.

## Goals

1. **Aim: 9 t/s on the iMac** (about 110 ms per token), phase C's workload, apps closed, MXFP4, at the
   largest cache phase C's C2 rule allows. An aim, not a pass rule: the result is reported either way, with
   its n.
2. **Prefetch earns its place:** on the iMac, at the cache the C2 rule picks when the run happens (2.5 GiB
   if Goal 3 lands, else 2 GiB), prefetch on saves at least 20 ms per token against prefetch off, same build
   (designer's call: the cache users get decides, not the one with the most stops to save). Below that,
   prefetch stays off and the work stops at a written verdict, without tuning.
3. **The 2.5 GiB cache fits at 32K context** under the unchanged C2 rule, found in buffers (owner's choice,
   2026-10-29). Fallback, only with the owner's approval: a smaller default context for streamed decode on
   machines where the cache would otherwise drop.
4. **Exact.** Prefetch decides only what is read early, never what is computed: the router and the gate
   still choose the experts. Streamed ids equal resident ids on the M5 (S4's rule), and the iMac's ids equal
   phase C's at the same cache.
5. **No regression elsewhere.** Prefetch runs only with `--ssd-streaming`. Resident decode is byte-identical
   and within noise. Streamed Q8_0 and MXFP4 on the M5 are not slower, by S4's A/B (n=6).
6. **Shared code.** The prefetch protocol lives in the model-free driver (`ds4_mellum_stream.c`), so Q8_0
   (S5's 16 GB path) gets it with no separate implementation, and CPU tests can pin it.

## Non-goals

- A target for Q8_0, the M1 Pro or the M5. The M5 is a no-regression guard. When S5 brings up the M1 Pro,
  prefetch is measured there under S5's own 15 t/s target. (Its 32 GB can hold the Q8 file in the page
  cache, which would flatter prefetch.)
- The fixed cost of a GPU round, the M1's per-layer kernel cost, faster encoding, windowed passes, cache
  eviction policy (S5 step 3). These are the "weird ideas" the owner ruled out for now.
- New Metal kernels, or a different router or gate.
- Answer quality.

## Anti-goals

- A prefetch that changes any token id.
- A prefetch that evicts an expert the current layer still needs.
- Prefetch work, threads or events when streaming is off.
- A speed claim taken on the M5 and presented as the iMac's.
- A smaller default context without the owner's say.

## Constraints

- ds4-mellum's working rules: one concern per commit, the failing behavioral test committed before its
  fix, about 150 product lines per commit.
- Every rate carries its n. Screens run freely; the decision in Goal 1 and Goal 2 gets a preregistration.
- No `iogpu.wired_limit_mb` change.
- The iMac is reached over SSH only (`pauleveritt@herveline.local`), no desktop. Its runs use the phase C
  workload and `tools/mellum_s4_measure`.
- **Lands after S6's merge into `mellum`** (designer's call, following the owner's 2026-10-29 plan to merge
  S6 once phase C closed): the work is cut from `mellum`, and the plan's first task waits for that merge.
  Goal 4's iMac reference is then a fresh run of the merged build at the chosen cache, prefetch off, not
  phase C's `69492de`. The 2.5 GiB fit counts as S6 work: it changes phase C's C2 outcome, which S6's
  README rows report.

## The design, as decided

**Order (designer's call, after the profile):**
1. **The 2.5 GiB fit.** Independent of prefetch, already measured at +25%, and it sets the cache the rest
   is judged at.
2. **Screen, no product code.** Port the M5's synchronous prefetch hack
   (`2026-10-29-mellum-stream-decode-screen/prefetch/prefetch-hack.patch`, which no longer applies) and run it
   on the iMac at the chosen cache beside the stop profile. It measures stops avoided per token and the cost
   of the extra reads at iMac speeds. If stops avoided × 6.4 ms is under 20 ms per token, the asynchronous
   form cannot meet Goal 2, and the work stops at a verdict.
3. **Asynchronous prefetch,** only if the screen clears its bar.
4. **The preregistered iMac run** that decides Goals 1 and 2: prefetch off and on at the chosen cache,
   n=3 each, interleaved. The M5's Goal 5 A/B runs after it, as a check.

**The default (designer's call).** Prefetch ships behind its own switch under `--ssd-streaming`. It is on by
default only if Goal 2 passes on the iMac and Goal 5 passes on the M5. If the iMac gains and the M5 loses,
it stays opt-in and the Mellum README tells 8 GB users to turn it on. A rule that picks the default per
machine is a reopen condition, not part of this work.

**How the asynchronous prefetch works:**
- At a stop at layer L, the host has layer L's FFN input. It computes layer L+1's predicted top-8 on the
  CPU: the router is 64 rows of F32, a few microseconds, as the hack did.
- It starts the reads of L's missing experts (as now) and of L+1's predicted misses. A prefetch read must not
  evict L's experts or L+1's resident predicted ones (the cache's existing protect mechanism).
- The resumed pass is encoded with a GPU wait on a shared event placed before layer L+1's gate check. It is
  committed once L's own reads finish, as today. Layer L's experts then run on the GPU while L+1's reads
  complete.
- When L+1's reads finish, the host marks them resident and signals the event. The GPU's gate at L+1 then
  sees them, and when the prediction was complete, no stop happens at L+1.
- A wrong or partial prediction still stops at L+1, as today; it costs the time the GPU waited plus the
  wasted reads. The preregistered run measures that.
- Prior art for the event: ds4 already makes a command buffer wait on a CPU-signaled `MTLSharedEvent` in
  the middle of a batch (`ds4_metal.m:11163`, `encodeWaitForEvent`, for tensor parallelism).
- Only the next layer is prefetched (k = 1). Two layers ahead predicts 6.65 of 8; trying k = 2 is the
  builder's call after k = 1 is measured.

**Measurement:** the stop profile's patch and analysis (`2026-10-29-mellum-imac-stop-profile/`) are the
screen instrument, extended with prefetch counts (issued, used, wasted, stops avoided).

## Left to the builder

- Where in the encode the event wait goes, how the wait is split from the gate check, and which thread
  signals the event.
- The driver's new ops and their protocol tests: prefetch begin, the signal, and what a failed or cancelled
  prefetch does. A failed prefetch read must leave the gate to stop and load on demand, never fail the
  token.
- Where the 24.8 MB comes from (for example the 0.32 GiB of buffers, sized by `prefill_cap=1024`), with
  time to first token on the iMac no more than 10% worse (designer's call).
- The switch's name and form (an environment variable for the screens, then a flag), within the default rule
  above.

## What would falsify this

- The iMac screen shows stops avoided per token × 6.4 ms under 20 ms per token. Then Goal 2 cannot pass.
- Prefetch on changes any token id.
- The 2.5 GiB cache cannot be brought under the C2 limit at 32K without slowing prefill by more than 10%.
- Prefetch on the iMac saves the stops the screen predicts, yet the per-token time does not fall: the
  waits and wasted reads ate the gain.

## Risks

- The 45.7% "all 8 predicted" figure comes from 64 positions of one prompt on the M5; the iMac's workload
  may differ. The screen measures it there.
- Memory ordering between a CPU write to the residency table and the GPU's read after the event is assumed,
  not shown; S4's stream identity tests must cover it.
- Wasted prefetches use SSD time and slots. At 2 GiB the cache is small, so they may evict experts that a
  later layer needs.
- One prompt, one machine, n=3 for the decision.
