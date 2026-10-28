# S3 scheduling assessment: generated locality against measured read service

Date: 2026-10-28. Branch `codex/mellum-ssd-replay`. Inputs:
[generated Phase 1 -02 locality](2026-10-28-mellum-generated-phase1-02-evidence.md)
and [storage-service read cost](2026-10-27-mellum-storage-service-evidence.md).

**Honesty header.** Measured: miss demand under one frozen policy (grouped
global LRU, 3 GiB, 489 slots, cold start) for **one** generated conversation
(n=1, 5120 decode tokens, all thinking-phase), and one-worker native read
service for **three** 512-expert repetitions of one aligned packed layout on
this 128 GiB M5 Max. In those repetitions, device reads were 100.09% of
logical bytes, but `physical_io_verified=false`. Conjecture: everything
that multiplies the two. That arithmetic assumes the measured service time
holds for these misses, in this order, at this queue depth. No streamed
inference ran, and no stall, generation rate or 8 GB result is measured.

## Assumptions

1. **Atomic group residency.** A layer computes only once all 8 routed experts
   are resident; misses are not split or reordered (the LRU's own schedule).
2. **On demand.** A layer's routing is known only after its router runs, so a
   miss cannot be fetched earlier without prediction. No prefetch is assumed.
3. **One worker, serial.** Each miss costs the measured median service
   0.655 ms (repetitions 1–2; 0.658 ms in repetition 0). Queueing, concurrency
   and recovery overhead are excluded.
4. **Read time adds to compute.** With compute `Tc` per token, token time is
   `Tc + reads`. `Tc` for resident Mellum decode without the recorder is not
   measured on this branch.

## Results

Decode, 5120 tokens: mean 56.54 expert misses per token, 2.02 per layer.
No token was all-hit. On 837 tokens (16.3%), all 28 layers had at least one
miss.

| Decode token | Misses | Serial reads at 0.655 ms |
|---|---:|---:|
| Mean | 56.5 | 37.0 ms |
| p50 | 53 | 34.7 ms |
| p95 | 102 | 66.8 ms |
| Worst (first token of response0) | 182 | 119.2 ms |

Serial reads exceed 50 ms on 915 tokens (17.9%), 66.7 ms on 259 (5.1%) and
100 ms on 20 (0.39%). The worst tokens include the first decode token after
each prefill: 182 misses at position 1950, 175 at 6120.

At the mean, the arithmetic gives `1000 / (Tc + 37.0)` tok/s:

- 27.0 tok/s at `Tc` = 0;
- 16.1 tok/s at 25 ms;
- 11.5 tok/s at 50 ms;
- 10.0 tok/s at 63 ms.

The mean only exhausts a 100 ms token by itself if service rises to 1.77 ms
per expert, 2.7 times the measured median. Prefill reads: initial 3484 misses
(2.28 s serial for 1950 tokens); append 1259 (0.82 s for 74 tokens).

The conversation touched 1782 of 1792 experts (99.4%), 10.9 GiB, 3.6 times the
cache.

## What this permits and rules out

**Ruled out, for this trace at 3 GiB:**

- **A warm-cache all-hit regime.** Any schedule assuming misses become rare
  after warmup is ruled out. No token was all-hit, a quarter of expert
  references miss, and the working set is nearly the whole expert set.
- **Per-token deadlines under on-demand serial one-worker reads.** Every
  token under 100 ms is ruled out by reads alone, at any `Tc`: 20 tokens
  exceed 100 ms with `Tc` = 0. At a plausible nonzero `Tc` the share is
  larger: at `Tc` = 50 ms, 17.9% of tokens miss the deadline.

**Not ruled out:**

- A mean of 10 tok/s under on-demand serial one-worker reads, *if* resident
  `Tc` ≤ about 63 ms/token and the measured service holds on misses.
- Concurrency, overlap with compute, or within-layer read parallelism could
  each lower the read term. None is measured, so none is credited here.

**Not assessable from S3:**

- An 8 GB machine: its SSD, its memory pressure, and whether 3 GiB of
  expert cache fits beside fixed tensors and session state.
- Answer-phase, tool-result and medium/8K locality.
- Cold physical reads beyond the storage-service classifier.
- Original GGUF gate/up/down range cost.

## Consequences

- **S4.** Miss recovery is the common path, not an exception: about two per
  layer per token, with no all-hit tokens. S4's proof should exercise it
  under that rate, not only in forced-miss tests.
- **Resident `Tc` is the missing input.** It decides whether 10 tok/s is
  plausible under the serial class above. Measuring it without recorder
  overhead is cheap next to SSD work.
- **The worst case is after a prefill.** First-token bursts (175–182 misses)
  suggest prefill-end or turn-start prefetch as an S5 candidate. That is a
  hypothesis, not a result.
- **S3's exit is partly met.** Read service and locality are both measured
  and combined under stated assumptions. A miss *curve* (more than one
  capacity) and physical cold-read cost are not. A capacity sweep on the
  held-out trace would be descriptive; the design forbids tuning from it, so
  it must not pick a capacity or policy.

Recompute the table from saved artifacts only:

```bash
python3 -I -c "import json;t=json.load(open('/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-generated-phase1-02/validated/result/artifact/analysis.json'))['token_series'];L=sorted(x['demand_loads'] for x in t);n=len(L);print(n,sum(L)/n,L[n//2],L[int(.95*n)],L[-1],sum(l*.655>100 for l in L))"
```

## What would falsify this

- **Resident `Tc` above about 63 ms/token.** That would rule out a 10 tok/s
  mean under the serial class, regardless of SSD.
- **Higher service times on misses.** Storage-observed cold reads of real
  miss streams averaging above about 1.77 ms/expert would make reads alone
  exhaust the mean token budget.
- **Better locality in another conversation.** An answer-phase or tool-result
  conversation with materially fewer misses per token would make this trace
  pessimistic. Worse locality would make it optimistic.

## Risks

n=1 locality, thinking-only, from one task. The read service is one layout,
one worker, uniform-order 512-expert batches, not the trace's miss order.
Adding medians ignores tail service (p99 up to 0.744 ms, max about 3.3 ms
per expert), so serial worst cases are understated. `Tc` is unknown, and
every rate above is conditional on it.
