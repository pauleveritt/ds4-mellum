# SSD streaming deep review and retarget to 16 GB first

Date: 2026-10-28. Branch `codex/mellum-ssd-replay`. Inputs:
[S3 assessment](2026-10-28-mellum-s3-scheduling-assessment.md),
[-02 locality](2026-10-28-mellum-generated-phase1-02-evidence.md),
[storage service](2026-10-27-mellum-storage-service-evidence.md).

**Honesty header.** This note records an independent architecture review
(Fable, read-only) and the owner's decision that followed. Measured inputs:
- one generated conversation's routing (n=1, thinking-only), which is also
  the held-out evaluation trace;
- one-worker read service on this laptop's SSD;
- the engine's own memory plan.

Descriptive, not evaluation: the miss curve and relaxed-optimal bound
below are replays of that held-out trace. They choose no capacity or policy.
Unverified paper figures: the Metal working-set limit on 16 GB Macs, and the
SSD bandwidth of target machines. Nothing here measures streamed inference
speed, 8 GB or 16 GB feasibility.

## The review's main correction

The one-worker median of 0.655 ms per 6,580,224-byte expert is 10.05 GB/s.
That is a bandwidth figure, not a latency. Serial read time per token is
therefore close to bytes ÷ bandwidth. Concurrency can save at most the
latency share, about 24 waiting layers (mean `blocked_layers` 23.95) × 0.655
ms ≈ 16 ms/token. **Bytes per token, not read latency or overlap, decide
the ceiling.** The S3 assessment's sums stand; its framing understated this.

## Miss curve and bound (descriptive, decode, cold start)

[`curve.py`](2026-10-28-mellum-ssd-retarget/curve.py) replays the saved trace
with the unchanged replay tool. Output is in
[`curve-output.txt`](2026-10-28-mellum-ssd-retarget/curve-output.txt).

| Cache | LRU loads/token | LRU MB/token | Relaxed-optimal MB/token |
|---:|---:|---:|---:|
| 2 GiB | 98.96 | 651.2 | 294.7 |
| 3 GiB | 56.54 | 372.1 | 165.2 |
| 4 GiB | 34.72 | 228.4 | 95.6 |
| 5 GiB | 21.02 | 138.3 | 55.7 |
| 6 GiB | 12.62 | 83.0 | 31.5 |
| 7 GiB | 7.13 | 46.9 | 16.5 |
| 8 GiB | 3.50 | 23.0 | 7.6 |
| 10 GiB | 0.31 | 2.0 | 0.5 |

Relaxed-optimal is future-aware and unreachable; it bounds what any
predictor could recover. The curve is steep: each added GiB roughly halves
LRU demand from 3 GiB up.

The review also reported:
- a static hot set covering 90% of decode references would need 5.3 GiB;
- a per-layer LRU split is about 8% better than global LRU;
- consecutive tokens share 4.12 of 8 experts per layer, which LRU already
  harvests.

These come from the reviewer's own script (`whatif.py`) and were not rerun
here. Treat them as reported, not reproduced.

## Why 8 GB is not the first target

Memory on an 8 GB Mac, from measured sizes:
- non-expert tensors are 1.056 GiB (12,925,417,760 − 11,791,761,408 bytes);
- the session is 314.8 MB plus 14,336 B/token, from the three measured
  plans in MELLUM.md, which predicts the -02 run's logged 432,496,896
  bytes at 8K;
- a 3 GiB cache puts one process near 4.7–5.0 GiB.

That makes 3 GiB the ceiling and 2–2.5 GiB realistic.

At 372 MB/token, a bandwidth of B GB/s caps decode at B/0.372 tok/s before
any compute: 4.0 at 1.5 GB/s, 8.1 at 3 GB/s. 10 tok/s needs about 3.7 GB/s
or more of sustained miss-order reads at the full 3 GiB. Typical 8 GB-class
SSD speeds (1.5–3 GB/s) are the reviewer's recollection, not measured. Lower-bit
streamed experts would about quarter the bytes at 3 GiB, but they change weights.

## Why 16 GB is

Metal's `recommendedMaxWorkingSetSize` on a 16 GB Mac is reported as about
10.67 GiB, two-thirds of RAM. This is from secondary sources only:
- [devnote on the Metal VRAM cap](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap);
- [llama.cpp issue 4495](https://github.com/ggml-org/llama.cpp/issues/4495);
- [a 16 GB M2 Pro recipe](https://smeltcore.com/recipes/gpt-oss-20b-m2-pro/).

It is not verified on hardware. Fully resident Q8 plans 12.44 GiB at 8K
context, as logged by the -02 run. If the reported limit holds, **Q8 cannot
run fully resident on 16 GB at the default limit, so streaming is how Q8
runs there at all.** Raising `iogpu.wired_limit_mb` is a system setting
the engine must not require.

The engine keeps routed-expert caches under 7/8 of that recommendation after
context and buffers ([ds4.c:5335](../../../ds4.c)). That gives roughly 9.3
GiB for the whole working set, and about **7 GiB of expert cache** after
fixed tensors and session. At 7 GiB, this trace's LRU demand is 46.9
MB/token, 7.13 misses or 0.25 per layer per token. That is about an eighth of
the 3 GiB rate. The read ceiling is about 32 tok/s at 1.5 GB/s, still before
compute, which is unmeasured on a 16 GB-class chip. Misses become the exception;
all-hit layers become the common path.

## Owner decision

1. SSD streaming targets **16 GB first**, with Q8 weights unchanged. 8 GB
   remains a later target, deferred behind the review's unknowns: a target-class
   SSD read bench, answer-phase locality, and the 8 GB memory plan.
2. Streaming stays opt-in (`--ssd-streaming`, as for the engine's other
   models). With it off, resident speed must match the pre-streaming baseline.
3. The two 16 GB checks are recorded on paper now and measured first when a
   16 GB Mac is available:
   - the real `recommendedMaxWorkingSetSize`;
   - whether fully resident Q8 runs without swapping.
4. Lower-bit (Q4) streamed experts remain an open owner decision, not a plan.

## What would falsify this

- A measured 16 GB working-set limit of at least 13 GiB with no swapping
  under resident Q8 would make streaming optional on 16 GB, not required.
- A non-thinking conversation with materially worse locality at 7 GiB
  (say 150 MB/token or more) would undo the "misses are rare" premise.
- A 16 GB-class chip whose resident compute dominates its tokens would make
  read optimization secondary there.

## Risks

- One thinking-only conversation sets every cache figure.
- The Metal limit is hearsay until measured.
- The 7 GiB budget is derived from the engine's rule, not run on a 16 GB
  machine.
- Typical 16 GB laptops run an editor and browser beside the model.
- The relaxed-optimal bound assumes atomic residency, like the LRU.
