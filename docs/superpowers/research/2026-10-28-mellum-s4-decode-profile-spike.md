# Spike: where a streamed Mellum decode token spends its time

Date: 2026-10-28, 17:22 EDT. A throwaway build of `fdad3ee` with timers around the stop/resume ops,
[`profile.patch`](2026-10-28-mellum-s4-decode-profile-spike/profile.patch), since removed. One driver run
on this 128 GiB M5 Max: a short chat prompt, 512 greedy tokens, `--stream-bytes` 6 GiB, device-true reads
(`d26dca0`).

**Honesty header.**
- **Measured:** cumulative host wall time per op, averaged per token. n=1 run, a short context, early
  tokens included.
- **Not measured:** GPU time per kernel, and the long-context case.
- **The cause below is arithmetic.**

## Result

Averages over 512 tokens, from [`profile.log`](2026-10-28-mellum-s4-decode-profile-spike/profile.log). The
run took 8.21 stops and 9.96 loads per token over its last 256 tokens.

| Op | ms per token |
|---|---:|
| encode (host, all passes of the token) | 2.25 |
| finish: GPU execution plus the wait | **21.93** |
| load (preads into slots) | 8.47 |
| ...of which `msync(MS_INVALIDATE)` | 0.36 |
| total | 32.65 |

A resident token at this context takes about 7.4 ms (CLI 136 t/s).

## Reading (arithmetic, not measured)

- **The GPU wait carries most of the gap.** At about 9.2 encodes per token, the GPU runs each layer's
  ungated work again after every stop.
- **The gate suppresses only some dispatch paths after a stop:** the qwen4 dispatcher, `mellum_dispatch`,
  the rows RMS norm, the exact decode matvec and add.
- **These paths run unsuppressed for every remaining layer:** the Q8_0 attention projections (the legacy
  `ds4_gpu_matmul_q8_0_tensor` chain), q/k norm with RoPE, and the f32-to-f16 copies. Each stop
  therefore repeats most of a forward's projection work.
- **Loads cost about 0.85 ms per stop,** for about 1.2 experts (about 8 MB). They run serially, one stop at
  a time. Invalidation is small.

## Next

1. **Route the remaining Mellum-layer dispatch paths through the gate:** the Q8_0 projection chain, q/k
   norm with RoPE, and the copy kernel. Measure the GPU wait per token before and after.
2. **Then consider overlapping loads,** for example starting the next stop's reads while the GPU runs.
   That only matters once the wait is down.

## What would falsify this

A GPU-time profile showing the extra wait is not repeated projection work: for example kernel launch
overhead per command buffer, or attention.

## After gating the projections and copies (`9e9757f`)

The same timer patch and the same driver run (512 tokens, 6 GiB, short prompt, n=1) gave identical token
ids. Data: [`profile-after-gating.log`](2026-10-28-mellum-s4-decode-profile-spike/profile-after-gating.log).

| ms per token | before | after |
|---|---:|---:|
| finish: GPU plus wait | 21.93 | **13.83** |
| load | 8.47 | 8.65 |
| encode | 2.25 | 2.72 |
| total | 32.65 | **25.21** |
| decode t/s (driver) | 30.6 | **39.6** |

**Reading:**
- The projection and copy reruns accounted for about 8 ms per token.
- The GPU wait is still about 6.4 ms above resident. Candidates: command-buffer overhead across about
  9 encodes per token, and paths still ungated.
- Loads, about 0.95 ms per stop and serial, are now the second-largest cost.

The 32K figures in [measurement 02](2026-10-28-mellum-s4-measurement-02.md) predate this change and are
not re-measured.

## The remaining GPU wait, split (`9e9757f`, the built-in `DS4_METAL_CB_TIMES`)

One run per arm on the same driver setup: a short prompt, 256 decode tokens. The streamed arm ran at
6 GiB and took 9.62 stops per token (more than the 512-token run, because early tokens are colder). Data:
[`cb-times-resident.log`](2026-10-28-mellum-s4-decode-profile-spike/cb-times-resident.log) and
[`cb-times-stream.log`](2026-10-28-mellum-s4-decode-profile-spike/cb-times-stream.log).

| Per token | resident | streamed |
|---|---:|---:|
| command buffers | 1.00 | 10.74 |
| GPU busy (Metal GPU span) | 6.82 ms | 12.25 ms |
| commit-to-done minus span (dead time) | 0.21 ms | 2.76 ms |
| driver wall | 7.41 ms | 26.65 ms |

Per buffer, the median span is 0.90 ms (streamed) and the median dead time 0.20 ms, about the same as
resident's.

**Reading (arithmetic, not measured):**
- The streamed GPU wait beyond resident is about 5.4 ms of extra GPU execution plus about 2.6 ms of fixed
  per-buffer cost. The fixed cost is about 0.26 ms × 10.7 buffers.
- About 10.7 passes per token, each from a stop to the head, push roughly 3,700 dispatches through the GPU,
  against about 700 resident. Most are suppressed, but even a zero-threadgroup indirect dispatch costs the
  command processor time. At about 3,000 suppressed dispatches, 5.4 ms is about 1.8 µs each.
- The per-layer check kernel also launches up to 4,096 threads on every pass.
- Neither cost depends on the SSD. Both scale with stops per token. So fewer stops (a larger cache or a
  better policy), or passes that end sooner, are the levers. Shaving the check kernel's thread count is a
  small, low-risk cut.

**Not pursued** (owner, 18:10): load overlap and other high-complexity changes for small gains. A
dispatch count per pass, from the gate's record count, would confirm the suppressed-dispatch arithmetic.
