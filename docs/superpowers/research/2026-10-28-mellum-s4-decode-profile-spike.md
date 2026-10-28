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
