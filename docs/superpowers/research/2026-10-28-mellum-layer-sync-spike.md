# Spike: cost of a host sync after every Mellum decode layer

Date: 2026-10-28, 13:03–13:09 EDT. Throwaway spike for the S4 design, on this
128 GiB M5 Max, from `27cce31`, in a detached worktree since removed.

**Honesty header.** Measured: resident Mellum decode speed through the CLI,
with and without a host commit-and-wait after each of the first 27 layers.
There were n=4 processes per arm, in rotated order with 20 s rests, and 256
greedy tokens per run. This is one prompt on one machine, with no streaming
and no SSD reads. The per-sync figure is a difference of medians, not a
distribution.

## Method

The hook is in [`spike-sync.patch`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-layer-sync-spike/spike-sync.patch),
13 lines in `mellum_graph_forward_tokens`. `DS4_SPIKE_LAYER_SYNC` selects:

- `1`: end the command buffer, wait, begin a new one after each layer but the
  last, for decode (`T == 1`) only;
- `2`: the same, plus a 32-byte read of the layer's selected expert IDs.

The command, from MELLUM.md's generation row:
`DS4_SPIKE_LAYER_SYNC=<arm> ./ds4 -m Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf -c 8192 --temp 0 -n 256 -p "Explain how a hash map works, with an example in C."`.
The arm order was 0 1 2 2 1 0 1 2 0 0 2 1; the raw lines are in
[`spike-ab.log`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-layer-sync-spike/spike-ab.log). All 12
outputs were byte-identical (`cmp` against run 1).

## Result

| Arm | Generation t/s (n=4) | Median | ms/token |
|---|---|---:|---:|
| 0, no sync | 134.62, 135.89, 135.95, 136.06 | 135.92 | 7.36 |
| 1, sync per layer | 84.88, 84.90, 84.91, 85.38 | 84.91 | 11.78 |
| 2, sync plus ID read | 84.79, 84.97, 85.08, 85.09 | 85.03 | 11.76 |

A sync after every layer adds 4.42 ms per token, about 164 µs per sync, and
cuts resident decode by 37.5%. Reading eight IDs adds nothing measurable. In
an earlier n=1 smoke run, the arms gave 138.36 against 86.75 t/s.

## Consequence for S4

On the -02 trace's 6 GiB LRU replay (descriptive, n=1), a mean of 9.41
layers per token has at least one miss (p95 19), and 1.3% of tokens are
all-hit. Syncing only at layers that miss would cost about 9.4 × 164 µs ≈ 1.5
ms per token, against 4.4 ms for syncing every layer. The owner chose
syncing at misses for S4.

## What would falsify this

- A per-sync cost that changes materially with context length, prompt or
  thermal state would make 164 µs unrepresentative.
- A slower chip with a different command-buffer overhead (a 16 GB machine)
  could move the per-sync cost either way.

## Risks

- One prompt and 256 tokens at short context only.
- The CLI's own t/s line is the only timer.
- The spike does not re-encode work after a stop, which a real resume must do.
