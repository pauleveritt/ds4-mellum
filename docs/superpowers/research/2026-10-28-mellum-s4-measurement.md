# S4 32K measurement: streamed Mellum is exact and fits about 8 GiB; its reads were cache-served

Date: 2026-10-28, 16:38–16:53 EDT, one run. [Preregistration](2026-10-28-mellum-s4-measurement-prereg.md)
(`a5788fe`). The measured build is `2f167fc`; its engine and driver code are unchanged since `c999213`. The
baseline is `b1994c6`. Raw outputs: `/Users/pauleveritt/.codex/visualizations/2026/10/28/mellum-s4-measure-01`.

**Honesty header.**
- **Measured:** on this 128 GiB M5 Max, one frozen prompt of 30,620 tokens at 32K context with 2,048
  greedy tokens. n per arm is as preregistered: A/B n=6 each, R n=1, C n=3.
- **Correctness, regression and memory** pass their preregistered rules.
- **All three streamed runs fail the preregistered read rules.** They loaded 480 GB of expert bytes,
  over the 400 GB cap, while the device read only 1.7–1.9% of that. So the streamed timings measure
  loads served mostly from macOS's page cache, not SSD reads.
- **Not claimed:** any SSD-bound speed, and 16 GB or 8 GB behavior.

## Results

| Rule or metric | Result |
|---|---|
| Goal 2, no regression | **Pass.** B/A medians are 1.006 for prefill (518.2 vs 515.0 t/s) and 0.999 for generation (139.1 vs 139.3 t/s); n=6 each. |
| Identity | **Pass.** All three C runs give R's 2,048 token ids exactly. |
| Read class | **Recorded failure, all three runs.** 479.6 GB logical expert bytes against 9.07, 7.96 and 7.94 GB device reads (`over-read-cap`; a device/logical ratio of 0.017–0.019 would also be `cache-contaminated`). |

| Arm | Time to first token | Decode t/s, all / first 256 / last 256 | Stops per token | Loads per decode token | Peak footprint / RSS / Metal |
|---|---:|---|---:|---:|---|
| R, resident | 29.0 s | 93.7 / 102.4 / 90.1 | 0 | 0 | 0.94 / 0.06 / 12.78 GiB |
| C, 6 GiB (n=3) | 39.9–46.4 s, median 41.5 | 29.4–31.7 / 27.2–31.6 / 29.5–31.8 | 8.72 | 10.79 | 6.93 / 7.12 / 7.81 GiB |

- **Loads.** Prefill plus the first token loaded 50,806 experts (334 GB). Decode tokens 1–2,047 loaded
  10.79 experts per token, near the -02 trace's descriptive 12.6 misses at 6 GiB (n=1, a different prompt).
- **Determinism.** The three C runs' stops and loads were identical.
- **R's memory.** R's footprint excludes its 12 GiB file-backed model map; its Metal figure includes it.
- **C's memory fits the about 8 GiB budget.** The peak Metal allocation, 7.81 GiB, includes the 6 GiB of
  slabs, the 0.48 GiB of KV and buffers.

## Why the reads did not reach the device (inference, not verified)

**Superseded by the [cached-reads spike](2026-10-28-mellum-s4-cached-reads-spike.md):** the main cause was
`fresh_copy` reading each copy back to verify it, which cached it before the run. The unaligned-read
inference below is withdrawn; F_NOCACHE re-reads are also partly cached here.

Streamed reads `pread` through an `F_NOCACHE` descriptor (`ds4_metal.m:13509`, with `g_model_fd` set at
engine open), and each run read a fresh uncached copy, written in 3.0–3.2 s.

Device reads of about 8–9 GB are close to one pass over the experts the run touched. Every later reload
was served from memory.

The builder check had warned that a Mellum expert's 2,193,408-byte slices are not 4 KiB-aligned. The likely
cause is that macOS serves unaligned `F_NOCACHE` reads through the unified buffer cache, which on 128 GiB
holds the whole 12.9 GB file. On a 16 GB machine running an about 8 GiB process, the page cache would be
far smaller, so far more reloads would reach the SSD. That machine was not measured.

## What this does establish

- Opt-in streaming of Mellum's Q8_0 experts is exact at 32K: decode, prefill and recovery tests, plus three
  identical 2,048-token runs.
- Streaming off does not slow resident Mellum, by the 2% rule.
- The streamed process stays under about 8 GiB of footprint and Metal allocation at a 6 GiB cache and 32K
  context.
- Prefill dominates read volume: 334 GB for 30,620 tokens, because every 1,024-row chunk re-reads nearly
  all experts. That supports S5's layer-major prefill hypothesis.

## What would falsify this

- A device-observed rerun, with aligned reads or a smaller page cache, that changes token ids. That
  would contradict exactness; nothing here depends on the cache.
- A rerun showing the 7.9–9.1 GB device reads were not first-touch reads, for example because the same
  bytes were read twice. That would undo the inferred cause.

## Risks

- One prompt, generated output only, thinking on.
- The 400 GB cap was set from a 350 GB estimate; prefill alone exceeded the estimate.
- The CLI A/B uses a short prompt and 256 tokens; regression at 32K is not separately measured.
- Peak RSS includes slab pages pinned with `mlock`.

## Next

1. Make streamed reads reach the device: page-align the expert slices in a packed copy, or read aligned
   superset ranges. Verify with the device counters before any speed claim.
2. Layer-major prefill (S5).
3. A new preregistered measurement. This one is spent and is not retried.
