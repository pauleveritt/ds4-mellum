# S4 32K measurement 02: device-true streamed Mellum on this laptop

Date: 2026-10-28, 17:09–17:19 EDT, one run. [Preregistration](2026-10-28-mellum-s4-measurement-02-prereg.md)
(`f2c64a7`), measured at `f2c64a7`. Raw outputs:
`~/.codex/visualizations/2026/10/28/mellum-s4-measure-02`.

**Honesty header.**
- **Measured:** this 128 GiB M5 Max with its APPLE SSD AP2048Z, one frozen prompt of 30,620 tokens, 32K
  context and 2,048 greedy tokens. R n=1, C n=3.
- **The streamed reads are device-observed this time.** They sat between 0.996 and 1.002 of logical
  bytes, against 01's 0.017–0.019.
- **Not measured:** 16 GB or 8 GB hardware, other SSDs, and any cause of the overheads below beyond
  arithmetic.

## Results

| Rule | Result |
|---|---|
| Read class | **Device-observed, 3 of 3.** 479.6 GB logical against 480.8, 478.5 and 477.9 GB device; under the 600 GB cap. Each copy verified against the model after its run. |
| Identity | **Pass, 3 of 3.** Every run gives R's 2,048 token ids. |

| Arm | Time to first token | Decode t/s, all / first 256 / last 256 | Stops / loads per decode token | Peak footprint / RSS / Metal |
|---|---:|---|---|---|
| R, resident | 29.1 s | 91.8 / 102.5 / 87.1 | 0 / 0 | 0.92 / 0.06 / 12.78 GiB |
| C, 6 GiB, n=3 | 58.6–60.9 s, median 59.9 | 26.1–27.2 / 24.0–25.7 / 24.3–26.0 | 8.72 / 10.79 | 6.93 / 7.12 / 7.81 GiB |
| 01's C, cache-served | 39.9–46.4 s | 29.4–31.7 | 8.72 / 10.79 | the same |

Prefill and the first token loaded 50,806 experts, 334 GB. Stops and loads were identical in every run,
in 01 and in 02.

## Arithmetic (not measured causes)

- **Prefill is read-bound at this SSD's bandwidth.** C's time to first token exceeds R's by about 31 s for
  334 GB of expert reads, about 10.8 GB/s, near the 9.9 GB/s measured in the 10-27 storage service. Each
  1,024-row chunk re-reads nearly every expert. Layer-major prefill (S5), reading each layer's experts once
  per prompt, would remove most of it.
- **Decode costs more than its reads explain.**
  - A C token takes about 38 ms against R's about 11 ms. Its reads are 10.79 × 6.58 MB = 71 MB per
    token, about 7 ms at 10 GB/s. Its syncs are 8.72 × about 164 µs = about 1.4 ms (the spike's figure).
  - That leaves about 18 ms per token unexplained. Candidates include per-stop load latency (about 1.24
    experts per stop, read serially), re-encoding after a stop, and the invalidation `msync`. The S4
    stop/resume and load path needs profiling before S5 chooses a policy.
  - 01's cache-served decode, 29–32 t/s, was only about 15% faster than 02's device-bound decode. Most of
    the overhead is not SSD time.

## What this establishes

On this laptop, Mellum streams its experts from SSD with the following properties:
- exact output: 2,048-token identity at 32K;
- a bounded process: 6.93 GiB footprint, 7.81 GiB Metal, against the about 8 GiB target;
- device-true reads;
- streaming off unaffected (01's A/B, which 02 does not re-run; unchanged since).

## What would falsify this

- **A device ratio below 0.99** in a rerun of this preregistration.
- **Token ids that differ from resident.**
- **A profile showing the decode gap is SSD time after all,** for example a high queue latency per read.
  That would move the lever from host overhead to the device.

## Risks

- One prompt, n=3 on one machine, thinking-phase decode.
- The invalidation semantics are observed, not documented.
- On a 16 GB machine with a slower SSD and chip, every figure moves. S3b and S5 measure there.

## Next

1. **Profile the decode stop/resume path.** Split per-stop time into wait, load, invalidation and
   re-encode, and decide what to cut.
2. **Layer-major prefill** (S5 hypothesis): cut 334 GB of prefill reads to about one pass per prompt.
3. **S3b on a real 16 GB Mac**, when one is available.
