# Spike: why S4's streamed reads were cache-served, and what makes reloads device-true

Date: 2026-10-28, after the [S4 measurement](2026-10-28-mellum-s4-measurement.md). The runs were CPU and
SSD only, on this 128 GiB M5 Max and its APPLE SSD AP2048Z, under macOS 27.2.

**Honesty header.**
- **Measured:** device/logical read ratios from block-driver counters, bracketing each read pass over
  fresh 2 GiB random files read through an `F_NOCACHE` descriptor with `F_RDAHEAD` off. There is
  n=1 per cell.
- **Not measured:** the fix inside the engine, and any streamed inference speed.

## Results

| File | Read | Pass 1 | Between passes | Pass 2 |
|---|---|---:|---|---:|
| copy verified by reading it back (`fresh_copy`) | aligned 4 MiB | 0.016 | — | 0.016 |
| copy verified by reading it back | Mellum-sized unaligned 2,193,408 B | 0.015 | — | 0.015 |
| write-only (`F_NOCACHE` writes, fsync, never read) | aligned | 1.000 | — | 0.250 |
| write-only | unaligned | 1.000 | — | 0.478 |
| write-only | aligned | 1.000 | `posix_madvise(DONTNEED)` | 0.257 |
| write-only | unaligned | 1.000 | `posix_madvise(DONTNEED)` | 0.492 |
| write-only | aligned | 1.000 | `msync(MS_INVALIDATE)` on a `MAP_SHARED` read-only mapping | 1.007 |
| write-only | unaligned | 1.000 | `msync(MS_INVALIDATE)` | 1.014 |

Scripts: [`readbench.py`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-s4-cached-reads-spike/readbench.py) and
[`readbench2.py`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-28-mellum-s4-cached-reads-spike/readbench2.py).

## What it says

1. **The measurement's copies were cached before the runs began.** `fresh_copy` verified each copy by
   reading it back, and that read left the copy in the unified buffer cache. The first read of a verified
   copy is 98% cache. This was the main cause of the S4 measurement's 1.7–1.9% device ratio, not
   alignment. The earlier "unaligned" inference is withdrawn.
2. **`F_NOCACHE` does not keep re-reads off the cache here.** After one pass over a never-read file,
   75% (aligned) and 52% (unaligned) of a second pass is served from cache. Alignment matters only at the
   margin.
3. **`msync(MS_INVALIDATE)` over the file's page-rounded range makes the next read device-true.**
   Upstream's eviction hint, `posix_madvise(DONTNEED)`, does not.

## Proposed fix

These are proposals; nothing is changed yet.

- **Engine.**
  - After a Mellum layer load, invalidate the page-rounded file ranges of the experts it just read.
  - Use a read-only `MAP_SHARED` mapping of the streaming descriptor, so every reload reaches the device
    on any page-cache size.
  - Never invalidate the static tensors' spans: a rounded boundary page may be shared, and only a page
    held wholly by expert tensors is dropped.
- **Runner.**
  - `fresh_copy` writes only.
  - It verifies the copy after the timed run, never before.
- **Then a new preregistered measurement.** It would be device-observed if fixes 1 and 2 hold. The read
  cap would be reset from the measured 480 GB.

## What would falsify this

- A device ratio below 0.99 on reloads after the engine invalidates its ranges.
- Invalidation that measurably slows a token beyond its read time, for example because the syscall per
  load costs more than the read.

## Risks

- `MS_INVALIDATE` semantics on a read-only shared mapping are observed, not documented. The cells are n=1.
- On a real 16 GB machine the page cache is small anyway, and the invalidation may only add syscall cost.
