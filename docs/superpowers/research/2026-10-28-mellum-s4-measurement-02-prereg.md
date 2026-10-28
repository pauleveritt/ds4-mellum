# S4 32K measurement 02 (device-true reads): preregistration

Date: 2026-10-28, committed before any timing.
- **Supersedes the reads of:** [measurement 01](2026-10-28-mellum-s4-measurement.md). Its streamed runs
  read 98% from the page cache.
- **Fixes since 01:** `efdb60f` (the fresh copy is never read back) and `d26dca0` (every loaded expert's
  file pages are invalidated). A live test showed reloads' device/logical ratio at 1.020, up from 0.596
  ([spike](2026-10-28-mellum-s4-cached-reads-spike.md)).

**Honesty header.** Same as 01:
- this 128 GiB M5 Max's SSD and compute only, one frozen prompt;
- n is as stated;
- one run, no retry;
- no 16 GB or 8 GB claims.

## What is the same as 01

- **Prompt:** the same frozen prompt ([01's `prompt.txt`](2026-10-28-mellum-s4-measurement-prereg/prompt.txt),
  SHA-256 `2164763c…6344da`, 30,620 tokens).
- **Run settings:** context 32,768; 2,048 greedy tokens; `--stream-bytes 6442450944` (979 slots).
- **Arms:** R (resident, n=1), then C (streamed, n=3), with 30 s rests.

## What changes

- **Each C run's copy.** A write-only fresh `F_NOCACHE` copy, bracketed by block-driver counters, and
  verified against the model after the run (`verify_copy`).
- **No A/B regression arm.** 01 passed it (B/A 1.006 and 0.999, n=6 each). Since then the only engine
  change is in the streamed layer load (`ds4_gpu_mellum_stream_load_group`). The streaming-off path is not
  touched (`git diff 2f167fc..d26dca0` of `ds4.c` and `metal/` is empty).
- **Read cap:** 600 GB of expert bytes, against 01's measured 480 GB (`classify_reads(..., cap=600e9)`).
  Over it is a recorded failure.

## Decisions fixed now

- **Identity:** every C run's 2,048 token ids equal R's.
- **Read class:** device-observed at a device/logical ratio of 0.99 or more; otherwise cache-contaminated.
- **Reported:**
  - time to first token;
  - decode t/s over all tokens, the first 256 and the last 256;
  - stops and loads per decode token;
  - logical and device bytes;
  - peak footprint, RSS and Metal allocation;
  - C against R and against 01's cache-served C.
- **Speed claims** from C are made only for runs classed device-observed, and only for this SSD.
