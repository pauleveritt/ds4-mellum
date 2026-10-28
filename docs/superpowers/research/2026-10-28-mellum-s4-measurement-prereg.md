# S4 32K measurement: preregistration

Date: 2026-10-28, committed before any timing. Branch `codex/mellum-ssd-replay`; the
measured build is the commit after this file's. Design:
[S4 spec](../specs/2026-10-28-mellum-s4-cached-execution-design.md), goals 2 and 6.

**Honesty header.** This fixes, in advance, what will be measured and how it is judged.
- **Measured:** this 128 GiB M5 Max's SSD and compute only.
- **Not claimed:** 16 GB or 8 GB behavior, or speed on any other machine.
- **The prompt:** one synthetic prompt, and n stated per arm.
- **One run:** no rerun or retry follows any outcome.

## Frozen inputs

- **Model:** `~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf` (Q8_0, 12,925,417,760 B).
- **Prompt:** [`prompt.txt`](2026-10-28-mellum-s4-measurement-prereg/prompt.txt).
  - SHA-256 `2164763cbe3b1fa7bb9d46107650e4cb0679f3d78329d1c3477aea3f2d6344da`, 85,844 B.
  - Built by `tools/mellum_s4_measure.py build_prompt(root, 86_000)`: 7 tracked `.c`/`.md` files, no research notes, no ac-demo.
  - Rendered by the driver as one user turn with thinking on, it is **30,620 tokens** (calibration run, resident, `--gen 1`).
- **Context:** 32,768. **Generation:** 2,048 greedy tokens, so 32,668 in total.
- **Streamed cache:** `--stream-bytes 6442450944`, which is 6 GiB = 979 slots.

## Arms

| Arm | Build | Tool | Runs |
|---|---|---|---|
| A | pre-S4 baseline `b1994c6`, built in a detached worktree | `ds4` CLI: `-c 8192 --temp 0 -n 256 -p "Explain how a hash map works, with an example in C."` | 6 |
| B | measured build, streaming off | same CLI command | 6 |
| R | measured build, resident | `tools/mellum_s4_measure`, frozen prompt | 1 |
| C | measured build, streamed at 6 GiB | `tools/mellum_s4_measure --stream-bytes 6442450944`, each from a fresh `F_NOCACHE` model copy | 3 |

A and B run in A/B/B/A blocks (`regression_order(6)`) with 30 s rests. Then R runs, then C, with 30 s
rests between processes. Each C process is bracketed by block-driver read counters (`mellum_device_observer`
snapshot before and after), and its fresh copy is deleted after it. Every run respects the shared model lock.
A held lock before the first run postpones the whole measurement; nothing is created.

## Decisions fixed now

- **Goal 2, no regression:** B's median prefill t/s and median generation t/s are each at least 0.98 of A's
  (`regression_verdict`). The verdict is one-sided: it detects a slowdown.
- **Identity:** every C run's 2,048 token ids equal R's.
- **Read class per C run:** device-observed when block-driver read bytes are at least 0.99 × (loads × 6,580,224).
  Otherwise cache-contaminated. Over 400 GB of expert bytes is a recorded failure (`classify_reads`).
- **Reported, not gated:**
  - time to first token;
  - decode t/s over tokens 0–255 and over the last 256;
  - stops and loads per token;
  - logical and device bytes;
  - peak `phys_footprint`, RSS and Metal allocation, against the about 8 GiB budget.

  C is reported against R, with n=3 shown as a range and a median.

## What would change the plan

Nothing in the run. Any result outside these rules is reported as it is. A second measurement needs its own
preregistration.
