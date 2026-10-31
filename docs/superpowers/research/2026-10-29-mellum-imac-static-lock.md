# On the 8 GB iMac, `ds4 --ssd-streaming` at a short context locks its static weights, and prefill collapses

**Corrected 2026-10-29 ~21:25, the same evening: the lock is not shown to be the cause.** The note below (kept as
written) read three locked and two unlocked processes, run in that order, as cause and effect. Then the README
runner, with the lock off (`A_ENV=DS4_METAL_DISABLE_STREAMING_STATIC_LOCK=1`, no "locked" line in its log),
prefilled at **2.79 t/s** ([capture](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-29-mellum-readme-speed/imac-static-lock/unlocked-aborted)). That run
followed the runner's token count, an 8-minute non-streamed load of the 7 GB model. So on this machine prefill
swings from about 3 to 121 t/s under the same flags, and the order of runs and the memory state (1.4 GB of swap
in use since phase D's bringup, high page-outs) are confounded with the lock. What is measured: the five rates
below and the 2.79. What is not: their cause. A controlled check needs a rebooted iMac and alternated locked and
unlocked processes.

Date: 2026-10-29. A screen, found while measuring the README's iMac rows. Captures:
[`2026-10-29-mellum-readme-speed/imac-static-lock/`](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/research/2026-10-29-mellum-readme-speed/imac-static-lock).

**Honesty header.** Measured: the 8 GB M1 iMac, build `67b4670`, MXFP4 streamed at 2 GiB (`--ssd-streaming
--ssd-streaming-cache-experts 2GB`), the README runner's 2,056-token prompt, one process per line below.
Conjecture: why locking slows prefill this much. The cause of the switch is read from code.

| Run | Static weights | Prefill t/s |
|---|---|---|
| `ds4 -c 3072` (the prefill runner's context), processes 1-3 | locked 1.05 GiB | 13.44, 1.70, 4.01 |
| `ds4 -c 3072`, `DS4_METAL_DISABLE_STREAMING_STATIC_LOCK=1` | pageable | 121.40 |
| `ds4 -c 32768` | pageable ("to preserve runtime headroom") | 103.34 |

Phase C's measure tool ran at 32K and so never locked: about 138 t/s on its 2,686-token prompt (n=3).

**The switch** (`ds4.c`, the streaming static lock in engine open, around line 74470): with streaming on, the
engine locks the weights every token uses whenever they and the expert cache fit
`ds4_streaming_manual_cache_safe_bytes(backend, context, …)`. The budget grows as the context shrinks, so a short
context locks and a long one does not. The lock exists to keep those weights from competing with streamed experts
in the file cache, which helps a large machine.

**On 8 GB it does the opposite.** The locked runs logged no expert-cache cap or mlock failure, so the cache kept
its 652 slots. Our guess, not shown: 1.05 GiB more wired memory on an 8 GB machine already 1.4 GB into swap
starves the rest of the system. Prefill drops 9-70× and varies wildly between processes.

**Consequences.**
- The README's iMac rows are measured with `DS4_METAL_DISABLE_STREAMING_STATIC_LOCK=1` (the runner's `A_ENV`),
  and the README says so.
- A user who runs the README's quick check on an 8 GB Mac gets the collapse by default. The product fix (when to
  lock: a RAM floor, a share of physical memory, or never for Mellum) is a decision, not made here.

## What would falsify this

- A locked run at `-c 3072` on the iMac with prefill near 120 t/s, or an unlocked one near 10.
- A 16 GB Mac showing the same collapse when locked (then it is not about 8 GB).

## Risks

- n=1 for each unlocked run, three locked processes. The iMac's swap was already 1.4 GB used.
