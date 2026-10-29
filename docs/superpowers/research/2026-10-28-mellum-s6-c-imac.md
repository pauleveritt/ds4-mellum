# S6 phase C: the 8 GB M1 iMac, result

Date: 2026-10-29. Build `69492de` (the kit's commit, recorded in each `bringup/commit.txt`). Rules:
[the preregistration](2026-10-28-mellum-s6-c-preregistration.md), with its notes of 08:12 and 12:19.
Captures: [`2026-10-28-mellum-s6-c-evidence/imac/`](2026-10-28-mellum-s6-c-evidence/imac/). Every figure
below is recomputed from them by `python3 -I docs/superpowers/research/2026-10-28-mellum-s6-c-evidence/imac/analyze.py`,
except the logits agreement, which is read from each `correct/agree.*.json`.

**Honesty header.**
- **Measured:** one 8 GB M1 iMac (iMac21,1, macOS 27.0.1, Metal limit 5,726,633,984 B), the
  preregistered workload, n stated per figure.
- **Not run:** C4's apps-open runs (owner's decision, note of 12:19: no desktop access) and C6 (no access
  to the M4 Max). So **C4 and C6 are undecided**, and C5 has only its apps-closed half.
- **Not claimed:** other 8 GB Macs, answer quality, the iMac under a desktop workload.
- **Conjecture, marked where it appears:** why the `ds4_test` lines failed, and where the decode time goes.

## Verdicts

| Item | Verdict | Evidence |
|---|---|---|
| C1 bring-up | **Pass** on the second run | First run (`s6-c-imac/`): `test-mellum` rc 2 on Apple's Python 3.9.6, the rest rc 0. Second run (`s6-c-imac-2/`, Python 3.14.10): `make`, `make tools`, `test-mellum` (14 tests OK) and `test-mellum-kernels` (697 checks, 0 failures) all rc 0. Model sha256 `cf09584a…`. |
| C2 cache | **2 GiB** (652 slots) | Rule limit: 5,726,633,984 − 1 GiB = 4,652,892,160 B. Metal peaks: 3 GiB 5,217,779,712; 2.5 GiB 4,677,697,536 (over by 24,805,376 B); 2 GiB 4,140,908,544, with 0 swap-out growth. The peaks are identical in both cache stages. |
| C3 correctness | **Pass** | At 2 GiB against the M5's resident logits, agreement 29/29, 64/64 and 64/64; mean KL 8.9e-08, 2.6e-04 and 3.7e-04; worst KL 5.6e-07, 3.4e-03 and 1.05e-02 (short, 1100, 2200). Bounds: ≥ 97%, ≤ 0.01, ≤ 1.0. Both `correct` runs give the same figures. |
| C4 footprint | **Undecided** | Apps-open runs not run. |
| C5 speed | Apps-closed half, reported | Below. |
| C6 falsifier | **Undecided** | The 27B was not run on the M4 Max. |

**The preregistration's own design falsifier fired:** "C2 picks 2 GiB or less. Decode on the iMac then
depends on the SSD, which C4's single request may hide." The spec's falsifier, "under about 2 GiB for the
cache", did not: 2 GiB qualifies.

## C1's recorded lines

The three `ds4_test` lines (`--mellum-stream-identity`, `-prefill`, `-faults`) returned rc 1 in both
bring-ups, as the preregistration allowed on 8 GB. The logs show memory failures, not wrong numbers:
- the resident reference could not open ("metal failed to map model views ... insufficient memory");
- the 6 GiB streamed run was capped to 4 GiB, could mlock only 0.15-0.19 GiB, and its prefill failed;
- the faults test's engine open failed (`tests/ds4_test.c:804`).

So those tests compared against a reference that was never produced. That is a reading of the logs; it was
not diagnosed further. On this machine C3 is the correctness gate, and it passed.

## C5, apps closed (streamed MXFP4, 2 GiB)

Decode t/s is the preregistered metric, (n − 1) / the summed ns of tokens 1-511.

| Series | When | n | Decode t/s, median (range) | First token s, median (range) | Swap-out growth per run |
|---|---|---|---|---|---|
| `s6-c-imac-3/footprint-closed-interrupted/footprint-closed/` | 11:40-11:46 | 3 | 5.85 (5.82-5.85) | 19.40 (19.37-20.01) | 13,600, 0, 0 |
| `s6-c-imac-3/footprint-closed/` | 11:51-11:57 | 3 | 5.90 (5.87-5.93) | 19.48 (19.41-19.60) | 0, 0, 0 |
| `s6-c-imac-3/footprint-closed-interrupted/` | 10:46-10:52 | 2 of 3 | 5.93 (5.93-5.94) | 19.39 (19.37-19.42) | 4,472, 0 |
| `s6-c-imac-2/footprint-closed/`, eight apps open by mistake | earlier | 3 | 5.83 (5.82-5.85) | 19.96 (19.38-20.49) | 34,412, 171,196, 0 |

**Which series is C4's apps-closed result.** The first series was interrupted by hand in run 3 (its
`NOTE.txt`). A complete series ran at 11:40 and was later found *inside* the interrupted folder. The
likely cause is that the rename for the interrupted attempt ran after it, moving it there; the folder's
times fit that. The 11:51 series was then run into a fresh folder. By the rule that each item runs once,
the 11:40 series is C4's apps-closed result: **5.85 t/s, n=3**. The 11:51 series (5.90, the figure the
owner reported and the 12:19 note quotes) is reported beside it. Every complete apps-closed run at 2 GiB
lies within 5.82-5.94 t/s (n=8), and C4 and C6 are undecided either way, so the choice decides nothing.

**Per token, at 2 GiB** (identical in all 13 runs at 2 GiB): 34.71 loads and 19.98 stops. Metal peak
4,140,908,544 B; footprint peak about 3.05 GB.

**The M5's figures for the same workload,** from the speed run at the merge, reported beside and not
compared (n=3 each): resident, first token 0.95 s and decode 142.8 t/s; streamed at 2.5 GiB, 1.19 s and
40.7 t/s, 22.62 loads/token.

**Prefill:** 2,686 prompt tokens in about 19.4 s, about 138 t/s (arithmetic on the medians).

**Where the decode time goes (conjecture).** 1 / 5.85 t/s is 171 ms/token; over 19.98 stops that is
about 8.6 ms per stop. On the M5 before step 2b, a stop's loads took about 0.95 ms (S4). If stops dominate on the iMac
as on the M5, its cost per stop is the lever, not the stop count alone. It was not profiled on the iMac.

## Also observed (not preregistered)

- **The same ids at every cache.** All 17 workload runs on the iMac, at 2, 2.5 and 3 GiB, produced the
  same 512 ids. They match the M5's MXFP4 runs for the first 310 tokens and then diverge. The two
  machines run different kernel paths (the iMac's tensor API is disabled pre-M5), and C3's logits agree,
  so this is taken to be a near-tie carried forward. That is not shown.
- **A larger cache is faster but does not fit.** In the cache stages, 2.5 GiB gave 7.31 t/s and 3 GiB
  9.14 and 9.84 t/s (n=1 each per stage), at 22.60 and 14.19 loads/token. Both are over the rule's Metal
  limit, and 3 GiB swapped in both stages (13,712 and 106,000 swap-outs).
- **Apps left open swap.** In `s6-c-imac-2/footprint-closed`, run with eight apps open by mistake, runs 1
  and 2 swapped (34,412 and 171,196 swap-outs) at about the same speed. That hints at Goal 4 but decides
  nothing, since that load was neither the preregistered one nor recorded.

## What would falsify this

- C3: a rerun at 2 GiB that misses the agreement or KL bounds on any file.
- The cache choice: a Metal peak at 2 GiB above 4,652,892,160 B, or swap-out growth at 2 GiB with the
  apps closed and the machine otherwise idle.
- The apps-closed speed: another series on this machine outside about 5.8-6.0 t/s with the same build,
  cache and workload.

## Risks

- One machine, n=3 per series, one flattened request (the preregistration's risks).
- The swap baseline was already high at the second bring-up (`machine.txt`: 2,891 MB of 4,096 MB swap used), and
  the disk was 97% full. Neither was varied.
- The interrupted series' bookkeeping was done by hand, and its order is inferred from file times.
- macOS 27.0.1 here, not the 27.1 the preregistration expected.
