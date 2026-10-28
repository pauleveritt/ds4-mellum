# M1 Pro bring-up for streamed Mellum

> **For agentic workers:** a short checklist plan, not SDD tasks. Steps use checkbox (`- [ ]`) syntax.
> No product code is planned here. A failure in any step becomes its own research note and, if it needs
> a fix, its own plan.

**Goal:** make the owner's M1 Pro / 32 GB MacBook Pro S5's second machine. It measures streamed Mellum
on a slower GPU, lower memory bandwidth and an older SSD than this M5 Max, then under memory pressure
that leaves about a 16 GB machine's headroom.

**Why this machine (judgment, not measured):** it is closer to a base-chip 16 GB laptop than the M5 Max
is. It is still optimistic for decode: from memory, to be checked, its memory bandwidth is above any
base-chip Air's. A streamed decode that misses S5's target here would miss it on a base 16 GB laptop too;
a pass is encouraging, not proof. It cannot measure a 16 GB Mac's Metal working-set limit (S3b, S5b).

**Constraints:** the branch `codex/mellum-ssd-replay` as it stands, with no rebase and no product change.
Every GPU or live run needs an owner-authorized window on that machine. One-shot measurements are
preregistered and not retried. Never `--raw-prompt`. Nothing discarded: failed runs keep their output.

## 0. Access

- [ ] The owner chooses how runs happen: an SSH connection the app can use, or the owner runs each
  command and the outputs are copied back into the evidence directory.
- [ ] Copy the Q8_0 GGUF (about 13 GB) to the M1 Pro and record its SHA-256. It must match this
  laptop's copy.

## 1. Machine facts (no GPU)

- [ ] Record `machine.txt` with `tools/mellum_machine_facts.sh machine.txt`: model, memory, chip,
  `vm.user_wire_limit`, macOS, Xcode and Metal toolchain, GPU, SSD, free disk, memory pressure, swap and
  power state. It is read-only.
- [ ] Check the macOS version against the engine's Metal 4 shader path (macOS 26 or later). If it is
  older, stop and record that; an OS update is the owner's call.

## 2. Build and model-free tests

- [ ] Check out the branch at a named commit and run `make` and `make test-mellum` (CPU-only).
- [ ] In a GPU window, run `make test-mellum-kernels`. It includes the address-table variants at decode
  and tiled levels and the stream-gate tests. Record the pass/fail counts against this laptop's 564/0.
- [ ] Record the startup line: whether the Metal 4 tensor API is on, and whether E1's tensor-tile experts
  run or prefill falls back to the tiled kernels.

## 3. Resident and streamed correctness (live)

- [ ] Resident: the MELLUM.md generation command (`-c 8192 --temp 0 -n 512`, the hash-map prompt), n=3.
  Record generation and prefill t/s. Output need not match the M5 Max's: other kernel paths may round
  differently.
- [ ] Streamed against resident on this machine: `tests/ds4_test` with `--mellum-stream-identity`,
  `--mellum-stream-prefill` and `--mellum-stream-faults`. All must pass before any timing.

## 4. 32K measurement (preregistered)

- [ ] Preregister before running, reusing measurement 02's frozen prompt (30,620 tokens) and
  `tools/mellum_s4_measure.py`. The arms are R (resident) and C (6 GiB cache), with C's n and the
  device-read rule as in 02.
- [ ] Add the per-turn case: time to first token after appending about 2K tokens to a warm 30K session.
  The appended text is frozen in the preregistration.
- [ ] Report against S5's targets, as measured on this machine with its n.

## 5. Memory pressure (preregistered)

- [ ] Choose the holder that keeps about 16 GB resident for the run. `mlock` is bounded by
  `vm.user_wire_limit`, so the method is chosen from step 1's facts and recorded.
- [ ] Re-run arm C with the holder running and an editor and a browser open. Report swap-ins, memory
  pressure, decode t/s and time to first token against step 4's.

## Done when

The steps above are recorded as one research note with its n per figure, and the roadmap's S5 status
names the M1 Pro's numbers. Failures are recorded, not retried.

## What would falsify the choice of this machine

- Resident Mellum cannot run here, or runs well below what its bandwidth predicts. That would make it
  unrepresentative of any 16 GB target.
- The holder cannot keep 16 GB resident without the system reclaiming it. Then step 5 measures
  something other than a 16 GB machine's headroom.
