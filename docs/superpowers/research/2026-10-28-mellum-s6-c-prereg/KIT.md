# The S6 phase C kit: what goes to the iMac, and the order of the stages

The rules are in [the preregistration](../2026-10-28-mellum-s6-c-preregistration.md), and the steps in
[plan C](https://github.com/pauleveritt/ds4-mellum/tree/mellum-evidence-2026-10-31/docs/superpowers/plans/2026-10-28-mellum-s6-c-imac.md). `run_imac.sh` was dry-run on the M5 Max on
2026-10-29: its `correct` stage reproduced the M5 reference exactly (157/157 positions, KL 0), and its
`cache` stage recorded swap-outs around each run. After the merge `633fd23` (amendment of 2026-10-29), the
`correct` stage was dry-run again, with the same result.

## Copy to the iMac

1. **The branch.** On the M5, run `git -C <worktree> bundle create s6.bundle research/mellum-8gb-q4`. On
   the iMac, run `git clone s6.bundle ds4-mellum-s6`.
2. **The model**, `Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf`. Its SHA-256 must be `cf09584a…`;
   `bringup` records it.
3. **The background-load project.** Copy a small Python project to `../mellum-demo` next to the clone, or
   name another in the preregistration before the run. `tools/mellum_background_load.sh` refuses a missing
   directory.
4. **The reference logits.** They are git-ignored, so copy them by hand:
   - copy `docs/superpowers/research/2026-10-28-mellum-s6-c-evidence/raw/m5-decode-resident.mx.*` to the
     same path in the clone;
   - in that directory, run `shasum -a 256 -c ../m5-reference-sha256.txt`. Every line must say OK.
5. **Python 3.10 or later as `python3`.** Apple's Command Line Tools ship 3.9.6, which fails `test-mellum`
   (the preregistration's note of 2026-10-29 08:12). For example, install python.org's macOS installer and
   put it first: `export PATH=/Library/Frameworks/Python.framework/Versions/3.14/bin:/usr/bin:/bin:/usr/sbin:/sbin`.

## Stages, in order (from the clone's root)

Set `MX` and `OUT` first, then run each stage as `MX=… OUT=… sh docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/run_imac.sh STAGE`:

1. **`bringup`** (C1). Stop if `make`, `test-mellum` or `test-mellum-kernels` is not `rc 0`. The three
   `ds4_test` lines are recorded only; on 8 GB they may fail for memory (C1, amended).
2. **`cache`** (C2), with the apps closed. Pick the cache by the preregistered rule from the three `*.measure.json`
   (`metal_allocated_peak`) and the `*.vmstat` files, and from `tools/mellum_metal_limit`'s line in
   `bringup/machine.txt`.
3. **`correct BYTES`** (C3), with the chosen cache in bytes: 3221225472, 2684354560 or 2147483648.
4. **`footprint closed BYTES`** (C4).
5. **Start the background load:** `sh tools/mellum_background_load.sh ../mellum-demo docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/urls.txt`.
   Wait until it prints `ready`, then run **`footprint open BYTES`**.
6. **C6, on the M4 Max** (the preregistration's addendum of 2026-10-29 07:23), with ds4-engine's
   submodule at its pin:
   `DS4=…/ds4-engine/external/ds4 MODEL=…/Qwen3.8-27B-Q4_K_M.gguf OUT=$OUT/c6 sh docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/run_c6_27b.sh`.
   It checks the pin and the model's sha256, builds `ds4-bench`, and runs three times.

Bring `$OUT` back whole, including failed stages. The agent writes the analysis from those captures alone.
