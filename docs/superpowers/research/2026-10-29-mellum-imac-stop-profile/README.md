# Where a streamed decode stop's time goes on the iMac (screen kit)

A screen, not a preregistered run (`docs/sdd.md`, "Evals"). Phase C's cache stages fit the iMac's decode at
about 35.4 ms per token plus 6.71 ms per stop (n=6 runs, 2 to 3 GiB;
[result](../2026-10-28-mellum-s6-c-imac.md)). This kit splits a stop into its parts.

- `profile.patch`: screen-only timing of the streamed decode callbacks (`ds4.c`, not committed to product).
  Per token: `encode` (CPU encoding of a pass), `finish` (commit, GPU execution, gate read), `load_begin`
  (starting a stop's reads), `load_wait` (what is left of the reads after the resumed pass is encoded).
  `DS4_MELLUM_STREAM_PROFILE_ROUNDS` logs every round and stop.
- `run_imac_profile.sh`: applies the patch, builds, runs the phase C workload at 2 GiB five times (two
  unprofiled, two profiled at 128 tokens, one 64-token run with `DS4_METAL_CB_TIMES`), reverts the patch and
  rebuilds clean.
- `analyze.py OUT`: the tables.

**Dry run on the M5 Max** (`m5-dryrun/`, `81906f8`, 128 GB, n=2 per arm): unprofiled 34.8 and 34.7 ms/token,
profiled 35.3 and 34.8, so the patch costs about 1%. Per token: finish 17.5 ms, load wait 11.5, encode 5.5;
about 20.8 stops per token, median 2 experts per stop. A decode command buffer spends a median 0.60 ms on the
GPU of 0.80 ms from commit to done. On this machine the model file sits in the page cache, so the read rates
(6-11 GB/s) are memory, not SSD; the iMac, with 8 GB, cannot cache the 7 GB file.

The engine's `DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE` prints nothing on the asynchronous read path, so the
read span is taken as read start + encode + read wait.
