#!/bin/sh
# Real Metal sessions only; never substitute a huge CPU model run.
set -eu
unset DS4_MELLUM_TRACE DS4_MELLUM_TRACE_AT DS4_MELLUM_EXACT_PREFILL DS4_MELLUM_MOE_NAX DS4_QWEN4_PREFILL_CHUNK
if [ -z "${DS4_TEST_MELLUM_MODEL:-}" ]; then
    echo 'SKIP Mellum routing GPU proof: set DS4_TEST_MELLUM_MODEL'
    exit 0
fi
proof_dir=$(mktemp -d "${TMPDIR:-/tmp}/mellum-routes-gpu.XXXXXX")
trap 'rm -rf "$proof_dir"' EXIT HUP INT TERM
mkdir "$proof_dir/short" "$proof_dir/tiled" "$proof_dir/exact"
./tests/mellum_routes_gpu "$DS4_TEST_MELLUM_MODEL" tests/mellum_oracle_short.tokens 20 "$proof_dir/short"
./tests/mellum_routes_gpu "$DS4_TEST_MELLUM_MODEL" tests/mellum_heldout_code.tokens 32 "$proof_dir/tiled"
DS4_MELLUM_EXACT_PREFILL=1 ./tests/mellum_routes_gpu "$DS4_TEST_MELLUM_MODEL" tests/mellum_heldout_code.tokens 32 "$proof_dir/exact"
