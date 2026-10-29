#!/bin/sh
# S6 phase D Task 3: the synchronous next-layer prefetch, screened (not preregistered).  From the clone's root,
# apps closed:
#
#   MX=/path/to/Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf OUT=/new/dir BYTES=2684354560 sh <this file>
#
# Applies prefetch-sync.patch (beside this file) and the stop profile's profile.patch, builds tools/mellum_s4_measure,
# reverts both, then runs phase C's workload at BYTES, 128 tokens, prefetch off/on/off/on, 20 s apart, every run
# profiled (DS4_MELLUM_STREAM_PROFILE=1).  Rebuilds the clean tool at the end.  Nothing in $OUT is overwritten.
set -u
: "${MX:?set MX to the MXFP4 GGUF}" "${OUT:?set OUT to a new directory}" "${BYTES:?set BYTES to the cache in bytes}"
D=$(cd "$(dirname "$0")" && pwd)
PROF=docs/superpowers/research/2026-10-29-mellum-imac-stop-profile/profile.patch
P=docs/superpowers/research/2026-10-28-mellum-s6-c-prereg
[ -e "$OUT" ] && { echo "run_sync_screen: $OUT exists; nothing is overwritten" >&2; exit 1; }
[ -z "$(git status --porcelain ds4.c ds4_metal.m)" ] || { echo "run_sync_screen: ds4.c or ds4_metal.m has local changes" >&2; exit 1; }
git apply --check "$D/prefetch-sync.patch" && git apply "$D/prefetch-sync.patch" || exit 1
git apply --check "$PROF" && git apply "$PROF" || { git apply -R "$D/prefetch-sync.patch"; exit 1; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt"
shasum -a 256 "$D/prefetch-sync.patch" "$PROF" > "$OUT/patch-sha256.txt"
make tools/mellum_s4_measure > "$OUT/build.log" 2>&1; rc=$?
git apply -R "$PROF"; git apply -R "$D/prefetch-sync.patch"
[ "$rc" = 0 ] || { echo "run_sync_screen: build failed, see $OUT/build.log" >&2; exit 1; }
swapouts() { vm_stat | awk '/Swapouts/ {gsub("\\.","",$2); print $2}'; }
run() {   # name env...
    n=$1; shift
    sleep 20
    echo "swapouts_before $(swapouts)" > "$OUT/$n.vmstat"
    env "$@" DS4_MELLUM_STREAM_PROFILE=1 tools/mellum_s4_measure --model "$MX" --prompt "$P/workload-prompt.txt" \
        --gen 128 --context 32768 --stream-bytes "$BYTES" --out "$OUT/$n.measure.json" > "$OUT/$n.stdout" 2> "$OUT/$n.stderr"
    echo "rc $?" >> "$OUT/$n.vmstat"
    echo "swapouts_after $(swapouts)" >> "$OUT/$n.vmstat"
    echo "$n $(grep rc "$OUT/$n.vmstat")"
}
run off-1 DS4_NONE=1
run on-1 SCREEN_PREFETCH=1
run off-2 DS4_NONE=1
run on-2 SCREEN_PREFETCH=1
make tools/mellum_s4_measure > "$OUT/rebuild-clean.log" 2>&1; echo "rebuild clean rc $?"
