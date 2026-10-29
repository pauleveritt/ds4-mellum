#!/bin/sh
# Where a streamed decode stop's time goes on the 8 GB M1 iMac (a screen, not preregistered).  From the clone's
# root, with the apps closed, over SSH:
#
#   MX=/path/to/Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf OUT=~/s6-stop-profile sh <this file>
#
# It applies the screen-only timing patch (profile.patch, beside this file) to ds4.c, builds
# tools/mellum_s4_measure, runs the phase C workload at the 2 GiB cache, and reverts the patch.  Five runs, 20 s
# apart: two unprofiled (the patch's own cost), two with the per-callback profile, and one 64-token run that
# adds the engine's pread and command-buffer timings.  Nothing in $OUT is overwritten.
set -u
: "${MX:?set MX to the MXFP4 GGUF}" "${OUT:?set OUT to a new directory}"
D=$(cd "$(dirname "$0")" && pwd)
P=docs/superpowers/research/2026-10-28-mellum-s6-c-prereg
B=2147483648
[ -e "$OUT" ] && { echo "run_imac_profile: $OUT exists; nothing is overwritten" >&2; exit 1; }
[ -z "$(git status --porcelain ds4.c)" ] || { echo "run_imac_profile: ds4.c has local changes" >&2; exit 1; }
git apply --check "$D/profile.patch" || exit 1
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt"
shasum -a 256 "$D/profile.patch" > "$OUT/patch-sha256.txt"
git apply "$D/profile.patch"
make tools/mellum_s4_measure > "$OUT/build.log" 2>&1; rc=$?
git apply -R "$D/profile.patch"
[ "$rc" = 0 ] || { echo "run_imac_profile: build failed, see $OUT/build.log" >&2; exit 1; }
swapouts() { vm_stat | awk '/Swapouts/ {gsub("\\.","",$2); print $2}'; }
run() {   # name gen env...
    n=$1; g=$2; shift 2
    sleep 20
    echo "swapouts_before $(swapouts)" > "$OUT/$n.vmstat"
    env "$@" tools/mellum_s4_measure --model "$MX" --prompt "$P/workload-prompt.txt" --gen "$g" --context 32768 \
        --stream-bytes "$B" --out "$OUT/$n.measure.json" > "$OUT/$n.stdout" 2> "$OUT/$n.stderr"
    echo "rc $?" >> "$OUT/$n.vmstat"
    echo "swapouts_after $(swapouts)" >> "$OUT/$n.vmstat"
    echo "$n $(grep rc "$OUT/$n.vmstat")"
}
PROF="DS4_MELLUM_STREAM_PROFILE=1 DS4_MELLUM_STREAM_PROFILE_ROUNDS=1"
run base-1 128 DS4_NONE=1
run prof-1 128 $PROF
run base-2 128 DS4_NONE=1
run prof-2 128 $PROF
run reads 64 $PROF DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE=1 DS4_METAL_CB_TIMES=1
# the build is the patched one: rebuild the clean tool so later kit runs use the commit's code
make tools/mellum_s4_measure > "$OUT/rebuild-clean.log" 2>&1; echo "rebuild clean rc $?"
