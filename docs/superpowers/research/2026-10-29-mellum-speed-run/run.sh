#!/bin/sh
# Speed run at one build: phase C's workload (2,686 tokens, 512 greedy, context 32,768) through
# tools/mellum_s4_measure, n=3 per configuration, rotated so no configuration runs twice in a row, a 20 s rest
# before each run.  From the repo root:
#   MX=<MXFP4 gguf> Q8=<Q8_0 gguf> OUT=<dir> sh docs/superpowers/research/2026-10-29-mellum-speed-run/run.sh
set -u
: "${MX:?}" "${Q8:?}" "${OUT:?}"
P=docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/workload-prompt.txt
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt"
run() {   # name model extra-args
    name=$1; model=$2; shift 2
    [ -e "$OUT/$name.measure.json" ] && { echo "speed-run: $name exists" >&2; exit 1; }
    sleep 20
    tools/mellum_s4_measure --model "$model" --prompt "$P" --gen 512 --context 32768 --out "$OUT/$name.measure.json" "$@" \
        > "$OUT/$name.stdout" 2> "$OUT/$name.stderr"
    echo "$name rc $?" >> "$OUT/runs.txt"
}
for i in 1 2 3; do
    run "q8-resident-$i" "$Q8"
    run "mx-resident-$i" "$MX"
    run "mx-stream2.5g-$i" "$MX" --stream-bytes 2684354560
    run "mx-stream3g-$i" "$MX" --stream-bytes 3221225472
    run "q8-stream6g-$i" "$Q8" --stream-bytes 6442450944
done
cat "$OUT/runs.txt"
