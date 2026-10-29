#!/bin/sh
# The preregistered A/B (../2026-10-29-mellum-stream-decode-prereg.md): arms A and B in their own built
# worktrees, A B B A A B at 2.5 GiB, 512 tokens, a 20 s rest before each run; then, not judged, a resident run
# of B and one 3 GiB run of each arm.  Each run starts in its arm's tree (the engine compiles metal/ from the
# working directory).
#   MX=... A=/path/to/arm-A-tree B=/path/to/arm-B-tree OUT=/path/to/this/dir sh run_ab.sh
set -u
: "${MX:?}" "${A:?}" "${B:?}" "${OUT:?}"
P=docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/workload-prompt.txt
run() {   # name tree extra-args
    name=$1; tree=$2; shift 2
    [ -e "$OUT/$name.measure.json" ] && { echo "run_ab: $name exists" >&2; exit 1; }
    sleep 20
    (cd "$tree" && tools/mellum_s4_measure --model "$MX" --prompt "$tree/$P" --gen 512 --context 32768 \
        --out "$OUT/$name.measure.json" "$@" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr")
    echo "$name rc $?" >> "$OUT/runs.txt"
}
for x in A1 B1 B2 A2 A3 B3; do
    arm=$(printf %s "$x" | cut -c1); [ "$arm" = A ] && tree=$A || tree=$B
    run "$x" "$tree" --stream-bytes 2684354560
done
run B-resident "$B"
run A-3g "$A" --stream-bytes 3221225472
run B-3g "$B" --stream-bytes 3221225472
cat "$OUT/runs.txt"
