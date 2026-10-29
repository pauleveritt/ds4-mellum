#!/bin/sh
# C6's 27B arm (../2026-10-28-mellum-s6-c-preregistration.md, addendum of 2026-10-29): the 27B on the
# flattened workload as one user turn, greedy, 512 tokens with EOS disallowed, n=3, through ds4-engine's
# engine at its pin (ds4-bench).  The metric is the CSV's gen_steady_tps: tokens 1..511 over the time from
# the first generated token to the last.  Run on the owner's 36 GB M4 Max, apps closed, on AC power.
#   DS4=/path/to/ds4-engine/external/ds4 MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf OUT=/path/to/new/dir \
#       sh run_c6_27b.sh
set -u
: "${DS4:?}" "${MODEL:?}" "${OUT:?}"
PIN=ebdcb0f
SHA=e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520
P=$(cd "$(dirname "$0")" && pwd)/workload-prompt.txt
[ -e "$OUT" ] && { echo "run_c6_27b: $OUT exists" >&2; exit 1; }
mkdir -p "$OUT" || exit 1
rev=$(git -C "$DS4" rev-parse --short=7 HEAD) || exit 1
[ "$rev" = "$PIN" ] || { echo "run_c6_27b: engine at $rev, want $PIN" >&2; exit 1; }
[ "$(shasum -a 256 "$MODEL" | cut -c1-64)" = "$SHA" ] || { echo "run_c6_27b: model sha256 differs" >&2; exit 1; }
{ echo "engine $rev"; echo "model $SHA"; sysctl -n machdep.cpu.brand_string hw.memsize; sw_vers; } > "$OUT/machine.txt"
(cd "$DS4" && make ds4-bench) > "$OUT/build.log" 2>&1 || { echo "run_c6_27b: build failed" >&2; exit 1; }
failed=0
for i in 1 2 3; do   # from inside $DS4: the engine compiles metal/ from the working directory
    sleep 20
    (cd "$DS4" && ./ds4-bench -m "$MODEL" --chat-prompt-file "$P" --ctx-start 2573 --ctx-max 2573 \
        --ctx-alloc 32768 --gen-tokens 512 --csv "$OUT/run$i.csv" > "$OUT/run$i.stdout" 2> "$OUT/run$i.stderr")
    rc=$?; echo "run$i rc $rc" >> "$OUT/runs.txt"; [ "$rc" = 0 ] || failed=1
done
cat "$OUT/runs.txt"
exit "$failed"
