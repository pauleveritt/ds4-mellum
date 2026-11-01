#!/bin/bash
# What a k-row verify costs, through ./ds4_test --mellum-bk (k 1, 2, 3, 4, 6, 8).
#   tests/mellum_bk.sh OUTDIR MODEL CONTEXTS      e.g. ... 2048,8192,32768
# One process per context, COOL seconds without GPU work before each; N rounds of D C C D per k (the first
# dropped by the test).  The test's own lines, its stderr, and power and thermal state before and after each
# process are kept.  TIMELINE=1 also writes the engine's per-dispatch GPU timeline (DS4_METAL_ENCODER_TIMELINE),
# which slows every dispatch a little: its runs are for the anatomy, not the ratios.
set -u
OUT=$1; MODEL=$2; CTXS=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=${N:-16}; COOL=${COOL:-45}; TIMELINE=${TIMELINE:-0}
mkdir -p "$OUT"

state() {
    echo "$(date +%H:%M:%S) $(pmset -g batt | head -1 | sed 's/Now drawing from //')" \
         "powermode=$(pmset -g | awk '/powermode/{print $2}')" \
         "$(pmset -g therm 2>&1 | tr '\n' ' ' | sed 's/Note: //g; s/ *$//')"
}

{ echo "machine: $(sysctl -n machdep.cpu.brand_string) $(sysctl -n hw.memsize) bytes"
  echo "head: $(git -C "$ROOT" rev-parse --short HEAD)  ds4_test: $(stat -f %Sm "$ROOT/ds4_test")"
  echo "model: $MODEL"
  echo "N=$N COOL=$COOL TIMELINE=$TIMELINE contexts=$CTXS"; } > "$OUT/setup.txt"

for c in ${CTXS//,/ }; do
    sleep "$COOL"
    echo "state before $c: $(state)" >> "$OUT/events.txt"
    tl=(); [ "$TIMELINE" = 1 ] && tl=(env DS4_METAL_ENCODER_TIMELINE="$OUT/timeline_$c.txt")
    (cd "$ROOT" && ${tl[@]+"${tl[@]}"} env DS4_TEST_MELLUM_MODEL="$MODEL" DS4_TEST_MELLUM_BK_CTX="$c" DS4_TEST_MELLUM_BK_N="$N" \
        ./ds4_test --mellum-bk) > "$OUT/bk_$c.out" 2> "$OUT/bk_$c.err" ||
        { echo "ds4_test failed at context $c"; tail -5 "$OUT/bk_$c.out" "$OUT/bk_$c.err"; exit 1; }
    echo "state after $c: $(state)" >> "$OUT/events.txt"
    grep '^mellum-bk: context' "$OUT/bk_$c.out" | tee -a "$OUT/summary.txt"
done
