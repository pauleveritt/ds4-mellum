#!/bin/bash
# D2: B0, a two-row verify against two decode steps (phase 8b), through ./ds4_test --mellum-b0.
#   tests/mellum_b0.sh OUTDIR MODEL CONTEXTS      e.g. ... 100,2048,8192,32768
# One process per context, COOL seconds without GPU work before each; N rounds of D V V D in each (the first
# dropped by the test).  The test's own lines, its stderr, and power and thermal state before and after each
# process are kept.
set -u
OUT=$1; MODEL=$2; CTXS=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
N=${N:-16}; COOL=${COOL:-45}
mkdir -p "$OUT"

state() {
    echo "$(date +%H:%M:%S) $(pmset -g batt | head -1 | sed 's/Now drawing from //')" \
         "powermode=$(pmset -g | awk '/powermode/{print $2}')" \
         "$(pmset -g therm 2>&1 | tr '\n' ' ' | sed 's/Note: //g; s/ *$//')"
}

{ echo "machine: $(sysctl -n machdep.cpu.brand_string) $(sysctl -n hw.memsize) bytes"
  echo "head: $(git -C "$ROOT" rev-parse --short HEAD)  ds4_test: $(stat -f %Sm "$ROOT/ds4_test")"
  echo "N=$N COOL=$COOL contexts=$CTXS"; } > "$OUT/setup.txt"

for c in ${CTXS//,/ }; do
    sleep "$COOL"
    echo "state before $c: $(state)" >> "$OUT/events.txt"
    (cd "$ROOT" && DS4_TEST_MELLUM_MODEL=$MODEL DS4_TEST_MELLUM_B0_CTX=$c DS4_TEST_MELLUM_B0_N=$N \
        ./ds4_test --mellum-b0) > "$OUT/b0_$c.out" 2> "$OUT/b0_$c.err" ||
        { echo "ds4_test failed at context $c"; tail -5 "$OUT/b0_$c.out" "$OUT/b0_$c.err"; exit 1; }
    echo "state after $c: $(state)" >> "$OUT/events.txt"
    grep '^mellum-b0:' "$OUT/b0_$c.out" | tee -a "$OUT/summary.txt"
done
