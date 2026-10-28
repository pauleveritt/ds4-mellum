#!/bin/bash
# D1: Mellum decode through ds4 itself, at several prompt lengths (phase 8b).
#   tests/mellum_decode_profile.sh OUTDIR MODEL TOKENS...      e.g. ... 100 2048 8192 32768
# The prompts are T1's (tests/mellum_cli_prefill_abba.sh): the start of speed-bench/promessi_sposi.txt and one
# request, sized by --dump-tokens to the smallest templated count at or above each target.  The context is that
# count plus GEN and 64, rounded up to 1,024.  At each length, after COOL seconds without GPU work:
#   speed:    PER processes, GAP seconds apart, each `ds4 --temp 0 -n GEN`; the first is dropped (it is the
#             block's first after the cooldown); the CLI's "generation: X t/s" line is the result;
#   cbtimes:  one process with DS4_METAL_CB_TIMES=1: driver, queue-wait, GPU time and the gap since the last
#             command buffer's GPU end, for each of the first 400 command buffers;
#   timeline: one process with DS4_METAL_ENCODER_TIMELINE: every dispatch group's kernel and GPU duration
#             (pass boundaries add serialization, so these durations are an upper bound and their shares the
#             point).
# Every process's stdout (the generated text) and stderr are kept; power and thermal state go to events.txt.
set -u
OUT=$1; MODEL=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DS4=${DS4:-$ROOT/ds4}
TEXT=$ROOT/speed-bench/promessi_sposi.txt
PER=${PER:-4}; COOL=${COOL:-45}; GAP=${GAP:-10}; GEN=${GEN:-256}
ASK=$'\n\nSummarize the text above in three sentences.'
mkdir -p "$OUT"

state() {
    echo "$(date +%H:%M:%S) $(pmset -g batt | head -1 | sed 's/Now drawing from //')" \
         "powermode=$(pmset -g | awk '/powermode/{print $2}')" \
         "$(pmset -g therm 2>&1 | tr '\n' ' ' | sed 's/Note: //g; s/ *$//')"
}

count() {
    "$DS4" -m "$MODEL" --prompt-file "$1" --dump-tokens 2>/dev/null | head -1 | python3 -I -c \
        'import json, sys; print(len(json.loads(sys.stdin.read())))'
}

cut_prompt() {
    python3 -I - "$TEXT" "$2" "$ASK" > "$1" <<'PY'
import sys
text = open(sys.argv[1], encoding='utf-8').read()
n = int(sys.argv[2])
head = text[:n]
if n < len(text) and '\n' in head:
    head = head[:head.rindex('\n')]
sys.stdout.write(head.rstrip('\n') + sys.argv[3])
PY
}

run() {   # run LOGBASE [VAR=VALUE ...]: one ds4 process
    local base=$1; shift
    env "$@" "$DS4" -m "$MODEL" -c "$ctx" --temp 0 -n "$GEN" --prompt-file "$prompt" \
        > "$base.out" 2> "$base.err" || { echo "process $base failed"; tail -3 "$base.err"; exit 1; }
}

{ echo "machine: $(sysctl -n machdep.cpu.brand_string) $(sysctl -n hw.memsize) bytes"
  echo "head: $(git -C "$ROOT" rev-parse --short HEAD)  ds4: $(stat -f %Sm "$DS4")"
  echo "PER=$PER COOL=$COOL GAP=$GAP GEN=$GEN"; } > "$OUT/setup.txt"

for target in "$@"; do
    lo=1; hi=$((target * 6)); prompt=$OUT/prompt_$target.txt
    while [ $((hi - lo)) -gt 4 ]; do
        mid=$(((lo + hi) / 2)); cut_prompt "$prompt" $mid
        if [ "$(count "$prompt")" -lt "$target" ]; then lo=$mid; else hi=$mid; fi
    done
    cut_prompt "$prompt" $hi
    ntok=$(count "$prompt")
    ctx=$((((ntok + GEN + 64 + 1023) / 1024) * 1024))
    echo "length $target: $ntok tokens, $(wc -c < "$prompt" | tr -d ' ') bytes, ctx $ctx" | tee -a "$OUT/setup.txt"
    sleep "$COOL"
    echo "state before $target speed: $(state)" >> "$OUT/events.txt"
    for p in $(seq 1 "$PER"); do
        [ "$p" -gt 1 ] && sleep "$GAP"
        run "$OUT/l${target}_speed_p$p"
        tps=$(sed -n 's/.*ds4: Mellum prefill: \([0-9.]*\) t\/s, generation: \([0-9.]*\) t\/s.*/\1 \2/p' \
              "$OUT/l${target}_speed_p$p.err" | tail -1)
        [ -n "$tps" ] || { echo "no timing line in l${target}_speed_p$p.err"; exit 1; }
        echo "$target $ntok $p $tps" >> "$OUT/runs.txt"
    done
    echo "state after $target speed: $(state)" >> "$OUT/events.txt"
    sleep "$GAP"
    run "$OUT/l${target}_cbtimes" DS4_METAL_CB_TIMES=1
    sleep "$GAP"
    rm -f "$OUT/l${target}_timeline.txt"
    run "$OUT/l${target}_timeline" DS4_METAL_ENCODER_TIMELINE="$OUT/l${target}_timeline.txt"
    echo "state after $target anatomy: $(state)" >> "$OUT/events.txt"
done

python3 -I - "$OUT/runs.txt" <<'PY' | tee "$OUT/summary.txt"
import statistics, sys
rows = [l.split() for l in open(sys.argv[1])]
for target in dict.fromkeys(r[0] for r in rows):
    kept = [r for r in rows if r[0] == target and r[2] != '1']
    g = [float(r[4]) for r in kept]
    print('%s (%s tokens): n %d, generation median %.2f t/s, range %.2f-%.2f' %
          (target, kept[0][1], len(g), statistics.median(g), min(g), max(g)))
PY
