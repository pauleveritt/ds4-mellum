#!/bin/bash
# T1: Mellum CLI prefill, E1 on (the default) against DS4_MELLUM_MOE_NAX=0, through ds4 itself.
#   tests/mellum_cli_prefill_abba.sh OUTDIR MODEL TOKENS...      e.g. ... 2048 8192 32768
# The prompt is the start of speed-bench/promessi_sposi.txt (cut at a line end) and one request, sized so its
# templated token count (--dump-tokens) is the smallest at or above each target.  The context is that count plus
# 64, rounded up to 1,024.  At each length the arms run in blocks A, B, B, A (A = DS4_MELLUM_MOE_NAX=0, B = the
# default; A_ENV and B_ENV, space-separated assignments, replace them, A_NAME and B_NAME their labels; with no B in
# ARMS, arm A defaults to no variables, the build as it is), COOL
# seconds without GPU work before a block and GAP seconds between its PER processes.  ARMS sets the block order
# (letters A and B; default ABBA); ARMS=AA measures one build, arm A only, with no ratio.  ds4 runs one
# prefill per process (after its own GPU warmup), so a block's first process is dropped, not a process's first run.
# Every process's stderr is kept; power and thermal state go to events.txt before and after each block.
set -u
OUT=$1; MODEL=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DS4=${DS4:-$ROOT/ds4}
TEXT=$ROOT/speed-bench/promessi_sposi.txt
PER=${PER:-4}; COOL=${COOL:-45}; GAP=${GAP:-10}
ARMS=${ARMS:-ABBA}
[[ $ARMS =~ ^[AB]+$ ]] || { echo "ARMS must be letters A and B only, got '$ARMS'" >&2; exit 2; }
# T1's arm A (E1 off) is a comparison's default; a single-arm run (no B) measures the build as it is
if [[ $ARMS == *B* ]]; then A_DEFAULT=DS4_MELLUM_MOE_NAX=0; else A_DEFAULT=; fi
read -r -a A_ARM <<< "${A_ENV-$A_DEFAULT}"
read -r -a B_ARM <<< "${B_ENV-}"
export A_NAME=${A_NAME:-${A_DEFAULT:-this build}} B_NAME=${B_NAME:-E1 on (default)}
ASK=$'\n\nSummarize the text above in three sentences.'
mkdir -p "$OUT"

state() {
    echo "$(date +%H:%M:%S) $(pmset -g batt | head -1 | sed 's/Now drawing from //')" \
         "powermode=$(pmset -g | awk '/powermode/{print $2}')" \
         "$(pmset -g therm 2>&1 | tr '\n' ' ' | sed 's/Note: //g; s/ *$//')"
}

# templated token count of a prompt file
count() {
    "$DS4" -m "$MODEL" --prompt-file "$1" --dump-tokens 2>/dev/null | head -1 | python3 -I -c \
        'import json, sys; print(len(json.loads(sys.stdin.read())))'
}

# write the first $2 characters of the text, cut back to a line end, and the request
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

{ echo "machine: $(sysctl -n machdep.cpu.brand_string) $(sysctl -n hw.memsize) bytes"
  echo "head: $(git -C "$ROOT" rev-parse --short HEAD)  ds4: $(stat -f %Sm "$DS4")"
  echo "PER=$PER COOL=$COOL GAP=$GAP ARMS=$ARMS"
  echo "A ($A_NAME): ${A_ARM[*]:-(no variables)}"; echo "B ($B_NAME): ${B_ARM[*]:-(no variables)}"; } > "$OUT/setup.txt"

for target in "$@"; do
    # bisect the character count: about 2.8 characters per token in this text
    lo=1; hi=$((target * 6)); prompt=$OUT/prompt_$target.txt
    while [ $((hi - lo)) -gt 16 ]; do
        mid=$(((lo + hi) / 2)); cut_prompt "$prompt" $mid
        if [ "$(count "$prompt")" -lt "$target" ]; then lo=$mid; else hi=$mid; fi
    done
    cut_prompt "$prompt" $hi
    ntok=$(count "$prompt")
    ctx=$((((ntok + 64 + 1023) / 1024) * 1024))
    echo "length $target: $ntok tokens, $(wc -c < "$prompt" | tr -d ' ') bytes, ctx $ctx" | tee -a "$OUT/setup.txt"
    b=0
    for arm in $(fold -w1 <<< "$ARMS"); do
        b=$((b + 1))
        if [ $arm = A ]; then envs=(${A_ARM[@]+"${A_ARM[@]}"}); else envs=(${B_ARM[@]+"${B_ARM[@]}"}); fi
        sleep "$COOL"
        echo "state before $target $b$arm: $(state)" >> "$OUT/events.txt"
        for p in $(seq 1 "$PER"); do
            [ "$p" -gt 1 ] && sleep "$GAP"
            log=$OUT/l${target}_b$b${arm}_p$p.err
            env ${envs[@]+"${envs[@]}"} "$DS4" -m "$MODEL" -c "$ctx" --temp 0 -n 1 --prompt-file "$prompt" \
                > /dev/null 2> "$log" || { echo "process $target $b$arm $p failed"; tail -3 "$log"; exit 1; }
            tps=$(sed -n 's/.*ds4: Mellum prefill: \([0-9.]*\) t\/s.*/\1/p' "$log" | tail -1)
            [ -n "$tps" ] || { echo "no prefill line in $log"; exit 1; }
            echo "$target $ntok $arm $b $p $tps" >> "$OUT/runs.txt"
        done
        echo "state after $target $b$arm: $(state)" >> "$OUT/events.txt"
    done
done

python3 -I - "$OUT/runs.txt" <<'PY' | tee "$OUT/summary.txt"
import os, statistics, sys
rows = [l.split() for l in open(sys.argv[1])]
for target in dict.fromkeys(r[0] for r in rows):
    kept = [r for r in rows if r[0] == target and r[4] != '1']
    med = {}
    for arm, name in (('A', os.environ['A_NAME']), ('B', os.environ['B_NAME'])):
        v = [float(r[5]) for r in kept if r[2] == arm]
        if not v:
            continue
        med[arm] = statistics.median(v)
        print('%s (%s tokens) %s: n %d, median %.0f t/s, range %.0f-%.0f' %
              (target, kept[0][1], name, len(v), med[arm], min(v), max(v)))
    if len(med) == 2:
        print('%s: B / A median %.3f' % (target, med['B'] / med['A']))
PY
