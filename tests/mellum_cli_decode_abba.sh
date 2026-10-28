#!/bin/bash
# CLI decode A/B/B/A on D1's prompts (phase 8c; first written for the D3 screens under misc/d-decode/).
#   tests/mellum_cli_decode_abba.sh OUTDIR MODEL PROMPTDIR TOKENS...    (PROMPTDIR holds prompt_<TOKENS>.txt,
#   e.g. tests/mellum_measurements/d1_decode_profile)
# Run from the tree whose metal/ the ds4 binary should load (DS4, default ./ds4).  A_ENV and B_ENV are
# space-separated assignments.  Blocks A B B A of PER processes, COOL s before a block, GAP s between
# processes, the first process of each block dropped; each process `ds4 --temp 0 -n GEN`, the CLI's
# generation line is the result.  Context: prompt tokens + GEN + 64, rounded up to 1,024.  Every process's
# stdout (the generated text) and stderr are kept, runs.txt has every process's rate, and power and thermal
# state before and after each block go to events.txt.
set -u
OUT=$1; MODEL=$2; PD=$3; shift 3
DS4=${DS4:-./ds4}
PER=${PER:-4}; COOL=${COOL:-30}; GAP=${GAP:-10}; GEN=${GEN:-256}
read -r -a A_ARM <<< "${A_ENV-}"
read -r -a B_ARM <<< "${B_ENV-}"
mkdir -p "$OUT"
state() {
    echo "$(date +%H:%M:%S) $(pmset -g batt | head -1 | sed 's/Now drawing from //')" \
         "powermode=$(pmset -g | awk '/powermode/{print $2}')" \
         "$(pmset -g therm 2>&1 | tr '\n' ' ' | sed 's/Note: //g; s/ *$//')"
}
{ echo "machine: $(sysctl -n machdep.cpu.brand_string) $(sysctl -n hw.memsize) bytes"
  echo "tree: $(pwd) head $(git rev-parse --short HEAD), $(git status --porcelain --untracked-files=no | wc -l | tr -d " ") changed tracked files"
  echo "PER=$PER COOL=$COOL GAP=$GAP GEN=$GEN"
  echo "A: ${A_ARM[*]:-(no variables)}"; echo "B: ${B_ARM[*]:-(no variables)}"; } > "$OUT/setup.txt"
for target in "$@"; do
    prompt=$PD/prompt_$target.txt
    ntok=$("$DS4" -m "$MODEL" --prompt-file "$prompt" --dump-tokens 2>/dev/null | head -1 |
           python3 -I -c 'import json, sys; print(len(json.loads(sys.stdin.read())))')
    ctx=$((((ntok + GEN + 64 + 1023) / 1024) * 1024))
    echo "length $target: $ntok tokens, ctx $ctx" >> "$OUT/setup.txt"
    b=0
    for arm in A B B A; do
        b=$((b + 1))
        if [ $arm = A ]; then envs=(${A_ARM[@]+"${A_ARM[@]}"}); else envs=(${B_ARM[@]+"${B_ARM[@]}"}); fi
        sleep "$COOL"
        echo "state before $target $b$arm: $(state)" >> "$OUT/events.txt"
        for p in $(seq 1 "$PER"); do
            [ "$p" -gt 1 ] && sleep "$GAP"
            log=$OUT/l${target}_b$b${arm}_p$p
            env ${envs[@]+"${envs[@]}"} "$DS4" -m "$MODEL" -c "$ctx" --temp 0 -n "$GEN" --prompt-file "$prompt" \
                > "$log.out" 2> "$log.err" || { echo "process $log failed"; exit 1; }
            tps=$(sed -n 's/.*generation: \([0-9.]*\) t\/s.*/\1/p' "$log.err" | tail -1)
            echo "$target $ntok $arm $b $p $tps" >> "$OUT/runs.txt"
        done
        echo "state after $target $b$arm: $(state)" >> "$OUT/events.txt"
    done
done
python3 -I - "$OUT/runs.txt" <<'PY' | tee "$OUT/summary.txt"
import statistics, sys
rows = [l.split() for l in open(sys.argv[1])]
for t in dict.fromkeys(r[0] for r in rows):
    kept = [r for r in rows if r[0] == t and r[4] != '1']
    med = {}
    for arm in 'AB':
        v = [float(r[5]) for r in kept if r[2] == arm]
        med[arm] = statistics.median(v)
        print('%s (%s tokens) %s: n %d, generation median %.2f t/s, range %.2f-%.2f' %
              (t, kept[0][1], arm, len(v), med[arm], min(v), max(v)))
    print('%s: B / A %.3f' % (t, med['B'] / med['A']))
PY
