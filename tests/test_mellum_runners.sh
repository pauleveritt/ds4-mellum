#!/bin/bash
# Model-free checks of the two A/B/B/A runners (tests/mellum_cli_prefill_abba.sh, tests/mellum_cli_decode_abba.sh)
# against a stub ds4: --dump-tokens answers a JSON list of (prompt bytes / 3) ids, any other call writes the timing
# line both runners parse.  The default block order is A B B A; ARMS sets another (A and B only), and ARMS=AA is
# the single-arm mode: one build, arm A only, no ratio.  DS4_ARGS adds arguments to every measured process (the
# streamed rows: --ssd-streaming and a cache size), recorded in setup.txt.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }

cat > "$T/ds4" <<'STUB'
#!/bin/bash
prompt=
all="$*"
while [ $# -gt 0 ]; do
    case $1 in --prompt-file) prompt=$2; shift ;; --dump-tokens) dump=1 ;; esac
    shift
done
if [ -n "${dump:-}" ]; then
    python3 -I -c 'import json, sys; print(json.dumps(list(range(int(sys.argv[1]) // 3))))' "$(wc -c < "$prompt")"
else
    echo "ds4: Mellum prefill: 1000.00 t/s, generation: 100.00 t/s" >&2
    echo "stub env: DS4_MELLUM_MOE_NAX=${DS4_MELLUM_MOE_NAX-unset}" >&2
    echo "stub args: $all" >&2
fi
STUB
chmod +x "$T/ds4"
mkdir -p "$T/prompts"
head -c 3000 "$ROOT/speed-bench/promessi_sposi.txt" > "$T/prompts/prompt_500.txt"

run() {   # run NAME RUNNER OUT [VAR=VALUE ...]
    local name=$1 runner=$2 out=$3; shift 3
    if [ "$runner" = prefill ]; then
        env DS4="$T/ds4" COOL=0 GAP=0 PER=4 "$@" "$ROOT/tests/mellum_cli_prefill_abba.sh" "$out" model 500
    else
        env DS4="$T/ds4" COOL=0 GAP=0 PER=4 "$@" "$ROOT/tests/mellum_cli_decode_abba.sh" "$out" model "$T/prompts" 500
    fi > "$T/$name.log" 2>&1
}

for runner in prefill decode; do
    run "$runner-default" $runner "$T/$runner-default"
    rows=$(awk '{print $3}' "$T/$runner-default/runs.txt" 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ';')
    check "$runner: default order runs both arms, 8 processes each" "[ \"$rows\" = ' 8 A; 8 B;' ]"
    check "$runner: default summary has the ratio" "grep -q 'B / A' '$T/$runner-default/summary.txt'"

    run "$runner-single" $runner "$T/$runner-single" ARMS=AA
    rows=$(awk '{print $3}' "$T/$runner-single/runs.txt" 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ';')
    check "$runner: ARMS=AA runs arm A only, 8 processes" "[ \"$rows\" = ' 8 A;' ]"
    check "$runner: ARMS=AA summary has n 6 for arm A" "grep -q ': n 6, ' '$T/$runner-single/summary.txt'"
    check "$runner: ARMS=AA summary has no B line and no ratio" \
        "! grep -q 'B / A\| B: \|) B\b' '$T/$runner-single/summary.txt'"
    check "$runner: ARMS=AA measures the build as it is, with no variables set" \
        "! grep -qh 'DS4_MELLUM_MOE_NAX=[^u]' '$T/$runner-single'/*.err && grep -q '^A.*(no variables)' '$T/$runner-single/setup.txt'"

    run "$runner-args" $runner "$T/$runner-args" ARMS=AA DS4_ARGS="--ssd-streaming --ssd-streaming-cache-experts 2GB"
    errs=$(ls "$T/$runner-args"/*.err 2>/dev/null | wc -l | tr -d ' ')
    with=$(grep -l 'stub args: .*--ssd-streaming --ssd-streaming-cache-experts 2GB' "$T/$runner-args"/*.err 2>/dev/null | wc -l | tr -d ' ')
    check "$runner: DS4_ARGS reaches every measured process ($with of $errs)" "[ $errs -gt 0 ] && [ $with -eq $errs ]"
    check "$runner: DS4_ARGS is recorded in setup.txt" \
        "grep -q 'ds4 args: --ssd-streaming --ssd-streaming-cache-experts 2GB' '$T/$runner-args/setup.txt'"
    check "$runner: without DS4_ARGS no measured process streams" "! grep -qh 'stub args: .*--ssd-streaming' '$T/$runner-single'/*.err"

    run "$runner-bad" $runner "$T/$runner-bad" ARMS=AXA
    rc=$?
    check "$runner: ARMS=AXA is refused, naming ARMS" "[ $rc -ne 0 ] && grep -q 'ARMS' '$T/$runner-bad.log'"
done

[ $fails -eq 0 ] && echo "mellum-runners: PASS" || { echo "mellum-runners: $fails FAILED"; exit 1; }
