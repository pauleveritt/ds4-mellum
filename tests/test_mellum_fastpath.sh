#!/bin/sh
# The release prefill's fast paths (batched and tiled from 32 rows) at chunk and window edges, with the model.
# Model gated: it needs DS4_TEST_MELLUM_MODEL, the real 2.1 GGUF, and Metal; without it the test says so and
# passes.  `make test-mellum-fastpath`.  Runs one model process at a time, about 60 of them.
#
# The tokens are prefixes of tests/mellum_heldout_long.tokens (Mellum-templated, 6,000 ids) of 31, 32, 1023,
# 1024, 1025 and 2048 tokens, prefilled in chunks of 1024, 512 and 100 rows (DS4_QWEN4_PREFILL_CHUNK).  Each run
# is traced at the chunk and window edges it has: 0, 31, 99, 100, 511, 512, 1022, 1023, 1024 and the last.
#
#   a. untraced against traced: a traced run waits after every layer, so it cannot see a hazard between the
#      layers of the one command buffer an untraced run uses; their session logits must be byte-equal;
#   b. the release prefill against the exact one (DS4_MELLUM_EXACT_PREFILL=1, which is byte-equal to decode):
#      byte-equal when every chunk is under 32 rows, else within the oracle's rule
#      (tests/mellum_logits_compare.py with tests/mellum_oracle_compare.sh's defaults);
#   c. decode steps that read a release-prefilled cache (mode prefill:N, eight evals after a sync of N, across
#      the window edge at N = 1023, 1024, 1025, and N = 32), against the exact path at those eight positions,
#      within the same rule;
#   d. greedy tokens from a release-prefilled cache (1,024 tokens, 32 greedy): under the exact path, teacher-
#      forced, each must be within DLOGP of the top log-probability at its position.  How many leading tokens
#      equal the exact path's own greedy tokens is printed, not checked (router flips at near-ties are legitimate
#      there; MELLUM.md, "Oracle caps").
set -u
cd "$(dirname "$0")/.." || exit 2
if [ -z "${DS4_TEST_MELLUM_MODEL:-}" ]; then
    echo "mellum-fastpath: DS4_TEST_MELLUM_MODEL unset, skipped"
    exit 0
fi
[ -x tests/mellum_trace ] || { echo "mellum-fastpath: build tests/mellum_trace first (make tests/mellum_trace)" >&2; exit 2; }
M=$DS4_TEST_MELLUM_MODEL
TOL=10 KLTOL=0.01 KLMAX=1.0 DLOGP=1.0
dir=$(mktemp -d "${TMPDIR:-/tmp}/mellum-fastpath.XXXXXX") || exit 2
trap 'rm -rf "$dir"' EXIT
fail=0
check() { # description, command...
    desc=$1; shift
    if "$@"; then echo "ok   $desc"; else echo "FAIL $desc"; fail=1; fi
}
ids() { grep -v '^#' tests/mellum_heldout_long.tokens | tr -s ' \n' '\n' | grep -v '^$'; }
prefix_file() { ids | head -n "$1" | tr '\n' ' ' > "$2"; }
edges() { # positions of the edge list inside n tokens, comma-separated
    python3 -I -c "import sys; n = int(sys.argv[1]); print(','.join(str(p) for p in sorted({0, 31, 99, 100, 511, 512, 1022, 1023, 1024, n - 1}) if p < n))" "$1"
}
# trace OUT TOKENS MODE CHUNK POSITIONS [exact]: one run; POSITIONS empty is an untraced run
trace() {
    if [ -n "${6:-}" ]; then
        DS4_MELLUM_EXACT_PREFILL=1 DS4_QWEN4_PREFILL_CHUNK=$4 DS4_MELLUM_TRACE=$1 DS4_MELLUM_TRACE_AT=$5 \
            tests/mellum_trace "$M" "$2" "$3" > "$1.out" 2> "$1.err"
    else
        DS4_QWEN4_PREFILL_CHUNK=$4 DS4_MELLUM_TRACE=$1 DS4_MELLUM_TRACE_AT=$5 \
            tests/mellum_trace "$M" "$2" "$3" > "$1.out" 2> "$1.err"
    fi || { echo "FAIL the driver failed on $2 $3 chunk $4:"; tail -3 "$1.err"; fail=1; return 1; }
}
compare() { # A B POSITIONS EXACT
    python3 -I tests/mellum_logits_compare.py "$1" "$2" "$3" "$4" "$TOL" "$KLTOL" "$KLMAX" "$DLOGP"
}

for n in 31 32 1023 1024 1025 2048; do
    f="$dir/t$n.tokens"
    prefix_file "$n" "$f"
    pos=$(edges "$n")
    trace "$dir/e$n" "$f" prefill 1024 "$pos" exact || continue
    for chunk in 1024 512 100; do
        trace "$dir/u$n.$chunk" "$f" prefill "$chunk" "" || continue
        trace "$dir/r$n.$chunk" "$f" prefill "$chunk" "$pos" || continue
        check "a. $n tokens, chunk $chunk: untraced session logits equal the traced run's" \
            cmp -s "$dir/u$n.$chunk.session.logits.f32" "$dir/r$n.$chunk.session.logits.f32"
        exact=0
        [ "$n" -lt 32 ] && exact=1
        r=$(compare "$dir/r$n.$chunk" "$dir/e$n" "$pos" "$exact")
        rc=$?
        check "b. $n tokens, chunk $chunk, release vs exact at $pos: $r" [ "$rc" -eq 0 ]
    done
done

for split in 32 1023 1024 1025; do
    n=$((split + 8))
    f="$dir/s$split.tokens"
    prefix_file "$n" "$f"
    pos=$(python3 -I -c "import sys; s = int(sys.argv[1]); print(','.join(str(p) for p in range(s, s + 8)))" "$split")
    trace "$dir/se$split" "$f" prefill 1024 "$pos" exact || continue
    for chunk in 1024 512; do
        trace "$dir/sr$split.$chunk" "$f" "prefill:$split" "$chunk" "$pos" || continue
        r=$(compare "$dir/sr$split.$chunk" "$dir/se$split" "$pos" 0)
        rc=$?
        check "c. sync $split (chunk $chunk), then 8 decode steps vs exact: $r" [ "$rc" -eq 0 ]
    done
done

# d. greedy from a release-prefilled cache, checked under the exact path, teacher-forced
f="$dir/g.tokens"
prefix_file 1024 "$f"
MELLUM_TRACE_GREEDY=32 DS4_MELLUM_TRACE="$dir/gr" tests/mellum_trace "$M" "$f" prefill > "$dir/gr.out" 2>&1 &&
MELLUM_TRACE_GREEDY=32 DS4_MELLUM_EXACT_PREFILL=1 DS4_MELLUM_TRACE="$dir/ge" tests/mellum_trace "$M" "$f" prefill \
    > "$dir/ge.out" 2>&1 || { echo "FAIL d. a greedy run failed"; fail=1; }
fast=$(sed -n 's/^mellum_trace: greedy 32://p' "$dir/gr.out")
slow=$(sed -n 's/^mellum_trace: greedy 32://p' "$dir/ge.out")
if [ -n "$fast" ] && [ -n "$slow" ]; then
    { cat "$f"; echo "$fast"; } > "$dir/gf.tokens"
    pos=$(python3 -I -c "print(','.join(str(p) for p in range(1023, 1023 + 32)))")
    if trace "$dir/gt" "$dir/gf.tokens" prefill 1024 "$pos" exact; then
        r=$(python3 -I - "$dir/gt" "$pos" "$fast" "$slow" "$DLOGP" <<'PY'
import math, struct, sys
prefix, positions, fast, slow, dlogp = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(), sys.argv[4].split(), float(sys.argv[5])
worst, at = 0.0, -1
for i, p in enumerate(positions):
    raw = open('%s.p%s.logits.f32' % (prefix, p), 'rb').read()
    v = struct.unpack('<%df' % (len(raw) // 4), raw)
    m = max(v)
    lse = m + math.log(sum(math.exp(t - m) for t in v))
    gap = (m - lse) - (v[int(fast[i])] - lse)
    if gap > worst: worst, at = gap, i
same = next((i for i, (a, b) in enumerate(zip(fast, slow)) if a != b), len(fast))
print('%s %d of 32 tokens match the exact greedy run before the first difference; worst log p gap to the '
      'top under the exact path %.3f (token %d)' % ('ok' if worst <= dlogp else 'FAIL', same, worst, at))
PY
)
        case "$r" in ok*) echo "ok   d. greedy from a release-prefilled cache: ${r#ok }" ;;
                     *) echo "FAIL d. greedy from a release-prefilled cache: ${r#FAIL }"; fail=1 ;; esac
    fi
fi

[ "$fail" -eq 0 ] && echo "mellum-fastpath: PASS" || echo "mellum-fastpath: FAILED"
exit "$fail"
