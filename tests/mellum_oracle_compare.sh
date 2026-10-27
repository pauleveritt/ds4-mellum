#!/bin/sh
# Compare this engine's Mellum forward pass with the old port's (the oracle):
#
#   tests/mellum_oracle_compare.sh ORACLE_DIR MODEL.gguf [GREEDY_TOKENS]
#
# ORACLE_DIR is a tree of the old port's mellum21-stage1 with its trace driver built:
#
#   git fetch <ds4-engine>/.git/worktrees/p42-2d-mellum-fit/modules/external/ds4 \
#       'refs/heads/mellum21-stage1:refs/old/mellum21-stage1'
#   git archive refs/old/mellum21-stage1 | tar -x -C ORACLE_DIR
#   make -C ORACLE_DIR tests/mellum_trace
#
# For each of tests/mellum_oracle_{short,1100,2200}.tokens, in prefill (one sync) and decode (one
# eval per token) mode, both engines' tests/mellum_trace write the logits at up to 64 positions
# spread over the file (98,304 f32 each) and the session's last-row logits.  Decode must equal the
# old engine's byte for byte (cmp).  So must a prefill whose chunks are all below 32 rows (the
# short file): from 32 rows this engine's projections and experts run as batched and tiled GEMMs
# on half-staged activations (ds4.c, MELLUM_TILED_MIN_ROWS), so a longer prefill is compared
# within a tolerance instead: the same top-1 token at every traced position, max|d logit| <= TOL
# (MELLUM_ORACLE_TOL, default 10) and mean KL divergence <= KLTOL (MELLUM_ORACLE_KL, default 0.01).
# This engine's prefill against its own decode is held to the same rule.  Every prefill also runs
# under the exact-prefill diagnostic (DS4_MELLUM_EXACT_PREFILL=1, "prefill-exact"), which must be
# byte-equal to the old engine and to its own decode on all three files.  Then GREEDY_TOKENS
# (default 64) greedy tokens from the short prompt, one decode run per token in each engine: every
# step's argmax and logits must be equal.  Every comparison prints max|d logit| and top-1 agreement.
#
# Runs one model process at a time.  Exit 0 when everything passes, 1 otherwise, 2 on bad usage.
set -u
[ $# -ge 2 ] || { echo "usage: $0 ORACLE_DIR MODEL.gguf [GREEDY_TOKENS]" >&2; exit 2; }
ORACLE=$(cd "$1" && pwd) || exit 2
MODEL=$2
GREEDY=${3:-64}
TOL=${MELLUM_ORACLE_TOL:-10}
KLTOL=${MELLUM_ORACLE_KL:-0.01}
cd "$(dirname "$0")/.." || exit 2
ROOT=$(pwd)
[ -x tests/mellum_trace ] || { echo "build tests/mellum_trace first (make tests/mellum_trace)" >&2; exit 2; }
[ -x "$ORACLE/tests/mellum_trace" ] || { echo "build $ORACLE/tests/mellum_trace first" >&2; exit 2; }

dir=$(mktemp -d "${TMPDIR:-/tmp}/mellum-oracle.XXXXXX") || exit 2
trap 'rm -rf "$dir"' EXIT
fail=0

# max|d| and argmax agreement of two logits files, for a report when they differ
diffstat() {
    python3 -I - "$1" "$2" <<'PY'
import struct, sys
a = open(sys.argv[1], 'rb').read(); b = open(sys.argv[2], 'rb').read()
n = len(a) // 4
x = struct.unpack('<%df' % n, a); y = struct.unpack('<%df' % n, b)
d = max(abs(p - q) for p, q in zip(x, y))
print('max|d logit| %.3e, argmax %d vs %d' % (d, max(range(n), key=x.__getitem__), max(range(n), key=y.__getitem__)))
PY
}

# compare A B POSITIONS EXACT: the logits of two traced runs at every position, and their session
# logits (tests/mellum_logits_compare.py has the rules).  Prints one summary and returns 1 on failure.
compare() {
    python3 -I "$ROOT/tests/mellum_logits_compare.py" "$1" "$2" "$3" "$4" "$TOL" "$KLTOL"
}

# run ENGINE_DIR TOKENS MODE OUT_PREFIX [POSITIONS]: the session logits land in
# OUT_PREFIX.session.logits.f32 and, with POSITIONS, each position's in OUT_PREFIX.pN.logits.f32;
# prints the driver's argmax.  Each engine runs from its own tree, which holds its Metal sources.
run() {
    (cd "$1" && DS4_MELLUM_TRACE="$4" DS4_MELLUM_TRACE_AT="${5:-}" "$1/tests/mellum_trace" "$MODEL" "$2" "$3") \
        > "$4.out" 2> "$4.err" || { echo "FAIL $1 $3 on $2: the driver failed:" >&2; tail -3 "$4.err" >&2; return 1; }
    sed -n 's/.*argmax \([0-9]*\),.*/\1/p' "$4.out" | head -1
}

for t in short 1100 2200; do
    tokens="$ROOT/tests/mellum_oracle_$t.tokens"
    n=$(grep -v '^#' "$tokens" | wc -w | tr -d ' ')
    positions=$(python3 -I -c "n = $n; k = min(64, n); print(','.join(str(round(i * (n - 1) / (k - 1))) for i in range(k)))")
    tiled=0
    [ "$n" -ge 32 ] && tiled=1
    # this engine's prefill twice: under the exact-prefill diagnostic it must equal the old engine byte for
    # byte; without it, the release path, within the tolerance when its chunks take the tiled paths
    for mode in prefill-exact prefill decode; do
        drv=$mode
        [ "$mode" = prefill-exact ] && drv=prefill
        if [ "$mode" = prefill-exact ]; then
            (export DS4_MELLUM_EXACT_PREFILL=1; run "$ROOT" "$tokens" "$drv" "$dir/new.$t.$mode" "$positions") > /dev/null ||
                { fail=1; continue; }
        else
            run "$ROOT" "$tokens" "$drv" "$dir/new.$t.$mode" "$positions" > /dev/null || { fail=1; continue; }
        fi
        [ -e "$dir/old.$t.$drv.session.logits.f32" ] ||
            run "$ORACLE" "$tokens" "$drv" "$dir/old.$t.$drv" "$positions" > /dev/null || { fail=1; continue; }
        exact=1
        [ "$mode" = prefill ] && [ "$tiled" -eq 1 ] && exact=0
        r=$(compare "$dir/new.$t.$mode" "$dir/old.$t.$drv" "$positions" "$exact") || fail=1
        echo "$(echo "$r" | cut -c1-4) $t $mode vs the old engine, $(echo "$r" | cut -c6-)"
    done
    r=$(compare "$dir/new.$t.prefill-exact" "$dir/new.$t.decode" "$positions" 1) || fail=1
    echo "$(echo "$r" | cut -c1-4) $t prefill-exact vs decode in this engine, $(echo "$r" | cut -c6-)"
    r=$(compare "$dir/new.$t.prefill" "$dir/new.$t.decode" "$positions" $((1 - tiled))) || fail=1
    echo "$(echo "$r" | cut -c1-4) $t prefill vs decode in this engine, $(echo "$r" | cut -c6-)"
done

# greedy tokens from the short prompt, decode mode, one run per token in each engine
grep -v '^#' tests/mellum_oracle_short.tokens | tr -s ' \n' ' ' > "$dir/greedy.tokens"
same=0
generated=""
i=0
while [ "$i" -lt "$GREEDY" ]; do
    a=$(run "$ROOT" "$dir/greedy.tokens" decode "$dir/gn") || { fail=1; break; }
    b=$(run "$ORACLE" "$dir/greedy.tokens" decode "$dir/go") || { fail=1; break; }
    if [ "$a" != "$b" ] || ! cmp -s "$dir/gn.session.logits.f32" "$dir/go.session.logits.f32"; then
        echo "FAIL greedy token $i: this engine $a, the old engine $b: $(diffstat "$dir/gn.session.logits.f32" "$dir/go.session.logits.f32")"
        fail=1
        break
    fi
    same=$((same + 1))
    generated="$generated $a"
    printf ' %s' "$a" >> "$dir/greedy.tokens"
    i=$((i + 1))
done
if [ "$same" -eq "$GREEDY" ]; then
    echo "ok   greedy: $same of $GREEDY tokens equal, with equal logits at every step:$generated"
else
    echo "FAIL greedy: $same of $GREEDY tokens equal before the first difference:$generated"
    fail=1
fi

[ "$fail" -eq 0 ] && echo "mellum-oracle: PASS" || echo "mellum-oracle: FAILED"
exit "$fail"
