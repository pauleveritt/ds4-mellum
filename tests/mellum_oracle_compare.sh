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
# eval per token) mode, both engines' tests/mellum_trace write the session's last-row logits
# (98,304 f32), and the two files must be equal byte for byte (cmp).  This engine's prefill and
# decode files must also be equal to each other.  Then GREEDY_TOKENS (default 64) greedy tokens
# from the short prompt, one decode run per token in each engine: every step's argmax and logits
# must be equal.  A difference prints max|d logit| and whether the argmax agrees.
#
# Runs one model process at a time.  Exit 0 when everything is equal, 1 otherwise, 2 on bad usage.
set -u
[ $# -ge 2 ] || { echo "usage: $0 ORACLE_DIR MODEL.gguf [GREEDY_TOKENS]" >&2; exit 2; }
ORACLE=$(cd "$1" && pwd) || exit 2
MODEL=$2
GREEDY=${3:-64}
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

# run ENGINE_DIR TOKENS MODE OUT_PREFIX: the session logits land in OUT_PREFIX.session.logits.f32;
# prints the driver's argmax.  Each engine runs from its own tree, which holds its Metal sources.
run() {
    (cd "$1" && DS4_MELLUM_TRACE="$4" "$1/tests/mellum_trace" "$MODEL" "$2" "$3") > "$4.out" 2> "$4.err" || {
        echo "FAIL $1 $3 on $2: the driver failed:" >&2; tail -3 "$4.err" >&2; return 1; }
    sed -n 's/.*argmax \([0-9]*\),.*/\1/p' "$4.out" | head -1
}

for t in short 1100 2200; do
    tokens="$ROOT/tests/mellum_oracle_$t.tokens"
    for mode in prefill decode; do
        run "$ROOT" "$tokens" "$mode" "$dir/new.$t.$mode" > /dev/null || { fail=1; continue; }
        run "$ORACLE" "$tokens" "$mode" "$dir/old.$t.$mode" > /dev/null || { fail=1; continue; }
        if cmp -s "$dir/new.$t.$mode.session.logits.f32" "$dir/old.$t.$mode.session.logits.f32"; then
            echo "ok   $t $mode: session logits equal the old engine's (cmp)"
        else
            echo "FAIL $t $mode: session logits differ from the old engine's: $(diffstat "$dir/new.$t.$mode.session.logits.f32" "$dir/old.$t.$mode.session.logits.f32")"
            fail=1
        fi
    done
    if cmp -s "$dir/new.$t.prefill.session.logits.f32" "$dir/new.$t.decode.session.logits.f32"; then
        echo "ok   $t: prefill equals decode in this engine (cmp)"
    else
        echo "FAIL $t: prefill differs from decode: $(diffstat "$dir/new.$t.prefill.session.logits.f32" "$dir/new.$t.decode.session.logits.f32")"
        fail=1
    fi
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
