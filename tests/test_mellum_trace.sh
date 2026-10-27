#!/bin/sh
# The Mellum trace driver on a 3-token file.  Model gated: it needs DS4_TEST_MELLUM_MODEL, the
# real 2.1 GGUF, and Metal; without it the test says so and passes.  `make test-mellum-trace`.
#
#   1. a listed position writes its three files, each with its exact size, and an unlisted
#      position writes none; the session's own logits are written too (<prefix>.session.logits.f32);
#      every listed position's logits are checked finite;
#   2. a listed position that is not one of the tokens is an error naming it (exit 1, nothing written);
#   3. DS4_MELLUM_TRACE_AT without DS4_MELLUM_TRACE is a usage error (exit 2); DS4_MELLUM_TRACE alone
#      is an untraced run that writes only the session logits;
#   4. a traced run ends its command buffer after every layer, so it cannot see a hazard between
#      layers of the one buffer an ordinary run uses: the session logits of an untraced run equal
#      those of a run traced at every position, as one chunk and as two (DS4_QWEN4_PREFILL_CHUNK=2);
#   5. files from an earlier run with the same prefix do not survive a run that fails, and a run
#      that succeeds replaces them.
set -u
cd "$(dirname "$0")/.." || exit 2
if [ -z "${DS4_TEST_MELLUM_MODEL:-}" ]; then
    echo "mellum-trace: DS4_TEST_MELLUM_MODEL unset, skipped"
    exit 0
fi
[ -x tests/mellum_trace ] || { echo "mellum-trace: build tests/mellum_trace first (make tests/mellum_trace)" >&2; exit 2; }

dir=$(mktemp -d "${TMPDIR:-/tmp}/mellum-trace.XXXXXX") || exit 2
trap 'rm -rf "$dir"' EXIT
printf '# three tokens\n27 1397\n233\n' > "$dir/tokens3.txt"
printf '27 99999999 233\n' > "$dir/badtoken.txt"
fail=0
check() { # description, command...
    desc=$1; shift
    if "$@"; then echo "ok   $desc"; else echo "FAIL $desc"; fail=1; fi
}
size_is() { [ "$(wc -c < "$1" | tr -d ' ')" = "$2" ]; }
M=$DS4_TEST_MELLUM_MODEL

# 1. positions 0 and 2 of 3, in each mode
for mode in prefill decode; do
    p="$dir/$mode"
    DS4_MELLUM_TRACE="$p" DS4_MELLUM_TRACE_AT=0,2 tests/mellum_trace "$M" "$dir/tokens3.txt" "$mode" \
        > "$dir/$mode.out" 2> "$dir/$mode.err"
    check "$mode: the driver exits 0" [ $? -eq 0 ]
    for pos in 0 2; do
        check "$mode: p$pos hidden is 28 x 2304 f32 (258048 bytes)" size_is "$p.p$pos.hidden.f32" 258048
        check "$mode: p$pos qk is 28 x 4608 f32 (516096 bytes)" size_is "$p.p$pos.qk.f32" 516096
        check "$mode: p$pos logits is 1 x 98304 f32 (393216 bytes)" size_is "$p.p$pos.logits.f32" 393216
    done
    check "$mode: the session logits are 1 x 98304 f32 (393216 bytes)" size_is "$p.session.logits.f32" 393216
    check "$mode: unlisted position 1 writes nothing" [ ! -e "$p.p1.hidden.f32" ]
    for pos in 0 2; do
        check "$mode: p$pos logits are checked finite" grep -q "position $pos logits: 0 non-finite" "$dir/$mode.out"
    done
done
# the session's last-row logits are the traced last position's logits
check "prefill: session logits equal the traced position 2 logits" cmp -s "$dir/prefill.session.logits.f32" "$dir/prefill.p2.logits.f32"

# 2. a position the file does not have
DS4_MELLUM_TRACE="$dir/miss" DS4_MELLUM_TRACE_AT=0,7 tests/mellum_trace "$M" "$dir/tokens3.txt" prefill \
    > "$dir/miss.out" 2> "$dir/miss.err"
rc=$?
check "a missing position: exit 1" [ "$rc" -eq 1 ]
check "a missing position: the error names 7" grep -q "trace position 7 is not one of the 3 tokens" "$dir/miss.err"
check "a missing position: nothing was written" [ ! -e "$dir/miss.p0.hidden.f32" ]

# 3. half a trace environment, and the untraced run
DS4_MELLUM_TRACE_AT=0 tests/mellum_trace "$M" "$dir/tokens3.txt" prefill > /dev/null 2> "$dir/half.err"
check "DS4_MELLUM_TRACE_AT without DS4_MELLUM_TRACE: exit 2" [ $? -eq 2 ]
DS4_MELLUM_TRACE="$dir/plain" tests/mellum_trace "$M" "$dir/tokens3.txt" prefill > /dev/null 2> "$dir/plain.err"
check "DS4_MELLUM_TRACE alone is an untraced run: exit 0" [ $? -eq 0 ]
check "the untraced run writes the session logits" size_is "$dir/plain.session.logits.f32" 393216
check "the untraced run writes no position file" [ ! -e "$dir/plain.p0.hidden.f32" ]

# 4. untraced against traced, one chunk and two
for chunk in 1024 2; do
    DS4_QWEN4_PREFILL_CHUNK=$chunk DS4_MELLUM_TRACE="$dir/u$chunk" tests/mellum_trace "$M" "$dir/tokens3.txt" prefill > /dev/null 2>&1
    check "chunk $chunk: the untraced run exits 0" [ $? -eq 0 ]
    DS4_QWEN4_PREFILL_CHUNK=$chunk DS4_MELLUM_TRACE="$dir/t$chunk" DS4_MELLUM_TRACE_AT=0,1,2 tests/mellum_trace "$M" "$dir/tokens3.txt" prefill > /dev/null 2>&1
    check "chunk $chunk: the fully traced run exits 0" [ $? -eq 0 ]
    check "chunk $chunk: untraced session logits equal the fully traced run's" cmp -s "$dir/u$chunk.session.logits.f32" "$dir/t$chunk.session.logits.f32"
done
check "one chunk and two chunks give the same session logits" cmp -s "$dir/u1024.session.logits.f32" "$dir/u2.session.logits.f32"

# 5. stale files
p="$dir/stale"
for f in p0.hidden p0.qk p0.logits session.logits; do head -c 4 /dev/zero > "$p.$f.f32"; done
DS4_MELLUM_TRACE="$p" DS4_MELLUM_TRACE_AT=0 tests/mellum_trace "$M" "$dir/badtoken.txt" prefill > /dev/null 2> "$dir/stale.err"
check "a failing run: exit 1" [ $? -eq 1 ]
check "a failing run removed the stale p0 hidden file" [ ! -e "$p.p0.hidden.f32" ]
check "a failing run removed the stale session logits file" [ ! -e "$p.session.logits.f32" ]
for f in p0.hidden p0.qk p0.logits session.logits; do head -c 4 /dev/zero > "$p.$f.f32"; done
DS4_MELLUM_TRACE="$p" DS4_MELLUM_TRACE_AT=0 tests/mellum_trace "$M" "$dir/tokens3.txt" prefill > /dev/null 2>&1
check "a good run replaces the stale files: exit 0" [ $? -eq 0 ]
check "a good run replaced the stale hidden file" size_is "$p.p0.hidden.f32" 258048

[ "$fail" -eq 0 ] && echo "mellum-trace: PASS" || echo "mellum-trace: FAILED"
exit "$fail"
