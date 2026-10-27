#!/bin/sh
# The ds4 CLI with Mellum 2.1, through the chat template (never --raw-prompt).  Model gated: it needs
# DS4_TEST_MELLUM_MODEL, the real 2.1 GGUF, and Metal; without it the test says so and passes.
# `make test-mellum-cli`.  One model process at a time.
#
#   1. thinking on: the answer is right, and the reasoning's <think> and </think> are not printed
#      as text (the printer turns them into the grey reasoning block);
#   2. --nothink on a simple question: a direct answer, no reasoning;
#   3. a turn that runs into -n says so on stderr, in the greedy and in the sampled path, and a
#      turn that ends on its own does not.
set -u
cd "$(dirname "$0")/.." || exit 2
if [ -z "${DS4_TEST_MELLUM_MODEL:-}" ]; then
    echo "mellum-cli: DS4_TEST_MELLUM_MODEL unset, skipped"
    exit 0
fi
[ -x ./ds4 ] || { echo "mellum-cli: build ./ds4 first (make)" >&2; exit 2; }
M=$DS4_TEST_MELLUM_MODEL
dir=$(mktemp -d "${TMPDIR:-/tmp}/mellum-cli.XXXXXX") || exit 2
trap 'rm -rf "$dir"' EXIT
fail=0
check() { # description, command...
    desc=$1; shift
    if "$@"; then echo "ok   $desc"; else echo "FAIL $desc"; fail=1; fi
}
ask() { # name, ds4 arguments...
    name=$1; shift
    ./ds4 -m "$M" -c 4096 "$@" > "$dir/$name.out" 2> "$dir/$name.err"
}
Q="What is the capital of France? Answer in one sentence."

ask think --temp 0 -n 400 -p "$Q"
check "thinking on: exit 0" [ $? -eq 0 ]
check "thinking on: the answer names Paris" grep -q Paris "$dir/think.out"
check "thinking on: no <think> text" sh -c "! grep -q '<think>' '$dir/think.out'"
check "thinking on: no </think> text" sh -c "! grep -q '</think>' '$dir/think.out'"
check "thinking on: a turn that ends on its own reports no cap" sh -c "! grep -q 'stopped at the -n cap' '$dir/think.err'"

ask nothink --temp 0 -n 400 --nothink -p "$Q"
check "--nothink: exit 0" [ $? -eq 0 ]
check "--nothink: the answer names Paris" grep -q Paris "$dir/nothink.out"
check "--nothink: a direct answer (under 20 words)" [ "$(wc -w < "$dir/nothink.out" | tr -d ' ')" -lt 20 ]

ask capg --temp 0 -n 20 -p "$Q"
check "greedy, -n 20: the cap is reported" grep -q 'stopped at the -n cap of 20 tokens' "$dir/capg.err"
ask caps --temp 0.7 --seed 7 -n 20 -p "$Q"
check "sampled, -n 20: the cap is reported" grep -q 'stopped at the -n cap of 20 tokens' "$dir/caps.err"

[ "$fail" -eq 0 ] && echo "mellum-cli: PASS" || echo "mellum-cli: FAILED"
exit "$fail"
