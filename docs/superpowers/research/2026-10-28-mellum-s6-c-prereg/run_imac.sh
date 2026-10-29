#!/bin/sh
# S6 phase C on the 8 GB M1 iMac, one preregistered stage per call (docs/superpowers/research/2026-10-28-mellum-s6-c-
# preregistration.md).  The owner runs it from the repo root, by hand, one stage at a time:
#
#   MX=/path/to/Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf OUT=~/s6-c-imac sh <this file> STAGE [ARG]
#
#   bringup             C1: build, CPU and kernel tests, the three streamed ds4_test checks, machine facts
#   cache               C2: the workload once at 3, 2.5 and 2 GiB, apps closed, with vm_stat around each
#   correct BYTES       C3: decode-mode streamed logits at the chosen cache on the oracle files, vs the M5 reference
#   footprint closed|open BYTES   C4: three workload runs at the chosen cache (start the background load before "open")
#
# Every stage writes under $OUT/<stage> and never deletes a capture; a stage that exists already is refused.
set -u
: "${MX:?set MX to the MXFP4 GGUF}" "${OUT:?set OUT to the evidence directory}"
stage=${1:?stage: bringup | cache | correct BYTES | footprint closed|open BYTES}
P=docs/superpowers/research/2026-10-28-mellum-s6-c-prereg
REF=${REF:-docs/superpowers/research/2026-10-28-mellum-s6-c-evidence/raw}
mkdir -p "$OUT"
new_dir() { [ -e "$OUT/$1" ] && { echo "run_imac: $OUT/$1 exists; nothing is overwritten" >&2; exit 1; }; mkdir -p "$OUT/$1"; }
swapouts() { vm_stat | awk '/Swapouts/ {gsub("\\.","",$2); print $2}'; }
measure() {   # name stream_bytes
    d=$1; b=$2
    echo "swapouts_before $(swapouts)" > "$d.vmstat"
    tools/mellum_s4_measure --model "$MX" --prompt "$P/workload-prompt.txt" --gen 512 --context 32768 \
        --stream-bytes "$b" --out "$d.measure.json" > "$d.stdout" 2> "$d.stderr"
    echo "rc $?" >> "$d.vmstat"
    echo "swapouts_after $(swapouts)" >> "$d.vmstat"
    vm_stat > "$d.vm_stat_full.txt"
    sleep 20
}
case "$stage" in
bringup)
    new_dir bringup; d=$OUT/bringup
    git rev-parse HEAD > "$d/commit.txt"; shasum -a 256 "$MX" > "$d/model-sha256.txt"
    make > "$d/make.log" 2>&1; echo "make rc $?" >> "$d/summary.txt"
    make tools/mellum_metal_limit tools/mellum_s4_measure tests/mellum_trace ds4_test > "$d/make-tools.log" 2>&1
    echo "make tools rc $?" >> "$d/summary.txt"
    tools/mellum_machine_facts.sh "$d/machine.txt"
    make test-mellum > "$d/test-mellum.log" 2>&1; echo "test-mellum rc $?" >> "$d/summary.txt"
    make test-mellum-kernels > "$d/test-mellum-kernels.log" 2>&1; echo "test-mellum-kernels rc $?" >> "$d/summary.txt"
    for t in mellum-stream-identity mellum-stream-prefill mellum-stream-faults; do
        DS4_TEST_MELLUM_MODEL="$MX" ./ds4_test --$t > "$d/ds4_test-$t.log" 2>&1; echo "$t rc $?" >> "$d/summary.txt"
    done
    cat "$d/summary.txt" ;;
cache)
    new_dir cache
    for b in 3221225472 2684354560 2147483648; do measure "$OUT/cache/$b" "$b"; done
    grep -H . "$OUT"/cache/*.vmstat ;;
correct)
    b=${2:?correct needs the chosen cache in bytes}; new_dir correct; d=$OUT/correct
    for t in short 1100 2200; do
        tok=tests/mellum_oracle_$t.tokens
        n=$(python3 -I -c "print(len([w for l in open('$tok') for w in l.split('#')[0].split()]))")
        pos=$(python3 -I -c "n = $n; k = min(64, n); print(','.join(str(round(i * (n - 1) / (k - 1))) for i in range(k)))")
        MELLUM_TRACE_STREAM_BYTES=$b DS4_MELLUM_TRACE="$d/imac.mx.$t" MELLUM_TRACE_SESSION_AT=$pos \
            tests/mellum_trace "$MX" "$tok" decode > "$d/trace-$t.log" 2>&1; echo "trace $t rc $?" >> "$d/summary.txt"
        python3 -I tools/mellum_logits_agree.py "$d/imac.mx.$t" "$REF/m5-decode-resident.mx.$t" "$pos" \
            > "$d/agree.$t.json" 2>> "$d/summary.txt"
    done
    cat "$d/summary.txt" ;;
footprint)
    cond=${2:?footprint needs closed or open}; b=${3:?footprint needs the chosen cache in bytes}
    new_dir "footprint-$cond"
    for i in 1 2 3; do measure "$OUT/footprint-$cond/$i" "$b"; done
    grep -H . "$OUT/footprint-$cond"/*.vmstat ;;
*) echo "run_imac: unknown stage $stage" >&2; exit 2 ;;
esac
