#!/bin/sh
# S6 phase B, preregistered item (b): our logits and llama.cpp's on the Q8_0 (control) and MXFP4 files, at the
# oracle's traced positions of tests/mellum_oracle_{short,1100,2200}.tokens, prefill mode.  Run from the repo root:
#   MX=... Q8=... sh docs/superpowers/research/2026-10-28-mellum-s6-b-evidence/logits/run_b.sh
# Raw logits land in this directory's raw/ (git-ignored, kept); JSON summaries and logs beside this script.
set -u
D=docs/superpowers/research/2026-10-28-mellum-s6-b-evidence/logits
mkdir -p "$D/raw"
{ echo "build $(git rev-parse HEAD)"; date; llama-cli --version 2>&1 | head -2; sw_vers -productVersion
  shasum -a 256 "$Q8" "$MX"; } > "$D/provenance.txt" 2>&1
for t in short 1100 2200; do
    tok=tests/mellum_oracle_$t.tokens
    n=$(python3 -I -c "import sys; print(len([w for l in open('$tok') for w in l.split('#')[0].split()]))")
    pos=$(python3 -I -c "n = $n; k = min(64, n); print(','.join(str(round(i * (n - 1) / (k - 1))) for i in range(k)))")
    for m in q8 mx; do
        if [ $m = q8 ]; then model=$Q8; else model=$MX; fi
        DS4_MELLUM_TRACE="$D/raw/ours.$m.$t" DS4_MELLUM_TRACE_AT="$pos" tests/mellum_trace "$model" "$tok" prefill \
            > "$D/ours.$m.$t.log" 2>&1 || echo "FAILED ours $m $t" >> "$D/failures.txt"
        tools/mellum_llama_logits "$model" "$tok" "$D/raw/llama.$m.$t" "$pos" \
            > "$D/llama.$m.$t.log" 2>&1 || echo "FAILED llama $m $t" >> "$D/failures.txt"
        python3 -I tools/mellum_logits_agree.py "$D/raw/ours.$m.$t" "$D/raw/llama.$m.$t" "$pos" \
            > "$D/agree.ours-vs-llama.$m.$t.json" 2>> "$D/failures.txt"
    done
    python3 -I tools/mellum_logits_agree.py "$D/raw/ours.mx.$t" "$D/raw/ours.q8.$t" "$pos" \
        > "$D/agree.ours-mx-vs-ours-q8.$t.json" 2>> "$D/failures.txt"
done
echo done
