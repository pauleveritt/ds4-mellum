#!/bin/sh
# M5 reference for phase C's workload (amended: ds4-server does not serve Mellum, see server-*.log): the flattened
# request through tools/mellum_s4_measure, resident then streamed at 2.5 GiB, n=3 each, --gen 512, context 32,768,
# 20 s rests.  From the repo root: MX=... sh <this file>
set -u
D=docs/superpowers/research/2026-10-28-mellum-s6-c-evidence/workload-m5
P=docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/workload-prompt.txt
for arm in resident streamed2.5g; do
    for i in 1 2 3; do
        extra=""; [ $arm = streamed2.5g ] && extra="--stream-bytes 2684354560"
        tools/mellum_s4_measure --model "$MX" --prompt "$P" --gen 512 --context 32768 --out "$D/$arm-$i.measure.json" $extra \
            > "$D/$arm-$i.stdout" 2> "$D/$arm-$i.stderr" || echo "FAILED $arm $i" >> "$D/failures-measure.txt"
        sleep 20
    done
done
echo done
