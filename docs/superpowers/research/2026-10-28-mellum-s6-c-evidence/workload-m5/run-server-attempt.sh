#!/bin/sh
# M5 reference for phase C's workload: ds4-server on MXFP4, resident then streamed at 815 slots (2.5 GiB), three
# greedy replays of the frozen request each (max_tokens 512).  From the repo root: MX=... sh <this file>
set -u
D=docs/superpowers/research/2026-10-28-mellum-s6-c-evidence/workload-m5
REQ=docs/superpowers/research/2026-10-28-mellum-s6-c-prereg/workload-request.json
for arm in resident streamed815; do
    extra=""; [ $arm = streamed815 ] && extra="--ssd-streaming --ssd-streaming-cache-experts 815"
    ./ds4-server -m "$MX" -c 32768 --port 8765 $extra > "$D/server-$arm.log" 2>&1 &
    pid=$!
    n=0; until curl -s --noproxy '*' http://127.0.0.1:8765/v1/models > /dev/null 2>&1; do n=$((n+1)); [ $n -gt 300 ] && break; sleep 1; done
    for i in 1 2 3; do
        python3 -I tools/mellum_replay_request.py http://127.0.0.1:8765 "$REQ" "$D/$arm-$i.json" --max-tokens 512 \
            > "$D/$arm-$i.stdout" 2> "$D/$arm-$i.stderr" || echo "FAILED $arm $i" >> "$D/failures.txt"
        sleep 10
    done
    kill $pid; wait $pid 2>/dev/null
    sleep 10
done
echo done
