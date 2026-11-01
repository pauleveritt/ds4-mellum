#!/bin/bash
set -u
W=/Users/pauleveritt/projects/pauleveritt/ds4-mellum/.worktrees/pi-real-use
S=$(ls -d ~/.cache/huggingface/hub/*/snapshots/4159d684ffdb09ba89dc2a58feb3e75c54eef036)
R=$HOME/ds4-runs/pi
cd "$W" || exit 9
echo "omlx-server: $(ps -o pid,%cpu,rss -p $(pgrep -f omlx-server | head -1) | tail -1)" > $R/screen-env.txt
first=1
for n in 1 2 3; do
  for q in q8:Q8_0 mx:MXFP4_MOE; do
    name=${q%%:*}-$n; file=${q#*:}
    if [ -e "$R/$name" ]; then echo "$name exists, refusing"; exit 8; fi
    [ $first = 1 ] || sleep 120
    first=0
    echo "$(date '+%H:%M:%S') start $name"
    uv run --no-project python -I tools/mellum_pi_run.py --model "$S/Mellum2.1-12B-A2.5B-Thinking-$file.gguf" --out "$R/$name" > "$R/$name.out" 2>&1
    echo "$(date '+%H:%M:%S') end $name exit=$? $(tail -1 $R/$name.out)"
  done
done
echo DONE
