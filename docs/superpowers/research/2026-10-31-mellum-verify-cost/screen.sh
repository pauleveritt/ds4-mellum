#!/bin/bash
set -u
W=/Users/pauleveritt/projects/pauleveritt/ds4-mellum/.worktrees/verify-cost
S=$(ls -d ~/.cache/huggingface/hub/*/snapshots/4159d684ffdb09ba89dc2a58feb3e75c54eef036)
R=$HOME/ds4-runs/bk; mkdir -p $R
for q in Q8_0 MXFP4_MOE; do
  [ -e $R/$q ] && { echo "$R/$q exists"; exit 8; }
  "$W/tests/mellum_bk.sh" $R/$q "$S/Mellum2.1-12B-A2.5B-Thinking-$q.gguf" 2048,8192,32768 || exit 1
done
for q in Q8_0 MXFP4_MOE; do
  N=4 TIMELINE=1 "$W/tests/mellum_bk.sh" $R/$q-timeline "$S/Mellum2.1-12B-A2.5B-Thinking-$q.gguf" 2048,32768 || exit 1
done
echo DONE
