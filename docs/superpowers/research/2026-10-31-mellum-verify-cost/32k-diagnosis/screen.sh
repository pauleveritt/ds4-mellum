#!/bin/bash
# The 32K step jump: each k in its own process (descending), then one process with k descending.
set -u
W=/Users/pauleveritt/projects/pauleveritt/ds4-mellum/.worktrees/verify-cost
S=$(ls -d ~/.cache/huggingface/hub/*/snapshots/4159d684ffdb09ba89dc2a58feb3e75c54eef036)
R=$HOME/ds4-runs/bk-32k; [ -e $R ] && { echo "$R exists"; exit 8; }; mkdir -p $R
for q in Q8_0 MXFP4_MOE; do
  M="$S/Mellum2.1-12B-A2.5B-Thinking-$q.gguf"
  for k in 8 6 4 3 2 1; do
    DS4_TEST_MELLUM_BK_K=$k "$W/tests/mellum_bk.sh" $R/$q-k$k "$M" 32768 || exit 1
  done
  DS4_TEST_MELLUM_BK_K=8,6,4,3,2,1 "$W/tests/mellum_bk.sh" $R/$q-descending "$M" 32768 || exit 1
done
echo DONE
