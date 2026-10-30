#!/bin/bash
export PATH=/opt/homebrew/opt/python@3.14/libexec/bin:$PATH
cd ~/ds4-mellum-release || exit 1
O=~/readme-m1pro; Q8=~/models/Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf; MX=~/models/Mellum2.1-12B-A2.5B-Thinking-MXFP4_MOE.gguf
P=tests/mellum_measurements/d1_decode_profile
echo "start $(date) head $(git rev-parse --short HEAD)"
ARMS=AA tests/mellum_cli_prefill_abba.sh $O/q8-prefill $Q8 2048 8192 32768 > $O/q8-prefill.log 2>&1; echo "q8-prefill rc $? $(date)"
ARMS=AA tests/mellum_cli_decode_abba.sh $O/q8-decode $Q8 $P 2048 8192 32768 > $O/q8-decode.log 2>&1; echo "q8-decode rc $? $(date)"
ARMS=AA tests/mellum_cli_prefill_abba.sh $O/mx-prefill $MX 2048 8192 32768 > $O/mx-prefill.log 2>&1; echo "mx-prefill rc $? $(date)"
ARMS=AA tests/mellum_cli_decode_abba.sh $O/mx-decode $MX $P 2048 8192 32768 > $O/mx-decode.log 2>&1; echo "mx-decode rc $? $(date)"
echo done
