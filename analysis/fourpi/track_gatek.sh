#!/bin/bash
# Gate tracker for a K-read-out head in training: after every new checkpoint run gatek4pi.py on a copy and append its table
# to out/4pi/sim/track_<tag>.log. Usage: STEPS=20000 CUDA_VISIBLE_DEVICES=0 setsid nohup bash analysis/fourpi/track_gatek.sh v9 &
M=${1:-v9}; cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
last=${FROM:-0}; LOG=out/4pi/sim/track_$M.log; TMP=/tmp/claude-1002/scratch/head4pi_${M}_trk.pt; mkdir -p /tmp/claude-1002/scratch
while true; do
  it=$(python -c "
import torch
try: print(torch.load('out/4pi/models/head4pi_$M.pt', weights_only=False, map_location='cpu')['it'])
except Exception: print(0)" 2>/dev/null)
  if [ "${it:-0}" -gt "$last" ]; then
    cp out/4pi/models/head4pi_$M.pt $TMP
    { echo "#### $M it $it $(date +%H:%M)"; python analysis/fourpi/gatek4pi.py --head $TMP 2>&1 | grep -v Warn | grep -E "^\||^==|error /"; } >> $LOG
    last=$it
    [ "$it" -ge "${STEPS:-20000}" ] && { echo TRACK_DONE >> $LOG; exit 0; }
  fi
  sleep 60
done
