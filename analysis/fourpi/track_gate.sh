#!/bin/bash
# Gate tracker for a head in training: after every new checkpoint run the review table on a copy of it and append the
# key rows to out/4pi/sim/track_<tag>.log. Exits after the checkpoint of step $STEPS.
# Usage: STEPS=40000 CUDA_VISIBLE_DEVICES=0 setsid nohup bash analysis/fourpi/track_gate.sh v8 > /dev/null 2>&1 &
M=${1:-v8}; cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
last=${FROM:-0}; LOG=out/4pi/sim/track_$M.log; TMP=/tmp/claude-1002/scratch/head4pi_${M}_trk.pt; mkdir -p /tmp/claude-1002/scratch
while true; do
  it=$(python -c "
import torch
try: print(torch.load('out/4pi/models/head4pi_$M.pt', weights_only=False, map_location='cpu')['it'])
except Exception: print(0)" 2>/dev/null)
  if [ "${it:-0}" -gt "$last" ]; then
    cp out/4pi/models/head4pi_$M.pt $TMP
    python analysis/fourpi/review4pi.py --head $TMP > /tmp/claude-1002/scratch/trk_$M.txt 2>&1
    { echo "== $M it $it $(date +%H:%M)"
      grep -A8 "== rms_trim60" /tmp/claude-1002/scratch/trk_$M.txt | grep -E "oracle T,C,cov \+ network prior|production"
      grep -A8 "== robust" /tmp/claude-1002/scratch/trk_$M.txt | grep production | sed "s/^/robust /"; } >> $LOG
    last=$it
    [ "$it" -ge "${STEPS:-40000}" ] && { echo TRACK_DONE >> $LOG; exit 0; }
  fi
  sleep 60
done
