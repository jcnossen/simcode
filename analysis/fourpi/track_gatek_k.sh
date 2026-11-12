#!/bin/bash
# Gate tracker for a K-range head (v11): after every new checkpoint, gatek4pi.py at K = 1, 4, 8 (6 neighbours max) -> out/4pi/sim/track_<tag>.log
# Usage: STEPS=30000 CUDA_VISIBLE_DEVICES=2 setsid nohup bash analysis/fourpi/track_gatek_k.sh v11 &
M=${1:-v11}; cd /home/jc/simcode2; export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
last=${FROM:-0}; LOG=out/4pi/sim/track_$M.log; TMP=/tmp/claude-1002/scratch/head4pi_${M}_trk.pt; mkdir -p /tmp/claude-1002/scratch
while true; do
  it=$(python -c "
import torch
try: print(torch.load('out/4pi/models/head4pi_$M.pt', weights_only=False, map_location='cpu')['it'])
except Exception: print(0)" 2>/dev/null)
  if [ "${it:-0}" -gt "$last" ] && [ $(( it % ${EVERY:-5000} )) -eq 0 ]; then
    cp out/4pi/models/head4pi_$M.pt $TMP
    for K in 1 4 8; do
      { echo "#### $M it $it K $K $(date +%H:%M)"; python analysis/fourpi/gatek4pi.py --head $TMP --ksum $K --nb_max ${NBMAX:-10} 2>&1 | grep -v Warn | grep -E "^\||^==|error /|^K = " | sed "s#$TMP#$M#g"; } >> $LOG
    done
    last=$it
    [ "$it" -ge "${STEPS:-30000}" ] && { echo TRACK_DONE >> $LOG; exit 0; }
  fi
  sleep 120
done
