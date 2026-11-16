#!/bin/bash
# DECODE model 2 (dec_m25v3: PSF v2 of 2.5 nM/2, for 2.5 nM/2 (Fig 4)) at the published scale, from scratch, on the next
# free GPU: waits until GPU ${GPU:-1} has < 20 % use and > 12 GB free in 10 checks over ~5 min, then trains + evaluates
# (iso movie with the m25 PSF, real-data chunk 2.5 nM (20 nM is not in the paper)).
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2-decodev3
G=${GPU:-1}; L=~/simcode2/out/decode_checks/v3/logs
ok=0
while [ $ok -lt 10 ]; do
  read u m t <<< $(nvidia-smi -i $G --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d ',')
  if [ "$u" -lt 20 ] && [ $((t - m)) -gt 12288 ]; then ok=$((ok + 1)); else ok=0; fi
  sleep 30
done
echo "$(date) GPU $G free, starting dec_m25v3" >> $L/run_m25.log
export CUDA_VISIBLE_DEVICES=$G TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python -u analysis/baselines/decode/train_v3.py --preset m25 --name dec_m25v3 --snap_every 5 --warmup 512 > $L/train_dec_m25v3.log 2>&1 &
sleep 120
nohup python -u analysis/decode_checks/v3/eval_watch.py --names dec_m25v3 --iso ~/simcode2/out/decode_checks/v3/iso_m25 --dens '' \
  --real 2.5nM_2 --tag dec_m25v3 --poll 600 > $L/eval_dec_m25v3.log 2>&1 &
wait
