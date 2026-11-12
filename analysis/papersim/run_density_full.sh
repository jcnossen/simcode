#!/bin/bash
# full density sweep (60 000 frames): base, DECODE (dec_1809_2, flagged), v1, v2 default + v2 m1809, metrics per net set.
# usage: CUDA_VISIBLE_DEVICES=2 bash analysis/papersim/run_density_full.sh "D4 D1"
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
for D in $1; do
  md=out/papersim/$D
  if [ "$D" = "D1_gauss" ]; then
    python analysis/papersim/run_all.py $md --steps v1,metrics --netset default >> out/papersim/logs/full_$D.log 2>&1
  else
    python analysis/papersim/run_all.py $md --steps base,decode,v1,v2,metrics --netset default --gt_diag 0 >> out/papersim/logs/full_$D.log 2>&1
    python analysis/papersim/run_all.py $md --steps v2,metrics --netset m1809 --gt_diag 0 >> out/papersim/logs/full_$D.log 2>&1
  fi
  echo "$(date) done $D rc $?" >> out/papersim/logs/full_stream.log
done
