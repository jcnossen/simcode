#!/bin/bash
# density sweep at the experimental 1809/2 photon / bg distribution (gen_hollow.py --radius_nm 0, on-time 20 frames): waits for
# the main density streams, then runs one stream of movies (base, DECODE, v1, v2 default + m1809, metrics).
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
while pgrep -f "run_density_full.sh $2" > /dev/null; do sleep 60; done
for D in $1; do
  md=out/papersim/$D
  python analysis/papersim/run_all.py $md --steps base,decode,v1,v2,metrics --netset default --gt_diag 0 >> out/papersim/logs/full_$D.log 2>&1
  python analysis/papersim/run_all.py $md --steps v2,metrics --netset m1809 --gt_diag 0 >> out/papersim/logs/full_$D.log 2>&1
  echo "$(date) done $D rc $?" >> out/papersim/logs/full_stream.log
done
