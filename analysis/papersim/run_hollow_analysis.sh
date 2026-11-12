#!/bin/bash
# hollow_analysis.py on every finished hollow run (metrics present), up to 4 in parallel; skips runs already analysed.
# usage: bash analysis/papersim/run_hollow_analysis.sh <netset>
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
NS=${1:-default}
for d in out/papersim/hollow/*_R*_x*; do
  [ -f $d/full/metrics_$NS.json ] || continue
  [ -f $d/full/hollow_$NS.json ] && continue
  python analysis/papersim/hollow_analysis.py $d/full --movie $d --netset $NS > $d/full/logs/hollow_$NS.log 2>&1 &
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 5; done
done
wait
