#!/bin/bash
# second queue (2026-11-12 01:20): waits for the main density stream $2, then runs the list $1 of "<movie>:<steps>:<netset>" items.
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
while pgrep -f "run_density_full.sh $2" > /dev/null; do sleep 60; done
for it in $1; do
  IFS=: read D ST NS <<< "$it"
  python analysis/papersim/run_all.py out/papersim/$D --steps $ST --netset $NS --gt_diag 0 >> out/papersim/logs/full_$D.log 2>&1
  echo "$(date) done $it rc $?" >> out/papersim/logs/full_stream.log
done
