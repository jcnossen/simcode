#!/bin/bash
# recompute metrics.py (current columns) for the given "<movie>:<netset>" items, 4 in parallel (CPU)
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2
for it in $1; do
  IFS=: read D NS <<< "$it"
  python analysis/papersim/metrics.py out/papersim/$D/full --movie out/papersim/$D --netset $NS > out/papersim/$D/full/logs/metrics_rerun_$NS.log 2>&1 &
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 5; done
done
wait
echo "$(date) metrics rerun done: $1" >> out/papersim/logs/full_stream.log
