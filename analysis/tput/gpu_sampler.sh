#!/bin/bash
# logs "<epoch> <gpu index> <compute pids>" every 10 s for the given GPUs (default 1 2): validity check of the timing sections
# (summarize.py reads <tput dir>/gpu_samples.log). 
# usage: bash analysis/tput/gpu_sampler.sh [gpus...] >> <simcode out>/tput/gpu_samples.log &
GPUS=${*:-1 2}
while true; do
  t=$(date +%s)
  for g in $GPUS; do echo "$t $g $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"; done
  sleep 10
done
