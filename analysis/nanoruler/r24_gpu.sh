#!/bin/bash
# R24 (nr40 / nr20 workers): run ONE GPU command at a time (global lock), on a GPU with enough free memory.
# usage: bash analysis/nanoruler/r24_gpu.sh <need_gb> <cmd ...>
#   takes the lock out/nanoruler/nr40/gpu.lock (waits for other R24 GPU jobs), then picks the first GPU of $R24_GPUS (default "1 2 0")
#   with free memory >= need_gb + 2 GB (waits, one line per 10 min), exports CUDA_VISIBLE_DEVICES and EV_DET_BATCH (default 8), runs the command.
need=$1; shift
exec 9>/home/jc/simcode2/out/nanoruler/nr40/gpu.lock; flock 9
n=0
while true; do
  for g in ${R24_GPUS:-1 2 0}; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    [ "$f" -ge $(( (need + 2) * 1024 )) ] && break 2
  done
  [ $((n % 10)) = 0 ] && echo "$(date +%F_%T) r24_gpu: waiting for a GPU with $((need + 2)) GB free" >&2
  n=$((n + 1)); sleep 60
done
echo "$(date +%F_%T) r24_gpu: GPU $g ($f MiB free): $*" >&2
export CUDA_VISIBLE_DEVICES=$g EV_DET_BATCH=${EV_DET_BATCH:-8}
"$@" 9>&-
rc=$?; echo "$(date +%F_%T) r24_gpu: rc $rc" >&2; exit $rc
