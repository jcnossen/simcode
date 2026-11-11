#!/bin/bash
# Pass-2 follower on a second GPU: solves every <cell> file whose front output exists and whose locs do not, while
# run_full4pi.sh is still in pass 1 (pass 1 is HDD-bound, pass 2 GPU-bound). The driver skips files solved here.
# Usage: source ~/miniconda3/etc/profile.d/conda.sh && conda activate simcode-py311 &&
#        CUDA_VISIBLE_DEVICES=0 setsid nohup bash analysis/fourpi/solve_follow.sh MT v2 > out/4pi/logs/follow_MT_v2.log 2>&1 &
set -u
CELL=$1; MODEL=${2:-v2}
cd /home/jc/simcode2
export PATH=$HOME/miniconda3/envs/simcode-py311/bin:$PATH
while true; do
  TODO=""
  for f in out/4pi/net/front_${CELL}_[0-9][0-9][0-9]_${MODEL}.npz; do
    [ -e "$f" ] || continue
    i=$(basename $f | cut -d_ -f3)
    [ -f out/4pi/net/locs_${CELL}_${i}_${MODEL}.npz ] || TODO="$TODO $((10#$i))"
  done
  if [ -n "$TODO" ]; then
    echo "$(date) solving:$TODO"
    python analysis/fourpi/solve_real4pi.py --cell $CELL --files $TODO --model $MODEL
  fi
  # stop when the driver has finished this cell's pass 1 (it then runs pass 2 itself for anything left)
  grep -q "$CELL pass 2" out/4pi/logs/full_v2.log && [ -z "$TODO" ] && break
  sleep 120
done
echo "$(date) follower done"
