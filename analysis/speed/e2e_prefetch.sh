#!/bin/bash
# pass-1 stage of run_uncalibrated (fast path, production cut 0.3) on NF frames in chunks of CH, prefetch 0 vs 1, alternating,
# REPS times; wall per run + output equality. usage: bash analysis/speed/e2e_prefetch.sh <movie> <calib> <tag> [NF] [CH] [REPS]
MOV=$1; CAL=$2; TAG=$3; NF=${4:-6000}; CH=${5:-1500}; REPS=${6:-2}
O=/home/jc/simcode2/out/speed/e2e_prefetch_$TAG
for r in $(seq 1 $REPS); do
  for pf in $( [ $((r % 2)) -eq 1 ] && echo "0 1" || echo "1 0"); do
    t0=$(date +%s.%N)
    python -m simcode2.pipeline.run_uncalibrated --movie $MOV --calib $CAL --out $O/pf$pf --fast 1 --cand_pmin 0.3 --stages pass1 \
      --nframes $NF --chunk $CH --redo 1 --prefetch $pf > /dev/null 2>&1
    t1=$(date +%s.%N)
    echo "rep $r prefetch $pf wall $(echo "$t1 - $t0" | bc) gpus $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')"
  done
done
python - <<PY
import numpy as np, glob, os
for f in sorted(glob.glob('$O/pf0/pass1_*.npz')):
    a, b = np.load(f), np.load(f.replace('/pf0/', '/pf1/'))
    print(os.path.basename(f), 'identical' if all(np.array_equal(a[k], b[k]) for k in a.files) else 'DIFFERENT')
PY
