#!/bin/bash
# Methods on a benchmark movie (recipes.sh output dir): ours (simcode, both passes, the sim-sigma calibrations as in the
# paper's sim tables), sign-fixed SIMCODE (analysis/baselines/run_simcode.py; needs ~/smlmtorch), optionally retrained DECODE.
# usage: GPU=1 bash analysis/simreal/eval_movie.sh <movie_dir> <out_dir> <ni6_3 (0.5 nM) | L64_5 (2.5 nM)> [decode_model]
set -euo pipefail
d=$1; o=$2; md=$3; dec=${4:-}
D=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$D/../.." && pwd)
export CUDA_VISIBLE_DEVICES=${GPU:-1} TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
mkdir -p "$o"
$PY -u -m simcode.processing.run_uncalibrated --movie "$d/sim_movie.tif" --calib insitu/calib_0.5nM.json --out "$o/ours" --frontend sc2p \
  --sc_cal calib/calib_sc_hd_sc5nbb15_pm_bright.json --p2_cal calib/calib_p2L_movie_b10.json \
  --chunk 6000 --crop 5,5,110 --stages pass1,pattern,pass2
$PY "$REPO/analysis/baselines/run_simcode.py" "$d" "$o/simcode_${md}_fix.npz" --model "$md" --cam paper --fix_sign 1 --net_cache "$o/simcode_${md}_net.npy"
if [ -n "$dec" ]; then (cd "$REPO/analysis/baselines/decode" && $PY decode_infer.py "$d" "$dec" sum6 "$o/decode_${dec}_sum6.npz"); fi
