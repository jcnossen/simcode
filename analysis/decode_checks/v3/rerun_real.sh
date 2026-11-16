#!/bin/bash
# DECODE v3 on the paper's real data, camera v2: 1809_2 with dec_m1809v3, 2.5nM_2 with dec_m25v3 (frozen plateau snapshots)
# -> out/paper2/<ds>/decode/locs_v3.npz next to the old locs.npz / locs_fixed.npz.  usage: GPU=2 rerun_real.sh <ds> <model> <weights>
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2-decodev3
export PYTHONPATH=/home/jc/simcode2/external/torch-psf-040c3f3${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${GPU:-2} TQDM_DISABLE=1
ds=$1; name=$2; mf=$3; L=~/simcode2/out/decode_checks/v3/logs/rerun_real_$ds.log
echo "$(date) $ds $name $mf" >> $L
python -u analysis/baselines/decode/decode_paper.py infer $ds --name $name --model_file $mf --camera v2 \
  --out ~/simcode2/out/paper2/$ds/decode/locs_v3.npz >> $L 2>&1 && echo "$(date) done" >> $L || echo "FAILED" >> $L
