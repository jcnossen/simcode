#!/bin/bash
# DECODE v3 trial phase (2026-11-15), GPU 2: model 1 (preset m1809, widened ranges) from scratch and continued from
# dec_sim1809v2 (lr reset; same input / output scaling as v2). Same seed = the same training data in both arms.
# (3 arms do not fit next to the other GPU-2 jobs: ~6.3 GB each; the sim-only reference (preset sim1809) runs later next to
# the chosen run, for a matched-epoch comparison.)
source ~/miniconda3/bin/activate simcode-py311; cd ~/simcode2-decodev3
export CUDA_VISIBLE_DEVICES=${GPU:-2} TQDM_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=~/simcode2/out/decode_checks/v3/logs; T=analysis/baselines/decode/train_v3.py
nohup python -u $T --preset m1809 --name v3t_m1809_scratch --snap_every 2 > $L/train_v3t_m1809_scratch.log 2>&1 &
nohup python -u $T --preset m1809 --name v3t_m1809_cont --snap_every 2 --init ~/simcode2/out/paper2/decode/lutpsf/dec_sim1809v2/model_final.pt > $L/train_v3t_m1809_cont.log 2>&1 &
wait
