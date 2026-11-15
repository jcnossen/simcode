#!/bin/bash
# Generic pass 2, v2 test (2026-11-13): continue the generic pass-2 DETECTOR (40k steps, lr 1e-4, pct 0.05, seed 51) from
# runs/p2generic/g1809/p2det/final.pt with p_std $2 (the detector carries about half of the g1809 loss, eval_mix.txt).
# usage: GPU=<g> setsid nohup bash analysis/p2generic/cont_det.sh <name> <p_std> >> runs/p2generic/contdet_<name>.log 2>&1 &
source /home/jc/simcode2/runs/psfmatch/env.sh
# $2 = p_std: share of SIMCODE-pattern FOVs (the former presets pgen_v1 = 0.25, pgen_v1s50 = 0.5)
N=$1; PS=$2; D=runs/p2generic/g1809; O=runs/p2generic/$N; mkdir -p $O
echo "$(date) contdet $N p_std $PS GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU P2G_TITLE=p2gen-d$N python -u analysis/p2generic/pwrap.py simcode.detector.train --out $O/p2det --steps 40000 --batch 16 --size 64 \
  --lr 1e-4 --pct 0.05 --w_off 1.0 --seed 51 --eval_every 5000 --ckpt_every 10000 --init $D/p2det/final.pt --pass1_ref models/det_m1809.pt \
  --calib ${PSF[m1809]} --p_std $PS > $O/p2det.log 2>&1 && echo "$(date) contdet $N done" || echo "$(date) contdet $N FAILED"
