#!/bin/bash
# Generic pass 2, v2 test (2026-11-13): continue the generic m1809-PSF head with a second p2headL-type stage (50k steps, lr 1e-4,
# pct 0.05, pert_model real, seed 41) from runs/p2generic/g1809/p2headL/final.pt (detector = g1809 p2det final), pgen preset $2.
# usage: GPU=<g> setsid nohup bash analysis/p2generic/cont_gen.sh <name> <pgen preset> >> runs/p2generic/cont_<name>.log 2>&1 &
source /home/jc/simcode2/runs/psfmatch/env.sh
N=$1; PG=$PWD/out/p2generic/presets/$2.json; D=runs/p2generic/g1809; O=runs/p2generic/$N; mkdir -p $O
echo "$(date) cont $N pgen $PG GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU P2G_TITLE=p2gen-$N python -u analysis/p2generic/pwrap.py simcode.p2.head2 train --out $O/p2headL --det $D/p2det/final.pt \
  --init $D/p2headL/final.pt --steps 50000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 41 --eval_every 5000 --ckpt_every 10000 \
  --no_illum 0 --pert_model real --calib ${PSF[m1809]} --pgen $PG > $O/p2headL.log 2>&1 && echo "$(date) cont $N done" || echo "$(date) cont $N FAILED"
