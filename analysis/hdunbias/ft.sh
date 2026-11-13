#!/bin/bash
# warm-start fine-tune of hd_m1809 (chain2.sh stage-hd recipe, 6000 steps, seed 1, ckpt every 2000), detector det_m1809,
# extra sc4 args ($3...; the experiment options --jit / --unb_w / --unb_clip were removed from sc4.py after the A/B, commit of
# 2026-11-14; to reproduce the J / P arms restore them from analysis/hdunbias/sc4_experiment.patch (git apply)). usage: ft.sh <tag> <gpu> [extra args]
source /home/jc/simcode2/runs/psfmatch/env.sh
T=$1; G=$2; shift 2; O=out/hdunbias/runs/$T; mkdir -p $O
CUDA_VISIBLE_DEVICES=$G python -u analysis/hdunbias/wrap.py $T sc4 --arch sc4n --out $O --steps 6000 --batch 16 \
  --lr 1.5e-4 --ch 96 --det models/det_m1809.pt --paper 1 --tau 15,80 --hold out/sc4/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 \
  --nnw 1 --seed 1 --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init models/hd_m1809.pt --pct 0.1 \
  --bright_tail 0.1,2000,20000 --ckpt_every 2000 --calib models/insitu/calib_1809_2_psf3d_avg.json "$@" > $O.log 2>&1
