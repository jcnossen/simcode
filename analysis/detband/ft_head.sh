#!/bin/bash
# pass-1 head range A/B: warm-start fine-tune of hd_m1809 (chain2.sh stage hd recipe, 4000 steps, seed 1), detector det_m1809
# (or $DET), prior range R = $1 (0.5 = package code, control). GPU 1. usage: ft_head.sh <R> <tag> [det]
source /home/jc/simcode2/runs/psfmatch/env.sh
R=$1; T=$2; DET=${3:-models/det_m1809.pt}; O=out/detband/runs/$T; mkdir -p $O
DETBAND_HRANGE=$R CUDA_VISIBLE_DEVICES=1 python -u analysis/detband/wrap.py $T sc4 --arch sc4n --out $O --steps 4000 --batch 16 \
  --lr 1.5e-4 --ch 96 --det $DET --paper 1 --tau 15,80 --hold out/sc4/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 \
  --nnw 1 --seed 1 --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init models/hd_m1809.pt --pct 0.1 \
  --bright_tail 0.1,2000,20000 --ckpt_every 2000 --calib models/insitu/calib_1809_2_psf3d_avg.json > $O.log 2>&1
