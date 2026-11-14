#!/bin/bash
# m1809 det recipe (runs/psfmatch/chain2.sh stage det), only --band differs. GPU 1. usage: train_ab.sh <band> <tag>
source /home/jc/simcode2/runs/psfmatch/env.sh
B=$1; T=$2; O=out/detband/runs/$T; mkdir -p $O
CUDA_VISIBLE_DEVICES=1 python -u analysis/detband/train.py $T --up 1 --out $O --steps 40000 --batch 16 --size 64 --lr 1e-3 \
  --w_off 1.0 --nnw 1 --em_photons 40,2500 --seed 0 --calib models/insitu/calib_1809_2_psf3d_avg.json --band $B > $O.log 2>&1
