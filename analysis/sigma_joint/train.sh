#!/bin/bash
# joint prior / intensity covariance (2026-11-17): fine-tuning of the emitter networks of a model with the joint (x, y, I_1..6)
# Gaussian NLL (simcode.emitter.train; the 12 new C output rows start at zero = the current network). Detectors frozen: the
# detector each network was trained with (its checkpoint args), the model's PSF calibration, the recipe flags of the model
# (models/README.md; g13: --dens 0.0005,0.06 --tau 3,80), shorter and at the recipe's final learning rate.
# usage: PKG=<package repo> GPU=<g> bash train.sh <m1809|m25|g13>:<p1|p2> [...]   (out: ~/simcode2/out/sigma_joint/train/<m>/<p>)
source ~/miniconda3/bin/activate simcode-py311
export CUDA_VISIBLE_DEVICES=${GPU:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PKG
cd $PKG
O=$HOME/simcode2/out/sigma_joint/train; HOLD=$HOME/simcode2/out/emitter_heldout/heldout5.pt; P=$HOME/simcode2/out/psf_v2/train
for spec in "$@"; do
  m=${spec%%:*}; ps=${spec#*:}; X=""
  case $m in
    m1809) C=models/m1809/calib_1809_2_psf_v2.json; D1=$P/m1809v2/det/final.pt; D2=$P/m1809v2/p2det/final.pt; DENS=0.0005,0.04; TAU=15,80 ;;
    m25)   C=models/insitu/calib_2p5nM_m2_psf_v2.json; D1=$P/m25v2/det/final.pt; D2=$P/m25v2/p2det/final.pt; DENS=0.0005,0.04; TAU=15,80 ;;
    g13)   C=models/insitu/calib_gauss1.3.json; D1=$HOME/simcode2/runs/psfmatch/g13/det/final.pt; D2=$HOME/simcode2/out/z3d/runs/g13g/p2det/final.pt
           DENS=0.0005,0.06; TAU=3,80; X="--dens $DENS --tau $TAU" ;;
  esac
  out=$O/$m/$ps; mkdir -p $O/$m
  [ -f $out/final.pt ] && { echo "train $m $ps exists"; continue; }
  t0=$(date +%s)
  if [ $ps = p1 ]; then
    python -u -m simcode.emitter.train --out $out --steps 12000 --batch 16 --lr 1e-4 --pct 0.1 --ch 96 --detector $D1 --tau $TAU --hold $HOLD \
      --pair_frac 0.25 --dens $DENS --med 40,2500 --nnw 1 --seed 101 --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 \
      --init $HOME/simcode2/models/$m/emitter_pass1.pt --ckpt_every 4000 --calib $C > $out.log 2>&1
  else
    python -u -m simcode.emitter.train --out $out --detector $D2 --init $HOME/simcode2/models/$m/emitter_pass2.pt --steps 20000 --batch 16 --lr 1e-4 --pct 0.05 \
      --w_mr 2.0 --w_slope 50.0 --seed 141 --eval_every 5000 --ckpt_every 5000 --no_illum 0 --hold_v1 0 --calib $C $X > $out.log 2>&1
  fi
  [ -f $out/final.pt ] && echo "train $m $ps done $(( $(date +%s) - t0 )) s" || echo "train $m $ps FAILED"
done
