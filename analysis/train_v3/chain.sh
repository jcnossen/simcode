#!/bin/bash
# v3 retrain (2026-11-18) of a paper model on the in-situ PSF v3 (out/psf_v3; frozen calibration copies out/train_v3/calib, with
# the fit sampling npupil 64 / oversample 2) with the dataset's measured brightness noise (--bnoise c,c_a; simulate/batches
# TrainingBatches bnoise). The full recipe of models/README.md ("How a model is made" + "Joint prior / intensity covariance"),
# stage names of simcode.pipeline.training:
#   detector 40k -> emitter 18k (init pretrain_emitter_pass1.pt)
#     -> [A] detector_pass2 40k (dedicated, --illum 1) -> detector_pass2_mix 40k -> detector_pass2_cont 40k       (detector chain)
#     -> [B] emitter_joint 12k (pass-1 joint fine-tune) -> emitter_pass2 24k (det detector_pass2/ckpt_25000, pert v1)
#            -> emitter_pass2_long 50k (det detector_pass2/final, hold_v1 15) -> emitter_pass2_final 50k (hold_v1 0)
#            -> emitter_pass2_joint 20k (pass-2 joint fine-tune)                                                     (emitter chain)
#   [A] and [B] run concurrently on the same GPU. Model = detector, emitter_joint, detector_pass2_cont, emitter_pass2_joint (final.pt).
# Training heights from the calibration (emitter_z_quantiles_um, clipped to emitter_z_range_um; no flag); camera = the
# calibration's gain 0.4256 / per-pixel read-noise map (v2 maps), no extra white read noise. Frozen package SRC (src_e8fc62f4,
# psfclean e8fc62f4) + torch-psf snapshot out/psf_v3/tpsf_64871e5 (env.sh). Stages with final.pt are skipped (resumable).
# usage: setsid nohup bash chain.sh <m1809|m25> <gpu> > logs/chain_<m>.log 2>&1 &
M=$1; G=$2
T=/home/jc/simcode2/out/train_v3
source $T/env.sh $G
cd $SRC
HOLD=/home/jc/simcode2/out/emitter_heldout/heldout5.pt
case $M in
  m1809) C=$T/calib/calib_1809_2_psf_v3.json; BN=0.093,0.084 ;;
  m25)   C=$T/calib/calib_2p5nM_m2_psf_v3.json; BN=0.190,0.065 ;;
  *) echo "unknown model $M"; exit 1 ;;
esac
D=$T/runs/$M; mkdir -p $D $T/logs
log() { echo "$(date +%F_%T) $M: $*" >> $T/stages.log; }
stage() {  # name cmd...
  local n=$1; shift
  if [ -f $D/$n/final.pt ]; then log "$n: final.pt exists, skip"; return 0; fi
  log "$n start on GPU $G"
  "$@" > $D/$n.log 2>&1; local rc=$?
  log "$n end rc=$rc"
  [ -f $D/$n/final.pt ] || { log "$n FAILED (no final.pt)"; return 1; }
}
X="--calib $C --bnoise $BN"
echo "$(date) chain $M calib $C bnoise $BN GPU $G src $SRC torch_psf $TPSF"
stage detector python -u -m simcode.detector.train --out $D/detector --steps 40000 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 --nnw 1 \
  --seed 0 $X || exit 1
stage emitter python -u -m simcode.emitter.train --out $D/emitter --steps 18000 --batch 16 --lr 1.5e-4 --ch 96 --detector $D/detector/final.pt \
  --tau 15,80 --hold $HOLD --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 --w_mr 2.0 --w_slope 50.0 --bf16 1 \
  --eval_every 2000 --init pretrain_emitter_pass1.pt --pct 0.1 --ckpt_every 6000 $X || exit 1
(
  stage detector_pass2 python -u -m simcode.detector.train --out $D/detector_pass2 --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 \
    --w_off 1.0 --seed 11 --eval_every 2500 --ckpt_every 5000 --illum 1 --init $D/detector/final.pt $X || exit 1
  stage detector_pass2_mix python -u -m simcode.detector.train --out $D/detector_pass2_mix --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 \
    --w_off 1.0 --seed 61 --eval_every 5000 --ckpt_every 10000 --init $D/detector_pass2/final.pt --pass1_ref $D/detector/final.pt $X || exit 1
  stage detector_pass2_cont python -u -m simcode.detector.train --out $D/detector_pass2_cont --steps 40000 --batch 16 --size 64 --lr 1e-4 --pct 0.05 \
    --w_off 1.0 --seed 71 --eval_every 5000 --ckpt_every 10000 --init $D/detector_pass2_mix/final.pt --pass1_ref $D/detector/final.pt $X || exit 1
) &
PD=$!
stage emitter_joint python -u -m simcode.emitter.train --out $D/emitter_joint --init $D/emitter/final.pt --detector $D/detector/final.pt \
  --steps 12000 --batch 16 --lr 1e-4 --pct 0.1 --ch 96 --tau 15,80 --hold $HOLD --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 \
  --seed 101 --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --ckpt_every 4000 $X || exit 1
until [ -f $D/detector_pass2/ckpt_25000.pt ] || [ -f $D/detector_pass2/final.pt ]; do
  kill -0 $PD 2>/dev/null || [ -f $D/detector_pass2/final.pt ] || { log "detector_pass2 died"; exit 1; }; sleep 30; done
P25=$D/detector_pass2/ckpt_25000.pt; [ -f $P25 ] || P25=$D/detector_pass2/final.pt
stage emitter_pass2 python -u -m simcode.emitter.train --out $D/emitter_pass2 --detector $P25 --init $D/emitter/final.pt --steps 24000 \
  --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0 --pert_model v1 $X || exit 1
until [ -f $D/detector_pass2/final.pt ]; do kill -0 $PD 2>/dev/null || [ -f $D/detector_pass2/final.pt ] || { log "detector_pass2 died"; exit 1; }; sleep 30; done
stage emitter_pass2_long python -u -m simcode.emitter.train --out $D/emitter_pass2_long --detector $D/detector_pass2/final.pt \
  --init $D/emitter_pass2/final.pt --steps 50000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 \
  --ckpt_every 10000 --no_illum 0 --hold_v1 15 $X || exit 1
stage emitter_pass2_final python -u -m simcode.emitter.train --out $D/emitter_pass2_final --detector $D/detector_pass2/final.pt \
  --init $D/emitter_pass2_long/final.pt --steps 50000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 41 --eval_every 5000 \
  --ckpt_every 10000 --no_illum 0 --hold_v1 0 $X || exit 1
stage emitter_pass2_joint python -u -m simcode.emitter.train --out $D/emitter_pass2_joint --init $D/emitter_pass2_final/final.pt \
  --detector $D/detector_pass2/final.pt --steps 20000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 141 --eval_every 5000 \
  --ckpt_every 5000 --no_illum 0 --hold_v1 0 $X || exit 1
wait $PD || { log "detector chain failed"; exit 1; }
log "chain done"
echo "$(date) chain $M done"
