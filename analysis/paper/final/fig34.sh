#!/bin/bash
# Final method (2026-11-17): Figs 3 / 4 with SIMCODE = the run's pass-2 solve table locs_p2_solve (frc_fair.V2_TABLE; posterior mean and
# posterior sd of the pattern solve; detection p >= 0.63; user decision 2026-11-17) and DECODE = the v3 models (dec_m1809v3 for 1809_2, dec_m25v3 for 2.5nM_2: <ds>/decode/locs_v3.npz). Common
# rule: each method's own filters + max(sigma) < 0.15 px on its own reported sigma (fig_exp_new.py, fair FRC: disjoint windows, 300-frame
# halves, own DME v2 drift). SMLM / SIMFLUX rows and drifts = those of the current paper figures (out/sigma_work/partA/fig); the
# DECODE v3 drift of 2.5nM_2 = the one of its decv3 figure (out/decode_checks/v3/fig4_decv3), the others are estimated here.
# usage: GPU=<g> bash fig34.sh <1809_2|2.5nM_2> <SIMCODE run dir> <out dir>
set -e
source ~/miniconda3/bin/activate simcode-py311
REPO=$(cd "$(dirname "$0")/../../.." && pwd); cd $REPO
export CUDA_VISIBLE_DEVICES=${GPU:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$REPO
DS=$1; RUN=$2; OUT=$3; P2=$HOME/simcode2/out/paper2/$DS; OLD=$HOME/simcode2/out/sigma_work/partA/fig; V3=$HOME/simcode2/out/decode_checks/v3/fig4_decv3
case $DS in 1809_2) TAG=fig3_psfv2; CAL=models/insitu/calib_1809_2_psf_v2.json ;; 2.5nM_2) TAG=fig4_psfv2; CAL=models/insitu/calib_2p5nM_m2_psf_v2.json ;; esac
mkdir -p $OUT
for k in smlm sf; do [ -f $OUT/${DS}_${TAG}_${k}_drift.npy ] || cp $OLD/${DS}_${TAG}_${k}_drift.npy $OUT/${DS}_${TAG}_${k}_drift.npy; done
[ $DS = 2.5nM_2 ] && { [ -f $OUT/${DS}_${TAG}_decode_drift.npy ] || cp $V3/${DS}_${TAG}_decv3_decode_drift.npy $OUT/${DS}_${TAG}_decode_drift.npy; }
NOTE='SIMCODE: the pass-2 pattern solve (emitter networks with the joint position / intensity covariance, brightness noise estimated with the pattern), its posterior mean and posterior uncertainty (on held-out simulations rms(error / posterior sigma) = 1.2-1.3 per photon bin with the estimated brightness noise, i.e. the posterior sigma is 20-30 % smaller than the actual error); detection p >= 0.63. DECODE: retrained on the in-situ PSF v2 with its published training settings.'
python -u analysis/paper/fig_exp_new.py $DS --smlm $P2/baselines_psfv2/smlm/smlm.npy --sf $P2/baselines_psfv2/sf/sf.hdf5 \
  --decode $P2/decode/locs_v3.npz --tag $TAG --outdir $OUT --v2 $RUN --v2_sigma_max 16.25 --psf $CAL --note "$NOTE"
