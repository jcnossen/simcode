#!/bin/bash
# sigma study (2026-11-17): Figs 3 / 4 (fig_exp_new.py, the paper's fair protocol: disjoint windows, max(sigma) < 0.15 px,
# 300-frame halves, own DME v2 drift) with a SIMCODE run of the joint-covariance networks; SIMCODE rows = the paper's selection at 16.25 nm.
# The other methods' rows and drifts are those of the previous figure (out/sigma_work/partA/fig, --reuse_drift). Output:
# out/sigma_joint/fig/<label>/ (not installed into the paper).
# usage: GPU=<g> bash fig.sh <1809_2|2.5nM_2> <SIMCODE run dir or table> <label> [--nopsf]
set -e
source ~/miniconda3/bin/activate simcode-py311
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd $REPO
export CUDA_VISIBLE_DEVICES=${GPU:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$REPO
DS=$1; V2=$2; LAB=$3; P2=$HOME/simcode2/out/paper2/$DS; OLD=$HOME/simcode2/out/sigma_work/partA/fig
case $DS in 1809_2) TAG=fig3_psfv2; CAL=models/insitu/calib_1809_2_psf_v2.json ;; 2.5nM_2) TAG=fig4_psfv2; CAL=models/insitu/calib_2p5nM_m2_psf_v2.json ;; esac
PSF="--psf $CAL"; [ "$4" = "--nopsf" ] && PSF=""
NOTE='SIMCODE: emitter networks with the joint prior / intensity covariance, uncertainty = the solve posterior sd (no calibration factor); selection p >= 0.63 and max(sigma_x, sigma_y) <= 16.25 nm.'
python -u analysis/paper/fig_exp_new.py $DS --smlm $P2/baselines_psfv2/smlm/smlm.npy --sf $P2/baselines_psfv2/sf/sf.hdf5 \
  --decode $P2/decode/locs_fixed.npz --tag $TAG --outdir $HOME/simcode2/out/sigma_joint/fig/$LAB --v2 $V2 --v2_sigma_max 16.25 \
  --reuse_drift $OLD/${DS}_$TAG $PSF --note "$NOTE"
