#!/bin/bash
# sigma study (2026-11-17): simulated movies with ground truth for the calibration / precision tests of m1809, m25, g13
# (4800 frames, 120 x 120 px; analysis/simreal/sim_movie_r.py with the knobs of out/psf_v2/eval/sims.sh (m05r3v2 / m25rc3v2:
# v2 PSF, v2 camera units), varied in seed, density and photons; g13: the m05r3 / m25rc3 recipes rendered with the Gaussian
# sigma 1.3 px PSF in the 0.56 e/ADU camera units of those recipes).
# usage: SRC=<paper repo> GPU=<g> bash sims.sh <name> [...]     (out: ~/simcode2/out/sigma_layer/sims/<name>)
source ~/miniconda3/bin/activate simcode-py311; cd $SRC
export CUDA_VISIBLE_DEVICES=${GPU:-1} CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$SRC
O=$HOME/simcode2/out/sigma_layer/sims; M=$SRC/models/insitu; D=$SRC/analysis/simreal
F05=$D/fields/field_own10_real05.pt; F25=$D/fields/field_own10_real25.pt
C05="SIMP_bg=2.67 SIMP_bg_sd=0.38 SIMP_bg_hp=0.19 SIMP_fluct=0.114"
C25="SIMP_bg=6.24 SIMP_bg_sd=2.28 SIMP_bg_hp=0.34 SIMP_fluct=0.27"
A05="SIM_PRESET=0.5nM SIM_STRUCT=1 SIMP_fil_on=0.97 SIMP_fil_resample=true SIM_SD=0.45 SIM_FRAME_SD=0.09 SIM_STEP_SD=0.14 SIMP_fil_density=0.6 SIM_PFIELD=$F05"
A25="SIM_PRESET=2.5nM SIM_SD=0.45 SIM_FRAME_SD=0.09 SIM_STEP_SD=0.10 SIM_STRUCT=1 SIMP_fil_on=0.97 SIMP_fil_resample=true SIMP_fil_density=0.8 SIM_PFIELD=$F25 SIM_DIM_TAU=20"
P18="SIM_CALIB=$M/calib_1809_2_psf_v2.json $C05"; P25="SIM_CALIB=$M/calib_2p5nM_m2_psf_v2.json $C25"; PG="SIM_CALIB=$M/calib_gauss1.3.json"
mk() { local nm=$1 seed=$2; shift 2; [ -f $O/$nm/sim_movie.tif ] && { echo "sim $nm exists"; return 0; }; mkdir -p $O/$nm
  local t0=$(date +%s)
  env "$@" python -u $D/sim_movie_r.py $O/$nm 4800 $seed > $O/$nm.log 2>&1 || { echo "sim $nm FAILED"; return 1; }; echo "sim $nm done $(( $(date +%s) - t0 )) s"; }
for nm in "$@"; do case $nm in
  # m1809 (PSF 1809/2 v2): base = m05r3v2 (SIM_MED 532, dens 0.0105)
  a1) mk a1 11 $A05 $P18 SIM_MED=532 SIMP_dens=0.0105 ;;
  a2) mk a2 12 $A05 $P18 SIM_MED=532 SIMP_dens=0.021 ;;
  a3) mk a3 13 $A05 $P18 SIM_MED=350 SIMP_dens=0.0105 ;;
  a4) mk a4 14 $A05 $P18 SIM_MED=850 SIMP_dens=0.006 ;;
  a5) mk a5 15 $A05 $P18 SIM_MED=1300 SIMP_dens=0.0105 ;;
  ah2) mk ah2 21 $A05 $P18 SIM_MED=650 SIMP_dens=0.015 ;;
  ah3) mk ah3 22 $A05 $P18 SIM_MED=1000 SIMP_dens=0.008 ;;
  # m25 (PSF 2.5 nM m2 v2): base = m25rc3v2 (SIM_MED 570, preset dens 0.024, dim population 0.02 / 228)
  b1) mk b1 11 $A25 $P25 SIM_MED=570 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 ;;
  b2) mk b2 12 $A25 $P25 SIM_MED=400 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 ;;
  b3) mk b3 13 $A25 $P25 SIM_MED=800 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 SIMP_dens=0.016 ;;
  b4) mk b4 14 $A25 $P25 SIM_MED=1100 SIM_DIM_DENS=0.01 SIM_DIM_MED=228 SIMP_dens=0.030 ;;
  bh2) mk bh2 21 $A25 $P25 SIM_MED=650 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 ;;
  bh3) mk bh3 22 $A25 $P25 SIM_MED=450 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 SIMP_dens=0.032 ;;
  # g13 (Gaussian sigma 1.3 px): the m05r3 / m25rc3 recipes in their 0.56 e/ADU units (preset background)
  c1) mk c1 11 $A05 $PG SIM_MED=700 SIMP_dens=0.0105 ;;
  c2) mk c2 12 $A05 $PG SIM_MED=1200 SIMP_dens=0.0105 ;;
  c3) mk c3 13 $A25 $PG SIM_MED=600 SIM_DIM_DENS=0.02 SIM_DIM_MED=300 ;;
  ch1) mk ch1 21 $A05 $PG SIM_MED=900 SIMP_dens=0.015 ;;
  ch2) mk ch2 22 $A25 $PG SIM_MED=450 SIM_DIM_DENS=0.02 SIM_DIM_MED=300 ;;
  *) echo "unknown sim $nm"; exit 1 ;;
esac; done
