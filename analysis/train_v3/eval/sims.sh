#!/bin/bash
# held-out v3 sim movies with ground truth (4800 frames, 120 x 120 px; analysis/simreal/sim_movie_r.py with the knobs of
# analysis/sigma_joint/sims.sh (a1 / a2 / a5, b1 / b2 / b4: v2 camera units), rendered with the PSF v3 calibrations (fit sampling,
# heights from the calibration) and the measured brightness noise of the dataset in the training parametrisation (frame_sd = c,
# step_sd = 0, ang_sd = c_a; simulate/batches bnoise); new seeds (61-65): not seen by any training (the trainers draw their own batches).
# usage: bash sims.sh <gpu> <name> [...]     (out: out/train_v3/eval/sims/<name>)
G=$1; shift; T=/home/jc/simcode2/out/train_v3; source $T/env.sh $G
P=/home/jc/simcode2; D=$P/analysis/simreal; O=$T/eval/sims; mkdir -p $O; cd $T/eval
F05=$D/fields/field_own10_real05.pt; F25=$D/fields/field_own10_real25.pt
C05="SIMP_bg=2.67 SIMP_bg_sd=0.38 SIMP_bg_hp=0.19 SIMP_fluct=0.114"
C25="SIMP_bg=6.24 SIMP_bg_sd=2.28 SIMP_bg_hp=0.34 SIMP_fluct=0.27"
A05="SIM_PRESET=0.5nM SIM_STRUCT=1 SIMP_fil_on=0.97 SIMP_fil_resample=true SIM_SD=0.45 SIMP_fil_density=0.6 SIM_PFIELD=$F05"
A25="SIM_PRESET=2.5nM SIM_SD=0.45 SIM_STRUCT=1 SIMP_fil_on=0.97 SIMP_fil_resample=true SIMP_fil_density=0.8 SIM_PFIELD=$F25 SIM_DIM_TAU=20"
P18="SIM_CALIB=$T/calib/calib_1809_2_psf_v3.json $C05 SIM_FRAME_SD=0.093 SIM_STEP_SD=0 SIM_ANG_SD=0.084"
P25="SIM_CALIB=$T/calib/calib_2p5nM_m2_psf_v3.json $C25 SIM_FRAME_SD=0.190 SIM_STEP_SD=0 SIM_ANG_SD=0.065"
mk() { local nm=$1 seed=$2; shift 2; [ -f $O/$nm/sim_movie.tif ] && { echo "sim $nm exists"; return 0; }; mkdir -p $O/$nm
  local t0=$(date +%s)
  env "$@" python -u $T/with_pkg.py $D/sim_movie_r.py $O/$nm 4800 $seed > $O/$nm.log 2>&1 || { echo "sim $nm FAILED"; return 1; }; echo "sim $nm done $(( $(date +%s) - t0 )) s"; }
for nm in "$@"; do case $nm in
  v3a1) mk v3a1 61 $A05 $P18 SIM_MED=532 SIMP_dens=0.0105 ;;
  v3a2) mk v3a2 62 $A05 $P18 SIM_MED=532 SIMP_dens=0.021 ;;
  v3a5) mk v3a5 65 $A05 $P18 SIM_MED=1300 SIMP_dens=0.0105 ;;
  v3b1) mk v3b1 61 $A25 $P25 SIM_MED=570 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 ;;
  v3b2) mk v3b2 62 $A25 $P25 SIM_MED=400 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 ;;
  v3b3) mk v3b3 63 $A25 $P25 SIM_MED=800 SIM_DIM_DENS=0.02 SIM_DIM_MED=228 SIMP_dens=0.016 ;;
  v3b4) mk v3b4 64 $A25 $P25 SIM_MED=1100 SIM_DIM_DENS=0.01 SIM_DIM_MED=228 SIMP_dens=0.030 ;;
  *) echo "unknown sim $nm"; exit 1 ;;
esac; done
