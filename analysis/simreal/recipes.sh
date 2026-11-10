#!/bin/bash
# Benchmark movies of the SIMCODE v2 evaluation (analysis/simreal/sim_movie_r.py, realistic RealSim knobs via env).
# 120 x 120 px uint16 TIFF + sim_gt.npz + gt_pattern.pt.
#   m05r / m25rc   realistic 0.5 / 2.5 nM-like (4800 frames; c-matched, the headline sims of "sim realism + p2L")
#   m05pd / m25pd  punctate repeated-site benchmarks (30k frames; evaluation protocol; frozen in the paper analysis)
#   m05pe / m25pe  m*pd + the measured illumination envelope (critic 37 #2)
# usage: GPU=1 bash analysis/simreal/recipes.sh <name> <out_dir> [n_frames]
set -euo pipefail
nm=$1; out=$2
D=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$D/../.." && pwd)
export CUDA_VISIBLE_DEVICES=${GPU:-1} TORCH_DISABLE_NATIVE_JIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
F05=$D/fields/field_own10_real05.pt; F25=$D/fields/field_own10_real25.pt
R05="SIM_PRESET=0.5nM SIM_MED=700 SIM_STRUCT=1 SIMP_fil_on=0.97 SIMP_fil_resample=true SIMP_dens=0.0105"
P_COMMON="SIM_LAZY=1 SIM_FRAME_SD=0.13 SIM_STEP_SD=0.05 SIM_STRUCT=1 SIMP_fil_on=0.97 SIMP_fil_resample=true SIMP_fil_density=0.45 SIMP_site_jit=2 SIMP_tau_tf=0.2 SIMP_tau_tm=3 SIM_DIM_MED=300 SIM_DIM_TAU=40 SIM_OFFERR=0.6 SIM_RNMULT=1.2 SIM_SHIFT_LIN=5 SIM_SHIFT_RMS=2 SIM_SHIFT_SIG=25"
case $nm in
  m05r)  NF=4800;  SEED=7;  ENV="$R05 SIM_SD=0.45 SIM_FRAME_SD=0.09 SIM_STEP_SD=0.14 SIMP_fil_density=0.6 SIM_PFIELD=$F05" ;;
  m25rc) NF=4800;  SEED=7;  ENV="SIM_PRESET=2.5nM SIM_MED=750 SIM_SD=0.45 SIM_FRAME_SD=0.09 SIM_STEP_SD=0.10 SIM_STRUCT=1 SIMP_fil_on=0.97 SIMP_fil_resample=true SIMP_fil_density=0.8 SIM_PFIELD=$F25 SIM_DIM_DENS=0.02 SIM_DIM_MED=300 SIM_DIM_TAU=20" ;;
  m05pd) NF=30000; SEED=11; ENV="$P_COMMON SIM_PRESET=0.5nM SIM_MED=700 SIM_SD=0.45 SIMP_dens=0.0105 SIMP_site_ev=2 SIM_DIM_DENS=0.004 SIM_PFIELD=$F25" ;;
  m25pd) NF=30000; SEED=12; ENV="$P_COMMON SIM_PRESET=2.5nM SIM_MED=750 SIM_SD=0.45 SIMP_site_ev=10 SIM_DIM_DENS=0.02 SIM_PFIELD=$F05" ;;
  m05pe) NF=30000; SEED=21; ENV="$P_COMMON SIM_PRESET=0.5nM SIM_MED=700 SIM_SD=0.39 SIMP_dens=0.0105 SIMP_site_ev=2 SIMP_bg_sd=0.38 SIMP_env_x0=73.5142 SIMP_env_y0=64.8650 SIMP_env_s1=61.9973 SIMP_env_s2=50.6720 SIMP_env_th=2.4998 SIMP_env_bg=0.28 SIM_DIM_DENS=0.004 SIM_PFIELD=$F25" ;;
  m25pe) NF=30000; SEED=22; ENV="$P_COMMON SIM_PRESET=2.5nM SIM_MED=750 SIM_SD=0.39 SIMP_site_ev=10 SIMP_bg_sd=1.3 SIMP_env_x0=64.9553 SIMP_env_y0=9.8454 SIMP_env_s1=78.5102 SIMP_env_s2=66.2635 SIMP_env_th=-1.1616 SIMP_env_bg=1.0 SIM_DIM_DENS=0.02 SIM_PFIELD=$F05" ;;
  *) echo "unknown recipe $nm"; exit 1 ;;
esac
NF=${3:-$NF}
mkdir -p "$out"
env $ENV $PY "$D/sim_movie_r.py" "$out" "$NF" "$SEED"
