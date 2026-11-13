#!/bin/bash
# Task 1 (2026-11-13): current production pass-2 sets m1809 / m25 on the standard gate scenes rendered with each movie's MEASURED
# per-step phase spacings (realsim opt-in key step_offs; presets out/p2generic/presets/real_<ds>.json), same seeds / emitters as
# the acceptance gates (out/psfmatch/gate_<tag>, fe_<tag>: trained-range spacing). gate.py = CRLB + reference front; dump s1 / s2.
# usage: GPU=<g> bash analysis/p2generic/t1_run.sh <m1809|m25>
source /home/jc/simcode2/runs/psfmatch/env.sh; set -o pipefail
S=$1; export CUDA_VISIBLE_DEVICES=${GPU:-0}
declare -A DS=([m1809]=1809_2 [m25]=2.5nM_2) SCN=([m1809]=iso,pair1,pair2,rs05qm,rs05pm [m25]=iso,pair1,pair2,rs25qm)
P=out/p2generic; G=$P/gate_${S}_real; FE=$P/fe_${S}_real; mkdir -p $G $FE
export P2G_PRESET=$PWD/$P/presets/real_${DS[$S]}.json P2G_TITLE=p2gen-t1 SIMCODE_GATE_CALIB=${PSF[$S]} GATE_PSF_CHUNK=1500
W="python -u analysis/p2generic/pwrap.py"
$W simcode.hd.gate --scen ${SCN[$S]} --fronts sc --sc hd_sc5nb.pt --sccal calib/calib_sc_hd_sc5nbb15_pm_bright.json --tag _hd_cur --out $G --ntarget 3500 --seeds 301,302,303 || exit 1
SIMCODE_GATE_DIR=$G $W simcode.p2.dump --scen ${SCN[$S]} --seeds 301,302,303 --fronts s1=det_$S.pt:hd_$S.pt,s2=p2det_$S.pt:p2headL_$S.pt --out $FE || exit 1
echo "t1 $S done"
