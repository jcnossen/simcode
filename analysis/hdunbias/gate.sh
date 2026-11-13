#!/bin/bash
# gate scenes (iso, rs05qm, pair1, pair2; seeds 301-303; 1809_2 averaged PSF), m1809 pass-1 front-end with detector det_m1809
# unchanged, head $2, solve calibration of hd_m1809. usage: gate.sh <tag> <head> <gpu>
source /home/jc/simcode2/runs/psfmatch/env.sh
T=$1; HEAD=$2; O=out/hdunbias/gate; mkdir -p $O
export SIMCODE_GATE_CALIB=models/insitu/calib_1809_2_psf3d_avg.json GATE_PSF_CHUNK=${GATE_PSF_CHUNK:-1500} CUDA_VISIBLE_DEVICES=$3
SC_DET=models/det_m1809.pt python -u analysis/hdunbias/wrap.py g$T gate --scen iso,rs05qm,pair1,pair2 --seeds 301,302,303 \
  --sc $HEAD --sccal calib/calib_sc_hd_m1809.json --out $O --tag _$T --ntarget 3500 > $O/run_$T.log 2>&1
if grep -q '^OOM' $O/run_$T.log; then echo "OOM in run_$T: outputs removed" >> $O/oom.log; rm -f $O/*__s30?_$T.npz; mv $O/run_$T.log $O/run_$T.oom.log; fi
