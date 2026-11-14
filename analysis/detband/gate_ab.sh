#!/bin/bash
# gate scenes (iso, rs05qm, pair1, pair2; seeds 301-303; 1809_2 averaged PSF) with the m1809 pass-1 front-end, only the
# detector ($2) and/or head ($3, range $4) swapped; pass-1 solve calibration of hd_m1809 unchanged. GPU 1.
# usage: gate_ab.sh <tag> <det> [head] [R]
source /home/jc/simcode2/runs/psfmatch/env.sh
T=$1; DET=$2; HEAD=${3:-models/hd_m1809.pt}; R=${4:-0.5}; O=out/detband/gate; mkdir -p $O
export SIMCODE_GATE_CALIB=models/insitu/calib_1809_2_psf3d_avg.json GATE_PSF_CHUNK=${GATE_PSF_CHUNK:-1500} CUDA_VISIBLE_DEVICES=1
SC_DET=$DET DETBAND_HRANGE=$R python -u analysis/detband/wrap.py g$T gate --scen iso,rs05qm,pair1,pair2 --seeds 301,302,303 \
  --sc $HEAD --sccal calib/calib_sc_hd_m1809.json --out $O --tag _$T --ntarget 3500 > $O/run_$T.log 2>&1
# a scene iteration skipped after an OOM keeps its GT but loses its detections (gate.py) -> discard and let the caller rerun
if grep -q '^OOM' $O/run_$T.log; then echo "OOM in run_$T: outputs removed" >> $O/oom.log; rm -f $O/*__s30?_$T.npz; mv $O/run_$T.log $O/run_$T.oom.log; fi
