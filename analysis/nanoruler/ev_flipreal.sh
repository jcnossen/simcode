#!/bin/bash
# nr80 evaluation, step-order flip on the real 80 nm movie: the reordered cache rewritten with the steps of each orientation
# reversed (ev_flipsteps.py, idle I/O priority, 60 MB/s) and both passes run the production way (as analysis/nanoruler/r24_run.sh:
# ev_run.py, crop 5,5,390, chunk 1500, EV_DET_BATCH=8, --autoc 0, nr80 networks, sim-sigma calibrations cal_sim / p2_cal_sim of
# netset_nr80.json, sigma rescale sigu_p1_nr80.json / the refit sigu_p2_nr80_flip.json). Movie and run dir on the data disk
# (/mnt/data/dmdsf/simcode2_out/nanoruler/{raw,runs/real_flip}), symlink out/nanoruler/runs/real_flip. One GPU job.
# NOT RUN YET (2026-11-13): writing the 9.6 GB movie to the data disk was not permitted in the session that wrote this script.
# usage: setsid nohup bash analysis/nanoruler/ev_flipreal.sh >> out/nanoruler/eval/flipreal.log 2>&1 < /dev/null &
source /home/jc/simcode2/runs/psfmatch/env.sh; set -o pipefail
A=analysis/nanoruler; NR=$PWD/out/nanoruler; MD=$NR/models; MC=$MD/calib; BIG=/mnt/data/dmdsf/simcode2_out/nanoruler
say() { echo "$(date +%F_%T) $*"; }; die() { say "FAILED: $*"; exit 1; }
MOV=$BIG/raw/sim4_1_reord_flip.tif; RD=$NR/runs/real_flip
[ -f $MOV ] || python $A/ev_flipsteps.py movie $BIG/raw/sim4_1_reord.tif $MOV > $NR/eval/logs/flip_real.log 2>&1 || die "flip (see $NR/eval/logs/flip_real.log)"
mkdir -p $BIG/runs/real_flip; [ -e $RD ] || ln -s $BIG/runs/real_flip $RD
if [ ! -f $RD/locs_p2.npz ]; then
  g=$(NEED_GB=6 pickgpu 6 "2 0 1"); say "real run on GPU $g"
  CUDA_VISIBLE_DEVICES=$g EV_DET_BATCH=8 python -u $A/ev_run.py --movie $MOV --calib $NR/calib_nr80.json --out $RD --crop 5,5,390 --chunk 1500 --autoc 0 --pass2 --cand_pmin 0.3 \
    --det $MD/det_nr80.pt --head $MD/hd_nr80.pt --cal $MC/calib_sc_hd_nr80.json --p2_det $MD/p2det_nr80.pt --p2_head $MD/p2headL_nr80.pt --p2_cal $MC/calib_p2L_nr80.json \
    --sig_u $MC/sigu_p1_nr80.json,$MC/sigu_p2_nr80_flip.json 2>&1 | grep -v "Warning\|^W1[0-9]\{3\} " > $NR/eval/logs/run_real_flip.log || die "real run (see $NR/eval/logs/run_real_flip.log)"
fi
[ -f $RD/locs_p2.npz ] && say "flipreal done" || die "no locs_p2.npz"
