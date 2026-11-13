#!/bin/bash
# R24: pass-1 pipeline run (nr80 pass-1 networks, sim-sigma calibration, fixed c = c_a = 0.10 i.e. --autoc 0, as out/nanoruler/runs/real_p1)
# of a simulated movie or of the real nr40 / nr20 movie, through ev_run.py + r24_gpu.sh (lock, EV_DET_BATCH=8). Skipped when locs_p1.npz exists.
# usage: bash analysis/nanoruler/r24_run.sh <nr40|nr20> <sim name | real> [extra simcode2.run arguments, e.g. --nframes 3000 --out DIR]
# R24_P2=1: the same run directory continued with pass 2 (--pass2 --cand_pmin 0.3, nr80 pass-2 networks and sim-sigma calibration; finished
#   pass-1 chunks are skipped by simcode2.run); marker locs_p2.npz, log run_<name>_p2.log.
source /home/jc/simcode2/runs/psfmatch/env.sh
MV=$1; N=$2; shift 2; NR=$PWD/out/nanoruler; O=$NR/$MV; A=analysis/nanoruler
if [ $N = real ]; then MOV=/mnt/data/dmdsf/simcode2_out/nanoruler/raw_$MV/movie_reord.tif; CAL=$NR/calib_$MV.json; CROP=5,5,390
else MOV=$O/big/sim/$N/sim_movie.tif; CAL=$O/big/sim/$N/calib_sim.json; CROP=5,5,246; fi
OUT=$O/big/runs/$N; EX="$*"
case "$EX" in *--out*) OUT=$(echo "$EX" | sed 's/.*--out \([^ ]*\).*/\1/'); EX=$(echo "$EX" | sed 's/--out [^ ]*//');; esac
MK=locs_p1.npz; SFX=""; if [ "${R24_P2:-0}" = 1 ]; then MK=locs_p2.npz; SFX=_p2; EX="$EX --pass2 --cand_pmin 0.3 --p2_det $NR/models/p2det_nr80.pt --p2_head $NR/models/p2headL_nr80.pt --p2_cal $NR/models/calib/calib_p2L_nr80.json"; fi
mkdir -p $OUT $O/logs; [ -f $OUT/$MK ] && { echo "$OUT: $MK exists, skip"; exit 0; }
bash $A/r24_gpu.sh 6 python -u $A/ev_run.py --movie $MOV --calib $CAL --out $OUT --crop $CROP --chunk 1500 --autoc 0 \
  --det $NR/train/det/final.pt --head $NR/train/hd/final.pt --cal $NR/models/calib/calib_sc_hd_nr80.json $EX 2>&1 | grep -v "Warning\|^W1[0-9]\{3\} " > $O/logs/run_$(basename $OUT)$SFX.log
[ -f $OUT/$MK ] && echo "$OUT $MK done" || { echo "$OUT FAILED (see $O/logs/run_$(basename $OUT)$SFX.log)"; exit 1; }
