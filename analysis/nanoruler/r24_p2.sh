#!/bin/bash
# R24: the pass-2 repeat for nr40 and nr20 with the nr80 pass-2 networks. Resumable (every finished output is skipped); REFUSES to start before
# the nr80 evaluation chain has finished ("eval nr80 done" in out/nanoruler/eval/chain.log and the pass-2 files of the set exist).
# usage: cd ~/simcode2; setsid nohup bash analysis/nanoruler/r24_p2.sh [nr40 nr20] >> out/nanoruler/nr40/logs/r24_p2.log 2>&1 < /dev/null &
# Per movie: (1) pass 2 of the simulated movies (same run directories, R24_P2=1 r24_run.sh) + GT evaluation; (2) pass 2 of the real movie;
# (3) method tables / site-pair analysis (r24_methods.sh <mv> p2); (4) captures of the pass-2 head outputs (abl_capture.py through r24_abl.py,
# detector batch 8) and the solve variants on the real movie and the two grid simulations (r24_variants.sh ... p2); (5) the reuse gates of
# pass 2 (analysis/nanoruler/r24r_p2.sh of the reuse worker, if present). Every GPU command goes through r24_gpu.sh (one at a time).
# Estimated GPU time: nr40 ~0.6 GPU-h (real pass 2 ~0.3, captures ~0.2, sims / solves ~0.1), nr20 ~2 GPU-h (real pass 2 ~0.8, capture ~0.6,
# solves ~0.4, sims ~0.2); no single job over 1 GPU-h. NOT applied: the selection-aware sigma rescale sigu_p2_nr80.json (simcode2.strict
# --sig_u): the tables carry the sim-sigma calibration only, as in the nr80 ablation.
source /home/jc/simcode2/runs/psfmatch/env.sh
A=analysis/nanoruler; NR=$PWD/out/nanoruler
grep -q "eval nr80 done" $NR/eval/chain.log && [ -f $NR/models/p2det_nr80.pt ] && [ -f $NR/models/p2headL_nr80.pt ] && [ -f $NR/models/calib/calib_p2L_nr80.json ] && [ -f $NR/eval/fe_nr80/rs05qm__s202_s2.npz ] \
  || { echo "$(date +%F_%T) r24_p2: the nr80 evaluation chain has not finished (no 'eval nr80 done' or pass-2 files missing): not started"; exit 1; }
say() { echo "$(date +%F_%T) $*"; }
for MV in ${@:-nr40 nr20}; do
  O=$NR/$MV; case $MV in nr40) SIM=s40;; nr20) SIM=s20;; esac
  for s in ${SIM}g_a ${SIM}g_c0 ${SIM}_a ${SIM}_c0; do [ -d $O/sim/$s ] || continue
    R24_P2=1 bash $A/r24_run.sh $MV $s || { say "FAILED pass 2 $s"; continue; }
    [ -f $O/simeval/gt_${s}_p2.json ] || python $A/r24_simeval.py $O/runs/$s/locs_p2.npz $O/sim/$s --out $O/simeval/gt_${s}_p2.json --name "$s pass 2" > /dev/null 2>&1
  done
  R24_P2=1 bash $A/r24_run.sh $MV real || { say "FAILED pass 2 real $MV"; continue; }
  bash $A/r24_methods.sh $MV p2 || say "FAILED methods $MV p2"
  for s in p2 simga_p2 simgc0_p2; do
    bash $A/r24_gpu.sh 6 python $A/r24_abl.py $MV capture $s 2>&1 | grep -v "Warning\|r24_gpu" > $O/logs/capture_$s.log || { say "FAILED capture $MV $s"; continue; }
    P=$O/pairs_auto_sf.json
    if [ $s != p2 ]; then P=$O/simeval/pairs_truth_${SIM}g.json
      [ -f $P ] || python $A/r24_rul.py pairs --truth $O/sim/${SIM}g_a/sites.npz --d0 ${MV#nr} --out $P --fov 0 16640 > /dev/null; fi
    bash $A/r24_variants.sh $MV $s $P || say "FAILED variants $MV $s"
  done
  say "r24_p2 $MV done"
done
[ -f $A/r24r_p2.sh ] && bash $A/r24r_p2.sh
say "r24_p2 all done"
