#!/bin/bash
# Nanoruler pass 2 with the GENERIC nr80 pass-2 set (gnr80, worker p2generic: runs/p2generic/gnr80, chain_gen2.sh / eval_nr.sh).
# PREPARED 2026-11-13 evening, NOT run by the nanoruler worker. Refuses to start until the generic set's files exist.
# Per movie: a new run directory <run>_p2gen whose pass-1 chunks, pattern and pass-1 solve are SYMLINKS to the finished pass-1 run (the
# valid pass-1 result is not touched or recomputed), then ev_run.py --pass2 (same pass-1 networks / calibration as the pass-1 run, fixed c,
# generic pass-2 detector + head + pass-2 sim-sigma calibration), then the tables: own DME v2 drift, selections m12 (paper's main rule),
# u45, sel; matched events vs published SIMFLUX; nanoruler metrics (80 nm fixed sites, nra_run.py; 40 / 20 nm site pairs, r24_rul.py),
# and pass 2 vs pass 1 on identical events (both tables in one r24_rul / nra_run call: same sites / pairs, paired differences).
# usage: bash analysis/nanoruler/nr_p2gen.sh <nr80|nr40|nr20> [gpu, default 1]
#   e.g. cd ~/simcode2; for m in nr80 nr40 nr20; do bash analysis/nanoruler/nr_p2gen.sh $m 1; done >> out/nanoruler/logs/p2gen.log 2>&1
# GPU estimate (pass 2 only; pass 1 reused): nr80 ~10 min, nr40 ~15 min, nr20 ~40 min on a shared RTX 3090 (det batch 8, < 6 GB).
source /home/jc/simcode2/runs/psfmatch/env.sh
MV=$1; GPU=${2:-1}; A=analysis/nanoruler; NR=$PWD/out/nanoruler; export R24_GPUS=$GPU
P2D=${P2D:-$PWD/models/p2det_gnr80.pt}; P2H=${P2H:-$PWD/models/p2headL_gnr80.pt}; P2C=${P2C:-$PWD/out/p2generic/models/calib/calib_p2L_gnr80.json}
for f in $P2D $P2H $P2C; do [ -f $f ] || { echo "$(date +%F_%T) missing $f: generic nr80 pass-2 set not ready (see out/p2generic/STATUS.md); not started"; exit 1; }; done
case $MV in
  nr80) SRC=$NR/runs/real_p1; MOV=$NR/raw/sim4_1_reord.tif; CAL=$NR/calib_nr80.json; P1CAL=$NR/models/calib/calib_sc_hd_nr80_real.json; NC=5000;;
  nr40) SRC=$NR/nr40/runs/real; MOV=/mnt/data/dmdsf/simcode2_out/nanoruler/raw_nr40/movie_reord.tif; CAL=$NR/calib_nr40.json; P1CAL=$NR/models/calib/calib_sc_hd_nr80.json; NC=6666;;
  nr20) SRC=$NR/nr20/runs/real; MOV=/mnt/data/dmdsf/simcode2_out/nanoruler/raw_nr20/movie_reord.tif; CAL=$NR/calib_nr20.json; P1CAL=$NR/models/calib/calib_sc_hd_nr80.json; NC=16666;;
  *) echo "usage: $0 <nr80|nr40|nr20> [gpu]"; exit 1;;
esac
OUT=${SRC}_p2gen; mkdir -p $OUT $NR/logs
for f in $SRC/pass1_*.npz $SRC/pass2_est_*.npz $SRC/pattern_est.pt $SRC/pattern_est.json; do [ -e $OUT/$(basename $f) ] || ln -s $f $OUT/$(basename $f); done
LOG=$NR/logs/p2gen_$MV.log
if [ ! -f $OUT/locs_p2.npz ]; then
  echo "$(date +%F_%T) $MV pass 2 (generic) -> $OUT"
  bash $A/r24_gpu.sh 6 python -u $A/ev_run.py --movie $MOV --calib $CAL --out $OUT --crop 5,5,390 --chunk 1500 --autoc 0 \
    --det $NR/train/det/final.pt --head $NR/train/hd/final.pt --cal $P1CAL --pass2 --p2_det $P2D --p2_head $P2H --p2_cal $P2C 2>&1 | grep -v "Warning\|^W1[0-9]\{3\} " > $LOG
  [ -f $OUT/locs_p2.npz ] || { echo "$(date +%F_%T) $MV pass 2 FAILED, see $LOG"; exit 1; }
fi
# tables (own DME v2 per selection, as the pass-1 tables) + matched events vs published SIMFLUX
if [ $MV = nr80 ]; then
  T=$NR/ablation/tabs_p2gen; PUB=$NR/analysis; TB=$NR/ablation/tabs; TC=$NR/ablation/tabs_cal; O=$NR/ablation/methods_p2gen; mkdir -p $T
  for s in m12 u45 sel; do [ -f $T/tab_p2_$s.npz ] || bash $A/r24_gpu.sh 3 python $A/abl_convert.py locs $OUT/locs_p2.npz $T/tab_p2_$s.npz --sel $s --sig cal; done
  [ -f $T/match_p2_pubsf_match.json ] || bash $A/r24_gpu.sh 3 python $A/abl_match.py $OUT/locs_p2.npz $T/tab_p2_u45_drift.npy $PUB/tab_pub_simflux.npz $T/match_p2_pubsf --also $PUB/tab_pub_smlm.npz
  [ -f $T/match_p2_pubsf_m12_pipe.npz ] || python $A/nr12_sub.py $OUT/locs_p2.npz $T/match_p2_pubsf
  python $A/nra_run.py run --sites $PUB/sites.json --out $O --bins 6 --nboot 500 \
    --methods pub_smlm=$PUB/tab_pub_smlm.npz pub_simflux=$PUB/tab_pub_simflux.npz bl_smlm=$TB/tab_bl_smlm_dme.npz bl_simflux=$TB/tab_bl_simflux_dme.npz \
              simcode_p1_m12=$TC/tab_p1_m12.npz simcode_p2_m12=$T/tab_p2_m12.npz simcode_p1_u45=$TC/tab_p1_u45.npz simcode_p2_u45=$T/tab_p2_u45.npz \
    --pairs simcode_p2_m12:simcode_p1_m12 simcode_p2_m12:pub_simflux simcode_p2_u45:simcode_p1_u45 simcode_p2_m12:pub_smlm > $NR/ablation/logs/nra_methods_p2gen.log 2>&1
  python $A/nra_run.py run --sites $PUB/sites.json --out $O --tag matched_m12 --bins 6 --nboot 500 \
    --methods simcode_p2=$T/match_p2_pubsf_m12_pipe.npz pub_simflux=$T/match_p2_pubsf_m12_ref.npz --pairs simcode_p2:pub_simflux > $NR/ablation/logs/nra_matched_m12_p2gen.log 2>&1
  python $A/nra_run.py run --sites $PUB/sites.json --out $O --tag matched_all --bins 6 --nboot 500 \
    --methods simcode_p2=$T/match_p2_pubsf_pipe.npz simcode_p1=$TC/match_p1_pubsf_pipe.npz pub_simflux=$T/match_p2_pubsf_ref.npz \
    --pairs simcode_p2:simcode_p1 simcode_p2:pub_simflux simcode_p1:pub_simflux > $NR/ablation/logs/nra_matched_all_p2gen.log 2>&1
else
  O=$NR/$MV; D0=${MV#nr}; REF=$O/ref; T=$O/methods_p2gen/tabs; T1=$O/methods_p1/tabs; M=$O/methods_p2gen; mkdir -p $T
  CMP="--cmp rcc=$REF/drift_rcc.npy pub_smlm=$REF/drift_pub_total_smlm.npy pub_simflux=$REF/drift_pub_total_simflux.npy"
  for s in m12 u45 sel; do [ -f $T/simcode_$s.npz ] || bash $A/r24_gpu.sh 3 python $A/r24_tab.py locs $OUT/locs_p2.npz $T/simcode_$s.npz --ncyc $NC --sel $s $CMP; done
  [ -f $T/match_pubsf_match.json ] || python $A/abl_match.py $OUT/locs_p2.npz $T/simcode_u45_drift.npy $REF/tab_pub_simflux.npz $T/match_pubsf --also $REF/tab_pub_smlm.npz
  [ -f $T/match_pubsf_m12_pipe.npz ] || python $A/nr12_sub.py $OUT/locs_p2.npz $T/match_pubsf
  RL="auto_sf auto_sc"; [ $MV = nr20 ] && RL="picks_g2d auto_sf auto_sc"; export R24_JOBS=8
  for r in $RL; do X="--pairs $O/pairs_$r.json --d0 $D0 --out $M --levels"
    python $A/r24_rul.py run $X --tag m12_$r --methods pub_smlm=$REF/tab_pub_smlm.npz pub_simflux=$REF/tab_pub_simflux.npz bl_simflux=$T1/bl_simflux.npz \
      simcode_p1_m12=$T1/simcode_m12.npz simcode_p2_m12=$T/simcode_m12.npz simcode_p1_u45=$T1/simcode_u45.npz simcode_p2_u45=$T/simcode_u45.npz \
      --cmp simcode_p2_m12:simcode_p1_m12 simcode_p2_m12:pub_simflux simcode_p2_u45:simcode_p1_u45 --frc 1 > $O/logs/rul_p2gen_m12_$r.log 2>&1
    python $A/r24_rul.py run $X --tag ${r}_matched_all --methods simcode_p2=$T/match_pubsf_pipe.npz simcode_p1=$T1/match_pubsf_pipe.npz pub_simflux=$T/match_pubsf_ref.npz \
      --cmp simcode_p2:simcode_p1 simcode_p2:pub_simflux simcode_p1:pub_simflux > $O/logs/rul_p2gen_${r}_matched_all.log 2>&1
  done
fi
echo "$(date +%F_%T) nr_p2gen $MV done"
