#!/bin/bash
# Generic pass-2 set (2026-11-13): the PSF-matched pass-2 recipe of runs/psfmatch/chain2.sh (p2det 40k, p2head 24k, p2headL 50k;
# same lr / seeds / init) with the generic pattern draw (share P_STD of SIMCODE patterns, default 0.25). Resumable (final.pt skips).
# usage: setsid nohup bash analysis/p2generic/chain_gen.sh <set g1809> "<gpu pref>" >> runs/p2generic/chain_<set>.log 2>&1 &
S=$1; PREF=${2:-"0 2"}
source /home/jc/simcode2/runs/psfmatch/env.sh
R=runs/p2generic                                   # stage() writes $R/$S/<stage>, logs $R/stages.log, lock $R/.gpulock
declare -A BASE=([g1809]=m1809) PSFG=([g1809]=${PSF[m1809]})
B=${BASE[$S]}; C=${PSFG[$S]}; [ -n "$B" ] || { echo "unknown set $S"; exit 1; }
PS=${P_STD:-0.25}   # share of SIMCODE-pattern FOVs (pgen_v1 = 0.25, pgen_v1s50 = 0.5; 2026-11-14 --pgen -> --p_std)
D=$R/$S; mkdir -p $D
W="python -u analysis/p2generic/pwrap.py"
echo "$(date) chain $S base $B calib $C p_std $PS pref $PREF"
( stage $S p2det 8 "$PREF" env P2G_TITLE=p2gen-det-$S $W simcode.detector.train --out $D/p2det --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11 --eval_every 2500 --ckpt_every 5000 --illum 1 --init models/det_$B.pt --calib $C --p_std $PS ) &
PD=$!
until [ -f $D/p2det/ckpt_25000.pt ] || [ -f $D/p2det/final.pt ]; do kill -0 $PD 2>/dev/null || [ -f $D/p2det/final.pt ] || { echo "p2det died"; exit 1; }; sleep 30; done
P25=$D/p2det/ckpt_25000.pt; [ -f $P25 ] || P25=$D/p2det/final.pt
stage $S p2head 10 "${PREF2:-2 0}" env P2G_TITLE=p2gen-hd-$S $W simcode.emitter.train --out $D/p2head --det $P25 --init models/hd_$B.pt --steps 24000 --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0 --pert_model v1 --calib $C --p_std $PS || exit 1
wait $PD || { echo "p2det failed"; exit 1; }
stage $S p2headL 10 "$PREF" env P2G_TITLE=p2gen-hL-$S $W simcode.emitter.train --out $D/p2headL --det $D/p2det/final.pt --init $D/p2head/final.pt --steps 50000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 --pert_model real --hold_v1 15 --calib $C --p_std $PS || exit 1
for n in p2det p2headL; do cp -n $D/$n/final.pt models/${n}_$S.pt; done
echo "$(date) chain $S done"
