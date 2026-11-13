#!/bin/bash
# Generic pass-2 set (2026-11-13): the PSF-matched pass-2 recipe of runs/psfmatch/chain2.sh (p2det 40k, p2head 24k, p2headL 50k;
# same lr / seeds / init) with the generic pattern draw --pgen (out/p2generic/presets/pgen_v1.json). Resumable (final.pt skips).
# env PGEN (pgen preset name, default pgen_v1), PM (p2headL pert_model, default real; realw = per-FOV error-sd ranges)
# chain_gen2.sh = chain_gen.sh + set gnr80 (separate file: chain_gen.sh was running when this was added)
# usage: setsid nohup bash analysis/p2generic/chain_gen2.sh <set g1809|gnr80> "<gpu pref>" >> runs/p2generic/chain_<set>.log 2>&1 &
S=$1; PREF=${2:-"0 2"}
source /home/jc/simcode2/runs/psfmatch/env.sh
R=runs/p2generic                                   # stage() writes $R/$S/<stage>, logs $R/stages.log, lock $R/.gpulock
declare -A BASE=([g1809]=m1809 [gnr80]=nr80) PSFG=([g1809]=${PSF[m1809]} [gnr80]=$PWD/out/nanoruler/calib_nr80.json)
B=${BASE[$S]}; C=${PSFG[$S]}; [ -n "$B" ] || { echo "unknown set $S"; exit 1; }
# init networks of the base set; gnr80: the nr80 pass-1 networks, trainers through nrwrap (nanoruler RealSim presets: camera map,
# emitters at the surface, bg) + the nr80 range extensions (as analysis/nanoruler/chain_nr80.sh)
MD=models; XT=""; NRW=""
if [ $S = gnr80 ]; then MD=$PWD/out/nanoruler/models; XT="--tau 3,80 --dens 0.0005,0.06"; NRW="analysis/nanoruler/nrwrap.py"; fi
PG=$PWD/out/p2generic/presets/${PGEN:-pgen_v1}.json
D=$R/$S; mkdir -p $D
W="python -u analysis/p2generic/pwrap.py"
echo "$(date) chain $S base $B calib $C pgen $PG pref $PREF"
( stage $S p2det 8 "$PREF" env P2G_TITLE=p2gen-det-$S $W $NRW simcode.p2.det2 train --out $D/p2det --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11 --eval_every 2500 --ckpt_every 5000 --init_n1 $MD/det_$B.pt --calib $C --pgen $PG $XT ) &
PD=$!
until [ -f $D/p2det/ckpt_25000.pt ] || [ -f $D/p2det/final.pt ]; do kill -0 $PD 2>/dev/null || [ -f $D/p2det/final.pt ] || { echo "p2det died"; exit 1; }; sleep 30; done
P25=$D/p2det/ckpt_25000.pt; [ -f $P25 ] || P25=$D/p2det/final.pt
stage $S p2head 10 "${PREF2:-2 0}" env P2G_TITLE=p2gen-hd-$S $W $NRW simcode.p2.head2 train --out $D/p2head --det $P25 --init $MD/hd_$B.pt --steps 24000 --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0 --calib $C --pgen $PG $XT || exit 1
wait $PD || { echo "p2det failed"; exit 1; }
stage $S p2headL 10 "$PREF" env P2G_TITLE=p2gen-hL-$S $W $NRW simcode.p2.head2 train --out $D/p2headL --det $D/p2det/final.pt --init $D/p2head/final.pt --steps 50000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 --pert_model ${PM:-real} --hold_v1 15 --calib $C --pgen $PG $XT || exit 1
for n in p2det p2headL; do cp -n $D/$n/final.pt models/${n}_$S.pt; done
echo "$(date) chain $S done"
