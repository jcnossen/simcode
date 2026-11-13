#!/bin/bash
# PSF-matched network set "nr80": runs/psfmatch/chain2.sh (m1809 recipe + the g13 range extensions --tau 3,80, --dens 0.0005,0.06)
# with every trainer run through analysis/nanoruler/nrwrap.py (RealSim presets patched: pattern pitch, step spacing, camera
# map). Outputs out/nanoruler/train/<stage>/, final networks copied to out/nanoruler/models/. Resumable per stage (final.pt).
# usage: NEED_GB=6 setsid nohup bash analysis/nanoruler/chain_nr80.sh "2 1 0" >> out/nanoruler/train/chain.log 2>&1 &
PREF=${1:-"2 1 0"}
source /home/jc/simcode2/runs/psfmatch/env.sh
R=out/nanoruler/train; S=.; D=$R; C=$PWD/out/nanoruler/calib_nr80.json; W="python -u analysis/nanoruler/nrwrap.py"
XT="--tau 3,80 --dens 0.0005,0.06"
mkdir -p $D out/nanoruler/models
echo "$(date) chain nr80 calib $C pref $PREF"
stage $S det 10 "$PREF" $W simcode2.det.up2 train --up 1 --out $D/det --steps 40000 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 --nnw 1 --em_photons 40,2500 --seed 0 --calib $C --dens 0.0005,0.06 || exit 1
( stage $S hd 10 "$PREF" $W simcode2.hd.sc4 train --arch sc4n --out $D/hd --steps 18000 --batch 16 --lr 1.5e-4 --ch 96 --det $D/det/final.pt --paper 1 --tau 15,80 --hold out/sc4/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init hd_sc5nb.pt --pct 0.1 --bright_tail 0.1,2000,20000 --ckpt_every 6000 --calib $C $XT ) &
PH=$!
( stage $S p2det 10 "$PREF" $W simcode2.p2.det2 train --out $D/p2det --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11 --eval_every 2500 --ckpt_every 5000 --init_n1 $D/det/final.pt --calib $C $XT ) &
PD=$!
wait $PH || { echo "hd failed"; exit 1; }
until [ -f $D/p2det/ckpt_25000.pt ] || [ -f $D/p2det/final.pt ]; do sleep 30; done
P25=$D/p2det/ckpt_25000.pt; [ -f $P25 ] || P25=$D/p2det/final.pt
stage $S p2head 10 "$PREF" $W simcode2.p2.head2 train --out $D/p2head --det $P25 --init $D/hd/final.pt --steps 24000 --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0 --calib $C $XT || exit 1
wait $PD || { echo "p2det failed"; exit 1; }
stage $S p2headL 10 "$PREF" $W simcode2.p2.head2 train --out $D/p2headL --det $D/p2det/final.pt --init $D/p2head/final.pt --steps 50000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 --pert_model real --hold_v1 15 --calib $C $XT || exit 1
for n in det hd p2det p2headL; do cp -n $D/$n/final.pt out/nanoruler/models/${n}_nr80.pt; done
echo "$(date) chain nr80 done"
