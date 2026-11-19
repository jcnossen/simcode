#!/bin/bash
# DECODE v4 (2026-11-18): dec_m1809v4 / dec_m25v4 = the v3 protocol (analysis/baselines/decode/train_v3.py, published DECODE
# schedule, 512-step lr warm-up, from scratch, out/decode_checks/v3/run_m1809.sh / run_m25.sh) with the in-situ PSF v3 (train_v3
# presets m1809v4 / m25v4, branch decv4 dc6ff554), + the progress evaluation (eval_watch.py: iso movie rendered with the same v3
# PSF, out/decode_checks/v4/iso_<m>; m1809 also the density movie D2_v2psf as v3; the real-data chunk), + the freeze: the epoch-300
# snapshot -> model_e0300_converged.pt, then the training is stopped (v3: both models frozen at epoch 300, metrics flat).
# usage: setsid nohup bash run_decode.sh <m1809|m25> <gpu> > logs/decode_<m>.log 2>&1 &
M=$1; G=$2; T=/home/jc/simcode2/out/train_v3
source $T/env.sh $G
cd /home/jc/simcode2-decv4
L=$T/logs; N=dec_${M}v4; MD=/home/jc/simcode2/out/paper2/decode/lutpsf/$N; ISO=/home/jc/simcode2/out/decode_checks/v4/iso_$M
case $M in m1809) SNAP=4; REAL=1809_2; DENS=""; POLL=300 ;; m25) SNAP=5; REAL=2.5nM_2; DENS="--dens ''"; POLL=600 ;; esac
echo "$(date) $N GPU $G"
python -u $T/with_pkg.py analysis/baselines/decode/train_v3.py --preset ${M}v4 --name $N --snap_every $SNAP --warmup 512 >> $L/train_$N.log 2>&1 &
TP=$!
until [ -f $ISO/movie.tif ] && [ -f $ISO/gt.npz ]; do sleep 60; done
sleep 60
if [ $M = m25 ]; then
  python -u $T/with_pkg.py analysis/decode_checks/v3/eval_watch.py --names $N --iso $ISO --dens '' --real $REAL --tag $N --poll $POLL >> $L/eval_$N.log 2>&1 &
else
  python -u $T/with_pkg.py analysis/decode_checks/v3/eval_watch.py --names $N --iso $ISO --real $REAL --tag $N --poll $POLL >> $L/eval_$N.log 2>&1 &
fi
WP=$!
until [ -f $MD/snapshots/model_e0300.pt ]; do kill -0 $TP 2>/dev/null || { echo "$(date) trainer ended before epoch 300"; break; }; sleep 120; done
if [ -f $MD/snapshots/model_e0300.pt ]; then
  sleep 30; cp -p $MD/snapshots/model_e0300.pt $MD/model_e0300_converged.pt
  echo "$(date) froze epoch 300 -> $MD/model_e0300_converged.pt; stopping the trainer"
  kill $TP
fi
# let the watcher evaluate the last snapshots, then stop it
sleep $((3 * POLL)); kill $WP 2>/dev/null
echo "$(date) $N done"
