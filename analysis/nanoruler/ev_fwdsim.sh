#!/bin/bash
# nr80 evaluation, positive control of the step-order hypothesis: a simulated nanoruler movie identical to sim/nr_a (same
# sim_nr.py defaults and seed) but with FORWARD (increasing) phase steps (out/nanoruler/eval/calib_nr80_fwd.json), and a fresh copy
# of the standard movie (decreasing steps, as the real setup) generated the same way; both through ev_run.py with the nr80 set
# (model nr80 with its sim calibrations, no sigma rescale), then pass 2 vs pass 1 on identical GT windows (ev_p12sim.py). One GPU job at a time.
# usage: setsid nohup bash analysis/nanoruler/ev_fwdsim.sh >> out/nanoruler/eval/fwdsim.log 2>&1 < /dev/null &
source /home/jc/simcode2/runs/psfmatch/env.sh; set -o pipefail
A=analysis/nanoruler; EV=out/nanoruler/eval; MD=$PWD/out/nanoruler/models; MC=$MD/calib; T=$EV/sim_order
say() { echo "$(date +%F_%T) $*"; }
until ! pgrep -f "^bash $A/ev_revtest.sh" > /dev/null; do sleep 20; done
mkdir -p $T
for v in fwd rev; do
  C=$PWD/out/nanoruler/calib_nr80.json; [ $v = fwd ] && C=$PWD/$EV/calib_nr80_fwd.json
  SD=out/nanoruler/sim/ev_order/$v; RD=$T/run_$v   # sim_nr.py only writes under out/nanoruler/sim/: ev_order is a symlink to $T (the data are in out/nanoruler/eval/sim_order)
  if [ ! -f $SD/sim_info.json ]; then
    g=$(NEED_GB=4 pickgpu 4 "2 0 1"); say "sim $v on GPU $g"
    CUDA_VISIBLE_DEVICES=$g python -u $A/sim_nr.py $SD 12000 1 --calib $C > $T/sim_$v.log 2>&1 || { say "FAILED sim $v"; exit 1; }
  fi
  if [ ! -f $RD/locs_p2.npz ]; then
    g=$(NEED_GB=4 pickgpu 4 "2 0 1"); say "run $v on GPU $g"
    CUDA_VISIBLE_DEVICES=$g python -u $A/ev_run.py --movie $SD/sim_movie.tif --calib $PWD/$SD/calib_sim.json --out $RD --crop 5,5,246 \
      --model $MD/netset_nr80.json --cal_kind sim --sig_u '' \
      > $T/run_$v.log 2>&1 || { say "FAILED run $v"; exit 1; }
  fi
  python $A/ev_p12sim.py $RD $SD nr > $T/p12_$v.md.tmp && mv $T/p12_$v.md.tmp $T/p12_$v.md || { say "FAILED p12 $v"; exit 1; }
done
say "fwdsim done"
