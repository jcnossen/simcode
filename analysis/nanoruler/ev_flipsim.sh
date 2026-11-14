#!/bin/bash
# nr80 evaluation, step-order flip, sim validation: the simulated nanoruler movies nr_a / nr_c0 rewritten with the steps of each
# orientation reversed (ev_flipsteps.py; big files in /mnt/data/dmdsf/simcode2_out/nanoruler/eval_flip, symlink
# out/nanoruler/eval/sim_flip), both passes with the nr80 set through ev_run.py exactly as ev_chain.sh step 4 (own pattern estimate,
# model nr80 = out/nanoruler/models/netset_nr80.json with its sim calibrations, no sigma rescale), the standard nr_c0 run for comparison, then the tables. One GPU job at a time.
# usage: setsid nohup bash analysis/nanoruler/ev_flipsim.sh >> out/nanoruler/eval/flipsim.log 2>&1 < /dev/null &
source /home/jc/simcode2/runs/psfmatch/env.sh; set -o pipefail
A=analysis/nanoruler; EV=out/nanoruler/eval; MD=$PWD/out/nanoruler/models; MC=$MD/calib; BIG=/mnt/data/dmdsf/simcode2_out/nanoruler/eval_flip
say() { echo "$(date +%F_%T) $*"; }; die() { say "FAILED: $*"; exit 1; }
mkdir -p $BIG; [ -e $EV/sim_flip ] || ln -s $BIG $EV/sim_flip
run() {  # movie dir, run dir
  [ -f $2/locs_p2.npz ] && return 0
  local g=$(NEED_GB=4 pickgpu 4 "2 0 1"); say "run $2 on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python -u $A/ev_run.py --movie $1/sim_movie.tif --calib $PWD/$1/calib_sim.json --out $2 --crop 5,5,246 \
    --model $MD/netset_nr80.json --cal_kind sim --sig_u '' \
    > $2.log 2>&1 || die "run $2 (see $2.log)"
}
for n in nr_a nr_c0; do
  S0=out/nanoruler/sim/$n; SF=$EV/sim_flip/$n; mkdir -p $SF
  [ -f $SF/sim_movie.tif ] || python $A/ev_flipsteps.py movie $S0/sim_movie.tif $SF/sim_movie.tif > $SF/flip.log 2>&1 || die "flip $n"
  [ -f $SF/sim_gt.npz ] || python $A/ev_flipsteps.py gt $S0/sim_gt.npz $SF/sim_gt.npz >> $SF/flip.log 2>&1 || die "flip gt $n"
  cp -n $S0/calib_sim.json $S0/sim_info.json $SF/
  run $SF $EV/sim_flip/run_$n
  python $A/ev_p12sim.py $EV/sim_flip/run_$n $SF nr > $EV/p12_flip_$n.md.tmp && mv $EV/p12_flip_$n.md.tmp $EV/p12_flip_$n.md || die "p12 flip $n"
done
mkdir -p $EV/sim_nr_c0; run out/nanoruler/sim/nr_c0 $EV/sim_nr_c0/run_nr80
python $A/ev_p12sim.py $EV/sim_nr_c0/run_nr80 out/nanoruler/sim/nr_c0 nr > $EV/p12_std_nr_c0.md || die "p12 std c0"
python $A/ev_flipclass.py --cross rev,flip fwd=$EV/sim_order/run_fwd:$EV/sim_order/fwd rev=$EV/sim_order/run_rev:$EV/sim_order/rev flip=$EV/sim_flip/run_nr_a:$EV/sim_flip/nr_a \
  > $EV/flipclass_nr_a.md.tmp && mv $EV/flipclass_nr_a.md.tmp $EV/flipclass_nr_a.md || die flipclass
python $A/ev_flipclass.py --cross rev,flip rev=$EV/sim_nr_c0/run_nr80:out/nanoruler/sim/nr_c0 flip=$EV/sim_flip/run_nr_c0:$EV/sim_flip/nr_c0 > $EV/flipclass_nr_c0.md || die "flipclass c0"
# sigma: pass 2 with the existing rescale (fitted on the mismatched movie), refit on the flipped nr_a first half, test on the second half; pass 1 with the existing rescale
RD=$EV/sim_flip/run_nr_a; SF=$EV/sim_flip/nr_a
python $A/ev_sigcal.py test none $RD:$SF:est2:6000:12000 > $EV/sigu_test_p2_nr80_flip_before.txt || die sig
python $A/ev_sigcal.py test $MC/sigu_p2_nr80.json $RD:$SF:est2:6000:12000 > $EV/sigu_test_p2_nr80_flip_oldsigu.txt || die sig
python $A/ev_sigcal.py fit $MC/sigu_p2_nr80_flip.json $RD:$SF:est2:0:6000 > $EV/sigu_fit_p2_nr80_flip.txt || die sigfit
python $A/ev_sigcal.py test $MC/sigu_p2_nr80_flip.json $RD:$SF:est2:6000:12000 > $EV/sigu_test_p2_nr80_flip.txt || die sig
python $A/ev_sigcal.py test $MC/sigu_p1_nr80.json $RD:$SF:est:6000:12000 > $EV/sigu_test_p1_nr80_flip.txt || die sig
python $A/ev_sigcal.py test $MC/sigu_p2_nr80_flip.json $EV/sim_flip/run_nr_c0:$EV/sim_flip/nr_c0:est2:0:12000 > $EV/sigu_test_p2_nr80_flip_c0.txt || die sig
say "flipsim done"
