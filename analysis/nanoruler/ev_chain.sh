#!/bin/bash
# nr80 evaluation / calibration chain (modelled on runs/psfmatch/eval_g13.sh). Resumable: every step is skipped when its output
# exists; a failed step prints "FAILED" to the chain log and stops the chain (no later step runs on a missing input).
# Waits by itself for the training chain (out/nanoruler/train) and the simulated movie (out/nanoruler/sim/nr_a).
# usage: cd ~/simcode2; setsid nohup bash analysis/nanoruler/ev_chain.sh >> out/nanoruler/eval/chain.log 2>&1 < /dev/null &
source /home/jc/simcode2/runs/psfmatch/env.sh
set -o pipefail
A=analysis/nanoruler; EV=out/nanoruler/eval; TR=$PWD/out/nanoruler/train; MD=$PWD/out/nanoruler/models; MC=$MD/calib
C=$PWD/out/nanoruler/calib_nr80.json; S=nr80; SC=iso,pair1,pair2,rs05qm; FIT=${EV_FIT:-rs05qm}; SEEDS=202,301,302,303
G=$EV/gate_$S; FE=$EV/fe_$S; CB=$EV/calibs; LG=$EV/logs; SD=out/nanoruler/sim/nr_a; SR=$EV/sim_nr_a; HALF=6000; NF=12000; CROP=5,5,246
B1=calib/calib_sc_hd_sc5nbb15_pm_bright.json; B1R=calib/calib_sc_hd_sc5nbb15_pm_bright_real.json; B2=calib/calib_p2L.json; B2R=calib/calib_p2L_b10_real.json
W="python -u $A/nrwrap.py"; export NR_TITLE=nr-eval
mkdir -p $G $FE $CB $LG $SR $MC
say() { echo "$(date +%F_%T) $*"; }
die() { say "FAILED: $*"; exit 1; }
gpu() {  # need_gb: the env.sh pickgpu rule (free >= need + RESERVE_GB, preference 2 1 0), with a log line (stderr -> chain log) every 30 min of waiting
  local need=${1:-4} n=0 g f
  while true; do
    for g in ${PREF:-2 1 0}; do
      f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
      if [ "$f" -ge $(( (need + RESERVE_GB) * 1024 )) ]; then echo $g; return 0; fi
    done
    [ $((n % 30)) = 0 ] && say "waiting for a GPU with $((need + RESERVE_GB)) GB free" >&2
    n=$((n + 1)); sleep 60
  done
}
waitfor() {  # file [grep pattern]: wait (a line every 30 min) until the file exists (and contains the pattern)
  local n=0
  until [ -f "$1" ] && { [ -z "$2" ] || grep -q "$2" "$1"; }; do [ $((n % 30)) = 0 ] && say "waiting for $1 $2"; n=$((n + 1)); sleep 60; done
}
dump() {  # fronts (name=det:head,...): p2.dump on the gate scenes; every dump must exist and have found its CRLB
  local names=$(echo $1 | sed 's/=[^,]*//g; s/,/ /g') g f sc sd need=0
  for f in $names; do for sc in ${SC//,/ }; do for sd in ${SEEDS//,/ }; do [ -f $FE/${sc}__s${sd}_$f.npz ] || need=1; done; done; done
  [ $need = 0 ] && { say "dump $names: exists, skip"; return 0; }
  g=$(gpu 4); say "dump $names on GPU $g"
  CUDA_VISIBLE_DEVICES=$g SIMCODE2_GATE_CALIB=$C SIMCODE2_GATE_DIR=$G GATE_PSF_CHUNK=1500 $W simcode2.p2.dump --scen $SC --seeds $SEEDS --fronts $1 --out $FE \
    >> $LG/dump_${names// /_}.log 2>&1 || die "dump $names (see $LG/dump_${names// /_}.log)"
  grep -q "CRLB from None" $LG/dump_${names// /_}.log && die "dump $names: CRLB reference not found (GT differs from the gate scenes)"
  for f in $names; do for sc in ${SC//,/ }; do for sd in ${SEEDS//,/ }; do [ -f $FE/${sc}__s${sd}_$f.npz ] || die "dump $f $sc $sd missing"; done; done; done
}
fitcal() {  # front base out_sim out_real real-src
  [ -f $3 ] && [ -f $4 ] && { say "calib $3: exists, skip"; return 0; }
  CUDA_VISIBLE_DEVICES=$(gpu 2) python $A/ev_fitcal.py $1 $FE $FIT $2 $3 $4 $5 >> $LG/fitcal.log 2>&1 || die "fitcal $1 (see $LG/fitcal.log)"
  say "calib $1 -> $3"
}
accept() {
  CUDA_VISIBLE_DEVICES=$(gpu 2) python $A/ev_accept.py > $LG/accept.log 2>&1 || die "accept (see $LG/accept.log)"
  [ -n "$1" ] && cp $EV/accept_$S.txt $EV/accept_${S}_$1.txt; say "accept table -> $EV/accept_$S.txt $1"
}
simrun() {  # name det head cal p2det p2head p2cal [gt]; gt: the run gets the TRUE pattern of the simulation instead of estimating it
  local RD=$SR/run_$1 id=$(md5sum < $SD/sim_gt.npz)
  # a run made on an earlier version of the simulated movie (the sim worker regenerates it) is moved aside and redone
  if [ -d $RD ] && [ "$(cat $RD/sim_id.txt 2>/dev/null)" != "$id" ]; then say "sim run $1: movie changed, old run -> $RD.stale"; rm -rf $RD.stale; mv $RD $RD.stale; fi
  [ -f $RD/locs_p2.npz ] && { say "sim run $1: exists, skip"; return 0; }
  mkdir -p $RD; echo "$id" > $RD/sim_id.txt
  if [ "$8" = gt ] && [ ! -f $RD/pattern_est.pt ]; then
    mkdir -p $RD; cp $SD/gt_pattern.pt $RD/pattern_est.pt
    echo "pattern_est.pt of this run is a COPY of $SD/gt_pattern.pt (the true pattern of the simulation), not an estimate (ev_chain.sh simrun ... gt)" > $RD/PATTERN_IS_GT.txt
    say "sim run $1: pattern_est.pt = the true pattern of the simulation (not estimated)"
  fi
  local g=$(gpu 4); say "sim run $1 on GPU $g"
  CUDA_VISIBLE_DEVICES=$g python -u $A/ev_run.py --movie $SD/sim_movie.tif --calib $PWD/$SD/calib_sim.json --out $RD --crop $CROP --pass2 --autoc 0 \
    --det $2 --head $3 --cal $4 --p2_det $5 --p2_head $6 --p2_cal $7 > $SR/run_$1.log 2>&1 || die "sim run $1 (see $SR/run_$1.log)"
  [ -f $RD/locs_p2.npz ] || die "sim run $1: no locs_p2.npz"
  [ "$(md5sum < $SD/sim_gt.npz)" = "$id" ] || die "sim run $1: $SD/sim_gt.npz changed during the run"
}
out() {  # file cmd...: stdout to file only on success
  local f=$1; shift
  "$@" > $f.tmp 2>> $LG/sigcal.err || { rm -f $f.tmp; die "$* (see $LG/sigcal.err)"; }
  mv $f.tmp $f
}
say "ev_chain $S start (pid $$)"

# 1. gate scenes (CRLB + the current production front as the gate's reference), reference dumps, their calibs on these scenes
if [ ! -f $G/rs05qm__s303_hd_cur.npz ]; then
  g=$(gpu 4); say "gate scenes on GPU $g"
  CUDA_VISIBLE_DEVICES=$g SIMCODE2_GATE_CALIB=$C GATE_PSF_CHUNK=1500 $W simcode2.hd.gate --scen $SC --fronts sc --sc hd_sc5nb.pt --sccal $B1 \
    --tag _hd_cur --out $G --ntarget 3500 --seeds $SEEDS >> $LG/gate.log 2>&1 || die "gate (see $LG/gate.log)"
fi
[ -f $EV/scenecheck_$S.txt ] || { CUDA_VISIBLE_DEVICES=$(gpu 4) SIMCODE2_GATE_CALIB=$C $W $A/ev_scenecheck.py > $EV/scenecheck_$S.txt.tmp 2>> $LG/scenecheck.err \
  && mv $EV/scenecheck_$S.txt.tmp $EV/scenecheck_$S.txt || die "scenecheck"; }
dump c1=det_n1.pt:hd_sc5nb.pt,c2=p2_det.pt:p2_head_long.pt,r1=det_m1809.pt:hd_m1809.pt,r2=p2det_m1809.pt:p2headL_m1809.pt
for f in c1 r1; do fitcal $f $B1 $CB/calib_${f}_$S.json $CB/calib_${f}_${S}_real.json $B1R; done
for f in c2 r2; do fitcal $f $B2 $CB/calib_${f}_$S.json $CB/calib_${f}_${S}_real.json $B2R; done
[ -f $EV/accept_${S}_ref.txt ] || accept ref
say "step 1 done"

# 4a. reference run of the current production nets on the simulated nanoruler movie (needs no new nets)
waitfor $SD/sim_gt.npz; waitfor $SD/sim_info.json; [ -f $SD/calib_sim.json ] || die "no $SD/calib_sim.json (calib with the simulated camera maps)"
# The production nets cannot estimate the pattern on this movie (2026-11-13: the mismatched detector splits every emitter into ~6
# candidates, 14 'clean' rows of 151739, estimate_pattern fails), so the reference run is given the TRUE pattern (this favours the reference).
simrun cur det_n1.pt hd_sc5nb.pt $PWD/$CB/calib_c1_$S.json p2_det.pt p2_head_long.pt $PWD/$CB/calib_c2_$S.json gt
for p in 1 2; do tg=est; [ $p = 2 ] && tg=est2
  [ -f $EV/sigu_test_p${p}_cur_before.txt ] || out $EV/sigu_test_p${p}_cur_before.txt python $A/ev_sigcal.py test none $SR/run_cur:$SD:$tg:$HALF:$NF
done
say "reference sim run done"

# 2. pass 1 after the hd stage
waitfor $TR/hd/final.pt; waitfor $TR/stages.log "hd end rc=0"
dump s1=$TR/det/final.pt:$TR/hd/final.pt
fitcal s1 $B1 $MC/calib_sc_hd_$S.json $MC/calib_sc_hd_${S}_real.json $B1R
accept pass1; say "step 2 (pass-1 accept) done"

# 3. pass 2 after the training chain
waitfor $TR/chain.log "chain $S done"
for n in det hd p2det p2headL; do [ -f $MD/${n}_$S.pt ] || die "$MD/${n}_$S.pt missing"; cmp -s $TR/$n/final.pt $MD/${n}_$S.pt || die "$MD/${n}_$S.pt differs from $TR/$n/final.pt"; done
dump s2=$MD/p2det_$S.pt:$MD/p2headL_$S.pt
fitcal s2 $B2 $MC/calib_p2L_$S.json $MC/calib_p2L_${S}_real.json $B2R
accept; say "step 3 (accept) done"

# 4. simulated nanoruler movie with the nr80 set, sigma rescale fit on [0, HALF) / test on [HALF, NF), GT comparison vs the reference run
# the nr80 run estimates its own pattern (EV_NR80_GTPAT=gt: fallback for a manual relaunch, true pattern; not the default)
simrun cur det_n1.pt hd_sc5nb.pt $PWD/$CB/calib_c1_$S.json p2_det.pt p2_head_long.pt $PWD/$CB/calib_c2_$S.json gt   # redone only if the movie changed
simrun $S $MD/det_$S.pt $MD/hd_$S.pt $MC/calib_sc_hd_$S.json $MD/p2det_$S.pt $MD/p2headL_$S.pt $MC/calib_p2L_$S.json ${EV_NR80_GTPAT:-}
RD=$SR/run_$S
for p in 1 2; do tg=est; [ $p = 2 ] && tg=est2
  [ -f $MC/sigu_p${p}_$S.json ] && [ $MC/sigu_p${p}_$S.json -nt $RD/locs_p2.npz ] || python $A/ev_sigcal.py fit $MC/sigu_p${p}_$S.json $RD:$SD:$tg:0:$HALF >> $LG/sigcal_fit.log 2>> $LG/sigcal.err || die "sigcal fit p$p (see $LG/sigcal.err)"
  out $EV/sigu_test_p${p}_$S.txt python $A/ev_sigcal.py test $MC/sigu_p${p}_$S.json $RD:$SD:$tg:$HALF:$NF
  out $EV/sigu_test_p${p}_${S}_before.txt python $A/ev_sigcal.py test none $RD:$SD:$tg:$HALF:$NF
  out $EV/simcmp_nr_a_p${p}_${S}_vs_cur.txt python $A/ev_sigcal.py cmp $SD $tg $RD $SR/run_cur
done
say "step 4 done"

# 5. gates + network-set entry
python $A/ev_gates.py > $EV/gates_$S.txt.tmp 2>> $LG/gates.err; rc=$?; mv $EV/gates_$S.txt.tmp $EV/gates_$S.txt
tail -1 $EV/gates_$S.txt
[ $rc = 0 ] || say "GATES FAIL or error (see $EV/gates_$S.txt, $LG/gates.err)"
say "eval $S done"
