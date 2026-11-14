#!/bin/bash
# Generic pass 2 (2026-11-13): evaluation of the generic nanoruler-PSF set gnr80 (runs/p2generic/gnr80) against nr80.
#  g  gate scenes of the nr80 evaluation (out/nanoruler/eval/gate_nr80, forward steps, seeds 202 + 301-303) and the reversed-step
#     scenes (gate_nr80rev, = the real step direction): dump g2 (nrwrap presets), sim calibration fit on seed 202 rs05qm (as nr80),
#     head-to-head g2 vs s2 (nr80 pass 2) / s1 (nr80 pass 1) (nr80 dumps linked in)
#  s  simulated nanoruler movies nr_a / nr_c0 (standard = reversed order), nr80 pass 1 + generic pass 2 (analysis/nanoruler/ev_run.py,
#     as the nr80 evaluation); pass 2 vs pass 1 on identical windows, fully-on and partially-on separately (ev_flipclass.py), next to
#     the existing nr80 runs (std)
# usage: setsid nohup bash analysis/p2generic/eval_nr.sh [g s] >> runs/p2generic/eval_gnr80.log 2>&1 < /dev/null &
source /home/jc/simcode2/runs/psfmatch/env.sh; set -o pipefail
G=gnr80; P=$PWD/out/p2generic; A=analysis/p2generic; EV=out/nanoruler/eval; MD=$PWD/out/nanoruler/models; MC=$MD/calib
C=$PWD/out/nanoruler/calib_nr80.json; GM=$PWD/runs/p2generic/$G; MY=$P/models; mkdir -p $MY/calib
W="python -u $A/pwrap.py $PWD/analysis/nanoruler/nrwrap.py"
PARTS=${@:-g s}
gpu() { NEED_GB=4 pickgpu 4 "${PREF:-0 2}"; }
say() { echo "$(date +%F_%T) $*"; }
CALS="s1=$MC/calib_sc_hd_nr80.json,s2=$MC/calib_p2L_nr80.json,g2=$MY/calib/calib_p2L_$G.json"
export SIMCODE_GATE_CALIB=$C GATE_PSF_CHUNK=1500
for part in $PARTS; do
case $part in
g)  for v in fwd rev; do
      if [ $v = fwd ]; then GD=$EV/gate_nr80; SRC=$EV/fe_nr80; PRE=$PWD/out/nanoruler/nr80_preset.json; SEEDS=202,301,302,303; SCN=iso,pair1,pair2,rs05qm
      else GD=$EV/gate_nr80rev; SRC=$EV/fe_nr80rev; PRE=$PWD/$EV/nr80_preset_rev.json; SEEDS=301,302,303; SCN=iso,rs05qm; fi
      FE=$P/fe_nr80_$v; mkdir -p $FE
      for f in $SRC/*__s*_s[12].npz; do ln -sfn $PWD/$f $FE/$(basename $f); done
      NR_PRESET=$PRE SIMCODE_GATE_DIR=$GD P2G_TITLE=p2gen-dump CUDA_VISIBLE_DEVICES=$(gpu) $W simcode.p2.dump --scen $SCN --seeds $SEEDS \
        --fronts g2=$GM/p2det/final.pt:$GM/p2headL/final.pt --out $FE || { say FAILED dump $v; exit 1; }
      if [ $v = fwd ] && [ ! -f $MY/calib/calib_p2L_$G.json ]; then
        CUDA_VISIBLE_DEVICES=$(gpu) python analysis/p2generic/fitcal_wrap.py analysis/nanoruler/ev_fitcal.py g2 $FE rs05qm calib/calib_p2L.json $MY/calib/calib_p2L_$G.json \
          $MY/calib/calib_p2L_${G}_real.json calib/calib_p2L_b10_real.json || { say FAILED cal; exit 1; }
      fi
      CUDA_VISIBLE_DEVICES=$(gpu) python $A/eval_cmp.py $FE $SCN g2:s2,g2:s1,s2:s1 $CALS $P/eval_nr80_$v.json > $P/eval_nr80_$v.txt && say "g $v done"
    done ;;
s)  mkdir -p $P/sims
    for n in nr_a nr_c0; do SD=out/nanoruler/sim/$n; RD=$P/sims/${n}_$G
      # simcode.run takes a model: nr80 pass 1 + the generic pass 2 $G, sim-sigma calibrations, no sigma rescale (analysis/nanoruler/mkmodel.py)
      [ -f $RD/locs_p2.npz ] || { python analysis/nanoruler/mkmodel.py $RD/model.json nr80_$G $MD/det_nr80.pt $MD/hd_nr80.pt $MC/calib_sc_hd_nr80.json \
        $GM/p2det/final.pt $GM/p2headL/final.pt $MY/calib/calib_p2L_$G.json > /dev/null && \
        CUDA_VISIBLE_DEVICES=$(gpu) python -u analysis/nanoruler/ev_run.py --movie $SD/sim_movie.tif --calib $PWD/$SD/calib_sim.json --out $RD \
        --crop 5,5,246 --model $RD/model.json --sig_u '' > $RD.log 2>&1; } || { say FAILED run $n; exit 1; }
      python analysis/nanoruler/ev_p12sim.py $RD $SD nr > $P/p12_${n}_$G.md 2> $P/p12_${n}_$G.err
    done
    python analysis/nanoruler/ev_flipclass.py --cross std,gen std=$EV/sim_nr_a/run_nr80:out/nanoruler/sim/nr_a gen=$P/sims/nr_a_$G:out/nanoruler/sim/nr_a > $P/flipclass_nr_a_$G.md
    python analysis/nanoruler/ev_flipclass.py --cross std,gen std=$EV/sim_nr_c0/run_nr80:out/nanoruler/sim/nr_c0 gen=$P/sims/nr_c0_$G:out/nanoruler/sim/nr_c0 > $P/flipclass_nr_c0_$G.md
    say "s done" ;;
esac
done
say "eval $G done ($PARTS)"
