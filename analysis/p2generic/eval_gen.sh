#!/bin/bash
# Generic pass 2 (2026-11-13): evaluation of the generic set g1809 against the PSF-matched m1809 set (pass 2 s2, pass 1 s1).
#  (a) standard gate scenes out/psfmatch/gate_m1809a (seeds 202 + 301-303): dump g2, calibration fit on seed 202 (rs05qm, rs05pm),
#      head-to-head g2 vs s2 / s1 (s1, s2 dumps of the acceptance run linked in)
#  (b) gate scenes at the real 1809_2 pattern (presets/realpat_1809_2.json)
#  (d) out-of-distribution patterns: nanoruler (ood_nr), 3-px 4-step (ood_p3s4), random phases (ood_rph)
#  (c) sim movie m05r3 with GT: both sets with the current package, identical windows (sigcal cmp) + pass 2 vs pass 1 (ev_p12sim)
# usage: setsid nohup bash analysis/p2generic/eval_gen.sh [a b d c] >> runs/p2generic/eval_g1809.log 2>&1 < /dev/null &
source /home/jc/simcode2/runs/psfmatch/env.sh; set -o pipefail
G=${G:-g1809}; GN=g2; SX=""; [ $G = g1809 ] || { GN=$G; SX=_$G; }; B=m1809; C=${PSF[$B]}; P=$PWD/out/p2generic; A=analysis/p2generic; W="python -u $A/pwrap.py"
PARTS=${@:-a b d c}
gpu() { NEED_GB=4 pickgpu 4 "${PREF:-0 2}"; }
say() { echo "$(date +%F_%T) $*"; }
F3="s1=det_$B.pt:hd_$B.pt,s2=p2det_$B.pt:p2headL_$B.pt,$GN=p2det_$G.pt:p2headL_$G.pt"
CALS="s1=calib/calib_sc_hd_$B.json,s2=calib/calib_p2L_$B.json,$GN=calib/calib_p2L_$G.json"
CMP="$GN:s2,$GN:s1,s2:s1"
export SIMCODE_GATE_CALIB=$C GATE_PSF_CHUNK=1500
for part in $PARTS; do
case $part in
a)  FE=$P/fe_m1809a_gen; [ $G = g1809 ] || FE=$P/fe_m1809a_$G; mkdir -p $FE
    for f in out/psfmatch/fe_m1809a/*__s*_s[12].npz; do ln -sfn $PWD/$f $FE/$(basename $f); done
    SIMCODE_GATE_DIR=out/psfmatch/gate_m1809a P2G_TITLE=p2gen-dump CUDA_VISIBLE_DEVICES=$(gpu) $W simcode.processing.gate_dump --scen iso,pair1,pair2,rs05qm,rs05pm \
      --seeds 202,301,302,303 --fronts $GN=p2det_$G.pt:p2headL_$G.pt --out $FE || { say FAILED dump a; exit 1; }
    [ -f models/calib/calib_p2L_$G.json ] || CUDA_VISIBLE_DEVICES=$(gpu) python $A/fitcal_wrap.py $A/fit_cal_gen.py $GN $FE rs05qm,rs05pm calib/calib_p2L.json \
      models/calib/calib_p2L_$G.json models/calib/calib_p2L_${G}_real.json calib/calib_p2L_b10_real.json || { say FAILED cal; exit 1; }
    CUDA_VISIBLE_DEVICES=$(gpu) python $A/eval_cmp.py $FE iso,pair1,pair2,rs05qm,rs05pm $CMP $CALS $P/eval_a$SX.json > $P/eval_a$SX.txt && say "a done" ;;
b|d) if [ $part = b ]; then LIST="realpat_1809_2"; SCN=iso,pair1,pair2,rs05qm,rs05pm; else LIST="ood_nr ood_p3s4 ood_rph"; SCN=iso,pair2,rs05qm; fi
    for pr in $LIST; do
      GD=$P/gate_$pr; FE=$P/fe_$pr; mkdir -p $GD $FE; g=$(gpu)
      export P2G_PRESET=$P/presets/$pr.json
      last=${SCN##*,}
      [ -f $GD/${last}__s303_hd_cur.npz ] || P2G_TITLE=p2gen-gate CUDA_VISIBLE_DEVICES=$g $W simcode.benchmark.gate --scen $SCN --fronts sc --sc hd_sc5nb.pt \
        --sccal calib/calib_sc_hd_sc5nbb15_pm_bright.json --tag _hd_cur --out $GD --ntarget 3500 --seeds 301,302,303 || { say FAILED gate $pr; exit 1; }
      SIMCODE_GATE_DIR=$GD P2G_TITLE=p2gen-dump CUDA_VISIBLE_DEVICES=$g $W simcode.processing.gate_dump --scen $SCN --seeds 301,302,303 --fronts $F3 --out $FE || { say FAILED dump $pr; exit 1; }
      unset P2G_PRESET
      CUDA_VISIBLE_DEVICES=$g python $A/eval_cmp.py $FE $SCN $CMP $CALS $P/eval_$pr$SX.json > $P/eval_$pr$SX.txt && say "$part $pr done"
    done ;;
c)  SD=out/psf3d_nets/sims/m05r3; mkdir -p $P/sims
    for s in $B $G; do RD=$P/sims/m05r3_$s
      # simcode.run takes a model: pass 1 of $B, pass 2 of $s, with these (sim-sigma) calibrations, no sigma rescale (analysis/nanoruler/mkmodel.py)
      [ -f $RD/locs_p2.npz ] || { python analysis/nanoruler/mkmodel.py $RD/model.json m05r3_$s det_$B.pt hd_$B.pt calib/calib_sc_hd_$B.json \
        p2det_$s.pt p2headL_$s.pt calib/calib_p2L_$s.json > /dev/null && \
        CUDA_VISIBLE_DEVICES=$(gpu) python -m simcode.run --movie $SD/sim_movie.tif --calib out/psf3d_nets/calibs/calib_1809_2_psf3d_simcam.json \
        --out $RD --model $RD/model.json --sig_u '' > $P/sims/m05r3_$s.log 2>&1; } || { say FAILED sim $s; exit 1; }
      python analysis/nanoruler/ev_p12sim.py $RD $SD sc > $P/p12_m05r3_$s.md 2> $P/p12_m05r3_$s.err
    done
    python $A/simcmp_ci.py $SD est2 $P/sims/m05r3_$G $P/sims/m05r3_$B > $P/simcmp_m05r3_p2_${G}_vs_$B.txt
    python $A/simcmp_ci.py $SD est $P/sims/m05r3_$G $P/sims/m05r3_$B > $P/simcmp_m05r3_p1_${G}_vs_$B.txt
    say "c done" ;;
esac
done
say "eval $G done ($PARTS)"
