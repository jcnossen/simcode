#!/bin/bash
# Nanoruler solve-variant ablation: everything, resumable (each step is skipped when its output exists; delete the output to redo).
# usage: cd ~/simcode2; GPU=1 setsid nohup bash analysis/nanoruler/abl_all.sh [stages] >> out/nanoruler/ablation/logs/abl_all.log 2>&1 < /dev/null &
#   stages (default: p1 pat sim1 p2 sim2 methods fig; extra: gtpat): 
#     p1      real movie, pass 1: automatic theta, per-variant gate calibrations (front s1), solves, nanoruler metrics
#     pat     pattern-model test on the pass-1 rows (per-step pattern, finer drift blocks): solves, theta, paired metrics
#     sim1    simulated movies nr_a / nr_c0, pass 1 (pipeline run through ev_run.py, EV_DET_BATCH=8), theta, solves, GT evaluation
#     p2      real movie, pass 2: needs the nr80 pass-2 networks, the s2 gate dumps and calib_p2L_nr80.json of the evaluation chain
#             ("eval nr80 done" in out/nanoruler/eval/chain.log); captures the pass-2 head outputs (det batch 8, ~5 GB, ~15 GPU-min), then as p1
#     sim2    simulated movies, pass 2 (same prerequisites)
#     methods method table (published / re-implemented baselines / SIMCODE v2 of every finished pass) + matched comparison
#     fig     excess-variance figure and theta table of all finished sets, report figures (abl_figs.py: gallery, per-method photon figure, ablation figure + tab_ablation.md)
#     gtpat   control: nr_c0 with the TRUE pattern given to the solve and the pass-2 networks, pass 1 versus pass 2 on identical GT windows (needs sim2)
# The pass-1 network outputs of the real movie are those of out/nanoruler/runs/real_p1 (same networks as det_nr80 / hd_nr80); the pass-2
# networks only need that run's pattern_est.pt. Each GPU step needs < 6 GB; the GPU is checked once per stage (free memory >= NEED_GB + 2 GB).
source /home/jc/simcode2/runs/psfmatch/env.sh
cd /home/jc/simcode2; A=analysis/nanoruler; NR=$PWD/out/nanoruler; O=$NR/ablation; LG=$O/logs; mkdir -p $LG $O/prod
export NR_TITLE=nr-abl EV_DET_BATCH=8
STAGES=${@:-p1 pat sim1 p2 sim2 methods fig}
say() { echo "$(date +%F_%T) $*"; }
gpu() {  # wait for the GPU of $GPU (default: first of 1 2 0) with enough free memory
  local need=$(( (${NEED_GB:-6} + 2) * 1024 )) g f n=0
  while true; do
    for g in ${GPU:-1 2 0}; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g); [ "$f" -ge $need ] && { export CUDA_VISIBLE_DEVICES=$g; say "GPU $g ($f MiB free)"; return 0; }; done
    [ $((n % 30)) = 0 ] && say "waiting for a GPU with $need MiB free"; n=$((n + 1)); sleep 60
  done
}
run() { "$@" 2>&1 | grep -v "Warning\|^W1[0-9]\{3\} "; return ${PIPESTATUS[0]}; }
p2ready() { grep -q "eval nr80 done" $NR/eval/chain.log && [ -f $NR/models/calib/calib_p2L_nr80.json ] && [ -f $NR/eval/fe_nr80/rs05qm__s202_s2.npz ] && [ -f $NR/models/p2headL_nr80.pt ]; }
variants() {  # set front: theta, calibrations, solves, metrics
  local s=$1 fr=$2 th=${1/p2/p1}
  [ -f $O/autoc_$th.json ] || { run python $A/abl_autoc.py est $th && run python $A/abl_autoc.py branch $th; } || return 1
  local NE=""; grep -q "\"$fr-$th\"" $O/gate_done.txt 2>/dev/null && NE=--noeval      # the gate evaluation (seeds 301-303) once per front and theta
  run python $A/abl_gate.py $fr all --auto $th $NE > $LG/gate_${fr}_$th.log || return 1
  echo "\"$fr-$th\"" >> $O/gate_done.txt
  run python $A/abl_solve.py $s all > $LG/solve_$s.log || return 1
  [ -f $O/nra_$s.json ] || run python $A/abl_nra.py $s > $LG/nra_$s.log || return 1
}
simrun() {  # movie: pass-1 pipeline run (pattern estimated from the data; the solve calibration of this run is not used by the ablation)
  local m=$1 SD=$NR/sim/$1
  [ -f $O/prod/$m/locs_p1.npz ] && return 0
  # model nr80 (its pass-1 files are copies of train/det, train/hd final.pt) with the sim calibrations; simcode.run also runs pass 2
  run python -u $A/ev_run.py --movie $SD/sim_movie.tif --calib $SD/calib_sim.json --out $O/prod/$m --crop 5,5,246 --chunk 1500 \
    --model $NR/models/netset_nr80.json --cal_kind sim --sig_u '' > $LG/prod_${m}_p1.log
}
for st in $STAGES; do
  say "stage $st"
  case $st in
  p1) gpu; variants p1 s1 || say "FAILED p1";;
  pat) gpu
    [ -f $O/patterns_p1.json ] || run python $A/abl_pattern.py make p1 --kf 30,60,150 > $LG/pattern_p1.log
    [ -f $O/patterns/p1_d91.pt ] || run python $A/abl_pattern.py depth p1 >> $LG/pattern_p1.log
    for t in ps psd shd k30 k60 d97 d94 d91; do
      [ -f $O/runs/p1_$t/A/locs.npz ] || run python $A/abl_solve.py p1 A --pattern $O/patterns/p1_$t.pt --ptag $t --auto 0 >> $LG/solve_p1_pat.log
      case $t in d*) continue;; esac          # depth scan: positions only (theta: ~7 GPU-min each, run by hand if wanted)
      [ -f $O/autoc_p1_$t.json ] || { run python $A/abl_autoc.py est p1 --pattern $O/patterns/p1_$t.pt --name $t && run python $A/abl_autoc.py branch p1 --pattern $O/patterns/p1_$t.pt --name $t; } >> $LG/autoc_p1_pat.log
    done
    T=$(python $A/abl_pattern.py link p1 | tail -1)
    [ -f $O/nra_p1_pat.json ] || run python $A/abl_nra.py p1 --ptag pat --variants $T --ref est > $LG/nra_p1_pat.log;;
  sim1) gpu
    for m in nr_a nr_c0; do s=sim${m#nr_}_p1; simrun $m && variants $s s1 || { say "FAILED $s"; continue; }
      [ -f $O/sim_$s.json ] || run python $A/abl_sim.py $s > $LG/sim_$s.log; done;;
  p2) p2ready || { say "p2: the evaluation chain has not finished (no 'eval nr80 done'); skipped"; continue; }
    gpu; [ -f $O/head/real/head_28500.npz ] || run python $A/abl_capture.py p2 > $LG/capture_p2.log || { say "FAILED capture p2"; continue; }
    variants p2 s2 || say "FAILED p2"
    [ -f $O/autoc_p2.json ] || { run python $A/abl_autoc.py est p2 && run python $A/abl_autoc.py branch p2; } > $LG/autoc_p2.log      # theta of the pass-2 head outputs (reported; the variants use the pass-1 theta)
    [ -f $O/p12_real.json ] || run python $A/abl_p12.py real > $LG/p12_real.log;;
  sim2) p2ready || { say "sim2: the evaluation chain has not finished; skipped"; continue; }
    gpu
    for m in nr_a nr_c0; do s=sim${m#nr_}_p2; simrun $m || continue
      [ -f $O/head/$m/head_10500.npz ] || run python $A/abl_capture.py $s > $LG/capture_$s.log || { say "FAILED capture $s"; continue; }
      variants $s s2 || { say "FAILED $s"; continue; }
      [ -f $O/sim_$s.json ] || run python $A/abl_sim.py $s > $LG/sim_$s.log
      [ -f $O/autoc_$s.json ] || { run python $A/abl_autoc.py est $s && run python $A/abl_autoc.py branch $s; } > $LG/autoc_$s.log
      [ -f $O/p12_$m.json ] || run python $A/abl_p12.py sim $m > $LG/p12_$m.log; done;;
  methods) gpu
    for p in p1 p2; do [ -f $O/runs/$p/A/locs.npz ] && [ ! -f $O/methods_$p/metrics.md ] && bash $A/abl_methods.sh $O/runs/$p/A/locs.npz $p $CUDA_VISIBLE_DEVICES > $LG/methods_$p.out 2>&1; done
    if [ -f $O/tabs/tab_p2_m12.npz ] && [ ! -f $O/methods_p12/metrics.md ]; then      # both passes next to the published tables and the baselines
      P=$NR/analysis; T=$O/tabs
      python $A/nra_run.py run --sites $P/sites.json --out $O/methods_p12 --bins 6 --nboot 500 --gallery_n 12 \
        --methods pub_smlm=$P/tab_pub_smlm.npz pub_simflux=$P/tab_pub_simflux.npz bl_smlm=$T/tab_bl_smlm_dme.npz bl_simflux=$T/tab_bl_simflux_dme.npz \
                  simcode_p1_m12=$T/tab_p1_m12.npz simcode_p1_sel=$T/tab_p1_sel.npz simcode_p2_m12=$T/tab_p2_m12.npz simcode_p2_sel=$T/tab_p2_sel.npz \
        --pairs simcode_p2_m12:simcode_p1_m12 simcode_p2_sel:simcode_p1_sel simcode_p2_m12:pub_simflux simcode_p2_sel:pub_simflux simcode_p1_m12:pub_simflux simcode_p2_m12:bl_simflux > $LG/nra_methods_p12.log 2>&1
    fi;;
  fig) N=""; for s in p1 p2 sima_p1 simc0_p1 sima_p2 simc0_p2 p1_ps p1_psd p1_shd p1_k60 p1_k30; do [ -f $O/autoc_$s.json ] && N="$N,$s"; done
    run python $A/abl_autoc.py fig ${N#,} $O/fig_excess_variance_all.png > $LG/fig_autoc.log        # all sets + the table tab_autoc_all.md
    M=""; for s in p1 p2 sima_p1 simc0_p1; do [ -f $O/autoc_$s.json ] && M="$M,$s"; done
    run python $A/abl_autoc.py fig ${M#,} $O/fig_excess_variance.png --notable >> $LG/fig_autoc.log
    for w in gallery methods ablation; do run python $A/abl_figs.py $w >> $LG/figs.log; done;;
  gtpat) p2ready || continue; gpu; D=$O/prod/nr_c0g
    if [ ! -f $D/pattern_est.pt ]; then mkdir -p $D; for f in $O/prod/nr_c0/pass1_*.npz; do ln -sf $f $D/; done; cp $NR/sim/nr_c0/gt_pattern.pt $D/pattern_est.pt; fi
    { run python $A/abl_capture.py simc0g_p2 && run python $A/abl_solve.py simc0g_p1 A --auto 0 && run python $A/abl_solve.py simc0g_p2 A --auto 0 && run python $A/abl_p12.py sim nr_c0g; } > $LG/gtpat.log;;
  esac
done
say "abl_all done ($STAGES)"
