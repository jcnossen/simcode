#!/bin/bash
# Nanoruler resume (2026-11-13 evening, pass 1 only): the paper's main selection 'm12' (strict_u finite and max(sx, sy) <= 12 nm; abl_lib.selection)
# added to the 80 / 40 / 20 nm method tables, the finished parts of the 14:50-interrupted 40 / 20 nm work, and the grid simulations. Resumable
# (existing outputs are skipped). GPU steps through r24_gpu.sh (global lock) on GPU 1 only (R24_GPUS=1); SIMCODE sigma = the stored calibrated
# sigma (no g(strict_u); the sigma the 12 nm rule tests) for every m12 table.
# usage: bash analysis/nanoruler/nr12_chain.sh <stage> ...   stages: t80 m80 tabs40 tabs20 meth40 meth20 sim40 sim20 var40 var40m12 simvar40 sim80
source /home/jc/simcode2/runs/psfmatch/env.sh
export R24_GPUS=${R24_GPUS:-1}
A=analysis/nanoruler; NR=$PWD/out/nanoruler; G="bash $A/r24_gpu.sh"; q() { grep -v "Warning\|^W1[0-9]\{3\} "; }
nc() { case $1 in nr40) echo 6666;; nr20) echo 16666;; esac; }

t80() {   # 80 nm: m12 table (own DME v2), m12 sub-set of the matched events (calibrated-sigma match tables)
  local T=$NR/ablation/tabs_cal L=$NR/ablation/runs/p1/A/locs.npz
  [ -f $T/tab_p1_m12.npz ] || $G 3 python $A/abl_convert.py locs $L $T/tab_p1_m12.npz --sel m12 --sig cal 2>&1 | q
  [ -f $T/match_p1_pubsf_m12_pipe.npz ] || python $A/nr12_sub.py $L $T/match_p1_pubsf
}
m80() {   # 80 nm: nanoruler metrics on the fixed site list (as abl_methods.sh) with the m12 row
  local T=$NR/ablation/tabs_cal TB=$NR/ablation/tabs PUB=$NR/analysis O=$NR/ablation/methods_p1m12
  [ -f $O/metrics.md ] || python $A/nra_run.py run --sites $PUB/sites.json --out $O --bins 6 --nboot 500 \
    --methods pub_smlm=$PUB/tab_pub_smlm.npz pub_simflux=$PUB/tab_pub_simflux.npz bl_smlm=$TB/tab_bl_smlm_dme.npz bl_simflux=$TB/tab_bl_simflux_dme.npz \
              simcode_p1_m12=$T/tab_p1_m12.npz simcode_p1_u45=$T/tab_p1_u45.npz \
    --pairs simcode_p1_m12:pub_simflux simcode_p1_m12:bl_simflux simcode_p1_m12:pub_smlm simcode_p1_u45:pub_simflux bl_simflux:pub_simflux bl_smlm:pub_smlm > $NR/ablation/logs/nra_methods_p1m12.log 2>&1 &
  [ -f $O/metrics_matched_m12.md ] || python $A/nra_run.py run --sites $PUB/sites.json --out $O --tag matched_m12 --bins 6 --nboot 500 \
    --methods simcode_p1=$T/match_p1_pubsf_m12_pipe.npz pub_simflux=$T/match_p1_pubsf_m12_ref.npz --pairs simcode_p1:pub_simflux > $NR/ablation/logs/nra_matched_m12_p1.log 2>&1 &
  wait
}
tabs() {  # 40 / 20 nm: m12 table of the real movie (own DME v2, common frame), m12 sub-set of the matched events
  local MV=$1 O=$NR/$1; local T=$O/methods_p1/tabs REF=$O/ref
  local CMP="--cmp rcc=$REF/drift_rcc.npy pub_smlm=$REF/drift_pub_total_smlm.npy pub_simflux=$REF/drift_pub_total_simflux.npy"
  [ -f $T/simcode_m12.npz ] || $G 3 python $A/r24_tab.py locs $O/runs/real/locs_p1.npz $T/simcode_m12.npz --ncyc $(nc $MV) --sel m12 $CMP 2>&1 | q
  [ -f $T/match_pubsf_m12_pipe.npz ] || python $A/nr12_sub.py $O/runs/real/locs_p1.npz $T/match_pubsf
}
meth() {  # 40 / 20 nm: finish r24_methods.sh (resumable), then the site-pair tables with the m12 row (tag m12_<list>) and the m12 matched events
  local MV=$1 O=$NR/$1 D0=${1#nr}; local M=$O/methods_p1 T=$O/methods_p1/tabs REF=$O/ref LG=$O/logs
  bash $A/r24_methods.sh $MV p1
  local ME="pub_smlm=$REF/tab_pub_smlm.npz pub_simflux=$REF/tab_pub_simflux.npz bl_smlm=$T/bl_smlm.npz bl_simflux=$T/bl_simflux.npz simcode_m12=$T/simcode_m12.npz simcode_u45=$T/simcode_u45.npz"
  local PA="simcode_m12:pub_simflux simcode_m12:bl_simflux simcode_m12:pub_smlm simcode_u45:pub_simflux bl_simflux:pub_simflux bl_smlm:pub_smlm"
  local RL="auto_sf auto_sc"; [ $MV = nr20 ] && RL="picks_g2d auto_sf auto_sc"
  export R24_JOBS=8
  for r in $RL; do local X="--pairs $O/pairs_$r.json --d0 $D0 --out $M --levels"
    [ -f $M/rul_m12_$r.json ] || python $A/r24_rul.py run $X --tag m12_$r --methods $ME --cmp $PA --frc 1 > $LG/rul_p1_m12_$r.log 2>&1
    [ -f $M/rul_${r}_matched_m12.json ] || python $A/r24_rul.py run $X --tag ${r}_matched_m12 --methods simcode=$T/match_pubsf_m12_pipe.npz pub_simflux=$T/match_pubsf_m12_ref.npz --cmp simcode:pub_simflux > $LG/rul_p1_${r}_matched_m12.log 2>&1
  done
  echo "$(date +%F_%T) nr12 meth $MV done"
}
sim() {   # grid simulations (s40g / s20g): GT evaluation (with m12), tables u45 / sel / m12 with GT drift, truth / auto pair lists, pair-tool validation
  local MV=$1 O=$NR/$1 D0=${1#nr}; local SIM=s$D0 LG=$O/logs; mkdir -p $O/simeval/tabs
  for s in ${SIM}g_a ${SIM}g_c0; do local NS=$(python -c "import json; print(json.load(open('$O/sim/$s/sim_info.json'))['args']['n_frames'] // 6)")
    [ -f $O/simeval/gt_${s}_p1m12.json ] || python $A/r24_simeval.py $O/runs/$s $O/sim/$s --out $O/simeval/gt_${s}_p1m12.json --name "$s pass 1" > /dev/null 2>&1
    for sel in u45 sel m12; do [ -f $O/simeval/tabs/${s}_p1_$sel.npz ] || $G 3 python $A/r24_tab.py locs $O/runs/$s/locs_p1.npz $O/simeval/tabs/${s}_p1_$sel.npz --ncyc $NS --sel $sel --gt $O/sim/$s 2>&1 | q; done
  done > $LG/nr12_simtabs.log 2>&1
  local PT=$O/simeval/pairs_truth_${SIM}g.json PA=$O/simeval/pairs_auto_${SIM}g.json
  [ -f $PT ] || python $A/r24_rul.py pairs --truth $O/sim/${SIM}g_a/sites.npz --d0 $D0 --out $PT --fov 0 16640 > $LG/pairs_truth_g.log
  [ -f $PA ] || python $A/r24_rul.py pairs --ref $O/simeval/tabs/${SIM}g_a_p1_sel.npz --d0 $D0 --out $PA --fov 0 16640 > $LG/pairs_auto_g.log
  export R24_JOBS=8
  for s in ${SIM}g_a ${SIM}g_c0; do for p in truth auto; do
    [ -f $O/simeval/rul_m12_${s}_$p.json ] || python $A/r24_rul.py run --pairs $O/simeval/pairs_${p}_${SIM}g.json --d0 $D0 --out $O/simeval --tag m12_${s}_$p \
      --methods m12=$O/simeval/tabs/${s}_p1_m12.npz u45=$O/simeval/tabs/${s}_p1_u45.npz sel=$O/simeval/tabs/${s}_p1_sel.npz --cmp u45:m12 sel:m12 --levels > $LG/rul_m12_${s}_$p.log 2>&1
  done; done
  echo "$(date +%F_%T) nr12 sim $MV done"
}
var40m12() {  # 40 nm real-movie solve variants: the common view on variant A's m12 rows (one drift trace: A's u45 own DME), site-pair metrics
  local O=$NR/nr40; local B=$O/abl; local R=$B/runs/p1 T=$B/tabs_p1 M="" P=""
  for v in A B C D E F G H auto autoS; do [ -f $R/$v/locs.npz ] || continue
    [ -f $T/${v}_cm12.npz ] || python $A/r24_tab.py locs $R/$v/locs.npz $T/${v}_cm12.npz --ncyc 6666 --sel m12 --mask_from $R/A/locs.npz --drift $T/A_u45_drift.npy >> $O/logs/tab_p1_m12.log 2>&1
    M="$M $v=$T/${v}_cm12.npz"; [ $v != A ] && P="$P $v:A"
  done
  export R24_JOBS=8
  [ -f $B/rul_p1_m12.json ] || python $A/r24_rul.py run --pairs $O/pairs_auto_sf.json --d0 40 --out $B --tag p1_m12 --methods $M --cmp $P --levels raw nb f > $O/logs/rul_p1_m12.log 2>&1
  echo "$(date +%F_%T) nr12 var40m12 done"
}
simvar40() {  # 40 nm grid simulations: solve variants with GT (r24_variants.sh; queued at 14:50)
  local PT=$NR/nr40/simeval/pairs_truth_s40g.json
  bash $A/r24_variants.sh nr40 simga_p1 $PT; bash $A/r24_variants.sh nr40 simgc0_p1 $PT
  echo "$(date +%F_%T) nr12 simvar40 done"
}
sim80() {  # 80 nm simulated movies: GT evaluation of variant A pass 1 with the m12 row
  for s in a c0; do local D=$NR/ablation/runs/sim${s}_p1/A/locs.npz
    [ -f $NR/ablation/gt_m12_nr_$s.json ] || python $A/r24_simeval.py $D $NR/sim/nr_$s --out $NR/ablation/gt_m12_nr_$s.json --name "nr_$s (80 nm) pass 1, variant A" > /dev/null 2>&1
  done
}
for st in "$@"; do case $st in t80) t80;; m80) m80;; tabs40) tabs nr40;; tabs20) tabs nr20;; meth40) meth nr40;; meth20) meth nr20;; sim40) sim nr40;; sim20) sim nr20;;
  var40m12) var40m12;; simvar40) simvar40;; sim80) sim80;; *) echo "unknown stage $st";; esac; echo "$(date +%F_%T) stage $st done"; done
