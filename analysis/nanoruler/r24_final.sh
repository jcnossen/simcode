#!/bin/bash
# R24: the remaining pass-1 steps for one movie, in order, resumable: grid-simulation tables + pair-tool validation, solve variants (real movie,
# grid simulations), pattern test (shared drift blocks of 30 / 60 frames instead of 150: abl_pattern.py make, solve A, theta for k30, paired pair metrics).
# usage: bash analysis/nanoruler/r24_final.sh <nr40|nr20>
source /home/jc/simcode2/runs/psfmatch/env.sh
MV=$1; A=analysis/nanoruler; O=$PWD/out/nanoruler/$MV; B=$O/abl; LG=$O/logs; D0=${MV#nr}; SIM=s$D0; G="bash $A/r24_gpu.sh"
case $MV in nr40) NC=6666;; nr20) NC=16666;; esac
q() { grep -v "Warning\|r24_gpu\|^W1[0-9]\{3\} "; }
for s in ${SIM}g_a ${SIM}g_c0; do NS=$(python -c "import json; print(json.load(open('$O/sim/$s/sim_info.json'))['args']['n_frames'] // 6)")
  [ -f $O/simeval/gt_${s}_p1.json ] || python $A/r24_simeval.py $O/runs/$s $O/sim/$s --out $O/simeval/gt_${s}_p1.json --name "$s pass 1" > /dev/null 2>&1
  for sel in u45 sel; do [ -f $O/simeval/tabs/${s}_p1_$sel.npz ] || $G 3 python $A/r24_tab.py locs $O/runs/$s/locs_p1.npz $O/simeval/tabs/${s}_p1_$sel.npz --ncyc $NS --sel $sel --gt $O/sim/$s 2>&1 | q; done
done > $LG/final_simtabs.log 2>&1
PT=$O/simeval/pairs_truth_${SIM}g.json; PA=$O/simeval/pairs_auto_${SIM}g.json
[ -f $PT ] || python $A/r24_rul.py pairs --truth $O/sim/${SIM}g_a/sites.npz --d0 $D0 --out $PT --fov 0 16640 > $LG/pairs_truth_g.log
[ -f $PA ] || python $A/r24_rul.py pairs --ref $O/simeval/tabs/${SIM}g_a_p1_sel.npz --d0 $D0 --out $PA --fov 0 16640 > $LG/pairs_auto_g.log
export R24_JOBS=6
for s in ${SIM}g_a ${SIM}g_c0; do for p in truth auto; do
  [ -f $O/simeval/rul_${s}_$p.json ] || python $A/r24_rul.py run --pairs $O/simeval/pairs_${p}_${SIM}g.json --d0 $D0 --out $O/simeval --tag ${s}_$p --methods u45=$O/simeval/tabs/${s}_p1_u45.npz sel=$O/simeval/tabs/${s}_p1_sel.npz --cmp u45:sel --levels > $LG/rul_${s}_$p.log 2>&1
done; done
bash $A/r24_variants.sh $MV p1 $O/pairs_auto_sf.json
bash $A/r24_variants.sh $MV simga_p1 $PT
bash $A/r24_variants.sh $MV simgc0_p1 $PT
# pattern test
[ -f $B/patterns_p1.json ] || $G 5 python $A/r24_abl.py $MV pattern make p1 --kf 30,60 2>&1 | q > $LG/pattern_p1.log
T=$B/tabs_p1; M="est=$T/A_csel.npz"; MU="est=$T/A_cu45.npz"; P=""
for t in k30 k60; do
  [ -f $B/runs/p1_$t/A/locs.npz ] || $G 5 python $A/r24_abl.py $MV solve p1 A --pattern $B/patterns/p1_$t.pt --ptag $t --auto 0 2>&1 | q >> $LG/solve_p1_pat.log
  for sel in u45 sel; do [ -f $T/pat_${t}_c$sel.npz ] || python $A/r24_tab.py locs $B/runs/p1_$t/A/locs.npz $T/pat_${t}_c$sel.npz --ncyc $NC --sel $sel --mask_from $B/runs/p1/A/locs.npz --drift $T/A_u45_drift.npy >> $LG/tab_p1_pat.log 2>&1; done
  M="$M $t=$T/pat_${t}_csel.npz"; MU="$MU $t=$T/pat_${t}_cu45.npz"; P="$P $t:est"
done
[ -f $B/rul_p1_pat_sel.json ] || python $A/r24_rul.py run --pairs $O/pairs_auto_sf.json --d0 $D0 --out $B --tag p1_pat_sel --methods $M --cmp $P --levels raw > $LG/rul_p1_pat_sel.log 2>&1
[ -f $B/rul_p1_pat_common.json ] || python $A/r24_rul.py run --pairs $O/pairs_auto_sf.json --d0 $D0 --out $B --tag p1_pat_common --methods $MU --cmp $P --levels raw > $LG/rul_p1_pat_common.log 2>&1
[ -f $B/autoc_p1_k30.json ] || { $G 5 python $A/r24_abl.py $MV autoc est p1 --pattern $B/patterns/p1_k30.pt --name k30 && $G 5 python $A/r24_abl.py $MV autoc branch p1 --pattern $B/patterns/p1_k30.pt --name k30; } 2>&1 | q > $LG/autoc_p1_k30.log
echo "$(date +%F_%T) r24_final $MV done"
