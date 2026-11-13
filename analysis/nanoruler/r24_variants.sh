#!/bin/bash
# R24: solve variants of one set (abl_all.sh 'variants' for nr40 / nr20 through r24_abl.py), resumable; then tables with ONE drift trace
# (own DME v2 on variant A's cycle-aligned u <= 0.45 rows), common views (rows u <= 0.45 / sel of variant A) and the site-pair metrics
# (r24_rul.py); for simulated sets also the GT evaluation per variant (r24_simeval.py, common rows, calibrated and uncalibrated sigma, paired to A).
# usage: bash analysis/nanoruler/r24_variants.sh <nr40|nr20> <set p1|sim[g]a_p1|sim[g]c0_p1|p2|...> <pairs.json> [variants, default "A B C D E F G H auto autoS"]
source /home/jc/simcode2/runs/psfmatch/env.sh
MV=$1; S=$2; RUL=$3; VARS=${4:-"A B C D E F G H auto autoS"}; A=analysis/nanoruler; O=$PWD/out/nanoruler/$MV; B=$O/abl; LG=$O/logs; mkdir -p $B $LG
case $MV in nr40) D0=40; NC=6666; SIM=s40;; nr20) D0=20; NC=16666; SIM=s20;; esac
case $S in simga*) SD=$O/sim/${SIM}g_a;; simgc0*) SD=$O/sim/${SIM}g_c0;; sima*) SD=$O/sim/${SIM}_a;; simc0*) SD=$O/sim/${SIM}_c0;; *) SD="";; esac
[ -n "$SD" ] && NC=$(python -c "import json; print(json.load(open('$SD/sim_info.json'))['args']['n_frames'] // 6)")
FR=s1; TH=$S; case $S in *p2) FR=s2; TH=${S/p2/p1};; esac
G="bash $A/r24_gpu.sh 5"; run() { "$@" 2>&1 | grep -v "Warning\|^W1[0-9]\{3\} \|r24_gpu"; return ${PIPESTATUS[0]}; }
[ -f $B/autoc_$TH.json ] || { run $G python $A/r24_abl.py $MV autoc est $TH && run $G python $A/r24_abl.py $MV autoc branch $TH; } > $LG/autoc_$TH.log || { echo "FAILED autoc $TH"; exit 1; }
grep -q branch $B/autoc_$TH.json || run $G python $A/r24_abl.py $MV autoc branch $TH >> $LG/autoc_$TH.log
run $G python $A/r24_abl.py $MV gate $FR all --auto $TH --noeval > $LG/gate_${FR}_$TH.log || { echo "FAILED gate"; exit 1; }
run $G python $A/r24_abl.py $MV solve $S all > $LG/solve_$S.log || { echo "FAILED solve $S"; exit 1; }
R=$B/runs/$S; T=$B/tabs_$S; mkdir -p $T
GT=""; [ -n "$SD" ] && GT="--gt $SD"
[ -f $T/A_u45_drift.npy ] || run $G python $A/r24_tab.py locs $R/A/locs.npz $T/A_u45.npz --ncyc $NC --sel u45 $GT > $LG/tab_$S.log
M=""; MS=""; P=""
for v in $VARS; do [ -f $R/$v/locs.npz ] || continue
  for sel in u45 sel; do [ -f $T/${v}_c$sel.npz ] || python $A/r24_tab.py locs $R/$v/locs.npz $T/${v}_c$sel.npz --ncyc $NC --sel $sel --mask_from $R/A/locs.npz --drift $T/A_u45_drift.npy >> $LG/tab_$S.log 2>&1; done
  M="$M $v=$T/${v}_cu45.npz"; MS="$MS $v=$T/${v}_csel.npz"; [ $v != A ] && { P="$P $v:A"; }
done
export R24_JOBS=5
[ -f $B/rul_${S}_common.json ] || python $A/r24_rul.py run --pairs $RUL --d0 $D0 --out $B --tag ${S}_common --methods $M --cmp $P --levels raw nb f > $LG/rul_${S}_common.log 2>&1 &
[ -f $B/rul_${S}_sel.json ] || python $A/r24_rul.py run --pairs $RUL --d0 $D0 --out $B --tag ${S}_sel --methods $MS --cmp $P --levels raw nb f > $LG/rul_${S}_sel.log 2>&1 &
if [ -n "$SD" ]; then mkdir -p $B/gt_$S
  for v in $VARS; do [ -f $R/$v/locs.npz ] || continue
    [ -f $B/gt_$S/${v}_cal.json ] || python $A/r24_simeval.py $R/$v/locs.npz $SD --ref $R/A/locs.npz --out $B/gt_$S/${v}_cal.json --name "$S $v calibrated sigma" > /dev/null 2>&1
    [ -f $B/gt_$S/${v}_raw.json ] || python $A/r24_simeval.py $R/$v/locs.npz $SD --ref $R/A/locs.npz --sig sx_raw,sy_raw --out $B/gt_$S/${v}_raw.json --name "$S $v uncalibrated sigma" > /dev/null 2>&1
  done
fi
wait; echo "$(date +%F_%T) r24_variants $MV $S done"
