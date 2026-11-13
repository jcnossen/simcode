#!/bin/bash
# R24: method tables (own DME v2 drift per method) and the nanoruler analysis of the real nr40 / nr20 movie; resumable (existing outputs are skipped).
# usage: bash analysis/nanoruler/r24_methods.sh <nr40|nr20> <p1|p2>     -> out/nanoruler/<mv>/methods_<p>/
#   tables (tabs/): simcode_u45, simcode_sel (pipeline table of the pass, own drift), bl_smlm, bl_simflux (re-implemented baselines, own drift),
#   published SMLM / SIMFLUX as published (their own drift corrections); simcode_u45_pubdrift = the pipeline rows with the PUBLISHED total drift
#   (drift check); match_pubsf_* = identical events (abl_match.py, every published SIMFLUX row matched to a cycle-aligned pipeline row).
#   Analysis: SITE PAIRS (r24_rul.py; the samples are 2D origami structures, not linear rulers), on several pair lists (see below).
source /home/jc/simcode2/runs/psfmatch/env.sh
MV=$1; N=$2; A=analysis/nanoruler; O=$PWD/out/nanoruler/$MV; REF=$O/ref; M=$O/methods_$N; T=$M/tabs; LG=$O/logs; mkdir -p $T $LG
case $MV in nr40) D0=40; NC=6666;; nr20) D0=20; NC=16666;; esac
LOCS=$O/runs/real/locs_$N.npz; [ -f $LOCS ] || { echo "no $LOCS"; exit 1; }
CMP="--cmp rcc=$REF/drift_rcc.npy pub_smlm=$REF/drift_pub_total_smlm.npy pub_simflux=$REF/drift_pub_total_simflux.npy"
G="bash $A/r24_gpu.sh 3"; q() { grep -v "Warning\|r24_gpu"; }
{
for s in u45 sel; do [ -f $T/simcode_$s.npz ] || $G python $A/r24_tab.py locs $LOCS $T/simcode_$s.npz --ncyc $NC --sel $s $CMP; done
[ -f $T/simcode_u45_pubdrift.npz ] || python $A/r24_tab.py locs $LOCS $T/simcode_u45_pubdrift.npz --ncyc $NC --sel u45 --drift $REF/drift_pub_total_simflux.npy
for b in smlm simflux; do [ -f $T/bl_$b.npz ] || $G python $A/r24_tab.py table $REF/tab_bl_$b.npz $T/bl_$b.npz --ncyc $NC $CMP; done
[ -f $M/driftcheck.json ] || $G python $A/r24_driftcheck.py $LOCS $NC $M/driftcheck.json --sel sel $CMP
[ -f $T/match_pubsf_match.json ] || python $A/abl_match.py $LOCS $T/simcode_u45_drift.npy $REF/tab_pub_simflux.npz $T/match_pubsf --also $REF/tab_pub_smlm.npz
} 2>&1 | q | tee -a $LG/methods_$N.log
ME="pub_smlm=$REF/tab_pub_smlm.npz pub_simflux=$REF/tab_pub_simflux.npz bl_smlm=$T/bl_smlm.npz bl_simflux=$T/bl_simflux.npz simcode_u45=$T/simcode_u45.npz simcode_sel=$T/simcode_sel.npz"
PA="simcode_u45:pub_simflux simcode_sel:pub_simflux simcode_u45:bl_simflux simcode_u45:pub_smlm bl_simflux:pub_simflux bl_smlm:pub_smlm"
MM="simcode=$T/match_pubsf_pipe.npz pub_simflux=$T/match_pubsf_ref.npz"; MU="simcode=$T/match_pubsf_u45_pipe.npz pub_simflux=$T/match_pubsf_u45_ref.npz"
M3="simcode=$T/match_pubsf_3way_pipe.npz pub_simflux=$T/match_pubsf_3way_ref.npz pub_smlm=$T/match_pubsf_3way_also.npz"
# site-pair lists (r24_rul.py pairs): auto_sf = sites found on the published SIMFLUX table; auto_sc = sites found on the pipeline 'sel' table of
# pass 1 (fixed once, also used for pass 2); nr20 also the paper's picks: picks_sf (simflux_picksfull, 1119 centres), picks_g2d (g2d_picksfull,
# 1426 centres, picked on the SMLM reconstruction: independent of both SIMFLUX and the pipeline)
P1T=$O/methods_p1/tabs/simcode_sel.npz
[ -f $O/pairs_auto_sf.json ] || python $A/r24_rul.py pairs --ref $REF/tab_pub_simflux.npz --d0 $D0 --out $O/pairs_auto_sf.json > $LG/pairs_auto_sf.log
[ -f $O/pairs_auto_sc.json ] || python $A/r24_rul.py pairs --ref $P1T --d0 $D0 --out $O/pairs_auto_sc.json > $LG/pairs_auto_sc.log
RL="auto_sf auto_sc"
if [ $MV = nr20 ]; then RL="picks_g2d auto_sf auto_sc"
  [ -f $O/pairs_picks_sf.json ] || python $A/r24_rul.py pairs --picks $REF/picks_simflux.npz --d0 $D0 --out $O/pairs_picks_sf.json > $LG/pairs_picks_sf.log
  [ -f $O/pairs_picks_g2d.json ] || python $A/r24_rul.py pairs --picks $REF/picks_g2d.npz --d0 $D0 --out $O/pairs_picks_g2d.json > $LG/pairs_picks_g2d.log
fi
export R24_JOBS=6
for r in $RL; do X="--pairs $O/pairs_$r.json --d0 $D0 --out $M --levels"
  [ -f $M/rul_$r.json ] || python $A/r24_rul.py run $X --tag $r --methods $ME --cmp $PA --frc 1 > $LG/rul_${N}_$r.log 2>&1
  [ -f $M/rul_${r}_pubdrift.json ] || python $A/r24_rul.py run $X --tag ${r}_pubdrift --methods own=$T/simcode_u45.npz pubdrift=$T/simcode_u45_pubdrift.npz --cmp own:pubdrift > $LG/rul_${N}_${r}_pubdrift.log 2>&1 &
  [ -f $M/rul_${r}_matched.json ] || python $A/r24_rul.py run $X --tag ${r}_matched --methods $MM --cmp simcode:pub_simflux > $LG/rul_${N}_${r}_matched.log 2>&1
  wait
  [ -f $M/rul_${r}_matched_u45.json ] || python $A/r24_rul.py run $X --tag ${r}_matched_u45 --methods $MU --cmp simcode:pub_simflux > $LG/rul_${N}_${r}_matched_u45.log 2>&1 &
  [ -f $M/rul_${r}_matched_3way.json ] || python $A/r24_rul.py run $X --tag ${r}_matched_3way --methods $M3 --cmp simcode:pub_simflux simcode:pub_smlm pub_simflux:pub_smlm > $LG/rul_${N}_${r}_matched_3way.log 2>&1
  wait
done
echo "$(date +%F_%T) r24_methods $MV $N done"
