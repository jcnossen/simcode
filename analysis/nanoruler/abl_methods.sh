#!/bin/bash
# Nanoruler ablation, part A: method tables (own DME v2 drift) and the nanoruler analysis on the fixed site list, plus the matched comparison.
# SIMCODE sigma: env SIG=u (default) = calibrated sigma x g(strict_u) of the chain (the production sigma, without the other setup's multiplier); SIG=cal = without the rescale.
# usage: bash analysis/nanoruler/abl_methods.sh <pipeline locs npz> <name, e.g. p1> [gpu]     (outputs: out/nanoruler/ablation/tabs, methods_<name>/)
source /home/jc/simcode2/runs/psfmatch/env.sh
cd /home/jc/simcode2; A=analysis/nanoruler; O=out/nanoruler/ablation; T=$O/tabs; LOCS=$1; N=$2; export CUDA_VISIBLE_DEVICES=${3:-0}
PUB=out/nanoruler/analysis; BL=out/nanoruler/baselines; mkdir -p $T $O/logs
{
for s in u45 sel; do [ -f $T/tab_${N}_$s.npz ] || python $A/abl_convert.py locs $LOCS $T/tab_${N}_$s.npz --sel $s --sig ${SIG:-u}; done
for b in smlm simflux; do [ -f $T/tab_bl_${b}_dme.npz ] || python $A/abl_convert.py table $BL/tab_bl_$b.npz $T/tab_bl_${b}_dme.npz; done
python $A/abl_match.py $LOCS $T/tab_${N}_u45_drift.npy $PUB/tab_pub_simflux.npz $T/match_${N}_pubsf --also $PUB/tab_pub_smlm.npz --sig ${SIG:-u}
} 2>&1 | grep -v Warning | tee $O/logs/methods_$N.log
python $A/nra_run.py run --sites $PUB/sites.json --out $O/methods_$N --bins 6 --nboot 500 \
  --methods pub_smlm=$PUB/tab_pub_smlm.npz pub_simflux=$PUB/tab_pub_simflux.npz bl_smlm=$T/tab_bl_smlm_dme.npz bl_simflux=$T/tab_bl_simflux_dme.npz \
            simcode_${N}_u45=$T/tab_${N}_u45.npz simcode_${N}_sel=$T/tab_${N}_sel.npz \
  --pairs simcode_${N}_u45:pub_simflux simcode_${N}_sel:pub_simflux simcode_${N}_u45:bl_simflux simcode_${N}_u45:pub_smlm bl_simflux:pub_simflux bl_smlm:pub_smlm > $O/logs/nra_methods_$N.log 2>&1 &
python $A/nra_run.py run --sites $PUB/sites.json --out $O/methods_$N --tag matched --bins 6 --nboot 500 \
  --methods simcode_${N}=$T/match_${N}_pubsf_pipe.npz pub_simflux=$T/match_${N}_pubsf_ref.npz --pairs simcode_${N}:pub_simflux > $O/logs/nra_matched_$N.log 2>&1 &
python $A/nra_run.py run --sites $PUB/sites.json --out $O/methods_$N --tag matched_u45 --bins 6 --nboot 500 \
  --methods simcode_${N}=$T/match_${N}_pubsf_u45_pipe.npz pub_simflux=$T/match_${N}_pubsf_u45_ref.npz --pairs simcode_${N}:pub_simflux > $O/logs/nra_matched_u45_$N.log 2>&1 &
python $A/nra_run.py run --sites $PUB/sites.json --out $O/methods_$N --tag matched_3way --bins 6 --nboot 500 \
  --methods simcode_${N}=$T/match_${N}_pubsf_3way_pipe.npz pub_simflux=$T/match_${N}_pubsf_3way_ref.npz pub_smlm=$T/match_${N}_pubsf_3way_also.npz \
  --pairs simcode_${N}:pub_simflux simcode_${N}:pub_smlm pub_simflux:pub_smlm > $O/logs/nra_matched_3way_$N.log 2>&1 &
wait; echo "methods $N done"
