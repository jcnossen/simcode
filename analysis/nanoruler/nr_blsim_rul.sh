# Merged-event calibration (2026-11-13 evening): re-implemented SMLM / SIMFLUX tables of the linear 20 nm simulation s20_a (r24_blsim.py) -> GT-drift tables + site-pair analysis on the truth pairs (per_pair npz for nr_merged.py).
source /home/jc/simcode2/runs/psfmatch/env.sh; export R24_GPUS=1 R24_JOBS=6
O=out/nanoruler/nr20; A=analysis/nanoruler; mkdir -p $O/merged/tabs
for b in smlm simflux; do [ -f $O/merged/tabs/bl_s20_a_$b.npz ] || bash $A/r24_gpu.sh 3 python $A/r24_tab.py table $O/big/blsim/s20_a_$b.npz $O/merged/tabs/bl_s20_a_$b.npz --ncyc 8000 --gt $O/sim/s20_a; done
[ -f $O/merged/rul_bl_s20_a_truth.json ] || python $A/r24_rul.py run --pairs $O/simeval/pairs_truth_s20.json --d0 20 --out $O/merged --tag bl_s20_a_truth --methods bl_smlm=$O/merged/tabs/bl_s20_a_smlm.npz bl_simflux=$O/merged/tabs/bl_s20_a_simflux.npz --levels
echo done
