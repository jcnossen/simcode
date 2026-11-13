#!/bin/bash
# Fig 3 / Fig 4 with the global precision selection: 4-method paper versions and 5-method internal versions.
# usage: bash analysis/selection_global/run_figs.sh [sigma_max nm, default 12]   (tags fig3_matched_sig<S>, fig4_matched_sig<S>)
# The other methods' drift caches are copied from the u045 versions (same inputs); v2 gets its own DME v2 drift.
set -e
cd "$(dirname "$0")/../.."
S=${1:-12}; T=sig$S
F=out/paper2/fig
N="SIMCODE = SIMCODE v2, PSF-matched networks, selection: calibrated precision max(sigma_x, sigma_y) <= $S nm."
NI="INTERNAL (not for the manuscript). SIMCODE v2 = PSF-matched networks, selection max(sigma_x, sigma_y) <= $S nm."
for k in smlm sf decode; do
  cp -n $F/1809_2_fig3_matched_u045_${k}_drift.npy $F/1809_2_fig3_matched_${T}_${k}_drift.npy
  cp -n $F/1809_2_fig3_matched_u045_internal_${k}_drift.npy $F/1809_2_fig3_matched_${T}_internal_${k}_drift.npy
  cp -n $F/2.5nM_2_fig4_matched_u045_${k}_drift.npy $F/2.5nM_2_fig4_matched_${T}_${k}_drift.npy
  cp -n $F/2.5nM_2_fig4_matched_u045_${k}_drift.npy $F/2.5nM_2_fig4_matched_${T}_internal_${k}_drift.npy
done
cp -n $F/1809_2_fig3_matched_u045_internal_v1_drift.npy $F/1809_2_fig3_matched_${T}_internal_v1_drift.npy
cp -n $F/2.5nM_2_fig4_internal_v1_drift.npy $F/2.5nM_2_fig4_matched_${T}_internal_v1_drift.npy
python analysis/paper/fig_exp_new.py 1809_2 --tag fig3_matched_$T --v2 out/paper2/1809_2/v2_matched/locs_p2_$T.npz --note "$N"
cp $F/1809_2_fig3_matched_${T}_v2_drift.npy $F/1809_2_fig3_matched_${T}_internal_v2_drift.npy
python analysis/paper/fig_exp_new.py 1809_2 --tag fig3_matched_${T}_internal --v2 out/paper2/1809_2/v2_matched/locs_p2_$T.npz \
  --v1 out/paper/1809_2/published/simflux_5.5ms_500mW/sfhd-ndi+smlm.hdf5 --note "$NI"
python analysis/paper/fig_exp_new.py 2.5nM_2 --tag fig4_matched_$T --v2 out/paper2/2.5nM_2/v2_matched/locs_p2_$T.npz --note "$N"
cp $F/2.5nM_2_fig4_matched_${T}_v2_drift.npy $F/2.5nM_2_fig4_matched_${T}_internal_v2_drift.npy
python analysis/paper/fig_exp_new.py 2.5nM_2 --tag fig4_matched_${T}_internal --v2 out/paper2/2.5nM_2/v2_matched/locs_p2_$T.npz \
  --v1 out/paper/2.5nM_2/published/sfhd-ndi+smlm.hdf5 --note "$NI"
echo FIGS DONE
