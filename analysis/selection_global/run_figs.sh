#!/bin/bash
# Fig 3 / Fig 4 with the global precision selection (sigma_max 12 nm): 4-method paper versions and 5-method internal versions.
# The other methods' drift caches were copied from the u045 versions (same inputs); v2 gets its own DME v2 drift.
set -e
cd "$(dirname "$0")/../.."
F=out/paper2/fig
N="SIMCODE = SIMCODE v2, PSF-matched networks, selection: calibrated precision max(sigma_x, sigma_y) <= 12 nm."
NI="INTERNAL (not for the manuscript). SIMCODE v2 = PSF-matched networks, selection max(sigma_x, sigma_y) <= 12 nm."
python analysis/paper/fig_exp_new.py 1809_2 --tag fig3_matched_sig12 --v2 out/paper2/1809_2/v2_matched/locs_p2_sig12.npz --note "$N"
cp $F/1809_2_fig3_matched_sig12_v2_drift.npy $F/1809_2_fig3_matched_sig12_internal_v2_drift.npy
python analysis/paper/fig_exp_new.py 1809_2 --tag fig3_matched_sig12_internal --v2 out/paper2/1809_2/v2_matched/locs_p2_sig12.npz \
  --v1 out/paper/1809_2/published/simflux_5.5ms_500mW/sfhd-ndi+smlm.hdf5 --note "$NI"
python analysis/paper/fig_exp_new.py 2.5nM_2 --tag fig4_matched_sig12 --v2 out/paper2/2.5nM_2/v2_matched/locs_p2_sig12.npz --note "$N"
cp $F/2.5nM_2_fig4_matched_sig12_v2_drift.npy $F/2.5nM_2_fig4_matched_sig12_internal_v2_drift.npy
python analysis/paper/fig_exp_new.py 2.5nM_2 --tag fig4_matched_sig12_internal --v2 out/paper2/2.5nM_2/v2_matched/locs_p2_sig12.npz \
  --v1 out/paper/2.5nM_2/published/sfhd-ndi+smlm.hdf5 --note "$NI"
echo FIGS DONE
