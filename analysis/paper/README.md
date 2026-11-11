# Paper figures (experimental data)

- `fig_exp_new.py`: the experimental figure. It covers the 3 movies (1809/2, 2.5 nM/2, 20 nM/2) and the 4 reprocessed methods (SMLM, SIMFLUX, DECODE, SIMCODE v2), with the fair FRC.
- `frc_fair.py`: the fair FRC protocol below. Its module docstring has the API.
- `paper_common.py`: zoom boxes, matplotlib settings and the RSP computation.

## Fair FRC protocol (`frc_fair.py`), identical for every method

1. **Rows.** Each method's full output:
   - SMLM: the chi²-filtered `smlm.npy`
   - SIMFLUX: `sf.hdf5`
   - DECODE: all rows
   - SIMCODE v2: pass-2 `sel`
2. **Disjoint windows only.** Every method localizes 6-frame windows (SMLM / SIMFLUX at every frame, DECODE / v2 every 3 frames). Only window starts `f % 6 == 0` are kept, so no photon is counted twice and all methods sample the same windows.
3. **Uncertainty filter.** Keep `max(sigma_x, sigma_y) < 0.15 px` (the paper's `crlb_filter(0.15)`), using each method's own sigma:
   - SMLM / SIMFLUX: CRLB
   - DECODE: predicted sigma
   - v2: calibrated sigma
4. **Common region.** Keep x, y in [5, 115) px.
5. **Drift.** Each method's own DME v2 drift, estimated on its rows from steps 1–4. Time = window start + 3.
6. **FRC.** Alternating 300-frame halves (`window start // 300 % 2`), 5-nm histogram renders, 5-bin smoothing, 1/7 threshold. This equals `simcode2.evalproto.lib.frc_std`; the code asserts it.
7. **Counts.** `n_frc` (the FRC set) is reported next to the FRC, together with `n_disjoint` and `n_all`.
   - Secondary numbers: the FRC without the sigma filter, and with the paper's 30-frame halves.

All figure panels show that same set:

- renders and zooms
- counts
- RSP (Pearson correlation with the widefield mean image; the paper's `compute_pc`)

The zoom AOIs are:

- 1809/2 and 2.5 nM: the paper's boxes
- 20 nM: `analysis/figures/zoompick.py` crossings, picked on the union of the 4 methods' sets
