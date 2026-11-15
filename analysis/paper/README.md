# Paper figures (experimental data)

- `fig_exp_new.py`: the experimental figure. It covers the 3 movies (1809/2, 2.5 nM/2, 20 nM/2) and the 4 reprocessed methods (SMLM, SIMFLUX, DECODE, SIMCODE v2), with the fair FRC.
- `frc_fair.py`: the fair FRC protocol below. Its module docstring has the API.
- `paper_common.py`: zoom boxes, matplotlib settings and the RSP computation.
- `si_pattern.py`: SI figures of the v2 pattern estimation of one run (default 1809_2): `si_phase_decouple` (per-step phases vs time = shared per-angle drift + fixed per-step offsets, with the per-block shared-drift test) and `si_pattern_plots` (preprint layout: SIMFLUX / SIMCODE rows x 1809_2 / 2.5nM_2 columns, normalized intensity vs pattern phase, included vs excluded) and `si_pattern_plots_extended`. CPU only.
- `si_dme_drift.py`: SI figure `si_dme_drift`: each method's own DME v2 drift in the preprint layout (the frc_fair drift caches; paths as arguments; --style extended adds difference panels).
- `si_single_frame.py`: SI reviewer figure `si_review-single-frame`: one raw 1809/2 frame in photo-electrons (v2 camera calibration) with an inset of an isolated spot picked with the v2 table, 2 um scale bar, PSF-v2 FWHM in the caption file; `_locs` variant with the selected v2 localizations. CPU only.
- `fig_pipeline_v2.py`: the main-text pipeline schematic (vector) -> `out/paper_figs_v2/fig_pipeline_v2.{pdf,svg,png}` + `_caption.txt`.
- `fig_network_structure.py`: the SI network-architecture figure (layer sizes and parameter counts read from the m1809 checkpoints, CPU) -> `out/paper_figs_v2/si_network_structure_v3.{pdf,svg,png}` + `_caption.txt`.

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
6. **FRC.** Alternating 300-frame halves (`window start // 300 % 2`), 5-nm histogram renders, 5-bin smoothing, 1/7 threshold. This equals `simcode.evaluation.metrics.frc_std`; the code asserts it.
7. **Counts.** `n_frc` (the FRC set) is reported next to the FRC, together with `n_disjoint` and `n_all`.
   - Secondary numbers: the FRC without the sigma filter, and with the paper's 30-frame halves.

All figure panels show that same set:

- renders and zooms
- counts
- RSP (Pearson correlation with the widefield mean image; the paper's `compute_pc`)

The zoom AOIs are:

- 1809/2 and 2.5 nM: the paper's boxes
- 20 nM: `analysis/figures/zoompick.py` crossings, picked on the union of the 4 methods' sets
