# Archive (2026-11-15)

Code, scripts and documents removed from the tree in the 2026-11-15 cleanup (decisions: `docs/CLEANUP_CANDIDATES.md` and
`docs/CLEANUP_LOG.md`). Nothing is lost: the git tag `archive-2026-11-15` is the state just before the removal.

```bash
git show archive-2026-11-15:<path>                       # one file
git checkout archive-2026-11-15 -- <path or directory>    # back into the tree
git ls-tree -r --name-only archive-2026-11-15 <dir>      # what was there
```

Mentions in the documents written `archive-2026-11-15:<path>` refer to these files. Two scripts that a kept script needed
moved instead of being archived: `analysis/nanoruler/mkmodel.py` -> `analysis/p2generic/mkmodel.py`, the SIMCODE-setup part
of `analysis/nanoruler/ev_p12sim.py` -> `analysis/p2generic/pass12_sim.py`. Output data (`out/`) and model weights are not part
of this; their clean-up is `analysis/maintenance/cleanup_outputs.py` (dry run by default).

## Archived (284 files: results recorded, code not needed)

- `analysis/baselines/` (6): compare.py, decode/decode_infer.py, decode/decode_train.py, run_classic.py, run_simcode_real.py, run_smlm_tuned.py
- `analysis/dme/` (1): dme_splithalf.py
- `analysis/evalproto/` (15): blockphase.py, covpool.py, datasets.py, envmap.py, example_real05.json, example_real20.json, example_sim.json, filt.py, frc_published.py, nnlag.py, realproto.py, spatial.py, valid.py, zcal.py, zcal_fix.py
- `analysis/figures/` (28): crlb_all_v2.py, crlb_vs_simcode.py, dme_full05.py, field/check1_offsets.py, field/check1_summary.py, field/check3_pattern.py, field/field_maps.py, field/plot_field_offsets.py, fig_gap.py, frc_common.py, frc_commonN.py, frc_plot.py, hdzoom.py, icrlb.py, icrlb_report.py, lib20.py, lm_gate.py, lm_icrlb.py, lm_robust.py, lmref.py, plot_crlb_sc3.py, recon_full_dme_cross.py, render_matched20.py, render_strict20.py, run_crlb_refs.sh, scorecard.py, sf_gate.py, smoke_metrics.py
- `analysis/fourpi/` (12): chpsf.py, field4pi.py, fo_analyze.py, frameorder.py, inventory.py, reg_xcorr.py, render_run.py, showcase4pi_strip.py, summary_fits.py, syncchk.py, tb_analyze.py, timeblocks.py
- `analysis/gate/` (5): gate_paired.py, merged_d1.py, p2eval.py, run_gate.sh, vs_simcode_p2.py
- `analysis/hd_compare/` (4): calib_b22/calib_p2L_movie_b22.json, calib_b22/calib_sc_hd_sc5nbb15_pm_bright_c22.json, render.py, score.py
- `analysis/nanoruler/` (104): abl_all.sh, abl_autoc.py, abl_capture.py, abl_convert.py, abl_driftcheck.py, abl_figs.py, abl_gate.py, abl_lib.py, abl_match.py, abl_methods.sh, abl_nra.py, abl_p12.py, abl_pattern.py, abl_sim.py, abl_solve.py, bl_run.py, bl_tables.py, chain_nr80.sh, ev_accept.py, ev_chain.sh, ev_fitcal.py, ev_flipclass.py, ev_flipsim.sh, ev_flipsteps.py, ev_fwdsim.sh, ev_gates.py, ev_p12.py, ev_revtest.sh, ev_run.py, ev_scenecheck.py, ev_sigcal.py, gain.py, insitu_any.sh, make_calib.py, nr12_chain.sh, nr12_sub.py, nr_blsim_rul.sh, nr_coon.py, nr_fit_insitu2.py, nr_gallery.py, nr_merged.py, nr_psf_heldout.py, nra_convert.py, nra_lib.py, nra_run.py, nra_selftest.py, nrwrap.py, pst_cfree.py, pst_compare.py, pst_fig.py, pst_fit.py, pst_jitter.py, pst_lagcov.py, pst_lib.py, pst_pairs.py, pst_report.py, pst_simcheck.py, pst_summary.py, quick_smlm.py, r24_abl.py, r24_blmerged.py, r24_blsim.py, r24_cut.py, r24_driftcheck.py, r24_fig.py, r24_final.sh, r24_gpu.sh, r24_merged.py, r24_methods.sh, r24_midc.py, r24_p2.sh, r24_rul.py, r24_run.sh, r24_sim.py, r24_simeval.py, r24_tab.py, r24_variants.sh, r24b_ref.py, r24b_run.py, r24b_sfcheck.py, r24b_tables.py, r24r_accept.py, r24r_accept.sh, r24r_chain.sh, r24r_gates.sh, r24r_insitu_off.py, r24r_insitu_off.sh, r24r_offset.py, r24r_p2.sh, r24r_pair031.py, r24r_psf.py, r24r_psf_fig.py, reorder.py, reorder_any.py, run_psf.sh, sim_nr.py, sim_nr_check.py, sim_nr_lib.py, sitesize/ss_common.py, sitesize/ss_fig.py, sitesize/ss_lib.py, sitesize/ss_run.py, sitesize/ss_slow.py, sitesize/ss_tables.py
- `analysis/p2generic/` (3): chain_gen2.sh, cont_gen2.sh, eval_nr.sh
- `analysis/papersim/` (6): add_pvalid.py, compare_v2.py, pvalid.py, pvalid_real.py, s10_figure.py, selection.py
- `analysis/psf3d/` (2): fisher.py, intensity.py
- `analysis/psfshape/` (1): evalgt.py
- `analysis/results/` (78): dme/compare_dmev2_05.log, dme/compare_dmev2_25.log, dme/report_p2_05.log, dme/report_p2_25.log, dme/report_real20.txt, evalproto/blockphase_0.5.txt, evalproto/blockphase_2.5.txt, evalproto/blockphase_20.txt, evalproto/covpool_simm05pd.txt, evalproto/covpool_simm05pe.txt, evalproto/covpool_simm25pd.txt, evalproto/covpool_simm25pe.txt, evalproto/envmap_0.5.json, evalproto/envmap_0.5.txt, evalproto/envmap_2.5.json, evalproto/envmap_2.5.txt, evalproto/envmap_20.json, evalproto/envmap_20.txt, evalproto/filt_real0.5.txt, evalproto/filt_real2.5.txt, evalproto/filt_real20.txt, evalproto/filt_simm05pd.txt, evalproto/filt_simm25pd.txt, evalproto/realproto_0.5.txt, evalproto/realproto_2.5.txt, evalproto/realproto_20.txt, evalproto/spatial_real0.5.txt, evalproto/spatial_real2.5.txt, evalproto/spatial_real20.txt, evalproto/spatial_simm05pd.txt, evalproto/spatial_simm05pe.txt, evalproto/spatial_simm25pd.txt, evalproto/spatial_simm25pe.txt, evalproto/zcal_fit_m05pd.json, evalproto/zcal_fit_m05pe.json, evalproto/zcal_fit_m25pd.json, evalproto/zcal_fit_m25pe.json, evalproto/zcal_real2.5_m25pe.txt, evalproto/zcal_real20_m25pe.txt, hd_compare/topn_vs_sc_m20qh.txt, hd_compare/topn_vs_sc_m25qh.txt, psfshape/evalgt_bc.txt, psfshape/evalgt_pd.txt, realism/validate_0.5nM_final.json, realism/validate_0.5nM_final.txt, realism/validate_2.5nM_final.json, realism/validate_2.5nM_final.txt, simcode_base2/cmp_m05q_b10.txt, simcode_base2/cmp_m05q_b22.txt, simcode_base2/cmp_m05qh_b10.txt, simcode_base2/cmp_m05qh_b22.txt, simcode_base2/cmp_m25q_b10.txt, simcode_base2/cmp_m25q_b22.txt, simcode_base2/cmp_m25qh_b10.txt, simcode_base2/cmp_m25qh_b22.txt, simcode_base2/gate_paired_scode_L64_5_vs_b10.txt, simcode_base2/gate_paired_scode_L64_5_vs_b22.txt, simcode_base2/gate_paired_scode_ni6_3_vs_b22.txt, simcode_base2/gate_paired_scodenf_L64_5_vs_b10.txt, simcode_base2/gate_paired_scodenf_L64_5_vs_b22.txt, simcode_base2/gate_paired_scodenf_L64_5_vs_pm.txt, simcode_base2/gate_paired_scodenf_ni6_3_vs_b10.txt, simcode_base2/gate_paired_scodenf_ni6_3_vs_b22.txt, simcode_base2/gate_paired_scodenf_ni6_3_vs_pm.txt, simcode_base2/summary_autoc.txt, simreal/cci_m05o_est.txt, simreal/cci_m05o_p2L.txt, simreal/cci_m05rb_est.txt, simreal/cci_m05rb_p2L.txt, simreal/cci_m25o_est.txt, simreal/cci_m25o_p2L.txt, simreal/cci_m25r_est.txt, simreal/cci_m25r_p2L.txt, tput/bench_0.5nM.json, tput/bench_0.5nM_contaminated_3.json, tput/bench_2.5nM.json, tput/clean_0.5nM.json, tput/clean_2.5nM.json
- `analysis/simreal/` (5): compare_ci_r.py, eval_movie.sh, merged_gross.py, sim_movie.py, sim_movie20.py
- `analysis/tput/` (4): bench_tput.py, datasets.json, gpu_sampler.sh, summarize.py
- `analysis/z3d/` (3): crlb_z.py, eval_head.py, psf_zscan.py
- `docs/` (1): CLEANUP_PLAN.md
- `simcode/benchmark/` (1): preprint_gate.py
- `simcode/calibration/` (4): README.md, fit_pattern.py, fit_psf_tracks.py, psf_heldout.py
- `simcode/processing/` (1): dedup.py

## Deleted (30 files: dead, superseded or one-off; also in the tag)

- `analysis/figures/` (4): crlb_gate.py, rcc_drift.py, recon_full.py, recon_full_dc.py
- `analysis/fourpi/` (1): real_w4pi.py
- `analysis/psf3d/` (1): export_zstack.py
- `analysis/z3d/` (1): pipe_check.py
- `analysis/zimflux/` (17): camera_jan13.py, crlb_jan13.py, crlb_s0.py, cycle6ms.py, kvec6ms.py, kvec_refine.py, meta.py, pattern_movie.py, period6ms.py, s1_astig.py, s1_common.py, s1_kz.py, s1_pattern.py, s1_post.py, s1_rois.py, s1_zf.py, stats6ms.py
- `simcode/benchmark/` (1): crlb_nuisance.py
- `simcode/calibration/` (2): camera_gain.py, fit_psf_sums.py
- `simcode/processing/` (1): compare_pattern.py
- `simcode/simulate/` (1): emitters.py
- `simcode/util/` (1): neighbours.py
