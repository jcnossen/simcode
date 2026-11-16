# Cleanup candidates (2026-11-15)

The model files and modules of this list were renamed on 2026-11-16 (docs/LAYOUT.md "Renamed (2026-11-16)"); the entries
below use the new names. Names of out/ directories and of the analysis/results snapshots are the names on disk.

**Decided and done (user, 2026-11-15).** Code, docs and git-tracked files: done on branch `rename` (archived files are in
the git tag `archive-2026-11-15`, index `docs/ARCHIVE.md`; log `docs/CLEANUP_LOG.md`, section "2026-11-15: processing options
and clean-up"). out/ and the models directory: `python analysis/maintenance/cleanup_outputs.py` (dry run; `--apply` after the
merge). The decisions on top of the categories below:

- zimflux: code and outputs deleted (raw data stay)
- temporary / development outputs (category delete): deleted
- nanoruler: raw data kept; code and outputs archived (not deleted); `mkmodel.py` and the SIMCODE part of `ev_p12sim.py`
  moved to `analysis/p2generic/` (`mkmodel.py`, `pass12_sim.py`), which still needs them
- tput: archived
- uncertain items: `selection_global/run_papersim.sh` keep; compare / hd_compare / simreal / evalproto archived together
  (the simreal items of category archive; the kept simreal files `sim_movie_r.py`, `recipes.sh`, the field files stay);
  `docs/RESULTS.md` 'Reproduce' paths point to the archive tag; the nr80 recipe archived
- the categories below as listed otherwise (keep / archive / delete)

The list as it was written follows (nothing in it is changed).

Candidate list for the clean-up after the package reorganisation. Criterion: needed for
(a) the SIMCODE v2 paper figures and tables (real-data figures 1809_2, 2.5 nM, 20 nM; simulation figures; the CRLB /
ablation / pixel-bias / generic-pass-2 / solve-gap analyses of the paper or SI), (b) the 4Pi-SIMCODE SI figure and the
multi-channel work, (c) the torch-psf integration, (d) reproducing the models (training recipes, calibrations).
A file that a kept file imports, calls or sources is keep too.

- **keep**: needed for (a)-(d)
- **archive**: results worth keeping (cited, recorded in a STATUS / report / docs/RESULTS.md), code or bulk data not needed
- **delete**: dead, superseded, or a one-off whose result is recorded elsewhere

One line per item: `- <category> | <item> | <reason>`. Each analysis section ends with its uncertain cases.
Not in the repository: the paper's solve-gap analysis scripts are in `~/simcode2/analysis/solvegap` (untracked: inputs.py,
report.py, sffit.py, solvegap.py) and should be brought into the repository (keep). `simcode/fourpi/` internals are left for
the 4Pi refactor.

| part | keep | archive | delete |
|---|---|---|---|
| Analysis: paper, papersim, baselines, figures, eva | 42 | 57 | 4 |
| Analysis: studies (pixbias, crlbcmp, ablation_cov, | 45 | 37 | 11 |
| Analysis: nanoruler, fourpi, fourpi_classical, res | 138 | 159 | 14 |
| Package modules | 3 | 5 | 7 |
| Docs | 10 | 2 | 0 |
| Model weights | 5 | 3 | 8 |
| Output directories | 34 | 34 | 112 |
| **total** | **277** | **297** | **156** |

## Analysis: paper, papersim, baselines, figures, evalproto, selection_global, dme

### analysis/baselines
- keep | analysis/baselines/paper_pipeline.py | SMLM/SIMFLUX baselines of the paper rerun; imported by papersim run_baselines, s10, hollow_summary
- keep | analysis/baselines/zpsf.py | PSF spec for all baselines; imported by paper_pipeline, decode_paper, papersim common/psfpanel, fourpi/refit
- keep | analysis/baselines/dme_drift.py | DME v2 drift used by frc_fair (paper FRC) and 4Pi post_w4pi / render_run / post_cl
- keep | analysis/baselines/run_simcode.py | SIMCODE preprint pipeline (v1); imported by papersim/run_simcode_v1 (preprint vs v2 comparison)
- keep | analysis/baselines/decode/decode_paper.py | DECODE baseline train/infer of the paper rerun; used by frc_fair, papersim run_decode(_preprint)
- keep | analysis/baselines/decode/dec_common.py | shared DECODE pieces imported by decode_paper.py
- keep | analysis/baselines/decode/real_stats_paper2.py | makes decode_real_stats.json: DECODE training ranges and gen_hollow conditions (model recipe)
- archive | analysis/baselines/run_simcode_real.py | sign-fixed SIMCODE (b) on real dev movies; evalproto / RESULTS.md numbers; paper uses published hdf5
- archive | analysis/baselines/compare.py | hd/sim like-for-like comparison library (critic 32); RESULTS.md; used by hd_compare, simreal, evalproto (all dev)
- archive | analysis/baselines/run_classic.py | hd_compare Gaussian classical baseline; superseded by paper_pipeline/run_baselines; numbers in RESULTS.md
- archive | analysis/baselines/run_smlm_tuned.py | hd_compare tuned-SMLM baseline (critic 35); numbers in RESULTS.md / hd_compare summary
- archive | analysis/baselines/decode/decode_infer.py | DECODE on hd_compare sim movies (simreal/eval_movie.sh); dev study, superseded by decode_paper
- archive | analysis/baselines/decode/decode_train.py | trains hd_compare-matched DECODE (hd_sum6); dev study, nothing refers to it

### analysis/paper
- keep | analysis/paper/README.md | documents the fair FRC protocol and the experimental figure
- keep | analysis/paper/fig_exp_new.py | paper Fig 3 / Fig 4 / 20 nM figure (1809_2, 2.5nM_2, 20nM_2)
- keep | analysis/paper/frc_fair.py | fair FRC protocol; used by fig_exp_new, papersim metrics, pixbias, tests/test_dataset
- keep | analysis/paper/paper_common.py | zoom boxes, rc params, RSP for fig_exp_new

### analysis/papersim
- keep | analysis/papersim/reproduce.sh | recipe of all paper simulations (density, gauss, hollow, s10, v2)
- keep | analysis/papersim/common.py | shared constants / GT / renderer for every papersim script
- keep | analysis/papersim/run_all.py | per-movie driver called by reproduce.sh
- keep | analysis/papersim/gen_density.py | Fig 2 / SI S1-S2 movie generator
- keep | analysis/papersim/gen_moving.py | SI S8 moving-MT movie generator
- keep | analysis/papersim/gen_hollow.py | hollow-MT movies (reviewer answer); called by reproduce.sh hollow
- keep | analysis/papersim/run_baselines.py | SMLM/SIMFLUX step of run_all
- keep | analysis/papersim/run_decode.py | DECODE step of run_all
- keep | analysis/papersim/run_decode_preprint.py | preprint DECODE model for the gauss conditions (decode_pre step)
- keep | analysis/papersim/run_simcode.py | SIMCODE v2 step of run_all
- keep | analysis/papersim/run_simcode_v1.py | preprint SIMCODE (v1) step of run_all
- keep | analysis/papersim/metrics.py | metrics of the paper sims (FRC, Jaccard); imported widely
- keep | analysis/papersim/gtfrc.py | GT-FRC library (newest paper sim metric), used by gtfrc_run / fig_gtfrc
- keep | analysis/papersim/gtfrc_run.py | gtfrc step of run_all
- keep | analysis/papersim/fig_gtfrc.py | GT-FRC summary + time-to-fidelity figure
- keep | analysis/papersim/fig_density.py | Fig 2 / SI S1 per-density figure + summary table
- keep | analysis/papersim/fig_overview.py | rendered overview of density sims (paper/SI)
- keep | analysis/papersim/psfpanel.py | PSF + CRLB panel imported by fig_overview
- keep | analysis/papersim/moving_fig.py | SI S8 figure
- keep | analysis/papersim/moving_matched.py | matched-count Jaccard for S8 (run_papersim.sh)
- keep | analysis/papersim/make_movie.py | SI moving-MT movie (cited in papersim STATUS)
- keep | analysis/papersim/hollow_analysis.py | hollow step of run_all (reviewer answer)
- keep | analysis/papersim/hollow_summary.py | hollow120k summary figure / numbers (reviewer answer)
- keep | analysis/papersim/s10.py | SI S10 precision vs CRLB data (reproduce.sh s10)
- keep | analysis/papersim/si_crlb_figure.py | paper-ready SI CRLB figure fig:crlb_vs_precision
- keep | analysis/papersim/repro_table.py | preprint vs reproduced vs v2 table (gauss conditions)
- keep | analysis/papersim/table.py | 12-line metrics printer, usage documented in papersim STATUS
- archive | analysis/papersim/s10_figure.py | earlier S10 figure + s10_table md; superseded by si_crlb_figure.py
- archive | analysis/papersim/compare_v2.py | one-off 3D vs 2D v2 comparison; result compare_v2_m1809_3d.md in STATUS
- archive | analysis/papersim/selection.py | selection-strictness study; superseded by global sigma<=12 nm rule (selection_global STATUS)
- archive | analysis/papersim/pvalid.py | p_valid study; results in papersim STATUS; not adopted (hook not made)
- archive | analysis/papersim/add_pvalid.py | p_valid column / revisit check; part of the p_valid study
- archive | analysis/papersim/pvalid_real.py | p_valid on real 1809_2; results in papersim STATUS

### analysis/figures
- keep | analysis/figures/zoompick.py | 20 nM zoom boxes for fig_exp_new; also 4Pi fig_cl / showcase_real_mt
- keep | analysis/figures/skel.py | skeletonize imported by zoompick and 4Pi fig_cl
- keep | analysis/figures/README.md | documents zoompick/skel (needs trimming to the kept scripts)
- archive | analysis/figures/crlb_all_v2.py | dev CRLB figure of the earlier development emitter networks (pretrain_emitter_pass1 and an unshipped variant); superseded by crlbcmp / si_crlb
- archive | analysis/figures/icrlb.py | per-frame I/bg vs CRLB of dev networks; input of crlb_all_v2
- archive | analysis/figures/icrlb_report.py | text report of icrlb npz (dev)
- archive | analysis/figures/lm_gate.py | LM reference lines for crlb_all_v2 (dev)
- archive | analysis/figures/lm_icrlb.py | LM I/bg reference for crlb_all_v2 (dev)
- archive | analysis/figures/sf_gate.py | SIMFLUX-MLE reference for crlb_all_v2 (dev)
- archive | analysis/figures/lmref.py | LM / SF-MLE reference library of the crlb_all_v2 group
- archive | analysis/figures/lm_robust.py | robust companion numbers of crlb_all_v2 (dev)
- archive | analysis/figures/plot_crlb_sc3.py | generic gate CRLB plot, curves() used by crlb_all_v2 group
- archive | analysis/figures/run_crlb_refs.sh | GPU driver for the crlb_all_v2 references (dev)
- archive | analysis/figures/crlb_vs_simcode.py | gate RMSD of dev sc front-ends vs preprint SIMCODE
- delete | analysis/figures/crlb_gate.py | early figure of dev hybrid front-ends pf/det/hd; superseded by crlb_all_v2
- archive | analysis/figures/recon_full_dme_cross.py | 0.5 nM dev-run render (pretrain_emitter_pass1); superseded by fig_exp_new
- archive | analysis/figures/dme_full05.py | DME drifts for recon_full_dme_cross (dev 0.5 nM)
- delete | analysis/figures/recon_full.py | first 0.5 nM render version; superseded by recon_full_dme_cross
- delete | analysis/figures/recon_full_dc.py | RCC-drift version of the 0.5 nM render; superseded by DME version
- delete | analysis/figures/rcc_drift.py | RCC drift only used by recon_full_dc; DME v2 is the drift method
- archive | analysis/figures/smoke_metrics.py | critic-30 smoke metrics; loaders of archived renders / frc_common
- archive | analysis/figures/lib20.py | 20 nM dev-run loaders for hdzoom / render_*20
- archive | analysis/figures/hdzoom.py | 20 nM dev render (dense zooms); superseded by fig_exp_new 20nM_2
- archive | analysis/figures/render_matched20.py | 20 nM dev matched render; cited in RESULTS.md
- archive | analysis/figures/render_strict20.py | 20 nM dev strict render; cited in RESULTS.md
- archive | analysis/figures/fig_gap.py | realgap summary with hard-coded dev numbers; RESULTS.md
- archive | analysis/figures/scorecard.py | sim-realism scorecard of the dev front-end (pretrain_emitter_pass1); cited by acquisition.py docstring
- archive | analysis/figures/frc_common.py | critic-34 FRC sets (dev runs)
- archive | analysis/figures/frc_commonN.py | critic-34 FRC vs N json
- archive | analysis/figures/frc_plot.py | frc_vs_N.png (critic 34); RESULTS.md
- archive | analysis/figures/field/check1_offsets.py | field-dependent offset check (2026-11-10); RESULTS.md
- archive | analysis/figures/field/check1_summary.py | field study summary json
- archive | analysis/figures/field/check3_pattern.py | field study pattern-depth maps
- archive | analysis/figures/field/field_maps.py | field study maps
- archive | analysis/figures/field/plot_field_offsets.py | field_dep_offsets.png; RESULTS.md

### analysis/evalproto
- archive | analysis/evalproto/datasets.py | loader of the dev evaluation protocol (EVALUATION.md / RESULTS.md); paper uses frc_fair
- archive | analysis/evalproto/nnlag.py | identical-row NN vs SIMCODE (b); numbers in RESULTS.md / EVALUATION.md
- archive | analysis/evalproto/blockphase.py | block-phase gap300 FRC; RESULTS.md / EVALUATION.md
- archive | analysis/evalproto/covpool.py | critic-37 coverage-matched pools; RESULTS.md
- archive | analysis/evalproto/frc_published.py | published SIMCODE on covpool protocol (critic 40); RESULTS.md
- archive | analysis/evalproto/realproto.py | real-data GT-free protocol restatement; RESULTS.md
- archive | analysis/evalproto/valid.py | GT validation of GT-free metrics on punctate sims; RESULTS.md
- archive | analysis/evalproto/filt.py | filtered-ours vs SIMCODE at matched N; RESULTS.md
- archive | analysis/evalproto/spatial.py | spatial drivers of the strict product (critic 37); RESULTS.md
- archive | analysis/evalproto/envmap.py | illumination envelope fit; opt-in env_* keys of AcquisitionSim, not used in training
- archive | analysis/evalproto/zcal.py | conditional sigma calibration study (critic 37 #3); RESULTS.md
- archive | analysis/evalproto/zcal_fix.py | out-of-sample sigma-calibration fixes; RESULTS.md
- archive | analysis/evalproto/example_real05.json | example spec of the dev protocol
- archive | analysis/evalproto/example_real20.json | example spec of the dev protocol
- archive | analysis/evalproto/example_sim.json | example spec of the dev protocol

### analysis/selection_global
- keep | analysis/selection_global/run_figs.sh | generates the current paper Fig 3 / Fig 4 (sigma <= 12 nm) via fig_exp_new
- archive | analysis/selection_global/run_papersim.sh | one-off v2sig merge with *_u045main backups; its figure commands belong in reproduce.sh

### analysis/dme
- archive | analysis/dme/dme_splithalf.py | DME v2 split-half numbers of dev runs; RESULTS.md + analysis/results/dme snapshot

### Uncertain cases
- `analysis/selection_global/run_papersim.sh`: archive because it is a migration step (backs up u045main copies), but it is the only script that lists the fig_density / fig_overview / moving_fig / moving_matched / repro_table commands; keep it, or move those commands into reproduce.sh first.
- `analysis/selection_global/run_figs.sh`: kept as the recipe of the current Fig 3/4. It also copies the u045 drift caches and makes the "internal" 5-method versions; it could be replaced by plain fig_exp_new calls.
- `analysis/baselines/compare.py`, `run_classic.py`, `run_smlm_tuned.py`, `decode/decode_infer.py`, `decode/decode_train.py`: archive, assuming hd_compare / simreal / evalproto are archived as well. compare.py is imported by hd_compare/score.py, simreal/compare_ci_r.py, simreal/merged_gross.py and evalproto valid/filt. If any of those is kept, compare.py (and run_simcode.py's gt_mp used by run_classic) must stay.
- `analysis/baselines/run_simcode_real.py`: archive. The paper figures use the published v1 hdf5. If the SI quotes a sign-fixed SIMCODE (b) on real data, it is keep.
- `analysis/papersim/s10_figure.py`: superseded as a figure by si_crlb_figure.py, but it also writes `s10_table_<model>.md`, which papersim STATUS cites. Archive, or delete if the table is not needed.
- p_valid / selection (`pvalid.py`, `add_pvalid.py`, `pvalid_real.py`, `selection.py`): archive, as long as the paper does not discuss p_valid or the selection-strictness analysis. selection.py may back an SI argument for the 12 nm rule; the selection_global sweep is the stated evidence.
- `analysis/papersim/compare_v2.py`: archive as a one-off. If the 3D-model reprocessing is to be compared again, keep it.
- figures/ CRLB group (`crlb_all_v2` and refs): judged dev-era (earlier development emitter networks, one of them not shipped). The paper's CRLB analysis is si_crlb_figure plus analysis/crlbcmp. Keep it if the SI still uses crlb_all_v2.png.
- `recon_full.py`, `recon_full_dc.py`, `rcc_drift.py`, `crlb_gate.py`: deleted as superseded versions. RESULTS.md cites `recon_full*` and README lists their PNGs, so archive is the safer choice.
- `analysis/dme/dme_splithalf.py`: archive. If the revised SI gets a DME drift-validation figure (PAPER_HANDOFF: "probably new drift figures"), it is keep.
- `analysis/figures/README.md`: kept for zoompick/skel, but most of its content describes archived scripts and needs trimming.
- refs.json basename collisions: run_simcode.py (baselines vs papersim), table.py (papersim vs simcode/table.py), valid.py, metrics.py, zcal.py (evalproto vs fourpi/zcal.py). The real import edges were checked with grep rather than taken from refs.json.

## Analysis: studies (pixbias, crlbcmp, ablation_cov, gate, calib, psfmatch, psf3d, psf3d_nets, z3d, p2generic, tput, hd_compare, simreal, psfshape, auto, zimflux)

Evidence: models/README.md, docs/{METHOD,AUTO_PIPELINE,RESULTS,EVALUATION,CLEANUP_LOG,PSF_POLARIZATION}.md, script docstrings,
/tmp/rename_check/refs.json, out/*/STATUS.md, out/HANDOFF_SESSION.md, models/m1809/model.json.
Note: refs.json matches by basename, so short names (render.py, report.py, real.py, sim.py, summary.py, score.py, gate_dump.py)
pick up many false references (e.g. analysis/fourpi's own render.py, simcode.processing.gate_dump). The import lists were checked
directly. Every `from simcode...` import in these files resolves on this branch.
The paper's solve-gap analysis is in /home/jc/simcode2/analysis/solvegap (untracked: inputs.py, report.py, sffit.py, solvegap.py;
out/solvegap/STATUS.md). It is not part of this repository and is not classified here. It should be brought into the repo (keep).

### pixbias (paper pixel-bias analysis; out/pixbias/STATUS.md)
- keep | analysis/pixbias/pblib.py | shared library of the pixel-bias study, imported by edge/frc_corr/real/sim
- keep | analysis/pixbias/real.py | paper pixel-locking on 1809_2 / 2.5nM_2 / 20nM_2 tables (fig_real_*)
- keep | analysis/pixbias/sim.py | GT bias curves on papersim D1/Dexp1 + gate scenes; imported by edge.py
- keep | analysis/pixbias/edge.py | pixel-edge deficit figure of the pixel-bias study (fig_edge_*)
- keep | analysis/pixbias/frc_corr.py | FRC with / without CDF equalisation on the paper table (fig_frc_corr_*)
- keep | analysis/pixbias/summ.py | compact tables of sim_*.json behind the STATUS numbers; tiny, part of the study set

### crlbcmp (paper CRLB analysis; out/crlbcmp/STATUS.md)
- keep | analysis/crlbcmp/prep.py | stage 1: joins m1809 dumps with CRLB tables, offline solve, matching
- keep | analysis/crlbcmp/crlb_free.py | free-model / two-stage CRLB dumps (the key pattern-free bound)
- keep | analysis/crlbcmp/run_crlb.sh | driver of crlb_free.py over scenes/seeds
- keep | analysis/crlbcmp/analyze.py | stage 2 -> crlbcmp.json (efficiency vs CRLB)
- keep | analysis/crlbcmp/report.py | stage 3 tables.md + fig_prior/fig_solve/fig_gap_iso

### ablation_cov (paper covariance / excess-variance ablation; out/ablation_cov/REPORT.md)
- keep | analysis/ablation_cov/variants.py | solve-variant definitions; imported by gate_abl, sigma_check, tables, nanoruler abl_*
- keep | analysis/ablation_cov/capture.py | captures pass-2 head outputs with Cholesky factor for offline variants
- keep | analysis/ablation_cov/gate_dump.py | extra gate dumps (mismatch overrides) for the ablation; simcode.benchmark.gate_scenes POVER/SOVER hook
- keep | analysis/ablation_cov/run_dumps.sh | driver of gate_dump.py (control / recovery / dense scenes)
- keep | analysis/ablation_cov/cov_desc.py | part 1: covariance structure, gate + real 1809_2
- keep | analysis/ablation_cov/cov_fig.py | part 1 figure fig_covariance.png
- keep | analysis/ablation_cov/gate_abl.py | part 2a: variants on gate scenes with refitted calibrations
- keep | analysis/ablation_cov/autoc_est.py | automatic c-term estimate on real paper runs + sims
- keep | analysis/ablation_cov/run_autoc.sh | driver of autoc_est.py
- keep | analysis/ablation_cov/sigma_check.py | part 4: c term vs reported uncertainty
- keep | analysis/ablation_cov/tables.py | REPORT.md tables; tests/test_tables.py and nanoruler scripts use it

### gate (development gate / SIMCODE-vs-ours comparisons on the old pretrain_emitter_pass1 head; RESULTS.md §2, analysis/results/gate)
- archive | analysis/gate/run_gate.sh | gate driver tied to old pretrain_emitter_pass1 head; iso 1.17 recorded in RESULTS.md
- archive | analysis/gate/gate_paired.py | paired preprint-vs-ours gate (critic 32); numbers in RESULTS.md / analysis/results
- archive | analysis/gate/merged_d1.py | critic-34 merged sub-pixel quantification; snapshot analysis/results/gate/merged_d1_vs_simcode.txt
- archive | analysis/gate/vs_simcode_p2.py | critic-34 repaired SIMCODE on pass-2 gates; recorded in RESULTS.md
- archive | analysis/gate/p2eval.py | helpers of merged_d1 / vs_simcode_p2 only (development calibs in out/pass2_gate)

### calib (solve-calibration library and the template-calibration fit)
- keep | analysis/calib/plib.py | pass-2 dump eval + fit_cal; used by fit_cal_set, acceptance docs, z3d, crlbcmp, ablation
- keep | analysis/calib/dump_fe.py | produced the pretrain_emitter_pass1 template σ tables (METHOD.md); templates of every model calibration
- keep | analysis/calib/blib.py | library of fit_cal.py (pass-1 template fit)
- keep | analysis/calib/fit_cal.py | fitted the calib/template_calib_emitter_pass1_* templates (cited in their json provenance, METHOD.md)

### psfmatch
- keep | analysis/psfmatch/fit_cal_set.py | model-calibration step of models/README.md recipe and simcode/pipeline/training.py
- keep | analysis/psfmatch/prod_m20_m25.sh | 20nM_2 paper production with m25 (v2_m25, used by pixbias/real figures)

### psf3d (in-situ 3D PSF)
- keep | analysis/psf3d/make_calib.py | writes calib_<movie>_psf3d(_avg).json, the paper PSFs; calibration README, AUTO_PIPELINE
- keep | analysis/psf3d/summary.py | regenerates the PSF_POLARIZATION.md tables (model sequence, held-out evidence)
- archive | analysis/psf3d/intensity.py | averaged-vs-per-axis intensity study; results out/psf3d/intensity, PSF_POLARIZATION.md
- archive | analysis/psf3d/fisher.py | one-off profiled Fisher information of PSF globals; nothing calls it
- delete | analysis/psf3d/export_zstack.py | z-stack export for external spline pipelines; no consumer (torch_psf / DEC_CALIB cover it)

### psf3d_nets
- keep | analysis/psf3d_nets/sigcal.py | sig_u sigma rescale fit: models/README.md recipe, training.py, table.py --sig_u

### z3d (3D emitter network; out/z3d/STATUS.md)
- keep | analysis/z3d/accept_z.py | 3D-vs-predecessor gate acceptance; called by out/z3d/runs/eval.sh for model swaps
- keep | analysis/z3d/swap_model.py | model takeover/export (m1809 swapped with it; m25/g13 pending); in model.json provenance
- archive | analysis/z3d/eval_head.py | smoke z evaluation of 3D heads; numbers in out/z3d/STATUS.md
- archive | analysis/z3d/crlb_z.py | z CRLB design check (37-64 nm) recorded in z3d STATUS
- archive | analysis/z3d/psf_zscan.py | PSF-vs-height design check; number cited in simcode/simulate/batches.py comment
- delete | analysis/z3d/pipe_check.py | one-off code-path smoke test of the 3D head (tests/test_fast_paths covers it)

### p2generic (paper generic-pass-2 analysis; out/p2generic/STATUS.md)
- keep | analysis/p2generic/chain_gen.sh | trained g1809 pass-2 detector/heads of shipped m1809 (model.json provenance)
- keep | analysis/p2generic/cont_det.sh | trained dg2s = m1809's pass-2 detector (model.json provenance)
- keep | analysis/p2generic/pwrap.py | preset-patching wrapper used by chain_gen, cont_det, eval_gen (realpat / OOD presets)
- keep | analysis/p2generic/eval_gen.sh | generic-pass-2 evaluation (standard, real-pattern, OOD gates, m05r3 sim)
- keep | analysis/p2generic/eval_cmp.py | head-to-head gate comparison used by eval_gen.sh
- keep | analysis/p2generic/simcmp_ci.py | sim-movie A-vs-B with CI used by eval_gen.sh (c)
- keep | analysis/p2generic/summary.py | one-table summary of the generic-pass-2 evaluation
- delete | analysis/p2generic/fit_cal_gen.py | copy of fit_cal_set.py; fit_cal_set now takes a dump dir (9ff618a); eval_gen.sh call to switch
- delete | analysis/p2generic/fitcal_wrap.py | sig_ph shim superseded by plib.fit_cal fix af74ef4; eval_gen.sh call to switch
- archive | analysis/p2generic/chain_gen2.sh | gnr80 nanoruler generic set (not adopted for real runs); recorded in STATUS
- archive | analysis/p2generic/cont_gen2.sh | v2-test head continuation; not the shipped m1809 pass-2 emitter chain (out/z3d/runs)
- archive | analysis/p2generic/eval_nr.sh | gnr80 evaluation; verdict recorded in out/p2generic/STATUS.md

### tput (throughput; RESULTS.md §4, analysis/results/tput)
- archive | analysis/tput/bench_tput.py | original-vs-fast per-stage benchmark on dev 0.5/2.5 nM data; numbers in RESULTS §4
- archive | analysis/tput/datasets.json | bench_tput inputs (dev movies, old pretrain_emitter_pass1 calibrations)
- archive | analysis/tput/gpu_sampler.sh | GPU-occupancy logger for bench validity; companion of bench_tput
- archive | analysis/tput/summarize.py | tables from the tput outputs; companion of bench_tput

### hd_compare (high-density sim comparison m25qh/m20qh; RESULTS.md §3b)
- archive | analysis/hd_compare/score.py | development high-density sim scoring vs SMLM/SF/DECODE/SIMCODE; recorded in RESULTS §3b
- archive | analysis/hd_compare/render.py | renders of that comparison (paper sim figures are analysis/papersim)
- archive | analysis/hd_compare/calib_b22/calib_p2L_movie_b22.json | b22 calibration input of the RESULTS §3b runs
- archive | analysis/hd_compare/calib_b22/calib_sc_hd_sc5nbb15_pm_bright_c22.json | b22 calibration input of the RESULTS §3b runs

### simreal (benchmark sim movies)
- keep | analysis/simreal/sim_movie_r.py | generates m05r3 / m25rc3: sigcal fit movies, papersim selection/pvalid, p2generic (c)
- keep | analysis/simreal/recipes.sh | the recipes (seeds, knobs) of m05r3 / m25rc3 and the other benchmark movies
- keep | analysis/simreal/fields/field_own10_real05.pt | SIM_PFIELD input of the m05r* recipes
- keep | analysis/simreal/fields/field_own10_real25.pt | SIM_PFIELD input of the m25r* recipes
- archive | analysis/simreal/sim_movie.py | older m05qh/m25qh movies (RESULTS §3b); recipes.sh qh branches call it
- archive | analysis/simreal/sim_movie20.py | m20qh movie (RESULTS §3b); recipes.sh m20qh branch calls it
- archive | analysis/simreal/eval_movie.sh | runs old-default ours + sign-fixed SIMCODE on benchmark movies (EVALUATION.md)
- archive | analysis/simreal/compare_ci_r.py | RESULTS headline bootstrap CIs (critic 33); snapshot in analysis/results
- archive | analysis/simreal/merged_gross.py | critic-34 gross/merged fractions on the 20 nM-like sim; recorded in RESULTS

### psfshape
- archive | analysis/psfshape/evalgt.py | PSF-shape mismatch GT readout (2026-11-11); recorded in RESULTS.md

### auto
- keep | analysis/auto/cmp_calib.py | auto-pipeline / in-situ calibration vs reference comparison (AUTO_PIPELINE.md §9, torch-psf validation)

### zimflux (dropped by the user 2026-11-14; S0/S1 results kept in out/zimflux and simcode_results next to the movie)
- archive | analysis/zimflux/crlb_s0.py | S0 CRLB (astig / zimflux / fact); imported by crlb_jan13, s1_common
- archive | analysis/zimflux/crlb_jan13.py | S0 CRLB on the paper movie; recorded in out/zimflux
- archive | analysis/zimflux/camera_jan13.py | gain/offset of the paper movie camera; results in simcode_results/camera
- archive | analysis/zimflux/s1_common.py | S1 shared models; imported by every s1_* script
- archive | analysis/zimflux/s1_rois.py | S1 step 1 (camera, detection, ROIs); results kept next to the data
- archive | analysis/zimflux/s1_astig.py | S1 step 2 astigmatic fits; results kept next to the data
- archive | analysis/zimflux/s1_pattern.py | S1 step 3 pattern regression; imported by s1_kz, s1_zf
- archive | analysis/zimflux/s1_zf.py | S1 step 4 k_z + ZIMFLUX joint MLE; results kept
- archive | analysis/zimflux/s1_kz.py | S1 k_z from tracks; results kept
- archive | analysis/zimflux/s1_post.py | S1 step 5 drift, filtering, nanoruler clusters; results kept
- delete | analysis/zimflux/pattern_movie.py | inventory exploration of 10 ms movies; results in inventory json / STATUS / DESIGN.md
- delete | analysis/zimflux/meta.py | Micro-Manager metadata inventory; meta.json recorded
- delete | analysis/zimflux/stats6ms.py | quick 6 ms movie stats (superseded gain); recorded in DESIGN.md
- delete | analysis/zimflux/period6ms.py | 6 ms pattern-cycle exploration; recorded in STATUS / DESIGN
- delete | analysis/zimflux/cycle6ms.py | 6 ms 8-frame cycle exploration; recorded in STATUS / DESIGN
- delete | analysis/zimflux/kvec6ms.py | 6 ms wave-vector exploration; recorded in STATUS / DESIGN
- delete | analysis/zimflux/kvec_refine.py | refinement of kvec6ms with placeholder code; one-off, recorded

### Uncertain cases
- **calib/dump_fe.py, blib.py, fit_cal.py:** I kept these as the provenance of the pretrain_emitter_pass1 template calibrations that every model calibration copies from (METHOD.md). The templates are fixed files, so archive would also be defensible if re-fitting the templates is never needed.
- **psfmatch/prod_m20_m25.sh:** this is the only tracked 20nM_2 production driver. It is really one `python -m simcode.run --model m25` command, and the planned 3D reprocessing may use a different driver, so archive is possible.
- **psf3d/summary.py:** kept only to regenerate the PSF_POLARIZATION.md tables. It becomes archive if that doc is frozen.
- **psf3d/export_zstack.py:** delete assumes no baseline needs plain z-stacks. If the DECODE or spline baselines want the in-situ PSF as a stack, keep it.
- **p2generic fit_cal_gen.py / fitcal_wrap.py:** delete only after eval_gen.sh (and eval_nr.sh, if kept) call analysis/psfmatch/fit_cal_set.py directly. Until then eval_gen.sh part (a) depends on them.
- **p2generic chain_gen.sh / cont_det.sh:** the recipe is now written out in models/README.md, so these are provenance (cited by models/m1809/model.json) rather than needed code. Archive is acceptable if that provenance pointer is updated.
- **tput/*:** I archived these. If the revised paper reports a v2 runtime (the preprint has "SIMCODE 23.19 min vs DECODE 3:07"), bench_tput.py would be the tool to re-measure it on the paper data.
- **gate/*, hd_compare/*, simreal compare/eval scripts, psfshape:** I archived these because they back the development headline in docs/RESULTS.md (old pretrain_emitter_pass1 head), not the paper figures (analysis/papersim, paper2). Archiving them breaks RESULTS.md §7 "Reproduce" paths, so that doc needs a note.
- **simreal/recipes.sh (keep) calls sim_movie.py / sim_movie20.py (archive)** for its qh branches, so those branches break if the scripts move.
- **auto/cmp_calib.py:** kept for validating the auto / torch-psf in-situ calibration. It becomes archive if out/torchpsf_validation has superseded it.
- **z3d eval_head.py:** archived, but it may still be wanted for the pending m25 / g13 3D retrains.

## Analysis: nanoruler, fourpi, fourpi_classical, results

Format: `- <category> | <file> | <reason>`. Evidence: refs.json, docstrings, README / docs/LAYOUT.md / docs/CLEANUP_LOG.md,
out/nanoruler/STATUS.md, out/p2generic/STATUS.md, out/HANDOFF_SESSION.md, out/paper/*.md, a static import check of every .py
(only the module `accept` does not resolve).

### analysis/nanoruler (106: keep 24, archive 69, delete 13)

The nanoruler study (nr80 / nr40 / nr20 movies of the original SIMFLUX setup) is not in the paper or SI: out/paper/*.md does not use it
(only the preprint's "m 0.9 from nanorulers"), and docs/LAYOUT.md calls it a development study. Results live in out/nanoruler (STATUS.md, report/,
ablation/, nr40/, nr20/, perstep/, sitesize/). Kept are (1) the model nr80 recipe (training, in-situ calibration, sigma calibration), which
simcode/pipeline/configs/nr80.{py,yaml} and docs/AUTO_PIPELINE.md 9.4 point at, and (2) whatever analysis/p2generic (generic pass 2: gnr80, m1809g)
calls or imports.

#### keep
- keep | analysis/nanoruler/chain_nr80.sh | training recipe of model nr80; chain_gen2.sh (gnr80) builds on it
- keep | analysis/nanoruler/nrwrap.py | AcquisitionSim preset patch for nr80; used by chain_nr80, ev_chain, p2generic chain_gen2.sh / eval_nr.sh
- keep | analysis/nanoruler/make_calib.py | builds calib_nr80.json (nr80 calibration); cited in calibration/README, AUTO_PIPELINE
- keep | analysis/nanoruler/nr_fit_insitu2.py | in-situ PSF fit of the nr80 calibration (run_psf.sh, insitu_any.sh)
- keep | analysis/nanoruler/nr_psf_heldout.py | held-out PSF check in the nr80 / insitu_any calibration recipe
- keep | analysis/nanoruler/run_psf.sh | the in-situ PSF recipe that made calib_nr80
- keep | analysis/nanoruler/insitu_any.sh | in-situ calibration recipe for any movie of the setup; simcode/pipeline/spots.py points to it
- keep | analysis/nanoruler/quick_smlm.py | track extraction / camera-map masks of the calibration; spots.py is this script as a function
- keep | analysis/nanoruler/gain.py | camera gain of the nr80 calibration; cited by pipeline/camera.py and the configs
- keep | analysis/nanoruler/reorder.py | blocked-order cache of sim4_1 (nr80.yaml cache); imported by sim_nr / sim_nr_lib
- keep | analysis/nanoruler/reorder_any.py | reference of simcode/pipeline/frame_order.py (AUTO_PIPELINE 'order' stage)
- keep | analysis/nanoruler/mkmodel.py | one-model json writer; used by p2generic eval_gen.sh / eval_nr.sh and ev_chain.sh
- keep | analysis/nanoruler/ev_run.py | simcode.run with nr80 overrides; p2generic eval_nr.sh, AUTO_PIPELINE 9.4 reference run
- keep | analysis/nanoruler/ev_fitcal.py | sim-sigma calibration fit of nr80; called by p2generic eval_nr.sh / fitcal_wrap.py
- keep | analysis/nanoruler/ev_sigcal.py | sigma rescale of model nr80 (sigu_p*_nr80); imported by ev_flipclass / ev_p12sim
- keep | analysis/nanoruler/sim_nr.py | simulated nanoruler movies nr_a / nr_c0 (sigma calibration, p2generic eval_nr.sh inputs)
- keep | analysis/nanoruler/sim_nr_lib.py | imported by sim_nr.py and ev_sigcal.py
- keep | analysis/nanoruler/ev_p12sim.py | called by p2generic eval_gen.sh (m1809g) and eval_nr.sh
- keep | analysis/nanoruler/ev_flipclass.py | called by p2generic eval_nr.sh (gnr80 verdict, out/p2generic/STATUS.md)
- keep | analysis/nanoruler/abl_lib.py | imported by ev_p12sim.py and ev_sigcal.py (load_table, SIGMA_MAX_NM)
- keep | analysis/nanoruler/nra_lib.py | imported by abl_lib.py
- keep | analysis/nanoruler/ev_chain.sh | nr80 evaluation / calibration chain (calib_sc_hd_nr80, sigu); its step calling ev_accept.py must go
- keep | analysis/nanoruler/ev_gates.py | step 5 of ev_chain.sh (gates + model entry of nr80)
- keep | analysis/nanoruler/ev_scenecheck.py | gate-scene check step of ev_chain.sh

#### archive
- archive | analysis/nanoruler/abl_all.sh | driver of the nr80 solve-variant ablation; results in out/nanoruler/ablation
- archive | analysis/nanoruler/abl_autoc.py | ablation part C (autoc estimate on nr80); results recorded
- archive | analysis/nanoruler/abl_capture.py | ablation: captures pass-2 head outputs for offline solves
- archive | analysis/nanoruler/abl_convert.py | ablation part A table converter (own DME drift)
- archive | analysis/nanoruler/abl_figs.py | ablation report figures
- archive | analysis/nanoruler/abl_gate.py | ablation part B1 per-variant calibration on nr80 gates
- archive | analysis/nanoruler/abl_match.py | ablation part A identical-event matching vs published SIMFLUX (STATUS headline numbers)
- archive | analysis/nanoruler/abl_methods.sh | ablation part A driver (method tables)
- archive | analysis/nanoruler/abl_nra.py | ablation part B3 nanoruler metrics per variant
- archive | analysis/nanoruler/abl_p12.py | ablation part E pass 1 vs pass 2 on identical events
- archive | analysis/nanoruler/abl_pattern.py | ablation part D pattern-model test
- archive | analysis/nanoruler/abl_sim.py | ablation part E on simulated movies vs GT
- archive | analysis/nanoruler/abl_solve.py | ablation part B2 solve variants on a run
- archive | analysis/nanoruler/bl_run.py | classical baselines on nr80 movie (study comparison, results recorded)
- archive | analysis/nanoruler/bl_tables.py | baseline run -> nanoruler tables
- archive | analysis/nanoruler/ev_flipsim.sh | step-order flip sim validation; superseded by generic pass 2, results in STATUS
- archive | analysis/nanoruler/ev_flipsteps.py | step-order flip tool; flip approach dropped (generic pass 2 instead)
- archive | analysis/nanoruler/ev_fwdsim.sh | positive control of the step-order hypothesis; finished, recorded
- archive | analysis/nanoruler/ev_p12.py | pass 2 - pass 1 on gate scenes (step-order test); finished
- archive | analysis/nanoruler/ev_revtest.sh | reversed-step gate test driver; finished, recorded
- archive | analysis/nanoruler/nr12_chain.sh | resume driver (m12 selection tables 80/40/20 nm); results recorded
- archive | analysis/nanoruler/nr12_sub.py | m12 subset of matched events (part of nr12_chain)
- archive | analysis/nanoruler/nr_blsim_rul.sh | merged-event calibration driver
- archive | analysis/nanoruler/nr_coon.py | merged-event co-ON estimator (nr20 result in STATUS)
- archive | analysis/nanoruler/nr_gallery.py | report galleries (out/nanoruler/report)
- archive | analysis/nanoruler/nr_merged.py | merged events on real 20 nm movie (STATUS result)
- archive | analysis/nanoruler/nra_convert.py | converters to the nra table format / published-frame transform
- archive | analysis/nanoruler/nra_run.py | nanoruler precision analysis CLI (sites, metrics)
- archive | analysis/nanoruler/pst_cfree.py | per-step pattern study part 3 (out/nanoruler/perstep)
- archive | analysis/nanoruler/pst_compare.py | per-step pattern study part 4
- archive | analysis/nanoruler/pst_fig.py | per-step pattern study figures
- archive | analysis/nanoruler/pst_fit.py | per-step pattern study part 1 fits
- archive | analysis/nanoruler/pst_jitter.py | per-step pattern study parts 2-3 (noise / jitter)
- archive | analysis/nanoruler/pst_lagcov.py | per-step pattern study lag covariance
- archive | analysis/nanoruler/pst_lib.py | library of the per-step pattern study
- archive | analysis/nanoruler/pst_pairs.py | per-step pattern study model-free variance
- archive | analysis/nanoruler/pst_report.py | per-step pattern study part 1 numbers
- archive | analysis/nanoruler/pst_summary.py | per-step pattern study summary json
- archive | analysis/nanoruler/r24_abl.py | runs abl_ scripts on nr40 / nr20
- archive | analysis/nanoruler/r24_blmerged.py | merged rows in re-implemented baseline (sim)
- archive | analysis/nanoruler/r24_blsim.py | baseline on simulated nanoruler movie
- archive | analysis/nanoruler/r24_cut.py | merged-row cut tables
- archive | analysis/nanoruler/r24_driftcheck.py | held-out DME check, called by archived r24_methods.sh
- archive | analysis/nanoruler/r24_fig.py | R24 figures (site pairs)
- archive | analysis/nanoruler/r24_final.sh | R24 remaining pass-1 steps driver
- archive | analysis/nanoruler/r24_gpu.sh | GPU-lock helper of the archived R24 drivers
- archive | analysis/nanoruler/r24_merged.py | merged-row separability on sim (STATUS 11-13 result)
- archive | analysis/nanoruler/r24_methods.sh | R24 method tables driver (nr40 / nr20)
- archive | analysis/nanoruler/r24_midc.py | merged-row midpoint estimator (did not calibrate, recorded)
- archive | analysis/nanoruler/r24_p2.sh | R24 pass-2 repeat driver (pass 2 not quoted per user decision)
- archive | analysis/nanoruler/r24_rul.py | site-pair analysis 40 / 20 nm (headline 40 nm numbers)
- archive | analysis/nanoruler/r24_run.sh | R24 pipeline run wrapper
- archive | analysis/nanoruler/r24_sim.py | simulated 40 / 20 nm movies (wraps sim_nr.py)
- archive | analysis/nanoruler/r24_simeval.py | R24 sim table vs GT
- archive | analysis/nanoruler/r24_tab.py | R24 table converter (also used by sitesize/ss_run.py)
- archive | analysis/nanoruler/r24_variants.sh | R24 solve-variant driver
- archive | analysis/nanoruler/r24b_ref.py | reference tables of nr40 / nr20 movies
- archive | analysis/nanoruler/r24b_run.py | classical baselines on nr40 / nr20
- archive | analysis/nanoruler/r24b_tables.py | nr40 / nr20 baseline tables
- archive | analysis/nanoruler/r24r_gates.sh | reuse check part C gate scenes (results in nr40/reuse)
- archive | analysis/nanoruler/r24r_pair031.py | reuse check 0.31 px pair analysis (result recorded)
- archive | analysis/nanoruler/r24r_psf.py | reuse check: nr40 / nr20 PSF vs nr80
- archive | analysis/nanoruler/r24r_psf_fig.py | figure of r24r_psf.py
- archive | analysis/nanoruler/sitesize/ss_common.py | site-size study common-mode part (out/nanoruler/sitesize)
- archive | analysis/nanoruler/sitesize/ss_fig.py | site-size study figure
- archive | analysis/nanoruler/sitesize/ss_lib.py | site-size study library
- archive | analysis/nanoruler/sitesize/ss_run.py | site-size study step 2
- archive | analysis/nanoruler/sitesize/ss_slow.py | site-size study slow-component diagnosis
- archive | analysis/nanoruler/sitesize/ss_tables.py | site-size study step 1 tables

#### delete
- delete | analysis/nanoruler/ev_accept.py | imports removed module `accept` (analysis/psf3d_nets/accept.py, deleted in 557c399)
- delete | analysis/nanoruler/r24r_accept.py | imports removed module `accept`; broken
- delete | analysis/nanoruler/r24r_accept.sh | driver whose main step (r24r_accept.py) is broken; finished one-off
- delete | analysis/nanoruler/r24r_chain.sh | 9-line driver of r24r_gates + broken r24r_accept.sh
- delete | analysis/nanoruler/r24r_p2.sh | pass-2 repeat of the broken r24r_accept.sh
- delete | analysis/nanoruler/abl_driftcheck.py | one-off held-out drift check; nothing calls it, result in STATUS
- delete | analysis/nanoruler/r24b_sfcheck.py | one-off SIMFLUX-SMLM offset check; no references
- delete | analysis/nanoruler/nra_selftest.py | synthetic self-test of nra_lib; no references
- delete | analysis/nanoruler/sim_nr_check.py | one-off validation of the simulated movie; no references
- delete | analysis/nanoruler/r24r_offset.py | one-off camera offset 99 vs 100 ADU check; no references
- delete | analysis/nanoruler/r24r_insitu_off.py | one-off offset-sensitivity refit; only its own .sh calls it
- delete | analysis/nanoruler/r24r_insitu_off.sh | driver of the one-off offset refit
- delete | analysis/nanoruler/pst_simcheck.py | semi-synthetic check of pst_lagcov; no references

### analysis/fourpi (55: keep 42, archive 12, delete 1)

#### keep
- keep | analysis/fourpi/bench_classical.py | pixel-MLE reference of the v10 isolated-emitter gate (CLEANUP_LOG keep list)
- keep | analysis/fourpi/camera.py | 4Pi calibration chain: smlm/camera/camera_<cell>.npz read by simcode.fourpi.front
- keep | analysis/fourpi/cmp_full_w4pi.py | README 4Pi entry point: whole-movie network vs classical comparison
- keep | analysis/fourpi/cmp_real4pi.py | imported by post_w4pi.py (pairs)
- keep | analysis/fourpi/dafl4pi.py | imported by fourpi_classical/fig_cl.py (adjacent-window precision)
- keep | analysis/fourpi/dc.py | 4Pi raw I/O module imported by the whole calibration chain
- keep | analysis/fourpi/diag_4pi.py | calibration step 5: channel / 4Pi phase calibration, angle phase difference (delta_deg source?)
- keep | analysis/fourpi/diag_pattern.py | writes smlm/diag/pattern_<cell>.json read by simcode.fourpi.run and real_cl.py
- keep | analysis/fourpi/fit_psf4pi.py | makes psf4pi/NPC_f_dxy15.json, the 4Pi PSF every kept script loads
- keep | analysis/fourpi/fourpi.py | imported by diag_4pi.py (per-emitter 4Pi phase)
- keep | analysis/fourpi/gatek4pi.py | isolated-emitter gate of head v10 (CLEANUP_LOG G4, LAYOUT "v10 gates")
- keep | analysis/fourpi/gate_movies_w.sh | movie gate of the v10 head (CLEANUP_LOG: kept)
- keep | analysis/fourpi/mle.py | Poisson MLE imported by process / refit / register / zcal
- keep | analysis/fourpi/pattern.py | imported by diag_pattern / diag_4pi (pattern step of the chain)
- keep | analysis/fourpi/post_w4pi.py | README 4Pi entry point: post-processing of network tables
- keep | analysis/fourpi/process.py | calibration chain pass 1 (input of refit -> diag_pattern)
- keep | analysis/fourpi/psffit.py | summed in-situ PSF on torch_psf, imported by run_psf.py (torch-psf integration)
- keep | analysis/fourpi/quickloc.py | quick photometry imported by process / register / zcal
- keep | analysis/fourpi/refit.py | calibration step 4: p2 tables that diag_pattern reads
- keep | analysis/fourpi/register.py | channel registration (PolyMap / reg_insitu); reference for N-channel registration work
- keep | analysis/fourpi/reg_npc_shift.py | wrote smlm/reg/reg_insitu_NPC.json (read by simcode.fourpi.front)
- keep | analysis/fourpi/render.py | imported by diag_pattern / diag_4pi (DME drift + renders)
- keep | analysis/fourpi/run_full_w4pi.sh | README 4Pi entry point: whole-movie network run
- keep | analysis/fourpi/run_psf.py | summed-channel PSF fit (bead + in-situ) used by refit
- keep | analysis/fourpi/run_reg_insitu.py | wrote smlm/reg/reg_insitu_MT.json (read by simcode.fourpi.front)
- keep | analysis/fourpi/select_curve_mt.py | README 4Pi entry point: MT selection curves (SI)
- keep | analysis/fourpi/select_curve_npc.py | README 4Pi entry point: NPC selection curves (SI)
- keep | analysis/fourpi/showcase4pi_eval.py | simulated showcase GT evaluation (LAYOUT "simulated showcase")
- keep | analysis/fourpi/showcase4pi_fig.py | showcase figures; layout reused by showcase_real_mt
- keep | analysis/fourpi/showcase4pi_movie.py | showcase movie; imported by the new SI sweep (fourpi_sim/sweep4pi_movie.py)
- keep | analysis/fourpi/showcase4pi_movies.sh | driver of the showcase movies
- keep | analysis/fourpi/showcase4pi_run.sh | driver of all methods on the showcase movies
- keep | analysis/fourpi/showcase4pi_table.py | showcase tables (out/4pi/showcase/tables.md)
- keep | analysis/fourpi/showcase_real_mt.py | README 4Pi entry point: real MT SI figure
- keep | analysis/fourpi/simeval4pi.py | GT evaluation; imported by sim_cl, showcase, fourpi_sim sweep
- keep | analysis/fourpi/simfront4pi.py | simulated front end; used by showcase, gates, fourpi_sim sweep
- keep | analysis/fourpi/simlin4pi.py | head + solve on simulated movies; showcase, gates, fourpi_sim sweep
- keep | analysis/fourpi/simmovie4pi.py | 4Pi movie simulator imported by all sim scripts
- keep | analysis/fourpi/train_det4pi.py | trainer of det4pi (models/README.md)
- keep | analysis/fourpi/train_head4pi.py | trainer of v3L, the frozen prior inside v10 (CLEANUP_LOG keep)
- keep | analysis/fourpi/train_head4pi_w.py | trainer of head v10
- keep | analysis/fourpi/zcal.py | bead z-stack ROIs (zstack_rois) that fit_psf4pi.py fits

#### archive
- archive | analysis/fourpi/chpsf.py | empirical per-channel PSF diagnostic; output read by nothing (smlm/diag/chpsf_*)
- archive | analysis/fourpi/field4pi.py | field-dependence study (out/4pi/FIELD_DEPENDENCE.md); no caller
- archive | analysis/fourpi/fo_analyze.py | frame-order sync analysis; result in out/4pi/SMLM_REPORT.md
- archive | analysis/fourpi/frameorder.py | frame-order sync check of all files; finished (SMLM_REPORT)
- archive | analysis/fourpi/inventory.py | file inventory / sync check; finished (out/4pi/inventory)
- archive | analysis/fourpi/reg_xcorr.py | alternative tile-xcorr registration refinement; not in the chain (maybe useful for multi-channel)
- archive | analysis/fourpi/render_run.py | classical SMLM step-5a renders / drift; superseded by post_w4pi / post_cl
- archive | analysis/fourpi/showcase4pi_strip.py | disk-budget helper stripping patch arrays; optional
- archive | analysis/fourpi/summary_fits.py | pass-1 / refit summary diagnostics (smlm/diag/fits_*)
- archive | analysis/fourpi/syncchk.py | within-file step-sync check; finished (~100k-frame event explained)
- archive | analysis/fourpi/tb_analyze.py | pattern / 4Pi phase vs time analysis; finished (SMLM_REPORT)
- archive | analysis/fourpi/timeblocks.py | sampled-block pattern tracking; finished (SMLM_REPORT)

#### delete
- delete | analysis/fourpi/real_w4pi.py | 12-line compatibility wrapper of simcode.fourpi.run; alias removal approved 2026-11-14

### analysis/fourpi_classical (10: keep 10)

- keep | analysis/fourpi_classical/__init__.py | package marker; referenced by tests/check_imports.py, util/drift.py, util/runconfig.py
- keep | analysis/fourpi_classical/cl4pi.py | classical 4Pi-SIMFLUX fitter imported by full_cl / real_cl / sim_cl / mov4pi_cl / val_iso
- keep | analysis/fourpi_classical/fig_cl.py | README 4Pi entry point; imported by cmp_full_w4pi and showcase_real_mt
- keep | analysis/fourpi_classical/full_cl.py | README 4Pi entry point: full classical reconstruction
- keep | analysis/fourpi_classical/mov4pi_cl.py | movie-level classical stages imported by full_cl / real_cl / sim_cl
- keep | analysis/fourpi_classical/npc_site.py | NPC site picks; layout and sites reused by cmp_full_w4pi.py
- keep | analysis/fourpi_classical/post_cl.py | README 4Pi entry point: classical merge + drift
- keep | analysis/fourpi_classical/real_cl.py | per-file classical driver imported by full_cl.py
- keep | analysis/fourpi_classical/sim_cl.py | classical side of the showcase / fourpi_sim SI sweep
- keep | analysis/fourpi_classical/val_iso.py | classical fitter validation (CLEANUP_LOG check C4)

### analysis/results (140: keep 62, archive 78, delete 0)

All are 2026-11-11 development-phase snapshots behind docs/RESULTS.md (earlier default head, not the paper revision's models).
analysis/results/README.md lists every group as a manifest. keep = file or file group cited as a source by docs/RESULTS.md / METHOD.md,
or read / cited by code; archive = in the README manifest only.

- keep | analysis/results/README.md | manifest of the snapshots; referenced by README, docs, package docstrings
- keep | analysis/results/checks/cand_pmin_chunk12000_0.5nM.txt | cited in RESULTS.md section 5 and METHOD.md
- archive | analysis/results/dme/compare_dmev2_05.log | README manifest only; DME v2 comparison log
- archive | analysis/results/dme/compare_dmev2_25.log | README manifest only
- archive | analysis/results/dme/report_p2_05.log | README manifest only
- archive | analysis/results/dme/report_p2_25.log | README manifest only
- archive | analysis/results/dme/report_real20.txt | README manifest only (split-half column of 20 nM)
- keep | analysis/results/dme/splithalf_all.txt | cited in RESULTS.md section 7 (split-half snapshot)
- archive | analysis/results/evalproto/blockphase_0.5.txt | script listed in RESULTS 7, output not cited
- archive | analysis/results/evalproto/blockphase_2.5.txt | script listed in RESULTS 7, output not cited
- archive | analysis/results/evalproto/blockphase_20.txt | script listed in RESULTS 7, output not cited
- keep | analysis/results/evalproto/covpool_real0.5.txt | source of RESULTS section 1 FRC table (covpool_real*)
- keep | analysis/results/evalproto/covpool_real2.5.txt | source of RESULTS section 1 FRC table
- keep | analysis/results/evalproto/covpool_real20.txt | source of RESULTS section 1 FRC table
- archive | analysis/results/evalproto/covpool_simm05pd.txt | sim covpool, not cited
- archive | analysis/results/evalproto/covpool_simm05pe.txt | sim covpool, not cited
- archive | analysis/results/evalproto/covpool_simm25pd.txt | sim covpool, not cited
- archive | analysis/results/evalproto/covpool_simm25pe.txt | sim covpool, not cited
- archive | analysis/results/evalproto/envmap_0.5.json | envmap input of spatial.py; output not cited
- archive | analysis/results/evalproto/envmap_0.5.txt | not cited
- archive | analysis/results/evalproto/envmap_2.5.json | not cited
- archive | analysis/results/evalproto/envmap_2.5.txt | not cited
- archive | analysis/results/evalproto/envmap_20.json | not cited
- archive | analysis/results/evalproto/envmap_20.txt | not cited
- archive | analysis/results/evalproto/filt_real0.5.txt | filt.py listed in RESULTS 7, output not cited
- archive | analysis/results/evalproto/filt_real2.5.txt | not cited
- archive | analysis/results/evalproto/filt_real20.txt | not cited
- archive | analysis/results/evalproto/filt_simm05pd.txt | not cited
- archive | analysis/results/evalproto/filt_simm25pd.txt | not cited
- keep | analysis/results/evalproto/frc_published_real0.5.json | RESULTS section 1 published row (frc_published_real*)
- keep | analysis/results/evalproto/frc_published_real0.5.txt | RESULTS section 1 published row
- keep | analysis/results/evalproto/frc_published_real2.5.json | RESULTS section 1 published row
- keep | analysis/results/evalproto/frc_published_real2.5.txt | RESULTS section 1 published row
- keep | analysis/results/evalproto/nnlag_real0.5.txt | source of RESULTS section 2 (nnlag_real*)
- keep | analysis/results/evalproto/nnlag_real2.5.txt | source of RESULTS section 2
- keep | analysis/results/evalproto/nnlag_real20.txt | source of RESULTS section 2
- keep | analysis/results/evalproto/nnlag_simm05pd.txt | RESULTS section 2 (nnlag_sim{m05pd,...})
- keep | analysis/results/evalproto/nnlag_simm05pe.txt | RESULTS sections 2 / 3b
- keep | analysis/results/evalproto/nnlag_simm25pd.txt | RESULTS section 2
- keep | analysis/results/evalproto/nnlag_simm25pe.txt | RESULTS sections 2 / 3b
- archive | analysis/results/evalproto/realproto_0.5.txt | script listed in RESULTS 7 (row counts); output not cited
- archive | analysis/results/evalproto/realproto_2.5.txt | not cited
- archive | analysis/results/evalproto/realproto_20.txt | not cited
- archive | analysis/results/evalproto/spatial_real0.5.txt | supports an uncited 'spatially neutral' claim in RESULTS 1
- archive | analysis/results/evalproto/spatial_real2.5.txt | same
- archive | analysis/results/evalproto/spatial_real20.txt | same
- archive | analysis/results/evalproto/spatial_simm05pd.txt | not cited
- archive | analysis/results/evalproto/spatial_simm05pe.txt | not cited
- archive | analysis/results/evalproto/spatial_simm25pd.txt | not cited
- archive | analysis/results/evalproto/spatial_simm25pe.txt | not cited
- keep | analysis/results/evalproto/strictq_real0.5_p1_autoc.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real0.5_p1_autoc.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real0.5_p1_fixedc.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real0.5_p1_fixedc.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real0.5_p2L.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real0.5_p2L.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real2.5_p1_autoc.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real2.5_p1_autoc.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real2.5_p1_fixedc.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real2.5_p1_fixedc.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real2.5_p2L.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real2.5_p2L.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real20_p1_autoc.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real20_p1_autoc.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real20_p1_fixedc.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real20_p1_fixedc.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real20_p2L.json | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/strictq_real20_p2L.txt | RESULTS section 5 cites strictq_*
- keep | analysis/results/evalproto/valid_m05pd.txt | cited in RESULTS section 3b
- keep | analysis/results/evalproto/valid_m25pd.txt | cited in RESULTS section 3b
- archive | analysis/results/evalproto/zcal_fit_m05pd.json | sigma-fit parameters, not cited
- archive | analysis/results/evalproto/zcal_fit_m05pe.json | not cited
- archive | analysis/results/evalproto/zcal_fit_m25pd.json | not cited
- archive | analysis/results/evalproto/zcal_fit_m25pe.json | not cited
- keep | analysis/results/evalproto/zcal_fix.txt | cited by analysis/evalproto/zcal_fix.py
- keep | analysis/results/evalproto/zcal_real0.5_m05pe.txt | cited by analysis/evalproto/zcal.py and the README
- archive | analysis/results/evalproto/zcal_real2.5_m25pe.txt | not cited (sibling of the cited 0.5 nM file)
- archive | analysis/results/evalproto/zcal_real20_m25pe.txt | not cited
- keep | analysis/results/evalproto/zcal_sim_m05pd_m25pd.txt | source of RESULTS section 3b table
- keep | analysis/results/evalproto/zcal_sim_m05pe_m25pe.txt | source of RESULTS 3b; cited by evalproto/zcal.py
- keep | analysis/results/gate/gate_long_vs_simcode.txt | RESULTS 3b / 7; reproduced by analysis/gate/vs_simcode_p2.py
- keep | analysis/results/gate/merged_d1_vs_simcode.txt | RESULTS 7 snapshot; cited by vs_simcode_p2.py
- keep | analysis/results/hd_compare/score_m20qh.txt | source of RESULTS 3b table (m20qh)
- keep | analysis/results/hd_compare/score_m25qh.txt | source of RESULTS 3b table (m25qh)
- keep | analysis/results/hd_compare/summary.txt | cited by analysis/figures/field/*.py and figures/README.md
- archive | analysis/results/hd_compare/topn_vs_sc_m20qh.txt | not cited
- archive | analysis/results/hd_compare/topn_vs_sc_m25qh.txt | not cited
- archive | analysis/results/psfshape/evalgt_bc.txt | evalgt.py listed in RESULTS 7, output not cited
- archive | analysis/results/psfshape/evalgt_pd.txt | not cited
- keep | analysis/results/realism/realstats.json | cited by simcode/simulate/acquisition.py
- archive | analysis/results/realism/validate_0.5nM_final.json | legacy hybrid front-end validation, not ported
- archive | analysis/results/realism/validate_0.5nM_final.txt | same
- archive | analysis/results/realism/validate_2.5nM_final.json | same
- archive | analysis/results/realism/validate_2.5nM_final.txt | same
- archive | analysis/results/simcode_base2/cmp_m05q_b10.txt | m05q / m25q comparisons; no RESULTS table row cites them
- archive | analysis/results/simcode_base2/cmp_m05q_b22.txt | same
- archive | analysis/results/simcode_base2/cmp_m05qh_b10.txt | same
- archive | analysis/results/simcode_base2/cmp_m05qh_b22.txt | same
- archive | analysis/results/simcode_base2/cmp_m25q_b10.txt | same
- archive | analysis/results/simcode_base2/cmp_m25q_b22.txt | same
- archive | analysis/results/simcode_base2/cmp_m25qh_b10.txt | same
- archive | analysis/results/simcode_base2/cmp_m25qh_b22.txt | same
- archive | analysis/results/simcode_base2/gate_paired_scode_L64_5_vs_b10.txt | gate-paired variant, not cited
- archive | analysis/results/simcode_base2/gate_paired_scode_L64_5_vs_b22.txt | not cited
- keep | analysis/results/simcode_base2/gate_paired_scode_ni6_3_vs_b10.txt | RESULTS 7 command; cited by analysis/gate/gate_paired.py
- archive | analysis/results/simcode_base2/gate_paired_scode_ni6_3_vs_b22.txt | not cited
- archive | analysis/results/simcode_base2/gate_paired_scodenf_L64_5_vs_b10.txt | not cited
- archive | analysis/results/simcode_base2/gate_paired_scodenf_L64_5_vs_b22.txt | not cited
- archive | analysis/results/simcode_base2/gate_paired_scodenf_L64_5_vs_pm.txt | not cited
- archive | analysis/results/simcode_base2/gate_paired_scodenf_ni6_3_vs_b10.txt | not cited
- archive | analysis/results/simcode_base2/gate_paired_scodenf_ni6_3_vs_b22.txt | not cited
- archive | analysis/results/simcode_base2/gate_paired_scodenf_ni6_3_vs_pm.txt | not cited
- keep | analysis/results/simcode_base2/simcode_config.txt | cited by baselines/run_simcode.py and simcode/benchmark/preprint_gate.py
- keep | analysis/results/simcode_base2/summary.txt | cited by analysis/figures/field/*.py and figures/README.md
- archive | analysis/results/simcode_base2/summary_autoc.txt | not cited
- archive | analysis/results/simreal/cci_m05o_est.txt | movie m05o not in the RESULTS 3b tables
- archive | analysis/results/simreal/cci_m05o_p2L.txt | same
- keep | analysis/results/simreal/cci_m05r_est.txt | source of RESULTS 3b row m05r
- keep | analysis/results/simreal/cci_m05r_p2L.txt | source of RESULTS 3b row m05r
- archive | analysis/results/simreal/cci_m05rb_est.txt | m05rb mentioned in text only, not a table source
- archive | analysis/results/simreal/cci_m05rb_p2L.txt | same
- keep | analysis/results/simreal/cci_m20qh_est.txt | source of RESULTS 3b (n20/cci_m20qh_*)
- keep | analysis/results/simreal/cci_m20qh_p2L.txt | source of RESULTS 3b
- archive | analysis/results/simreal/cci_m25o_est.txt | not in the RESULTS tables
- archive | analysis/results/simreal/cci_m25o_p2L.txt | same
- keep | analysis/results/simreal/cci_m25qh_est.txt | source of RESULTS 3b (p2/long/cci_m25qh_*)
- keep | analysis/results/simreal/cci_m25qh_p2L.txt | source of RESULTS 3b
- archive | analysis/results/simreal/cci_m25r_est.txt | m25r superseded by m25rc in the tables
- archive | analysis/results/simreal/cci_m25r_p2L.txt | same
- keep | analysis/results/simreal/cci_m25rc_est.txt | source of RESULTS 3b row m25rc
- keep | analysis/results/simreal/cci_m25rc_p2L.txt | source of RESULTS 3b row m25rc
- keep | analysis/results/simreal/merged_gross_m20qh.txt | cited in RESULTS 3b
- keep | analysis/results/simreal/merged_gross_m25qh.txt | cited in RESULTS 3b
- archive | analysis/results/tput/bench_0.5nM.json | raw input of summarize.txt; not cited itself
- archive | analysis/results/tput/bench_0.5nM_contaminated_3.json | contaminated benchmark run; not cited
- archive | analysis/results/tput/bench_2.5nM.json | raw input of summarize.txt
- archive | analysis/results/tput/clean_0.5nM.json | raw input of summarize.txt
- archive | analysis/results/tput/clean_2.5nM.json | raw input of summarize.txt
- keep | analysis/results/tput/stages_real20.log | cited in RESULTS section 7 (20 nM stage times)
- keep | analysis/results/tput/summarize.txt | throughput summary behind RESULTS section 4; README cites it

### Uncertain cases

- Nanoruler scope: nothing of the nanoruler study is in the paper or SI. I kept the nr80 model recipe and calibration chain (configs/nr80.*
  and AUTO_PIPELINE 9.4 refer to model nr80; nr80 is not in simcode/models.json), plus the scripts that analysis/p2generic calls (eval_gen.sh for
  m1809g, eval_nr.sh / chain_gen2.sh for gnr80). If nr80 / gnr80 is not a model to reproduce, and the generic-pass-2 SI uses only eval_gen.sh,
  then only mkmodel.py, ev_p12sim.py, ev_sigcal.py, sim_nr_lib.py, abl_lib.py and nra_lib.py stay keep. The other 18 nanoruler keeps become archive.
  ev_chain.sh is kept, but its ev_accept.py step (line 47) has to be removed or it fails. ev_gates.py and ev_scenecheck.py are kept only
  because ev_chain.sh calls them.
- Nanoruler archive vs delete: the R24 / nr12 / abl drivers (*.sh) are one-night orchestration (the earlier plan deletes such scripts). I put
  them in archive because they are the only record of the commands behind the STATUS numbers. The one-off checks are in delete because
  nothing calls them and their results are in STATUS.md. r24r_pair031.py works on its own, but its only caller (r24r_accept.sh) is broken.
- 4Pi: diag_4pi.py and fourpi.py are kept on the assumption that they are where the per-cell delta_deg in simcode/fourpi/datasets.json
  came from. I did not confirm this. Their json output is read only by tb_analyze.py (archived). reg_xcorr.py (archive) and register.py
  (keep) may serve as references for the planned N-channel in-situ registration stage. showcase4pi_* is the earlier simulated showcase.
  The new SI sweep (untracked analysis/fourpi_sim) imports showcase4pi_movie, simfront4pi, simlin4pi, simeval4pi and sim_cl, but not
  showcase4pi_eval, _fig or _table. real_w4pi.py is in delete as a compatibility alias, but cmp_real4pi.py and post_w4pi.py docstrings
  still name it.
- analysis/results: the whole directory is a 2026-11-11 development-phase snapshot (old default head). The paper revision regenerates
  its figures and tables from the new models and does not read these files. "keep" here means "evidence cited by docs/RESULTS.md / code".
  If RESULTS.md is retired, all 140 can go to archive. realstats.json, simcode_config.txt and the two summary.txt files are named by code
  (acquisition.py, preprint_gate.py, run_simcode.py, figures/field/*), so archiving them means editing those references. The
  spatial_real* and tput bench/clean json files are borderline (they support uncited claims and summaries).

## Package modules (`simcode/`; `fourpi/` left for the 4Pi refactor)

Every module not listed is keep (used by `simcode.run`, the trainers, the automatic pipeline or the tests).

- delete | simcode/simulate/emitters.py | nothing imports it; the first detector training data, superseded by simulate/acquisition.py
- delete | simcode/benchmark/crlb_nuisance.py | nothing imports or runs it; nuisance-CRLB study of 2026-11-09 (numbers in docs/RESULTS.md)
- delete | simcode/calibration/camera_gain.py | not used; the pipeline's camera stage (pipeline/camera.py) generalises it
- delete | simcode/calibration/fit_psf_sums.py | not used; PSF refit on multi-frame sums, superseded by the in-situ fit (pipeline/psf_fit.py, torch_psf)
- archive | simcode/calibration/fit_psf_tracks.py, fit_pattern.py, psf_heldout.py, psf_compare.py, README.md | the manual in-situ recipe of the stored calibrations (models/insitu/*); the pipeline's psf stage (psf_fit.py = fit_psf_tracks as a function) is the current path; keep only if the stored calibrations must be re-made by the old recipe
- keep | simcode/calibration/__init__.py, extract_tracks.py, camera_maps.py | called by the automatic pipeline (tracks, cammaps stages)
- delete | simcode/processing/compare_pattern.py | nothing imports or runs it (pattern comparison of the development runs)
- archive | simcode/processing/pattern_field.py | analysis only (2 evalproto / simreal scripts, both archive candidates); the pipeline option was removed on 2026-11-14
- archive | simcode/processing/dedup.py | one analysis user (archive candidate); output modes for overlapping windows not used by simcode.run
- archive | simcode/solve/brightness_noise_fit.py | analysis only (ablation_cov / nanoruler); keep if the ablation SI table on the automatic c estimate is regenerated (ablation_cov is keep)
- archive | simcode/benchmark/preprint_gate.py | the preprint model on the gate scenes (development comparison; analysis/gate, figures)
- keep | simcode/benchmark/gate_scenes.py, gate_scenes_report.py, gate_scenes_report_detector_matched.py | gate scenes of a model's calibration (pipeline/training.py, gate_dump)
- delete | simcode/util/neighbours.py | only fit_psf_sums.py (delete candidate) uses it
- delete | simcode/detector/model.conv_flops | function; unused (FLOP count of the detector ablation)
- keep | simcode/evaluation/metrics.py | 21 analysis users, among them papersim / paper scripts (check with the analysis list before removing)

## Docs (`docs/`, top-level)

- keep | README.md | entry point
- keep | docs/LAYOUT.md | the tree and the rename table
- keep | docs/METHOD.md | method description (paper)
- keep | docs/USAGE.md, docs/DATASET_FORMAT.md | working with the tables (torch-psf Dataset)
- keep | docs/AUTO_PIPELINE.md | design and validation of the automatic pipeline (model recipe stage e)
- keep | docs/EVALUATION.md | evaluation protocol of the paper (FRC, GT-free metrics)
- keep | docs/PSF_POLARIZATION.md | in-situ PSF polarization terms (SI)
- archive | docs/RESULTS.md | results of the development phase (earlier default networks); many paths point to archive candidates; keep as record, no longer maintained
- archive | docs/CLEANUP_PLAN.md | the executed plan of the 2026-11 cleanup (its outcome is CLEANUP_LOG.md)
- keep | docs/CLEANUP_LOG.md | record of removals / renames (also of this cleanup)
- keep | models/README.md | model files, sha256, recipe
- keep | simcode/calibration/README.md | only with the calibration recipe (see above)

## Model weights (`/home/jc/simcode2/models`, not in git)

- keep | m1809/, m25/, g13/ (detector_pass1, emitter_pass1, detector_pass2, emitter_pass2) | the models of models.json (3D pass 1, generic pass 2 for m1809)
- keep | pretrain_emitter_pass1.pt | init of every model's emitter network (recipe)
- keep | pretrain_detector_pass1.pt | init of the 4Pi detector (analysis/fourpi/train_det4pi.py)
- keep | decode/hd_frame, decode/hd_sum6 | DECODE baseline models
- keep | insitu/beads/psf_fit_*.pt | bead PSF fit state (start of the in-situ fit)
- archive | archive_2d/{m1809,m25,g13}/ | the 2D models before 2026-11-14 (production of the earlier paper tables; not referenced)
- archive | archive_pass2_dedicated/m1809/ | dedicated (non-generic) pass 2 of m1809 (not referenced)
- delete | the top-level copies of the networks of m1809, m25, g13 (the flat layout before the model folders) | byte-identical copies of files in the model folders / archive_2d / archive_pass2_dedicated (deleted 2026-11-15)
- delete | the top-level copy of the g2ss pass-2 detector | byte-identical to m1809/detector_pass2.pt (the adopted generic pass-2 detector; deleted 2026-11-15)
- archive | archive_pass2_generic/{g1809,g2c,g2cc,g2s,g2ss,g2w}/ (detector_pass2, emitter_pass2, calib/) | generic pass-2 study variants (out/p2generic/STATUS.md); g2ss pair adopted into m1809; keep only if the generic-pass-2 SI table is regenerated
- delete | archive_superseded/gnr80/ | nanoruler generic pass 2 (not adopted)
- delete | archive_superseded/psf3dmix/, psf3dmix_pair/ | superseded mixture set (models/README.md: not needed by anything)
- delete | archive_superseded/ablation_1809_avg/, ablation_1809_ax/ | ablation emitter networks of the PSF-matching study (results recorded)
- delete | archive_superseded/earlier_architecture/emitter_pass1.pt | earlier emitter-network architecture, not loadable by the package
- delete | archive_superseded/previous_default/ (detector_pass2, emitter_pass2, calib/) | earlier default pass 2 (superseded by the models' own pass 2)

## Scripts that work on the file names before 2026-11-16 (added 2026-11-16)

One-off or old-checkout scripts that were not rewritten by the 2026-11-16 rename (their inputs carry the earlier names):

- archive | analysis/maintenance/cleanup_outputs.py | the executed 2026-11-15 out/ + models/ clean-up (its file lists are the names of that day)
- archive | analysis/z3d/swap_model.py | the executed 2026-11-14 model swaps (reads the staged out/z3d model folders with the earlier names)
- archive | analysis/p2generic/chain_gen.sh, cont_det.sh | run only from a checkout before 2026-11-15 (master f48ab14 = the former main 710d34b; --p_std)

## Output directories (`/home/jc/simcode2/out`; size with --apparent-size, symlink target)

Many entries are symlinks into /mnt/data/dmdsf/simcode2_out (or /mnt/data/4pi-sf); deleting them frees space there.

- keep | out/4pi (107G -> /mnt/data/4pi-sf/simcode2_out/4pi) | 4Pi data, models, tables of the 4Pi-SIMCODE SI figure
- delete | out/ablate_frr_hd_b.json (730) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/ablate.log (4,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/ablate_polish_hd_b.json (1,7K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/ablate_prior_hd_b.json (1,3K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/ablate_r11.json (4,3K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/ablate_r41.json (4,3K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/ablation_cov (14G -> /mnt/data/dmdsf/simcode2_out/ablation_cov) | covariance ablation of the paper
- keep | out/auto (2,2G) | automatic-pipeline validation (AUTO_PIPELINE.md section 9)
- archive | out/autoc (545K -> /mnt/data/dmdsf/simcode2_out/autoc) | automatic-c estimates (ablation SI if regenerated)
- archive | out/bench (5,2M -> /mnt/data/dmdsf/simcode2_out/bench) | benchmarks
- delete | out/bench_hd_a_quick.log (6,5K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/bright (161M -> /mnt/data/dmdsf/simcode2_out/bright) | bright-emitter study
- archive | out/calib_inputs (2,3G -> /mnt/data/dmdsf/simcode2_out/calib_inputs) | inputs of the pretrain_emitter_pass1 template calibrations
- delete | out/calib_sc_hd_sc1.json (284) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc1.json.p1 (189) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2b15.json (322) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2b15.json.p1 (230) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2.json (290) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2.json.p1 (191) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2Lb15.json (322) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2Lb15.json.p1 (230) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2Ln1b15.json (321) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc2Ln1b15.json.p1 (229) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc3b15.json (321) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc3b15.json.p1 (230) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5db15.json (322) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5db15.json.p1 (230) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5nb15.json (322) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5nb15.json.p1 (231) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5nb15_pm.json (299) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5nbb15.json (323) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5nbb15.json.p1 (230) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/template_calib_emitter_pass1_brightness_noise_fit.json (2,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5nbb15_pm_autoc.json (2,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/template_calib_emitter_pass1_sim.json (2,0K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/template_calib_emitter_pass1_real.json (2,8K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc5nbb15_pm.json (548) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc6nb15.json (322) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc6nb15.json.p1 (230) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc6nb15_pm.json (444) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc7bgb15.json (320) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc7bgb15.json.p1 (229) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/calib_sc_hd_sc7bgb15_pm.json (441) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/cleanpass (678M) | clean pass 2026-11-14 record (STATUS)
- delete | out/critic34 (43M -> /mnt/data/dmdsf/simcode2_out/critic34) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/critic35 (2,8M -> /mnt/data/dmdsf/simcode2_out/critic35) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/crlbcmp (48M) | CRLB study of the paper
- delete | out/decomp (98M -> /mnt/data/dmdsf/simcode2_out/decomp) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/decomp_sc2L (48M -> /mnt/data/dmdsf/simcode2_out/decomp_sc2L) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/decomp_sc2Ln1 (111M -> /mnt/data/dmdsf/simcode2_out/decomp_sc2Ln1) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/decomp_sc3 (48M -> /mnt/data/dmdsf/simcode2_out/decomp_sc3) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/decomp_sc5d (111M -> /mnt/data/dmdsf/simcode2_out/decomp_sc5d) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/decomp_sc5n (111M -> /mnt/data/dmdsf/simcode2_out/decomp_sc5n) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/decomp_sc5n_pm (67M -> /mnt/data/dmdsf/simcode2_out/decomp_sc5n_pm) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/detband (1,2G) | detector --band A/B study (STATUS)
- delete | out/det_up2 (275M -> /mnt/data/dmdsf/simcode2_out/det_up2) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/drift (1,8G -> /mnt/data/dmdsf/simcode2_out/drift) | earlier drift study (RESULTS.md)
- archive | out/drift25 (3,2G -> /mnt/data/dmdsf/simcode2_out/drift25) | earlier drift study 2.5 nM
- keep | out/emitter_heldout (526M -> /mnt/data/dmdsf/simcode2_out/sc4) | held-out set of the emitter trainer (= sc4)
- archive | out/evalproto (739K -> /mnt/data/dmdsf/simcode2_out/evalproto) | evaluation-protocol development (EVALUATION.md, RESULTS.md)
- archive | out/ext (124M -> /mnt/data/dmdsf/simcode2_out/ext) | external data checks
- archive | out/field (11G -> /mnt/data/dmdsf/simcode2_out/field) | spatial pattern-field study (RESULTS.md)
- keep | out/fourpi_sim (5,1M) | 4Pi SI simulation sweep
- delete | out/full_0.5nM_p2L_real (2,4G -> /mnt/data/dmdsf/simcode2_out/full_0.5nM_p2L_real) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_0.5nM_retr_real (5,5G -> /mnt/data/dmdsf/simcode2_out/full_0.5nM_retr_real) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_0.5nM_sc5nb_autoc (1,4G -> /mnt/data/dmdsf/simcode2_out/full_0.5nM_sc5nb_autoc) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_0.5nM_sc5nb_autoc_v1s (1,4G -> /mnt/data/dmdsf/simcode2_out/full_0.5nM_sc5nb_autoc_v1s) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_0.5nM_sc5nb_bright_real (1,4G -> /mnt/data/dmdsf/simcode2_out/full_0.5nM_sc5nb_bright_real) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_20nM_m2_fast (14G -> /mnt/data/dmdsf/simcode2_out/full_20nM_m2_fast) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_20nM_m2_fast_autoc (7,9G -> /mnt/data/dmdsf/simcode2_out/full_20nM_m2_fast_autoc) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_2.5nM_m2_p2L_real (17G -> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_p2L_real) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_2.5nM_m2_retr_real (72G -> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_retr_real) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_2.5nM_m2_sc5nb_autoc (9,0G -> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_sc5nb_autoc) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/full_2.5nM_m2_sc5nb_bright_real (9,0G -> /mnt/data/dmdsf/simcode2_out/full_2.5nM_m2_sc5nb_bright_real) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/gate (2,0G -> /mnt/data/dmdsf/simcode2_out/gate) | gate scenes / npz (CRLB source of gate_dump; model calibration)
- keep | out/gate_dumps (500M -> /mnt/data/dmdsf/simcode2_out/p2/fe) | gate dumps of the models (= p2/fe)
- delete | out/gate_gpu1.log (754) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/gate_gpu1.pid (8) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/gate_gpu2.log (1,9K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/gate_gpu2.pid (8) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/gate_report_sc1.json (58K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/gate_report_sc.json (125K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/gate_report_v1.json (136K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/gate_report_v2.json (215K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/HANDOFF_SESSION.md (19K) | session hand-off
- archive | out/hd_compare (45M -> /mnt/data/dmdsf/simcode2_out/hd_compare) | high-density comparison (RESULTS.md)
- archive | out/hdunbias (7,9G) | pass-1 prior pixel-locking study (STATUS final)
- keep | out/insitu (146M -> /mnt/data/dmdsf/simcode2_out/insitu) | in-situ calibration outputs (tracks, camera maps, PSF fits)
- delete | out/logs_move_paper2.sh (1,3K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/miss_anatomy.py (2,3K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/move_paper2_b.log (2,5K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/move_paper2_b.sh (1,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/move_paper2.log (564) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/n20 (302K -> /mnt/data/dmdsf/simcode2_out/n20) | 20 nM small study
- archive | out/nanoruler (131G) | nanoruler nr80 study (STATUS.md, report/); not in the paper; 131 GB: keep report/STATUS, drop runs
- keep | out/newpsf_validation (1,9G) | torch-psf (redesigned PSF classes) validation
- archive | out/nuis (114K -> /mnt/data/dmdsf/simcode2_out/nuis) | nuisance CRLB numbers
- archive | out/optclean (2,5G) | options clean-up record (STATUS)
- keep | out/p2 (505M -> /mnt/data/dmdsf/simcode2_out/p2) | gate dumps p2/fe and pass-2 gate evaluation; new names gate_dumps / pass2_gate (symlinks)
- keep | out/p2generic (4,1G -> /mnt/data/dmdsf/simcode2_out/p2generic) | generic pass-2 study (paper / SI; STATUS.md)
- archive | out/p2prior (344K -> /mnt/data/dmdsf/simcode2_out/p2prior) | pass-2 prior study
- archive | out/p2real (44M -> /mnt/data/dmdsf/simcode2_out/p2real) | pass-2 real-data checks
- keep | out/paper (45G -> /mnt/data/dmdsf/simcode2_out/paper) | paper figures / runs (earlier production, baselines)
- keep | out/paper2 (375G) | paper production runs (1809_2, 2.5nM_2, 20nM_2 v2 tables, baselines, figures)
- keep | out/papersim (69G -> /mnt/data/dmdsf/simcode2_out/papersim) | paper simulations (all methods, metrics, figures)
- keep | out/pass2_gate (505M -> /mnt/data/dmdsf/simcode2_out/p2) | pass-2 gate evaluation (= p2)
- keep | out/pixbias (29M) | pixel-bias study of the paper
- delete | out/plots (37M -> /mnt/data/dmdsf/simcode2_out/plots) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/plots_check (9,3M -> /mnt/data/dmdsf/simcode2_out/plots_check) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/plots_check2 (1,3M -> /mnt/data/dmdsf/simcode2_out/plots_check2) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/post_hd_b.log (2,5K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/post_hd_b.pid (8) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/psf3d (446M -> /mnt/data/dmdsf/simcode2_out/psf3d) | PSF calibration of the paper data (out/psf3d/final)
- keep | out/psf3d_nets (18G -> /mnt/data/dmdsf/simcode2_out/psf3d_nets) | sigma-rescale fits (sigcal sims m05r3 / m25rc3)
- keep | out/psf_compare_1809_25 (979K) | in-situ PSF comparison 1809_2 vs 2.5 nM (2026-11-15)
- keep | out/psfmatch (9,9G -> /mnt/data/dmdsf/simcode2_out/psfmatch) | runs of the model recipe (calibrations, gate dumps of the models)
- keep | out/PSFMATCH_STATUS.md (8,8K) | status of the PSF-matched models
- archive | out/psfshape (166M -> /mnt/data/dmdsf/simcode2_out/psfshape) | PSF-shape study (RESULTS.md)
- delete | out/psf_v2 (6,4M) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/real20 (2,5G -> /mnt/data/dmdsf/simcode2_out/real20) | earlier 20 nM real runs (RESULTS.md)
- archive | out/realgap (818M -> /mnt/data/dmdsf/simcode2_out/realgap) | sim-vs-real gap study
- archive | out/realism (46M -> /mnt/data/dmdsf/simcode2_out/realism) | simulator realism checks (c calibration)
- keep | out/realstats.json (11K) | read by simulate/acquisition.py
- archive | out/refs (76M -> /mnt/data/dmdsf/simcode2_out/refs) | reference runs (RESULTS.md)
- keep | out/registration (179M) | in-situ channel registration (multi-channel work)
- keep | out/rename (1,2G) | checks of the 2026-11-15 reorganisation (outputs before / after); delete after the merge
- archive | out/resolvability (467K -> /mnt/data/dmdsf/simcode2_out/resolvability) | pair-resolvability bounds
- delete | out/retrain_accept.txt (4,2K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/retrain_r (297M -> /mnt/data/dmdsf/simcode2_out/retrain_r) | retrain study (RETRAIN_STATUS.md)
- keep | out/RETRAIN_STATUS.md (8,2K) | retrain status record
- delete | out/run_baselines.log (16K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_baselines.pid (8) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_baselines.sh (832) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_chain_b.sh (388) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_gate.sh (486) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_gpu2_a.pid (8) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_gpu2_a.sh (603) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_post.sh (775) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_sc1.log (2,9K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_sc1.pid (8) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_sc2.log (1,8K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/run_sc.sh (855) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/sc3 (567K -> /mnt/data/dmdsf/simcode2_out/sc3) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/sc4 (526M -> /mnt/data/dmdsf/simcode2_out/sc4) | heldout4/5.pt of the emitter trainer; new name emitter_heldout (symlink), remove the old name after the z3d workers
- delete | out/sc6 (6,7M -> /mnt/data/dmdsf/simcode2_out/sc6) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/sc7 (70M -> /mnt/data/dmdsf/simcode2_out/sc7) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/scratch (61G -> /mnt/data/dmdsf/simcode2_out/scratch) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/selection_global (1,1G -> /mnt/data/dmdsf/simcode2_out/selection_global) | selection study behind Fig 3/4 (sigma <= 12 nm)
- keep | out/shareloc_u373 (355M) | torch-psf real-data test case
- archive | out/simcode_base (1,8M -> /mnt/data/dmdsf/simcode2_out/simcode_base) | preprint baseline numbers (gate comparison)
- archive | out/simcode_base2 (1,7M -> /mnt/data/dmdsf/simcode2_out/simcode_base2) | preprint baseline numbers
- archive | out/simreal2 (554K -> /mnt/data/dmdsf/simcode2_out/simreal2) | sim-real study
- delete | out/smoke_0.5nM_sc5n (101M -> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5n) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/smoke_0.5nM_sc5nb (101M -> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5nb) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/smoke_0.5nM_sc5nb_bright (101M -> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5nb_bright) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/smoke_0.5nM_sc5nb_bright_real (101M -> /mnt/data/dmdsf/simcode2_out/smoke_0.5nM_sc5nb_bright_real) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- archive | out/solvefix (928M) | solve-side fixes, all rejected (STATUS final)
- keep | out/solvegap (4,8M) | solve-gap analysis (paper)
- archive | out/speed (1,4G -> /mnt/data/dmdsf/simcode2_out/speed) | speed branch measurements
- archive | out/static25 (2,4G -> /mnt/data/dmdsf/simcode2_out/static25) | static-background study 2.5 nM
- delete | out/summary_baselines.json (40K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/time_sc.py (2,0K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/torchpsf_validation (4,9G) | torch-psf integration validation
- archive | out/tput (859M) | throughput benchmark (re-measure if the paper quotes v2 runtime)
- delete | out/validate_0.5nM_final.json (3,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_0.5nM_final.txt (1,1K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_0.5nM.json (3,1K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_0.5nM_v2.json (3,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_0.5nM_v3.json (3,5K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_0.5nM_v4.json (3,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_2.5nM_final.json (3,3K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_2.5nM_final.txt (1,1K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_2.5nM.json (3,0K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_2.5nM_v2.json (3,3K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_2.5nM_v3.json (3,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- delete | out/validate_2.5nM_v4.json (3,4K) | development output of earlier networks / one-off logs and scripts (numbers in docs/RESULTS.md or superseded)
- keep | out/WORKLOG.md (72K) | work log
- keep | out/z3d (7,0G) | 3D models and generic pass-2 evaluation (current models)
- archive | out/zimflux (1,3G) | ZIMFLUX / 3D SIMCODE design study (STATUS / DESIGN.md)
