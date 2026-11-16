# Fig 2 (simulation, webSMLM hollow-tube ground truth): facts sheet (2026-11-15)

**Change log 2026-11-16 (DECODE swap):** Fig 2 (`fig_sim_Dexp_v2psf_wsm3d.*`) and the wsm3d SI sweeps (`out/papersim/figures/fig_sim_{D,Dexp}_v2psf_wsm3d_common.*`) now use DECODE = dec_m1809v3 (published training settings, converged, `<run>/decode/locs_v3.npz`); previous files kept as `*_decold.*`. Dexp1_wsm3d DECODE: GT-FRC 28.5 -> 24.0 nm, 124k vs 102k localizations, frames to 30 nm 49.5k -> 21.8k, false positives 2 246 -> 1 843. The D-sweep table was also regenerated with master's fig_sim_main.py, so its non-DECODE fair-FRC column moves too (script version, not DECODE; e.g. SIMCODE 40.1 -> 38.0 nm, SMLM 91.5 -> 85.0 nm at density 0.5). Diff: out/decode_checks/v3/decv3_caption_diff.md; training note: SI_DECODE_TRAINING_NOTE.md.

**Note: the SIMCODE selection may still change.** An FRC-optimal-selection experiment and an aggregated-detection-probability fix are being evaluated. All numbers here are for the current selection: p >= 0.63 inside the field, calibrated max(sigma_x, sigma_y) <= 12 nm.

Format: one fact per line, with its source in brackets. Paths are relative to the repository root. Abbreviations:
- RUN<d> = out/papersim/Dexp<d>_v2psf_wsm (d = 0.5 / 1 / 2 / 4).
- GTF = RUN<d>/full/gtfrc_m1809v2.json; MET = RUN<d>/full/metrics_m1809v2.json.
- RULES = out/paper_figs_v2/fig_sim_Dexp_v2psf_wsm_gtfrc_rules.json, written by out/paper_figs_v2/fig_sim_Dexp_v2psf_wsm_gtfrc_rules.py. It recomputes GT-FRC and the frames-to-30-nm time course under both crossing rules (CPU), and it reproduces the stored first-crossing values exactly.

Labels: the method is called "SIMCODE" throughout. The internal run names (m1809v2, v2sig, dec_sim1809v2) are file names only.

## 1. Files

### Main Fig 2 (fig_sim_fig2.py, paper version)
- Fig 2 PDF: out/paper_figs_v2/fig_sim_Dexp_v2psf_wsm.pdf [ls, 2026-11-15 18:41]
- Fig 2 SVG: out/paper_figs_v2/fig_sim_Dexp_v2psf_wsm.svg [ls]
- Fig 2 PNG: out/paper_figs_v2/fig_sim_Dexp_v2psf_wsm.png [ls]
- Fig 2 caption: out/paper_figs_v2/fig_sim_Dexp_v2psf_wsm_caption.txt [ls]
- Fig 2 values JSON: **does not exist**. fig_sim_fig2.py writes no values file; the numbers come from RUN1/full/{metrics,gtfrc}_m1809v2.json. [analysis/papersim/fig_sim_fig2.py l.57-58 and docstring "output"]
- Crossing-rule values (both rules, all densities, plus the time courses): RULES (new, this sheet). [RULES]
- Internal variant (with the preprint column; NOT for the manuscript): out/paper_figs_v2/fig_sim_Dexp_v2psf_wsm_internal.{pdf,svg,png}, _internal_caption.txt [ls]
- Generator: `python analysis/papersim/fig_sim_fig2.py --prefix Dexp --suffix _v2psf_wsm --model m1809v2`, density 1 by default [fig_sim_fig2.py usage; out/papersim/logs/figs_wsm_Dexp.log]
- Zoom box: gen.json zoom_crop = [13, 17, 16] px (x0, y0, w), i.e. 1.73 um, picked at a filament crossing (zoompick) [RUN1/gen.json; wsm_gt.py docstring]

### SI table (fig_sim_main.py)
- SI table: out/papersim/figures/fig_sim_Dexp_v2psf_wsm_table.md and .csv. Columns: n, GT-FRC, frames to 30 nm, fair (half-split) FRC, Jaccard 10 nm, false-positive fraction; SIMFLUX rows with the estimated and with the true pattern. [files]
- The table's legend text is the header paragraph of the .md file. Its title line contains "SIMCODE = m1809v2"; drop that internal name in the paper. [table.md l.1]

### SI figure: per-density panels (fig_sim_main.py)
- Files: out/papersim/figures/fig_sim_Dexp_v2psf_wsm.{pdf,svg,png} and _caption.txt [ls 18:43-18:44]
- Content: (A) renders of all methods x densities, (B) PSF, (C) GT-FRC vs density, (D) frames to 30 nm vs density [its caption]
- This figure is NOT the same file as Fig 2 (the two PDFs differ). [cmp]

### SI overview (fig_overview.py)
- Files: out/papersim/overview_Dexp_v2psf_wsm_m1809v2.{pdf,svg,png} and _caption.txt [ls 18:44]
- Location: at the top level of out/papersim, not in figures/. [ls]
- Internal variant: overview_Dexp_v2psf_wsm_m1809v2_internal.* [ls]
- Wording to fix before use: the caption footnote names "dec_sim1809v2", and it says "SIMCODE with the PSF-matched networks". [overview caption]

### Per-density panels and summary (internal working files)
- Panels: out/papersim/fig_density_Dexp{0.5,1,2,4}_v2psf_wsm_m1809v2.png [ls]
- Full metrics table: out/papersim/summary_density_Dexp_v2psf_wsm_m1809v2.{md,json} [ls]

### D condition (the preprint condition: 500 detected photons per window, bg 2.5 photons/px/frame), wsm GT: NOT FINISHED
- No metrics_m1809v2.json or gtfrc_m1809v2.json exists in any D{0.5,1,2,4}_v2psf_wsm/full. [ls, 19:12]
- Lane progress at 19:12: the base lane has started D4 base on GPU 0. The net lane (DECODE + SIMCODE) finished D4 at 18:52. The post lane (metrics, gtfrc) has not started any movie. [out/papersim/logs/run_wsm_{base,net,post}_D.log]
- **Blocker:** the D1 preprint-SIMCODE step (v1) crashed with "ValueError: spots_per_bin should be set to result in either 1 or more than 2 frame bins (now 2, for 10618 spots.)". run_wsm.sh still logged rc 0. D1_v2psf_wsm/full/v1/v1_info.json therefore does not exist. [out/papersim/logs/full_D1_v2psf_wsm.log; D1_v2psf_wsm/full/logs/v1.log; ls D1_v2psf_wsm/full/v1]
- Consequence: the post lane runs D1 first and waits for v1/v1_info.json, so it will wait forever. D1, and then D0.5 / D2 / D4, get no metrics unless v1 is rerun or the wait is removed. [out/papersim/logs/run_wsm.sh 'post' case]
- So no D-condition wsm SI material exists yet. [above]

## 2. Numbers for the Results text

### 2a. Crossing rule: Fig 2, the SI table and the captions use the FIRST crossing
- Fig 2 titles, panels I and J, the SI table and the SI figure all take GT-FRC = gtfrc.crossing: 1 / (the first frequency from shell 3 on where the 5-sample-smoothed curve is < 1/7). [analysis/papersim/gtfrc.py crossing(), gt_frc(); gtfrc_run.py; fig_sim_fig2.py / fig_sim_main.py read GTF gt_frc.res_nm]
- Frames to 30 nm (Fig 2 J, SI table) use the same first-crossing rule on the first N frames, log-interpolated between N = 250, 500, 1k, 2k, 4k, 8k, 15k, 30k, 45k, 60k. [gtfrc.time_to_fidelity]
- The "fair FRC" (half-split) column of the SI table is also the first crossing (MET frc_nm). MET also stores frc_nm_sustained. [metrics.py]
- The sustained rule used elsewhere in the paper is metrics.frc_sustained: the first frequency after which the curve stays below 1/7 for 0.01 /nm. [metrics.py l.31-32, frc_sustained]
- **Recommendation: switch Fig 2, the SI table and the text to the sustained rule.** That needs gtfrc.crossing -> metrics.frc_sustained in gt_frc / time_to_fidelity, then a rerun of the gtfrc step and the figure scripts (CPU). This sheet does not change the figures. RULES already holds the sustained values. [RULES]
- Under the first rule, the SMLM GT-FRC at density 1 is 69.9 nm (it stops inside the dip); sustained, it is 40.1 nm. [RULES '1'.smlm]
- The fair FRC is affected too: SIMFLUX (estimated pattern) at density 1 is 77.6 nm first-crossing vs 34.3 nm sustained. SIMCODE is 32.3 vs 29.7 nm. [MET frc_nm, frc_nm_sustained]

### 2b. The spectral null: measured
- Projected distance of the labels from their microtubule axis: p50 15.6 nm, p90 24.5 nm, p99 27.6 nm, histogram peak at 20-24 nm, none beyond 30 nm. This is the arcsine-type profile of a shell. [out/papersim/websmlm_gt/gt_11_3_3.npz pts vs centre; computed for this sheet]
- Label geometry: attachment on the 12.5 nm tube surface, then a 12 nm radial binder, then the dye 2-5 nm from the binder tip in a uniform direction. Labels therefore sit about 25-30 nm from the axis (about 24 nm effective in projection). [analysis/fourpi_sim/mt_cells.py l.19-20, l.140; out/fourpi_sim/README.md 'Labels']
- Radial power spectrum of the rendered GT (5 nm pixels): a broad minimum between about 70 and 55 nm, lowest at 61 nm. [computed for this sheet from RUN1/gt.npz]
- Local minima of the density-1 GT-FRC curves between 120 and 45 nm: SMLM 59 nm (0.107, below 1/7), SIMFLUX 70 nm (0.199), DECODE 64 nm (0.233), SIMCODE 64 nm (0.432). [GTF curves, density 1]
- Theory: the first zero of J0(2 pi f R) is at 1/f = 2 pi R / 2.405. That is 63 nm for R = 24 nm and 65 nm for R = 25 nm. [metrics.frc_sustained docstring; arithmetic]

### 2c. Suggested caption / text sentences (the null)
"The labels of the simulated microtubules sit on a shell of about 25-30 nm radius around the tube axis (antibody-sized binder on the 25 nm tube). The projected cross-section of such a hollow tube has a Fourier transform close to J0(2 pi f R), which has its first zero near 1/(60-70 nm). Every FRC against the ground truth therefore dips at about 1/65 nm and recovers at higher frequencies. We report the resolution where the FRC falls below 1/7 for good (the sustained crossing used throughout this paper), so that a curve dipping into this structural null is not mistaken for the resolution limit."

### 2d. Density 1 (Fig 2), both rules
Next best = the best of SMLM, SIMFLUX (estimated pattern) and DECODE, the methods shown in Fig 2. Gain = (other - SIMCODE) / other.
- GT-FRC, first crossing (Fig 2 as drawn): SMLM 69.9, SIMFLUX 28.0, DECODE 32.1, SIMCODE 18.8 nm. [GTF '1'; table.md]
- GT-FRC, sustained: SMLM 40.1, SIMFLUX 22.0, DECODE 32.1, SIMCODE 18.2 nm. [RULES '1']
- SIMCODE gain over the next best (SIMFLUX), first rule: 33 % (28.0 -> 18.8). Sustained: 17 % (22.0 -> 18.2). [RULES]
- SIMCODE gain over DECODE: 41 % first rule, 43 % sustained. Over SMLM: 73 % first, 55 % sustained. [RULES]
- SIMCODE gain over SIMFLUX with the true pattern (SI only; 20.5 nm under both rules): 9 % first, 11 % sustained. [RULES '1'.sf_gtp]
- Frames to 30 nm, first rule (Fig 2 J): SMLM not reached, SIMFLUX 44.4k, DECODE not reached, SIMCODE 10.0k (4.4x fewer than SIMFLUX). [GTF ttf.frames_to; caption]
- Frames to 30 nm, sustained: SMLM not reached, SIMFLUX 31.9k, DECODE not reached, SIMCODE 8.2k (3.9x fewer). SIMFLUX with the true pattern: 42.7k first, 16.8k sustained. [RULES f30_sust]
- Localizations in the comparison set: SMLM 45 291, SIMFLUX 18 761, DECODE 74 186, SIMCODE 109 086. That is 1.47x DECODE and 5.8x SIMFLUX. [GTF n; caption E]
- False-positive fraction (no emitter ON in the window within 3 own sigma): SMLM 41.4 %, SIMFLUX 20.9 %, DECODE 7.3 %, SIMCODE 2.5 %. [GTF ghost.ghost_fraction]
- Jaccard 10 nm (preprint definition, all rows, 60k frames): SMLM 0.725, SIMFLUX 0.876, DECODE 0.830, SIMCODE 0.950. [MET jaccard_end]
- sigma-Jaccard (3-sigma Mahalanobis gate vs all labels): SMLM 0.832, SIMFLUX 0.721, DECODE 0.993, SIMCODE 0.994. [GTF sigma_jaccard.jaccard]
- Recall, structure (fraction of labels within 3 sigma of some localization): SMLM 0.702, SIMFLUX 0.596, DECODE 0.985, SIMCODE 0.980. [GTF sigma_jaccard.frac_gt_recovered]
- Recall, emitters (GT emitters with >= 150 photons in a window, matched in that window): SMLM 0.248, SIMFLUX 0.124, DECODE 0.371, SIMCODE 0.538. [MET gt_match.recall_gt_ph150]
- GT RMS error x, y (comparison set vs its ON emitter): SMLM 28.0/27.3, SIMFLUX 8.3/7.9, DECODE 13.8/13.8, SIMCODE 8.3/8.1 nm. [MET gt_match.rms_nm]
- Fair half-split FRC: SMLM 88.1, SIMFLUX 77.6, DECODE 59.3, SIMCODE 32.3 nm (first). Sustained: 88.1 / 34.3 / 46.6 / 29.7 nm. [MET frc_nm, frc_nm_sustained]

### 2e. All densities (SI table), both rules
SF = SIMFLUX with the estimated pattern; SFt = SIMFLUX with the true pattern (SI only). Values are first / sustained (nm).
- GT-FRC at 0.5: SMLM 68.8/42.5, SF 25.1/23.1, SFt 22.7/19.6, DECODE 33.3/33.3, SIMCODE 19.7/17.9. [RULES '0.5']
- GT-FRC at 1: SMLM 69.9/40.1, SF 28.0/22.0, SFt 20.5/20.5, DECODE 32.1/32.1, SIMCODE 18.8/18.2. [RULES '1']
- GT-FRC at 2: SMLM 72.2/72.2, SF 31.2/31.2, SFt 30.5/21.8, DECODE 31.9/31.9, SIMCODE 19.0/19.0. [RULES '2']
- GT-FRC at 4: SMLM 78.8/71.1, SF 66.7/56.3, SFt 66.7/37.7, DECODE 54.2/35.0, SIMCODE 19.7/19.1. [RULES '4']
- SIMCODE gain over the next best Fig-2 method, first rule: 21 % (SF) at 0.5, 33 % (SF) at 1, 39 % (SF) at 2, 64 % (DECODE) at 4. [RULES]
- The same, sustained rule: 22 % (SF), 17 % (SF), 39 % (SF), 45 % (DECODE). [RULES]
- SIMCODE gain over the best method including SFt, first / sustained: 0.5: 13 % / 9 %; 1: 9 % / 11 %; 2: 38 % / 13 %; 4: 64 % / 45 % (DECODE). [RULES]
- SIMCODE GT-FRC range over 0.5-4 emitters/um^2/frame: 18.8-19.7 nm first, 17.9-19.1 nm sustained. It is essentially flat with density. [RULES]
- Frames to 30 nm, first rule. SIMCODE: 18.0k / 10.0k / 7.8k / 9.0k. SF: 42.8k / 44.4k / - / -. SFt: 31.8k / 42.7k / - / -. SMLM and DECODE: not reached at any density. [GTF ttf.frames_to; table]
- Frames to 30 nm, sustained. SIMCODE: 15.6k / 8.2k / 6.4k / 5.1k. SF: 21.8k / 31.9k / - / -. SFt: 22.6k / 16.8k / 46.7k / -. SMLM and DECODE: not reached. [RULES f30_sust]
- Localizations (0.5 / 1 / 2 / 4). SMLM: 31 613 / 45 291 / 50 799 / 38 173. SF: 16 071 / 18 761 / 14 633 / 6 946. SFt: 16 049 / 18 655 / 14 366 / 6 838. DECODE: 45 988 / 74 186 / 102 546 / 126 700. SIMCODE: 59 767 / 109 086 / 172 958 / 222 591. [GTF n; table]
- False-positive fraction (0.5 / 1 / 2 / 4). SMLM: 0.278 / 0.414 / 0.571 / 0.700. SF: 0.126 / 0.209 / 0.340 / 0.545. SFt: 0.103 / 0.165 / 0.277 / 0.485. DECODE: 0.046 / 0.073 / 0.120 / 0.209. SIMCODE: 0.020 / 0.025 / 0.032 / 0.045. [GTF ghost; table]
- Jaccard 10 nm (0.5 / 1 / 2 / 4). SMLM: 0.732 / 0.725 / 0.698 / 0.673. SF: 0.867 / 0.876 / 0.843 / 0.720. SFt: 0.870 / 0.882 / 0.852 / 0.729. DECODE: 0.837 / 0.830 / 0.813 / 0.787. SIMCODE: 0.946 / 0.950 / 0.945 / 0.938. [MET jaccard_end; table]
- sigma-Jaccard (0.5 / 1 / 2 / 4). SMLM: 0.817 / 0.832 / 0.823 / 0.792. SF: 0.694 / 0.721 / 0.652 / 0.415. DECODE: 0.991 / 0.993 / 0.993 / 0.991. SIMCODE: 0.990 / 0.994 / 0.996 / 0.996. [GTF sigma_jaccard]
- Structure recall (labels within 3 sigma, 0.5 / 1 / 2 / 4). SMLM: 0.693 / 0.702 / 0.705 / 0.697. SF: 0.570 / 0.596 / 0.537 / 0.323. DECODE: 0.981 / 0.985 / 0.985 / 0.987. SIMCODE: 0.977 / 0.980 / 0.981 / 0.981. [GTF frac_gt_recovered]
- Emitter recall (>= 150 photons, 0.5 / 1 / 2 / 4). SMLM: 0.361 / 0.248 / 0.136 / 0.051. SF: 0.222 / 0.124 / 0.055 / 0.020. DECODE: 0.460 / 0.371 / 0.257 / 0.156. SIMCODE: 0.592 / 0.538 / 0.430 / 0.274. [MET gt_match.recall_gt_ph150]
- Fair half-split FRC, SIMCODE, first / sustained: 62.8/32.6, 32.3/29.7, 29.5/26.6, 27.9/23.9 nm. At 0.5 the first crossing (62.8) is also caught by the null. [MET]

### 2f. The five key Results numbers (density 1; first-crossing value with the sustained value in brackets)
1. GT-FRC: SIMCODE 18.8 nm (18.2), next best SIMFLUX 28.0 nm (22.0), so a 33 % gain (17 %). DECODE 32.1 nm, SMLM 69.9 nm (40.1). [GTF, RULES]
2. Frames to 30 nm: SIMCODE 10.0k (8.2k) vs SIMFLUX 44.4k (31.9k); SMLM and DECODE never reach it. [GTF, RULES]
3. Localizations: SIMCODE 109 086 vs DECODE 74 186, SMLM 45 291, SIMFLUX 18 761. [GTF]
4. False positives: SIMCODE 2.5 % vs DECODE 7.3 %, SIMFLUX 20.9 %, SMLM 41.4 %. [GTF]
5. Jaccard 10 nm: SIMCODE 0.950 vs SIMFLUX 0.876, DECODE 0.830, SMLM 0.725. sigma-Jaccard 0.994; emitter recall 0.538 vs DECODE 0.371. [MET, GTF]

## 3. SI "Simulation of a microtubule dataset"

### PSF
- PSF: the in-situ vectorial PSF v2 of 1809/2 (models/insitu/calib_1809_2_psf_v2.json), from torch_psf.InSituEstimator, single raw frames < 30 000, 15 px ROI, 8000 train / 4000 held-out spots. [calib json 'source'; out/paper_v2psf/FACTS.md g]
- Zernike set: Noll 5-21 (17 modes), fitted. [calib json zernike_noll]
- rms aberration over Noll >= 5: 68 mlambda (68.2). [calib json psf_v2.rms_mwaves]
- Optics: NA 1.49, wavelength 0.70 um, n_imm 1.518, n_med 1.33, no blur, no per-angle dipole terms. [calib json; FACTS.md g]
- Focus: 0.20 um, fixed for all movies (within 0.0014 /px of each movie's held-out-NLL optimum). [calib json focus_um; FACTS.md g]
- Render: a 41 x 41 px ROI and a LUT over 0-0.40 um in 10 nm planes. [out/papersim/STATUS.md 2026-11-15 14:30]
- Held-out fit quality: median chi2 1.206. [calib json psf_v2]

### Emitter heights
- Rule: one height per label, the inverse CDF of the calibration's non-pinned emitter_z_quantiles_um, clipped to [0, 0.40] um, no TIRF brightness factor. webSMLM's own z is not used; it is kept as wsm_z_nm. [RUN1/gen.json z_rule; wsm_gt.py docstring]
- Quantiles p05 / p25 / p50 / p75 / p95: 0 / 21 / 116 / 207 / 400 nm. 21 % of the labels sit at 0 and 7 % at 0.40 um. [RUN1/gen.json z_dist_um, z_frac_at_0, z_frac_at_040]
- Median height 116 nm (0.1159 um). This is also the fit plane of SMLM / SIMFLUX. [RUN1/gen.json z_um]
- Detected photons: the intensity parameter is the condition photons / collected(z); collected(z) p05 / p50 / p95 = 0.54 / 0.72 / 0.96. [RUN1/gen.json photons, collected_p05_p50_p95]

### Geometry
- Pixel size 108.33 nm. [RUN1/gen.json px_nm]
- Field 40 x 40 px = 4.33 um x 4.33 um (18.8 um^2). [RUN1/gen.json W; wsm fov_um 4.3332]
- 60 000 frames per movie. [RUN1/gen.json nframes]

### Structure and labelling
- Source: webSMLM "microtubules (cells)" sample, i.e. the inSiliScope CellField world model (Martens, BSD-3), as a Python line-by-line port (analysis/fourpi_sim/mt_cells.py) checked against webSMLM (compare_websmlm.py). [wsm_gt.py docstring; out/fourpi_sim/README.md]
- Seed 11 at webSMLM's seed-fixed start view, plus an offset of (3, 3) um; view centre (231.8, -230.5) um. [RUN1/gen.json wsm.seed, offset_um, center_um]
- 31 microtubules in the field, 74.9 um of microtubule, 37 427 labels. [RUN1/gen.json wsm.n_mt_in_fov, mt_length_um, n_labels]
- Label efficiency 500/1625 = 0.308 of the lattice sites, giving 0.50 labels per nm of filament (the preprint GT's line density). webSMLM's default is 0.7. [wsm_gt.py LABEL_EFF; RUN1/gen.json wsm.label_eff, labels_per_nm]
- Lattice: a 25 nm diameter tube (12.5 nm radius), 13_3 protofilament lattice, 13 sites per 8 nm (1625 per um), random seam phase. [out/fourpi_sim/README.md 'Labels']
- Binder 12 nm radially from the tube surface; dye 2-5 nm from the binder tip (uniform direction, radius uniform in volume). [README; mt_cells.py l.19-20]
- Label shell: about 25-30 nm from the axis; in projection the labels peak at 20-24 nm and reach at most about 28 nm (p99 27.6 nm). [README; section 2b]
- Other webSMLM defaults: optical section focus 250 nm +- 750 nm; 0.9 microtubules per um^2 of cell footprint. [wsm_gt.py; RUN1/gen.json wsm.focus_nm, z_range_nm]
- GT points md5: 965fdb374b5cb04fddab921bbe1a3bae. The D and Dexp wsm sets share it. [RUN1/gen.json; D1_v2psf_wsm/gen.json]

### Kinetics
- Blinking: the preprint's two-state Markov sampler (smlmtorch blinking_spots), 10 sub-steps per frame. On-times are therefore geometric/exponential with mean 20 frames: measured mean 20.05, median 13.9 frames (= 20 ln 2). [common.blink_events; gen_hollow.py docstring; RUN1/gt.npz ev_t1 - ev_t0]
- Density = emitters ON per um^2 per frame: 0.5 / 1 / 2 / 4. [RUN<d>/gen.json density]
- How it is set: on_fraction = density x field area / number of labels; k_on follows from k_off = 1/(on-time x 10 sub-steps). [gen_density.py docstring; common.blink_events]
- Measured mean ON per frame: 9.31 / 18.79 / - / 74.85 emitters in the 18.8 um^2 field, i.e. 0.50 / 1.00 / 4.0 per um^2. [RUN0.5/RUN1/RUN4 gen.json mean_on_per_frame]
- Blink events: 27 985 / 56 233 / 112 337 / 224 882 at 0.5 / 1 / 2 / 4. [gen.json n_events; summary_density floor rows]

### SIM pattern
- 2 angles (0 deg and 90 deg) x 3 phase steps; frame f has step f % 6, angle = step // 3 (order [[0,1,2],[3,4,5]]). [RUN1/gt.npz k; common.py pattern doc]
- Pitch 224 nm (k = 3.039 rad/px). [common.PITCH_NM; RUN1/gt.npz k]
- Modulation depth 0.9 for all 6 steps. [RUN1/gt.npz m]
- Phases 0, 2 pi/3, 4 pi/3 per angle; relative intensity 1. [RUN1/gt.npz phi, rel]
- Illumination model: 1 + m cos(k_a . r - phi_s). [common.py pattern doc]
- No phase drift: the pattern is constant over the movie, and the sample does not move. [common.py "no phase drift"; FACTS.md h]

### Photons and background
- Photons are detected photons, matched to the real 1809/2 SIMCODE pass-1 statistics converted to v2 camera units (x 0.76). [gen_hollow.py docstring; STATUS.md 'Condition Dexp']
- Per event, the brightness I6 (photons per fully-on 6-frame window) is lognormal with median 694 and log-sd 0.40. These values are fitted so that the GT windows with >= 228 photons have the measured median and log-sd. [RUN1/gen.json brightness_fit.med0, sd0, window_threshold]
- Result: windows >= 228 photons have median 621 and log-sd 0.45 (p5 / p95 281 / 1274). All ON windows: median 551. [RUN1/gen.json window_photons_ge300 (key name; threshold 228), window_photons_all]
- Background: flat over the field, one level per 6-frame block drawn from the measured 1809/2 bg-per-6-sum distribution (quantile interpolation), / 6 per frame. Mean 1.95 photons/px/frame (p1-p99 1.36-2.58). [gen_hollow.py docstring; RUN1/gen.json bg_per_frame]

### Camera
- The v2 camera of 1809/2 (Andor Zyla 4.2 PLUS) on a 40 x 40 crop at (40, 40) of camera_maps_1809_2_v2.npz. [RUN1/gen.json camera; FACTS.md f]
- Gain 0.4256 e/ADU. [RUN1/gen.json camera.gain]
- Offset map: fixed pattern + 103.93 ADU, sd 1.69 ADU. [STATUS.md 2026-11-15 14:30 'Camera']
- Per-pixel Gaussian read noise: median 0.90 e, rms 1.34 e. The captions' "1.3 e- rms" is this value. [RUN1/gen.json camera.read_noise; STATUS.md]
- CRLB at the Dexp condition (16 x 16 px ROI, z free, mean over position and phase): SMLM x/y/z 12.7/12.7/58 nm; SIMFLUX (known pattern) 5.0/5.0/50 nm. [Fig 2 caption]

### Method settings (short)
- SMLM:
  - Fitter: analysis/baselines/paper_pipeline.py PP.PAPER, 8 x 8 px ROI, Gaussian sigma 1.3 px detection template on the 6-frame sum, threshold 3.5, a window at every frame. [FACTS.md b-i]
  - Fit: 2D Poisson LM-MLE (50 iterations) with the in-situ PSF v2 at the fixed plane z = 0.116 um. [FACTS.md b-i; RUN1/full/base/run.json psf]
  - Camera: the movie's gain and offset map; read noise not modelled. Pearson chi2 <= 4 x ROI^2; spots_per_bin 10 000. [FACTS.md b-i; STATUS.md; run.json]
- SIMFLUX:
  - Fit: the SMLM ROIs with modulation error < 0.1, a SIMFLUX LM fit (50 iterations), kept if |SIMFLUX - SMLM| < 0.5 px; depth fixed at 0.9. [FACTS.md b-i]
  - Pattern: estimated from the SMLM fits (Fig 2). The true pattern (sf_gtp) appears only in the SI table. [FACTS.md b-i; fig_sim_main.py TABLE_PAPER]
- DECODE:
  - Model: retrained dec_sim1809v2 (SigmaMUNet, 3-sum context) on PSF v2, LUT 0-400 nm, z U(0, 400) nm. [FACTS.md c; RUN1/full/decode/locs.npz info]
  - Training ranges: photons lognormal median 450 (log-sd 0.6) per 6-sum, bg U(6, 30) per 6-sum, density U(4, 160) per 40 x 40 px, read noise 3.28 e per 6-sum, gain 0.4256. Stopped at the 4.5 h limit (263 epochs). [locs.npz info train_variant; FACTS.md c]
  - Inference: 6-frame sums starting every 3 frames (starts 0, 3), frame labels with the 2026-11-15 pad-same fix, offset flat 103.9 ADU. [RUN1/full/decode/locs.npz info]
- SIMCODE:
  - Command: `simcode.run --cal_kind sim --crop 0,0,40`, networks m1809 (trained on PSF v2 + camera v2; staged copy out/psf_v2/eval/m1809v2/model_m1809). [RUN1/full/v2_m1809v2/command.txt, papersim_run.json]
  - Pattern estimated from the data; pass 1 + pattern solve + pass 2; no ground-truth diagnostics (--gt_diag 0). [run_wsm.sh; papersim_run.json]
  - Runtime: 157 s for 60 000 frames, density 1. [papersim_run.json run_s]

### Selection rules per method (the "comparison set" = rows that are rendered and scored)
- All methods: disjoint 6-frame windows only (window start % 6 == 0), max(sigma_x, sigma_y) < 0.15 px (16.2 nm) with each method's own sigma, inside the field [0, 40) px, no drift correction (the simulations have no drift). [metrics.py frc_set; frc_fair.py protocol]
- Sigma per method: CRLB for SMLM / SIMFLUX, predicted sigma for DECODE, calibrated sigma for SIMCODE. [frc_fair.py step 3]
- SMLM: additionally chi2 <= 4 x ROI^2 (rows of smlm.npy). [FACTS.md b-i]
- SIMFLUX: additionally modulation error < 0.1 and |SF - SMLM| < 0.5 px (rows of sf.hdf5). [FACTS.md b-i]
- DECODE: all rows of the decode table (network detection threshold raw_th 0.5 from its config). [frc_fair.py step 1; dec param_train.yaml raw_th]
- SIMCODE: pass-2 rows with p >= 0.63 inside the field and calibrated max(sigma_x, sigma_y) <= 12 nm (simcode.table.select). **May still change; see the note at the top.** [metrics.py l.8-10, l.85-95]
- Jaccard 10 nm: the preprint's compute_jaccard_points (10 nm, 120 time bins, all GT labels), computed on each method's full rows, not only the comparison set. [metrics.py doc; MET protocol]
- False positive: a comparison-set row with no emitter ON in its own window within 3 of its own sigma (Mahalanobis). [gtfrc.ghost_fraction]
