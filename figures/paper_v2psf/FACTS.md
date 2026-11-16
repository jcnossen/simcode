# SIMCODE v2 paper: facts sheet (2026-11-15)

One fact per line, with the file that proves it. Read-only collection; no new measurements (GPU 0 was busy at 42 %, so all timings are recorded ones).
Paths relative to the repository root unless absolute. RUN = /mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/runs/simcode_v2_psfv2_20261115.

## b) SMLM / SIMFLUX baselines

### b-i) Paper simulations (analysis/papersim/run_baselines.py -> analysis/baselines/paper_pipeline.py PP.PAPER)
- Wrapper: it reuses paper_pipeline's fitter, SIMFLUX fit and pattern estimate without changes. The camera comes from the movie (camera_sim.npz). [run_baselines.py docstring]
- PP.PAPER: roisize 8, detection_threshold 3.5, pattern_frames [[0,1,2],[3,4,5]], 108.33 nm px, smlm_max_chisq 4, mod_error_threshold 0.1, fix_depth 0.9, sf_iterations 50, distFromSMLM 0.5, zrange [-0.3, 0.3]. [paper_pipeline.py:65-67]
- Detection: the paper's Gaussian sigma 1.3 px template (det_template='paper', psf_calib=[1.3,1.3]) on the 6-frame sum, threshold 3.5, moving window (every window start). [run_baselines.py main(); paper_pipeline.py docstring]
- Fit: a torch Levenberg-Marquardt Poisson MLE, 50 iterations, with parameters (x, y, I, bg). It is 2D at a fixed emitter plane psf.z_fit; z is not fitted. [paper_pipeline.py docstring, limits() l.143]
- Parameter limits: x, y in (1.5, P-2.5) px; I > 10; bg > 1 (the fastpsf limits). [paper_pipeline.py:143-144]
- chi2 filter: Pearson chi2 = sum over the ROI of (d-mu)^2/mu <= 4 x ROI^2 (= 256 for the 8 x 8 ROI). [paper_pipeline.py docstring l.6, l.110]
- SIMFLUX: modulation-error < 0.1 selection, SIMFLUX LM fit (50 iterations) of the SMLM ROIs, then |SIMFLUX - SMLM| < 0.5 px. [run_baselines.py sf_step]
- Sim-specific: smlm_spots_per_bin 10000 (the real data use 20000). [run_baselines.py SPOTS_PER_BIN_SIM]
- Two SIMFLUX variants: 'sf' (pattern estimated from the SMLM fits, as on real data) and 'sf_gtp' (ground-truth pattern). [run_baselines.py docstring]
- fig_sim_main.py renders 'sf' (estimated pattern). 'sf_gtp' appears only in the SI table. [analysis/papersim/fig_sim_main.py METH / TABLE]
- v2psf movies: fit PSF = in-situ PSF v2, models/insitu/calib_1809_2_psf_v2.json. It is NOT C.PSF_CALIB and NOT a Gaussian. [out/papersim/D*_v2psf/full/base/run.json args.psf, all 8 v2psf movies]
- v2psf movies: fit plane z_fit = 0.1159 um (the calib's emitter_z_median_um). The simulated heights span 0-0.40 um. [out/papersim/D1_v2psf/full/base/smlm/info.json psf_z_fit_um; analysis/papersim/common.py V2 doc]
- v2psf movies: camera = the movie's v2 crop (gain 0.4256 e/ADU, per-pixel offset map). [D1_v2psf/full/base/run.json gain; common.py camera_v2]
- Older sets (D*, Dexp*, moving*) used calib_1809_2_psf3d_avg.json at z_fit 0.055 um. The *_gauss sets used gauss:1.3. [out/papersim/*/full/base/run.json]
- Example D1_v2psf: 478 296 ROIs, 472 797 after chi2, median chi2/px 1.63, total 620 s. [D1_v2psf/full/base/smlm/info.json, run.json]

### b-ii) Real-data Figs 3/4 (analysis/paper/fig_exp_new.py, driven by out/paper_v2psf/run_ds.sh)
- Fig 3 tables: smlm = out/paper2/1809_2/smlm/smlm.npy, sf = out/paper2/1809_2/sf/sf.hdf5, decode = out/paper2/1809_2/decode/locs_fixed.npz. [out/paper_v2psf/1809_2_fig3_psfv2_values.json 'paths']
- Fig 4 (2.5nM_2) uses the same pattern: out/paper2/2.5nM_2/{smlm/smlm.npy, sf/sf.hdf5, decode/locs_fixed.npz}. [fig_exp_new.py dflt; run_ds.sh]
- Produced by analysis/baselines/paper_pipeline.py on 2026-11-11 (chain out/paper2/logs/chain.sh, GPU 1). [out/paper2/<ds>/run.json, file times]
- Fit PSF on real data = each movie's OLD averaged in-situ PSF (calib_1809_2_psf3d_avg / calib_2p5nM_m2_psf3d_avg / calib_20nM_m2_psf3d_avg.json). It is NOT PSF v2. [out/paper2/<ds>/run.json args.psf]
- Fit plane on real data = 0.055 um (2D, z not fitted). [out/paper2/1809_2/smlm/info.json psf_z_fit_um]
- Camera on real data = OLD camera: gain 0.56 e/ADU plus the in-situ per-pixel offset map (--camera maps). It is NOT camera v2. [out/paper2/1809_2/run.json gain 0.56, offset 'maps']
- Real-data settings otherwise as PP.PAPER: ROI 8, Gaussian sigma 1.3 detection template, threshold 3.5, chi2 <= 4 x ROI^2, mod-error < 0.1, |SF - SMLM| < 0.5 px, spots_per_bin 20000, sf_init fixed, torch fitter. [out/paper2/1809_2/run.json; paper_pipeline.py]
- 1809_2 SMLM: 9 195 522 ROIs, 8 925 529 after chi2. SIMFLUX: 1 316 157 after mod-error, 1 313 812 out. [out/paper2/1809_2/smlm/info.json, sf/info.json]
- The figures' fair-FRC filter for all methods: disjoint windows (start % 6 == 0), max(sigma) < 0.15 px, crop 5-115 px, own DME v2 drift. [fig_exp_new.py docstring; analysis/paper/frc_fair.py masks()]
- The v2psf figures reuse the SMLM / SIMFLUX / DECODE / preprint drift from the previous figure's caches (out/paper2/fig/1809_2_fig3_matched_sig12_*). [values.json drift_reused_from; out/paper_v2psf/STATUS.md]

### b-iii) Paper text "8 x 8 px ROI, Gaussian sigma 1.3 px, chi2 filter" vs now
- 8 x 8 px ROI: unchanged. [PP.PAPER roisize 8]
- Gaussian sigma 1.3 px: now only the DETECTION template. The FIT PSF is an in-situ vectorial spline PSF at a fixed height: psf3d_avg at 0.055 um on real data, PSF v2 at 0.116 um in the v2psf sims. [run.json files above; paper_pipeline.py docstring]
- chi2 filter: unchanged in form (Pearson chi2 <= 4 x ROI^2, from the published config). [paper_pipeline.py docstring]
- Not in the text: camera = per-pixel offset map + gain 0.56 (real) instead of the scalar 106.34 ADU. The scalar over-subtracts about 7 ADU. [paper_pipeline.py docstring 'Camera']
- Not in the text: SIMFLUX starts each ROI from its own SMLM fit. The published code indexed another ROI's fit (--sf_init fixed vs paper). [paper_pipeline.py docstring]
- Not in the text: torch LM-MLE (same likelihood / model as fastpsf) instead of fastpsf, which has no spline-PSF SIMFLUX. [paper_pipeline.py docstring]
- Inconsistency to note: on real data the baselines use the v1 PSF (psf3d_avg) and the old camera (gain 0.56). SIMCODE v2 uses PSF v2 + camera v2 (gain 0.4256). [out/paper2/1809_2/run.json; RUN/summary.json calib]

## c) DECODE

- Driver: analysis/baselines/decode/decode_paper.py, one model per PSF. It uses DECODE's simulator through dec_common.LutDecodeInstance, from the fork the DECODE fork decode2. [decode_paper.py docstring]
- Network: SigmaMUNet, channels_in 3 (3-sum context), depth_union 2, sigmoid intensity scale, MixtureModel loss. [out/paper2/decode/lutpsf/dec_1809_2/param_train.yaml]
- Training: batch 32, lr 2e-4, ReduceLROnPlateau, pseudo_ds_size 8192 frames per epoch, 40 x 40 px frames. [same param_train.yaml]
- LUT: zpsf PSF rendered to 41 x 41 px, 33 x 33 sub-pixel shifts, bilinear sub-pixel / linear z, 10 nm planes. [decode_paper.py psf_luts/setup_psf]
- Real-data models: dec_1809_2 / dec_2.5nM_2 / dec_20nM_2, each trained on its movie's psf3d_avg PSF (NOT v2). [lutpsf/dec_*/psf.json; out/paper2/logs/chain.sh]
- Real-data model ranges: LUT 0-200 nm; emitter z U(10, 150) nm; gain 0.56; read noise 2.20 e per 6-sum (0.9 sqrt 6). [dec_1809_2/psf.json, param_train.yaml]
- Real-data model ranges: photons lognormal median 1000 / 6-sum, log-sd 0.6; bg U(22, 120); density U(4, 45) per 40 x 40 px; on-time 10.8 sums; label_min 150. [dec_1809_2/psf.json]
- Real-data epochs: dec_1809_2 380 epochs (0-379), stopped at the 4.5 h limit. dec_2.5nM_2 400 epochs (4.14 h). dec_20nM_2 400 epochs (3.90 h). [out/paper2/logs/dectrain_<ds>.log]
- v2psf-sim model: dec_sim1809v2, PSF v2 (calib_1809_2_psf_v2.json), LUT 0-400 nm (41 planes), z U(0, 400) nm, gain 0.4256, read 3.28 e per 6-sum. [lutpsf/dec_sim1809v2/psf.json; out/papersim/decode_v2psf/train_dec_sim1809v2.sh]
- v2psf-sim ranges: photons median 450, log-sd 0.6; bg U(6, 30); density U(4, 160); on-time 1 sum; label_min 60. [dec_sim1809v2/psf.json]
- v2psf-sim epochs: 263 (0-262), stopped at the 4.5 h limit. [out/papersim/decode_v2psf/train_dec_sim1809v2.log]
- dec_sim1809v2 param_train.yaml has an excitation_pattern block, but the variant has pattern False. Whether it is used: UNKNOWN (not traced). [dec_sim1809v2/param_train.yaml, psf.json]
- Inference: non-overlapping 6-frame sums starting at s + 6k for s in {0, 3}, i.e. window starts w % 3 == 0. The preprint used s = 0 only. [decode_paper.py infer(), docstring; out/paper2/1809_2/decode/locs.npz info 'starts' "0,3"]
- Inference camera (real): the movie's old in-situ offset map, gain 0.56, photons = (sum of 6 frames - 6 offset) x gain. [decode_paper.py infer; locs.npz info]
- x/y convention: the fork the DECODE fork decode2 swaps XY internally (x = last image axis). [the DECODE fork decode2/decode/neuralfitter/coord_transform.py:36, loss.py:230/383 'JC: XY swapped'; fork commit 5ed7932]
- x/y convention: our pipeline applies no further swap. The output x = column, pixel centre = integer, which is the smlmtorch / fastpsf / simcode convention. [decode_paper.py docstring; dec_common.py:270; simcode/dataset.py from_decode]
- Sim smoke check: offsets < 2 nm between methods ("conventions consistent"). [out/WORKLOG.md l.344]
- Frame-label bug: the fork's InferenceDataset has pad=None, so output i belongs to centre sum i + 1. All DECODE rows were one 6-frame window early, and the first window was never localized. [out/decode_checks/STATUS.md 'Finding 0']
- Evidence: iso-movie recall 0.003 as stored vs 0.636 with +6 frames. [out/decode_checks/STATUS.md]
- Fix in code: dec_common.process_padded repeats the first / last sum (pad 'same'). It is on branch decodefix (commit 8c3b19d), NOT yet merged into main. [out/decode_checks/STATUS.md; git merge-base check]
- Fix for existing tables: locs_fixed.npz (frame + 6) next to every locs.npz; the originals are unchanged. info.frame_fix.shift = 6. [out/paper2/<ds>/decode/locs_fixed.npz info]
- Figs 3/4 v2psf use locs_fixed.npz. [1809_2_fig3_psfv2_values.json paths.decode; run_ds.sh comment]
- Effect: GT-FRC, Jaccard and renders are unchanged. Window-based false-positive fraction, recall and RMS change (e.g. D1_v2psf FP 0.258 -> 0.118). [out/decode_checks/decode_fix_compare.md]
- Fig 3 DECODE FRC 28.14 nm, n_frc 1 037 823: identical before and after the fix. Its drift is reused from the pre-fix cache. [out/paper2/fig/1809_2_fig3_matched_sig12_values.json vs out/paper_v2psf/1809_2_fig3_psfv2_values.json]
- DECODE pixel locking: x A_bias +6.0 to +9.4 nm at D0.5-D4. Isolated emitters: +3.9 / +1.3 nm (x / y). [out/decode_checks/STATUS.md section 2]
- DECODE precision on isolated emitters: RMS/CRLB 0.89-1.24 (200-1200 photons); recall about 0.75 at 500-750 photons. [out/decode_checks/STATUS.md section 1]

## d) Throughput

- SIMCODE v2 1809_2 run: command `python -m simcode.run --dataset 1809_2 --out RUN`, started 2026-11-15T10:42:00+01:00, git ba22cbe (clean). [RUN/command.txt; RUN/run_config.yaml l.297-300]
- 120 000 frames (2 TIFF parts), 20 chunks of 6000 frames. [RUN/log.txt line 1]
- Done line: {"pass1_pattern_s": 601.1, "solve_s": 76.1, "pass2_s": 397.1, "total_s": 1179.5}. [RUN.log l.91 (= /mnt/data/.../runs/simcode_v2_psfv2_20261115.log); RUN/summary.json timing]
- Total 1179.5 s = 19.7 min; 120 000 / 1179.5 = 102 frames/s, end to end. [derived from total_s]
- GPU: one NVIDIA GeForce RTX 3090 (24 GB), CUDA_VISIBLE_DEVICES=1, torch 2.13.0, CUDA 13.0. [RUN/run_config.yaml l.306-313]
- Per chunk at steady state (chunks 12000-114000): pass 1 about 24.4-26.9 s, pass-2 pattern solve about 2.9-3.1 s, pass 2 about 18.5-20.3 s wall. Pattern estimate 42.5 s once. [RUN/runtime.json]
- Steady-state rate about (25.0 + 3.0 + 19.6) s per 6000 frames, about 126 frames/s. [derived from RUN/runtime.json]
- Chunks 0 and 6000 took 53.4 s / 43.4 s in pass 1. [RUN/runtime.json]
- GPU sharing: the GPU sampler shows a second process on GPU 1 from 10:42:07 to 10:43:28 (9 samples, 10 s apart). After that only one PID until the end. The identity of the second PID 785814: UNKNOWN. [out/archive/tput/gpu_samples.log, epochs 1794735720-1794736920]
- Share of the chunk 0/6000 slowdown due to sharing vs compile / warm-up: UNKNOWN. [runtime.json 'fast.compile': auto]
- DECODE on 1809_2 (dec_1809_2, 120 000 frames, starts 0,3): infer_s 1612.6 s = 26.9 min = 74 frames/s, GPU 1. [out/paper2/1809_2/decode/locs.npz info; out/paper2/logs/decinfer_1809_2.log; chain.sh]
- That DECODE 1809_2 timing is NOT clean: in every sample GPU 1 also held another PID (1117789; 1134801 present in 99 of 160 samples). [gpu_samples.log, 2026-11-11 10:52-11:19]
- DECODE 2.5nM_2 (180 000 frames): 334.2 s = 539 frames/s, with a single PID on GPU 1 during the run. [out/paper2/2.5nM_2/decode/locs.npz info; gpu_samples.log]
- DECODE 20nM_2 (120 000 frames): 957.0 s, GPU 1 shared (3 PIDs). [out/paper2/20nM_2/decode/locs.npz info; gpu_samples.log]
- Excluded from the DECODE time: training (4.5 h limit per model, one RTX 3090) and the frame relabel. [dectrain logs]
- SMLM + SIMFLUX on 1809_2 (real): total 1037 s (detect 199, SMLM fit 650, pattern 78, SF fit 102), GPU 1. GPU sharing: UNKNOWN. [out/paper2/1809_2/run.json timing]
- SIMCODE v2 training of m1809 (PSF v2): chain 05:06-10:27 on 2026-11-15, one RTX 3090 (GPU 1); the p2det and p2head stages ran concurrently. [out/psf_v2/train/stages.log]
- Archived throughput study (tag archive-2026-11-15: analysis/tput/bench_tput.py, summarize.py; data out/archive/tput): one 6000-frame chunk, RTX 3090 on GPU 1. Networks: the earlier hd_sc5nb / det_n1 / p2_head_long, not today's models. [bench_tput.py header; bench_*.json]
- Archive 2.5 nM chunk: original path, pass 1 31.3 + solve 89.4 + pass 2 144.9 = 266 s (23 frames/s). Fast path: 31.0 + 7.8 + 45.9 = 85 s (71 frames/s). Fast compiled: 30.9 + 5.1 + 42.5 = 79 s (76 frames/s). [out/archive/tput/bench_2.5nM.json]
- Archive 0.5 nM chunk: original 12.8 + 29.9 + 46.7 = 89 s (67 frames/s); fast compiled 12.3 + 1.6 + 14.6 = 28.5 s (210 frames/s). [out/archive/tput/bench_0.5nM.json]
- Archive m20qh (sim, 4800 frames): original 260 s (18 frames/s); fast compiled 86 s (56 frames/s). [out/archive/tput/bench_m20qh.json]
- Archive cleanliness: summarize.py keeps only sections with no foreign GPU process. Only one 2.5 nM orig_pass1 measurement passed (clean_2.5nM.json); none passed for 0.5 nM (clean_0.5nM.json = {}). The bench numbers above are therefore from a shared GPU. [out/archive/tput/clean_*.json; summarize.py docstring]

## e) Which v2 outputs the figures use

- Real-data figures: v2 rows = Dataset.select(12) of RUN locs_p2: p >= 0.63, calibrated max(sx, sy) <= 12 nm, run crop. [run_ds.sh --v2_sigma_max 12; RUN/summary.json main_rule]
- Columns plotted / evaluated: x, y, frame, sx, sy (FRC, counts, renders, Pearson). Photons are loaded but the renders use intensities=False. [analysis/paper/frc_fair.py load(); fig_exp_new.py shared render args]
- Drift: DME v2 in 2D on x, y. [frc_fair.py dme_v2]
- z on real data: used ONLY in the PSF panel / caption. The median pass-1 z of the plotted set (112 nm) sets the xy-section height and the collected fraction (0.73) for the detected-photon CRLB. [analysis/paper/psfdata.py; out/paper_v2psf/STATUS.md]
- The PSF-panel photons are the v2 network's photons = the PSF model's intensity parameter (median 861 / window), shown as about 626 detected. [out/paper_v2psf/STATUS.md 'Photon units']
- Pass-2 rows take z from pass 1 (linked candidate); 0.18 % of the selection has no z. [docs/METHOD.md stage 5; out/paper_v2psf/STATUS.md table]
- Sims: no z in the metrics; grep finds no z use in analysis/papersim/metrics.py, gtfrc.py, fig_sim_main.py. The sim PSF panel is drawn at the calib's emitter_z_median_um (116 nm). [psfpanel.psf_info z0 = psf.z_fit; out/papersim/figures/fig_sim_Dexp_v2psf_caption.txt]

## f) Camera calibration v2

- Gain 0.4256 e/ADU: joint white-noise photon transfer of 4 movies. Power of lag-6 same-step frame differences at |f| > 0.50 cycles/px (beyond the OTF cutoff 0.461), binned by level above the offset map. Leave-one-out 0.424-0.428. [out/psf_v2/STATUS.md section 1; out/psf_v2/whitegain.py; build_camera.py docstring]
- Why the old 0.56 was wrong: pixels were binned by absolute level, so the offset fixed pattern flattened the slope. [out/psf_v2/STATUS.md Bottom line 1]
- Offset map: per-pixel fixed-pattern offset + global 99.3 + DELTA = 103.93 ADU. 1809_2 uses its own stored map. 2.5 / 20 nM use the 0.5 nM map shifted (22, 2), with the uncovered pixels from the 2.5 nM map minus its sample structure. [out/psf_v2/build_camera.py docstring; STATUS.md]
- Global offset / read-noise split: not identifiable without darks. It was fixed by assuming a 0.9 e median read noise (the stored value, Zyla 4.2 PLUS spec) on the 18 Sep movies. [build_camera.py docstring]
- Read-noise map: per pixel rn^2 = var6 - (level - DELTA)/g from same-step frame differences, high-passed (median 7), plus the 0.9 e median, floor 0.3 e. Result: median 0.90 e, mean rn^2 1.8-1.9 e^2. [build_camera.py docstring; STATUS.md]
- Dark-frame check: no 2023 darks exist. The 2022 darks cannot be registered to the 2023 sensor regions. A dark-based offset would be 104.17 ADU (1809_2) vs 103.93 used; effect bg -0.11 ph/px, photons < 0.2 %, xy ~1 nm. Recommendation: keep v2. [out/camera_dark/STATUS.md items 1, 3, 7, Recommendation]
- Camera: Andor Zyla 4.2 PLUS (ZYLA-4.2P-USB3, serial 8984), 16-bit low noise & high well capacity, 540 MHz, baseline clamp on, 5.5 ms. [out/camera_dark/STATUS.md item 2; out/psf_v2/STATUS.md Metadata]
- Validation: white noise of spot residuals / model = 1.03 / 1.03 / 1.00 (1809_2 / 2.5 nM / 20 nM). [out/psf_v2/STATUS.md Bottom line 3]

## g) In-situ PSF v2 (Methods)

- Estimator: torch_psf.InSituEstimator, vectorial PSF, Zernike Noll 5-21, refine=False. [out/psf_v2/psf_refit.py docstring; models/insitu/calib_1809_2_psf_v2.json 'source']
- Optics: NA 1.49, lambda 0.70 um, n_imm 1.518, n_med 1.33, 108.33 nm px. [out/psf_v2/STATUS.md section 2; calib json]
- Spots: single raw frames 0-30 000, ROI 15 px, threshold 6. 8000 train / 4000 held-out spots (checkerboard). 1809_2: 24 000 spots detected in 2600 frames read. [psf_refit.py; out/psf_v2/result_1809_2_v2p.json report]
- Overlap rejection: median-deviance rule (median + 3 x scale). 1809_2 rejected 2.9 % (235); 2.4-3.0 % across movies. [result_1809_2_v2p.json report.deviance; STATUS.md]
- Camera: camera v2 maps (gain 0.4256). [result_1809_2_v2p.json camera]
- Fitting: two starts, A neutral and B stored Zernikes, each polished with 40-iteration restart rounds until the held-out NLL gains < 1e-4 / px. The lower held-out NLL wins; for 1809_2, A won. [STATUS.md section 2]
- Focus: fixed at 0.20 um for all movies. It is within 0.0014 / px of each movie's held-out-NLL optimum. The mirrored defocus branch is worse in all movies. [STATUS.md 'Focus']
- Blur: none. On 2.5 / 20 nM, blur and wavelength ran away (no converged optimum); for 1809_2 blur gave no gain. [STATUS.md 'Blur']
- Dipole model: default; no per-angle photoselection terms (not carried over from psf3d). [STATUS.md section 3]
- Result 1809_2: rms aberration 68 mlambda (Noll >= 5). Held-out NLL -0.00128 / px vs the stored PSF; held-out chi2 1.206. [STATUS.md table]
- Heights: 4000 held-out spots re-fit. Fits pinned at the limits were dropped (1809_2: 2.4 % / 2.5 %). Non-pinned quantiles define the training heights, clipped to [0, 0.40] um; 1809_2 median 0.116 um. [STATUS.md 2026-11-15 05:05; zfit_nonpinned.py, write_heights.py]
- Open issue: for 2.5 / 20 nM, 1/3 and 1/5 of the spots are pinned at the z limits (broader than the model). [STATUS.md OPEN ISSUE]

## h) Preprint SIMCODE runs

- Sims: analysis/papersim/run_simcode_v1.py runs the published pipeline (analysis/baselines/run_simcode.py) unchanged. Network sf_conv_g1.3_tio2_L48 checkpoint_3, --pattern est, --fix_sign 0, i.e. with the published drift-sign behaviour. [run_simcode_v1.py docstring; D1_v2psf/full/v1/v1_info.json fix_sign 0]
- Sims, v1gt: the same with --pattern gt, also fix_sign 0. [D1_v2psf/full/v1gt/v1_info.json]
- Sim settings: p >= 0.7, crlb 0.2 px, depth 0.9, mod error 0.1, |ndi - net| < 0.2 px, MVU merge ('sfhd-ndi+smlm'). [v1_info.json settings]
- Sim camera for v1: GAIN = 0.4256 and a scalar OFFSET = the mean of the offset map (103.906 ADU); no per-pixel map. [D1_v2psf/full/v1/v1_info.json; run_simcode_v1.py]
- The v1 network was trained on a Gaussian sigma 1.3 px PSF, so it is mismatched to the vectorial-PSF movies. [run_simcode_v1.py docstring]
- The sim movies have a constant pattern (no phase drift in gen.json / gen_density.py). The sign issue therefore acts only on the estimated, noise-level per-frame offsets. Its size in the sims: UNKNOWN. [out/papersim/D1_v2psf/gen.json; grep gen_density.py]
- Fig 3 internal preprint column: out/paper/1809_2/published/simflux_5.5ms_500mW/sfhd-ndi+smlm.hdf5. This is the published result file of the 2025-01-20 processing (config.pickle dated 2025-01-20), read through the same fair protocol (loaded as kind 'sf'). [1809_2_fig3_psfv2_internal_values.json paths.v1; fig_exp_new.py --v1]
- Fig 4 internal preprint column: out/paper/2.5nM_2/published/sfhd-ndi+smlm.hdf5. There is no preprint column for 20 nM. [run_ds.sh V1]
- Internal figures are marked "INTERNAL (not for the manuscript)". [run_ds.sh NOTE]
- Nature of the sign issue: estimate_phases stores the per-frame phase offset as phase(f) - mean. ModulationPattern.mod_at_frame applies phase = mod.phase - offset (changed from += in smlmtorch fc2cc68, 2024-05-05, before the 2025-01-20 processing). The estimated phase drift is therefore applied with the wrong sign (error about 2x the drift). [analysis/baselines/run_simcode.py docstring l.19-22]
- Effect on real data: after own-DME drift correction no FRC gain on the same rows (25.5 vs 25.5 nm). The paper sign keeps about 18 % fewer rows. Raw published positions carry a time-varying error up to +-20 nm. [docs/RESULTS.md l.41-55]
- Which movie those RESULTS.md sign numbers refer to: not stated in the excerpt; UNKNOWN without further reading.

## i) Model training summary (Methods)

- One model = 4 networks: pass-1 detector, pass-1 emitter network (head), pass-2 detector, pass-2 head, plus solve calibrations. [models/README.md 'Models']
- m1809 (current, 2026-11-15) is trained on PSF v2 + camera v2 (calib_1809_2_psf_v2.json). Commit ba22cbe; the previous model is in models/archive_psf_v1/m1809. [models/m1809/model.json psf / _recipe.note; git log ba22cbe]
- Stale doc: the models/README.md table still lists m1809's PSF as calib_1809_2_psf3d_avg.json. [models/README.md table vs models/m1809/model.json]
- m25 (2.5 / 20 nM) is trained on calib_2p5nM_m2_psf_v2.json (commit ddd3e14). [models/m25/model.json; README table]
- Recipe m1809 (PSF v2): detector 40k steps (lr 1e-3, batch 16, 64 px). Head 18k steps fine-tuned from hd_sc5nb.pt (lr 1.5e-4, 96 ch, bf16), 3D. [models/m1809/model.json _recipe.note; out/psf_v2/train/chain.sh]
- Recipe m1809 pass 2: detector p2det 40k (p_std 0.25) -> p2det_c 40k (p_std 0.5). Head p2head 24k -> p2headL 50k -> p2headL2 50k (p_std 0.25 / 0.25 / 0.5). [chain.sh; model.json]
- The "p_std 1 -> 0.5" detector recipe applies to m25 ONLY: dedicated d2ded p_std 1, 40k, lr 3e-4 -> gft1 p_std 0.5, 40k, lr 3e-4 -> gft2 p_std 0.5, 40k, lr 1e-4. m1809 does NOT use it. [models/README.md 'Pass-2 detector of m25'; models/m25/model.json _recipe]
- m25 pass-2 detector vs dedicated: RMS/CRLB iso +0.002, pair1 -0.020, pair2 -0.024, rs25qm -0.012; recall >= dedicated. [models/README.md; commit ddd3e14]
- Generic pass 2: in a share 1 - p_std of the FOVs, pitch log-uniform 1.8-4.0 px, orientations -40..40 / 50..130 deg, random phases / frame order, depth 0.5-1.0, power 0.8-1.25; the other FOVs use the SIMCODE pattern. [models/README.md; simcode/simulate/acquisition.py]
- Simulator: AcquisitionSim. With a v2 calibration, heights = piecewise-linear inverse CDF of emitter_z_quantiles_um, clipped to [0, 0.40] um, with no TIRF brightness factor. The read noise is the calibration's camera map. [simcode/simulate/acquisition.py docstring l.19-25, height_quantiles, draw_z, tirf_factor]
- Simulator photons: drawn as the PSF model's intensity parameter (render = phot x psf), so detected = photons x collected(z). [analysis/papersim/common.py V2 'photons'; out/paper_v2psf/STATUS.md]
- 3D output: the pass-1 head outputs a pattern-free (x, y, z) prior with a 3 x 3 covariance. z is conditioned on the solved x, y (z_given_xy). [docs/METHOD.md stages 1 and 4]
- Stale doc: METHOD.md says "trained on heights 0-0.3 um, half uniform" (the pre-v2 rule). For v2 models the heights come from the calibration. [docs/METHOD.md stage 1 vs acquisition.py draw_z]
- z of pass-2 rows comes from pass 1: one-to-one link within 1 px, mutual nearest neighbours; unlinked rows get z = NaN. The pass-2 head's own z is biased (+32 nm at low photons) and not used. [docs/METHOD.md stage 5]
- Solve: per-row LM of (x, y, B[, t_on, t_off]) on the network's I_t with covariance Sigma = s Sigma_net + c^2 diag(mu^2) + c_a^2 mu mu^T. Starts: prior + 8 fringe-shifted starts; fringe posterior mean. [docs/METHOD.md stages 3-4]
- c term: fixed, cmult = 0.1, cmult_ang = 0.1, from the model's solve calibrations (pass 1 and pass 2); autoc = 0. [RUN/run_config.yaml l.20-21, 144-145, 213-214]
- Selection: p >= 0.63 inside the crop and calibrated max(sx, sy) <= 12 nm (user decision 2026-11-13). Stored candidate cut p >= 0.3. [docs/METHOD.md stage 6 / production recipe; RUN/run_config.yaml l.18-19, 345, 355]
- 1809_2 run: pass-2 selection n = 2 127 865 (12 nm rule), median photons 860; pass-1 n = 1 837 311. [RUN/summary.json]
- Calibration fit: gate scenes rendered with the model's PSF -> analysis/psfmatch/fit_cal_set.py. The real-data multiplier sig_ph_real_mult was copied from the earlier calibrations (not re-measured). [models/m1809/calib/calib_sc_hd_m1809_real.json note]
