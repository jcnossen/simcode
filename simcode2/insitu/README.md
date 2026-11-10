# In-situ calibration (PSF, camera, pattern)

Everything is estimated from the DNA-PAINT movie itself plus the SIMCODE pipeline's own `smlm` localizations of it
(Picasso hdf5 in the dataset's `results/` folder), which are used only to find isolated emitters and to mask emitters
out of the camera maps. A bead z-stack fit only provides the starting Zernikes. The result is a
`models/insitu/calib_<movie>.json` plus its camera maps `camera_maps_<movie>.npz`; `--calib insitu/calib_<movie>.json`
then works with every entry point.

Inputs are absolute or relative to the data root (`simcode2.json` key `data`). Default outputs go to `<out>/insitu/`.
Run with `CUDA_VISIBLE_DEVICES=1` and `PYTHONPATH=<repo>` (or `pip install -e`).
The defaults are the ones used for the shipped 0.5 nM calibration (`calib_0.5nM.json`).

## Recipe (0.5 nM, as shipped)

```bash
# 0. (once per setup) bead z-stacks -> starting Zernikes. Output: <out>/insitu/psf_fit_sign-1_na149_ball_drift.{json,pt}
#    (shipped as models/insitu/beads/). NA 1.49, lambda 0.70 um, n_med 1.33 fixed; bead = 32-point ball; per-stack xy drift; stack 7 held out.
python -m simcode2.psf.fit_beads --sign -1 --nball 32 --tag _na149_ball_drift

# 1a. gain / read noise: robust variance-vs-level fit of same-step frame differences (0.5 nM, 12000 frames:
#     gain 0.552 e/ADU -> 0.56 used; rn 0.88 e at offset 99.3)
python -m simcode2.insitu.camera_gain --movie dnapaint/0.5nM/cover1/1/simflux_5.5ms_500mW.tif

# 1. camera maps: per-pixel offset (loc-masked temporal median minus its smooth part, + global offset 99.3) and read noise
#    (MAD of same-step frame differences minus shot noise, floor 0.9 e), gain 0.56. The shipped map used the first
#    10000 frames (--nframes 10000; the argparse default is 20000): this command reproduces camera_maps_0.5nM.npz bit for bit.
python -m simcode2.insitu.camera_maps --movie dnapaint/0.5nM/cover1/1/simflux_5.5ms_500mW.tif --nframes 10000 \
    --locs dnapaint/0.5nM/cover1/1/results/simflux_5.5ms_500mW/smlm.hdf5 --out <out>/insitu/camera_maps_0.5nM.npz

# 2. isolated emitter tracks: smlm locs of frames < 30000 linked over frames (<= 0.8 px, 1-frame gaps allowed),
#    >= 6 frames, no other loc within 7 px; cut into 11 x 11 x 12-frame ADU samples (0.5 nM: 52863 samples).
python -m simcode2.insitu.extract_tracks --out <out>/insitu/tracks_0.5nM.npz

# 3. stage A, camera: Zernikes + focus + offset + read noise (+ optionally gain) by Adam on 8000 samples, per-frame I
#    and bg. Gain is not identifiable from sparse data: fix it (0.56, robust var-vs-level fit). The development runs
#    (only their json outputs are recorded, not the full command lines): --gain 0.56 -> offset 99.34 ADU, rn 1.90 e;
#    + --fit-apod 1 -> 99.21 ADU, rn 1.89 e. The shipped values are offset 99.3 ADU and rn 1.9 e (the read noise absorbs
#    model error; the solve uses 0.9 e and the per-pixel map).
python -m simcode2.insitu.fit_insitu --gain 0.56 --out <out>/insitu/insitu_0.5nM_g056

# 4. stage A, PSF (variable projection; this is what calib_0.5nM.json holds): per-frame I and per-(sample, angle) bg
#    solved exactly (varpro.py) each iteration, x, y, z per sample + Zernike 5..21 + focus by Adam (600 iterations,
#    8000 samples), gain 0.56 / offset 99.3 / rn 1.9 fixed, z in [0, 0.2] um. Writes <tag>_psf.json/.pt and the
#    per-frame intensities of all samples (<tag>_intensities.pt, PSF frozen) for stage B.
python -m simcode2.insitu.fit_insitu2 --gain 0.56 --zmax 0.2 --out <out>/insitu/insitu2_0.5nM

# 5. stage B, pattern (initial estimate): I = A relint_s (1 + m_s cos(k_a.r - phi_s - dphi_a(t))) on fully-on frames,
#    k / phases initialised from the SIMCODE pipeline's sf_fits_cfg.pickle, piecewise-linear phase drift (1500-frame knots).
python -m simcode2.insitu.fit_pattern --intensities <out>/insitu/insitu2_0.5nM_intensities.pt --out <out>/insitu/pattern_0.5nM
```

Then assemble `models/insitu/calib_<movie>.json` (see `calib_0.5nM.json` for the format):

- `zernike_noll` = 5..21, `zernike_waves` = `zernike`, `focus_um`, `NA`, `apod` from step 4's `_psf.json`
- `pixelsize_um` 0.10833, `wavelength_um` 0.70, `n_imm` 1.518, `n_med` 1.33, `tirf_depth_um` 0.082,
  `emitter_z_median_um` (step 4)
- `camera`: `gain_e_per_adu` 0.56, `offset_adu` 99.3, `readnoise_e` 0.9, and `maps` = the camera-map file name, relative
  to the json (copy the npz next to it)
- `pattern`: k, phase and relint from step 5. The shipped depth (0.925 / 0.955) is the later cross-cycle estimate on the
  nn_overlap ≥ 3 subset. Step 5's regression depth is diluted and must not be written there (see `depth_note`).
  The localization pipeline re-estimates the pattern from the data (`pipeline/pattern_est.py`). It uses only the PSF and
  camera of `--calib`; the pattern block is read only by the simulator.

Other movies of the same setup (`calib_2p5nM_m2.json`, `calib_20nM_m2*.json`) reuse the 0.5 nM PSF, i.e. they copy the
fields above. They only get their own camera maps:

```bash
python -m simcode2.insitu.camera_maps --movie dnapaint/2.5nM/2/cover1_600mW_5.5ms_cf660.tif \
    --locs <its results/.../smlm.hdf5> --readnoise_from insitu/camera_maps_0.5nM.npz --out <out>/insitu/camera_maps_2p5nM_m2.npz
```

The read-noise map is a camera property. It is taken from the 0.5 nM map, shifted by the sensor-ROI offset found by
cross-correlation (22, 2). Uncovered pixels get the high-passed own estimate. The 20 nM movie-2 offset map is the
shifted 0.5 nM map. The `camera.note` fields of the shipped jsons record which variant each file uses.

`compare_pattern` (`python -m simcode2.pipeline.compare_pattern --est <pattern_est.pt> --ref <pattern.pt> --nframes N`)
compares an estimated pattern with a reference one (k, phases with the per-angle gauge, depth, relint, eps(t), and the
per-tile residual phase of clean pass-1 emitters with `--pass1`).

## Diagnostics (not needed for a calibration)

- `fit_psf_sums.py`: PSF refit on high-SNR 12-frame sums of bright isolated emitters. Free: Zernike 5..21, focus, and
  with `--field 1` a linear field dependence of Noll 5..11 (the field-dependent PSF estimate). 50 % held out.
- `coma_anchor_scan.py`: pattern-anchored profile scan of the odd aberrations (Noll 7 / 8, or the SAF terms with
  `--scan saf`), using the full-model LM of `lm_refine.py`. This is a calibration/diagnostic helper only, not part of the
  localization method.

Both read a locs npz in the format of the development tree's legacy localize_real output (`x_ref, y_ref, I_ref [N,12],
I_pre, I_post, phot, z, wstart, crop, f0, f1`). The two files used (`real_v6_p4_f0.npz`, `real_v6_p5m_f0.npz`) are
copied to `<out>/insitu/legacy/`; that front-end is not part of simcode2.

## Check (2026-11-11)

- `camera_maps --nframes 10000` reproduces `models/insitu/camera_maps_0.5nM.npz` exactly (all arrays).
- `extract_tracks` (defaults) reproduces the development track file exactly (52863 samples; copy in `<out>/insitu/tracks_0.5nM_dev.npz`).
- `fit_insitu2` on the development track file reproduces `calib_0.5nM.json` exactly (focus and all 17 Zernikes, residual
  variance 0.996).
- An independent subset of 8000 other samples from the same tracks: focus 0.1127 vs 0.1128 um, Zernikes within 0.003 waves
  (rms 0.0014).
- Fresh tracks from frames 0–6000 only (10576 samples): focus 0.1116 um, Zernikes within 0.010 waves (rms 0.0035; largest:
  Noll 5 −0.010, Noll 15 −0.007), residual variance 1.006.
