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

# 3. stage A, camera: the shipped camera values (gain 0.56 e/ADU from the robust var-vs-level fit, camera_gain.py;
#    offset 99.3 ADU, read noise 1.9 e) come from an earlier Adam fit of Zernikes + focus + offset + read noise on 8000
#    samples (fit_insitu.py, removed from the tree in the 2026-11 cleanup; it also produced the bead-fit initialisation
#    used below). They are fixed inputs of step 4.

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

## 3D PSF extension (psf3d, 2026-11-11; opt-in, defaults unchanged)

`fit_insitu2` has extra in-situ PSF freedom. A calib json without the new keys gives exactly the old PSF.

| flag | calib key | model |
|---|---|---|
| `--blur line` | `psf_blur_px2` [sxx, syy, sxy] px² | constant image-plane blur, rank 1 (σ ≈ 0.46 px along x at 0.5 nM). Applied to the oversampled intensity before pixel integration (`VectorialPSF.blur`, FFT on a grid padded by 3 px). |
| `--init-fit <stage-1 _psf.pt> --freeze-base 1 --blur angle_delta` | `psf_blur_angle_px2` [2][3] | per pattern angle (frame % 6 // 3), the total blur covariance: base ± (d, −d, e) |
| `--photosel 1/2/3` | `psf_dipole_M_angle` [2][3][3] | photoselection: emission-dipole tensor per angle (`VectorialPSF.dipM`); see `docs/PSF_POLARIZATION.md` |

Other flags:
- `--split cb0` trains on one checkerboard parity. `psf_heldout.py --test cb1` evaluates the other parity, and
  `psf_compare.py` compares the outputs (paired NLL, residual second moments per angle).

Consumers of the new keys:
- `sim.simulator.Simulator` builds one LUT per angle (`lut_angle`; `lut` = their exact average) and renders each frame with its
  angle's PSF.
- So do `hd.realsim.RealSim` (LUT-bank entries are (angle 0, angle 1) tuples), `analysis/simreal/sim_movie_r.py` and
  `sim.crlb` (`psf_stack` returns 8 channels and `crlb_all` uses the per-frame PSF).
- The pattern-free consumers (`pipeline.common.load_calib`, `psf.lut.make_lut`, the DECODE LUT with `DEC_CALIB=...`) use the
  angle average.

The recipe for the paper movies is in `out/psf3d/run_final.sh` and `out/psf3d/run_final_ps.sh`:
- stage 1: line blur (`run_final.sh`)
- stage 2: frozen + `--photosel 3` (the recommended per-axis model, `run_final_ps.sh`)
- or stage 2: frozen + `angle_delta` (the alternative, `run_final.sh`)

`analysis/psf3d/make_calib.py` writes `calib_<movie>_psf3d.json` (per axis) and `..._psf3d_avg.json` (average).
`analysis/psf3d/export_zstack.py` writes plain z-stacks for spline PSFs.
