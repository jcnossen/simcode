# Pattern estimation: the facts behind today's 1809_2 / 2.5 nM patterns (SI paragraph)

Code as run: branch main of the repository (runs 1809/cover1/2/runs/simcode_v2_psfv2_20261115 and dnapaint/2.5nM/2/runs/simcode_v2_psfv2_final).
Line numbers: figfix rebased on main (2026-11-15 evening); the pattern code is unchanged on main since these runs.
Both runs' run_config.json 'fixed': nn_min 3.0, min_phot 300.0, cand_pmin 0.3, p_good 0.63, pattern_pitch_search_px [1.6, 2.8], window_step 3.

## Input rows
- Input = all stored pass-1 candidates, detector p >= CAND_PMIN = 0.3 (simcode/processing/common.py:13; front_fast.py:189). There is no p >= 0.63 cut and no sigma cut anywhere in the pattern estimation.
- Each window is taken once across chunk boundaries, and the head's constant prior-position bias prior_off is subtracted from x, y first (run_uncalibrated.py:147-148; common.py:27-38).
- Position = the pass-1 head's pattern-free position prior. Per-frame intensities I_j = the pass-1 network's linear read-out intensities (front_fast.py:264).
- Photon units: photo-electrons = detected photons, frames = (ADU - offset map) x gain of the model calibration, camera v2 gain 0.4256 e/ADU (front_fast.py:248).
- Rows that are evaluated ('out'): time centroid tc of the 6-frame intensities in [1, 4) (central half of the window) and inside the crop (common.py:40).

## Selection of 'clean' emitter-windows (pattern_estimation.select, pattern_estimation.py:59-87)
- **On through the window:** some detection (any stored candidate) lies within link_r = 1.0 px of the row in BOTH windows w - 6 and w + 6. These are the photon-disjoint neighbouring windows, so the event spans the whole window (pattern_estimation.py:59, 67, 76-77).
- **Positive intensities:** all six I_j > 0 (pattern_estimation.py:80).
- **Photon count:** phot >= min_phot = 300 photons (detected), where phot = the sum of the six pass-1 network intensities of the window, I.sum(1) (front_fast.py:264; pattern_estimation.py:80; default run_uncalibrated.py:55). The threshold is on the window sum, not per frame.
- **Isolation:** nn >= nn_min = 3.0 px (325 nm), where nn = the distance to the nearest OTHER detection of the SAME window (same window start w; all stored candidates p >= 0.3 count as neighbours, not across frames / windows) (pattern_estimation.py:66, 85; default run_uncalibrated.py:54).
- **How nn is computed:** spatial hash (simcode/util/spatial_hash.nearest, spatial_hash.py:162: sorted-grid cells keyed by window start, exact distances); NB_BACKEND = 'hash' (pattern_estimation.py:44). Search radius max(NN_CAP = 5 px, nn_min); beyond that nn = inf (pattern_estimation.py:47, 66).
- **Averaged position:** for on-through rows the position used by the estimator is the mean of the linked positions in windows w - 6, w, w + 6. Its sigma is divided by sqrt(3) (pattern_estimation.py:81-83).
- **Counts (log.txt):** 1809_2: 10 758 658 rows, 4 634 045 on-through + positive + >= 300 ph, 980 396 clean (also nn >= 3). 2.5 nM: 93 507 946 / 48 459 874 / 966 735.

## Estimation steps and their filters
- **k (angle + pitch):** uses the on-through + positive + >= 300 ph rows with a looser isolation, nn >= 2.0 px (pattern_estimation.py:117).
  - DFT power is summed incoherently over 3000-frame blocks (pattern_estimation.py:116, 123).
  - Pitch search annulus 1.6-2.8 px (pattern_estimation.py:116).
- **Per-step phases psi_s / rel_s:** a weighted regression on the clean rows (pattern_estimation.py:164-183).
  - Weights = phot clipped at 4000 (pattern_estimation.py:168).
  - Two passes with outlier rejection |residual| >= 4 x 1.4826 MAD (pattern_estimation.py:179).
- **Shared phase drift eps_a(t):** estimated on the clean rows in 150-frame blocks (kf = 150; pattern_estimation.py:195, 276).
  - Weights 1 / var(phase), with var = |k|^2 s^2 + 2 / (m^2 A) (pattern_estimation.py:208).
  - Blocks with < 20 rows are interpolated (pattern_estimation.py:216).
  - 3-block running mean, mean removed (pattern_estimation.py:195, 219-221).
- **Depth m_a:** the cross-cycle estimator on pairs of clean windows (w, w + 6) of the same site (clean detection within 1.0 px in w + 6) (pattern_estimation.py:231-236, 241-262).
  - Per-photon-bin diagnostics only for bins with > 50 pairs (pattern_estimation.py:264).
  - Bootstrap se over 1500-frame blocks (pattern_estimation.py:267).
- **Iterations:** phases / drift / depth are iterated n_iter = 3 times on the clean rows (pattern_estimation.py:276, 282-292).
- **Diagnostic only:** depth vs nn_min 2 / 3 / 4 / 5 px, not used for the pattern (pattern_estimation.py:298-301).
- **No kinetics test:** there is no on/off-kinetics model test in the estimation; 'on through the window' is the linking test above. The kinetics test (Delta(-2 log L) > 9.21) belongs to the explicit solve, not to the pattern estimation.
- **Call:** estimate_pattern(R, movie.shape, nn_min=a.nn_min, min_phot=a.min_phot) (run_uncalibrated.py:152).
