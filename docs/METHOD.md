# SIMCODE v2: method

SIMCODE v2 localizes emitters in DMD-SIMFLUX DNA-PAINT movies. In that setup a sinusoidal illumination pattern is stepped
through 3 phases for each of 2 orientations, so every 6 frames form one full pattern cycle. SIMCODE v2 is a per-window
method: each 6-frame window of an emitter gives one localization.

The networks never see the illumination pattern in pass 1. Instead they output per-frame intensities with a full
covariance. The pattern-encoded position then comes from an explicit intensity-domain solve. No pixel-level linear
intensity fit is done anywhere in the method.

## Pipeline (`python -m simcode2.run`)

| stage | what | code |
|---|---|---|
| 1. pass 1: detection | Pattern-free detector **det_n1** (`models/det_n1.pt`) on each 6-frame window. Windows start every 3 frames. It outputs a logit map and a sub-pixel offset. A sub-pixel NMS keeps the top-k (640 per 64² px). Candidates with p ≥ `--cand_pmin` (0.3) are stored. The cut also feeds the pattern stage's isolation test and the automatic c: 0.3 vs 0.5 (the RESULTS runs) moves the pattern phases by 0.4–0.9 nm and depth by +0.01, and splits c / c_a differently (one-chunk check, `analysis/results/checks/`). | `det/up2.py`, `det/model.py` |
| 1. pass 1: head | Neighbour-aware head **SCHead4 / sc5nb** (`models/hd_sc5nb.pt`) on a 21 × 21 × 6 patch per candidate. Inputs: log and linear frames, coordinate maps, the PSF template, a splat of the other candidates (p ≥ 0.2), and the detector-probability map. A learned linear read-out gives I_t = Σ w_t,q (y_t,q − bg_t) plus a learned correction. Outputs: the 6 per-frame intensities I_t with a full 6 × 6 Cholesky covariance, the per-frame bg, and a pattern-free position prior (x, y) ± σ. bf16 trunk, fp32 read-out. | `hd/sc4.py`, `hd/sc.py`, `pipeline/sc_front_fast.py` |
| 2. pattern estimation | Estimates k (FFT, then DFT refinement), per-step phases, modulation depth, relative intensities, and a per-angle phase drift ε(t). It uses only the pass-1 network intensities and prior positions; the constant prior bias `prior_off` is removed first. | `pipeline/pattern_est.py` |
| 3. automatic c | Fits θ = (s, c, c_a) of the intensity noise model Σ(θ) = s Σ_net + c² diag(μ²) + c_a² μμᵀ (within angle) to the solve's own residuals. It uses isolated, non-kinetic rows and minimises the per-frame and angle-contrast PIT medians binned by SNR, with 3 fixed-point re-solves. s is a nuisance and is not applied. Default on; `--autoc 0` uses a fixed c = c_a = 0.10. | `autoc.py` |
| 4. pattern solve | Per row, a Levenberg–Marquardt fit of (x, y, B[, t_on, t_off]) to I_t with the covariance Σ(μ). The prior is a Gaussian term. Starts: the prior centre plus the 8 fringe-shifted starts (±1 period per angle). The on/off kinetics model is kept only if it lowers the NLL by > 9.21. The output is the fringe posterior mean over the deduplicated starts. σ_cal = (chi²-aware Fisher σ + posterior spread) × a per-photon-bin factor, calibrated on sims and rescaled for real data. | `hd/msolve.py`, `hd/msolve_fast.py`, `hd/sc.py:sc_solve` |
| 5. pass 2 (`--pass2`) | Pattern-aware detector **p2_det** and head **p2_head_long** (p2L). They are det_n1 / sc5nb with extra input channels for the estimated illumination, zero-initialised and then fine-tuned with a realistic pattern-error model. Their output goes through the same pattern solve (`models/calib/calib_p2L_b10_real.json`). | `p2/front.py`, `p2/head2.py`, `p2/det2.py` |
| 6. table | Localization table. The phase-calibrated selection `sel` keeps, per photon band, the pass fraction of p ≥ 0.63, taken as the top-p rows in each 6 × 6 sub-pixel cell. `strict` is the recommended strict product: a per-10-px-tile σ quantile; q from the GT-free row density (anchors where it matches SIMCODE's coverage on the real movies, in-sample, under the automatic and the fixed c alike). | `strict.py` |

### Fast path

The fast path is the default. Its outputs match the reference path to rounding (`tests/test_fast_paths.py`).

- **Queued batching.** The detector runs on 64 windows at a time. Candidates go into a GPU queue, and the head drains it in
  fixed batches of 4096 on 2 CUDA streams.
- **Exact solve restructuring.** The pattern solve avoids batched bmm, uses reduced normal equations and one triangular
  solve, and computes the covariance only for the returned rows.
- **Exact kinetics gate.** Rows that can never select the kinetic model skip that fit.
- **Compiled LM step.** `torch.compile` of the LM step is used when available.

The original modules (`pipeline/sc_front.py`, `p2/front.py`, `hd/sc.py:sc_solve`) are kept as the reference.

## Training

All networks are trained only on the experiment-matched simulator (`hd/realsim.py`; `hd/data.py`, `p2/data.py` batches):

- calibrated vectorial PSF from the in-situ fit (`models/insitu/calib_0.5nM.json`) with LUT jitter and a halo
- the SIMFLUX pattern with estimation errors
- DNA-PAINT kinetics, filaments, close pairs
- per-frame / per-step / per-angle emitter brightness variation
- structured and fluctuating background, sCMOS camera maps
- paper-matched photon distributions

| network | trainer | notes |
|---|---|---|
| det_n1 | `python -m simcode2.det.up2 train --up 1` | focal loss on the nearest-pixel targets, offset NLL |
| sc5nb | `python -m simcode2.hd.sc4 prep5`, then `train --arch sc4n --init <sc5n> --bright_tail ...` | full-covariance Gaussian NLL of the GT per-frame photons, plus a mean-removed anti-shrinkage term and the modulation-slope penalty, plus the position NLL; per-nn-bin target weights |
| p2_det / p2_head_long | `python -m simcode2.p2.det2 train`, `python -m simcode2.p2.head2 train --pert real` | illumination channels zero-initialised from det_n1 / sc5nb |
| hd_sc7bg (optional) | `python -m simcode2.hd.sc7bg train` | sc5nb + a dense BgNet background (bg 3.5× → 1.1× CRLB, positions unchanged); not production |

`hd/realgen.py` (`--gen r1`) is an optional realistic training generator. The networks retrained with it (p1r / p2r) were
**not** adopted: the pre-registered rules failed.

## Calibrations (`models/`)

| file | use |
|---|---|
| `calib/calib_sc_hd_sc5nbb15_pm_bright_real.json` | pass-1 solve with fixed c (0.10 + 0.10). The σ table is fitted on the gate tuning seed and rescaled for real data with the real 0.5 nM M4 zcal × the sim (true / M4) ratio. |
| `calib/calib_sc_hd_sc5nbb15_pm_bright.json` | same, sim σ (gate scenes, sim movies; start point of the autoc fit) |
| `calib/calib_sc_hd_sc5nbb15_pm_autoc1.json` | σ table for the automatic-c solve; `run` writes `<out>/calib_autoc.json` = this table + the estimated (c, c_a) |
| `calib/calib_autoc_r05 / r25 / r20.json` | the automatic-c calibrations of the full real 0.5 / 2.5 / 20 nM runs (θ 0.117 / 0.055, 0.143 / 0, 0.131 / 0) |
| `calib/calib_p2L_b10_real.json` | pass-2 solve, real σ (production) |
| `calib/calib_p2L_movie_b10.json`, `calib_p2L.json` | pass-2, sim σ (movie benchmark), gate |
| `insitu/calib_0.5nM.json` | in-situ PSF (Zernikes, focus), camera (gain 0.56 e/ADU, per-pixel offset / read-noise maps), pattern of the 0.5 nM movie; the PSF of the simulator |
| `insitu/calib_2p5nM_m2*.json`, `insitu/calib_20nM_m2*.json` | the same PSF with the camera maps of the 2.5 nM / 20 nM movie 2 (`*_sc.json` = the versions used by the runs) |

The σ tables were fitted per network-photon bin on the gate tuning seed 202, which is disjoint from the evaluation
seeds 301–303: `analysis/calib/dump_fe.py`, then `fit_cal.py`. The pass-2 equivalent is `analysis/calib/plib.py`
(`fit_cal`). The real-data rescale uses the real M4 zcal × the sim (true / M4) ratio.

Only the camera maps and the PSF of the in-situ calibration are used by the pipeline. Its pattern entry is never used:
the pattern is always estimated from the data.

## Known limitations

See `RESULTS.md` for the numbers.

- **Absolute accuracy is unknown.** This holds for ours and for SIMCODE. Per-site static errors and the ±3–6 nm linear
  terms are invisible to every GT-free metric, and no calibration sample exists.
- **PSF ellipticity.** On real data the PSF is x/y-elliptic relative to the model (x wider, ΔMxx +0.066 px²); the best
  one-parameter description is an x-blur of about 0.36 px beyond the sims. About 20 % of it flips with pattern angle. Our
  head was trained on one vectorial PSF and loses some prior precision on real data; SIMCODE's Gaussian-trained net is
  insensitive to it. Fine-tunes with PSF diversity were not adopted.
- **The angle-0 brightness-dependent x term** is a ±3 nm accuracy systematic. Both methods show it.
- **Sub-pixel pairs.** Pairs below about 1 px (108 nm) are only partly resolved: on gate pair1, recall is 0.64 (p1) / 0.78 (p2L),
  and gross errors fall from 23 % (p1) to 4.5 % (p2s).
- **2.5 nM.** There is an event-static prior error: about 40 % of an event-specific image-vs-pattern offset leaks into
  the final position. Per window vs SIMCODE there: p1 a tie, p2L ~7 % (indicative, NN bias-corrected).
- **σ of the strict product.** It is overconfident in its best bins (winner's curse, 1.3–1.9× in sim). Quote NN or GT
  RMS for it, never its σ threshold.
