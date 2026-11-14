# SIMCODE v2: method

SIMCODE v2 localizes emitters in DMD-SIMFLUX DNA-PAINT movies. In that setup a sinusoidal illumination pattern is stepped
through 3 phases for each of 2 orientations, so every 6 frames form one full pattern cycle. SIMCODE v2 is a per-window
method: each 6-frame window of an emitter gives one localization.

The networks never see the illumination pattern in pass 1. Instead they output per-frame intensities with a full
covariance. The pattern-encoded position then comes from an explicit intensity-domain solve. No pixel-level linear
intensity fit is done anywhere in the method.

## Pipeline (`python -m simcode.run`)

| stage | what | code |
|---|---|---|
| 1. pass 1: detection | Pattern-free detector **det_n1** (`models/det_n1.pt`) on each 6-frame window. Windows start every 3 frames. It outputs a logit map and a sub-pixel offset. A sub-pixel NMS keeps the top-k (640 per 64² px). Candidates with p ≥ 0.3 (`pipeline/common.CAND_PMIN`) are stored. The cut also feeds the pattern stage's isolation test and the automatic c: 0.3 vs 0.5 (the RESULTS runs) moves the pattern phases by 0.4–0.9 nm and depth by +0.01, and splits c / c_a differently (one-chunk check, `analysis/results/checks/`). | `det/up2.py`, `det/model.py` |
| 1. pass 1: head | Neighbour-aware head **SCHead4 / sc5nb** (`models/hd_sc5nb.pt`) on a 21 × 21 × 6 patch per candidate. Inputs: log and linear frames, coordinate maps, the PSF template, a splat of the other candidates (p ≥ 0.2), and the detector-probability map. A learned linear read-out gives I_t = Σ w_t,q (y_t,q − bg_t) plus a learned correction. Outputs: the 6 per-frame intensities I_t with a full 6 × 6 Cholesky covariance, the per-frame bg, and a pattern-free 3D position prior (x, y, z) with a 3 × 3 covariance (x, y uncorrelated; z | x, y = z + b·Δxy, sd s_z|xy; z in px of the lateral pixel relative to the PSF calibration's focal plane; trained on heights 0–0.3 µm, half uniform). bf16 trunk, fp32 read-out. | `hd/sc4.py`, `hd/sc.py`, `pipeline/sc_front_fast.py` |
| 2. pattern estimation | Estimates k (FFT, then DFT refinement), per-step phases, modulation depth, relative intensities, and a per-angle phase drift ε(t). It uses only the pass-1 network intensities and prior positions; the constant prior bias `prior_off` is removed first. | `pipeline/pattern_est.py` |
| (3. automatic c) | Not a pipeline stage since 2026-11-14 (the paper uses the fixed c of the model's calibration). `simcode/autoc.py` remains as an analysis estimator: it fits θ = (s, c, c_a) of the intensity noise model Σ(θ) = s Σ_net + c² diag(μ²) + c_a² μμᵀ (within angle) to the solve's own residuals (isolated, non-kinetic rows; per-frame and angle-contrast PIT medians binned by SNR; 3 fixed-point re-solves). | `autoc.py` |
| 4. pattern solve | Per row, a Levenberg–Marquardt fit of (x, y, B[, t_on, t_off]) to I_t with the covariance Σ(μ). The prior is a Gaussian term. Starts: the prior centre plus the 8 fringe-shifted starts (±1 period per angle). The on/off kinetics model is kept only if it lowers the NLL by > 9.21. The output is the fringe posterior mean over the deduplicated starts. z: the head's 3D prior conditioned on the solved x, y (`sc.z_given_xy`: z + b·(r − r_prior), var = s_z|xy² + Σ b_i² σ_i²); the pattern carries no z information, and the in-focus TIRF PSF little (z CRLB 40–60 nm at 1000 photons), so z is mostly the prior for dim emitters. σ_cal = (chi²-aware Fisher σ + posterior spread) × a per-photon-bin factor, calibrated on sims and rescaled for real data. | `hd/msolve.py`, `hd/msolve_fast.py`, `hd/sc.py:sc_solve` |
| 5. pass 2 | Pattern-aware detector **p2_det** and head **p2_head_long** (p2L). They are det_n1 / sc5nb with extra input channels for the estimated illumination, zero-initialised and then fine-tuned with a realistic pattern-error model. Their output goes through the same pattern solve (the model's pass-2 calibration, e.g. `calib/calib_p2L_m1809_real.json`). | `p2/front.py`, `p2/head2.py`, `p2/det2.py` |
| 6. table | Localization table. The phase-calibrated selection `sel` keeps, per photon band, the pass fraction of p ≥ 0.63, taken as the top-p rows in each 6 × 6 sub-pixel cell. The paper's selection is max(σx, σy) ≤ 12 nm on the rows with p ≥ 0.63 inside the crop (`strict.select`; the per-tile strict product and the u ≤ 0.45 rule were removed on 2026-11-14). | `strict.py` |

### Run record

Every run writes `<out>/run_config.json` (readable copy `run_config.yaml`): the fully resolved settings, the fixed values
of the code path, absolute path + sha256 of the movie, the calibration, the camera maps and every network / solve
calibration (with the trainer arguments stored in each checkpoint), git commit and dirty flag, DME version, torch / CUDA /
GPU, the pattern estimate and the counts per selection rule. `python -m simcode.strict` appends its settings to the `steps`
list of the same file. `python -m simcode.run --config <run_config.json> --out <new dir>` re-runs with those settings and
warns about every input or code difference; on the same code and GPU the outputs are bit-identical (README, "Run record";
`simcode/util/runconfig.py`; `tests/test_runconfig.py`).

### Models (2026-11)

The paper's runs use **PSF-matched models** (`simcode/models.json`: `m1809`, `m25`, `g13`, `m1809g`; `--model`,
required; or the path of a model folder with a `model.json`). A model has the same four architectures and training recipes as described below; the simulator's PSF is the
averaged in-situ PSF of the target movie (`--calib` of the trainers), and the pass-1 head is fine-tuned from `hd_sc5nb`
(`models/README.md` has the recipe and the file list). In the table above, read det_n1 / sc5nb / p2_det / p2_head_long as
the detector / head / pass-2 detector / pass-2 head of the model (`det_<set>.pt`, `hd_<set>.pt`, `p2det_<set>.pt`,
`p2headL_<set>.pt`) and the solve calibrations as the model's (`calib/calib_sc_hd_<set>_real.json`,
`calib/calib_p2L_<set>_real.json`). The earlier default networks named in this document (det_n1, hd_sc5nb, p2_det,
p2_head_long) are the ancestors of the models; they are no longer a selectable model.

Production recipe (`python -m simcode.run --dataset <name>`; `_production` in `models.json`; since 2026-11-14 every run does
this): fixed c, stored-candidate cut 0.3, pass 2, the model's selection-aware sigma rescale, and the main selection **calibrated max(σx, σy) ≤ 12 nm** (`simcode.strict.select(table)`; rows with a rank,
i.e. p ≥ 0.63 inside the crop, whose calibrated precision is 12 nm or better in both axes; user decision 2026-11-13, study in
`analysis/selection_global`). The earlier per-tile rule (sigma rank quantile ≤ 0.45 within each 10-px tile) is `select(table, 'u')`.

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
| det_n1 | `python -m simcode.det.up2 train` | focal loss on the nearest-pixel targets, offset NLL |
| sc5nb | `python -m simcode.hd.sc4 prep5`, then `train --init <sc5n>` | full-covariance Gaussian NLL of the GT per-frame photons, plus a mean-removed anti-shrinkage term and the modulation-slope penalty, plus the position NLL; per-nn-bin target weights |
| p2_det / p2_head_long | `python -m simcode.p2.det2 train`, `python -m simcode.p2.head2 train` (error model `--pert_model real`, the default; v1 for the intermediate p2head stage) | illumination channels zero-initialised from det_n1 / sc5nb |

### Exact training commands of the earlier default networks (the ancestors of the models)

(As run at the time. Since 2026-11-14 `--up 1`, `--em_photons 40,2500`, `--arch sc4n`, `--paper 1`, `--bright_tail
0.1,2000,20000` and `--pert_model real` are the trainers' fixed behaviour or defaults; the removed flags are accepted with
these values. Two differences when re-running them: `--calib` is required (the earlier networks used the 0.5 nM in-situ PSF,
`insitu/calib_0.5nM.json`), and a `p2.head2` command without `--pert_model` now means `real` (add `--pert_model v1`).)

The recipe of a PSF-matched set (the same commands with `--calib <averaged PSF>` and the fine-tune from `hd_sc5nb.pt`) is in
`models/README.md`.

From the `args` stored in each checkpoint (`torch.load(f, weights_only=False)['args']`); `runs/` is any work directory.
Paths of the dev runs are given relative; the shipped files are the `final.pt` of each run.

```bash
export CUDA_VISIBLE_DEVICES=1
# det_n1 (40k steps)
python -m simcode.det.up2 train --up 1 --out runs/det_n1 --steps 40000 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 \
    --nnw 1 --em_photons 40,2500 --seed 0
# held-out set of the head trainer (<out>/sc4/heldout5.pt, det_n1 candidates)
python -m simcode.hd.sc4 prep5
# hd_sc5n (100k steps from scratch), then hd_sc5nb = 18k-step fine-tune with the bright tail
python -m simcode.hd.sc4 train --arch sc4n --out runs/hd_sc5n --steps 100000 --batch 16 --lr 5e-4 --ch 96 --det det_n1.pt \
    --paper 1 --tau 15,80 --hold <out>/sc4/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 0 \
    --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000
python -m simcode.hd.sc4 train --arch sc4n --out runs/hd_sc5nb --steps 18000 --batch 16 --lr 1.5e-4 --ch 96 --det det_n1.pt \
    --paper 1 --tau 15,80 --hold <out>/sc4/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 \
    --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init runs/hd_sc5n/final.pt --pct 0.1 \
    --bright_tail 0.1,2000,20000 --ckpt_every 6000
# p2_det (40k steps; illumination channels zero-initialised from det_n1)
python -m simcode.p2.det2 train --out runs/p2_det --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 \
    --seed 11 --eval_every 2500 --ckpt_every 5000
# p2_head (24k steps from hd_sc5nb; its --det was p2_det's 25k-step checkpoint, identical to runs/p2_det/ckpt_25000.pt),
# then p2_head_long = 50k steps from p2_head with the 'real' pattern-error model
python -m simcode.p2.head2 train --out runs/p2_head --det runs/p2_det/ckpt_25000.pt --init hd_sc5nb.pt --steps 24000 \
    --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0
python -m simcode.p2.head2 train --out runs/p2_head_long --det runs/p2_det/final.pt --init runs/p2_head/final.pt --steps 50000 \
    --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 \
    --pert_model real --hold_v1 15
```

All four entry points were run for 3 steps with these arguments (2026-11-11, `--init` from the shipped files). On this
server, put the env's `bin/` on PATH (`conda activate`): otherwise triton cannot find `ptxas` in the head trainers'
backward pass.

The realistic training generator `hd/realgen.py` (`--gen r1`; networks p1r / p2r) was **not** adopted (the pre-registered
rules failed) and was removed on 2026-11-14.

## Calibrations (`models/`)

| file | use |
|---|---|
| `calib/calib_sc_hd_<set>_real.json`, `calib_sc_hd_<set>.json`, `calib_p2L_<set>_real.json`, `calib_p2L_<set>.json`, `sigu/sigu_p{1,2}_<set>.json` | the calibrations of a model (pass-1 / pass-2 solve with real-data or sim σ; selection-aware σ rescale): fitted with `analysis/psfmatch/fit_cal_set.py` on the gate scenes rendered with the model's PSF, templates = the hd_sc5nb files below; `analysis/psf3d_nets/sigcal.py` for the rescale |
| `calib/calib_sc_hd_sc5nbb15_pm_bright_real.json` | pass-1 solve with fixed c (0.10 + 0.10). The σ table is fitted on the gate tuning seed and rescaled for real data with the real 0.5 nM M4 zcal × the sim (true / M4) ratio. |
| `calib/calib_sc_hd_sc5nbb15_pm_bright.json` | same, sim σ (gate scenes, sim movies; start point of the autoc fit) |
| `calib/calib_sc_hd_sc5nbb15_pm_autoc1.json` | σ table for the automatic-c solve (`autoc.make_calib`; analysis only since 2026-11-14) |
| `calib/calib_autoc_r05 / r25 / r20.json` | the automatic-c calibrations of the full real 0.5 / 2.5 / 20 nM runs (θ 0.117 / 0.055, 0.143 / 0, 0.131 / 0) |
| `calib/calib_p2L_b10_real.json` | pass-2 solve, real σ (production) |
| `calib/calib_p2L_movie_b10.json`, `calib_p2L.json` | pass-2, sim σ (movie benchmark), gate |
| `insitu/calib_0.5nM.json` | in-situ PSF (Zernikes, focus), camera (gain 0.56 e/ADU, per-pixel offset / read-noise maps), pattern of the 0.5 nM movie; the PSF of the simulator |
| `insitu/calib_2p5nM_m2*.json`, `insitu/calib_20nM_m2*.json` | the same PSF with the camera maps of the 2.5 nM / 20 nM movie 2 (`*_sc.json` = the versions used by the runs) |

The σ tables were fitted per network-photon bin on the gate tuning seed 202, which is disjoint from the evaluation
seeds 301–303: `analysis/calib/dump_fe.py`, then `fit_cal.py`. The pass-2 equivalent is `analysis/calib/plib.py`
(`fit_cal`). The real-data rescale uses the real M4 zcal × the sim (true / M4) ratio.

The in-situ calibration itself (bead fit → camera maps → tracks → stage A → stage B → `calib_<movie>.json`) is described
in `simcode/insitu/README.md`.

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
- **σ of a per-tile σ-quantile selection** (the strict product, removed 2026-11-14) is overconfident in its best bins
  (winner's curse, 1.3–1.9× in sim). Quote NN or GT RMS for such a subset, never its σ threshold.
