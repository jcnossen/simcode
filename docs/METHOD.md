# SIMCODE v2: method

Paths under `analysis/`, the model `m25`, the in-situ calibrations of the other movies (`models/insitu/`) and the
development notes (`docs/RESULTS.md`, `docs/AUTO_PIPELINE.md`, `docs/CLEANUP_LOG.md`, `docs/ARCHIVE.md`) named in this
document are on the `paper` branch of the repository.

SIMCODE v2 localizes emitters in DMD-SIMFLUX DNA-PAINT movies. In that setup a sinusoidal illumination pattern is stepped
through 3 phases for each of 2 orientations, so every 6 frames form one full pattern cycle. SIMCODE v2 is a per-window
method: each 6-frame window of an emitter gives one localization.

The networks never see the illumination pattern in pass 1. Instead they output per-frame intensities with a full
covariance. The pattern-encoded position then comes from an explicit intensity-domain solve. No pixel-level linear
intensity fit is done anywhere in the method.

## Pipeline (`python -m simcode.run`)

| stage | what | code |
|---|---|---|
| 1. pass 1: detection | Pattern-free detector **pretrain_detector_pass1** (`models/pretrain_detector_pass1.pt`) on each 6-frame window. Windows start every 3 frames. It outputs a logit map and a sub-pixel offset. A sub-pixel NMS keeps the top-k (640 per 64² px). Candidates with p ≥ 0.3 (`processing/common.CAND_PMIN`) are stored. The cut also feeds the pattern stage's isolation test and the automatic c: 0.3 vs 0.5 (the RESULTS runs) moves the pattern phases by 0.4–0.9 nm and depth by +0.01, and splits c / c_a differently (one-chunk check, `analysis/results/checks/`). | `detector/train.py`, `detector/model.py` |
| 1. pass 1: head | Neighbour-aware emitter network **EmitterNet** (`models/pretrain_emitter_pass1.pt`) on a 21 × 21 × 6 patch per candidate. Inputs: log and linear frames, coordinate maps, a splat of the other candidates (p ≥ 0.2), and the detector-probability map. The PSF template g of the candidate offset is not an input channel: it only shapes the read-out weights (w_t = w0·exp(m) normalised to w·g = 1, w0 = g / Σg², plus per-frame signed maps D_t with the g-component projected out). The learned linear read-out gives I_t = Σ w_t,q (y_t,q − bg_t) plus a learned correction. Outputs: the 6 per-frame intensities I_t with a full 6 × 6 Cholesky covariance, the per-frame bg, and a pattern-free 3D position prior (x, y, z) with a 3 × 3 covariance (x, y uncorrelated; z | x, y = z + b·Δxy, sd s_z|xy; z in px of the lateral pixel relative to the PSF calibration's focal plane; trained on heights drawn from the PSF calibration: the quantiles `emitter_z_quantiles_um` of the fitted in-situ spot heights not pinned at the fit limits, piecewise-linear inverse CDF, clipped to `emitter_z_range_um` [0, 0.40] µm (PSF-v2 models, 2026-11-15; `simulate/acquisition.py` height_quantiles / draw_z, `out/psf_v2/STATUS.md`)). bf16 trunk, fp32 read-out. | `emitter/train.py`, `emitter/model.py`, `processing/inference_fast.py` |
| 2. pattern estimation | Estimates k (FFT, then DFT refinement), per-step phases, modulation depth, relative intensities, and a per-angle phase drift ε(t). It uses only the pass-1 network intensities and prior positions; the constant prior bias `prior_off` is removed first. | `processing/pattern_estimation.py` |
| (3. automatic c) | Not a pipeline stage since 2026-11-14 (the paper uses the fixed c of the model's calibration). `simcode/solve/brightness_noise_fit.py` remains as an analysis estimator: it fits θ = (s, c, c_a) of the intensity noise model Σ(θ) = s Σ_net + c² diag(μ²) + c_a² μμᵀ (within angle) to the solve's own residuals (isolated, non-kinetic rows; per-frame and angle-contrast PIT medians binned by SNR; 3 fixed-point re-solves). | `solve/brightness_noise_fit.py` |
| 4. pattern solve | Per row, a Levenberg–Marquardt fit of (x, y, B[, t_on, t_off]) to I_t with the covariance Σ(μ). The prior is a Gaussian term. Starts: the prior centre plus the 8 fringe-shifted starts (±1 period per angle). The on/off kinetics model is kept only if it lowers the solve cost −2 log L (χ² incl. the prior, plus log det Σ) by > 9.21 (the 99 % point of χ² with 2 dof; an NLL drop of 4.6). The output is the fringe posterior mean over the deduplicated starts. z: the head's 3D prior conditioned on the solved x, y (`solve/z_conditioning.z_given_xy`: z + b·(r − r_prior), var = s_z|xy² + Σ b_i² σ_i²); the pattern carries no z information, and the in-focus TIRF PSF little (z CRLB 40–60 nm at 1000 photons), so z is mostly the prior for dim emitters. σ_cal = (chi²-aware Fisher σ + posterior spread) × a per-photon-bin factor, calibrated on sims; real movies use the same calibration (no real-data rescale since 2026-11-17). | `solve/solve.py`, `solve/solve_fast.py`, `solve/solve.py:solve_emitters` |
| 5. pass 2 | Pattern-aware pass-2 detector and emitter network (of the earlier default: `models/archive_superseded/previous_default/detector_pass2.pt`, `emitter_pass2.pt`). They are the pass-1 networks with extra input channels for the estimated illumination, zero-initialised and then fine-tuned with a realistic pattern-error model on generic illumination patterns (any pitch 1.8–4 px, orientation, per-frame phase, frame order, depth and power in half of the training FOVs, the SIMCODE pattern in the others; `simulate/acquisition.AcquisitionSim._generic`), so pass 2 accepts any measured two-orientation pattern. Their output goes through the same pattern solve (the model's pass-2 calibration, e.g. `archive_2d/m1809/calib/calib_emitter_pass2_real.json`). z of the pass-2 rows comes from pass 1 (2026-11-14; the pass-2 head's z is biased upward at low photons, e.g. +32 nm on the gate scene rs05pm, and is not used): each pass-2 row is linked one-to-one to a pass-1 candidate of the same window (detector points within 1 px, iterated mutual nearest neighbours, `z_conditioning.link_pass1`), and that candidate's 3D prior is conditioned on the pass-2 solved x, y (`z_conditioning.z_given_xy` with the pass-1 3 × 3 covariance; `z_conditioning.pass1_z_columns`). Rows without a pass-1 partner get z = NaN (real 2.5 nM / 0.5 nM chunks: 2 % / 0.7 % of the paper's selection, 30 % / 12 % of all rows; gate scenes: 0.1–2.8 % of the targets, close pairs pair1 10.7 %). | `processing/pass2_inference.py`, `emitter/train.py`, `detector/train.py` |
| 6. table | Localization table. The phase-calibrated selection `sel` keeps, per photon band, the pass fraction of p ≥ 0.63, taken as the top-p rows in each 6 × 6 sub-pixel cell. The paper's selection is max(σx, σy) ≤ 12 nm on the rows with p ≥ 0.63 inside the crop (`table.select`; the per-tile strict product and the u ≤ 0.45 rule were removed on 2026-11-14). | `table.py` |

### Run record

Every run writes `<out>/run_config.json` (readable copy `run_config.yaml`): the fully resolved settings, the fixed values
of the code path, absolute path + sha256 of the movie, the calibration, the camera maps and every network / solve
calibration (with the trainer arguments stored in each checkpoint), git commit and dirty flag, DME version, torch / CUDA /
GPU, the pattern estimate and the counts per selection rule. `python -m simcode.table` appends its settings to the `steps`
list of the same file. `python -m simcode.run --config <run_config.json> --out <new dir>` re-runs with those settings and
warns about every input or code difference; on the same code and GPU the outputs are bit-identical (README, "Run record";
`simcode/util/runconfig.py`; `tests/test_runconfig.py`).

### Models (2026-11)

The paper's runs use **PSF-matched models** (`simcode/models.json`: `m1809`, `g13`; the paper branch adds `m25`; `--model`,
required; or the path of a model folder with a `model.json`). A model has the same four architectures and training recipes as described below; the simulator's PSF is the
in-situ PSF v2 of the target movie (`insitu/calib_<movie>_psf_v2.json`, the model's `model.json` 'psf'; `--calib` of the trainers; the pre-v2 models used the axis-averaged `_psf3d_avg`, `models/archive_psf_v1/`), and the pass-1 head is fine-tuned from `pretrain_emitter_pass1`
(`models/README.md` has the recipe and the file list). In the table above, read the networks named there as
the detector / emitter network / pass-2 detector / pass-2 emitter network of the model (`<set>/detector_pass1.pt`,
`emitter_pass1.pt`, `detector_pass2.pt`, `emitter_pass2.pt`) and the solve calibrations as the model's
(`<set>/calib/calib_emitter_pass1_sim.json`, `calib_emitter_pass2_sim.json`; the only calibrations of a model since 2026-11-17). The earlier default networks named in this
document (`pretrain_detector_pass1.pt`, `pretrain_emitter_pass1.pt`, and the pass-2 pair `archive_superseded/previous_default/`) are the ancestors
of the models; they are no longer a selectable model.

Production recipe (`python -m simcode.run --dataset <name>`; `_production` in `models.json`; since 2026-11-14 every run does
this): fixed c, stored-candidate cut 0.3, pass 2, the model's selection-aware sigma rescale, and the main selection **calibrated max(σx, σy) ≤ 12 nm** (`simcode.table.select(table)`; rows with a rank,
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

The original modules (`processing/pass1_inference.py`, `processing/pass2_inference.py`, `solve/solve.py:solve_emitters`) are kept as the reference.

## Training

All networks are trained only on the experiment-matched simulator (`simulate/acquisition.py`; `simulate/batches.py`, `simulate/batches.py` batches):

- calibrated vectorial PSF from the in-situ fit (`models/insitu/calib_0.5nM.json`) with LUT jitter and a halo
- the SIMFLUX pattern with estimation errors
- DNA-PAINT kinetics, filaments, close pairs
- per-frame / per-step / per-angle emitter brightness variation
- structured and fluctuating background, sCMOS camera maps
- paper-matched photon distributions

| network | trainer | notes |
|---|---|---|
| pretrain_detector_pass1 | `python -m simcode.detector.train` | focal loss on the nearest-pixel targets, offset NLL |
| pretrain_emitter_pass1 | `python -m simcode.emitter.heldout --set 5` (held-out set), then `python -m simcode.emitter.train --init <the network trained from scratch>` | full-covariance Gaussian NLL of the GT per-frame photons, plus a mean-removed anti-shrinkage term and the modulation-slope penalty, plus the position NLL; per-nn-bin target weights |
| `archive_superseded/previous_default/detector_pass2.pt`, `emitter_pass2.pt` | `python -m simcode.detector.train --illum 1`, `python -m simcode.emitter.train` with a pass-2 `--detector` (error model `--pert_model real`, the default; v1 for the intermediate pass-2 stage) | illumination channels zero-initialised from the pass-1 networks |

### Exact training commands of the earlier default networks (the ancestors of the models)

(As run at the time, written with the present trainers (2026-11-15: one trainer per network, `simcode.detector.train` and
`simcode.emitter.train`, the pass taken from the checkpoints; before: `simcode.det.up2`, `simcode.p2.det2`, `simcode.hd.sc4`,
`simcode.p2.head2`). The flags `--up 1`, `--em_photons 40,2500`, `--arch` (the network tag), `--paper 1`, `--bright_tail 0.1,2000,20000`
of the original commands are the trainers' fixed behaviour since 2026-11-14 and are left out below. Two differences when
re-running them: `--calib` is required (the earlier networks used the 0.5 nM in-situ PSF, `insitu/calib_0.5nM.json`), and a
pass-2 emitter-network command without `--pert_model` now means `real` (add `--pert_model v1`).)

The recipe of a PSF-matched set (the same commands with `--calib <averaged PSF>` and the fine-tune from `pretrain_emitter_pass1.pt`) is in
`models/README.md`.

From the `args` stored in each checkpoint (`torch.load(f, weights_only=False)['args']`); `runs/` is any work directory.
Paths of the dev runs are given relative; the shipped files are the `final.pt` of each run.

```bash
export CUDA_VISIBLE_DEVICES=1
# pretrain_detector_pass1 (40k steps)
python -m simcode.detector.train --out runs/pretrain_detector_pass1 --steps 40000 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 \
    --nnw 1 --seed 0
# held-out set of the head trainer (<out>/emitter_heldout/heldout5.pt, pretrain_detector_pass1 candidates)
python -m simcode.emitter.heldout --set 5
# emitter network from scratch (100k steps), then pretrain_emitter_pass1 = 18k-step fine-tune with the bright tail
python -m simcode.emitter.train --out runs/emitter_scratch --steps 100000 --batch 16 --lr 5e-4 --ch 96 --detector pretrain_detector_pass1.pt \
    --tau 15,80 --hold <out>/emitter_heldout/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 0 \
    --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000
python -m simcode.emitter.train --out runs/pretrain_emitter_pass1 --steps 18000 --batch 16 --lr 1.5e-4 --ch 96 --detector pretrain_detector_pass1.pt \
    --tau 15,80 --hold <out>/emitter_heldout/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 \
    --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init runs/emitter_scratch/final.pt --pct 0.1 --ckpt_every 6000
# archive_superseded/previous_default/detector_pass2.pt (40k steps; illumination channels zero-initialised from pretrain_detector_pass1)
python -m simcode.detector.train --out runs/detector_pass2 --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 \
    --seed 11 --eval_every 2500 --ckpt_every 5000 --illum 1 --init pretrain_detector_pass1.pt
# pass-2 emitter network, stage 1 (24k steps from pretrain_emitter_pass1; its --detector was the pass-2 detector's 25k-step checkpoint,
# identical to runs/detector_pass2/ckpt_25000.pt), then archive_superseded/previous_default/emitter_pass2.pt = 50k steps from stage 1 with the 'real'
# pattern-error model
python -m simcode.emitter.train --out runs/emitter_pass2_stage1 --detector runs/detector_pass2/ckpt_25000.pt --init pretrain_emitter_pass1.pt --steps 24000 \
    --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0
python -m simcode.emitter.train --out runs/emitter_pass2 --detector runs/detector_pass2/final.pt --init runs/emitter_pass2_stage1/final.pt --steps 50000 \
    --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 \
    --pert_model real --hold_v1 15
```

The four trainings were run for 3 steps with these arguments (2026-11-11, `--init` from the shipped files; the merged trainers again on 2026-11-15). On this
server, put the env's `bin/` on PATH (`conda activate`): otherwise triton cannot find `ptxas` in the head trainers'
backward pass.

The realistic training generator `hd/realgen.py` (`--gen r1`; networks p1r / p2r) was **not** adopted (the pre-registered
rules failed) and was removed on 2026-11-14.

## Calibrations (`models/`)

| file | use |
|---|---|
| `<set>/calib/calib_emitter_pass1_sim.json`, `calib_emitter_pass2_sim.json`, `sigma_rescale_pass{1,2}.json` | the calibrations of a model (pass-1 / pass-2 solve; selection-aware σ rescale), used for real and simulated movies: fitted with `simcode.calibration.fit_cal_set` on the gate scenes rendered with the model's PSF, templates = the pretrain_emitter_pass1 files below; `analysis/psf3d_nets/sigcal.py` for the rescale |
| `calib/template_calib_emitter_pass1_sim.json` | pass-1 solve with fixed c (0.10 + 0.10), σ table fitted on the gate tuning seed (template of a new model's calibration; start point of the automatic-c fit, solve/brightness_noise_fit.py) |
| `calib/template_calib_emitter_pass1_brightness_noise_fit.json` | σ table for the automatic-c solve (made by the automatic-c estimator's make_calib, removed 2026-11-15; analysis only since 2026-11-14) |
| `archive_superseded/previous_default/calib/calib_emitter_pass1_brightness_noise_fit_{0.5nM,2.5nM,20nM}.json` | the automatic-c calibrations of the full real 0.5 / 2.5 / 20 nM runs (θ 0.117 / 0.055, 0.143 / 0, 0.131 / 0) |
| `archive_superseded/previous_default/calib/calib_emitter_pass2_sim_movie.json`, `template_calib_emitter_pass2_sim.json` | pass-2, sim σ (movie benchmark), gate |
| `insitu/calib_0.5nM.json` | in-situ PSF (Zernikes, focus), camera (gain 0.56 e/ADU, per-pixel offset / read-noise maps), pattern of the 0.5 nM movie; the PSF of the simulator |
| `insitu/calib_2p5nM_m2*.json`, `insitu/calib_20nM_m2*.json` | the same PSF with the camera maps of the 2.5 nM / 20 nM movie 2 (`*_sc.json` = the versions used by the runs) |

The σ tables were fitted per network-photon bin on the gate tuning seed 202, which is disjoint from the evaluation
seeds 301–303: `analysis/calib/dump_fe.py`, then `fit_cal.py`. The pass-2 equivalent is `analysis/calib/plib.py`
(`fit_cal`). Until 2026-11-17 a second, `_real` calibration multiplied this σ factor by a per-photon-bin factor 0.83–1.00 (pass 1)
/ 0.90–1.00 (pass 2), fitted on real 0.5 nM data (real M4 zcal × the sim (true / M4) ratio) and copied unchanged to later models
(`sig_ph_real_mult`); it was removed, real movies use the simulation-fitted calibration.

The in-situ calibration itself (bead fit → camera maps → tracks → stage A → stage B → `calib_<movie>.json`) is described
in `archive-2026-11-15:simcode/calibration/README.md`.

Only the camera maps and the PSF of the in-situ calibration are used by the pipeline. Its pattern entry is never used:
the pattern is always estimated from the data.

## Known limitations

The numbers are in the paper (and `docs/RESULTS.md` of the paper branch).

- **Absolute accuracy is unknown.** This holds for ours and for SIMCODE. Per-site static errors and the ±3–6 nm linear
  terms are invisible to every GT-free metric, and no calibration sample exists.
- **PSF ellipticity.** On real data the PSF is x/y-elliptic relative to the model (x wider, ΔMxx +0.066 px²); the best
  one-parameter description is an x-blur of about 0.36 px beyond the sims. About 20 % of it flips with pattern angle. Our
  head was trained on one vectorial PSF and loses some prior precision on real data; SIMCODE's Gaussian-trained net is
  insensitive to it. Fine-tunes with PSF diversity were not adopted.
- **The angle-0 brightness-dependent x term** is a ±3 nm accuracy systematic. Both methods show it.
- **Sub-pixel pairs.** Pairs below about 1 px (108 nm) are only partly resolved: on gate pair1, recall is 0.64 (pass 1) / 0.78 (pass 2),
  and gross errors fall from 23 % (p1) to 4.5 % (p2s).
- **2.5 nM.** There is an event-static prior error: about 40 % of an event-specific image-vs-pattern offset leaks into
  the final position. Per window vs SIMCODE there: pass 1 a tie, pass 2 ~7 % (indicative, NN bias-corrected).
- **σ of a per-tile σ-quantile selection** (the strict product, removed 2026-11-14) is overconfident in its best bins
  (winner's curse, 1.3–1.9× in sim). Quote NN or GT RMS for such a subset, never its σ threshold.
