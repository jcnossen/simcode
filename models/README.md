# models/

This directory is where the code looks up the names of networks and calibrations (`simcode/config.py`: `MODELS` =
`<repo>/models`; there is no setting to move it: give a path, or a model folder with a `model.json`, to use files elsewhere).

**The repository holds source only.** Network weights (`*.pt`) are not in git. The weights of the example models m1809 and
g13 and the pretraining networks are assets of the GitHub release `models-v2` (2026-11-17: the emitter networks fine-tuned with the joint prior / intensity covariance; `models-v1` = before) (`python -m simcode.download_models [--set
m1809 g13 pretrain | all]`, sha256-checked); the other weights below (m25, DECODE, the archived models, the 4Pi networks) are
not release assets and have to be copied into this directory.
The small calibration json files are in git.

## What is in git

| path | what |
|---|---|
| `<set>/calib/calib_emitter_pass1_sim.json` | pass-1 solve calibration of a model: prior_off (simulated gate scenes), the fixed c, start box and posterior temperature (no sigma factors since 2026-11-17: the networks' joint prior / intensity covariance), used for real and simulated movies (the `_real` variant with a real-data sigma multiplier was removed on 2026-11-17; the archived folders `archive_*/<set>/` keep theirs as a record); the same names in every model folder (`m1809/`, `m25/`, `g13/`, `archive_*/<set>/`) |
| `<set>/calib/calib_emitter_pass2_sim.json` | the same for pass 2 |
| `<set>/calib/sigma_rescale_pass1.json`, `sigma_rescale_pass2.json` | selection-aware sigma rescale per pass (the model's `sig_u`; applied by `simcode.run` / `simcode.table`; fitted by `analysis/psf3d_nets/sigcal.py`) |
| `calib/template_calib_emitter_pass1_sim.json`, `template_calib_emitter_pass2_sim.json` | calibrations of the earlier default networks (`pretrain_detector_pass1.pt` + `pretrain_emitter_pass1.pt`; pass 2 `archive_superseded/previous_default/`): the templates from which the calibration of a new model is fitted (`analysis/psfmatch/fit_cal_set.py`, `simcode.pipeline.training`; `config.TEMPLATE_CAL_*`); the pass-1 one is also the start of the automatic-c fit |
| `calib/template_calib_emitter_pass1_brightness_noise_fit.json`, `template_calib_emitter_pass1_base.json` | the solve calibration of the automatic-c estimator `simcode/solve/brightness_noise_fit.py` (analysis only; `config.CAL_BRIGHTNESS_NOISE_FIT`); the base calibration from which the pass-1 templates were fitted (`analysis/calib/fit_cal.py`) |
| `archive_superseded/previous_default/calib/`, `archive_pass2_generic/g2ss/calib/` | calibrations of superseded networks, not used by the code: the first automatic-c table, the pass-2 movie-benchmark sim calibration, the automatic-c calibrations of the full real 0.5 / 2.5 / 20 nM runs, the earlier sigma rescale; the generic pass-2 study set g2ss |
| `insitu/calib_<movie>*.json` | in-situ PSF + camera calibration per movie (`_psf3d`: per-axis PSF; `_psf3d_avg`: averaged over the two pattern axes = the PSF the pre-v2 models were trained on (archive_psf_v1; the current models use `_psf_v2`); `calib_gauss1.3.json`: Gaussian sigma 1.3 px, the preprint's simulation PSF) |
| `insitu/calib_<movie>_psf_v2.json`, `insitu/camera_maps_<movie>_v2.npz` (1809_2, 2p5nM_m2, 20nM_m2) | in-situ PSF + camera v2 (2026-11-15; source: `out/psf_v2/calib/`, scripts and results `out/psf_v2/` (`psf_refit.py`, `write_calib.py`, `build_camera.py`, STATUS.md); also in each json's `source` / `psf_v2` keys): vectorial PSF (torch_psf InSituEstimator, Zernike Noll 5-21, focus 0.20 um, no blur, default dipole model), corrected camera (gain 0.4256 e/ADU, offset 103.9 ADU, new offset / read-noise maps; the 2.5 nM offset map without the sample structure), fitted spot heights `emitter_z_median_um` / `emitter_z_quantiles_um` of the fits not pinned at the fit limits, clipped to `emitter_z_range_um` [0, 0.40] (the training height distribution of a model trained on it: `simulate/acquisition.height_quantiles`; pinned fractions in `emitter_z_fit`, `out/psf_v2/zfit_nonpinned.py`, `write_heights.py`) |
| `insitu/camera_maps_<movie>.npz` | per-pixel offset and read-noise maps of the four movies (85-130 kB each). Binary, kept in git for now because every calibration json refers to one; **to be decided** whether they move out with the weights |
| `insitu/beads/psf_fit_*.json` | bead z-stack PSF fit (Zernike start of the in-situ fit) |
| `insitu/calib_<movie>_psf_v2.psf.zip`, `insitu/calib_gauss1.3.psf.zip`, `insitu/beads/psf_fit_*.psf.zip`, `fourpi/NPC_f_dxy15.psf.zip`; copies `m1809/`, `m25/`, `g13/` | the paper PSFs as **psf.zip** (torch-psf docs/PSF_FORMAT.md 0.2.0; 2026-11-15, `analysis/psfzip/convert_paper_psfs.py`): one file with the PSF, the camera and its per-pixel maps (no separate npz), the pattern (`patterns`), TIRF depth / emitter heights (`acquisition`), the provenance (the original json in `provenance.converted_from`), a z-stack TIFF, a preview PNG and a fit-report PDF (the in-situ v2 PSFs: with the InSituEstimator report, also in `psf.json` `report`). `simcode/models.json` and the model folders' `model.json` point at them; the json files stay and give identical results (`tests/test_psfzip.py`). SIMCODE reads both through `simcode.psf.read_calib` / `camera_maps` / `PSF.load`; the 4Pi PSF through `simcode.psf.fourpi_state` (FourPiPSF, the 4 channels with their camera crop / mirror). `fourpi/NPC_f_dxy15_spline.psf.zip` (250 kB, `spline_knots='rebuild'`: no knots stored, torch-psf tabulates the source FourPiPSF on the LUT4Pi grid when it loads; on the GPU bit-identical to the LUT4Pi of the 4Pi heads): the FourPiSplinePSF of the 4Pi heads. Camera blocks keep the calibrated scalars, the free-form keys and the extra npz entries; the pattern note is `patterns.pattern_note` (torch-psf 52c75c5) |
| `decode/*/param_*.yaml` | training / inference parameters of the two DECODE baseline models |

## Models (weights, not in git)

The shipped models are defined in `simcode/models.json` (`config.model_set(name)`, `python -m simcode.run --model <name>`);
any other model is a folder with a `model.json` (`--model <folder>`, e.g. one written by `simcode.pipeline.training.export_model`).
One model = four networks, trained on one averaged in-situ PSF, and their calibrations:

| model | PSF it was trained on | used for | files (sha256, first 16 hex digits) |
|---|---|---|---|
| `m1809` (default) | `insitu/calib_1809_2_psf_v2.json` (2026-11-15, copy `m1809/calib_1809_2_psf_v2.json`; since 2026-11-15 the model points at the psf.zip of it; before: `calib_1809_2_psf3d_avg.json`, folder `archive_psf_v1/m1809/`) | the 1809/2 movie; the realistic-PSF paper simulations | `m1809/`: `detector_pass1.pt` 18bd16f99613a67b, `emitter_pass1.pt` 92ab26c0fd0799cd, `detector_pass2.pt` 92d4487a7fe8ced5, `emitter_pass2.pt` 5329890b80311d02 |
| `m25` | `insitu/calib_2p5nM_m2_psf_v2.json` (2026-11-15; before: `calib_2p5nM_m2_psf3d_avg.json`, folder `archive_psf_v1/m25/`) | the 2.5 nM / 2 and 20 nM / 2 movies | `m25/`: `detector_pass1.pt` b3b03b7ee668d6b2, `emitter_pass1.pt` 394bb07bd409545d, `detector_pass2.pt` e7b92a4595681c60, `emitter_pass2.pt` 7ea586c87301cea6 (2026-11-17 joint-covariance fine-tune; before 81a4ca87744aaf44 / 53a7373811a6c33d; not a release asset) |
| `g13` | `insitu/calib_gauss1.3.json` | the preprint-condition simulations (Gaussian PSF) | `g13/`: `detector_pass1.pt` 7a6130f3d34ca111, `emitter_pass1.pt` 17be7aa32028e232, `detector_pass2.pt` d1037970932898d3, `emitter_pass2.pt` 0e23e4a66f8e6e06 |

File names inside every model folder (`config.MODEL_FILES`; until 2026-11-16 they carried the development names, docs/LAYOUT.md
"Renamed (2026-11-16)"): `detector_pass1.pt`, `emitter_pass1.pt`, `detector_pass2.pt`, `emitter_pass2.pt`, the calibrations
in `calib/` above, the training PSF, `model.json`.

Sizes: detector 7.3 MB, emitter network 5.5 MB, pass-2 detector 7.3 MB, pass-2 emitter network 5.6 MB.
Each model is a folder `<models>/<name>/` (networks, `calib/`, `model.json` with the recipe and sha256 of every file; the json
files are in git). Since 2026-11-14: pass-1 head with the 3D position prior (z output), GENERIC pass 2 (recipe below);
the earlier versions are the folders `archive_2d/<name>/` (2D head) and `archive_pass2_dedicated/<name>/` (3D head, pass 2
trained on the SIMCODE pattern only), not referenced by any config.

Other weights the kept code can use:

| file | what | sha256 (16) |
|---|---|---|
| `decode/hd_frame/model_final.pt`, `decode/hd_sum6/model_final.pt` | DECODE baseline models (13.9 MB each) | 4195cafc33e46b5b, 8a7cfd422df2f1fa |
| `pretrain_emitter_pass1.pt` | the earlier default pass-1 emitter network. **Needed to retrain a model**: the emitter network of every model above was fine-tuned from it (`--init pretrain_emitter_pass1.pt`, see the recipe below) | 238cea16191109f5 |
| `pretrain_detector_pass1.pt` | the earlier default detector. Needed to retrain the 4Pi detector (`analysis/fourpi/train_det4pi.py --init pretrain_detector_pass1.pt`) and the default `--detector` of a pass-1 emitter training from scratch | 3bfc6c64a75e63d8 |
| `insitu/beads/psf_fit_sign-1_na149_ball_drift.pt` | state of the bead PSF fit (2.7 kB; its json is in git) | 71e93f3b49a4c0a6 |

4Pi networks live in the 4Pi work folder (`<4Pi work folder>/models`, on qnano1 `~/simcode2/out/4pi/models`; the `--work` argument of the 4Pi tools): `head4pi_v10.pt` (learned-window
head, 9.0 MB, a44a951d4d61a1cb; contains its frozen prior), `head4pi_v3L.pt` (the prior alone = the candidate front of the
simulated movies and the start of a v10 training, 6.9 MB, 08d067f517dff6d0), `det4pi/final.pt` (detector, 7.3 MB,
5069f76c17b73c92).

Superseded models, removed from the code and from the model definitions in the 2026-11 cleanup (their weights are not needed by
anything in the tree), one folder per experiment in `archive_superseded/` with the role names above: `psf3dmix/` (the
mixture set) and `psf3dmix_pair/` (its close-pair pass 2), `gnr80/` (nanoruler generic pass 2), `ablation_1809_avg/`,
`ablation_1809_ax/` (ablation emitter networks), `earlier_architecture/emitter_pass1.pt` (not loadable by the package),
`previous_default/` (the earlier default pass 2 `detector_pass2.pt` / `emitter_pass2.pt`, calibrations in its `calib/`). The
generic pass-2 study sets (out/p2generic) are `archive_pass2_generic/<set>/` (g1809, g2c, g2cc, g2s, g2ss, g2w).

## How a model is made (m1809 recipe)

```bash
C=models/insitu/calib_1809_2_psf_v2.json         # the PSF + camera of the model (PSF v2; training heights from its emitter_z_quantiles_um)
python -m simcode.detector.train --out runs/S/detector --steps 40000 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 --nnw 1 \
    --seed 0 --calib $C
python -m simcode.emitter.train --out runs/S/emitter --steps 18000 --batch 16 --lr 1.5e-4 --ch 96 --detector runs/S/detector/final.pt \
    --tau 15,80 --hold <heldout5.pt of python -m simcode.emitter.heldout --set 5> --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 \
    --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init pretrain_emitter_pass1.pt --pct 0.1 --ckpt_every 6000 --calib $C
# pass 2: the detector starts DEDICATED (SIMCODE pattern only; --illum 1 from the pass-1 detector), then two fine-tuning stages
# on the 50 / 50 mix of SIMCODE and generic patterns (the pattern share is fixed in the trainers, see below)
python -m simcode.detector.train --out runs/S/detector_pass2 --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11 \
    --eval_every 2500 --ckpt_every 5000 --illum 1 --init runs/S/detector/final.pt --calib $C
python -m simcode.detector.train --out runs/S/detector_pass2_mix --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 61 \
    --eval_every 5000 --ckpt_every 10000 --init runs/S/detector_pass2/final.pt --pass1_ref runs/S/detector/final.pt --calib $C
python -m simcode.detector.train --out runs/S/detector_pass2_cont --steps 40000 --batch 16 --size 64 --lr 1e-4 --pct 0.05 --w_off 1.0 --seed 71 \
    --eval_every 5000 --ckpt_every 10000 --init runs/S/detector_pass2_mix/final.pt --pass1_ref runs/S/detector/final.pt --calib $C
# pass-2 emitter network: three stages (candidates from the first pass-2 detector runs/S/detector_pass2; a pass-2 --detector makes it a pass-2
# network, zero-initialised from the pass-1 network runs/S/emitter)
python -m simcode.emitter.train --out runs/S/emitter_pass2 --detector runs/S/detector_pass2/ckpt_25000.pt --init runs/S/emitter/final.pt --steps 24000 \
    --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0 \
    --pert_model v1 --calib $C
python -m simcode.emitter.train --out runs/S/emitter_pass2_long --detector runs/S/detector_pass2/final.pt --init runs/S/emitter_pass2/final.pt --steps 50000 \
    --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 \
    --hold_v1 15 --calib $C
python -m simcode.emitter.train --out runs/S/emitter_pass2_final --detector runs/S/detector_pass2/final.pt --init runs/S/emitter_pass2_long/final.pt --steps 50000 \
    --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 41 --eval_every 5000 --ckpt_every 10000 --no_illum 0 \
    --hold_v1 0 --calib $C
# -> <set>/detector_pass1.pt, emitter_pass1.pt = runs/S/detector, emitter; detector_pass2.pt = runs/S/detector_pass2_cont;
#    emitter_pass2.pt = runs/S/emitter_pass2_final (final.pt). The stage names are those of simcode.pipeline.training; the recipe
#    scripts of the shipped models (out/psf_v2/train/chain.sh, det2.sh) name these directories det, hd, p2det, p2det_m,
#    p2det_c, p2head, p2headL, p2headL2.
```
**Pattern share of the pass-2 training (fixed since 2026-11-15, no `--p_std` option; `simulate/batches.P_STD_DEDICATED`,
`P_STD`):** a pass-2 detector initialised from a pass-1 detector sees the SIMCODE pattern only (share 1, the dedicated
detector); a pass-2 detector initialised from a pass-2 detector (fine-tuning) and every pass-2 emitter-network stage see the
SIMCODE pattern in half the FOVs and the generic draw in the other half (share 0.5). The value is stored in each checkpoint
(`args['p_std']`). Why the detector starts dedicated: the generic pass-2 cost of m25 / g13 was in the generic detector
(close-pair recall, false positives; out/generic_mix/STATUS.md). On m25 (v2 gate scenes, the fine-tuned detector vs the
dedicated one with the same emitter network, out/psf_v2/eval/m25v2/det2/accept_f2_vs_dd.txt): RMS/CRLB 200-1e9 iso +0.002,
pair1 -0.020, pair2 -0.024, rs25qm -0.012; recall at 0.5 equal or higher on every scene (the earlier generic detector_pass2_cont: pair1
recall at matched FP below the dedicated one). Recipe scripts of the shipped models: out/psf_v2/train/chain.sh, det2.sh.
**History of the shipped models:** the m1809 pass-2 detector (and g13's) comes from the earlier chain detector_pass2 (share 0.25, seed 11)
-> detector_pass2_cont (0.5, lr 1e-4, seed 51); the m25 pass-2 detector is the recipe above (out/psf_v2/train/det2.sh: d2ded -> gft1 -> gft2
= detector_pass2 -> detector_pass2_mix -> detector_pass2_cont); every shipped pass-2 emitter network was trained with the shares 0.25 / 0.25 / 0.5 in
emitter_pass2 / emitter_pass2_long / emitter_pass2_final (now 0.5 in all three stages; the generic share of the emitter network was about neutral,
out/generic_mix). Retraining those exact networks needs a checkout before the 2026-11-15 cleanup (master f48ab14, the former main 710d34b).
Pass 2 is trained on generic illumination patterns (since 2026-11-14 the only pass-2 training draw; `simulate/acquisition.AcquisitionSim._generic`,
out/p2generic): per FOV, with probability 1 - p_std (above), pitch log-uniform 1.8-4.0 px, orientations -40..40 / 50..130 deg
(separation 60-120 deg), random k sign, independent per-frame phases, frame order cycle / permutation / interleaved, per-angle
depth 0.5-1.0 and per-slot / per-angle power 0.8-1.25; the other FOVs keep the SIMCODE (DMD) pattern. So a model's pass 2
accepts any measured two-orientation pattern (nanoruler, 4-step, random phases), at 0-2 % cost on SIMCODE patterns vs the
earlier pass 2 trained on the SIMCODE pattern only (out/p2generic/STATUS.md, out/z3d/STATUS.md).
One trainer per network, for both passes (`simcode.detector.train`, `simcode.emitter.train`; until 2026-11-15
`simcode.det.up2` / `simcode.p2.det2` and `simcode.hd.sc4` / `simcode.p2.head2`, docs/CLEANUP_LOG.md): the pass follows from the
checkpoints (detector: `--illum 1`, or a pass-2 `--init`; emitter network: a pass-2 `--detector`, or a pass-2 `--init`), and the
flags whose defaults differ between the passes (lr, seed, steps, ...) take the default of that pass. Since 2026-11-14 the
trainers' defaults are this recipe's former flags (`--up 1`, `--em_photons 40,2500`, `--arch` (the network tag), `--paper 1`,
`--bright_tail 0.1,2000,20000`, `--pert_model real`). `--calib` is required.
Model g13 adds `--dens 0.0005,0.06` to the detector and `--tau 3,80 --dens 0.0005,0.06` to the other trainers.
**Joint prior / intensity covariance (2026-11-17).** The shipped emitter networks are the networks of the recipe above fine-tuned
with the joint (x, y, I_1..6) Gaussian NLL (the 12 new output rows C start at zero, i.e. at the recipe's network; detectors frozen):
```bash
# pass 1 (detector of the pass-1 stage; the recipe flags of the pass-1 emitter network)
python -m simcode.emitter.train --out runs/S/emitter_joint --init runs/S/emitter/final.pt --detector runs/S/detector/final.pt --steps 12000 \
    --batch 16 --lr 1e-4 --pct 0.1 --ch 96 --tau 15,80 --hold <heldout5.pt> --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 \
    --seed 101 --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --ckpt_every 4000 --calib $C
# pass 2 (the detector the pass-2 emitter network was trained with, runs/S/detector_pass2/final.pt)
python -m simcode.emitter.train --out runs/S/emitter_pass2_joint --init runs/S/emitter_pass2_final/final.pt --detector runs/S/detector_pass2/final.pt \
    --steps 20000 --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 141 --eval_every 5000 --ckpt_every 5000 --no_illum 0 \
    --hold_v1 0 --calib $C
# -> emitter_pass1.pt, emitter_pass2.pt (final.pt); g13: + --dens 0.0005,0.06 --tau 3,80 (pass 2 too). Paper branch: analysis/sigma_joint/train.sh
```
Calibrations: `python -m simcode.processing.gate_dump` of the new networks on the gate scenes rendered with the model's PSF
(`--calib $C --gate_dir <gate npz folder>`), then `python -m simcode.calibration.fit_cal_set` (prior_off; templates: `calib/template_calib_emitter_pass{1,2}_sim.json` above), then the
sigma rescale `analysis/psf3d_nets/sigcal.py fit` on a simulated movie with ground truth.
