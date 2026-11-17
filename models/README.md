# models/

This directory is where the code looks up the names of networks and calibrations (`simcode/config.py`: `MODELS` =
`<repo>/models`; there is no setting to move it: give a path, or a model folder with a `model.json`, to use files elsewhere).

**Network weights (`*.pt`) are not in git.** They are assets of the GitHub release `models-v2` of the repository (2026-11-17: the emitter networks fine-tuned with the joint prior / intensity covariance; `models-v1` = before); fetch,
verify (sha256) and unpack them with

```bash
python -m simcode.download_models                    # m1809 and g13 -> models/m1809/, models/g13/
python -m simcode.download_models --set pretrain     # pretrain_detector_pass1.pt, pretrain_emitter_pass1.pt (to train a new model)
python -m simcode.download_models --set all --dest <models folder>
```

| release asset | contents (paths relative to `models/`) |
|---|---|
| `simcode_model_m1809.zip` | `m1809/detector_pass1.pt`, `emitter_pass1.pt`, `detector_pass2.pt`, `emitter_pass2.pt` |
| `simcode_model_g13.zip` | `g13/detector_pass1.pt`, `emitter_pass1.pt`, `detector_pass2.pt`, `emitter_pass2.pt` |
| `simcode_pretrain.zip` | `pretrain_detector_pass1.pt`, `pretrain_emitter_pass1.pt` |

The sha256 of every asset and file are in `simcode/download_models.py`.

## What is in git

| path | what |
|---|---|
| `<set>/model.json` | the model (format of `simcode/models.json`, paths relative to the folder) with its training record |
| `<set>/calib/calib_emitter_pass1_sim.json` | pass-1 solve calibration of a model: prior_off (simulated gate scenes), start box and posterior temperature (no brightness noise c since 2026-11-17: it is estimated per dataset with the pattern) (no sigma factors since 2026-11-17: the networks' joint prior / intensity covariance); used for real and simulated movies (the `_real` variant with a real-data sigma multiplier was removed on 2026-11-17) |
| `<set>/calib/calib_emitter_pass2_sim.json` | the same for pass 2 |
| `m1809/calib_1809_2_psf_v2.psf.zip` (+ `.json`, `camera_maps_1809_2_v2.npz`) | the PSF + camera m1809 was trained on: in-situ vectorial PSF v2 of the 1809/2 movie (Zernike Noll 5-21, focus 0.20 um, gain 0.4256 e/ADU, offset / read-noise maps, fitted emitter heights) as psf.zip (torch-psf PSF_FORMAT 0.2.0: PSF, camera with its per-pixel maps, pattern, heights, provenance, z-stack TIFF, preview, fit report); the json + npz give identical results (`tests/test_psfzip.py`) |
| `g13/calib_gauss1.3.psf.zip` (+ `.json`) | Gaussian PSF sigma 1.3 px (the preprint's simulation PSF), the PSF of g13 |
| `calib/template_calib_emitter_pass{1,2}_sim.json` | calibrations of the pretraining networks: the templates from which the calibration of a new model is fitted (`simcode.pipeline.training`, `simcode.calibration.fit_cal_set`; `config.TEMPLATE_CAL_*`); the pass-1 one is also the start of the automatic-c fit |
| `calib/template_calib_emitter_pass1_base.json` | the base calibration from which the pass-1 templates were fitted (`analysis/calib/fit_cal.py`, paper branch) |
| `insitu/calib_0.5nM.json`, `camera_maps_0.5nM.npz`, `camera_maps_2p5nM_m2.npz` | in-situ PSF + camera of the 0.5 nM movie and the camera maps of the 2.5 nM movie: the two acquisition presets of the training simulator (`simcode/simulate/acquisition.PRESETS`), the default PSF of the gate scenes / CRLB / held-out sets (`config.INSITU_05`), the read-noise map of the example pipeline configs |
| `insitu/camera_maps_1809_2.npz`, `calib_gauss1.3.json` | camera maps of the 1809/2 movie (referred to by `g13/calib_gauss1.3.json` and the psf3d calibrations) and the Gaussian PSF json (`tests/test_psfzip.py`) |
| `insitu/calib_1809_2_psf3d.json`, `calib_1809_2_psf3d_avg.json`, `calib_2p5nM_m2_psf_v2.json`, `camera_maps_2p5nM_m2_v2.npz` | in-situ calibrations used by the tests (`tests/test_fast_paths.py`, `tests/test_heights.py`) |

The paper branch of the repository adds the model `m25` (2.5 nM / 2 and 20 nM / 2 movies), the in-situ calibrations of
all paper movies, the bead PSF fit, the 4Pi PSF, the DECODE baseline parameters and the superseded / archived models.

## Models

The shipped models are defined in `simcode/models.json` (`config.model_set(name)`, `python -m simcode.run --model <name>`);
any other model is a folder with a `model.json` (`--model <folder>`, e.g. one written by `simcode.pipeline.training.export_model`).
One model = four networks, trained on one averaged in-situ PSF, and their calibrations:

| model | PSF it was trained on | used for | files (sha256, first 16 hex digits) |
|---|---|---|---|
| `m1809` (default) | `m1809/calib_1809_2_psf_v2.psf.zip` | the 1809/2 movie; the realistic-PSF paper simulations | `m1809/`: `detector_pass1.pt` 18bd16f99613a67b, `emitter_pass1.pt` 92ab26c0fd0799cd, `detector_pass2.pt` 92d4487a7fe8ced5, `emitter_pass2.pt` 5329890b80311d02 |
| `g13` | `g13/calib_gauss1.3.psf.zip` | the preprint-condition simulations (Gaussian PSF) | `g13/`: `detector_pass1.pt` 7a6130f3d34ca111, `emitter_pass1.pt` 17be7aa32028e232, `detector_pass2.pt` d1037970932898d3, `emitter_pass2.pt` 0e23e4a66f8e6e06 |

Sizes: detector 7.3 MB, emitter network 5.5 MB, pass-2 detector 7.3 MB, pass-2 emitter network 5.6 MB. Both models have the
pass-1 head with the 3D position prior (z output) and a GENERIC pass 2 (recipe below: accepts any measured two-orientation
pattern).

Pretraining networks (`simcode_pretrain.zip`):

| file | what | sha256 (16) |
|---|---|---|
| `pretrain_emitter_pass1.pt` | the earlier default pass-1 emitter network. **Needed to train a model**: the emitter network of every model above was fine-tuned from it (`--init pretrain_emitter_pass1.pt`, recipe below; `simcode.pipeline.training`) | 238cea16191109f5 |
| `pretrain_detector_pass1.pt` | the earlier default detector: the start of a new 4Pi detector and the default `--detector` of a pass-1 emitter training from scratch | 3bfc6c64a75e63d8 |

## How a model is made (m1809 recipe)

```bash
C=models/m1809/calib_1809_2_psf_v2.json          # the PSF + camera of the model (PSF v2; training heights from its emitter_z_quantiles_um)
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
#    scripts of the shipped models named these directories det, hd, p2det, p2det_m, p2det_c, p2head, p2headL, p2headL2.
```
**Pattern share of the pass-2 training (fixed since 2026-11-15, no `--p_std` option; `simulate/batches.P_STD_DEDICATED`,
`P_STD`):** a pass-2 detector initialised from a pass-1 detector sees the SIMCODE pattern only (share 1, the dedicated
detector); a pass-2 detector initialised from a pass-2 detector (fine-tuning) and every pass-2 emitter-network stage see the
SIMCODE pattern in half the FOVs and the generic draw in the other half (share 0.5). The value is stored in each checkpoint
(`args['p_std']`). Why the detector starts dedicated: the cost of a generic pass 2 was in a generic detector (close-pair
recall, false positives); a detector fine-tuned from a dedicated one has equal or better RMS / CRLB and recall on the gate
scenes.
Pass 2 is trained on generic illumination patterns (`simulate/acquisition.AcquisitionSim._generic`): per FOV, with
probability 1 - p_std (above), pitch log-uniform 1.8-4.0 px, orientations -40..40 / 50..130 deg (separation 60-120 deg),
random k sign, independent per-frame phases, frame order cycle / permutation / interleaved, per-angle depth 0.5-1.0 and
per-slot / per-angle power 0.8-1.25; the other FOVs keep the SIMCODE (DMD) pattern. So a model's pass 2 accepts any measured
two-orientation pattern (nanoruler, 4-step, random phases), at 0-2 % cost on SIMCODE patterns vs a pass 2 trained on the
SIMCODE pattern only.
One trainer per network, for both passes (`simcode.detector.train`, `simcode.emitter.train`; until 2026-11-15
`simcode.det.up2` / `simcode.p2.det2` and `simcode.hd.sc4` / `simcode.p2.head2`): the pass follows from the
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
(`--calib $C --gate_dir <gate npz folder>`), then `python -m simcode.calibration.fit_cal_set` (prior_off; templates: `calib/template_calib_emitter_pass{1,2}_sim.json` above).
Nothing else is fitted: no sigma rescale (removed 2026-11-17) and no brightness noise (estimated per dataset by the pipeline).
The automatic pipeline (`python -m simcode.pipeline`, stage e: `simcode/pipeline/training.py`) runs this recipe for a new
acquisition.
