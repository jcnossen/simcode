# models/

This directory is where the code looks for networks and calibrations (`simcode/config.py`: root `models`, default
`<repo>/models`; env `SIMCODE_MODELS` or the config key `models` to put it elsewhere).

**The repository holds source only.** Network weights (`*.pt`) are not in git: they are distributed separately and have to
be copied into this directory (or the directory `models` points to). Where they are distributed from is not decided yet.
The small calibration json files are in git.

## What is in git

| path | what |
|---|---|
| `calib/calib_sc_hd_<set>.json`, `calib_sc_hd_<set>_real.json` | pass-1 solve calibration of a model: sigma table from the gate scenes (`_real`: with the real-data mismatch multiplier) |
| `calib/calib_p2L_<set>.json`, `calib_p2L_<set>_real.json` | the same for pass 2 |
| `calib/sigu/sigu_p1_<set>.json`, `sigu_p2_<set>.json` | selection-aware sigma rescale per pass (applied by `simcode.run` / `simcode.table`; fitted by `analysis/psf3d_nets/sigcal.py`) |
| `calib/calib_sc_hd_sc5nbb15_pm*.json`, `calib_p2L.json`, `calib_p2L_b10_real.json`, `calib_p2L_movie_b10.json`, `calib_autoc_r*.json`, `sigu/sigu_p*_cur.json` | calibrations of the earlier default head (hd_sc5nb / p2_head_long). Still used: as the templates from which the calibration of a new model is fitted (`analysis/psfmatch/fit_cal_set.py`), and as the sigma table of the automatic-c estimator `simcode/solve/autoc.py` (analysis only) |
| `insitu/calib_<movie>*.json` | in-situ PSF + camera calibration per movie (`_psf3d`: per-axis PSF; `_psf3d_avg`: averaged over the two pattern axes = the PSF a model is trained on; `calib_gauss1.3.json`: Gaussian sigma 1.3 px, the preprint's simulation PSF) |
| `insitu/calib_<movie>_psf_v2.json`, `insitu/camera_maps_<movie>_v2.npz` (1809_2, 2p5nM_m2, 20nM_m2) | in-situ PSF + camera v2 (2026-11-15; source: `out/psf_v2/calib/`, scripts and results `out/psf_v2/` (`psf_refit.py`, `write_calib.py`, `build_camera.py`, STATUS.md); also in each json's `source` / `psf_v2` keys): vectorial PSF (torch_psf InSituEstimator, Zernike Noll 5-21, focus 0.20 um, no blur, default dipole model), corrected camera (gain 0.4256 e/ADU, offset 103.9 ADU, new offset / read-noise maps; the 2.5 nM offset map without the sample structure), fitted spot heights `emitter_z_median_um` / `emitter_z_quantiles_um` of the fits not pinned at the fit limits, clipped to `emitter_z_range_um` [0, 0.40] (the training height distribution of a model trained on it: `simulate/acquisition.height_quantiles`; pinned fractions in `emitter_z_fit`, `out/psf_v2/zfit_nonpinned.py`, `write_heights.py`) |
| `insitu/camera_maps_<movie>.npz` | per-pixel offset and read-noise maps of the four movies (85-130 kB each). Binary, kept in git for now because every calibration json refers to one; **to be decided** whether they move out with the weights |
| `insitu/beads/psf_fit_*.json` | bead z-stack PSF fit (Zernike start of the in-situ fit) |
| `decode/*/param_*.yaml` | training / inference parameters of the two DECODE baseline models |

## Models (weights, not in git)

The shipped models are defined in `simcode/models.json` (`config.model_set(name)`, `python -m simcode.run --model <name>`);
any other model is a folder with a `model.json` (`--model <folder>`, e.g. one written by `simcode.pipeline.training.export_model`).
One model = four networks, trained on one averaged in-situ PSF, and their calibrations:

| model | PSF it was trained on | used for | files (sha256, first 16 hex digits) |
|---|---|---|---|
| `m1809` (default) | `insitu/calib_1809_2_psf3d_avg.json` | the 1809/2 movie; the realistic-PSF paper simulations | `m1809/`: `det_m1809.pt` 8da168c18b993a54, `hd_m1809.pt` 75d26a156dcde604, `p2det_m1809.pt` b2610b51ccd07436, `p2headL_m1809.pt` ccfab3f89eec40ba |
| `m25` | `insitu/calib_2p5nM_m2_psf_v2.json` (2026-11-15; before: `calib_2p5nM_m2_psf3d_avg.json`, folder `archive_psf_v1/m25/`) | the 2.5 nM / 2 and 20 nM / 2 movies | `m25/`: `det_m25.pt` b3b03b7ee668d6b2, `hd_m25.pt` 81a4ca87744aaf44, `p2det_m25.pt` e7b92a4595681c60, `p2headL_m25.pt` 53a7373811a6c33d |
| `g13` | `insitu/calib_gauss1.3.json` | the preprint-condition simulations (Gaussian PSF) | `g13/`: `det_g13.pt` 7a6130f3d34ca111, `hd_g13.pt` cbd17c8b098ccb3e, `p2det_g13.pt` d1037970932898d3, `p2headL_g13.pt` 7e23fbd57c05e2f2 |

Sizes: detector 7.3 MB, head 5.5 MB, pass-2 detector 7.3 MB, pass-2 head 5.6 MB.
Each model is a folder `<models>/<name>/` (networks, `calib/`, `model.json` with the recipe and sha256 of every file; the json
files are in git). Since 2026-11-14: pass-1 head with the 3D position prior (z output), GENERIC pass 2 (recipe below);
the earlier versions are the folders `archive_2d/<name>/` (2D head) and `archive_p2dedicated/<name>/` (3D head, pass 2
trained on the SIMCODE pattern only), not referenced by any config.

Other weights the kept code can use:

| file | what | sha256 (16) |
|---|---|---|
| `decode/hd_frame/model_final.pt`, `decode/hd_sum6/model_final.pt` | DECODE baseline models (13.9 MB each) | 4195cafc33e46b5b, 8a7cfd422df2f1fa |
| `hd_sc5nb.pt` | the earlier default pass-1 head. **Needed to retrain a model**: the head of every model above was fine-tuned from it (`--init hd_sc5nb.pt`, see the recipe below) | 238cea16191109f5 |
| `det_n1.pt` | the earlier default detector. Needed to retrain the 4Pi detector (`analysis/fourpi/train_det4pi.py --init det_n1.pt`) | 3bfc6c64a75e63d8 |
| `insitu/beads/psf_fit_sign-1_na149_ball_drift.pt` | state of the bead PSF fit (2.7 kB; its json is in git) | 71e93f3b49a4c0a6 |

4Pi networks live in the 4Pi work directory (`<out>/4pi/models`, config root `fourpi`): `head4pi_v10.pt` (learned-window
head, 9.0 MB, a44a951d4d61a1cb; contains its frozen prior), `head4pi_v3L.pt` (the prior alone = the candidate front of the
simulated movies and the start of a v10 training, 6.9 MB, 08d067f517dff6d0), `det4pi/final.pt` (detector, 7.3 MB,
5069f76c17b73c92).

Superseded models, removed from the code and from the model definitions in the 2026-11 cleanup (their weights are not needed by
anything in the tree): the mixture set `psf3dmix` and its close-pair variant, the ablation heads `hd_abl1809_avg / ax`,
`hd_sc7bg`, `p2_det.pt`, `p2_head_long.pt`.

## How a model is made (m1809 recipe)

```bash
C=models/insitu/calib_1809_2_psf3d_avg.json      # the averaged PSF of the model
python -m simcode.detector.train --out runs/S/det --steps 40000 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 --nnw 1 \
    --seed 0 --calib $C
python -m simcode.emitter.train --out runs/S/hd --steps 18000 --batch 16 --lr 1.5e-4 --ch 96 --detector runs/S/det/final.pt \
    --tau 15,80 --hold <out>/emitter_heldout/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 \
    --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init hd_sc5nb.pt --pct 0.1 --ckpt_every 6000 --calib $C
# pass 2 (generic pattern draw, share --p_std of the SIMCODE pattern): detector, then its continuation
python -m simcode.detector.train --out runs/S/p2det --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11 \
    --eval_every 2500 --ckpt_every 5000 --illum 1 --init runs/S/det/final.pt --calib $C --p_std 0.25
python -m simcode.detector.train --out runs/S/p2det_c --steps 40000 --batch 16 --size 64 --lr 1e-4 --pct 0.05 --w_off 1.0 --seed 51 \
    --eval_every 5000 --ckpt_every 10000 --init runs/S/p2det/final.pt --pass1_ref runs/S/det/final.pt --calib $C --p_std 0.5
# pass-2 emitter network: three stages (candidates from the first pass-2 detector runs/S/p2det; a pass-2 --detector makes it a pass-2
# network, zero-initialised from the pass-1 network runs/S/hd)
python -m simcode.emitter.train --out runs/S/p2head --detector runs/S/p2det/ckpt_25000.pt --init runs/S/hd/final.pt --steps 24000 \
    --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0 \
    --pert_model v1 --calib $C --p_std 0.25
python -m simcode.emitter.train --out runs/S/p2headL --detector runs/S/p2det/final.pt --init runs/S/p2head/final.pt --steps 50000 \
    --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 \
    --hold_v1 15 --calib $C --p_std 0.25
python -m simcode.emitter.train --out runs/S/p2headL2 --detector runs/S/p2det/final.pt --init runs/S/p2headL/final.pt --steps 50000 \
    --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 41 --eval_every 5000 --ckpt_every 10000 --no_illum 0 \
    --hold_v1 0 --calib $C --p_std 0.5
# -> det_S.pt, hd_S.pt = runs/S/det, hd; p2det_S.pt = runs/S/p2det_c; p2headL_S.pt = runs/S/p2headL2 (final.pt)
```
**Pass-2 detector of m25 (2026-11-15, replaces p2det / p2det_c above for m25):** the generic pass-2 cost of m25 / g13 was in the
generic detector (close-pair recall, false positives; out/generic_mix/STATUS.md), so the detector starts from a DEDICATED one
(SIMCODE pattern only, `--p_std 1`) and is then fine-tuned on the 50 / 50 pattern mix (`--p_std 0.5`), 40k + 40k:
```bash
python -m simcode.detector.train --out runs/S/d2ded --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11 \
    --eval_every 2500 --ckpt_every 5000 --illum 1 --init runs/S/det/final.pt --calib $C --p_std 1
python -m simcode.detector.train --out runs/S/gft1 --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 61 \
    --eval_every 5000 --ckpt_every 10000 --init runs/S/d2ded/final.pt --pass1_ref runs/S/det/final.pt --calib $C --p_std 0.5
python -m simcode.detector.train --out runs/S/gft2 --steps 40000 --batch 16 --size 64 --lr 1e-4 --pct 0.05 --w_off 1.0 --seed 71 \
    --eval_every 5000 --ckpt_every 10000 --init runs/S/gft1/final.pt --pass1_ref runs/S/det/final.pt --calib $C --p_std 0.5
# -> p2det_S.pt = runs/S/gft2/final.pt; the pass-2 emitter network stays the one trained on runs/S/p2det (above); the pass-2
# calibrations and sigma rescale are fitted with this detector
```
On m25 (v2 gate scenes, vs the dedicated d2ded with the same emitter network, out/psf_v2/eval/m25v2/det2/accept_f2_vs_dd.txt):
RMS/CRLB 200-1e9 iso +0.002, pair1 -0.020, pair2 -0.024, rs25qm -0.012; recall at 0.5 equal or higher on every scene
(the generic p2det_c: pair1 recall at matched FP below the dedicated one). Recipe scripts: out/psf_v2/train/chain.sh, det2.sh.
Pass 2 is trained on generic illumination patterns (since 2026-11-14 the only pass-2 training draw; `simulate/acquisition.AcquisitionSim._generic`,
out/p2generic): per FOV, with probability 1 - `--p_std`, pitch log-uniform 1.8-4.0 px, orientations -40..40 / 50..130 deg
(separation 60-120 deg), random k sign, independent per-frame phases, frame order cycle / permutation / interleaved, per-angle
depth 0.5-1.0 and per-slot / per-angle power 0.8-1.25; the other FOVs keep the SIMCODE (DMD) pattern. So a model's pass 2
accepts any measured two-orientation pattern (nanoruler, 4-step, random phases), at 0-2 % cost on SIMCODE patterns vs the
earlier pass 2 trained on the SIMCODE pattern only (out/p2generic/STATUS.md, out/z3d/STATUS.md).
One trainer per network, for both passes (`simcode.detector.train`, `simcode.emitter.train`; until 2026-11-15
`simcode.det.up2` / `simcode.p2.det2` and `simcode.hd.sc4` / `simcode.p2.head2`, docs/CLEANUP_LOG.md): the pass follows from the
checkpoints (detector: `--illum 1`, or a pass-2 `--init`; emitter network: a pass-2 `--detector`, or a pass-2 `--init`), and the
flags whose defaults differ between the passes (lr, seed, steps, ...) take the default of that pass. Since 2026-11-14 the
trainers' defaults are this recipe's former flags (`--up 1`, `--em_photons 40,2500`, `--arch sc4n`, `--paper 1`,
`--bright_tail 0.1,2000,20000`, `--pert_model real`). `--calib` is required.
Model g13 adds `--dens 0.0005,0.06` to the detector and `--tau 3,80 --dens 0.0005,0.06` to the other trainers.
Calibrations: `python -m simcode.processing.gate_dump` of the new networks on the gate scenes rendered with the model's PSF
(`SIMCODE_GATE_CALIB=$C`), then `analysis/psfmatch/fit_cal_set.py` (templates: the hd_sc5nb calibrations above), then the
sigma rescale `analysis/psf3d_nets/sigcal.py fit` on a simulated movie with ground truth.
