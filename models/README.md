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
| `calib/sigu/sigu_p1_<set>.json`, `sigu_p2_<set>.json` | selection-aware sigma rescale per pass (`simcode.strict --sig_u`; fitted by `analysis/psf3d_nets/sigcal.py`) |
| `calib/calib_sc_hd_sc5nbb15_pm*.json`, `calib_p2L.json`, `calib_p2L_b10_real.json`, `calib_p2L_movie_b10.json`, `calib_autoc_r*.json`, `sigu/sigu_p*_cur.json` | calibrations of the earlier default head (hd_sc5nb / p2_head_long). Still used: as the templates from which the calibration of a new model is fitted (`analysis/psfmatch/fit_cal_set.py`), and as the sigma table of the automatic-c estimator `simcode/autoc.py` (analysis only) |
| `insitu/calib_<movie>*.json` | in-situ PSF + camera calibration per movie (`_psf3d`: per-axis PSF; `_psf3d_avg`: averaged over the two pattern axes = the PSF a model is trained on; `calib_gauss1.3.json`: Gaussian sigma 1.3 px, the preprint's simulation PSF) |
| `insitu/camera_maps_<movie>.npz` | per-pixel offset and read-noise maps of the four movies (85-130 kB each). Binary, kept in git for now because every calibration json refers to one; **to be decided** whether they move out with the weights |
| `insitu/beads/psf_fit_*.json` | bead z-stack PSF fit (Zernike start of the in-situ fit) |
| `decode/*/param_*.yaml` | training / inference parameters of the two DECODE baseline models |

## Models (weights, not in git)

The shipped models are defined in `simcode/models.json` (`config.model_set(name)`, `python -m simcode.run --model <name>`);
any other model is a folder with a `model.json` (`--model <folder>`, e.g. one written by `simcode.auto.train.export_model`).
One model = four networks, trained on one averaged in-situ PSF, and their calibrations:

| model | PSF it was trained on | used for | files (sha256, first 16 hex digits) |
|---|---|---|---|
| `m1809` (default) | `insitu/calib_1809_2_psf3d_avg.json` | the 1809/2 movie; the realistic-PSF paper simulations | `det_m1809.pt` 8da168c18b993a54, `hd_m1809.pt` c12e923724267195, `p2det_m1809.pt` c443106b6d9c10bb, `p2headL_m1809.pt` 71f5b5d1e310dd5a |
| `m25` | `insitu/calib_2p5nM_m2_psf3d_avg.json` | the 2.5 nM / 2 movie | `det_m25.pt` 96b7d846258d9b9d, `hd_m25.pt` 6da9df67df5f9460, `p2det_m25.pt` 5d052eb228ea53cb, `p2headL_m25.pt` 01e24eacec9abf33 |
| `g13` | `insitu/calib_gauss1.3.json` | the preprint-condition simulations (Gaussian PSF) | `det_g13.pt` 7a6130f3d34ca111, `hd_g13.pt` 02f4768f849df0a9, `p2det_g13.pt` 065435555ff1ceed, `p2headL_g13.pt` 17a3ba6324da5841 |

Sizes: detector 7.3 MB, head 5.5 MB, pass-2 detector 7.3 MB, pass-2 head 5.6 MB.

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
python -m simcode.det.up2 train --out runs/S/det --steps 40000 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 --nnw 1 \
    --seed 0 --calib $C
python -m simcode.hd.sc4 train --out runs/S/hd --steps 18000 --batch 16 --lr 1.5e-4 --ch 96 --det runs/S/det/final.pt \
    --tau 15,80 --hold <out>/sc4/heldout5.pt --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 \
    --w_mr 2.0 --w_slope 50.0 --bf16 1 --eval_every 2000 --init hd_sc5nb.pt --pct 0.1 --ckpt_every 6000 --calib $C
python -m simcode.p2.det2 train --out runs/S/p2det --steps 40000 --batch 16 --size 64 --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11 \
    --eval_every 2500 --ckpt_every 5000 --init_n1 runs/S/det/final.pt --calib $C
python -m simcode.p2.head2 train --out runs/S/p2head --det runs/S/p2det/ckpt_25000.pt --init runs/S/hd/final.pt --steps 24000 \
    --batch 16 --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --eval_every 4000 --ckpt_every 8000 --no_illum 0 \
    --pert_model v1 --calib $C
python -m simcode.p2.head2 train --out runs/S/p2headL --det runs/S/p2det/final.pt --init runs/S/p2head/final.pt --steps 50000 \
    --batch 16 --lr 1e-4 --pct 0.05 --w_mr 2.0 --w_slope 50.0 --seed 31 --eval_every 2000 --ckpt_every 10000 --no_illum 0 \
    --hold_v1 15 --calib $C
# -> models/det_S.pt, hd_S.pt, p2det_S.pt, p2headL_S.pt (the four final.pt)
```
Since 2026-11-14 the trainers' defaults are this recipe's former flags (`--up 1`, `--em_photons 40,2500`, `--arch sc4n`,
`--paper 1`, `--bright_tail 0.1,2000,20000`, `--pert_model real`); the old command lines still run (the removed flags are
accepted with exactly these values). `--calib` is required.
Model g13 adds `--dens 0.0005,0.06` to the detector and `--tau 3,80 --dens 0.0005,0.06` to the other three trainers.
Calibrations: `python -m simcode.p2.dump` of the new networks on the gate scenes rendered with the model's PSF
(`SIMCODE_GATE_CALIB=$C`), then `analysis/psfmatch/fit_cal_set.py` (templates: the hd_sc5nb calibrations above), then the
sigma rescale `analysis/psf3d_nets/sigcal.py fit` on a simulated movie with ground truth.
