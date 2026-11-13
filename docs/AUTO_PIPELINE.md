# Automatic per-dataset pipeline (`simcode.auto`): design

Status: design 2026-11-13; implementation in `simcode/auto/` (stages a, b, c, d, f, g; e as a wrapper with `--quick`).
Validation numbers: section 9 (filled in after the runs).

Goal: one config file per dataset, then one command that

1. calibrates the camera from the movie,
2. fits the bead z-stack (optional),
3. localizes the movie with a pattern-free spot finder to collect isolated emitter tracks (no external SMLM table),
4. fits the in-situ PSF on those tracks (variable projection, line blur, photoselection) with a held-out check,
5. trains the PSF-matched network set (recipe of `runs/psfmatch/chain2.sh`),
6. processes the whole movie (`simcode.run` with that set),
7. writes a report page.

```bash
python -m simcode.auto <config.yaml> [--stages camera,spots,...|all] [--quick] [--redo <stage>] [--gpus 1,2]
```

```python
from simcode.auto import Dataset
ds = Dataset('simcode/auto/configs/1809_2.yaml')
ds.run()                                   # all stages, skipping finished ones
ds.run(stages=['camera', 'spots'])         # some stages
ds.stage('psf')                            # one stage, callable on its own
```

A console script `simcode-auto = simcode.auto.pipeline:main` is proposed for `pyproject.toml` (section 8); until
then the entry point is `python -m simcode.auto`.

## 1. Config (yaml or json)

Only `name`, `movie` and `out` are required; everything else has the DMD-SIMFLUX default (the 1809 / 0.5 nM setup).
Unknown keys are an error (typos must not pass silently). The full config, with the defaults filled in, is written to
`<out>/config_resolved.json` at the start of every run.

```yaml
name: 1809_2                       # dataset tag (file names, network-set name 'a<name>' unless netset.name is given)
movie: /mnt/data/.../simflux_5.5ms_500mW.tif   # first part; further parts (_X1.tif ...) are found by the reader
out: /mnt/data/dmdsf/simcode2_out/auto/1809_2  # work directory (large); a symlink out/auto/<name> is made
beads:                             # optional (stage b): bead z-stacks of the same optics
  stacks: [/mnt/data/.../zstack2-1.tif, ...]
  dz_um: 0.010
  sign: -1                         # stage direction
  offset_adu: 106.34               # camera offset of the bead acquisition
  holdout: 7
camera:
  type: scmos                      # scmos | emccd (emccd: excess-noise factor 2 in the variance-vs-level fit; maps not per pixel)
  gain_e_per_adu: null             # null = estimate from the movie (variance vs level of same-step frame differences)
  offset_adu: 99.3                 # null = estimate (dark frames if given, else from the movie; flagged in the checks)
  readnoise_e: null                # null = estimate
  dark: null                       # optional dark-frame movie
  readnoise_from: null             # optional read-noise map of the same sensor (npz, shifted by cross-correlation)
  map_frames: 20000                # frames used for the per-pixel maps
optics:
  pixel_um: 0.10833
  NA: 1.49
  wavelength_um: 0.70
  n_imm: 1.518
  n_med: 1.33
  tirf_depth_um: 0.082
pattern:
  orientations: 2                  # only 2 is supported by the pipeline (section 7)
  steps: 3                         # phase steps per orientation; only 3 (6-frame cycle) is supported
  frame_order: blocked             # blocked = [a0s0 a0s1 a0s2 a1s0 a1s1 a1s2]; interleaved = [a0s0 a1s0 a0s1 a1s1 ...];
                                   # or an explicit list: acquisition frame j of a cycle -> slot (orientation * 3 + step)
  cache: null                      # reordered movie (frame_order != blocked); null = <out>/movie_blocked.tif, written by stage 'order'
  pitch_px: [1.6, 2.8]             # search range of the pitch (stage d pattern, stage f pattern estimation)
hints:
  photons_window: null             # typical photons per 6-frame window (gate scenes / training brightness); null = from the spots
  density_per_um2: null            # emitters on per frame per um^2; null = from the spots
  bg_photons_px: null              # background per px per frame; null = from the spots
spots:                             # stage c
  maxframe: 30000                  # spot finding / tracks on frames [0, maxframe)
  sum: 6                           # moving-sum window of the Gaussian SMLM pass (one window per frame start)
  thr: 5.0                         # detection threshold (x robust noise of the smoothed difference image)
  roi: 11                          # fit ROI (px)
  track_roi: 11                    # sample ROI of the tracks (px; 15 at 65 nm px)
  isolation_px: 7.0
psf:                               # stage d
  init: beads                      # beads | zero | <_psf.pt of an earlier fit>
  nsamples: 8000
  iters: 600
  zmax_um: 0.2
  blur: line                       # none | line   (stage d1)
  photosel: 3                      # 0 | 3          (stage d2: frozen d1 + photoselection; needs k from the d pattern step)
  lut: false                       # true: alternating fit, per-sample x, y, z on a spline table of the current model (section 4)
  heldout: true                    # checkerboard-parity fit + test (cb0 / cb1)
  readnoise_e: 1.9                 # fixed read noise / offset of the stage-A model (as fit_insitu2); null = camera stage values
  offset_adu: null
train:                             # stage e
  netset: null                     # null = 'a<name>'; an existing set name = skip training and use it
  quick: false                     # few steps per trainer (plumbing test)
  steps: {det: 40000, hd: 18000, p2det: 40000, p2head: 24000, p2headL: 50000}
  init: {det: null, hd: hd_sc5nb.pt}
  extra: {det: '', hd: '', p2: ''}   # extra trainer flags (e.g. g13: --dens 0.0005,0.06 --tau 3,80)
  gates: {sigma_zcal: [0.9, 1.15], rms_ratio_max: 1.05}
process:                           # stage f
  netset: null                     # null = the set of stage e
  pass2: true
  crop: null                       # x0,y0,size px; null = the whole sensor minus 5 px
  chunk: 6000
  nframes: 0                       # 0 = all
  det_batch: 64
  cycle_aligned_only: false        # true for interleaved acquisitions (windows at 6c + 3 mix two cycles)
  extra: []                        # further simcode.run flags
gpus: [1, 2]
```

## 2. Stages

Run order (letters = the request's stage list): `order`, `camera` (a1), `spots` (c1), `cammaps` (a2), `tracks` (c2),
`beads` (b), `psf` (d), [`psfspline` (d2, slot)], `train` (e), `process` (f), `report` (g).

Every stage writes `<out>/<stage>/summary.json` LAST and atomically (`simcode.fileio.work.atomic_json`); that file is the
done marker. A finished stage is skipped unless `--redo <stage>` (which also invalidates the stages after it).
Long stages are resumable inside: `spots` and `process` work in chunks through `simcode.fileio.work.Units`, `train`
per trainer (a trainer with `final.pt` is skipped). Each summary holds the settings used, the timing, the main numbers
and a list `checks` of `{name, value, range, ok}`; a failed check stops the run (unless `--force`) and is listed in
the report. Process title: `simcode:auto-<stage>` (at most 20 characters).

| stage | what | output (in `<out>/<stage>/`) | checks |
|---|---|---|---|
| `order` | only for `frame_order != blocked`: writes the movie with every complete 6-frame cycle permuted to the blocked order (as `analysis/nanoruler/reorder_any.py`), or verifies an existing `pattern.cache` (frame count, first-cycle checksum against the permutation) | `movie_blocked.tif` (on the data disk) | frame count = 6 x cycles; per-slot mean level |
| `camera` (a1) | gain, read noise (and offset, if not given) from the variance-vs-level fit of same-step frame differences (frames 6 apart; trimmed variance, robust to blinking; `insitu/camera_gain.py` generalised to any level range); dark frames, if given, give offset and read noise directly | `camera.json` | gain within 20 % of the config value if one is given; fit r^2; number of level bins >= 5 (else gain must be given) |
| `spots` (c1) | pattern-free Gaussian SMLM on `sum`-frame moving sums (window start every frame; as the SIMCODE-pipeline smlm table): smoothed difference to the per-pixel temporal median, local maxima > thr x noise, Poisson LM fit of x, y, I, bg, sigma (`analysis/nanoruler/quick_smlm.py` as a function); good-fit filter (chi2, snr, sigma range from the optics) | `spots.hdf5` (Picasso-style `locs`), `spots_good.hdf5` | spots per frame; median sigma vs the diffraction estimate 0.21 lambda / NA (+-25 %); median photons |
| `cammaps` (a2) | per-pixel offset deviation and read-noise maps (`python -m simcode.insitu.camera_maps`, masked by `spots_good.hdf5`) | `camera_maps.npz` | corr(offset_dev, loc density) < 0.05; read-noise median |
| `tracks` (c2) | isolated emitter tracks from `spots_good.hdf5` (`python -m simcode.insitu.extract_tracks`: link <= 0.8 px with 1-frame gaps, >= 6 frames, isolation radius, cut into T = 12 frame samples) | `tracks.npz` | number of samples >= 2 x psf.nsamples |
| `beads` (b) | optional bead z-stack fit (`python -m simcode.psf.fit_beads` with the optics of the config); without beads stage d starts from zero aberrations | `beads_psf.pt/json` | held-out stack NLL; |Z| < 0.15 waves |
| `psf` (d) | d1: Zernike 5-21 + focus + line blur (variable projection: per-frame I and per-(sample, angle) bg exact, x, y, z per sample + PSF by Adam; `simcode/auto/psffit.py`, the same algorithm and defaults as `insitu/fit_insitu2.py` but with the optics / camera from the config); intensities of the fitted samples; pattern from the track intensities (k by a DFT over the config pitch range, per-step phases, depth, relint); d2: d1 frozen + photoselection 3 (k directions from the pattern step); held-out: the same two fits on checkerboard parity cb0, tested on cb1 (PSF frozen, x, y, z refit); calib json per axis and averaged (`analysis/psf3d/make_calib.py` format) | `calib.json`, `calib_avg.json`, `camera_maps.npz` copy, `fit_*`, `heldout.json`, `psf_images.npz` | held-out NLL per sample: d1 <= base, photosel <= d1 (paired z); resid var in [0.85, 1.1]; |Z| < 0.15 waves; focus inside [0, zmax]; z median inside (0, zmax) |
| `psfspline` (d2, slot, not implemented) | non-parametric spline refinement (section 5) | `psf_spline.npz` + `calib_spline.json` | held-out acceptance |
| `train` (e) | network set with the PSF-matched recipe (section 6); registers it in `<out>/train/netset.json` | `models/*.pt`, calibs, `gates.txt` | gate acceptance |
| `process` (f) | `simcode.run` on the movie with the set (pass 1, pattern, solve, pass 2, table, selections), through the in-memory overrides of section 7 | `run/locs_p*.npz`, `run/summary.json` | pattern pitch inside the config range; rows per frame; median sigma |
| `report` (g) | `report.html` + `report.json`: settings, counts per stage, camera maps, PSF (images, residual moments per angle, held-out table), pattern (k, phases, depth vs the in-situ estimate), drift (DME on the table, if available), FRC (`simcode.evalproto.frc`), gate table, all checks | `report.html` | none |

## 3. Spot finding without an external SMLM table (stage c)

The in-situ recipe up to now used the SIMCODE-pipeline `smlm.hdf5` of each movie (Picasso, 6-frame moving sums) to
find isolated emitters and to mask emitters out of the camera maps. Stage c replaces it by a built-in Gaussian SMLM on
the same kind of input (6-frame moving sums, one window per frame start), so that tracks and masks are made the same way.
A Gaussian fit is enough for both uses: the track extraction only needs positions to 0.8 px and isolation, and the
camera-map mask only needs positions to ~1 px. The pass-1 detector of an existing network set is the other option
(PSF-agnostic within one setup, but it depends on a set trained for the pixel size and PSF width); it is not used by
default because a new instrument has no set yet.

## 4. Stage d speed-up: alternating fit on a spline table (`psf.lut: true`)

Direct fit (`fit_insitu2`): each Adam step evaluates the vectorial model for every sample (8000 samples x 2 angles with
photoselection), with gradients to x, y, z and to the PSF parameters.

Alternating fit: per outer iteration

1. the knots of a Catmull-Rom 3D spline table of the CURRENT vectorial model are built (`simcode/psf/lut.py` geometry:
   the pixel-integrated PSF sampled at S x S sub-pixel shifts and z steps hz over the fitted z range, per pattern angle
   when the model has per-angle terms), with autograd kept to the PSF parameters;
2. the per-sample x, y, z steps (and the varpro I / bg) run on the spline (conv2d evaluation, value and gradients), which
   costs a fraction of a vectorial evaluation;
3. the PSF parameters get the exact gradient of the spline-approximated loss over ALL samples by one backward pass
   through the table construction (d loss / d knots from the spline evaluation, then the vector-Jacobian product of the
   knots with respect to Zernikes, focus, blur, gamma, zeta). The cost of that step does not grow with the number of samples.

Acceptance: fitted parameters equal to the direct fit within the run-to-run scatter of the direct fit (Zernikes rms
< 0.003 waves, focus < 3 nm, blur < 0.01 px^2, gamma / zeta < 0.005), the same held-out NLL, and the speed-up measured on
the same GPU. The final intensities / residuals are always computed with the vectorial model.

## 5. Stage d2 slot: non-parametric spline PSF (not implemented here; explored in a separate session)

Interface:
- input: the stage-d calib json (vectorial PSF, per-angle terms) and `tracks.npz`;
- output: `psf_spline.npz` = spline knots per pattern angle (same geometry as `simcode.psf.lut.PSFLUT`: S, hz, z0, u0,
  knot array [nz, nv, nu]) plus `calib_spline.json` = the stage-d calib with a key `psf_spline` (relative path);
- acceptance: held-out (cb0 train / cb1 test) NLL per sample improves with a paired z > 5 and the chi2 per pixel
  improves; CRLB change reported; training is possible because the simulators render through tabulated PSFs
  (`sim.simulator.Simulator.lut`, `lut_angle`), which can be filled from the knots (proposal P5).

## 6. Stage e: network set (recipe of `runs/psfmatch/chain2.sh` + `eval.sh`)

Trainers (unchanged entry points, run through `simcode/auto/simwrap.py`, which patches `simcode.hd.realsim.PRESETS` in
memory with the dataset's pattern k, step spacing, camera maps, background and brightness, as `analysis/nanoruler/nrwrap.py`):

| step | command (all with `--calib <out>/psf/calib_avg.json` + `train.extra`) | steps (full / quick) |
|---|---|---|
| det | `simcode.det.up2 train --up 1 --batch 16 --size 64 --lr 1e-3 --w_off 1.0 --nnw 1 --em_photons 40,2500 --seed 0` | 40000 / 30 |
| hd | `simcode.hd.sc4 train --arch sc4n --init hd_sc5nb.pt --det <det> --lr 1.5e-4 --ch 96 --paper 1 --tau 15,80 --pair_frac 0.25 --dens 0.0005,0.04 --med 40,2500 --nnw 1 --seed 1 --w_mr 2.0 --w_slope 50.0 --bf16 1 --pct 0.1 --bright_tail 0.1,2000,20000 --hold <heldout5.pt>` | 18000 / 30 |
| p2det | `simcode.p2.det2 train --init_n1 <det> --lr 3e-4 --pct 0.05 --w_off 1.0 --seed 11` | 40000 / 30 |
| p2head | `simcode.p2.head2 train --det <p2det ckpt 25k> --init <hd> --lr 1.5e-4 --pct 0.1 --w_mr 2.0 --w_slope 50.0 --seed 21 --no_illum 0` (+ `--pgen <json>` when the generic pass-2 option is adopted) | 24000 / 30 |
| p2headL | `simcode.p2.head2 train --det <p2det> --init <p2head> --lr 1e-4 --pct 0.05 --seed 31 --no_illum 0 --pert_model real --hold_v1 15` | 50000 / 30 |
| gates | `simcode.hd.gate` scenes (iso, pair1, pair2, rs*qm) with the dataset PSF (`SIMCODE2_GATE_CALIB`), `simcode.p2.dump` of pass 1 and pass 2 | seeds 202, 301-303 / 202 only, small ntarget |
| calibs | `analysis/psfmatch/fit_cal_set.py` (sim sigma table on seed 202; the `_real` multiplier copied from m1809's) | same |
| accept | sigma zcal on the held-out seeds in `train.gates.sigma_zcal`; pass-2 RMS / CRLB not worse than pass 1 on rs*qm | same |

The set is registered in `<out>/train/netset.json` (format of `simcode/netsets.json`, absolute paths); stage f reads it
through an in-memory addition to `config.netsets()` (no edit of `netsets.json`). The selection-aware sigma rescale (`sig_u`,
fitted on a simulated movie with ground truth, `analysis/psf3d_nets/sigcal.py`) is not in the quick path; without it stage f
runs with `--sig_u ''`.

Cost of a full stage e on an idle RTX 3090 (measured on the m1809 / m25 / g13 chains): det 0.65 + hd 0.65 + p2det 1.1 +
p2head 1.9 + p2headL 2.5 = 6.8 GPU-h, gates + dumps + calibs ~1.5 GPU-h, sigma-rescale sim movie ~0.5 GPU-h.

## 7. Instrument assumptions that block a new setup, and how the config resolves them

| # | assumption (where) | resolution |
|---|---|---|
| 1 | 6-frame cycle, blocked order, angle = frame % 6 // 3 (`pipeline/pattern_est.py`, `pipeline/common.pattern_frames`, `sc_front*.py` (ws + t) % 6 // 3, `insitu/*`, `psf/vectorial` per-angle PSF, `sim/*`) | `pattern.frame_order`: stage `order` writes a blocked-order cache; every later stage reads the cache. For interleaved data the windows starting at 6c + 3 mix two cycles: `process.cycle_aligned_only` keeps windows with frame % 6 == 0 in the selections. 4-step or 3-orientation schemes: refused by the config check (needs P1) |
| 2 | 2 orientations classified by |kx| > |ky| (`core.angle_index`, `p2/illum.angle_of`, `hd/msolve`) | config check: the two fitted k (stage d pattern) must lie in different 45-degree sectors; otherwise stop with an explanation (P2) |
| 3 | pitch search 1.6-2.8 px (`pattern_est.estimate_k(pitch_px=...)`) | stage f sets the default of `estimate_k` in memory to `pattern.pitch_px` (as `analysis/nanoruler/ev_run.py`; P3) |
| 4 | strict CROP fixed to (5, 5, 110) (`strict.py`), `--crop` default 5,5,110 | stage f passes `--crop` = `process.crop` (default: the sensor minus 5 px) and sets `strict.CROP` and the bound defaults of `tile_of` / `bin20` in memory (P4) |
| 5 | detector batch 64 windows (`run_uncalibrated --det_batch`, not exposed by `run.py`); 400 x 400 px needs > 12 GB | `process.det_batch` is set as the argparse default in memory |
| 6 | optics / camera constants in `insitu/fit_insitu2.py` (wavelength 0.70, pixel 0.10833, offset 99.3, rn 1.9), `camera_gain.py` level bins 101-139 ADU | stage d uses `simcode/auto/psffit.py` (same algorithm, constants from the config); stage a its own generalised gain fit |
| 7 | `hd/realsim.PRESETS` (k, step spacing, camera maps, bg, brightness of the DMD setup), `PX_NM` 108.33 in realsim / plib / sigcal | stage e: in-memory preset patch (`simcode/auto/simwrap.py`) from the stage-d pattern and the stage-c statistics; `PX_NM` likewise patched (P6) |
| 8 | gate photon settings hard-coded in `hd/gate.py` (rs05qm median 1260 ph) | reported; P7 |
| 9 | strict-product density anchors fitted on the DMD movies (`strict.Q_DENSITY`) | not changed; the main selection (global sigma threshold) does not use them |
| 10 | pass-2 networks trained on the DMD pattern distribution (equal steps 2.09-2.12 rad, pitch 2.07 px) | stage e trains pass 2 with the dataset's pattern; the generic pass-2 option (`--pgen`, other worker) is used when it is adopted |

## 8. Proposed changes in files this work does not edit

- P1 (`pipeline/common.py`, `pattern_est.py`, `sc_front*.py`, `insitu/*`, `sim/*`): replace `frame % 6 // 3` by a
  per-dataset slot table (`cycle`, `angle_of_slot`), stored in the calib json (`pattern.frame_slots`), default the blocked
  6-frame cycle. Removes the reorder cache.
- P2 (`core.angle_index`, `p2/illum.angle_of`, `hd/msolve`): take the orientation index from the pattern (per step) instead of |kx| > |ky|.
- P3 (`pattern_est.estimate_k`, `run.py`, `run_uncalibrated.py`): a `--pitch_px lo,hi` flag (default 1.6,2.8).
- P4 (`strict.py`): `build_table(..., crop=)` already takes the crop; `tile_of` / `bin20` should default to it instead of
  the module constant `CROP`; `run.py --crop` default = the sensor minus 5 px.
- P5 (`sim/simulator.py`, `det/psfpert.build_lut`, `hd/realsim.py`): optional calib key `psf_spline` (knots per angle): the
  Simulator LUTs are filled from the spline, the jitter bank adds the spline-minus-vectorial residual to each LUT.
- P6 (`hd/realsim.py`, `analysis/calib/plib.py`, `analysis/psf3d_nets/sigcal.py`): pixel size and preset values from the
  calib json / a preset json argument (`--preset`), instead of `PRESETS` / `PX_NM`.
- P7 (`hd/gate.py`): scene photon / background settings from the preset.
- P8 (`run.py`): expose `--det_batch`; accept a network set given as a json file (`--netset path.json`).
- P9 (`pyproject.toml`): `[project.scripts] simcode-auto = "simcode.auto.pipeline:main"`.

## 9. Validation (filled in after the runs)

See `out/auto/STATUS.md`.
