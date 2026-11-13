# Automatic per-dataset pipeline (`simcode.auto`): design

Status: design 2026-11-13; implementation in `simcode/auto/` (stages a, b, c, d, f, g; e as a wrapper with `--quick`).
Validation numbers: section 9 (filled in after the runs).

Goal: one config file per dataset, then one command that

1. calibrates the camera from the movie,
2. fits the bead z-stack (optional),
3. localizes the movie with a pattern-free spot finder to collect isolated emitter tracks (no external SMLM table),
4. fits the in-situ PSF on those tracks (variable projection, line blur, photoselection) with a held-out check,
5. trains the PSF-matched model (recipe of `runs/psfmatch/chain2.sh`),
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
name: 1809_2                       # dataset tag (file names, model name 'a<name>' unless train.model is given)
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
  thr: 3.5                         # detection threshold (x robust noise of the smoothed difference image); low = more
                                   # neighbours seen by the isolation test (1809_2: resid var 0.996 at 5, 0.955 at 3.5)
  roi: 11                          # fit ROI (px)
  track_roi: 11                    # sample ROI of the tracks (px; 15 at 65 nm px)
  isolation_px: 7.0
  tracks_from: all                 # spots used for linking / isolation / camera-map mask: all detections | good fits only
psf:                               # stage d
  init: [zero, f0.1, f0.2, f0.3, beads]   # multi-start of the base fit, lowest NLL wins: zero | f<focus um> | beads | <_psf.pt>
  nsamples: 8000
  iters: 600
  zmax_um: 0.2
  blur: line                       # none | line   (stage d1)
  photosel: 3                      # 0 | 3          (stage d2: frozen d1 + photoselection; needs k from the d pattern step)
  lut: false                       # true: per-sample x, y, z on a spline table of the current model (section 4)
  lut_sub: 8                       # spline knots per pixel
  lut_polish: 50                   # last iterations with the direct vectorial model
  heldout: true                    # checkerboard-parity fit + test (cb0 / cb1)
  readnoise_e: 1.9                 # fixed read noise / offset of the stage-A model (as fit_insitu2); null = camera stage values
  offset_adu: null
train:                             # stage e
  model: null                      # null = 'a<name>'; an existing model name = skip training and use it (key netset before 2026-11-14)
  quick: false                     # few steps per trainer (plumbing test)
  steps: {det: 40000, hd: 18000, p2det: 40000, p2head: 24000, p2headL: 50000}
  init: {det: null, hd: hd_sc5nb.pt}
  extra: {det: '', hd: '', p2: ''}   # extra trainer flags (e.g. g13: --dens 0.0005,0.06 --tau 3,80)
  gates: {sigma_zcal: [0.9, 1.15], rms_ratio_max: 1.05}
process:                           # stage f
  model: null                      # null = the model of stage e; a name of simcode/models.json; or a model json file
  calib: null                      # null = the stage-d calib.json
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
`beads` (b), `psf` (d), [`psfrefine` (d2, slot)], `train` (e), `process` (f), `report` (g).

Every stage writes `<out>/<stage>/summary.json` LAST and atomically (`simcode.fileio.work.atomic_json`); that file is the
done marker. A finished stage is skipped unless `--redo <stage>` (which also invalidates the stages after it).
Long stages are resumable inside: `spots` and `process` work in chunks through `simcode.fileio.work.Units`, `train`
per trainer (a trainer with `final.pt` is skipped). The chunk directories are keyed by a hash of their inputs
(`spots/chunks_<key>`: movie, gain, offset, spot settings; `process/run_<key>`: the simcode.run arguments, the calibration
file content, the model and the overrides), so a redo with other settings never reuses stale chunks; stage d reuses a
fit only if it is newer than the tracks and was made from the same start. Each summary holds the settings used, the timing, the main numbers
and a list `checks` of `{name, value, range, ok}`; a failed check stops the run (unless `--force`) and is listed in
the report. Process title: `simcode:auto-<stage>` (at most 20 characters).

| stage | what | output (in `<out>/<stage>/`) | checks |
|---|---|---|---|
| `order` | only for `frame_order != blocked`: writes the movie with every complete 6-frame cycle permuted to the blocked order (as `analysis/nanoruler/reorder_any.py`), or verifies an existing `pattern.cache` (frame count, first-cycle checksum against the permutation) | `movie_blocked.tif` (on the data disk) | frame count = 6 x cycles; per-slot mean level |
| `camera` (a1) | gain, read noise (and offset, if not given) from the variance-vs-level fit of same-step frame differences (frames 6 apart; trimmed variance, robust to blinking; `insitu/camera_gain.py` generalised: quantile bins of the per-pixel MEAN level (integer-valued movies), the 10 % noisiest pixels excluded); dark frames, if given, give offset and read noise directly | `camera.json` | gain within 20 % of the config value if one is given; fit r^2; number of level bins >= 5 (else gain must be given) |
| `spots` (c1) | pattern-free Gaussian SMLM on `sum`-frame moving sums (window start every frame; as the SIMCODE-pipeline smlm table): smoothed difference to the per-pixel temporal median, local maxima > thr x noise, Poisson LM fit of x, y, I, bg, sigma (`analysis/nanoruler/quick_smlm.py` as a function); good-fit filter (chi2, snr, sigma range from the optics) | `spots.hdf5` (Picasso-style `locs`), `spots_good.hdf5` | spots per frame; median sigma vs the diffraction estimate 0.21 lambda / NA (+-25 %); median photons |
| `cammaps` (a2) | per-pixel offset deviation and read-noise maps (`python -m simcode.insitu.camera_maps`, masked by `spots_good.hdf5`) | `camera_maps.npz` | corr(offset_dev, loc density) < 0.05; read-noise median |
| `tracks` (c2) | isolated emitter tracks from `spots_good.hdf5` (`python -m simcode.insitu.extract_tracks`: link <= 0.8 px with 1-frame gaps, >= 6 frames, isolation radius, cut into T = 12 frame samples) | `tracks.npz` | number of samples >= 2 x psf.nsamples |
| `beads` (b) | optional bead z-stack fit (`python -m simcode.psf.fit_beads` with the optics of the config); without beads stage d starts from zero aberrations | `beads_psf.pt/json` | held-out stack NLL; |Z| < 0.15 waves |
| `psf` (d) | base fit from every start of `psf.init` (zero, focus-staggered f0.1-f0.3, beads), lowest NLL wins (focus / z gauge, section 9); d1: Zernike 5-21 + focus + line blur (variable projection: per-frame I and per-(sample, angle) bg exact, x, y, z per sample + PSF by Adam; `simcode/auto/psffit.py`, the same algorithm and defaults as `insitu/fit_insitu2.py` but with the optics / camera from the config); intensities of the fitted samples; pattern from the track intensities (k by a DFT over the config pitch range, per-step phases, depth, relint); d2: d1 frozen + photoselection 3 (k directions from the pattern step); held-out: the same two fits on checkerboard parity cb0, tested on cb1 (PSF frozen, x, y, z refit); calib json per axis and averaged (`analysis/psf3d/make_calib.py` format) | `calib.json`, `calib_avg.json`, `camera_maps.npz` copy, `fit_*`, `heldout.json`, `psf_images.npz` | held-out NLL per sample: d1 <= base, photosel <= d1 (paired z); resid var in [0.85, 1.1]; |Z| < 0.15 waves; focus inside [0, zmax]; z median inside (0, zmax) |
| `psfrefine` (d2, slot, not implemented) | refined PSF (free pupil map through the spline-gradient route, then the pattern term; section 5) | refined PSF file + `calib_refined.json` | held-out acceptance |
| `train` (e) | model with the PSF-matched recipe (section 6); registers it in `<out>/train/model.json` | `models/*.pt`, calibs, `gates.txt` | gate acceptance |
| `process` (f) | `simcode.run` on the movie with the set (pass 1, pattern, solve, pass 2, table, selections), through the in-memory overrides of section 7 | `run_<key>/locs_p*.npz`, `run_<key>/summary.json` | pattern pitch inside the config range; rows per frame; median sigma |
| `report` (g) | `report.html` + `report.json`: settings, counts per stage, camera maps, PSF (images, residual moments per angle, held-out table), pattern (k, phases, depth vs the in-situ estimate), drift (DME on the table, if available), FRC (`simcode.evalproto.frc`), gate table, all checks | `report.html` | none |

## 3. Spot finding without an external SMLM table (stage c)

The in-situ recipe up to now used the SIMCODE-pipeline `smlm.hdf5` of each movie to find isolated emitters and to mask
emitters out of the camera maps. Stage c replaces it by a built-in Gaussian SMLM on 6-frame moving sums (one window per frame
start; `simcode/auto/spots.py` = `analysis/nanoruler/quick_smlm.py` as a function, chunked through `Units`). A Gaussian fit
is enough for both uses: the track extraction needs positions to 0.8 px and the isolation test, the camera-map mask
positions to ~1 px. The pass-1 detector of an existing model is the other option; it is not the default because a new
instrument has no set yet.
- All detections (`spots.tracks_from: all`) are used for linking, the isolation test and the mask: a good-fit filter breaks
  tracks and hides neighbours from the isolation test (1809_2: 16 361 samples with the filter, 20 893 without).
- The Picasso table of the earlier recipe is not the same kind of list (median photons 317 vs 637 per window, background
  2x, 42.6 vs 26.2 detections per window; ~47 % of the stage-c spots match it within 1 px at any frame offset), so the
  track sets differ by construction (section 9).
- The samples are cut at round(mean of the track's fitted Gaussian centres), not at the smoothed peak (section 5 note).

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

Implementation (`simcode/auto/psffit.py`, `fit(..., lut=True)`): knots S = 8 per pixel, hz = 10 nm (spline error 4e-5
of the peak, rms 5e-6), |x|, |y| <= 2 px; the spline is evaluated separably (only the 4 x 4P x 4P knots the P x P pixels use
are gathered, then contracted along z, y, x: equal to `PSFLUT.evaluate` to 1.5e-8 and independent of S); the 100 warm-up
iterations (positions only) use one table; then one table per iteration with the graph to the PSF parameters; the last
`lut_polish` = 50 iterations are direct vectorial steps (the focus / z direction is nearly flat and the spline optimum is
offset along it by a few nm). Final residuals and intensities always use the vectorial model.

Measured (1809_2 reference tracks, 8000 samples, 600 iterations, idle RTX 3090): per iteration the vectorial PSF with
gradients costs 0.056 s, the spline 0.032 s with knot gradients and 0.022 s with position gradients only; varpro + NLL are
~0.06 s either way, so the gain is bounded:
| fit | direct | spline + polish | parameters vs direct |
|---|---|---|---|
| Zernike + focus + line blur (1 PSF per sample) | 48 s | 39 s (1.24x) | Zernikes rms 0.0003 waves (max 0.0007), focus -4.4 nm, blur -0.0016 px^2, NLL/px +3.5e-6 |
| bl frozen + photoselection 3 (2 PSFs per sample) | 111 s | 71 s (1.56x) | gamma -1.9e-5, zeta +5.4e-5, NLL/px +2e-7 |
The focus difference of the first fit is within the direct fit's own non-convergence (its focus still moves by ~1 nm per
100 iterations at the end). Opt-in (`psf.lut: true`); the held-out refits of stage d always use the spline (positions only).

## 5. Stage d2 slot: refined PSF beyond Zernike + blur + photoselection (not implemented here; separate session)

Interface (slot `psfrefine`, between `psf` and `train`):
- input: the stage-d calib json (vectorial PSF, per-angle terms), `tracks.npz`, the stage-d pattern (k, phases, relint);
- output: a refined PSF file usable by the simulators and the pipeline, plus `calib_refined.json` = the stage-d calib with a
  key naming it (relative path);
- acceptance: held-out (cb0 train / cb1 test) NLL per sample improves clearly (paired z > 5) and chi2 per pixel improves;
  CRLB change reported; networks can be trained on it because the simulators render through tabulated PSFs
  (`sim.simulator.Simulator.lut`, `lut_angle`), which can be filled from the refined model (proposal P5).

Recommended method (results of the user's spline-PSF study, simulation, one seed; code in the user's torch-spline-3d
branch `insitu-study`: `study/insitu/run_pattern.py`, `fit.py` (`fit_pupil_insitu`), `docs/insitu_spline_psf.md` 3.3-3.5):
- refine a free PUPIL map (phase + log-amplitude pixels on top of Zernike 5-21), fitted through the spline-gradient route of
  section 4 (spline table of the current model, gradients to the pupil through the table construction); NOT free spline
  knots (knots keep a z-dependent lateral tilt and a z relabelling);
- with the SIM pattern term in the localization step (each emitter's x, y, z shared between the PSF term and
  I_t = N relint_t (1 + m cos(k_t . r - phi_t))), the pupil fit reaches < 1.3 nm y / z error over 700 nm z;
- order matters: first refine the Zernikes on the summed frames without the pattern, then switch the pattern on (a pattern
  fit started from a poor PSF locks emitters onto the wrong fringe). The stage-d output (Zernikes + blur + photoselection,
  pattern from the track intensities) is that first step.

Note for stages c / d from the same study: a z-dependent lateral tilt of ~30 nm appeared only on patches cut from a movie by
spot detection (crop around the smoothed peak), not on clean patches; the cause (centring or selection) is not established.
Here the track samples are cut at round(mean of the track's fitted Gaussian centres) (`insitu/extract_tracks.py`), i.e. a
refined-fit centre, not the smoothed peak; the validation checks the slope of (vectorial-fit position - Gaussian centre)
against the fitted z (section 9).

## 6. Stage e: model (recipe of `runs/psfmatch/chain2.sh` + `eval.sh`)

Trainers (unchanged entry points, run through `simcode/auto/runwrap.py`, which patches `simcode.hd.realsim.PRESETS` in
memory with the dataset's pattern k, step spacing, camera maps, background and brightness, as `analysis/nanoruler/nrwrap.py`):

Quick mode (`--quick`): `train.quick_steps` (30) steps per trainer, the detector starts from the template set's detector
(det_m1809; from scratch it detects nothing after 30 steps), gate scenes iso + rs05qm with seeds 202 / 301 and 400 targets;
if a calibration fit has no matched targets the template calibrations are copied (flagged `calib_fallback`); the set is named
`q<name>` and is a plumbing artefact only.

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

The model is registered in `<out>/train/model.json` (format of `simcode/models.json`, absolute paths; `netset.json` before
2026-11-14, still read); stage f reads it through an in-memory addition to `config.model_sets()` (no edit of `models.json`). The selection-aware sigma rescale (`sig_u`,
fitted on a simulated movie with ground truth, `analysis/psf3d_nets/sigcal.py`) is not in the quick path; without it stage f
runs without it (the model's `sig_u` is empty).

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
| 7 | `hd/realsim.PRESETS` (k, step spacing, camera maps, bg, brightness of the DMD setup), `PX_NM` 108.33 in realsim / plib / sigcal | stage e: in-memory preset patch (`simcode/auto/runwrap.py`) from the stage-d pattern and the stage-c statistics; `PX_NM` likewise patched (P6) |
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
- P5 (`sim/simulator.py`, `det/psfpert.build_lut`, `hd/realsim.py`, `pipeline/common.load_calib`): optional calib key for the
  refined PSF of stage d2 (e.g. a pupil map): the Simulator / pipeline LUTs are built from it, the jitter bank applies its
  Zernike jitter on top of it.
- P6 (`hd/realsim.py`, `analysis/calib/plib.py`, `analysis/psf3d_nets/sigcal.py`): pixel size and preset values from the
  calib json / a preset json argument (`--preset`), instead of `PRESETS` / `PX_NM`.
- P7 (`hd/gate.py`): scene photon / background settings from the preset.
- P8 (`run.py`): expose `--det_batch`; accept a model given as a json file (`--model path.json`).
- P9 (`pyproject.toml`): `[project.scripts] simcode-auto = "simcode.auto.pipeline:main"`.

## 9. Validation (2026-11-13)

Scripts and logs: `out/auto/val/` (`t_*.py`), `analysis/auto/cmp_calib.py`; work directories `out/auto/<name>` ->
`/mnt/data/dmdsf/simcode2_out/auto/<name>`.

### 9.1 Stage d code against fit_insitu2
- `psffit.fit` on the reference tracks with the reference start: bit-identical to `out/psf3d/final/m1809_bl` (Zernikes,
  focus, blur, residual variance) and, with photoselection, to `m1809_blps3` (gamma / zeta differ by 3e-8).
- Spline-table mode: section 4 (1.24x / 1.56x, parameters within 0.0007 waves, 4 nm focus, 5e-5 gamma / zeta).

### 9.2 1809_2 (DMD-SIMFLUX), stages a-d against `calib_1809_2_psf3d.json` / `camera_maps_1809_2.npz`
| item | result |
|---|---|
| camera gain | config 0.56; measured 0.560 (the generalised estimator; `camera_gain.py` returns a negative slope on this movie: 7 ADU of background levels, integer medians); 0.5 nM movie 0.570 vs `camera_gain.py` 0.552 |
| camera maps | offset map corr 0.977, rms difference 0.33 ADU (reference sd 1.55); read-noise map (both from the 0.5 nM map, shifted) corr 0.99977, rms 0.02 e |
| spots (thr 3.5) | 34.8 detections per 6-frame window (Picasso table 42.6), median Gaussian sigma 1.28 px |
| tracks | 22 438 samples from 9 632 isolated tracks (reference, Picasso table: 50 422 / 33 971); thr 5: 20 893, good-fit spots only: 16 361 |
| beads (stage b) | NLL -8487532.94 vs the shipped fit -8487532.89 |
| start | bead Zernikes are in another basin (NLL/px +5e-4 on the reference tracks); the reference started from the 0.5 nM base fit; multi-start picks the focus 0.3 um basin (lowest base NLL on every track set, including the reference tracks: 2.53389 vs 2.53400) |
| PSF vs reference | Zernike rms 0.017 waves (max Noll 6, 0.058), focus 302 vs 148 nm, emitter z median 0.10 vs 0.077 um, blur 0.061 vs 0.204 px^2 (stage d on the reference tracks, same basin: 0.002); PSF images at matched emitter z: rms 1.4 % of the peak (1.4 % also for stage d on the reference tracks), at equal z 2.6 % |
| photoselection | gamma 0.066 vs 0.094, zeta 0.36 vs 0.36; held-out ps3 - bl: z -7.2 (reference tracks -6.1) |
| held-out line blur | bl - base: z -1.6 (thr 3.5), +6.9 on the reference tracks in this basin (blur not supported there) |
| residual variance | 0.955 (thr 3.5), 0.966 (thr 4), 0.996 (thr 5); reference tracks 0.928: fewer detections leave dim neighbours in the "isolated" samples |
| CRLB (1000 ph, bg 6, matched z) | x +2.2 %, y +3.4 % vs the reference calibration (reference tracks: -1.7 %, +0.6 %) |
| crop centre vs z | slope of (vectorial-fit position - Gaussian crop centre) vs z: x -0.9, y +1.6 nm per 100 nm z (reference tracks -0.3 / +2.3): no z-dependent crop offset |
Conclusion: the remaining difference to the reference calibration is the near-degenerate focus / emitter-z / spherical /
blur direction of in-situ 2D data (both calibrations fit the data about equally; the basin depends on the start), not the
spot finder. Downstream on the same 6000 frames (stage f, m1809 set, FRC of 300-frame block halves after DME): main
selection 18.84 nm / 126 618 rows with the final auto calibration (thr 3.5), 18.27 nm / 127 455 rows with the thr-5 one,
18.75 nm / 118 490 rows with the reference calibration; u <= 0.45 17.21 nm for the thr-5 and the reference calibration.

### 9.3 Stage e (quick) and stage f
- Stage e `--quick` on 1809_2: all five trainers (30 steps), gate scenes, dumps, calibration fits (fallback used: no
  targets), acceptance (zcal 0.88-1.07 with the 30-step nets, gates FAIL as expected), `netset.json`; 163 s. The quick set
  runs through stage f (3000 frames: pass 1, pattern k found, pass 2, tables).
- Stage f on 1809_2 frames 0-6000 (m1809, reference calibration): all 6 output npz bit-identical to `python -m simcode.run`
  with the same arguments (the wrapper applies no override for the DMD defaults).

### 9.4 Nanoruler (sim4_1, 80 nm; Pockels / mechanical steps, interleaved order, 65 nm px)
| item | result |
|---|---|
| order | the existing blocked cache verified (30 000 frames, permutation checked on 4 cycles) |
| camera | gain 0.45 / offset 100 assumed (config); measured gain 0.66 with r^2 flagged (not measurable: background ~3 ADU above the offset, as found before) |
| spots (thr 5) | 13.7 good fits per window (reference quick SMLM 13.65), photons p5 / 50 / 95 406 / 1157 / 2065 (409 / 1158 / 2059), sigma 1.848 px (1.845) |
| tracks | 31 859 samples / 14 247 isolated tracks (reference 33 794 / 11 082) |
| camera maps | offset corr 0.998 (rms 0.07 ADU), read noise corr 0.9998 (rms 0.009 e) |
| pattern (track intensities) | k (-0.00236, 1.85712), (1.86051, 0.00184) rad/px vs reference (-0.0024, 1.8572), (1.8605, 0.0019); pitch 3.383 / 3.377 px; phases 0.73, -1.33, 2.80 / 2.37, 0.47, -1.62 vs 0.82, -1.38, 2.81 / 2.28, 0.42, -1.50 |
| PSF | Zernike rms 0.00018 waves (max 0.0004), focus 69 vs 72 nm, blur ~0 vs 0.012 px^2, gamma 0.154 vs 0.151, zeta 0.385 vs 0.384, dipole tensors within 0.002, images rms 0.08 % of the peak, CRLB within 0.2 %; held-out photoselection z -20 (reference -23.6) |
| stage f | chunk 0 (1500 frames, nr80 set, reference calibration): `pass1_0.npz` bit-identical to `out/nanoruler/runs/real_p1` (made with `analysis/nanoruler/ev_run.py`); with the auto calibration (3000 frames): pitch 3.383 / 3.377 px found, 108 447 rows, 54 459 cycle-aligned |

### 9.5 Cost of a full run (GPU-hours on an idle RTX 3090; measured here on shared GPUs unless noted)
| stage | 1809_2 (120 x 120 px, 120k frames) | nanoruler (400 x 400 px, 30k frames) |
|---|---|---|
| order | 0 (blocked) | ~45 min I/O once (cache existed) |
| camera | 30 s CPU | 20 s CPU |
| spots (30k frames) | 3 min GPU | 6 min GPU |
| cammaps + tracks | 1 min CPU | 2 min CPU |
| beads | ~10 min GPU (shared) | - |
| psf (5 starts, bl, ps3, held-out) | ~9 min GPU (~6 min with `psf.lut`) | ~25 min on a shared GPU |
| train (full recipe) | 6.8 GPU-h training + ~1.5 GPU-h gates / dumps / calibrations (+ ~0.5 GPU-h for a sig_u sim movie, not wrapped) | same (+ wall time on shared GPUs 2x) |
| process | ~1.8 GPU-h (pass 1 + 2, full movie) | ~0.2 GPU-h pass 1 (pass 2 needs the generic pass-2 nets) |
| report (DME drift + FRC) | ~2 min CPU / GPU | ~1.5 min |
| total | ~10.5 GPU-h | ~9 GPU-h |
