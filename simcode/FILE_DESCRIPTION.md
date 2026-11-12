# SIMCODE v2: what each file does

One line per file of the `simcode` package, grouped by folder. Written from the module docstrings (2026-11-11); the
docstring at the top of each file has the details. Method description: `docs/METHOD.md`. Evaluation: `docs/EVALUATION.md`.

## How the pieces fit

A movie is processed in 6-frame windows (2 pattern angles × 3 phase steps), one window every 3 frames:

1. **Pass 1** (pattern-free): a detector finds candidates; a per-candidate head network gives per-frame intensities with a
   covariance, the background and a coarse position. No pixel-level fit is used anywhere.
2. **Pattern estimation**: the illumination pattern (pitch, angle, phases, depth, drift) is estimated from the pass-1 output.
3. **Explicit solve**: the position follows from the network intensities and the pattern.
4. **Pass 2** (optional, pattern-aware): detector and head get the estimated pattern as extra input, then the same solve.
5. **Table**: localizations with calibrated uncertainties, and a recommended strict subset.

## Top level (`simcode/`)

| file | what it does |
|---|---|
| `run.py` | User entry point: `python -m simcode.run --movie … --calib … --out … [--pass2]`. Runs all stages with production settings; finished stages are skipped. |
| `strict.py` | Builds the localization table from a pipeline output directory and selects the strict subset. Also a CLI. |
| `config.py` | All paths (models, data, outputs, external DME) from environment variables, `simcode.json` or defaults. No path is hard-coded elsewhere. |
| `core.py` | Pattern geometry helpers used by the explicit solve (illumination at a position, fringe shifts, kinetics start values). |
| `autoc.py` | Automatic per-dataset noise scale of the network intensities, estimated from the solve's own residuals (no ground truth). |
| `fastutil.py` | Optional `torch.compile` support for the fast inference paths. |
| `__init__.py` | Package marker. |

## `pipeline/`: running on a movie

| file | what it does |
|---|---|
| `run_uncalibrated.py` | Expert entry point with all flags; `run.py` calls its stages. |
| `common.py` | Movie reading (multi-part TIFF, absolute frame index, prefetch), camera conversion, PSF model loading. |
| `sc_front.py` | Reference implementation of pass 1 (detector + head per window) and of the pass-2 chunk loop. |
| `sc_front_fast.py` | Faster versions of the same chunks (queued fixed-size batches); same networks and outputs. Used with `--fast`. |
| `pattern_est.py` | Estimates the illumination pattern from pass-1 localizations and intensities. |
| `pattern_field.py` | Optional per-cell phase map and smooth depth field on top of the global pattern. Not used by default. |
| `compare_pattern.py` | Compares an estimated pattern with a reference pattern, in rad and nm. |
| `dedup.py` | Output modes for overlapping windows: non-overlapping (disjoint) windows, or merged chains. |

## `det/`: pass-1 detector

| file | what it does |
|---|---|
| `model.py` | Pattern-free dense detector network (U-Net): per pixel a detection score and a sub-pixel offset. |
| `up2.py` | The production detector `det_n1` (model wrapper, training targets and training loop). |
| `train.py` | Earlier detector trainer, kept for the detector variants. |
| `data.py` | Detector training scenes with DNA-PAINT kinetics. |
| `psfpert.py` | Bank of perturbed PSF lookup tables (width, aberrations) and camera errors, for robustness in training. |

## `hd/`: pass-1 head, explicit solve, simulation for training and benchmarks

| file | what it does |
|---|---|
| `sc.py` | Base per-candidate head and the wrapper around the explicit pattern solve. |
| `sc4.py` | The production head family (neighbour and detector-probability inputs, full intensity covariance, position prior) and its training. |
| `sc7bg.py` | Head variant with a separate background network. |
| `msolve.py` | Multi-start explicit pattern solve on the network intensities (reference implementation). |
| `msolve_fast.py` | Faster drop-in for the same solve. |
| `realsim.py` | Experiment-matched simulator: measured PSF with jitter, camera maps, structured and fluctuating background, filaments. |
| `realgen.py` | One realistic training-batch generator for all four networks. |
| `data.py` | Training batches for the head from `realsim.py` with widened parameter ranges. |
| `gate.py` | The gate benchmark: fixed simulated scenes (isolated emitters, close pairs, realistic scenes) run through a front end. |
| `gate_report.py` | Detection, precision and intensity metrics for gate outputs. |
| `gate_report_xdet.py` | Second version of the gate report (precision matched on the detector position). |

## `p2/`: pass 2 (pattern-aware networks)

| file | what it does |
|---|---|
| `det2.py` | Pass-2 detector: the pass-1 detector plus illumination inputs; model and training. |
| `head2.py` | Pass-2 head: the pass-1 head plus illumination patch inputs; model and training. |
| `illum.py` | Illumination input maps from a pattern, and perturbed pattern estimates for training. |
| `data.py` | Realistic training batches for the pass-2 networks. |
| `front.py` | Pass 2 inside the pipeline. |
| `dump.py` | Dumps raw network outputs on the gate scenes for the pass-2 benchmark. |

## `sim/`: basic simulator and Cramér–Rao bounds

| file | what it does |
|---|---|
| `simulator.py` | GPU simulator of modulated DNA-PAINT movies from the calibrated physical model. |
| `crlb.py` | Scene generator and multi-emitter CRLB for the benchmark scenes. |
| `simtools.py` | Emitter layouts for the gate scenes (grids, pairs, random scenes) and the pattern-error model. |

## `psf/`: point spread function

| file | what it does |
|---|---|
| `vectorial.py` | Vectorial PSF model (dipole emitter, Zernike aberrations, optional per-angle terms), differentiable. |
| `lut.py` | Spline lookup table of the PSF for fast rendering. |
| `fit_beads.py` | Fits the PSF model to bead z-stacks. |

## `insitu/`: calibration from the movie itself

| file | what it does |
|---|---|
| `__init__.py` | Default input and output locations. The recipe is in this folder's README. |
| `extract_tracks.py` | Extracts isolated emitter tracks (raw regions over consecutive frames). |
| `fit_insitu.py`, `fit_insitu2.py` | Stage A: PSF and camera parameters from the tracks (`fit_insitu2` solves intensities and background in closed form). |
| `varpro.py` | The closed-form linear solve used by `fit_insitu2.py`. |
| `fit_pattern.py` | Stage B: illumination pattern from the track intensities. |
| `fit_psf_sums.py` | PSF refit on bright multi-frame sums. |
| `camera_gain.py` | Camera gain and read noise from the movie (no dark frames needed). |
| `camera_maps.py` | Per-pixel offset and read-noise maps from the movie. |
| `psf_heldout.py`, `psf_compare.py` | Held-out evaluation of a fitted PSF, and comparison of several fits on the same samples. |
| `coma_anchor_scan.py` | Diagnostic scan of odd aberrations against the pattern position. |
| `lm_refine.py` | Pixel-level fit used only by that diagnostic; not part of the localization method. |

## `fourpi/`: 4Pi-SIMFLUX (separate data type; does not touch the pipeline above)

| file | what it does |
|---|---|
| `psf4pi.py` | Coherent two-arm vectorial 4Pi PSF with the four interference channels. |
| `lut4pi.py` | Spline lookup table of that PSF with the 4Pi phase as an exact analytic dependence. |
| `sim4pi.py` | Simulator of candidate patches (6 frames × 4 channels). |
| `head4pi.py` | Per-candidate head for the 24 channel patches. |
| `solve4pi.py` | Explicit solve: lateral position from the pattern, z from the 4Pi phase. |
| `lm.py` | Generic batched Poisson fit and CRLB, used for the classical 4Pi baselines. |

## Smaller folders

| file | what it does |
|---|---|
| `nn/blocks.py` | Shared network blocks (residual block, patch extraction). |
| `util/drift.py` | Access to the external DME v2 drift-estimation package. |
| `util/spatial_hash.py` | GPU/CPU neighbour search used by the pattern estimation and `autoc.py`. |
| `util/neighbours.py` | Small nearest-neighbour helper. |
| `evalproto/lib.py` | Evaluation metrics without ground truth (FRC variants, revisit statistic, nearest-neighbour proxies). |
| `eff/crlb_nuis.py` | Multi-emitter CRLB with nuisance parameters, for the gate scenes. |
| `eff/simcode_gate.py` | Runs the preprint's SIMCODE model on the gate scenes, for comparison. |

## Outside the package

- `analysis/`: scripts that use the package (baselines, calibration, figures, paper figures, simulated movies, speed tests,
  4Pi processing). They are not needed to run the method. `analysis/paper/README.md` and `analysis/figures/README.md`
  describe the figure scripts.
- `models/`: trained networks (`*.pt`) and calibrations (`calib/`, `insitu/`).
- `tests/`: `test_fast_paths.py` checks that the fast paths match the reference implementations.
- `docs/`: method, evaluation, results and the PSF polarization note.
