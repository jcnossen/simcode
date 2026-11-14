# SIMCODE v2: what each file does

One line per file of the `simcode` package, grouped by folder (state after the 2026-11 cleanup, `docs/CLEANUP_LOG.md`). The
docstring at the top of each file has the details. Method description: `docs/METHOD.md`. Evaluation: `docs/EVALUATION.md`.

The repository holds two comparisons and what they need:
- **SIMCODE preprint vs SIMCODE v2** on DMD-SIMFLUX DNA-PAINT movies: the v2 pipeline below with the PSF-matched network
  sets, the preprint's model as a baseline (`analysis/baselines/run_simcode*.py`), and the paper's other baselines (SMLM,
  SIMFLUX, DECODE).
- **4Pi-SIMFLUX (classical, as published) vs 4Pi-SIMCODE** (network): `fourpi/` below and `analysis/fourpi_classical/`.

## How the pieces fit (2D pipeline)

A movie is processed in 6-frame windows (2 pattern angles × 3 phase steps), one window every 3 frames:

1. **Pass 1** (pattern-free): a detector finds candidates; a per-candidate head network gives per-frame intensities with a
   covariance, the background and a coarse position. No pixel-level fit is used anywhere.
2. **Pattern estimation**: the illumination pattern (pitch, angle, phases, depth, drift) is estimated from the pass-1 output.
3. **Explicit solve**: the position follows from the network intensities and the pattern.
4. **Pass 2** (pattern-aware): detector and head get the estimated pattern as extra input, then the same solve.
5. **Table**: localizations with calibrated uncertainties; the paper's selection is max(σx, σy) ≤ 12 nm on the rows with p ≥ 0.63 inside the crop.

## Top level (`simcode/`)

| file | what it does |
|---|---|
| `run.py` | User entry point: `python -m simcode.run --movie … --calib … --model m1809 --out …` (`--model`: a name of `models.json` or a model folder; required), or `--dataset <name>` for a paper dataset. Runs all stages; finished stages are skipped. |
| `strict.py` | Builds the localization table from a pipeline output directory (with the model's sigma rescale), and the paper's selection `select(table, max_precision_nm=12, min_score=0.63, in_region=True)`; writes the selection as subset files (`--subset 12`). Also a CLI. |
| `dataset.py` | `Dataset`: a localization table as an object. Columns as numpy arrays (extra fields: `add_fields`), metadata, drift. Loads a run directory (`from_run(dir, pass_=2)`), v2 tables (`load_npz`), the preprint code's `.npy` (`load_npy`), Picasso hdf5 (`load_hdf5`), ThunderSTORM csv (`load_csv_thunderstorm`), the paper's DECODE tables; `select()` (= `strict.select`), filters, crop, concatenate (copying, and in place with a trailing underscore: `filter_`, `crop_`, `crop_from_edge_`, `select_`, `crlb_filter_`, `autocrop_`), DME v2 drift (`estimate_drift` / `apply_drift` / `undo_drift`), render, FRC; saves npz / Picasso hdf5 / ThunderSTORM csv / `.npy`. `docs/USAGE.md`. |
| `models.json` | The shipped models (m1809, m25, g13, m1809g), the default model, and the paper's datasets (`_production`); any other model is a folder with a `model.json` (`config.model_set` takes both). |
| `config.py` | All paths (models, data, outputs, 4Pi work directory, 4Pi data, external DME) from `SIMCODE_*` environment variables, `simcode.json` or defaults; `model_set(name or model folder)`. No path is hard-coded elsewhere. |
| `core.py` | Pattern geometry helpers used by the explicit solve (illumination at a position, fringe shifts, kinetics start values). |
| `autoc.py` | Opt-in automatic per-dataset noise scale of the network intensities, estimated from the solve's own residuals (no ground truth). Its sigma table exists only for the earlier default head. |
| `fastutil.py` | Optional `torch.compile` support for the fast inference paths. |
| `__init__.py` | Package marker. |

## `fileio/`: file handling shared by the 2D pipeline and the 4Pi pipelines

| file | what it does |
|---|---|
| `movie.py` | One reader interface (`n`, `shape`, `parts`, `read(f0, f1, y0, x0, S)`) with three back ends: `TiffMovie` (multi-part TIFF), `DcimgMovie` (Hamamatsu dcimg, optional channel-crop layout), `ArrayMovie` (array / `.npy`); `open_movie`; `Prefetch` (the next read runs in a background thread). |
| `work.py` | The resumable work driver: `Units` (a chunk of a movie or a file of an acquisition = one output file; done / claim with dead-process takeover / failed marker / per-unit record json / the worker loop `todo`), atomic writes (`atomic_savez`, …), `disk_guard`. |
| `launch.py` | Several GPUs = the same command once per GPU: `python -m simcode.fileio.launch --gpus 1,2 -- <command … --claim 1>`. |

## `pipeline/`: running on a movie

| file | what it does |
|---|---|
| `run_uncalibrated.py` | Expert entry point with all flags; `run.py` calls its stages. Chunks are the units of `fileio/work.py`. |
| `common.py` | Read plan of the chunks, the rows of the pass-1 chunk files, camera conversion, PSF model loading (torch_psf.load_psf). |
| `sc_front.py` | Reference implementation of pass 1 (detector + head per window) and of the pass-2 chunk loop. |
| `sc_front_fast.py` | Fast versions of the same chunks (queued fixed-size batches); same networks and outputs. The pipeline's code path (`sc_front.py`, `p2/front.py` are the test oracle). |
| `pattern_est.py` | Estimates the illumination pattern from pass-1 localizations and intensities. |
| `pattern_field.py` | Per-cell phase map and smooth depth field on top of the global pattern (`field_at`, `estimate_field`). Analysis only (`analysis/evalproto/spatial.py`, `analysis/simreal/sim_movie_r.py`); the pipeline option was removed on 2026-11-14. |
| `compare_pattern.py` | Compares an estimated pattern with a reference pattern, in rad and nm. |
| `dedup.py` | Output modes for overlapping windows: non-overlapping (disjoint) windows, or merged chains. |

## `det/`: pass-1 detector

| file | what it does |
|---|---|
| `model.py` | Pattern-free dense detector network (U-Net): per pixel a detection score and a sub-pixel offset. |
| `up2.py` | The production detector (model wrapper, training targets and training loop). |
| `data.py` | Detector training scenes with DNA-PAINT kinetics. |
| `psfpert.py` | Bank of perturbed PSF lookup tables (width, aberrations) and camera errors, for robustness in training. |

## `hd/`: pass-1 head, explicit solve, simulation for training and benchmarks

| file | what it does |
|---|---|
| `sc.py` | Shared pieces of the per-candidate head (patch extraction, PSF template, head call on detector candidates, loader) and the wrapper around the explicit pattern solve. |
| `sc4.py` | The production head (neighbour and detector-probability inputs, full intensity covariance, position prior) and its training. |
| `msolve.py` | Multi-start explicit pattern solve on the network intensities (reference implementation). |
| `msolve_fast.py` | Faster drop-in for the same solve (production). |
| `realsim.py` | Experiment-matched simulator: measured PSF with jitter, camera maps, structured and fluctuating background, filaments. |
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
| `dump.py` | Dumps raw network outputs on the gate scenes (input of the calibration fit of a model). |

## `sim/`: basic simulator and Cramér–Rao bounds

| file | what it does |
|---|---|
| `simulator.py` | GPU simulator of modulated DNA-PAINT movies from the calibrated physical model. |
| `crlb.py` | Scene generator and multi-emitter CRLB for the benchmark scenes. |
| `simtools.py` | Emitter layouts for the gate scenes (grids, pairs, random scenes) and the pattern-error model. |

## `psf/`: point spread function

| file | what it does |
|---|---|
| `__init__.py` | The only module that imports torch_psf (PSF models, calibration reader, fits, TIFF reader; re-exported for the rest of SIMCODE) plus the SIMCODE glue: per-pattern-angle PSF terms, `render`, the spline of the solve (`spline`). |
| `beads.py` | Fits the PSF model to bead z-stacks (torch_psf.estimate_psf_beads). |

## `insitu/`: calibration from the movie itself

| file | what it does |
|---|---|
| `__init__.py` | Default input and output locations. The recipe is in this folder's README. |
| `extract_tracks.py` | Extracts isolated emitter tracks (raw regions over consecutive frames). |
| `fit_insitu2.py` | Stage A: PSF from the tracks (intensities and background solved in closed form). |
| `fit_pattern.py` | Stage B: illumination pattern from the track intensities. |
| `fit_psf_sums.py` | PSF refit on bright multi-frame sums. |
| `camera_gain.py` | Camera gain and read noise from the movie (no dark frames needed). |
| `camera_maps.py` | Per-pixel offset and read-noise maps from the movie. |
| `psf_heldout.py`, `psf_compare.py` | Held-out evaluation of a fitted PSF, and comparison of several fits on the same samples. |

## `fourpi/`: 4Pi-SIMFLUX (separate data type; does not touch the 2D pipeline)

| file | what it does |
|---|---|
| `run.py` | **4Pi-SIMCODE**, the network pipeline on raw dcimg files: `python -m simcode.fourpi.run --cell NPC --files 10 --head v10` (`process_file`: read → detector → patches → head → in-situ pattern and 4Pi phase field → explicit solve). Files are the units of `fileio/work.py`. |
| `setup.py`, `datasets.json` | The 4Pi set-up in one place: work directory, data directory, channel crop layout, gain, pixel size, patch size, per-cell phase offset, file lists. |
| `front.py` | Front end shared with the classical pipeline: registration map (`PolyMap`), camera conversion, registered 4-channel sum, detection, candidate patches. |
| `insitu.py` | In-situ estimates per file: SIM pattern from per-frame intensities (slot DFT), per-candidate pattern phases, 4Pi phase field (tile map + time blocks). |
| `head4pi_w.py` | The learned-window head (v10): learned read-out windows with a frame-symmetric, emitter-orthogonal guard; contains a frozen `Head4Pi` as its prior. |
| `head4pi.py` | Per-candidate head for the 24 channel patches (the prior inside `head4pi_w.py`; alone = the candidate front of the simulated movies, checkpoint v3L). |
| `loadhead.py` | Loader of the 4Pi head checkpoints. |
| `windows4pi.py` | The canonical read-out windows (matched filter, cross-term quadratures, flat) of the head and the solve. |
| `linread4pi.py` | Fixed linear window read-out with analytic covariance: the reference the learned head is gated against. |
| `solve4pi.py` | Explicit solve: lateral position from the pattern, z from the 4Pi phase (base class, kinetics starts). |
| `solvek4pi.py` | The same solve on K window read-outs per (frame, channel) (used by the pipeline). |
| `solveloss4pi.py` | Solve-aware training loss of the learned-window head. |
| `psf4pi.py` | Coherent two-arm vectorial 4Pi PSF with the four interference channels. |
| `lut4pi.py` | Spline lookup table of that PSF with the 4Pi phase as an exact analytic dependence. |
| `sim4pi.py` | Simulator of candidate patches (6 frames × 4 channels). |
| `realsim4pi.py` | Full-scene 4Pi simulator for the detector training. |
| `lm.py` | Generic batched Poisson fit and CRLB, used for the classical 4Pi baseline. |
| `zdrift.py` | Sample axial drift estimated from the structure itself. |

## Smaller folders

| file | what it does |
|---|---|
| `nn/blocks.py` | Shared network blocks (residual block, patch extraction). |
| `util/drift.py` | Lateral drift of a localization table with the external DME v2 package (`drift_of_table`; the row selection is here). |
| `util/runconfig.py` | The run record `<out>/run_config.json` (+ yaml): resolved settings, file hashes, code / environment, pattern and selection outcome, later steps; re-run arguments of a record (`simcode.run --config`), `verify`, and `backfill` for runs made before 2026-11-14. |
| `util/ckpt.py` | Atomic checkpoint save with a disk guard, used by every trainer. |
| `util/proctitle.py` | Recognizable process names (`simcode:<stage> <unit>`) in `nvidia-smi` / `ps`. |
| `util/cfgdict.py` | `ConfigDict`: nested settings with attribute access (`cfg.optics.NA`), deep merge (`base.updated(...)`), freeze, yaml / json / Python-file loading; used to define acquisitions of the automatic pipeline in code (`auto/configs/*.py`, docs/AUTO_PIPELINE.md section 1.1). Not `config.py` (the paths). |
| `util/spatial_hash.py` | GPU/CPU neighbour search used by the pattern estimation and `autoc.py`. |
| `util/neighbours.py` | Small nearest-neighbour helper. |
| `evalproto/frc.py` | The one FRC implementation (reference estimator in camera px or nm, block halves; the Hann and fixed-field variants of two scripts). |
| `evalproto/lib.py` | Evaluation metrics without ground truth (FRC block variants, revisit statistic, nearest-neighbour proxies). |
| `eff/crlb_nuis.py` | Multi-emitter CRLB with nuisance parameters, for the gate scenes. |
| `eff/simcode_gate.py` | Runs the preprint's SIMCODE model on the gate scenes, for comparison. |

## Outside the package

- `analysis/`: scripts that use the package. They are not needed to run the method.
  - `baselines/`: SMLM, SIMFLUX, DECODE and the preprint's SIMCODE on simulated and real movies; `dme_drift.py` (CLI of `util/drift.py`).
  - `paper/`: figure scripts of the experimental figures and the fair FRC protocol (`README.md` there).
  - `papersim/`: the paper simulations (generators, all methods, metrics, figures); `reproduce.sh` lists the commands.
  - `fourpi/`: 4Pi calibration chain (camera, registration, PSF, pattern checks), the v10 trainers and gates, post-processing, real-data comparisons and the simulated showcase. `real_w4pi.py` = `simcode.fourpi.run`.
  - `fourpi_classical/`: the classical 4Pi-SIMFLUX pipeline (published-style detection and fits, pixel MLE) on the same reader and work driver.
  - `calib/`, `psfmatch/`, `psf3d/`, `psf3d_nets/sigcal.py`: calibration of a model (sigma tables, sigma rescale, PSF calibration files).
  - `gate/`, `figures/`, `evalproto/`, `hd_compare/`, `simreal/`, `speed/`, `tput/`, `dme/`, `psfshape/`: benchmarks and development figures; `results/` holds text snapshots of their outputs.
- `models/`: calibrations (`calib/`, `insitu/`); network weights are not in git (`models/README.md`).
- `tests/`: `test_fast_paths.py` (GPU: fast paths against the reference implementations), `test_fileio.py` (CPU: readers, work driver, config names), `test_tables.py` (CPU: table columns, selections, FRC), `test_dataset.py` (CPU: the `Dataset` class: formats, selection, drift, render, FRC against `analysis/paper/frc_fair.py`; one short DME call on the GPU when available), `check_imports.py` (static check of the tree), `check_checkpoints.py`, `compare_npz.py` / `compare_ckpt.py` (bit-identity of outputs).
- `docs/`: method, evaluation, results, the PSF polarization note, the cleanup plan and log.
