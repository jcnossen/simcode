# SIMCODE v2: package layout

One line per directory and file of the `simcode` package (reorganised 2026-11-15, see "Renamed" at the end; the 2026-11
cleanup before it: `docs/CLEANUP_LOG.md`). The docstring at the top of each file has the details. Method: `docs/METHOD.md`.
Evaluation: `docs/EVALUATION.md`. Working with the results: `docs/USAGE.md`.

The repository holds two comparisons and what they need:
- **SIMCODE preprint vs SIMCODE v2** on DMD-SIMFLUX DNA-PAINT movies: the v2 pipeline below with the PSF-matched models, the
  preprint's model as a baseline (`analysis/baselines/run_simcode*.py`), and the paper's other baselines (SMLM, SIMFLUX, DECODE).
- **4Pi-SIMFLUX (classical, as published) vs 4Pi-SIMCODE** (network): the 4Pi modules below (`setups/fourpi.py`, `channels/`, `*/fourpi*.py`) and `analysis/fourpi_classical/`.
- **Plain 4Pi (4Pi-SMS, no SIM pattern)**: the same 4Pi modules with single-frame windows (T = 1; setup `fourpi_plain`), on the public data set of Li & Ries (`pipeline/configs/fourpi_plain.py`, `analysis/fourpi_plain/`).

## How the pieces fit (2D pipeline)

A movie is processed in 6-frame windows (2 pattern angles × 3 phase steps), one window every 3 frames:

1. **Pass 1** (pattern-free): the detector (`detector/`) finds candidates; the emitter network (`emitter/`) gives per
   candidate the per-frame intensities with a full covariance, the background and a 3D position prior. No pixel-level fit
   is used anywhere.
2. **Pattern estimation** (`processing/pattern_estimation.py`): the illumination pattern (pitch, angle, phases, depth, drift)
   from the pass-1 output.
3. **Explicit solve** (`solve/`): the position follows from the network intensities and the pattern; z from the network's
   prior conditioned on the solved x, y.
4. **Pass 2** (pattern-aware): the same two network classes with illumination inputs (`pattern/illumination.py`) get the
   estimated pattern, then the same solve; z from pass 1.
5. **Table** (`table.py`): localizations with calibrated uncertainties; the paper's selection is max(σx, σy) ≤ 12 nm on
   the rows with p ≥ 0.63 inside the crop.

`processing/` runs this per chunk of a movie; `run.py` is the user entry point; `pipeline/` is the automatic
per-acquisition pipeline around it (camera, PSF, model training, processing, report). The networks are trained on
`simulate/` (acquisition simulator, training batches) and checked on the `benchmark/` gate scenes.

## Tree

```
simcode/
  __init__.py                 package marker, version
  download_models.py          python -m simcode.download_models [--set m1809 g13 pretrain | all] [--dest models/]: the network weights (GitHub release assets, sha256-checked)
  run.py                      user entry point: python -m simcode.run --movie … --calib … --model m1809 --out … (or --dataset 1809_2 --data-root …); all stages, finished ones skipped
  config.py                   the models: MODELS = <repo>/models (where model / calibration names are looked up: resolve, model), model_set(name or model folder), production(); no path settings (paths are arguments)
  models.json                 the shipped models (m1809, m25, g13: detector, emitter, cal_sim, detector_pass2, emitter_pass2,
                              cal_sim_pass2, psf), the default model, the paper's datasets (_production)
  dataset.py                  Dataset = torch_psf.Dataset + SIMCODE's loaders (from_run, load_table, from_columns, columns(), load_npz, from_decode, load_npy / from_smlmtorch) and the paper's selection select() / mask()
  table.py                    the four localization tables of a run directory (pass 1 / 2 x network / solve; locs_p2_solve = the result), torch_psf Dataset HDF5 table files, the paper's selection select() (applied at load); CLI simcode-table
  detector/
    __init__.py               package doc
    model.py                  Detector: the U-Net detector of pass 1 and pass 2 (illum=True: + illumination inputs, zero-initialised from pass 1); load() of every detector checkpoint
    select.py                 candidate selection: sub-pixel greedy NMS + top-k (select_subpixel), select_points
    train.py                  trainer of both passes: python -m simcode.detector.train (pass 2: --illum 1 or a pass-2 --init); focal loss + offset NLL with nn-bin weights
  emitter/
    __init__.py               package doc
    model.py                  EmitterNet: the per-candidate network of pass 1 and pass 2 (I_t with full covariance, bg_t, 3D position prior; illum=True: + illumination patches); neighbour inputs, PSF template, detect_candidates / apply_to_candidates, load() of every checkpoint (2D ones with zero z rows)
    train.py                  trainer of both passes: python -m simcode.emitter.train (pass 2: a pass-2 --detector or --init); intensity / prior / z NLL, anti-shrinkage and slope terms
    heldout.py                held-out gate-type patches of the pass-1 trainer (python -m simcode.emitter.heldout [--set 5]) and their metrics
    fourpi.py                 the 4Pi emitter network: FourPiEmitterNet (learned windows, v10; arch tag 'w') with the frozen FourPiPrior (v3L) inside, T frames per window (6; plain 4Pi 1: arch tags 'fourpi_emitter' / 'fourpi_prior' carry T, zmax, vread), the read-out windows, the fixed-window read-out, the training losses (solve_loss), load() of the 4Pi checkpoints by their arch tag
  solve/
    __init__.py               package doc
    solve.py                  the multi-start explicit pattern solve (reference implementation) and solve_emitters: the solve of the emitter-network outputs with a model calibration
    solve_fast.py             the same solve, fast (production; solve_emitters_fast, optional compiled LM step)
    pattern.py                pattern geometry of the solve (illumination at a position, angle index, fringe shifts, kinetics starts, chi2 p-value)
    z_conditioning.py         z after the solve: the network's 3D prior conditioned on the solved x, y; pass-2 rows linked to pass-1 candidates for their z
    brightness_noise_fit.py   the dataset's brightness noise c from the solve's own residuals on the clean emitter-windows (a step of the pattern estimation)
    fourpi.py                 FourPiSolve: the explicit 4Pi solve on the window read-outs (pattern + 4Pi phase, kinetics modes, phase mode, residual background; T frames, k None = no pattern), kinetics starts
  pattern/
    __init__.py               package doc
    illumination.py           illumination input maps of a pattern (cos / sin per frame) and perturbed pattern estimates (training error models v1 / real)
  processing/
    __init__.py               package doc
    run_uncalibrated.py       expert entry point with all flags; run.py calls its stages; chunks are the units of io/work.py
    common.py                 read plan of the chunks, pass-1 chunk rows, camera conversion, calibration / PSF loading, pattern_frames, CAND_PMIN
    pass1_inference.py        reference pass 1 (Pass1Front; test oracle) and z_inputs (stored pass-1 z as solve input)
    pass2_inference.py        reference pass 2 with the pattern-aware networks (Pass2Front); test oracle
    inference_fast.py         the pipeline's code path: queued fixed-size batches for pass 1 / pass 2 (Pass1FrontFast, Pass2FrontFast) and the chunk-wide pattern solve
    pattern_estimation.py     the illumination pattern from the pass-1 rows (k, per-step phases, depth, relative intensities, per-angle phase drift)
    pattern_field.py          evaluation of a per-cell phase map and smooth depth field on top of the global pattern (field_at; analysis only)
    gate_dump.py              raw network outputs of pass 1 / pass 2 on the gate scenes (input of a model's calibration fit): python -m simcode.processing.gate_dump
    fourpi.py                 4Pi-SIMCODE per file (the process stage of the fourpi / fourpi_plain set-ups): read, detect, patches, network, pattern (fourpi.pattern), phase field, solve, windows of fourpi.window frames; python -m simcode.processing.fourpi <config>
    fourpi_pattern_phase.py   per-file SIM pattern (slot DFT, per-candidate pattern phases) and 4Pi phase field (tile map + block drift) of the 4Pi pipelines
  simulate/
    __init__.py               package doc
    acquisition.py            AcquisitionSim: the experiment-matched acquisition simulator (measured PSF with jitter, camera maps, structured / fluctuating bg, filaments, kinetics, SIMCODE or generic patterns)
    batches.py                TrainingBatches: training batches of the detector and the emitter network, pass 1 and pass 2 (emitter_batch, detector_batch, illumination_batch)
    targets.py                training targets and weights from the ground truth: detector cell targets, nn-bin weights, candidate-GT matching, emitter-network targets
    simulator.py              Simulator: the calibrated physical model (PSF rendering: simcode.psf.calib_psf, calibration, camera)
    crlb.py                   scene generator and multi-emitter CRLB of the benchmark scenes
    gate_layouts.py           emitter layouts of the gate scenes (grids, pairs, Poisson scenes, wide brightness) and the pattern-estimate error model
    psf_perturbation.py       perturbed PSFs of the PSF bank (width, aberrations, focus, blur)
    fourpi_patches.py         FourPiPatchSim: 4Pi candidate patches (T x 4 channel patches; pattern=False: plain 4Pi) with GT read-out targets (4Pi emitter-network training, solve gates)
    multichannel.py           MultiChannelAcquisitionSim: full multi-channel scenes through the registration maps of CellSpec acquisitions (4Pi detector training, simulated movies; patterns=False: plain 4Pi with T frames; random registration error), candidate_patches (plain-4Pi emitter-network training)
  benchmark/
    __init__.py               package doc
    gate_scenes.py            the gate suite: fixed simulated scenes (isolated, pairs, realistic) through a front end: python -m simcode.benchmark.gate_scenes
    gate_scenes_report.py     detection, precision and intensity metrics of gate outputs
    gate_scenes_report_detector_matched.py  gate report with the precision targets matched on the detector position
  calibration/
    __init__.py               default input / output locations of the in-situ calibration
    extract_tracks.py         isolated emitter tracks (raw regions over consecutive frames)
    camera_maps.py            per-pixel offset and read-noise maps from the movie
    psf_compare.py            several PSF fits compared on the same samples (analysis/psf3d/summary.py)
    gate_stats.py             statistics of the gate-scene dumps: matching, explicit solve on dumped outputs, prior_off fit (fit_cal), RMS / CRLB (until 2026-11-16 analysis/calib/plib.py)
    fit_cal_set.py            python -m simcode.calibration.fit_cal_set: the solve calibration (prior_off) of a new model from its gate dumps (stage train; until 2026-11-16 analysis/psfmatch/fit_cal_set.py)
  pipeline/
    __init__.py               the automatic per-acquisition pipeline (LocalizationPipeline, ConfigDict)
    __main__.py               python -m simcode.pipeline <config>
    localization_pipeline.py  LocalizationPipeline: the stages of the acquisition's set-up in order (widefield: order, camera, spots, cammaps, tracks, beads, psf, train, process, report; fourpi: registration, process, report), done markers, checks
    acquisition_config.py     the acquisition config: defaults (+ the set-up's section), setup / channels / registration keys, yaml / json / Python files, validation, frame-order helpers
    configs/                  acquisitions: 1809_2.py / .yaml, example_setup.py; fourpi.py (the 4Pi MT and NPC movies), fourpi_plain.py (plain 4Pi: Li & Ries Cell11): configs(work, data), python -m simcode.pipeline <file> --work … --data …
    frame_order.py            stage order: blocked-order cache of a movie with another frame order
    camera.py                 stage camera: gain, read noise, offset from the variance-vs-level fit
    spots.py                  stage spots: spot finding and Gaussian fits for the tracks
    psf_fit.py                in-situ PSF fit of stage psf (torch_psf fit_tracks with the sample selection, camera conversion and photoselection hook; held-out check by torch_psf heldout_tracks, refit with simcode.psf.calib_psf)
    psf_stage.py              stage psf: multi-start in-situ PSF, line blur, photoselection, held-out, calibration files
    training.py               stage train: a PSF-matched model with the recipe of models/README.md, gate dumps, calibrations, acceptance, model.json
    acceptance.py             stage train acceptance on the gate dumps (sigma calibration, pass-2 vs pass-1 RMS / CRLB)
    process.py                stage process: simcode.run on the movie with the dataset's calibration and model
    run_with_overrides.py     a module run with the pipeline's instrument overrides applied in memory (AUTO_WRAP)
    report.py                 stage report: summary page of an acquisition
  io/
    __init__.py               package doc
    movie.py                  one reader interface with TIFF / dcimg / interleaved two-channel TIFF / array back ends (open_movie), Prefetch
    work.py                   the resumable work driver (Units: claims, dead-process takeover, records), atomic writes, disk guard
    launch.py                 the same command once per GPU: python -m simcode.io.launch --gpus 1,2 -- <command … --claim 1>
  evaluation/
    __init__.py               package doc
    frc.py                    the one FRC implementation (camera px or nm, block halves)
    metrics.py                metrics without ground truth (FRC block variants, revisit statistic, nearest-neighbour proxies)
  layers/
    __init__.py               package doc
    blocks.py                 ResBlock, norm_photons, extract_patches
  psf/
    __init__.py               the only module that imports torch_psf (PSF models, calibration reader, fits, TIFF reader) + the SIMCODE glue (per-angle PSF terms, render, spline)
    beads.py                  PSF fit to bead z-stacks (torch_psf.ZStackEstimator)
  channels/                   acquisitions with several channels on one camera (1, 2, 4)
    __init__.py               package doc
    layout.py                 Channel (crop, mirror, page, psf, weight), ChannelLayout (from the config key channels; split of a frame), ChannelReader
    registration.py           PolyMap (channel c <-> channel 0, json with feats_norm) and the in-situ registration per file (per-channel fits, coarse alignment incl. flips, matching, robust order-2 map + inverse, held-out residual, self-consistency): register_insitu, checks
    front.py                  ChannelFront: camera conversion, registered channel sum, classical detection, per-channel patches at the mapped positions
  setups/                     set-ups of the pipeline: what differs between acquisition types
    __init__.py               get(name): widefield | fourpi | fourpi_plain
    base.py                   the Setup interface (defaults, validate, stages, st_<stage>, movie_files, reader)
    widefield.py              SIMCODE on DMD-SIMFLUX movies (the pipeline's own stages; registration with several channels)
    fourpi.py                 4Pi-SIMFLUX: config section fourpi, files, dcimg channel reader, camera, per-file registration, front, stages registration / process / report; acquisition(name, work, data); FourPiPlainSetup (fourpi_plain): no pattern, fourpi.window frames, the training recipe (section fourpi_train, sim_kw)
  util/
    __init__.py               package marker
    checkpoint.py             atomic checkpoint save with a disk guard (every trainer)
    compile.py                optional torch.compile support of the fast paths
    drift.py                  lateral drift of a table with the external DME v2 package (drift_of_table)
    runconfig.py              the run record run_config.json (+ yaml): settings, hashes, code / environment, outcome; re-run, verify
    config_dict.py            ConfigDict: nested settings with attribute access, deep merge, freeze, yaml / json / Python loading
    legacy_names.py           the one map of names that exist only in stored data (earlier checkpoint arch tags, format-1 run-record keys), used by the readers
    proctitle.py              recognizable process names in nvidia-smi / ps
    spatial_hash.py           GPU / CPU neighbour search (pattern estimation, brightness_noise_fit, pass-1 linking)
```

## Checkpoints

Network checkpoints are plain state dicts with the training arguments (`{'model', 'args', 'step'}`; no pickled classes,
`tests/check_checkpoints.py`). The readers map them to the two classes:
- `detector.model.load`: pass 1 or pass 2 from the state dict (illumination weights `cin_il`), channel widths from the args
  (new checkpoints also carry `args['arch'] = 'detector'`).
- `emitter.model.load`: the architecture tag `args['arch'] = 'emitter_net'` (new checkpoints; the earlier pass-1 checkpoints store
  an earlier tag of the same network, read through `util/legacy_names.EMITTER_ARCH_TAGS`; the earlier pass-2 ones carry none),
  pass 1 or pass 2 from the state dict (`cin_il`), 2D checkpoints (before 2026-11-14) with zero z rows (`net.z3d` False).
- `emitter.fourpi.load`: `'fourpi_emitter'` / `'fourpi_prior'`; the 4Pi checkpoints trained before 2026-11-15 through
  `util/legacy_names.FOURPI_ARCH_TAGS`.
The stored tags are not rewritten: every shipped pass-1 checkpoint has the earlier tag, and rewriting would change the sha256
recorded in model.json files and run records.
Every checkpoint of `models/` (models.json models, model folders, `archive_*`, the other weights) and the 4Pi detector /
heads give bit-identical outputs with the code before and after the 2026-11-15 reorganisation and the 2026-11-16 rename
(`tests/check_model_outputs.py`).

## Model files

A model folder (`models/<name>/`, written by `simcode.pipeline.training`; names in `config.MODEL_FILES`):
`detector_pass1.pt`, `emitter_pass1.pt`, `detector_pass2.pt`, `emitter_pass2.pt`, `calib/calib_emitter_pass1_sim.json`,
`calib/calib_emitter_pass2_sim.json`, the PSF it was trained on, `model.json`. The networks
a new model starts from: `models/pretrain_detector_pass1.pt`, `models/pretrain_emitter_pass1.pt`; the calibration templates
of a new model: `models/calib/template_calib_emitter_pass{1,2}_sim.json` (`models/README.md`).

Removed 2026-11-17: the real-data sigma calibrations `calib/calib_emitter_pass{1,2}_real.json` of every model, the templates
`calib/template_calib_emitter_pass{1,2}_real.json`, the model keys `cal_real` / `cal_real_pass2` (a definition that still has
them is refused, `config.REMOVED_KEYS`), `config.CAL_REAL` / `CAL_REAL_PASS2` / `TEMPLATE_CAL_REAL*`, `simcode.run --cal_kind`
(`--config` of an older record drops it with a warning), the `_real` outputs of `simcode.calibration.fit_cal_set` and the
pipeline key `train.real_mult_from` (now `train.template_model`: the shipped model a quick run starts from). The archived
model folders (`models/archive_*/`, not referenced by any config) keep their `_real` files as a record; their `model.json`
files name `cal_real` and are refused by `config.model_set`.
Removed 2026-11-17 (later): the selection-aware sigma rescale (`<set>/calib/sigma_rescale_pass{1,2}.json`, model key `sig_u`
(refused), `table.tile_rank` / `model_sig_u`, `build_table(sig_u=)`), the fixed brightness noise `cmult` / `cmult_ang` of
the solve calibrations (estimated per run with the pattern; a calibration that still holds it is refused by
`solve.solve_cal`), `calib/template_calib_emitter_pass1_brightness_noise_fit.json` and `config.CAL_BRIGHTNESS_NOISE_FIT*`, the
kinetic selection threshold 9.21 and the kin_gate of the fast solve (both models are fitted and averaged), the pass-2
network-prior variant files `pass2_est2b_<f0>.npz` (`pass2_inference.direct_view`; replaced by the network columns
x_net, ... of `pass2_est2_<f0>.npz`), the tables `locs_p1` / `locs_p2` (now `locs_p{1,2}_{net,solve}`).
Removed 2026-11-19 (one noise model: the intensity map + one c + the full network covariance): the per-orientation brightness
noise c_a (solve `cang`, `cmult_ang` in `pattern_est.pt` (an older file is read with c only and a warning), its estimation in
the pattern step), the free brightness per orientation (calibration key `B_per_angle`, solve `two_b`), the component
brightness-noise estimator with its per-model network-transfer tables (`brightness_noise_fit.estimate_components`,
`load_transfer`, `bn_transfer_pass1.json`; c is the plain residual fit), the option without the intensity map
(`pattern.intensity_map` / `--intensity_map` are accepted, 0 is ignored with a warning), and in `pipeline/psf_fit.heldout` the
8 x Catmull-Rom refit table with its `xy_limit` option and the `vp_grad` option (the refit renders with `simcode.psf.calib_psf`;
torch-psf always differentiates through the closed-form I / bg). The default table of a run is the pass-2 solve
`locs_p2_solve` again (`locs_p2_net` 2026-11-17 to 2026-11-19).
Changed 2026-11-19: the PSF stage fits with `psf.pixel_integration` 'exact' by default (the exact pixel integral of the
band-limited image; `pipeline/psf_fit.FIT_PIXEL_INTEGRATION`); 'box' is still accepted, with a deprecation warning, to
reproduce older fits, and old 'box' fits / calibrations stay readable. Work folders whose psf stage was fitted with 'box'
(before the key existed) become stale under the new default (intended; `localization_pipeline.LATE_DEFAULTS` keeps 'box' as
the value of the records before the key, so they stay up to date only with an explicit 'box').

## Outside the package

- `analysis/`: scripts that use the package; not needed to run the method.
  - `baselines/`: SMLM, SIMFLUX, DECODE and the preprint's SIMCODE on simulated and real movies; `dme_drift.py` (CLI of `util/drift.py`).
  - `paper/`: figure scripts of the experimental figures and the fair FRC protocol (`README.md` there).
  - `papersim/`: the paper simulations (generators, all methods, metrics, figures); `reproduce.sh` lists the commands.
  - `fourpi/`: 4Pi calibration chain, the v10 trainers and gates, post-processing, real-data comparisons, the simulated showcase.
  - `fourpi_classical/`: the classical 4Pi-SIMFLUX pipeline (published-style detection and fits, pixel MLE).
  - `fourpi_plain/`: plain 4Pi on the Li & Ries data: `psf_astig.py` (astigmatism scale of the bead-fitted PSF), `compare_published.py` (counts, z relation, precision on the same emitters, Nup96 radius / two-layer z vs their localizations), `sim_bench.py` (simulated movie through the real driver vs GT, sigma calibration), `quick_eval.py` (no-GT metrics of a table); training: `fourpi/train_det4pi.py --acq`, `fourpi/train_head4pi_w.py --acq`
  - `calib/`, `psfmatch/`, `psf3d/`, `psf3d_nets/sigcal.py`: calibration of a model (sigma rescale, PSF files); `sigma_joint/`: the sigma study of 2026-11-17 (joint prior / intensity covariance fine-tunes, full-sim acceptance, movie diagnosis, kap = 1 precision, Figs 3 / 4 reruns).
  - `pixbias/`, `crlbcmp/`, `ablation_cov/`, `p2generic/`, `selection_global/`: the pixel-bias, CRLB, ablation, generic-pass-2 and selection analyses of the paper / SI; `simreal/`: the benchmark sim movies (sigma-rescale fits); `z3d/`: 3D models (acceptance, model-folder swap); `figures/`: zoom picking and skeletons of the figures; `auto/`: calibration comparison of the automatic pipeline; `results/`: text snapshots cited by docs/RESULTS.md
  - `fourpi_sim/`: 4Pi-SIMCODE simulations on microtubule cell fields (`mt_cells.py`: port of the inSiliScope world model, BSD-3-Clause, notice in the file), density sweeps, the hd-head experiment (`hd_run.sh`); `solvegap/`: the paper's solve-gap analysis (pattern solve vs the SIMFLUX fit on the same frames)
  - `maintenance/`: `rename_model_keys.py` (model.json key names), `cleanup_outputs.py` (out/ + models/ clean-up, dry run by default)
  - archived on 2026-11-15 (git tag `archive-2026-11-15`, `docs/ARCHIVE.md`): nanoruler, tput, hd_compare, evalproto, gate, dme, psfshape, most of figures, the development parts of baselines / papersim / simreal / psf3d / z3d / fourpi; deleted: zimflux.
- `models/`: calibrations (`calib/`, `insitu/`), model folders (`model.json`, calibrations, README); network weights are not in git (`models/README.md`).
- `tests/`: `test_fast_paths.py` (GPU: fast paths against the reference implementations), `test_z3d.py` (3D prior, layouts), `test_joint_solve.py` (joint prior / intensity covariance in the solve), `test_fileio.py`, `test_tables.py`, `test_dataset.py`, `test_config_dict.py`, `test_runconfig.py`, `test_solve4pi_oracle.py`; `check_imports.py` (static check of the tree), `check_checkpoints.py`, `check_model_outputs.py` (outputs of every checkpoint on a fixed batch), `check_fourpi_T.py` (4Pi simulators / networks / solve / loss with 6-frame windows on the CPU, before vs after a change), `compare_npz.py` / `compare_ckpt.py` (bit-identity of outputs).
- `docs/`: this file, method, evaluation, results (development phase), usage, dataset format, automatic pipeline, PSF polarization note, the cleanup log, the cleanup candidate list and the archive index (`ARCHIVE.md`).

## Renamed (2026-11-16)

File names of the models directory and module names from the development iterations. No aliases: the only old names left
in the code are the tags stored in earlier checkpoints / run records (`util/legacy_names.py`). Run-output file names
(`pass1_<f0>.npz`, gate-scene files `<scene>__s<seed>_hd_<tag>.npz`, dump front names given on the command line) and the
directories under `out/` keep their names.

| before | after |
|---|---|
| `processing/front.py`, `processing/pass2_front.py`, `processing/front_fast.py` | `processing/pass1_inference.py`, `processing/pass2_inference.py`, `processing/inference_fast.py` |
| `solve/autoc.py`; `config.CAL_AUTOC` / `CAL_AUTOC_START`; run record `fixed.autoc` | `solve/brightness_noise_fit.py`; `config.CAL_BRIGHTNESS_NOISE_FIT` / `CAL_BRIGHTNESS_NOISE_FIT_START`; `fixed.brightness_noise_fit` |
| `fourpi/psf4pi.py`, `fourpi/lut4pi.py`; then `fourpi/coherent_psf.py` (PSF4Pi, PSF4PiA, psf_from_state), `fourpi/spline_table.py` (LUT4Pi, evaluate_ch / evaluate_ch_grad), removed 2026-11-20 | torch_psf FourPiPSF / FourPiSplinePSF through `simcode.psf`: `fourpi_psf(state)`, `fourpi_spline(state, roisize, z_range, oversample, ...)` (LUT4Pi(state, P, zmin, zmax, hz, sub, zpad) -> fourpi_spline(state, P, (zmin, zmax), sub, hz, zpad)), `fourpi_channel(spline, pos, phi, c, grad)`; PSF4PiA -> FourPiPSF(depth_um=..., zmode=...) |
| `util/cfgdict.py`, `util/ckpt.py`; `tests/test_cfgdict.py`, `tests/check_lut4pi_insitu.py` | `util/config_dict.py`, `util/checkpoint.py`; `tests/test_config_dict.py`, `tests/check_spline_table_insitu.py` |
| `benchmark/gate.py`, `gate_report.py`, `gate_report_detector_matched.py` | `benchmark/gate_scenes.py`, `gate_scenes_report.py`, `gate_scenes_report_detector_matched.py` |
| `emitter.model.ARCH_TAGS`, the legacy part of `emitter.fourpi.ARCH_TAGS`, `runconfig.KEYS_FORMAT1` | `util/legacy_names.py` (EMITTER_ARCH_TAGS, FOURPI_ARCH_TAGS, RUN_RECORD_FORMAT1_KEYS / _FIXED) |
| model folder `<set>/`: `det_<set>.pt`, `hd_<set>.pt`, `p2det_<set>.pt`, `p2headL_<set>.pt` | `detector_pass1.pt`, `emitter_pass1.pt`, `detector_pass2.pt`, `emitter_pass2.pt` |
| `<set>/calib/calib_sc_hd_<set>.json`, `calib_sc_hd_<set>_real.json`, `calib_p2L_<set>.json`, `calib_p2L_<set>_real.json`, `sigu_p1_<set>.json`, `sigu_p2_<set>.json` | `calib_emitter_pass1_sim.json`, `calib_emitter_pass1_real.json`, `calib_emitter_pass2_sim.json`, `calib_emitter_pass2_real.json`, `sigma_rescale_pass1.json`, `sigma_rescale_pass2.json` (the same in `archive_2d/`, `archive_psf_v1/`, `archive_pass2_dedicated/`) |
| `det_n1.pt`, `hd_sc5nb.pt` | `pretrain_detector_pass1.pt`, `pretrain_emitter_pass1.pt` (`config.PRETRAIN_DETECTOR` / `PRETRAIN_EMITTER`) |
| `calib/calib_sc_hd_sc5nbb15_pm.json`, `_pm_bright.json`, `_pm_bright_real.json`, `_pm_autoc1.json`, `calib/calib_p2L.json`, `calib/calib_p2L_b10_real.json` | `calib/template_calib_emitter_pass1_base.json`, `template_calib_emitter_pass1_sim.json`, `template_calib_emitter_pass1_real.json`, `template_calib_emitter_pass1_brightness_noise_fit.json`, `template_calib_emitter_pass2_sim.json`, `template_calib_emitter_pass2_real.json` (`config.TEMPLATE_CAL_*`) |
| `calib/calib_sc_hd_sc5nbb15_pm_autoc.json`, `calib_p2L_movie_b10.json`, `calib_autoc_r05 / r25 / r20.json`, `sigu/sigu_p1_cur.json`, `sigu_p2_cur.json` (unused) | `archive_superseded/previous_default/calib/calib_emitter_pass1_brightness_noise_fit_first.json`, `calib_emitter_pass2_sim_movie.json`, `calib_emitter_pass1_brightness_noise_fit_{0.5nM,2.5nM,20nM}.json`, `sigma_rescale_pass{1,2}.json` |
| `calib/calib_{sc_hd,p2L}_{m1809,m25,g13}{,_real}.json`, `calib/sigu/sigu_p{1,2}_{m1809,m25,g13}.json` (copies of the 2D models' calibrations) | removed; the byte-identical originals are `archive_2d/<set>/calib/` (new names above) |
| `archive_p2dedicated/`, `archive_p2generic/p2det_<g>.pt`, `p2headL_<g>.pt`, `calib/calib_p2L_<g>{,_real}.json`, `calib/sigu/sigu_p2_g2ss.json` | `archive_pass2_dedicated/`, `archive_pass2_generic/<g>/detector_pass2.pt`, `emitter_pass2.pt`, `calib/calib_emitter_pass2_{sim,real}.json`, `calib/sigma_rescale_pass2.json` |
| `archive_superseded/p2_det.pt`, `p2_head_long.pt`, `{det,hd,p2det,p2headL}_psf3dmix.pt`, `p2{det,headL}_psf3dmix_pair.pt`, `p2{det,headL}_gnr80.pt`, `hd_abl1809_{avg,ax}.pt`, `hd_sc7bg.pt` | `archive_superseded/previous_default/{detector,emitter}_pass2.pt`, `psf3dmix/{detector,emitter}_pass{1,2}.pt`, `psf3dmix_pair/…_pass2.pt`, `gnr80/…_pass2.pt`, `ablation_1809_{avg,ax}/emitter_pass1.pt`, `earlier_architecture/emitter_pass1.pt` |
| `processing.gate_dump` front names `p1`, `p2L` (default `--fronts p1,p2L`) | `pass1`, `pass2` |
| process titles `train-p2det` / `train-det` | `train-detector-pass2` / `train-detector` |

## Renamed (2026-11-15)

Old names in the column "before"; the "after" column is the state of 2026-11-15 (renamed again on 2026-11-16 above). Checkpoint files are unchanged (read by format readers); run records, model.json files and
CLI flags use the new names; run-output file names (pass1_<f0>.npz, pass2_est2_<f0>.npz; the tables locs_p*.npz became locs_p*.hdf5 on 2026-11-15) and the gate-dump
column names are unchanged.

| before | after |
|---|---|
| `auto/` (pipeline.py, config.py, order.py, psffit.py, stage_psf.py, train.py, runwrap.py, accept.py, …) | `pipeline/` (localization_pipeline.py, acquisition_config.py, frame_order.py, psf_fit.py, psf_stage.py, training.py, run_with_overrides.py, acceptance.py, …) |
| `pipeline/` (sc_front.py, sc_front_fast.py, pattern_est.py, …), `p2/front.py`, `p2/dump.py` | `processing/` (front.py, front_fast.py, pattern_estimation.py, …, pass2_front.py, gate_dump.py) |
| `det/model.py` DetNet, `det/up2.py` DetUp, `p2/det2.py` DetP2 | `detector/model.py` Detector (+ `detector/select.py`); one trainer `detector/train.py` |
| `hd/sc4.py` SCHead4, `p2/head2.py` SCHeadP2, `hd/sc.py` (network part) | `emitter/model.py` EmitterNet; one trainer `emitter/train.py`; `emitter/heldout.py` |
| `core.py`, `hd/msolve.py`, `hd/msolve_fast.py`, `autoc.py`, `hd/sc.py` (sc_solve, z conditioning) | `solve/pattern.py`, `solve/solve.py` (solve_emitters), `solve/solve_fast.py` (solve_emitters_fast), `solve/autoc.py`, `solve/z_conditioning.py` |
| `p2/illum.py` | `pattern/illumination.py` |
| `hd/realsim.py` RealSim, `hd/data.py` Gen + `p2/data.py` Gen2 + `det/up2.batch`, `det/data.py`, `det/psfpert.py`, `sim/` | `simulate/acquisition.py` AcquisitionSim, `simulate/batches.py` TrainingBatches (emitter_batch, illumination_batch, detector_batch), `simulate/targets.py`, `simulate/emitters.py` (removed 2026-11-15), `simulate/psf_perturbation.py`, `simulate/simulator.py`, `simulate/crlb.py`, `simulate/gate_layouts.py` (simtools.py) |
| `hd/gate*.py`, `eff/crlb_nuis.py`, `eff/simcode_gate.py` | `benchmark/gate*.py`, `benchmark/crlb_nuisance.py`, `benchmark/preprint_gate.py` (the last two removed 2026-11-15, `docs/ARCHIVE.md`) |
| `insitu/`, `evalproto/` (lib.py), `nn/`, `fileio/`, `fastutil.py`, `strict.py` | `calibration/`, `evaluation/` (metrics.py), `layers/`, `io/`, `util/compile.py`, `table.py` |
| `fourpi/realsim4pi.py` RealSim4Pi | `fourpi/acquisition4pi.py` AcquisitionSim4Pi |
| Pass1SC, Pass1SCFast, Pass2P2, Pass2P2Fast, run_chunk_pass2 | Pass1Front, Pass1FrontFast, Pass2Front, Pass2FrontFast, run_chunk_pattern_solve (removed 2026-11-15; front_fast.run_chunk_pattern_solve_fast) |
| trainer modules `simcode.det.up2 train` / `simcode.p2.det2 train` (`--init_n1`) | `python -m simcode.detector.train` (pass 2: `--illum 1 --init <pass-1 detector>`; continuation: `--init <pass-2> --pass1_ref <pass-1>`) |
| trainer modules `simcode.hd.sc4 train` / `prep` / `prep5`, `simcode.p2.head2 train` | `python -m simcode.emitter.train` (pass 2 when `--detector` / `--init` is pass 2), `python -m simcode.emitter.heldout [--set 5]` |
| `simcode-strict` | `simcode-table` |
| model definition keys (models.json, model.json) `det`, `head`, `p2_det`, `p2_head`, `p2_cal_sim`, `p2_cal_real`; `config.DET` / `HEAD` / `P2_DET` / `P2_HEAD` / `CAL_P1_*` / `CAL_P2_*` | `detector`, `emitter`, `detector_pass2`, `emitter_pass2`, `cal_sim_pass2`, `cal_real_pass2`; `config.DETECTOR` / `EMITTER` / `DETECTOR_PASS2` / `EMITTER_PASS2` / `CAL_REAL` / `CAL_SIM` / `CAL_REAL_PASS2` / `CAL_SIM_PASS2` / `CAL_AUTOC*` (files with the old keys are refused; `analysis/maintenance/rename_model_keys.py` rewrites them) |
| run record keys (settings / model files) `det`, `head`, `p2_det`, `p2_head`, `p2_cal`, `sig_u_p1/2`; format `simcode.run_config/1` | `detector`, `emitter`, `detector_pass2`, `emitter_pass2`, `cal_pass2`, `sig_u_pass1/2`; format `simcode.run_config/2` (format-1 records are converted when read, `util/runconfig.upgrade`) |
| `run_uncalibrated --frontend sc / sc2p` (2026-11-15: stages `pass1_solve` / `pass2` instead), `--sc_det`, `--sc_head`, `--sc_cal`, `--p2_det`, `--p2_head`, `--p2_cal`, `--head_batch` | stages `pass1_solve` / `pass2` (`--pass2_mode solve / networks` until 2026-11-15), `--detector`, `--emitter`, `--cal`, `--detector_pass2`, `--emitter_pass2`, `--cal_pass2`, `--emitter_batch` |
| `simcode.benchmark.gate --det`, `--sc`, `--sccal`; `simcode.emitter.train --det` | `--detector`, `--emitter`, `--cal`; `--detector` |
| front constructors `Pass1Front(calib, head=, det=)` (attributes `.head`, `.det`), `QueuedFront(det, heads)`; `emitter.model.head_z` | `Pass1Front(calib, emitter=, detector=)` (`.emitter`, `.detector`), `QueuedFront(detector, emitters)`; `emitter_z` |
| pipeline training stages det, hd, p2det, p2det_c, p2head, p2headL, p2headL2 (`train.steps` / `init` / `extra` keys) | detector, emitter, detector_pass2, detector_pass2_cont, emitter_pass2, emitter_pass2_long, emitter_pass2_final (`extra`: detector, emitter, pass2) |
| batch keys `patt_p2`, `illum_p2` | `patt_input`, `illum_input` |
| checkpoint architecture tag `'sc4n'` | `'emitter_net'` in new checkpoints (`emitter.model.ARCH`; the stored `'sc4n'` is read through `ARCH_TAGS`); new detector checkpoints carry `'detector'` |
| `calibration/fit_insitu2.py`, `benchmark/gate_report_xdet.py` | `calibration/fit_psf_tracks.py` (archived 2026-11-15), `benchmark/gate_report_detector_matched.py` |
| output paths `<out>/sc4/heldout{4,5}.pt`, `<out>/p2/fe`, `<out>/p2/long` | `<out>/emitter_heldout/heldout{4,5}.pt`, `<out>/gate_dumps`, `<out>/pass2_gate/long` |

## 4Pi refactor (branch fourpi, 2026-11-15)

One LocalizationPipeline for every acquisition type: the set-up (config key `setup`) supplies the stages and what differs.
No aliases remain; checkpoints, PSF states and registration maps are read unchanged (the 4Pi tables, every network's
outputs and the seeded simulator batches are bit-identical to the code before).

| before | now |
|---|---|
| `fourpi/run.py` (CLI `--cell MT --files 5 --head v10`) | the pipeline: `python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --work <4Pi work folder> --data <raw movies> --only fourpi_MT` (stages registration, process, report); worker `processing/fourpi.py` |
| `fourpi/setup.py`, `fourpi/datasets.json` (P4, DATA4, LAYOUT, GAIN, PX_NM, P, XLIM, DELTA, file lists) | `pipeline/configs/fourpi.py` (acquisitions fourpi_MT, fourpi_NPC) read through `setups/fourpi.py` (`acquisition(name, work, data)`, `FourPiSetup`); the 4Pi work folder (P4) and the raw-movie folder are arguments (`--work`, `--data`; 2026-11-16, was `config.root('fourpi')` / `data4pi`) |
| `fourpi/front.py` Front(cell), PolyMap, feats, blur, nn_dist | `channels/front.py` ChannelFront (C channels), `channels/registration.py` PolyMap (+ feats_norm), `FourPiSetup.front(c, fi)` |
| `DcimgMovie(path, layout=dict)`, `io.movie.crops` | `channels.layout.ChannelReader(DcimgMovie(path), ChannelLayout)`, `ChannelLayout.split` |
| `fourpi/insitu.py` | `processing/fourpi_pattern_phase.py` |
| `fourpi/solve4pi.py` Solve4Pi, `fourpi/solvek4pi.py` SolveK | `solve/fourpi.py` FourPiSolve (the T / C solve = windows ('flat', 'gauss' 1.0)) |
| `fourpi/head4pi_w.py` Head4PiW, nbr_map, load_v10; `fourpi/head4pi.py` Head4Pi; `fourpi/windows4pi.py` WINS_K, DEFAULT; `fourpi/loadhead.py` load_head; `fourpi/linread4pi.py`; `fourpi/solveloss4pi.py` | `emitter/fourpi.py` FourPiEmitterNet, neighbour_map, init_from_v10; FourPiPrior; WINDOWS_K, WINDOWS_TC; load; linear_readout, border_bg; solve_loss |
| `fourpi/sim4pi.py` Sim4Pi, `fourpi/acquisition4pi.py` AcquisitionSim4Pi, RegMap | `simulate/fourpi_patches.py` FourPiPatchSim, `simulate/multichannel.py` MultiChannelAcquisitionSim, TorchMap |
| `fourpi/zdrift.py` zdrift_struct | `util/drift.py` zdrift_struct |
| `fourpi/lm.py` | `analysis/fourpi_classical/pixel_lm.py` (classical baselines only) |
| run output `<P4>/net/locs_<cell>_<file>_<head><tag>.npz` | `<out>/process/locs_<file><tag>.npz` (same arrays) |
