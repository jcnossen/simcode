# SIMCODE v2: package layout

One line per directory and file of the `simcode` package (reorganised 2026-11-15, see "Renamed" at the end; the 2026-11
cleanup before it: `docs/CLEANUP_LOG.md`). The docstring at the top of each file has the details. Method: `docs/METHOD.md`.
Evaluation: `docs/EVALUATION.md`. Working with the results: `docs/USAGE.md`.

The repository holds two comparisons and what they need:
- **SIMCODE preprint vs SIMCODE v2** on DMD-SIMFLUX DNA-PAINT movies: the v2 pipeline below with the PSF-matched models, the
  preprint's model as a baseline (`analysis/baselines/run_simcode*.py`), and the paper's other baselines (SMLM, SIMFLUX, DECODE).
- **4Pi-SIMFLUX (classical, as published) vs 4Pi-SIMCODE** (network): `fourpi/` below and `analysis/fourpi_classical/`.

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
  run.py                      user entry point: python -m simcode.run --movie … --calib … --model m1809 --out … (or --dataset 1809_2); all stages, finished ones skipped
  config.py                   all paths (models, data, outputs, 4Pi work directory and data, external DME) from SIMCODE_* variables / simcode.json / defaults; model_set(name or model folder)
  models.json                 the shipped models (m1809, m25, g13), the default model, the paper's datasets (_production)
  dataset.py                  Dataset = torch_psf.Dataset + SIMCODE's loaders (from_run, load_npz, from_decode, load_npy / from_smlmtorch) and the paper's selection select() / mask()
  table.py                    localization table of a run directory (with the model's sigma rescale), the paper's selection select(), subset files; CLI simcode-table
  detector/
    __init__.py               package doc
    model.py                  Detector: the U-Net detector of pass 1 and pass 2 (illum=True: + illumination inputs, zero-initialised from pass 1); load() of every detector checkpoint
    select.py                 candidate selection: sub-pixel greedy NMS + top-k (select_subpixel), select_points
    train.py                  trainer of both passes: python -m simcode.detector.train (pass 2: --illum 1 or a pass-2 --init); focal loss + offset NLL with nn-bin weights
  emitter/
    __init__.py               package doc
    model.py                  EmitterNet: the per-candidate network of pass 1 and pass 2 (I_t with full covariance, bg_t, 3D position prior; illum=True: + illumination patches); neighbour inputs, PSF template, detect_candidates / apply_to_candidates, load() of every checkpoint (2D ones with zero z rows)
    train.py                  trainer of both passes: python -m simcode.emitter.train (pass 2: a pass-2 --det or --init); intensity / prior / z NLL, anti-shrinkage and slope terms
    heldout.py                held-out gate-type patches of the pass-1 trainer (python -m simcode.emitter.heldout [--set 5]) and their metrics
  solve/
    __init__.py               package doc
    solve.py                  the multi-start explicit pattern solve (reference implementation) and solve_emitters: the solve of the emitter-network outputs with a model calibration
    solve_fast.py             the same solve, fast (production; solve_emitters_fast, optional compiled LM step)
    pattern.py                pattern geometry of the solve (illumination at a position, angle index, fringe shifts, kinetics starts, chi2 p-value)
    z_conditioning.py         z after the solve: the network's 3D prior conditioned on the solved x, y; pass-2 rows linked to pass-1 candidates for their z
    autoc.py                  opt-in automatic noise scale (s, c, c_a) of the network intensities from the solve's own residuals (analysis only)
  pattern/
    __init__.py               package doc
    illumination.py           illumination input maps of a pattern (cos / sin per frame) and perturbed pattern estimates (training error models v1 / real / realw)
  processing/
    __init__.py               package doc
    run_uncalibrated.py       expert entry point with all flags; run.py calls its stages; chunks are the units of io/work.py
    common.py                 read plan of the chunks, pass-1 chunk rows, camera conversion, calibration / PSF loading, pattern_frames, CAND_PMIN
    front.py                  reference pass 1 (Pass1Front) and the pattern solve of its rows (run_chunk_pattern_solve); test oracle
    pass2_front.py            reference pass 2 with the pattern-aware networks (Pass2Front); test oracle
    front_fast.py             the pipeline's code path: queued fixed-size batches for pass 1 / pass 2 (Pass1FrontFast, Pass2FrontFast) and the chunk-wide pattern solve
    pattern_estimation.py     the illumination pattern from the pass-1 rows (k, per-step phases, depth, relative intensities, per-angle phase drift)
    pattern_field.py          per-cell phase map and smooth depth field on top of the global pattern (analysis only)
    compare_pattern.py        an estimated pattern against a reference pattern, in rad and nm
    dedup.py                  output modes for overlapping windows (disjoint windows, merged chains)
    gate_dump.py              raw network outputs of pass 1 / pass 2 on the gate scenes (input of a model's calibration fit): python -m simcode.processing.gate_dump
  simulate/
    __init__.py               package doc
    acquisition.py            AcquisitionSim: the experiment-matched acquisition simulator (measured PSF with jitter, camera maps, structured / fluctuating bg, filaments, kinetics, SIMCODE or generic patterns)
    batches.py                TrainingBatches: training batches of the detector and the emitter network, pass 1 and pass 2 (emitter_batch, detector_batch, illumination_batch)
    targets.py                training targets and weights from the ground truth: detector cell targets, nn-bin weights, candidate-GT matching, emitter-network targets
    simulator.py              Simulator: the calibrated physical model (PSF LUT rendering, calibration json, camera)
    crlb.py                   scene generator and multi-emitter CRLB of the benchmark scenes
    gate_layouts.py           emitter layouts of the gate scenes (grids, pairs, Poisson scenes, wide brightness) and the pattern-estimate error model
    emitters.py               emitter lists with DNA-PAINT kinetics on the crlb scene model (the first detector training data)
    psf_perturbation.py       bank of perturbed PSF LUTs (width, aberrations, focus) and camera errors
  benchmark/
    __init__.py               package doc
    gate.py                   the gate suite: fixed simulated scenes (isolated, pairs, realistic) through a front end: python -m simcode.benchmark.gate
    gate_report.py            detection, precision and intensity metrics of gate outputs
    gate_report_xdet.py       gate report with precision matched on the detector position
    crlb_nuisance.py          multi-emitter CRLB with intensity nuisance parameters for the gate scenes
    preprint_gate.py          the preprint's SIMCODE model on the gate scenes (comparison)
  calibration/
    __init__.py               default input / output locations of the in-situ calibration
    README.md                 the in-situ calibration recipe
    extract_tracks.py         isolated emitter tracks (raw regions over consecutive frames)
    fit_insitu2.py            stage A: PSF from the tracks (intensities and bg in closed form)
    fit_pattern.py            stage B: illumination pattern from the track intensities
    fit_psf_sums.py           PSF refit on bright multi-frame sums
    camera_gain.py            camera gain and read noise from the movie
    camera_maps.py            per-pixel offset and read-noise maps from the movie
    psf_heldout.py            held-out evaluation of a fitted PSF
    psf_compare.py            several PSF fits compared on the same samples
  pipeline/
    __init__.py               the automatic per-acquisition pipeline (LocalizationPipeline, ConfigDict)
    __main__.py               python -m simcode.pipeline <config>
    localization_pipeline.py  LocalizationPipeline: the stages in order (order, camera, spots, cammaps, tracks, beads, psf, train, process, report), done markers, checks
    acquisition_config.py     the acquisition config: defaults, yaml / json / Python files, validation, frame-order helpers
    configs/                  example acquisitions: 1809_2.py / .yaml, nr80.py / .yaml, example_setup.py
    frame_order.py            stage order: blocked-order cache of a movie with another frame order
    camera.py                 stage camera: gain, read noise, offset from the variance-vs-level fit
    spots.py                  stage spots: spot finding and Gaussian fits for the tracks
    psf_fit.py                in-situ PSF fit of stage psf (fit_insitu2 as a function; spline-table fit)
    psf_stage.py              stage psf: multi-start in-situ PSF, line blur, photoselection, held-out, calibration files
    training.py               stage train: a PSF-matched model with the recipe of models/README.md, gate dumps, calibrations, acceptance, model.json
    acceptance.py             stage train acceptance on the gate dumps (sigma calibration, pass-2 vs pass-1 RMS / CRLB)
    process.py                stage process: simcode.run on the movie with the dataset's calibration and model
    run_with_overrides.py     a module run with the pipeline's instrument overrides applied in memory (AUTO_WRAP)
    report.py                 stage report: summary page of an acquisition
  io/
    __init__.py               package doc
    movie.py                  one reader interface with TIFF / dcimg / array back ends (open_movie), Prefetch
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
  fourpi/                     4Pi-SIMFLUX (separate data type; does not touch the 2D pipeline)
    __init__.py               package marker
    run.py                    4Pi-SIMCODE on raw dcimg files: python -m simcode.fourpi.run --cell NPC --files 10 --head v10
    setup.py                  the 4Pi set-up: work / data directory, channel layout, gain, pixel size, patch size, phase offsets, file lists
    datasets.json             the 4Pi datasets (cells, files, data root)
    front.py                  front end shared with the classical pipeline: registration (PolyMap), camera conversion, 4-channel sum, detection, patches
    insitu.py                 in-situ estimates per file: SIM pattern, per-candidate pattern phases, 4Pi phase field
    head4pi_w.py              the learned-window head (v10) with a frozen Head4Pi prior
    head4pi.py                per-candidate head on the 24 channel patches (the prior of head4pi_w; checkpoint v3L)
    loadhead.py               loader of the 4Pi head checkpoints
    windows4pi.py             the canonical read-out windows of the head and the solve
    linread4pi.py             fixed linear window read-out with analytic covariance (reference of the learned head)
    solve4pi.py               explicit solve: lateral position from the pattern, z from the 4Pi phase
    solvek4pi.py              the same solve on K window read-outs per (frame, channel) (the pipeline's)
    solveloss4pi.py           solve-aware training loss of the learned-window head
    psf4pi.py                 coherent two-arm vectorial 4Pi PSF with four interference channels
    lut4pi.py                 spline lookup table of that PSF, 4Pi phase analytic
    sim4pi.py                 simulator of candidate patches (6 frames × 4 channels)
    acquisition4pi.py         AcquisitionSim4Pi: full-scene 4Pi simulator for the detector training
    lm.py                     batched Poisson fit and CRLB (classical 4Pi baseline)
    zdrift.py                 sample axial drift from the structure itself
  util/
    __init__.py               package marker
    ckpt.py                   atomic checkpoint save with a disk guard (every trainer)
    compile.py                optional torch.compile support of the fast paths
    drift.py                  lateral drift of a table with the external DME v2 package (drift_of_table)
    runconfig.py              the run record run_config.json (+ yaml): settings, hashes, code / environment, outcome; re-run, verify, backfill
    cfgdict.py                ConfigDict: nested settings with attribute access, deep merge, freeze, yaml / json / Python loading
    proctitle.py              recognizable process names in nvidia-smi / ps
    spatial_hash.py           GPU / CPU neighbour search (pattern estimation, autoc, pass-1 linking)
    neighbours.py             small nearest-neighbour helper
```

## Checkpoints

Network checkpoints are plain state dicts with the training arguments (`{'model', 'args', 'step'}`; no pickled classes,
`tests/check_checkpoints.py`). The readers map them to the two classes:
- `detector.model.load`: pass 1 or pass 2 from the state dict (illumination weights `cin_il`), channel widths from the args.
- `emitter.model.load`: the architecture tag `args['arch'] = 'sc4n'` (pass-1 checkpoints; pass-2 ones carry none), pass 1 or
  pass 2 from the state dict (`cin_il`), 2D checkpoints (before 2026-11-14) with zero z rows (`net.z3d` False).
Every checkpoint of `models/` (models.json models, model folders, `archive_*`, the other weights) and the 4Pi detector /
heads give bit-identical outputs with the code before and after the 2026-11-15 reorganisation
(`tests/check_model_outputs.py`).

## Outside the package

- `analysis/`: scripts that use the package; not needed to run the method.
  - `baselines/`: SMLM, SIMFLUX, DECODE and the preprint's SIMCODE on simulated and real movies; `dme_drift.py` (CLI of `util/drift.py`).
  - `paper/`: figure scripts of the experimental figures and the fair FRC protocol (`README.md` there).
  - `papersim/`: the paper simulations (generators, all methods, metrics, figures); `reproduce.sh` lists the commands.
  - `fourpi/`: 4Pi calibration chain, the v10 trainers and gates, post-processing, real-data comparisons, the simulated showcase.
  - `fourpi_classical/`: the classical 4Pi-SIMFLUX pipeline (published-style detection and fits, pixel MLE).
  - `calib/`, `psfmatch/`, `psf3d/`, `psf3d_nets/sigcal.py`: calibration of a model (sigma tables, sigma rescale, PSF files).
  - `gate/`, `figures/`, `evalproto/`, `hd_compare/`, `simreal/`, `tput/`, `dme/`, `psfshape/`, `z3d/`, `p2generic/`, `nanoruler/`: benchmarks and development studies; `results/` holds text snapshots of their outputs.
- `models/`: calibrations (`calib/`, `insitu/`), model folders (`model.json`, calibrations, README); network weights are not in git (`models/README.md`).
- `tests/`: `test_fast_paths.py` (GPU: fast paths against the reference implementations), `test_z3d.py` (3D prior), `test_fileio.py`, `test_tables.py`, `test_dataset.py`, `test_cfgdict.py`, `test_runconfig.py`, `test_solve4pi_oracle.py`; `check_imports.py` (static check of the tree), `check_checkpoints.py`, `check_model_outputs.py` (outputs of every checkpoint on a fixed batch), `compare_npz.py` / `compare_ckpt.py` (bit-identity of outputs).
- `docs/`: this file, method, evaluation, results, usage, dataset format, automatic pipeline, PSF polarization note, the cleanup plan and log.

## Renamed (2026-11-15)

No aliases or old names remain; checkpoints, model folders, output files and CLI flags of `simcode.run` are unchanged.

| before | now |
|---|---|
| `auto/` (pipeline.py, config.py, order.py, psffit.py, stage_psf.py, train.py, runwrap.py, accept.py, …) | `pipeline/` (localization_pipeline.py, acquisition_config.py, frame_order.py, psf_fit.py, psf_stage.py, training.py, run_with_overrides.py, acceptance.py, …) |
| `pipeline/` (sc_front.py, sc_front_fast.py, pattern_est.py, …), `p2/front.py`, `p2/dump.py` | `processing/` (front.py, front_fast.py, pattern_estimation.py, …, pass2_front.py, gate_dump.py) |
| `det/model.py` DetNet, `det/up2.py` DetUp, `p2/det2.py` DetP2 | `detector/model.py` Detector (+ `detector/select.py`); one trainer `detector/train.py` |
| `hd/sc4.py` SCHead4, `p2/head2.py` SCHeadP2, `hd/sc.py` (network part) | `emitter/model.py` EmitterNet; one trainer `emitter/train.py`; `emitter/heldout.py` |
| `core.py`, `hd/msolve.py`, `hd/msolve_fast.py`, `autoc.py`, `hd/sc.py` (sc_solve, z conditioning) | `solve/pattern.py`, `solve/solve.py` (solve_emitters), `solve/solve_fast.py` (solve_emitters_fast), `solve/autoc.py`, `solve/z_conditioning.py` |
| `p2/illum.py` | `pattern/illumination.py` |
| `hd/realsim.py` RealSim, `hd/data.py` Gen + `p2/data.py` Gen2 + `det/up2.batch`, `det/data.py`, `det/psfpert.py`, `sim/` | `simulate/acquisition.py` AcquisitionSim, `simulate/batches.py` TrainingBatches (emitter_batch, illumination_batch, detector_batch), `simulate/targets.py`, `simulate/emitters.py`, `simulate/psf_perturbation.py`, `simulate/simulator.py`, `simulate/crlb.py`, `simulate/gate_layouts.py` (simtools.py) |
| `hd/gate*.py`, `eff/crlb_nuis.py`, `eff/simcode_gate.py` | `benchmark/gate*.py`, `benchmark/crlb_nuisance.py`, `benchmark/preprint_gate.py` |
| `insitu/`, `evalproto/` (lib.py), `nn/`, `fileio/`, `fastutil.py`, `strict.py` | `calibration/`, `evaluation/` (metrics.py), `layers/`, `io/`, `util/compile.py`, `table.py` |
| `fourpi/realsim4pi.py` RealSim4Pi | `fourpi/acquisition4pi.py` AcquisitionSim4Pi |
| Pass1SC, Pass1SCFast, Pass2P2, Pass2P2Fast, run_chunk_pass2 | Pass1Front, Pass1FrontFast, Pass2Front, Pass2FrontFast, run_chunk_pattern_solve |
| trainer modules `simcode.det.up2 train` / `simcode.p2.det2 train` (`--init_n1`) | `python -m simcode.detector.train` (pass 2: `--illum 1 --init <pass-1 detector>`; continuation: `--init <pass-2> --pass1_ref <pass-1>`) |
| trainer modules `simcode.hd.sc4 train` / `prep` / `prep5`, `simcode.p2.head2 train` | `python -m simcode.emitter.train` (pass 2 when `--det` / `--init` is pass 2), `python -m simcode.emitter.heldout [--set 5]` |
| `simcode-strict` | `simcode-table` |
