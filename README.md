# simcode

Using SIMCODE on your own microscope: `docs/QUICKSTART.md`.

SIMCODE v2 localizes emitters in DMD-SIMFLUX DNA-PAINT data, where the illumination pattern is modulated and stepped every
frame. It is a per-window deep-learning method:

- **The networks give the per-frame intensities.**
  - A pattern-free detector finds candidates in each 6-frame window, i.e. one full pattern cycle of 2 angles × 3 phases.
  - A neighbour-aware per-candidate head outputs, per candidate:
    - the 6 per-frame intensities with their full covariance
    - the per-frame background
    - a pattern-free position prior
- **The pattern-encoded position is then solved explicitly from those intensities.**
  - The illumination pattern, with a per-orientation intensity map over the field of view, is estimated from the data.
  - The intensity-domain solve has fringe hypotheses and on/off kinetics, both averaged by their posterior weights, the
    network's full intensity covariance plus one brightness-noise term c (c² μ² per frame) estimated from each dataset
    with the pattern, and the posterior σ.
- **Pass 2** reruns pattern-aware networks with the estimated pattern as input, then the same solve. The pass-2 solve is
  the result table; the networks' own outputs (position and σ from the emitter network's joint covariance) are written too.
- **No pixel-level linear intensity fit** is used anywhere.

More in the docs:

- `docs/METHOD.md`: method
- `docs/EVALUATION.md`: evaluation protocol
- `docs/USAGE.md`: working with the results (`simcode.dataset.Dataset` = `torch_psf.Dataset` + SIMCODE's loaders and selection: load, select, drift, render, FRC, hdf5 / Picasso / ThunderSTORM files); `docs/DATASET_FORMAT.md`: the field layout
- `docs/LAYOUT.md`: the package tree with one line per directory and file (and what was renamed on 2026-11-15)
- `docs/PSF_POLARIZATION.md`: the in-situ PSF model and its polarization terms

This branch (`master`) holds the package, its tests and two example models (`m1809`, `g13`). The branch `paper_figures` is
`master` plus the paper material: the analysis scripts (the baselines SMLM / SIMFLUX / DECODE under `analysis/baselines`,
the paper figures and simulations, the classical 4Pi-SIMFLUX pipeline `analysis/fourpi_classical`), the model `m25` and the
in-situ calibrations of all paper movies, the development notes in `docs/` and the text sources of the paper figures
(`figures/`).

## Installation

```bash
git clone https://github.com/qnano/simcode && cd simcode
pip install -e .                                     # numpy, scipy, torch, torch-psf, dme, ... (pyproject.toml)
python -m simcode.download_models                    # network weights of the example models m1809, g13 (GitHub release models-v2)
python -m simcode.download_models --set pretrain     # only to train a new model: the pretraining networks
```

Network weights (`*.pt`) are not in git: `simcode.download_models` fetches the release assets
(`simcode_model_m1809.zip`, `simcode_model_g13.zip`, `simcode_pretrain.zip`), checks their sha256 and unpacks them into
`models/` (`--dest <dir>` for another folder, `--base-url <url>` for a mirror or a local copy, `file://...`).
`models/README.md` lists the files of each model. Tested with Python 3.11 and PyTorch 2.13 + CUDA.

Dependencies:

- numpy, scipy, tifffile, torch (CUDA GPU required); matplotlib for the calibration / figure scripts; h5py for
  the table files (`locs_p*.hdf5`) and for reading Picasso hdf5 locs in the in-situ calibration; triton optional (`torch.compile`)
- torch-psf (MIT, https://github.com/jcnossen/torch-psf, pinned in pyproject.toml): the PSF models (vectorial PSF,
  convolutional 3D spline, Gaussian), the calibration json reader, the bead and in-situ PSF fits (variable projection)
  and the TIFF movie reader. `simcode/psf/__init__.py` is the only module that imports it (the rest of SIMCODE imports from `simcode.psf`) and holds the SIMCODE-specific glue (per-pattern-angle PSF terms, the
  spline settings of the solve).
- DME v2 drift: the `dme` package (`github.com/jcnossen/drift-estimation`), a pinned git dependency in `pyproject.toml`
  (installed by `pip install -e .`); the installed package is the only one used, through `simcode/util/drift.py`

The baselines of the paper (`analysis/baselines`, on the `paper_figures` branch) need more: the preprint's SIMCODE needs
`smlmtorch` and its paper models (`--smlmtorch_models`), DECODE the `decode` fork.

## Run

```bash
export CUDA_VISIBLE_DEVICES=0
python -m simcode.run --movie /mnt/data/dmdsf/bn_dnapaint/1809/cover1/2/simflux_5.5ms_500mW.tif \
    --calib m1809/calib_1809_2_psf_v2.psf.zip --model m1809 --crop 5,5,110 --out out_1809_2
python -m simcode.run --dataset 1809_2 --data-root /mnt/data/dmdsf/bn_dnapaint --out out_1809_2   # the same: a paper dataset
```

What every run does (= the paper's settings; the options that switched these were removed on 2026-11-14):

- the model (`--model`, required unless `--dataset` / `--config` gives it): a name of `simcode/models.json` (`m1809`,
  `g13`) or the path of a model folder with a `model.json` (as written by the automatic pipeline's training
  stage / `simcode.pipeline.training.export_model`); one model = pass-1 detector + head, pass-2 detector + head and their solve
  calibrations. There are no single-file overrides: other files = another model folder
- pass 1, pattern estimation (with the per-orientation intensity map and the dataset's brightness noise c), the pattern
  solve, and pass 2 (`--intensity_map` is deprecated: the map is always estimated, `--intensity_map 0` is ignored with a
  warning; the per-orientation brightness noise c_a was removed on 2026-11-19, an older `pattern_est.pt` with `cmult_ang`
  is read with c only and a warning)
- the solve calibrations of the model (`cal_sim`, `cal_sim_pass2`: prior_off from simulated gate scenes, start box, posterior
  temperature; used as they are for real and simulated movies) + the brightness noise estimated from the run's own data
  (the fixed c = c_a = 0.1, the σ factors, the real-data σ multiplier and `--cal_kind` were removed on 2026-11-17)
- the kinetic model: every row is fitted with and without on/off kinetics, the two fits are averaged by their posterior
  weights (BIC penalty 2 ln 6 for the 2 kinetic parameters)
- stored-candidate cut p ≥ 0.3 (`simcode/processing/common.CAND_PMIN`)
- the fast code path, with a compiled solve step when available
- the output region: `--crop x0,y0,size` (px); default the whole frame (square sensors; else the largest top-left square);
  the paper datasets carry their crop (5,5,110) in `models.json` `_production`
- four tables in the torch_psf Dataset format (CSV / Picasso: exports of the loaded Dataset): `locs_p1_net`, `locs_p1_solve`,
  `locs_p2_net`, `locs_p2_solve` (**the result**: the pass-2 pattern solve's position and posterior σ; `Dataset.from_run(<out>)`;
  2026-11-17 to 2026-11-19 the default was `locs_p2_net`, now `Dataset.from_run(<out>, kind='net')`)
- the selection (since 2026-11-13) is applied when a table is loaded, nothing is stored for it: a global threshold
  on the reported precision, max(σx, σy) ≤ 12 nm on the rows with score p ≥ 0.63 inside the crop
  (`Dataset.from_run(<out>).select()` = `simcode.table.select(table, max_precision_nm=12, min_score=0.63, in_region=True)`;
  another threshold: `select(14)`); the column `sel` (phase-calibrated detection selection) is the recommended full output
  for renders and drift. (The per-tile rule "u ≤ 0.45" and the strict product were removed on 2026-11-14.)

`--dataset <name>` takes movie, calibration, model and crop of a paper dataset from the `_production` entry of `models.json`;
its movie path is relative to `--data-root` (the folder of the DMD-SIMFLUX DNA-PAINT data).
The options removed on 2026-11-14 (`--pass2`, `--autoc`, `--cand_pmin`, `--fast`, `--compile`, `--netset`, `--det` ...,
`--u_max`, `--strict_q`), on 2026-11-15 (`--format`, `--subset`, `--sigma_max`, `--prefetch` (always on), `--sig_u`) and
on 2026-11-17 (`--cal_kind`; the model key `sig_u`) no longer exist.

Several GPUs: start the same command once per GPU with `--claim 1`; the copies share the chunks of pass 1 and pass 2:

```bash
python -m simcode.io.launch --gpus 0,1 --log out_1809_2/run -- python -m simcode.run --dataset 1809_2 --data-root /mnt/data/dmdsf/bn_dnapaint \
    --out out_1809_2 --claim 1
```

A run can be killed at any time: chunk files are written atomically, a restart continues with the missing chunks, and a
chunk claimed by a process that no longer exists is taken over. All movies of the lab server are on one hard disk, so
two workers on one TIFF movie are not necessarily faster than one (not measured).

Other options:

- `--chunk_list 12000` processes only the chunk of frames 12000–18000.
- `--drift drift.npy` (DME output, [F, 2] px, or a DME v2 result npz; `simcode/util/drift.py`) adds the
  drift-corrected columns `x_dc, y_dc`; the selection region then uses them.

Outputs in `--out`:

| file | content |
|---|---|
| `locs_p2_solve.hdf5` | **the result**: the pass-2 pattern solve (one row per 6-frame window of an emitter; windows start every 3 frames), torch_psf Dataset format (`docs/DATASET_FORMAT.md`), see below |
| `locs_p2_net.hdf5`, `locs_p1_net.hdf5`, `locs_p1_solve.hdf5` | the pass-2 emitter network output (same rows as `locs_p2_solve`), the pass-1 emitter network and the pass-1 solve (same rows as each other) |
| `locs_p*_info.json` | counts, crop, the size of the selection (`n_main`) and its median σ, column list, which table it is |
| `run_config.json`, `run_config.yaml` | the complete record of the run: fully resolved settings, file hashes, code and environment, pattern and selection outcome (section "Run record") |
| `summary.json` | settings, model, per-stage timing |
| `command.txt` | plain log of the commands run on the directory |
| `pass1_<f0>.npz`, `pattern_est.pt/.json` (pattern + intensity map + brightness noise), `pass2_est_<f0>.npz`, `pass2_est2_<f0>.npz`, `runtime.json`, `log.txt` | stage outputs; a re-run skips chunks whose files exist (`--redo 1` recomputes). Each chunk also leaves a small record `pass1_<f0>.json` … (timing, counts); `runtime.json` is merged from them |

Table columns (as built by `simcode.table`; in the file as Dataset fields: `pos` = (x, y), `crlb` = (sx, sy), `score` = p,
`intensities` = I, the others under their names; `Dataset.columns()` gives them back under these names):

- `frame` (window start), `t` (time centroid)
- `x, y` (sensor px, not drift corrected), `sx, sy` (σ, px): net tables = the emitter network's position (prior_off
  removed) and its sd; solve tables = the solve's posterior mean and sd; no factor in either
- `z, sz, cov_xz, cov_yz, z_prior`: z (px) of the network's 3D prior conditioned on the table's x, y (pass 2: of the linked
  pass-1 candidate; NaN without one)
- `photons`, `I[6]` (network per-frame intensities)
- `p` (detector), `x_prior, y_prior` (raw network position); solve tables also `q_dof`, `kin` (posterior weight of the
  kinetic model), `dnll`
- `good` (p ≥ 0.63), `sel` (phase-calibrated selection: the recommended full output for renders and drift), `disjoint`
  (frame % 6 == 0), pass 2: `p1_index` (row of the linked pass-1 candidate in the pass-1 tables, -1 none)
- the metadata `pixel_nm`, `crop`, `tag`, `kind`, `table`, `info`; with `--drift`: `x_dc, y_dc`
- runs made before 2026-11-17 have `locs_p1` / `locs_p2` (the solve, with the removed σ rescale; before 2026-11-15 `.npz`):
  `Dataset.from_run(<out>, table='locs_p2')` reads them; their chunk files lack the network columns, so their four tables
  need a re-run

The tables can be rebuilt at any time with
`python -m simcode.table <out> [--table locs_p2_solve | ... | all] [--drift drift.npy] [--crop x0,y0,size]`.
The crop of the selection is the output region of the run (`--crop` of the run; read from its chunk files).

The stage-level entry point is `python -m simcode.processing.run_uncalibrated` (`--stages pass1,pattern,pass1_solve,pass2`,
the networks and calibrations, batch sizes).

Speed: on an RTX 3090, a 6000-frame 0.5 nM chunk takes about 35 s for pass 1 + pattern and 11 s for the solve. Pass 2
adds about 25 s per chunk.

## Run record (`run_config.json`)

Every output directory says how it was calculated. `python -m simcode.run` writes `<out>/run_config.json` at the start
(status `running`) and completes it at the end (`done`; `failed` with the error text); `run_config.yaml` is the same content
for reading. It holds (`simcode/util/runconfig.py`):

| entry | content |
|---|---|
| `settings` | every argument after the defaults: movie, calibration, model and its files, crop, chunking, frames; files as absolute paths |
| `fixed` | what is not an option: pass 2, 6-frame windows every 3 frames, stored-candidate cut 0.3, p ≥ 0.63, the intensity map and the brightness-noise estimate, the kinetic model averaging, the four tables and which is the result, code path, compile mode, batch sizes, margin, pattern search range, default crop |
| `model` | name and definition; per file the absolute path, size, sha256; for a network the trainer arguments and step stored in its checkpoint; for a calibration its scalar parameters (`prior_off`, ...) |
| `calib` | in-situ PSF / camera calibration: path, sha256, camera maps + sha256, the PSF parameters (NA, wavelength, indices, pixel size, focus, Zernike coefficients, extra terms) and the camera gain / offset / read noise |
| `movie` | path, parts with size and sha256, frame count and size, frames and chunks processed, frame order / pattern scheme |
| `code`, `environment` | package version, git commit, dirty flag with the modified files, DME version and commit; python / numpy / torch / CUDA versions, GPU, host (no user name or e-mail) |
| `invocations` | one entry per process: command line, GPU, start / end, timing (a resumed run or one copy per GPU with `--claim 1` adds entries) |
| `pattern`, `tables`, `selection`, `counts`, `timing` | the pattern estimate (pitch, angle, depth, phases, drift range, intensity map) and the brightness noise c used by every solve, the four tables (which is the result), the counts per selection rule and table, detections, per-stage time |
| `outputs` | every table / subset file with its size and `arrays_sha256` (hash of the array contents; the file hash of an `.npz` changes with its zip time stamps) |
| `steps` | later work on the directory: `python -m simcode.table` appends its settings, input hashes, counts and outputs here and marks the files it replaced (`written_by: steps[i]`) |

Re-run with exactly the settings of a record (the model of the record is registered from its definition; a record made
with single network files re-runs with them as model `<name>@record`; arguments after `--config` override single settings):

```bash
python -m simcode.run --config out_1809_2/run_config.json --out out_1809_2_again
```

A warning is printed (and stored under `rerun_of.differences`) for every input file whose sha256 differs from the record,
for a different git commit or modified working tree, and for different torch / CUDA / numpy versions or fixed values. On the
same code and GPU the re-run is bit-identical (chunk files, pattern, tables; `python tests/test_runconfig.py --gpu`, 2 chunks
of 1809_2). If the settings or input files of a directory change between invocations, the earlier ones are kept under
`earlier_settings` with a warning, because existing chunk files are reused unless `--redo 1`.

```bash
python -m simcode.util.runconfig show <dir>       # one-page summary
python -m simcode.util.runconfig verify <dir>     # recorded file hashes / commit / versions against the present ones
```

`command.txt` stays a plain log (one line per `simcode.run` / `simcode.table` command); `summary.json`, `runtime.json` and
`locs_p*_info.json` are unchanged. The automatic pipeline (`simcode.pipeline`) writes the same kind of record per
acquisition and per stage, and the training recipe with file hashes into the model json.

## Paths

There are no path settings: no config file, no environment variables. Every tool takes its paths on the command line
and every library function as arguments:

| what | given as | default |
|---|---|---|
| movie | `--movie <file>` (or `--dataset <name> --data-root <dir>`) | none |
| outputs | `--out <dir>` | required for `simcode.run`; `./simcode_out/...` for the benchmark tools |
| model | `--model <name or model folder>` | none for `simcode.run` / the pipeline |
| networks / calibrations | a path, or a name in the repository's `models/` folder (also `models/calib/`, `models/insitu/`) | `<repo>/models` (network weights: `python -m simcode.download_models`) |
| 4Pi work folder, raw 4Pi movies | `--work <dir>`, `--data <dir>` (4Pi pipeline configs, 4Pi analysis scripts) | none |
| DME | the installed `dme` package | |

```bash
python -m simcode.run --movie <movie.tif> --calib <calib or name in models/> --model m1809 --out <dir>
python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --work <4Pi work folder> --data <raw 4Pi movies> --only fourpi_MT
```
`--calib m1809/calib_1809_2_psf_v2.psf.zip` works from anywhere (names are looked up in `<repo>/models`). The path
configuration of earlier versions (`simcode.json`, `~/.config/simcode.json`, `SIMCODE_CONFIG`, `SIMCODE_MODELS`,
`SIMCODE_DATA`, `SIMCODE_OUT`, `SIMCODE_4PI`, `SIMCODE_4PI_DATA`, `DME_PATH`) was removed on 2026-11-16 and is not read.

## Models and calibrations (`models/`)

Calibrations (json, psf.zip, camera maps) are in git; network weights are not: they are release assets of the GitHub
repository, fetched with `python -m simcode.download_models` ("Installation" above; `models/README.md`: the models, their
files with checksums, the recipe of a model). The models are defined in `simcode/models.json`.

## 4Pi

```bash
python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --work <4Pi work folder> --data <raw 4Pi movies> \
    --only fourpi_MT --gpus 0,1                                                         # 4Pi-SIMCODE (network): registration, every file, report
python -m simcode.processing.fourpi <out>/config_resolved.json --files 10              # one file of an acquisition (the process stage's worker)
```
The dcimg files are read through `simcode/io/movie.py` and the files of an acquisition run through `simcode/io/work.py`
(skip finished files, atomic outputs, claims). The acquisitions (paths, channel layout, camera, settings):
`simcode/pipeline/configs/fourpi.py` (`configs(work, data)`: the 4Pi work folder and the raw-movie folder are its
arguments), read through the set-up `simcode/setups/fourpi.py` (`acquisition('MT', work, data)`). The 4Pi networks, the
4Pi PSF and the classical 4Pi-SIMFLUX pipeline it is compared with are on the `paper_figures` branch.

## Repository layout

`docs/LAYOUT.md` has the tree with one line per directory and file. In short, `simcode/` the package: `run.py`, `table.py`,
`models.json`, `config.py`, `dataset.py`; `detector/` and `emitter/` the two networks (one class each for pass 1 and pass 2)
and their trainers; `solve/` the explicit pattern solve and z after it; `pattern/` the illumination inputs; `processing/` the
per-chunk fronts and the pattern estimation; `simulate/` the acquisition simulator and the training batches; `benchmark/`
the gate suite; `calibration/` the in-situ calibration; `pipeline/` the automatic per-acquisition pipeline; `io/` readers and
work driver; `channels/` multi-channel layout, registration and front end; `setups/` the set-ups of the pipeline (widefield,
fourpi); `evaluation/`, `layers/`, `psf/`, `util/`; `fourpi/` the 4Pi networks, PSF and simulators. Besides it: `models/`
calibrations and model folders, `tests/`, `docs/` (the `paper_figures` branch adds `analysis/`: baselines, paper figures and
simulations, 4Pi calibration / classical pipeline / comparisons, benchmarks; and `figures/`).

Tests: `python tests/check_imports.py` and `python tests/test_fileio.py` (CPU), `CUDA_VISIBLE_DEVICES=0 python
tests/test_fast_paths.py` (GPU, ~3 min, needs the m1809 weights), or `pytest tests` (the GPU tests fail without a GPU).

## History

Reproduction checks of this package against the development runs, on the same settings:

- **Real 0.5 nM, frames 12000–18000:**
  - identical candidates
  - bit-identical head outputs, pattern, automatic c, pattern solve and pass 2
  - vs the stored production run (reference code path): positions within the bf16 run-to-run noise
- **Gate iso:** RMS / CRLB 1.17 [1.14, 1.20], identical.
