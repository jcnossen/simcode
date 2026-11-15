# simcode

SIMCODE v2 localizes emitters in DMD-SIMFLUX DNA-PAINT data, where the illumination pattern is modulated and stepped every
frame. It is a per-window deep-learning method:

- **The networks give the per-frame intensities.**
  - A pattern-free detector finds candidates in each 6-frame window, i.e. one full pattern cycle of 2 angles × 3 phases.
  - A neighbour-aware per-candidate head outputs, per candidate:
    - the 6 per-frame intensities with their full covariance
    - the per-frame background
    - a pattern-free position prior
- **The pattern-encoded position is then solved explicitly from those intensities.**
  - The illumination pattern is estimated from the data.
  - The intensity-domain solve has fringe hypotheses with a posterior mean, on/off kinetics, an automatically estimated
    intensity-noise term c, and a calibrated σ.
- **An optional pass 2** reruns pattern-aware networks with the estimated pattern as input.
- **No pixel-level linear intensity fit** is used anywhere.

More in the docs:

- `docs/METHOD.md`: method
- `docs/RESULTS.md`: results of the development phase (earlier default networks) and the limitations
- `docs/EVALUATION.md`: evaluation protocol
- `docs/USAGE.md`: working with the results (`simcode.dataset.Dataset` = `torch_psf.Dataset` + SIMCODE's loaders and selection: load, select, drift, render, FRC, hdf5 / Picasso / ThunderSTORM files); `docs/DATASET_FORMAT.md`: the field layout
- `docs/LAYOUT.md`: the package tree with one line per directory and file (and what was renamed on 2026-11-15)
- `docs/CLEANUP_LOG.md`: the 2026-11 cleanup (what was removed, renamed, and how results were checked)

The repository holds two comparisons and what they need: the SIMCODE preprint against SIMCODE v2 (this pipeline, the
preprint's model and the paper's baselines SMLM / SIMFLUX / DECODE under `analysis/baselines`), and 4Pi-SIMFLUX as published
(classical, `analysis/fourpi_classical`) against 4Pi-SIMCODE (the network pipeline `simcode/fourpi`, see "4Pi" below).

## Install

The lab server has the conda env `simcode-py311` (Python 3.11, PyTorch 2.13 + CUDA, triton).

```bash
conda activate simcode-py311            # or call ~/miniconda3/envs/simcode-py311/bin/python directly
pip install -e <repo>                   # or: export PYTHONPATH=<repo>
```

The repository holds source only: network weights are not in git and have to be copied into `models/` (or the directory
the config key `models` points to); `models/README.md` lists the files of each model.

Dependencies:

- numpy, scipy, tifffile, torch (CUDA GPU required); matplotlib for the calibration / figure scripts; h5py for
  the table files (`locs_p*.hdf5`) and for reading Picasso hdf5 locs in the in-situ calibration; triton optional (`torch.compile`)
- torch-psf (MIT, https://github.com/jcnossen/torch-psf, pinned in pyproject.toml): the PSF models (vectorial PSF,
  convolutional 3D spline, Gaussian), the calibration json reader, the bead and in-situ PSF fits (variable projection)
  and the TIFF movie reader. `simcode/psf/__init__.py` is the only module that imports it (the rest of SIMCODE imports from `simcode.psf`) and holds the SIMCODE-specific glue (per-pattern-angle PSF terms, the
  spline settings of the solve).

Only the baselines need more:

| baseline | needs |
|---|---|
| SIMCODE | `smlmtorch` (`~/simcode`) and its paper models (`~/smlmtorch/scripts/models`, or env `SMLMTORCH_MODELS`) |
| DECODE | the `decode` fork (`~/decode2`) |
| DME v2 drift | the `dme` package (`github.com/jcnossen/drift-estimation`), a pinned git dependency in `pyproject.toml` (installed by `pip install -e <repo>`); a local checkout given by env `DME_PATH` / config key `dme` overrides it; used through `simcode/util/drift.py` |

## Run

```bash
export CUDA_VISIBLE_DEVICES=1           # GPU 1 or 2 on the lab server (keep GPU 0 free)
python -m simcode.run --movie 1809/cover1/2/simflux_5.5ms_500mW.tif --calib insitu/calib_1809_2_psf3d.json \
    --model m1809 --out out_1809_2
python -m simcode.run --dataset 1809_2 --out out_1809_2      # the same: movie, calibration and model of a paper dataset
```

What every run does (= the paper's settings; the options that switched these were removed on 2026-11-14):

- the model (`--model`, required unless `--dataset` / `--config` gives it): a name of `simcode/models.json` (`m1809`,
  `m25`, `g13`) or the path of a model folder with a `model.json` (as written by the automatic pipeline's training
  stage / `simcode.pipeline.training.export_model`); one model = pass-1 detector + head, pass-2 detector + head, their solve
  calibrations and the sigma rescale. There are no single-file overrides: other files = another model folder
- pass 1, pattern estimation, the pattern solve, and pass 2
- the fixed-c real-data solve calibrations of the model (`--cal_kind sim` for simulated movies)
- stored-candidate cut p ≥ 0.3 (`simcode/processing/common.CAND_PMIN`)
- the fast code path, with a compiled solve step when available
- the model's selection-aware sigma rescale (always; a model without one has an empty `sig_u`)
- the output region: `--crop x0,y0,size` (px); default the whole frame (square sensors; else the largest top-left square);
  the paper datasets carry their crop (5,5,110) in `models.json` `_production`
- the tables `locs_p1.hdf5`, `locs_p2.hdf5` in the torch_psf Dataset format (CSV / Picasso: exports of the loaded Dataset)
- the paper's selection (since 2026-11-13) is applied when a table is loaded, nothing is stored for it: a global threshold
  on the calibrated precision, max(σx, σy) ≤ 12 nm on the rows with score p ≥ 0.63 inside the crop
  (`Dataset.from_run(<out>).select()` = `simcode.table.select(table, max_precision_nm=12, min_score=0.63, in_region=True)`;
  another threshold: `select(14)`); the column `sel` (phase-calibrated detection selection) is the recommended full output
  for renders and drift. (The per-tile rule "u ≤ 0.45" and the strict product were removed on 2026-11-14.)

`--dataset <name>` takes movie, calibration, model and crop of a paper dataset from the `_production` entry of `models.json`.
The options removed on 2026-11-14 (`--pass2`, `--autoc`, `--cand_pmin`, `--fast`, `--compile`, `--netset`, `--det` ...,
`--u_max`, `--strict_q`) and on 2026-11-15 (`--format`, `--subset`, `--sigma_max`, `--prefetch` (always on), `--sig_u`
(always the model's)) no longer exist (`docs/CLEANUP_LOG.md`).

Several GPUs: start the same command once per GPU with `--claim 1`; the copies share the chunks of pass 1 and pass 2:

```bash
python -m simcode.io.launch --gpus 1,2 --log out_1809_2/run -- python -m simcode.run --dataset 1809_2 --out out_1809_2 --claim 1
```

A run can be killed at any time: chunk files are written atomically, a restart continues with the missing chunks, and a
chunk claimed by a process that no longer exists is taken over. All movies of the lab server are on one hard disk, so
two workers on one TIFF movie are not necessarily faster than one (not measured).

Other options:

- `--chunk_list 12000` processes only the chunk of frames 12000–18000.
- `--drift drift.npy` (DME output, [F, 2] px, or a DME v2 result npz; `analysis/baselines/dme_drift.py`) adds the
  drift-corrected columns `x_dc, y_dc`; the selection region and the tiles of the sigma rescale then use them.

Outputs in `--out`:

| file | content |
|---|---|
| `locs_p1.hdf5`, `locs_p2.hdf5` | the localization table (one row per 6-frame window of an emitter; windows start every 3 frames), torch_psf Dataset format (`docs/DATASET_FORMAT.md`), see below |
| `locs_p*_info.json` | counts, crop and the size of the paper's selection (`n_main`), column list |
| `run_config.json`, `run_config.yaml` | the complete record of the run: fully resolved settings, file hashes, code and environment, pattern and selection outcome (section "Run record") |
| `summary.json` | settings, model, per-stage timing |
| `command.txt` | plain log of the commands run on the directory |
| `pass1_<f0>.npz`, `pattern_est.pt/.json`, `pass2_est_<f0>.npz`, `pass2_est2_<f0>.npz`, `runtime.json`, `log.txt` | stage outputs; a re-run skips chunks whose files exist (`--redo 1` recomputes). Each chunk also leaves a small record `pass1_<f0>.json` … (timing, counts); `runtime.json` is merged from them |

Table columns (as built by `simcode.table`; in the file as Dataset fields: `pos` = (x, y), `crlb` = (sx, sy), `score` = p,
`intensities` = I, the others under their names; `Dataset.columns()` gives them back under these names):

- `frame` (window start), `t` (time centroid)
- `x, y` (final, sensor px, not drift corrected), `sx, sy` (calibrated σ, px)
- `photons`, `I[6]` (network per-frame intensities)
- `p` (detector), `q_dof`, `kin`, `dnll`, `x_prior, y_prior`
- `good` (p ≥ 0.63), `sel` (phase-calibrated selection: the recommended full output for renders and drift), `disjoint`
  (frame % 6 == 0)
- the metadata `pixel_nm`, `crop`, `tag`, `info`; with `--drift`: `x_dc, y_dc`
- runs made before 2026-11-15 have `.npz` tables (and subset files `locs_p*_sig12.npz`); `Dataset.from_run` reads them too;
  `python -m simcode.table <out> --tag est / est2` writes the `.hdf5` tables of such a run from its chunk files

The table can be rebuilt at any time with
`python -m simcode.table <out> [--tag est2] [--drift drift.npy] [--crop x0,y0,size]` (the sigma rescale of the run's model).
The crop of the selection and of the sigma-rescale tiles is the output region of the run (`--crop` of the run; read from its chunk files).

Which settings produced the numbers in `docs/RESULTS.md` (options of the time: stored-candidate cut 0.5, fixed c, calibrations, SIMCODE model per
movie, DME, grid) is listed in its section 7, "Reproduce".

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
| `settings` | every argument after the defaults: movie, calibration, model and its files, crop, chunking, frames, σ_max, sigma rescale, subsets, formats; files as absolute paths |
| `fixed` | what is not an option: pass 2, 6-frame windows every 3 frames, stored-candidate cut 0.3, p ≥ 0.63, fixed c, code path, compile mode, batch sizes, margin, pattern search range, default crop |
| `model` | name and definition; per file the absolute path, size, sha256; for a network the trainer arguments and step stored in its checkpoint; for a calibration its scalar parameters (`cmult` = the fixed c, `prior_off`, ...) |
| `calib` | in-situ PSF / camera calibration: path, sha256, camera maps + sha256, the PSF parameters (NA, wavelength, indices, pixel size, focus, Zernike coefficients, extra terms) and the camera gain / offset / read noise |
| `movie` | path, parts with size and sha256, frame count and size, frames and chunks processed, frame order / pattern scheme |
| `code`, `environment` | package version, git commit, dirty flag with the modified files, DME version and commit; python / numpy / torch / CUDA versions, GPU, host (no user name or e-mail) |
| `invocations` | one entry per process: command line, GPU, start / end, timing (a resumed run or one copy per GPU with `--claim 1` adds entries) |
| `pattern`, `selection`, `counts`, `timing` | the pattern estimate (pitch, angle, depth, phases, drift range), the counts per selection rule and table, detections, per-stage time |
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
python -m simcode.util.runconfig backfill <dir>   # reconstruct the record of a run made before 2026-11-14 (nothing is recomputed)
```

`command.txt` stays a plain log (one line per `simcode.run` / `simcode.table` command); `summary.json`, `runtime.json` and
`locs_p*_info.json` are unchanged. The three production runs of the paper (`out/paper2/1809_2/v2_matched`,
`2.5nM_2/v2_matched`, `20nM_2/v2_m25`) predate the record: theirs has status `reconstructed` and lists under
`reconstructed.not_recovered` what could not be recovered (git commit and environment at run time, hashes at run time,
unlogged table rebuilds). The automatic pipeline writes the same kind of record per acquisition and per stage, and the
training recipe with file hashes into the model json (`docs/AUTO_PIPELINE.md`, section 2).

## Paths and configuration

No path is hard-coded (`simcode/config.py`). The roots:

| root | env var | config key | default |
|---|---|---|---|
| models | `SIMCODE_MODELS` | `models` | `<repo>/models` |
| data | `SIMCODE_DATA` | `data` | none: pass full movie paths, or names relative to it |
| out | `SIMCODE_OUT` | `out` | `./simcode_out`; used by the benchmark / analysis scripts |
| 4Pi work directory | `SIMCODE_4PI` | `fourpi` | `<out>/4pi` |
| 4Pi raw data | `SIMCODE_4PI_DATA` | `data4pi` | `root` of `simcode/fourpi/datasets.json` |

The config file is `$SIMCODE_CONFIG`, else `<repo>/simcode.json`, else `~/.config/simcode.json`; see
`simcode.example.json`. Network and calibration arguments are resolved relative to the models dir (also `calib/` and
`insitu/`), so `--calib insitu/calib_1809_2_psf3d.json` works from anywhere. (The names from before the package was renamed,
`SIMCODE2_*` and `simcode2.json`, are no longer read since 2026-11-14.)

## Models and calibrations (`models/`)

Calibrations (json) are in git; network weights are not (`models/README.md`: the sets, their files with checksums, the
recipe of a model). The models are defined in `simcode/models.json`.

## 4Pi

```bash
python -m simcode.pipeline simcode/pipeline/configs/fourpi.py --only fourpi_MT --gpus 1,2   # 4Pi-SIMCODE (network): registration, every file, report
python -m simcode.processing.fourpi <out>/config_resolved.json --files 10              # one file of an acquisition (the process stage's worker)
python analysis/fourpi_classical/full_cl.py --stage read --cells NPC                    # classical 4Pi-SIMFLUX: reader ...
python analysis/fourpi_classical/full_cl.py --stage fit --cells NPC                     # ... and fitter(s)
```
Both read the dcimg files through `simcode/io/movie.py` and run their files through `simcode/io/work.py`
(skip finished files, atomic outputs, claims). The acquisitions (paths, channel layout, camera, settings):
`simcode/pipeline/configs/fourpi.py`, read through the set-up `simcode/setups/fourpi.py` (`acquisition('MT')`).
Post-processing and comparisons: `analysis/fourpi/post_w4pi.py`, `cmp_full_w4pi.py`, `showcase_real_mt.py`,
`select_curve_{mt,npc}.py`; `analysis/fourpi_classical/post_cl.py`, `fig_cl.py`.

## Repository layout

`docs/LAYOUT.md` has the tree with one line per directory and file. In short, `simcode/` the package: `run.py`, `table.py`,
`models.json`, `config.py`, `dataset.py`; `detector/` and `emitter/` the two networks (one class each for pass 1 and pass 2)
and their trainers; `solve/` the explicit pattern solve and z after it; `pattern/` the illumination inputs; `processing/` the
per-chunk fronts and the pattern estimation; `simulate/` the acquisition simulator and the training batches; `benchmark/`
the gate suite; `calibration/` the in-situ calibration; `pipeline/` the automatic per-acquisition pipeline; `io/` readers and
work driver; `channels/` multi-channel layout, registration and front end; `setups/` the set-ups of the pipeline (widefield,
fourpi); `evaluation/`, `layers/`, `psf/`, `util/`; `fourpi/` the 4Pi networks, PSF and simulators. Besides it: `analysis/` scripts that use
it (baselines, paper figures and simulations, 4Pi calibration / classical pipeline / comparisons, benchmarks), `models/`
calibrations and model folders, `tests/`, `docs/`.

Tests: `python tests/check_imports.py` and `python tests/test_fileio.py` (CPU), `CUDA_VISIBLE_DEVICES=1 python
tests/test_fast_paths.py` (GPU, ~1 min, needs the m1809 weights), or `pytest tests`.

## History

Reproduction checks of this package against the development runs, on the same settings:

- **Real 0.5 nM, frames 12000–18000:**
  - identical candidates
  - bit-identical head outputs, pattern, automatic c, pattern solve and pass 2
  - vs the stored production run (reference code path): positions within the bf16 run-to-run noise
- **Gate iso:** RMS / CRLB 1.17 [1.14, 1.20], identical.
