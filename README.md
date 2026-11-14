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
- `docs/USAGE.md`: working with the results (`simcode.dataset.Dataset`: load, select, drift, render, FRC, Picasso / ThunderSTORM export)
- `simcode/FILE_DESCRIPTION.md`: what each file does
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
  `--format hdf5` and for reading Picasso hdf5 locs in the in-situ calibration; triton optional (`torch.compile`)
- The PSF lookup tables use the convolutional 3D spline (Catmull-Rom spline evaluated as a grouped conv2d, value and
  gradients) from torch-spline-3d by Jelmer Cnossen (MIT, https://gitlab.com/jcnossen/torch-spline-3d). `simcode/psf/conv_spline.py`
  re-implements its evaluation for the PSF LUT (strided, single-index knot gather), so the package itself is not a
  dependency.

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

- model `m1809` (`--model`, older name `--netset`; models of `simcode/models.json`: `m1809`, `m25`, `g13`, `m1809g`; one
  model = pass-1 detector + head, pass-2 detector + head, their solve calibrations and the sigma rescale; `--det / --head /
  --cal / --p2_det / --p2_head / --p2_cal` override single files)
- pass 1, pattern estimation, the pattern solve, and pass 2
- the fixed-c real-data solve calibrations of the model (`--cal_kind sim` for simulated movies)
- stored-candidate cut p ≥ 0.3 (`simcode/pipeline/common.CAND_PMIN`)
- the fast code path, with a compiled solve step when available
- the model's selection-aware sigma rescale (`--sig_u set`) and the subset files `locs_p2_sig12.npz` / `locs_p2_u045.npz` /
  `locs_p2_strictonly.npz` (`--subset main,u,strict`)
- selections: the paper's main selection (since 2026-11-13) = a global threshold on the calibrated precision, rows with a
  rank (p ≥ 0.63, inside the crop) and max(σx, σy) ≤ σ_max, σ_max = 12 nm (`--sigma_max`, stored in the table; no column of
  its own); the earlier per-tile rule "u ≤ 0.45" = `strict_u ≤ u_max` (`--u_max`, rule `u`); and the columns `sel`, `strict`

`--dataset <name>` takes movie, calibration and model of a paper dataset from the `_production` entry of `models.json`.
Older scripts may still pass the removed options; they are accepted only with their fixed value (`--pass2`, `--autoc 0`,
`--cand_pmin 0.3`, `--fast 1`, `--compile auto`; `simcode/util/retired.py`).

Several GPUs: start the same command once per GPU with `--claim 1`; the copies share the chunks of pass 1 and pass 2:

```bash
python -m simcode.fileio.launch --gpus 1,2 --log out_1809_2/run -- python -m simcode.run --dataset 1809_2 --out out_1809_2 --claim 1
```

A run can be killed at any time: chunk files are written atomically, a restart continues with the missing chunks, and a
chunk claimed by a process that no longer exists is taken over. All movies of the lab server are on one hard disk, so
two workers on one TIFF movie are not necessarily faster than one (not measured).

Other options:

- `--chunk_list 12000` processes only the chunk of frames 12000–18000.
- `--drift drift.npy` (DME output, [F, 2] px, or a DME v2 result npz; `analysis/baselines/dme_drift.py`) is used for the
  strict-product coverage and tiles.
- `--format npz,csv,hdf5` sets the table formats.

Outputs in `--out`:

| file | content |
|---|---|
| `locs_p1.npz`, `locs_p2.npz` | the localization table (one row per 6-frame window of an emitter; windows start every 3 frames), see below |
| `locs_p*_info.json` | counts, sigma_max and the size of the main selection, u_max and the size of the u rule, strict-product q and coverage, column list |
| `locs_p2_sig12.npz`, `locs_p2_u045.npz`, `locs_p2_strictonly.npz` (and the same for pass 1; `--subset`) | the rows of a selection as a small table (x, y, frame, sx, sy, photons), the input format of the FRC and figure scripts |
| `run_config.json`, `run_config.yaml` | the complete record of the run: fully resolved settings, file hashes, code and environment, pattern and selection outcome (section "Run record") |
| `summary.json` | settings, model, per-stage timing |
| `command.txt` | plain log of the commands run on the directory |
| `pass1_<f0>.npz`, `pattern_est.pt/.json`, `pass2_est_<f0>.npz`, `pass2_est2_<f0>.npz`, `runtime.json`, `log.txt` | stage outputs; a re-run skips chunks whose files exist (`--redo 1` recomputes). Each chunk also leaves a small record `pass1_<f0>.json` … (timing, counts); `runtime.json` is merged from them |

Table columns:

- `frame` (window start), `t` (time centroid)
- `x, y` (final, sensor px, not drift corrected), `sx, sy` (calibrated σ, px)
- `photons`, `I[6]` (network per-frame intensities)
- `p` (detector), `q_dof`, `kin`, `dnll`, `x_prior, y_prior`
- `good` (p ≥ 0.63), `sel` (phase-calibrated selection: the recommended full output for renders and drift), `disjoint`
  (frame % 6 == 0)
- `strict_u` (per-tile σ rank quantile among the good rows in the crop; finite = the rows with a rank), the stored
  `sigma_max` (12 nm) and `px_nm`: the paper's selection is `strict_u` finite and max(sx, sy) · px_nm ≤ sigma_max, row mask
  `simcode.strict.select(table)`; the stored `u_max` (0.45) for the earlier rule `select(table, 'u')`; and `strict` (the
  density-rule strict product)

`strict` is a per-10-px-tile σ quantile among the good rows: in every tile the fraction q with the smallest calibrated σ
is kept. By default q comes from the GT-free row density ρ (good rows in the crop per window start per µm²), interpolated
between anchors measured on the three real movies, where q reproduces SIMCODE's coverage (occupied 20-nm bins): pass 1
ρ 0.49 → q 0.275, ρ ≥ 0.87 → 0.12; pass 2 ρ 0.64 → 0.25, 1.69 → 0.12, ≥ 2.22 → 0.10 (`simcode/strict.py`). The per-tile
q does not depend on the solve calibration (checked with the automatic c of `simcode/autoc.py` and with the fixed c of
the pipeline): on the real 0.5 / 2.5 / 20 nM movies its coverage is 0.98–1.02 × SIMCODE's in both cases (pass 1;
`analysis/evalproto/strictq.py`, `analysis/results/evalproto/strictq_*.txt`). This match is in-sample (the anchors were set on these movies). Without `--drift` the crop, tiles and ρ use raw
(not drift-corrected) coordinates; the effect is negligible at ~1 px drift. On data far from these densities, check the
coverage (`locs_p*_info.json`) or set q yourself. Options:

- `--strict_q 0.25` fixes q. (The earlier rule `--strict_sigma_ref`, q from the coverage of a global σ threshold, was
  removed on 2026-11-14.)

The table can be rebuilt at any time with
`python -m simcode.strict <out> [--tag est2] [--drift drift.npy] [--strict_q 0.25] [--u_max 0.45] [--subset main,strict] [--sig_u calib/sigu/sigu_p2_m1809.json]`.
The crop of the tiles and of the coverage is the output region of the run (`--crop` of the run; read from its chunk files).

Which settings produced the numbers in `docs/RESULTS.md` (options of the time: stored-candidate cut 0.5, fixed c, calibrations, SIMCODE model per
movie, DME, grid) is listed in its section 7, "Reproduce".

The stage-level entry point is `python -m simcode.pipeline.run_uncalibrated` (`--frontend sc|sc2p`, `--stages`,
`--ref_pattern` for a comparison solve with a given pattern, batch sizes).

Speed: on an RTX 3090, a 6000-frame 0.5 nM chunk takes about 35 s for pass 1 + pattern and 11 s for the solve. Pass 2
adds about 25 s per chunk.

## Run record (`run_config.json`)

Every output directory says how it was calculated. `python -m simcode.run` writes `<out>/run_config.json` at the start
(status `running`) and completes it at the end (`done`; `failed` with the error text); `run_config.yaml` is the same content
for reading. It holds (`simcode/util/runconfig.py`):

| entry | content |
|---|---|
| `settings` | every argument after the defaults: movie, calibration, model, the single network / calibration files, crop, chunking, frames, σ_max, u_max, strict q, sigma rescale, subsets, formats; files as absolute paths |
| `fixed` | what is not an option: pass 2, 6-frame windows every 3 frames, stored-candidate cut 0.3, p ≥ 0.63, fixed c, code path, compile mode, batch sizes, margin, pattern search range, strict crop and density anchors |
| `model` | name and definition; per file the absolute path, size, sha256; for a network the trainer arguments and step stored in its checkpoint; for a calibration its scalar parameters (`cmult` = the fixed c, `prior_off`, ...) |
| `calib` | in-situ PSF / camera calibration: path, sha256, camera maps + sha256, the PSF parameters (NA, wavelength, indices, pixel size, focus, Zernike coefficients, extra terms) and the camera gain / offset / read noise |
| `movie` | path, parts with size and sha256, frame count and size, frames and chunks processed, frame order / pattern scheme |
| `code`, `environment` | package version, git commit, dirty flag with the modified files, DME version and commit; python / numpy / torch / CUDA versions, GPU, host (no user name or e-mail) |
| `invocations` | one entry per process: command line, GPU, start / end, timing (a resumed run or one copy per GPU with `--claim 1` adds entries) |
| `pattern`, `selection`, `counts`, `timing` | the pattern estimate (pitch, angle, depth, phases, drift range), the counts per selection rule and table, detections, per-stage time |
| `outputs` | every table / subset file with its size and `arrays_sha256` (hash of the array contents; the file hash of an `.npz` changes with its zip time stamps) |
| `steps` | later work on the directory: `python -m simcode.strict` appends its settings, input hashes, counts and outputs here and marks the files it replaced (`written_by: steps[i]`) |

Re-run with exactly the settings of a record (the files of the record are passed explicitly; arguments after `--config`
override single settings):

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

`command.txt` stays a plain log (one line per `simcode.run` / `simcode.strict` command); `summary.json`, `runtime.json` and
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
`insitu/`), so `--calib insitu/calib_1809_2_psf3d.json` works from anywhere. The names from before the package was renamed
(`SIMCODE2_*`, `simcode2.json`) are still read when the new ones are absent.

## Models and calibrations (`models/`)

Calibrations (json) are in git; network weights are not (`models/README.md`: the sets, their files with checksums, the
recipe of a model). The models are defined in `simcode/models.json`.

## 4Pi

```bash
python -m simcode.fourpi.run --cell NPC --files 10 --head v10 --coh 1.56                # 4Pi-SIMCODE (network), one file
GPUS=1,2 bash analysis/fourpi/run_full_w4pi.sh MT v10 1.56                               # a whole movie, one worker per GPU
python analysis/fourpi_classical/full_cl.py --stage read --cells NPC                    # classical 4Pi-SIMFLUX: reader ...
python analysis/fourpi_classical/full_cl.py --stage fit --cells NPC                     # ... and fitter(s)
```
Both read the dcimg files through `simcode/fileio/movie.py` and run their files through `simcode/fileio/work.py`
(skip finished files, atomic outputs, claims). Paths and the constants of the data set: `simcode/fourpi/setup.py`.
Post-processing and comparisons: `analysis/fourpi/post_w4pi.py`, `cmp_full_w4pi.py`, `showcase_real_mt.py`,
`select_curve_{mt,npc}.py`; `analysis/fourpi_classical/post_cl.py`, `fig_cl.py`.

## Repository layout

`simcode/FILE_DESCRIPTION.md` has one line per file. In short: `simcode/` the package (`run.py`, `strict.py`, `models.json`,
`config.py`; `fileio/` readers and work driver; `pipeline/`, `det/`, `hd/`, `p2/` the 2D pipeline and its trainers; `sim/`,
`psf/`, `insitu/` simulator, PSF and calibration; `fourpi/` the 4Pi pipeline), `analysis/` scripts that use it (baselines,
paper figures and simulations, 4Pi calibration / classical pipeline / comparisons, benchmarks), `models/` calibrations,
`tests/`, `docs/`.

Tests: `python tests/check_imports.py` and `python tests/test_fileio.py` (CPU), `CUDA_VISIBLE_DEVICES=1 python
tests/test_fast_paths.py` (GPU, ~1 min, needs the m1809 weights), or `pytest tests`.

## History

Reproduction checks of this package against the development runs, on the same settings:

- **Real 0.5 nM, frames 12000–18000:**
  - identical candidates
  - bit-identical head outputs, pattern, automatic c, pattern solve and pass 2
  - vs the stored production run (reference code path): positions within the bf16 run-to-run noise
- **Gate iso:** RMS / CRLB 1.17 [1.14, 1.20], identical.
