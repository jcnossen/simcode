# simcode2

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
- `docs/RESULTS.md`: the current honest comparison with SIMCODE, and the limitations
- `docs/EVALUATION.md`: evaluation protocol

## Install

The lab server has the conda env `simcode-py311` (Python 3.11, PyTorch 2.13 + CUDA, triton).

```bash
conda activate simcode-py311            # or call ~/miniconda3/envs/simcode-py311/bin/python directly
pip install -e /home/jc/simcode2        # or: export PYTHONPATH=/home/jc/simcode2
```

Dependencies:

- numpy, scipy, tifffile, torch (CUDA GPU required); h5py only for `--format hdf5`
- `torch_convspline`: the user's Catmull-Rom spline package, `~/torch-spline-3d`, installed editable in the env; used by
  the PSF LUT

Only the baselines need more:

| baseline | needs |
|---|---|
| SIMCODE | `smlmtorch` (`~/simcode`) and its paper models (`~/smlmtorch/scripts/models`, or env `SMLMTORCH_MODELS`) |
| DECODE | the `decode` fork (`~/decode2`) |
| DME v2 drift | `~/dme` (env `DME_PATH`), imported read-only |

## Run

```bash
export CUDA_VISIBLE_DEVICES=1           # GPU 1 or 2 on the lab server (keep GPU 0 free)
python -m simcode2.run --movie /mnt/data/dmdsf/bn_dnapaint/dnapaint/0.5nM/cover1/1/simflux_5.5ms_500mW.tif \
    --calib insitu/calib_0.5nM.json --out out_05nM [--pass2]
```

Production defaults:

- fast path, with a compiled solve step when available
- detector `det_n1`, head `hd_sc5nb`
- stored-candidate cut p ≥ 0.3 (`--cand_pmin`; the RESULTS runs used 0.5, see below)
- automatic c per dataset (`--autoc 0` uses the fixed-c real calibration instead; the RESULTS runs used `--autoc 0`)
- strict product with the density-rule q (below)
- pass 2 (`--pass2`): `p2_det` + `p2_head_long`, calibration `calib_p2L_b10_real`

Other options:

- `--chunk_list 12000` processes only the chunk of frames 12000–18000.
- `--pattern_field est_depth` is the opt-in depth-only pattern field (not recommended by default).
- `--drift drift.npy` (DME output, [F, 2] px, or a DME v2 result npz; `analysis/baselines/dme_drift.py`) is used for the
  strict-product coverage and tiles.
- `--format npz,csv,hdf5` sets the table formats.

Outputs in `--out`:

| file | content |
|---|---|
| `locs_p1.npz` (and `locs_p2.npz` with `--pass2`) | the localization table (one row per 6-frame window of an emitter; windows start every 3 frames), see below |
| `locs_p*_info.json` | counts, strict-product q and coverage, column list |
| `summary.json` | settings, automatic θ, per-stage timing |
| `pass1_<f0>.npz`, `pattern_est.pt/.json`, `pass2_est_<f0>.npz`, `pass2_est2_<f0>.npz`, `calib_autoc.json`, `runtime.json`, `log.txt` | stage outputs; a re-run skips stages whose files exist (`--redo 1` recomputes) |

Table columns:

- `frame` (window start), `t` (time centroid)
- `x, y` (final, sensor px, not drift corrected), `sx, sy` (calibrated σ, px)
- `photons`, `I[6]` (network per-frame intensities)
- `p` (detector), `q_dof`, `kin`, `dnll`, `x_prior, y_prior`
- `good` (p ≥ 0.63), `sel` (phase-calibrated selection: the recommended full output for renders and drift), `disjoint`
  (frame % 6 == 0)
- `strict_u` (per-tile σ rank quantile) and **`strict`** (the recommended strict product)

`strict` is a per-10-px-tile σ quantile among the good rows: in every tile the fraction q with the smallest calibrated σ
is kept. By default q comes from the GT-free row density ρ (good rows in the crop per window start per µm²), interpolated
between anchors measured on the three real movies, where q reproduces SIMCODE's coverage (occupied 20-nm bins): pass 1
ρ 0.49 → q 0.275, ρ ≥ 0.87 → 0.12; pass 2 ρ 0.64 → 0.25, 1.69 → 0.12, ≥ 2.22 → 0.10 (`simcode2/strict.py`). The per-tile
q does not depend on the solve calibration, so the default holds with the automatic c (`--autoc 1`, default) and with the
fixed c (`--autoc 0`): on the real 0.5 / 2.5 / 20 nM movies its coverage is 0.98–1.02 × SIMCODE's in both cases (pass 1;
`analysis/evalproto/strictq.py`, `analysis/results/evalproto/strictq_*.txt`). This match is in-sample (the anchors were set on these movies). Without `--drift` the crop, tiles and ρ use raw
(not drift-corrected) coordinates; the effect is negligible at ~1 px drift. On data far from these densities, check the
coverage (`locs_p*_info.json`) or set q yourself. Options:

- `--strict_q 0.25` fixes q.
- `--strict_sigma_ref 7.0` is the earlier rule (q such that the product covers as many bins as the global σ ≤ 7.0 nm set;
  7.4 nm for pass 2). It is valid only with the fixed-c calibration (`--autoc 0`): the automatic c gives ~10–17 % larger σ,
  and then 7.0 nm keeps only 0.68–0.82 × SIMCODE's coverage (q 0.17 / 0.05 / 0.06 instead of 0.27 / 0.11 / 0.13).

The table can be rebuilt at any time with
`python -m simcode2.strict <out> [--tag est2] [--drift drift.npy] [--strict_q 0.25]`.

Which settings produced the numbers in `docs/RESULTS.md` (`--cand_pmin 0.5 --autoc 0`, calibrations, SIMCODE model per
movie, DME, grid) is listed in its section 7, "Reproduce".

The expert entry point with every development flag is `python -m simcode2.pipeline.run_uncalibrated`
(`--frontend sc|sc2p`, `--fast`, `--compile`, `--cand_pmin`, `--p1_solve`, `--pattern_field`, `--ref_pattern`, ...).

Speed: on an RTX 3090, a 6000-frame 0.5 nM chunk takes about 35 s for pass 1 + pattern and 11 s for the solve. The
automatic c adds about 80 s, once per dataset (`--autoc_fast 1`: ~25 s, but θ then differs by ~1e-3 from the reference
estimator). Pass 2 adds about 25 s per chunk.

## Paths and configuration

No path is hard-coded (`simcode2/config.py`). There are three roots:

| root | env var | config key | default |
|---|---|---|---|
| models | `SIMCODE2_MODELS` | `models` | `<repo>/models` |
| data | `SIMCODE2_DATA` | `data` | none: pass full movie paths, or names relative to it |
| out | `SIMCODE2_OUT` | `out` | `./simcode2_out`; used by the benchmark / analysis scripts |

The config file is `$SIMCODE2_CONFIG`, else `<repo>/simcode2.json`, else `~/.config/simcode2.json`; see
`simcode2.example.json`. Network and calibration arguments are resolved relative to the models dir (also `calib/` and
`insitu/`), so `--calib insitu/calib_0.5nM.json` works from anywhere.

## Models and calibrations (`models/`, committed; every file < 10 MB)

| file | what |
|---|---|
| `det_n1.pt` | pass-1 detector (pattern-free) |
| `hd_sc5nb.pt` | pass-1 head (SCHead4, neighbour-aware) |
| `p2_det.pt`, `p2_head_long.pt` | pass-2 (pattern-aware) detector and head (p2L) |
| `hd_sc7bg.pt` | optional head with a dense BgNet background; not production |
| `calib/` | solve calibrations: pass 1 real / sim / autoc, the full-run autoc results r05 / r25 / r20, pass 2 real / sim / gate; see `docs/METHOD.md` |
| `insitu/` | in-situ PSF + camera calibrations of the 0.5 nM, 2.5 nM movie 2 and 20 nM movie 2 data, with their camera maps |

## Repository layout

```
simcode2/            the package
  run.py               production CLI
  strict.py            localization table + strict product
  autoc.py             automatic c
  config.py            paths
  pipeline/            run_uncalibrated (stages), sc_front_fast (fast pass 1 / solve / pass 2), sc_front (reference path),
                       pattern_est, pattern_field (opt-in), common (movie / calib I/O), dedup
  hd/                  sc.py (head base + sc_solve), sc4.py (SCHead4 + trainer), sc7bg.py, msolve / msolve_fast (pattern solve),
                       realsim / data / realgen (simulator + training batches), gate / gate_report(_xdet) (CRLB benchmark)
  det/                 detector (model.py, up2.py = det_n1 trainer, data.py, psfpert.py)
  p2/                  pass 2 (front, det2, head2, data, illum, dump)
  sim/                 simulator.py (calibrated renderer), crlb.py (scenes + multi-emitter CRLB), simtools.py (gate layouts)
  psf/, core.py, nn/blocks.py, fastutil.py   vectorial PSF + LUT, pattern helpers, network blocks, torch.compile helper
  eff/                 simcode_gate.py (SIMCODE on the gate scenes), crlb_nuis.py (nuisance CRLB)
  evalproto/lib.py     evaluation-protocol metrics
analysis/            scripts (not imported by the package)
  evalproto/           nnlag, blockphase, covpool + datasets.py (json dataset specs)
  simreal/             sim_movie_r.py, recipes.sh (m05r / m25rc / m05pd / m25pd / m05pe / m25pe), eval_movie.sh, fields/
  baselines/           run_simcode.py (sims) / run_simcode_real.py (real; --fix_sign 1 = sign-fixed SIMCODE (b)),
                       decode/ (DECODE retrain + inference), dme_drift.py (DME v2, read-only import)
  gate/run_gate.sh     gate scenes + report
  calib/               sigma-calibration fitting on the gate tuning seed (dump_fe.py -> fit_cal.py; plib.py for pass 2)
models/              networks and calibrations
tests/               test_fast_paths.py (fast vs reference paths; GPU, ~1 min) + data/
docs/                METHOD, RESULTS, EVALUATION
```

Tests: `CUDA_VISIBLE_DEVICES=1 python tests/test_fast_paths.py` (or `pytest tests`).

## History

Reproduction checks of this package against the development runs, on the same settings:

- **Real 0.5 nM, frames 12000–18000:**
  - identical candidates
  - bit-identical head outputs, pattern, automatic c, pattern solve and pass 2
  - vs the stored production run (reference code path): positions within the bf16 run-to-run noise
- **Gate iso:** RMS / CRLB 1.17 [1.14, 1.20], identical.
