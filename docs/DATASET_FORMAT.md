# Dataset format

SIMCODE's localizations are `torch_psf.Dataset` objects: the specification (fields, units, metadata, drift, the HDF5
container, Picasso / ThunderSTORM exports) is torch-psf's `docs/DATASET_FORMAT.md`, at the commit pinned in
`pyproject.toml` (`external/torch-psf-<sha>/docs/DATASET_FORMAT.md` in a checkout with the export). In short: `pos` [n, D]
px (pixel centres at integers, x = column; z in px of the lateral pixel size), `crlb` [n, D] px, `frame` = first frame,
`n_frames` = frames covered, `photons`, `bg`, `id`, any further field [n, ...]; metadata `pixel_nm`, `image_size`,
`n_frames`, ...; `ds.save('x.hdf5')` / `Dataset.load('x.hdf5')`.

Files are written with torch-psf's dataset format_version 2 (since the psf-unify pin, 2026-11-20; the layout of the
fields and metadata is unchanged, and readers of version 1 do not check the version). torch-psf's own `Dataset` no longer
holds a drift (its `estimate_drift` returns a `torch_psf.Drift`); SIMCODE's `Dataset` keeps the drift as state of the
object (`ds.drift`, `set_drift` / `apply_drift` / `undo_drift` / `estimate_drift`, docs/USAGE.md) and saves it as the
`/drift` dataset of the file (attributes `applied`, `info`), as before; `torch_psf.Dataset.load` of such a file reads the
record of an applied drift and warns about an unapplied curve (read it with `torch_psf.Drift.load(path)`).

`simcode.dataset.Dataset` is that class plus SIMCODE's loaders (run directories; the tables of `simcode.run` / `simcode.table` are files of
this format, `locs_p{1,2}_{net,solve}.hdf5`, with the metadata `pixel_nm`, `crop`, `tag`, `kind`, `table`, `info`; the `.npz` tables of runs before 2026-11-15, the paper's DECODE npz, the preprint code's `.npy`) and the paper's
selection `select()` (docs/USAGE.md). A selection is not stored: it is computed when needed.

## SIMCODE's fields and metadata

A v2 table (SIMCODE v2, `simcode/table.py`; four per run: `locs_p1_net`, `locs_p1_solve`, `locs_p2_net`,
`locs_p2_solve` = the result) in this layout:

| field | dtype | shape | meaning |
|---|---|---|---|
| `pos` | float64 | [n, 2] or [n, 3] | position (x, y), px, sensor coordinates, not drift corrected: net tables = the emitter network's position (the calibration's prior_off removed), solve tables = the posterior mean of the explicit pattern solve; 3D heads (2026-11-14): + z = the network's z prior conditioned on the table's x, y (px of the lateral pixel size, relative to the focal plane of the model's PSF calibration: z = (height above the coverslip - focus_um) / pixel size) |
| `crlb` | float32 | [n, 2] or [n, 3] | sigma of x, y (px), not a CRLB: net tables = the network's sd (position block of its joint (x, y, I_1..6) covariance), solve tables = the solve's posterior sd; no factor (the selection-aware rescale was removed 2026-11-17); z: sd of z after the conditioning; pass-2 tables: z, its sd and cov are the pass-1 network's prior conditioned on the table's x, y (NaN for rows without a linked pass-1 candidate) |
| `cov` | float32 | [n, 3] | 3D heads: covariances (xy, xz, yz) px^2 of the position (xy = 0: x, y uncorrelated) |
| `z_prior` | float64 | [n] | 3D heads: the head's pattern-free z (px) before the conditioning |
| `frame` | int32 | [n] | window start (absolute movie frame); windows start every 3 frames |
| `n_frames` | int32 | [n] | 6: the window length (drift time = window centre, frame + 3) |
| `photons` | float32 | [n] | sum of the 6 network per-frame intensities (label units) |
| `score` | float32 | [n] | detector probability (`p` of the table; `select(min_score=)`) |
| `intensities` | float32 | [n, 6] | network per-frame intensities (`I` of the table) |
| `t` | float64 | [n] | time centroid of the window's intensities (absolute frame, fractional) |
| `q_dof`, `dnll` | float32 | [n] | solve tables: intensity-domain chi2 / dof of the solve at its MAP mode; NLL gap to the best competing fringe (pass 1) |
| `kin` | float32 | [n] | solve tables: posterior weight of the on/off-kinetics model (0-1; both models are fitted and averaged, 2026-11-17; bool before) |
| `p1_index` | int64 | [n] | pass-2 tables: row of the linked pass-1 candidate in the pass-1 tables (-1: none) |
| `x_prior`, `y_prior` | float64 | [n] | pattern-free network position (px) |
| `good`, `sel`, `disjoint` | bool | [n] | p >= 0.63; the phase-calibrated detection selection; frame % 6 == 0 |
| `x_dc`, `y_dc` | float64 | [n] | only for runs with `--drift`: drift-corrected x, y |
| `strict_u`, `strict` | | [n] | tables before 2026-11-14 only; unused |

Metadata added by the loaders: `pixel_nm` (the table's px_nm), `crop` [x0, y0, size] (output region of the run; the
selection's region), `sigma_max` (npz tables before 2026-11-15: the run's selection threshold, nm), `tag` (est / est2), `kind` (net /
solve), `table` (its name), `info` (the table's info json), `run_config` (the run
record), `model`, `pass` (1 / 2), `n_frames`, `image_size`, `source`, `format`; `selection` after `select()`.

Other sources:

| source | mapping |
|---|---|
| DECODE npz of the paper (`analysis/baselines`) | phot -> `photons`, sphot -> `crlb_photons`, prob -> `score`, sx, sy -> `crlb`; `z` (nm) kept as a field (the datasets are 2D); `n_frames` 6 |
| preprint code `.npy` (Dataset / SFDataset) | estim.pos / crlb.pos -> `pos` / `crlb` (a z column -> fields `z`, `crlb_z`), estim.ibg -> `intensities`, `intensities_bg`; `chisq`, `group`, `roi_id`, `roipos`; `n_frames` = `window=` (6 for the paper's tables) |
| Picasso hdf5 (e.g. the paper's sf.hdf5) | torch_psf's `load_picasso` (lpx, lpy -> `crlb`; sx, sy -> `psf_sx`, `psf_sy`); `n_frames` = `window=` |

## Where results live: `runs/<run name>/` next to the movie (planned layout)

```
<movie dir>/
  movie.tif                         raw data (never modified)
  runs/
    <run name>/
      run_config.yaml               the complete record (settings, file hashes, commit, environment); also .json
      final.hdf5                    the result (this format)
      prior.hdf5, detector.hdf5, pass1.hdf5   optional related estimates, same ids
      psf/                          the PSF / camera calibration used or estimated by this run (calib json, camera maps)
      model/                        only if this run trained a model: networks, calibrations, model.json (training record)
      report.html                   summary page
      work/                         intermediate files (chunks, caches); may be deleted after the run
```
