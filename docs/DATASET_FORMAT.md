# Dataset format

SIMCODE's localizations are `torch_psf.Dataset` objects: the specification (fields, units, metadata, drift, the HDF5
container, Picasso / ThunderSTORM exports) is torch-psf's `docs/DATASET_FORMAT.md`, at the commit pinned in
`pyproject.toml` (`external/torch-psf-<sha>/docs/DATASET_FORMAT.md` in a checkout with the export). In short: `pos` [n, D]
px (pixel centres at integers, x = column; z in px of the lateral pixel size), `crlb` [n, D] px, `frame` = first frame,
`n_frames` = frames covered, `photons`, `bg`, `id`, any further field [n, ...]; metadata `pixel_nm`, `image_size`,
`n_frames`, ...; `ds.save('x.hdf5')` / `Dataset.load('x.hdf5')`.

`simcode.dataset.Dataset` is that class plus SIMCODE's loaders (run directories, the v2 table `.npz` of `simcode.run` /
`simcode.strict`, the earlier SIMCODE npz save, the paper's DECODE npz, the preprint code's `.npy`) and the paper's
selection `select()` (docs/USAGE.md). A selection is not stored: it is computed when needed.

## SIMCODE's fields and metadata

A v2 table (SIMCODE v2, `simcode/strict.py`) in this layout:

| field | dtype | shape | meaning |
|---|---|---|---|
| `pos` | float64 | [n, 2] or [n, 3] | final position (x, y) of the explicit pattern solve, px, sensor coordinates, not drift corrected; 3D heads (2026-11-14): + z = the head's z prior conditioned on the solved x, y (px of the lateral pixel size, relative to the focal plane of the model's PSF calibration: z = (height above the coverslip - focus_um) / pixel size) |
| `crlb` | float32 | [n, 2] or [n, 3] | calibrated sigma of x, y (px; after the model's selection-aware rescale), not a CRLB; z: sd of z after the conditioning (network sd, not recalibrated); pass-2 tables: z, its sd and cov are the pass-1 head's prior conditioned on the pass-2 x, y (NaN for rows without a linked pass-1 candidate) |
| `cov` | float32 | [n, 3] | 3D heads: covariances (xy, xz, yz) px^2 of the position (xy = 0: the solve's x, y are uncorrelated) |
| `z_prior` | float64 | [n] | 3D heads: the head's pattern-free z (px) before the conditioning |
| `frame` | int32 | [n] | window start (absolute movie frame); windows start every 3 frames |
| `n_frames` | int32 | [n] | 6: the window length (drift time = window centre, frame + 3) |
| `photons` | float32 | [n] | sum of the 6 network per-frame intensities (label units) |
| `score` | float32 | [n] | detector probability (`p` of the table; `select(min_score=)`) |
| `intensities` | float32 | [n, 6] | network per-frame intensities (`I` of the table) |
| `t` | float64 | [n] | time centroid of the window's intensities (absolute frame, fractional) |
| `q_dof`, `dnll` | float32 | [n] | intensity-domain chi2 / dof of the solve; NLL gap to the best competing fringe (pass 1) |
| `kin` | bool | [n] | on/off-kinetics model chosen |
| `x_prior`, `y_prior` | float64 | [n] | pattern-free network position (px) |
| `good`, `sel`, `disjoint` | bool | [n] | p >= 0.63; the phase-calibrated detection selection; frame % 6 == 0 |
| `x_dc`, `y_dc` | float64 | [n] | only for runs with `--drift`: drift-corrected x, y |
| `strict_u`, `strict` | | [n] | tables before 2026-11-14 only; unused |

Metadata added by the loaders: `pixel_nm` (the table's px_nm), `crop` [x0, y0, size] (output region of the run; the
selection's region), `sigma_max` (the run's selection threshold, nm), `info` (the table's info json), `run_config` (the run
record), `model`, `pass` (1 / 2), `n_frames`, `image_size`, `source`, `format`; `selection` after `select()`.

Other sources:

| source | mapping |
|---|---|
| DECODE npz of the paper (`analysis/baselines`) | phot -> `photons`, sphot -> `crlb_photons`, prob -> `score`, sx, sy -> `crlb`; `z` (nm) kept as a field (the datasets are 2D); `n_frames` 6 |
| preprint code `.npy` (Dataset / SFDataset) | estim.pos / crlb.pos -> `pos` / `crlb` (a z column -> fields `z`, `crlb_z`), estim.ibg -> `intensities`, `intensities_bg`; `chisq`, `group`, `roi_id`, `roipos`; `n_frames` = `window=` (6 for the paper's tables) |
| Picasso hdf5 (e.g. the paper's sf.hdf5) | torch_psf's `load_picasso` (lpx, lpy -> `crlb`; sx, sy -> `psf_sx`, `psf_sy`); `n_frames` = `window=` |
| earlier SIMCODE npz save (`locs_meta`, `locs_drift`, `locs_drift_applied`) | the v2 table mapping plus its metadata (px_nm -> pixel_nm, imgshape -> image_size, nframes -> n_frames) and drift |

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
