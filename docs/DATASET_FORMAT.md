# Dataset format (draft for approval, 2026-11-14)

One file format for every localization result: SIMCODE v2 (2D SIM), 4Pi-SIMFLUX / 4Pi-SIMCODE (3D), astigmatic 3D,
classical SMLM / SIMFLUX fits, DECODE, simulations. Read and written only through `simcode.dataset.Dataset`.
A selection is never stored; it is computed on load (`ds.select(...)`). Other estimates of the same localizations
(the network's PSF-only position, the detector position, pass 1) are separate datasets, not extra columns.

## Container

One HDF5 file per dataset, `<name>.hdf5`:

| path | content |
|---|---|
| `/localizations` | one column per field (table, one row per localization) |
| `/metadata` | attributes (below) |
| `/drift` | optional, drift trace [n_frames, n_dims] in px, plus attributes: method per axis, applied yes / no |

## Fields

Required for every dataset (as the v1 Dataset: one position array and one precision array, 2D or 3D):

| field | dtype | unit | meaning |
|---|---|---|---|
| `id` | int64 | | localization id; the same id in related datasets of one run (final, prior, pass 1) links the rows |
| `frame` | int32 | frames | first frame of the localization's window (single-frame methods: the frame) |
| `pos` | float64 [n, D] | px | position (x, y) or (x, y, z), D = `metadata.n_dims`; pixel centres at integer coordinates, x = column; z relative to the focal plane of the PSF calibration (4Pi: the common focal plane of the aligned system), in px of the lateral pixel size |
| `crlb` | float32 [n, D] | px | reported precision per axis (SIMCODE: calibrated sigma; fits: CRLB) |
| `photons` | float32 | e- | total signal in the window |
| `bg` | float32 | e- / px / frame | background |

Optional for 3D: `cov` float32 [n, 3] = (xy, xz, yz) covariances of the position estimate, px^2.

Optional, by method:

| field | dtype | meaning |
|---|---|---|
| `score` | float32 | detection score 0-1 (uncalibrated network score; absent for fits) |
| `chi2` | float32 | goodness of fit per degree of freedom |
| `intensities` | float32 [n, frames, channels] | per-frame (per-channel) intensities, e-; layout in `metadata.intensity_layout` |
| `blink` | bool | the emitter switched on / off within the window |
| `branch` | int16 | ambiguity index (4Pi fringe / phase branch); `branch_margin` float32 = how clearly it was chosen |
| `group` | int64 | localizations of one binding event linked across windows |
| user fields | any | added with `add_fields`; carried through indexing, merge, save / load |

## Metadata (attributes of `/metadata`)

`format_version`, `pixel_size_nm`, `image_size` [h, w], `n_frames`, `window_length`, `window_step`, `n_dims`,
`n_channels`, `intensity_layout`, `method` (e.g. 'simcode-v2 pass 2',
'smlm', 'simflux', 'decode', '4pi-simcode'), `estimate` ('final' | 'prior' | 'detector' | 'pass1'), `model`,
`source_run` (path) and `run_config_sha256`, `software_version`, `git_commit`, `created`, `converted_from` (path + sha256,
for files converted from older formats).

## Where results live: `runs/<run name>/` next to the movie

Every run gets a name (`name` in its config, required) and is written to a `runs/` folder in the movie's own directory:

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

A run that reuses a model refers to it by path (e.g. `../<other run>/model` or a model in another movie's runs folder) and
records its hashes; the model itself is not copied. `LocalizationPipeline` returns the datasets as `pipe.dataset()`
(final) and `pipe.dataset(estimate='prior')` etc.

## Export (not the storage format)

`ds.save_picasso(path)` (Picasso hdf5: precision in lpx / lpy, PSF width in sx / sy) and `ds.save_csv(path)`
(ThunderSTORM conventions: nm, frames from 1). `save_npz` for quick numpy use.
