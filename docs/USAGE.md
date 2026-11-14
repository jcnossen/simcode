# Using SIMCODE v2 results: the `Dataset` class

`simcode.dataset.Dataset` holds the localizations of one sample: columns as numpy arrays, metadata and the drift. It is the
counterpart of the preprint code's `Dataset` / `SFDataset` (the same methods, with snake_case names: `save_hdf5`,
`load_hdf5`, `num_frames`, `apply_drift`, ...) and it reads the files of that code, of Picasso, of ThunderSTORM and the
paper's DECODE tables. Every method has a docstring; the module docstring lists units and formats. (The camelCase names and
the `Localizations` / `simcode.locs` aliases were removed on 2026-11-14.)

## Units

| field | unit / meaning |
|---|---|
| `x`, `y` | camera pixels, sensor coordinates of the movie; an integer is a pixel centre; `x` = column, `y` = row |
| `sx`, `sy` | localization precision (one sigma) in camera pixels; `ds.precision_nm` = max(sx, sy) · px_nm |
| `frame` | first movie frame of the localization (0-based); v2 localizes 6-frame windows that start every 3 frames |
| `photons` | sum of the network per-frame intensities of the window; `I` = the `[n, 6]` per-frame intensities |
| `p` | the detector score (`select(min_score=...)`) |
| `good`, `sel`, `disjoint`, `kin`, ... | the other columns of the v2 table (`simcode/strict.py`); older tables also have `strict_u`, `strict` (unused) |
| `ds.px_nm` | camera pixel size in the sample (nm); `ds.pos_nm` = positions in nm |
| `ds.drift` | `[frames, 2]` camera pixels; corrected x = x − drift[frame + window // 2] |

## Run a movie, load the result

```bash
export CUDA_VISIBLE_DEVICES=1
python -m simcode.run --movie movie.tif --calib insitu/calib_1809_2_psf3d.json --model m1809 --out out/run1
```

```python
from simcode.dataset import Dataset

ds = Dataset.from_run('out/run1', pass_=2)      # locs_p2.npz + its info json (+ run_config.json when the run has one)
ds = Dataset.load('out/run1')                   # the same (pass 2 if it exists, else pass 1); also takes a table file
print(ds)                                       # rows, frames, pixel size, median precision, selection counts, columns
ds.info()                                       # the same as a dict
ds.x, ds.photons, ds['p']                       # columns; ds.columns lists them; ds.meta holds the metadata
```

With the automatic pipeline (`simcode.auto`): `ds = pipe.dataset_pass2()` (or `pipe.dataset_pass1()`) gives the same object.

A full production table (6 million rows, 660 MB) loads in a few seconds; `Dataset.from_run(d, columns=('sx', 'sy', 'p'))`
loads only what is needed (`x`, `y`, `frame` always).

## Select and filter

```python
sel = ds.select()                               # the paper's selection: max(sx, sy) <= 12 nm, p >= 0.63, inside the run's crop
tight = ds.select(max_precision_nm=8.0)         # another threshold (nm); min_score=..., in_region=False; None = no condition
m = ds.mask()                                   # the boolean mask (= simcode.strict.select(table, ...))
full = ds.filter(ds.sel)                        # the column sel: the recommended full output for renders and drift

sub = ds[m]                                     # masks, slices and index arrays give a new Dataset
sub = ds.filter_photons(500, 5000).filter_frames(0, 30000).filter_region(20, 20, 100, 100)
sub = ds.crlb_filter(0.15)                      # max(sx, sy) < 0.15 px, the preprint's filter
roi = ds.crop(40, 40, size=20)                  # x, y in [40, 60) px; shift=True moves the origin there
both = Dataset.concatenate([a, b])              # or a + b
ds.add_fields(cluster=labels)                   # extra per-localization fields: kept by indexing, concatenate, save, load
ds.select_(); ds.crop_(40, 40, size=20)         # in place (trailing underscore): filter_, crop_, crop_from_edge_, select_,
                                                # crlb_filter_, autocrop_; they return the object itself
```

Filters, crops and selections return a new object; the original is unchanged. The variants with a trailing underscore
change the object itself (the PyTorch convention).

## Drift

```python
ds.estimate_drift()                             # DME v2 on the good, non-overlapping windows -> ds.drift [frames, 2] px
sel = ds.select().apply_drift()                 # the drift goes with every object indexed from ds
sel.undo_drift()                                # back to sensor coordinates; sel.is_drift_corrected tells the state
ds.set_drift(np.load('drift.npy'))              # or use a saved estimate
```

Estimate on the full table and select afterwards. `estimate_drift` needs the `dme` package (see the README) and a GPU.

## Render and FRC

```python
sel = ds.select().apply_drift()
img = sel.render(pixel_nm=10)                               # 2D histogram [ny, nx], whole image or the cropped region
img, ext = sel.render(pixel_nm=5, region=(40, 40, 60, 60), sigma_nm=5, return_extent=True)
res = sel.frc()                                             # nm; halves = alternating 300-frame blocks, 1/7 threshold
res, freq, curve = sel.frc(return_curve=True)
```

`frc` is `simcode.evalproto.frc.frc`, the implementation behind the paper's numbers (the paper's method comparison adds its
own row protocol: `analysis/paper/frc_fair.py`).

## Save and load

```python
sel.save('sel.npz')          # v2 table format, lossless (all columns, metadata, drift); = sel.save_npz(...)
sel.save('sel.hdf5')         # Picasso (+ sel.yaml); = sel.save_hdf5(...)
sel.save('sel.csv')          # ThunderSTORM; = sel.save_csv_thunderstorm(...)
sel.save('sel.npy')          # the preprint code's numpy layout; = sel.save_npy(...)
again = Dataset.load('sel.hdf5')   # by extension: load_npz, load_hdf5, load_csv_thunderstorm, load_npy
```

- **Picasso hdf5**: `lpx`, `lpy` = this table's `sx`, `sy` (the localization precision). Picasso's own `sx`, `sy` are PSF
  widths: they hold a nominal 1.3 px (not measured) unless the table has `psf_sx`, `psf_sy`. Positions are float32 camera
  pixels, drift-corrected if the drift is applied. All other columns are written as extra columns (`extra=False`: only
  Picasso's).
- **ThunderSTORM csv**: nm, frames counted from 1, centre of the first pixel at 0.5 px (converted both ways).
- Other tables: `Dataset.load('smlm.npy', window=6)` (preprint code), `Dataset.load('sf.hdf5', window=6)`,
  `Dataset.load('decode/locs.npz')`, `Dataset.from_estimates(x, y, frame, sx, sy, photons=...)`. `window` = frames per
  localization (it sets the time of a row for the drift: frame + window // 2).

Command line: `python -m simcode.dataset out/run1 sel.hdf5 12` prints the summary and writes the selection (<= 12 nm).
