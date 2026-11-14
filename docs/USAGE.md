# Using SIMCODE v2 results: the `Dataset` class

`simcode.dataset.Dataset` holds the localizations of one sample: columns as numpy arrays, metadata and the drift. It is the
counterpart of the preprint code's `Dataset` / `SFDataset` (same method names where they make sense) and it reads the files
of that code, of Picasso, of ThunderSTORM and the paper's DECODE tables. `Localizations` is an alias
(`from simcode.locs import Localizations`). Every method has a docstring; the module docstring lists units and formats.

## Units

| field | unit / meaning |
|---|---|
| `x`, `y` | camera pixels, sensor coordinates of the movie; an integer is a pixel centre; `x` = column, `y` = row |
| `sx`, `sy` | localization precision (one sigma) in camera pixels; `ds.precision_nm` = max(sx, sy) · px_nm |
| `frame` | first movie frame of the localization (0-based); v2 localizes 6-frame windows that start every 3 frames |
| `photons` | sum of the network per-frame intensities of the window; `I` = the `[n, 6]` per-frame intensities |
| `p`, `strict_u`, `good`, `sel`, `strict`, `disjoint`, `kin`, ... | the other columns of the v2 table (`simcode/strict.py`) |
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
ds.x, ds.photons, ds['strict_u']                # columns; ds.columns lists them; ds.meta holds the metadata
```

With the automatic pipeline (`simcode.auto`): `ds = pipe.dataset_pass2()` (or `pipe.dataset_pass1()`) gives the same object.

A full production table (6 million rows, 660 MB) loads in a few seconds; `Dataset.from_run(d, columns=('sx', 'sy', 'strict_u'))`
loads only what is needed (`x`, `y`, `frame` always).

## Select and filter

```python
main = ds.select('main')                        # the paper's selection: ranked rows with max(sx, sy) <= 12 nm
tight = ds.select('main', sigma_max=8.0)        # another threshold (nm)
u = ds.select('u')                              # the earlier per-tile rule strict_u <= 0.45;  also 'strict', 'sel', 'good'
m = ds.mask('main')                             # the boolean mask (= simcode.strict.select)

sub = ds[m]                                     # masks, slices and index arrays give a new Dataset
sub = ds.filter_photons(500, 5000).filter_frames(0, 30000).filter_region(20, 20, 100, 100)
sub = ds.crlb_filter(0.15)                      # max(sx, sy) < 0.15 px, the preprint's filter
roi = ds.crop(40, 40, size=20)                  # x, y in [40, 60) px; shift=True moves the origin there
both = Dataset.merge([a, b])                    # or a + b
ds.addFields(cluster=labels)                    # extra per-localization fields: kept by indexing, merge, save, load
```

Filters, crops and selections return a new object; the original is unchanged.

## Drift

```python
ds.estimateDrift()                              # DME v2 on the good, non-overlapping windows -> ds.drift [frames, 2] px
main = ds.select('main').applyDrift()           # the drift goes with every object indexed from ds
main.undoDrift()                                # back to sensor coordinates; main.isDriftCorrected tells the state
ds.set_drift(np.load('drift.npy'))              # or use a saved estimate
```

Estimate on the full table and select afterwards. `estimateDrift` needs the `dme` package (see the README) and a GPU.

## Render and FRC

```python
main = ds.select('main').applyDrift()
img = main.render(pixel_nm=10)                              # 2D histogram [ny, nx], whole image or the cropped region
img, ext = main.render(pixel_nm=5, region=(40, 40, 60, 60), sigma_nm=5, return_extent=True)
res = main.frc()                                            # nm; halves = alternating 300-frame blocks, 1/7 threshold
res, freq, curve = main.frc(return_curve=True)
```

`frc` is `simcode.evalproto.frc.frc`, the implementation behind the paper's numbers (the paper's method comparison adds its
own row protocol: `analysis/paper/frc_fair.py`).

## Save and load

```python
main.save('main.npz')        # v2 table format, lossless (all columns, metadata, drift); other v2 readers still read it
main.save('main.hdf5')       # Picasso (+ main.yaml); = main.saveHDF5(...)
main.save('main.csv')        # ThunderSTORM; = main.saveCSVThunderStorm(...)
main.save('main.npy')        # the preprint code's numpy layout; = main.saveNumpy(...)
again = Dataset.load('main.hdf5')
```

- **Picasso hdf5**: `lpx`, `lpy` = this table's `sx`, `sy` (the localization precision). Picasso's own `sx`, `sy` are PSF
  widths: they hold a nominal 1.3 px (not measured) unless the table has `psf_sx`, `psf_sy`. Positions are float32 camera
  pixels, drift-corrected if the drift is applied. All other columns are written as extra columns (`extra=False`: only
  Picasso's).
- **ThunderSTORM csv**: nm, frames counted from 1, centre of the first pixel at 0.5 px (converted both ways).
- Other tables: `Dataset.load('smlm.npy', window=6)` (preprint code), `Dataset.load('sf.hdf5', window=6)`,
  `Dataset.load('decode/locs.npz')`, `Dataset.from_arrays(x, y, frame, sx, sy, photons=...)`. `window` = frames per
  localization (it sets the time of a row for the drift: frame + window // 2).

Command line: `python -m simcode.dataset out/run1 main.hdf5 main` prints the summary and writes the main selection.
