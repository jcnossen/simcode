# Using SIMCODE v2 results: the `Dataset` class

`simcode.dataset.Dataset` holds the localizations of one sample. It is `torch_psf.Dataset` (fields as CPU tensors,
metadata, drift; indexing, filters, crop, concatenate, drift correction, render, FRC, HDF5 / Picasso / ThunderSTORM files:
torch-psf `docs/API.md`) with SIMCODE's loaders and the paper's selection. Field layout and SIMCODE's fields:
`docs/DATASET_FORMAT.md`. (Until 2026-11-14 it was a numpy class with columns x, y, sx, sy, p, I; those are now `pos`,
`crlb`, `score`, `intensities`.)

## Run a movie, load the result

```bash
export CUDA_VISIBLE_DEVICES=0
python -m simcode.run --movie /path/to/movie.tif --calib insitu/calib_1809_2_psf_v2.psf.zip --model m1809 --out out/run1
python -m simcode.run --dataset 1809_2 --data-root /mnt/data/dmdsf/bn_dnapaint --out out/run1   # a paper dataset
```
Paths are always arguments (there is no config file and no path environment variable): `--movie` a file, `--out` the
output folder, `--model` a name of `simcode/models.json` or a model folder; calibration and network names are looked up in
the repository's `models/` folder.

```python
from simcode.dataset import Dataset

ds = Dataset.from_run('out/run1')               # the result table locs_p2_solve.hdf5 (the pass-2 pattern solve) + its info json (+ run_config.json)
ds = Dataset.load('out/run1')                   # the same; also takes a table file
nt = Dataset.from_run('out/run1', pass_=2, kind='net')     # the pass-2 emitter network's own output (same rows); pass_=1: the pass-1 tables
print(ds)                                       # rows, frames, pixel size, median precision and photons, drift, fields
ds.pos, ds.crlb, ds['score'], ds.keys()         # fields (tensors); ds.meta holds the metadata; ds.numpy(): dict of arrays
```

With the automatic pipeline (`simcode.pipeline`): `ds = pipe.dataset_pass2()` (or `pipe.dataset_pass1()`; `kind='net'` for the
network tables) gives the same object.
`Dataset.from_run(d, columns=('sx', 'sy', 'p'))` loads only those table columns (x, y, frame always).

## Select and filter

```python
sel = ds.select()                               # the paper's selection: max(crlb) * pixel_nm <= 12 nm, score >= 0.63, in the crop
tight = ds.select(max_precision_nm=8.0)         # another threshold (nm); min_score=..., in_region=False; None = no condition
m = ds.mask()                                   # the boolean mask (= simcode.table.select on the table)
full = ds[ds.sel]                               # the field sel: the recommended full output for renders and drift
sub = ds.filter_photons(500, 5000).filter_frames(0, 30000).crop(20, 20, 100, 100)
sub = ds.filter_crlb(max_nm=10)                 # torch_psf filters; in-place variants end with _
both = Dataset.concatenate([a, b])              # or a + b
ds.add_fields(cluster=labels)                   # extra fields: kept by indexing, concatenate, save, load
```

## Drift

```python
from simcode.util.drift import dme_package; dme_package()     # the installed DME v2 package (pip install -e <drift-estimation>)
ds.estimate_drift(rows=ds.good & ds.disjoint)   # DME v2 on the good, non-overlapping windows -> ds.drift [frames, 2] px
sel = ds.select().apply_drift()                 # the drift goes with every object indexed from ds; x - drift[frame + 3]
sel.undo_drift()
ds.set_drift(np.load('drift.npy'))              # or use a saved estimate
```

## Render, FRC, files

```python
img = sel.render(pixel_nm=10)                   # 2D histogram [ny, nx]
res = sel.frc()                                 # nm; halves = alternating 300-frame blocks, 1/7 threshold (= simcode.evaluation.frc)
sel.save('sel.hdf5')                            # torch-psf format (docs/DATASET_FORMAT.md of torch-psf)
again = Dataset.load('sel.hdf5')                # also torch_psf.Dataset.load('sel.hdf5') without SIMCODE
sel.save_picasso('sel_picasso.hdf5')            # Picasso (+ .yaml); sel.save('sel.csv'): ThunderSTORM
```

Other tables: `Dataset.load('smlm.npy', window=6)` (preprint code), `Dataset.load('sf.hdf5', window=6)` (Picasso),
`Dataset.load('decode/locs.npz')`. `window` = frames per localization (the `n_frames` field; drift time = frame +
window // 2).

Command line: `python -m simcode.dataset out/run1 sel.hdf5 12` prints the summary and writes the selection (<= 12 nm).
