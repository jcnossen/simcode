# SIMCODE quick start

To run SIMCODE on your own DMD-SIMFLUX data, you install the package, describe your setup in one config file, and run the automatic pipeline. The pipeline calibrates the camera and the PSF from your movie, trains a model for your PSF and localizes the movie.

## 1. Install

You need:

- a Linux machine with an NVIDIA GPU (CUDA, at least 10 GB of free memory for the training and processing stages);
- Python 3.10 or later (we test with 3.11);
- PyTorch with CUDA.

Your movies must use the SIMCODE pattern scheme: 2 pattern orientations × 3 phase steps, one per frame.

```bash
git clone https://github.com/qnano/simcode && cd simcode
pip install -e .
python -m simcode.download_models                  # weights of the example models m1809, g13
python -m simcode.download_models --set pretrain   # needed to train your own model
```

Check the install. The test takes about 3 minutes on an RTX 3090:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_fast_paths.py
```

To try the whole chain before using your own data, simulate a small example movie (6000 frames of 120 x 120 px, about 10 s) with a matching config, and run the pipeline on it with `--quick` (sections 2 and 3 explain the config and the stages):

```bash
python examples/simulate_movie.py /data/simcode_example
python -m simcode.pipeline /data/simcode_example/quickstart.yaml --quick
```

## 2. Describe your setup

Write one YAML file per movie. A key you leave out takes the default of our setup: NA 1.49 oil, 700 nm emission, 108.33 nm pixels, sCMOS camera, blocked frame order. The camera gain and offset are the exception: left out, they are taken from the movie. A misspelled key is reported as an error.

```yaml
name: "my_movie"           # quote it if it looks like a number
movie: /data/my_movie.tif
out: /data/simcode_out/my_movie
optics:
  pixel_um: 0.10833        # pixel size in the sample
  NA: 1.49
  wavelength_um: 0.70      # emission
  n_imm: 1.518
  n_med: 1.33
camera:
  gain_e_per_adu: 0.56     # leave out to have it measured from the movie
  offset_adu: 99.3         # leave out: estimated from the movie (with a warning), or give dark frames: dark: /data/dark.tif
pattern:
  frame_order: blocked     # blocked: 3 steps of orientation 1, then 3 of orientation 2; interleaved: orientations alternate
  pitch_px: [1.6, 2.8]     # search range of the pattern period (camera pixels)
```

No bead z-stack is needed: the PSF is estimated from the movie itself (section 3). Bead stacks are optional, as an extra starting point for the PSF fit (`beads:`, see `simcode/pipeline/configs/1809_2.yaml`).

To see the settings the pipeline will use, without running anything or writing any file, run `python -m simcode.pipeline my_movie.yaml --show`. A worked example of our own setup is in `simcode/pipeline/configs/1809_2.yaml`.

## 3. Run the pipeline

One command does everything, from the calibration to the localizations:

```bash
python -m simcode.pipeline my_movie.yaml
```

The stages run in this order. Each stage writes its own folder under `out`.

1. **order, camera:** the frame order of the pattern, and the camera gain, offset and read noise.
2. **spots, cammaps, tracks:** finds the isolated emitters in the movie and makes the per-pixel camera maps.
3. **psf:** estimates the PSF of your microscope from the movie itself (below). Steps 1–3 took about 20 minutes on our movie.
4. **train:** trains the detector and emitter networks on simulations of your PSF and camera. This takes by far the longest: about 7 hours on one RTX 3090.
5. **process:** localizes the movie. Our 120000-frame movie took about 1 hour on one RTX 3090. The tables go to `out/process/run_<id>/`.
6. **report:** writes `out/report/report.html`, with the checks of every stage, the PSF, the pattern and the FRC.

How the PSF is estimated, without a z-stack:

- The spots and tracks stages find isolated emitters in the movie and cut out their blinking events over consecutive frames.
- The psf stage fits a vectorial PSF model to those events with torch-psf's in-situ fit: Zernike aberrations plus focus, then a blur term, then the polarization (photoselection) terms. It starts from several focus guesses and keeps the best fit.
- The same fit gives the first estimate of the illumination pattern.
- A held-out check refits on half of the events and tests on the other half. The result, `out/psf/calib.json`, is the PSF the networks are trained on. `docs/PSF_POLARIZATION.md` has the details.

Useful to know:

- Each stage ends with checks. A failed check stops the run and says why. Add `--force` to continue anyway.
- You can stop a run at any time. When you run the same command again, it skips the finished stages (and, in the train stage, the networks that are already trained).
- `--redo psf` redoes that stage and every stage after it.
- `--stages` runs only some stages, for example `--stages order,camera`.
- The pipeline uses GPU 0. With `--gpus 1,0` (or `gpus: [1, 0]` in the config) each stage takes the first GPU in that list with enough free memory.
- `--quick` trains only a few steps per network. Use it once to test the whole chain; the quick model is not usable. Then start the real run with `--redo train`, because the train stage of the quick run counts as finished.
- Already have a model for this microscope? Put `train: {model: <model folder>}` in the config to skip the training. For example, use the `out/train` folder of an earlier run. The other stages still run, so the movie gets its own camera and PSF calibration. The models we ship (`m1809`, `g13`) only fit our own setup.
- To localize more movies with a model you already have, there are two ways:
  - Write a config with `train: {model: ...}` as above. This is the recommended way.
  - Run the localization directly with the first movie's calibration:

    ```bash
    python -m simcode.run --movie <movie.tif> --calib <out>/psf/calib.json --model <out>/train --out <dir>
    ```

## 4. Use the results

Each run writes four tables:

- `locs_p1_net` and `locs_p1_solve`: pass 1;
- `locs_p2_net` and `locs_p2_solve`: pass 2, which is pattern-aware.

`_net` is the emitter network's own position and σ, and `_solve` is the explicit pattern solve on the network intensities. The two tables of a pass have the same rows. By default you load `locs_p2_net`, the result table. Add `kind='solve'` to load the pass-2 solve, which is the table the paper's figures use. Add `pass_=1` to load pass 1.

The tables are HDF5 files in the torch-psf format. You can load, filter, drift-correct, render and export them in Python:

```python
from simcode.pipeline import LocalizationPipeline
ds = LocalizationPipeline('my_movie.yaml').dataset_pass2()   # after a pipeline run; dataset_pass2(kind='solve')

from simcode.dataset import Dataset
ds = Dataset.load('<out dir>')               # the --out folder of python -m simcode.run; Dataset.load(dir, kind='solve')

sel = ds.select()                            # precision <= 12 nm, detection p >= 0.63, inside the crop; ds.select(16) for 16 nm
ds.estimate_drift(rows=ds.good & ds.disjoint)   # DME drift (a few minutes)
sel = ds.select().apply_drift(ds.drift)
img = sel.render(pixel_nm=10)                # image array: plt.imshow(img)
print(sel.frc())                             # resolution estimate (nm)
sel.save_picasso('my_movie_picasso.hdf5')    # or sel.save('my_movie.csv') for ThunderSTORM
```

There is more in the repository:

- `README.md`: all run options and output files;
- `docs/USAGE.md`: working with the tables;
- `docs/METHOD.md`: how it works.
