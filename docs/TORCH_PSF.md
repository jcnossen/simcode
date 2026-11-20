# SIMCODE and torch-psf

SIMCODE gets its PSF models, PSF / calibration files, in-situ PSF fits, movie readers and localization `Dataset` from
[torch-psf](https://github.com/jcnossen/torch-psf). This page states what SIMCODE relies on, so that a torch-psf change
can be checked against it, and records the migration to torch-psf's unified PSF classes (psf-unify, 2026-11-20).

## One import point (imports only)

Only `simcode/psf/__init__.py` imports `torch_psf` (checked 2026-11-20: no other module in `simcode/` does). Every other
module imports the names re-exported there; its docstring lists them. Keep it that way: new code imports from
`simcode.psf`, not from `torch_psf`.

This covers the imports only. Other modules call methods of the torch-psf objects they get from `simcode.psf`, and a
torch-psf API change can break them too. The method-level touch points (what SIMCODE calls, by module):

| Module | torch-psf objects and what is used |
|---|---|
| `simcode/psf/__init__.py` | `PSF.load(path \| dict, roisize, device, angle)`; `to_spline(model or per-angle list, kernel, value, backend, z_range, roisize, xy_limit, oversample, dz_um, zpad)`; `SplinePSF.n_angles / for_angle / resized / roisize / z_range / knots / table_map / evaluate_tables`; `FourPiPSF.from_state`, `ZMode(model, mode)`, `FourPiPSF.top / bot` (`n_med`), `channel_offsets`, `to_spline(roisize, z_range, value, kernel, backend, oversample, dz_um, zpad, xy_limit, chunk)`; `FourPiSplinePSF(roisize, z_range, knots, ..., kernel, backend)`, `.table`, `.knots`; `psfzip.read_zip`, `convert_calibration(..., integrated_spline=)`, `save_zip`, `is_psf_zip` |
| `simcode/dataset.py` | `Dataset` (subclass: `__init__(fields, meta)`, `__getitem__`, `copy`, `to`, `info`, `save_hdf5` / `load_hdf5`, `estimate_drift` -> `Drift`), `Drift(curve).at(center_frame)` |
| `simcode/pipeline/psf_fit.py` | `VectorialPSF.from_optics(optics, R, npupil, oversample, device, pixel_integration)` (torch-psf names: `'spectral'`, not SIMCODE's `'exact'`), `.resized(R)`, `.focus_um`, `.blur`, `.zcoef`; `fit_tracks` (`group_psf_kwargs` -> `dict(dipM=...)`), `heldout_tracks`, `paired_nll`, `split_checkerboard`; the `SplinePSF` of `calib_psf`: `for_angle`, `render(pos, grad)` |
| `simcode/pipeline/psf_stage.py`, `simcode/psf/beads.py` | `VectorialPSF.render(pos)`, `.zcoef / .NA / .n_med / .zidx`; `ZStackEstimator` |
| `simcode/emitter/model.py`, `simcode/simulate/{simulator, acquisition, crlb, psf_perturbation}.py` | the `SplinePSF` of `calib_psf`: `render(pos, grad)`, `resized(P)`, `roisize`, `for_angle(k)` |
| `simcode/emitter/fourpi.py`, `simcode/solve/fourpi.py`, `simcode/simulate/{multichannel, fourpi_patches}.py` | the `FourPiSplinePSF` of `fourpi_spline`: `n_channels`, `evaluate(pos, Phi, grad)`, and through `simcode.psf.fourpi_component` / `fourpi_channel` one table of its knot bank (`table.evaluate_tables`) |
| `simcode/channels/registration.py`, `simcode/pipeline/{spots, camera}.py`, `simcode/calibration/camera_maps.py`, `simcode/processing/*`, `simcode/io/movie.py` | `filter_frames`, `extract_rois`, `AstigmaticGaussianPSF`, `spot_deviance`, `fit_gaussian_free_sigma`, `white_photon_transfer`, `maps_from_movie`, `TiffMovie` / `ArrayMovie` / `open_movie`, `solve_linear` |
| `tests/test_psfzip.py`, `tests/test_heights.py`, `tests/test_psf_pixint.py`, `tests/test_dataset.py` | `SplinePSF.knots`, `FourPiPSF(pos, phi=)`, `VectorialPSF.resampled(P, pixel_integration=)`, `VectorialPSF(..., pixel_integration='spectral')`, `Drift.load` |

(`simcode/solve/pattern.py` has its own copy of the cubic B-spline kernel for the intensity map; it does not import
torch-psf.)

The 4Pi PSF is torch-psf's too: `fourpi_psf(state)` = `FourPiPSF.from_state` (+ `ZMode` for another z mode),
`fourpi_spline(state, ...)` = `FourPiPSF.to_spline` on SIMCODE's grid, `fourpi_component(spline, k, c, pos)` = one table
(component k: inc, cr, ci; channel c) of the `FourPiSplinePSF` knot bank, `fourpi_channel` = channel c at a 4Pi phase.
SIMCODE keeps three state keys of its in-situ 4Pi states that torch-psf does not read, in `simcode/psf/__init__.py`
(module doc): per-arm `n_med` in `arms` (with `bot` for torch-psf's `bottom`), `beta` (a 4Pi phase linear in z, applied to
the table's knots) and `spline` (a refined knot table plus the table change of a perturbed state); `reg` is torch-psf's
`channel_offsets`.

## The pin

`pyproject.toml` pins torch-psf to one commit. Current pin (2026-11-20): `3cf144c2d449`, the head of torch-psf's
`psf-unify` branch; it is to become the torch-psf master commit that contains psf-unify once that is merged there. The
models m1809 / m25 and the final in-situ PSF files of the paper data (`<movie>/simcode_results/psf_v4/`) were made with
the previous pin `4b0ffc3` (2026-11-19); the checks below show that the new pin renders them identically.

## What must not change: the rendered pixels

The networks are trained on images rendered by `calib_psf` (the integrated cubic B-spline of the calibration, 4 knots per
pixel, `SPLINE` and `SPLINE_KIND` in `simcode/psf/__init__.py`), and the pattern solve, the CRLB and the simulator render
from the same files at run time. A torch-psf update that changes those pixels makes a trained model inconsistent with the
solve and the simulator, without any error. So any new pin must give, for the final calibrations:

* reading the file: `PSF.load(<calib json | psf.zip>, angle=None / 0 / 1)` renders (values and x / y / z derivatives)
  identical to the current pin, bit for bit as the target and at most 1e-6 of the peak;
* building from the json: the integrated spline built with SIMCODE's grid (`SPLINE`) and the cubic B-spline kernel
  (`SPLINE_KIND`, passed explicitly, so a new torch-psf default for new tables does not change it) has the same
  coefficients and renders, under the same tolerance.

Reference files: `calib_1809_2_psf_v4_axis.{json,psf.zip}` and `calib_2p5nM_m2_psf_v4_axis.{json,psf.zip}` (per
excitation angle; psf.zip format 0.3.0 with an `integrated_spline` block).

The same holds for the 4Pi PSF (the 4Pi networks are trained on the `FourPiSplinePSF` of their checkpoint's state, and
the solve evaluates it; its interpolant `FOURPI_KIND`, Catmull-Rom of the pixel values, is passed explicitly too):
`fourpi_psf(state)` renders and `fourpi_spline(state, 15)` knots identical to the current pin for
`<4Pi work folder>/psf4pi/NPC_f_dxy15.json` and `MT_f_dxy15.json`, and `PSF.load` of `models/fourpi/NPC_f_dxy15*.psf.zip`.

The checks (run once with the current pin, `--save`, once with the new one, `--ref`):

    PYTHONPATH=<code>:<torch-psf> python tests/check_psf_pins.py --calib <calibration files> --work4pi <4Pi work folder> \
        [--plain <plain-4Pi work folder>] --save pins.npz          (then --ref pins.npz with the new pin)
    PYTHONPATH=<code>:<torch-psf> python tests/check_fourpi_T.py --work <4Pi work folder> --save ref.pt   (then --ref ref.pt)
    python -m pytest tests --work4pi <4Pi work folder>             (tests/test_psfzip.py: the zips vs their jsons)

plus a fixed-seed training batch (`simcode.simulate.batches.TrainingBatches`) and one end-to-end `simcode.run` on a short
simulated movie before / after.

## psf-unify (done 2026-11-20)

torch-psf's unified PSF classes (design note `docs/notes/psf_unify_design.md` in torch-psf; migration tables
`docs/notes/psf_unify_stage3.md` (API) and `psf_unify_stage4.md` (file versions)). What changed for SIMCODE:

| torch-psf before (4b0ffc3) | psf-unify | SIMCODE |
|---|---|---|
| `IntegratedSplinePSF.from_vectorial(vecs, z_range, roisize, xy_limit, oversample, dz_um, zpad)` | `to_spline(vecs, kernel='bspline', value='density', ...)`: one `SplinePSF` | `integrated_spline` passes `SPLINE_KIND` and the `SPLINE` grid explicitly |
| `IntegratedSplinePSF.coefs` / `coefs_angle` | `SplinePSF.knots` (bank [T, nz, n, n], per-angle tables on the `angle` axis) | tests use `knots` |
| `for_angle(k)` on a spline without angle terms: the same model | refused | `calib_psf` passes the angle only when the spline has angle terms |
| `CubicSplinePSF` | `SplinePSF(kernel='catmull', value='pixel', backend='conv')` | not re-exported any more (`SplinePSF` is) |
| `FourPiSplinePSF.luts[comp][c]` (CubicSplinePSF per table) | one `SplinePSF` bank `FourPiSplinePSF.table`, axes (phase, channel) | `fourpi_component`; `knots` is the bank [3 C, nz, n, n] (`fourpi_knots` reshapes it to [3, C, ...]) |
| `FourPiPSF.from_state(s, zmode=)`, `FourPiPSF(..., zmode=)` | `ZMode(model, mode)`; `from_state` applies the stored mode | `fourpi_psf(..., zmode=)` unchanged (wraps) |
| `FourPiPSF.to_spline(roisize, z_range, oversample, dz_um, zpad, xy_limit, chunk)` | `to_spline(roisize, z_range, **grid)` with `kernel` / `value` / `backend` | `fourpi_knots` passes `value='pixel'`, `FOURPI_KIND` |
| `FourPiPSF.forward(pos, M=, phi_extra=)` | `forward(pos, dipM=, phi=)`; one render signature `render(pos, grad, *, dipM, phi, xy, focus, angle)` | tests / analysis use `phi=` |
| `VectorialPSF.resized(P, oversample=, pixel_integration=)` | `resampled(P, ...)` (`resized(P)` keeps the sampling) | `tests/test_heights.py` |
| `VectorialPSF(..., pixel_integration='exact')` | `'spectral'` (`'exact'` only read from files) | `psf_fit.Fit` maps SIMCODE's `'exact'` (config `psf.pixel_integration`) to `'spectral'` |
| `Dataset` with drift state (`drift`, `drift_applied`, `set_drift`, `apply_drift`, `undo_drift`, `drift_at`, `estimate_drift` storing it) | drift is a result: `estimate_drift` returns a `torch_psf.Drift`, `drift_corrected(drift)`; HDF5 format_version 2 | `simcode.dataset.Dataset` keeps the drift state API unchanged on top of `Drift` (and writes / reads `/drift` in its HDF5 files) |

Checked on 2026-11-20 against the previous pin (`tests/check_psf_pins.py`, CPU and CUDA): the 2D renders and coefficient
banks of the final psf_v4 calibrations (json and zip) and of the v2 / v3 / v4 files of `models/` bit-identical; the 4Pi
models, gradients, knots and tables bit-identical; `tests/check_fourpi_T.py` (networks, simulators, solve) 130 of 130
outputs bit-identical. One difference on CUDA only: the Catmull-Rom tables (the 4Pi tables) evaluated for a few emitters
at once (about 1 - 16 on an RTX 3090) differ by up to 2e-7 of the peak in value and up to 7e-6 of the d/dz peak, because
torch-psf now computes the per-axis weights `[1, t, t^2, t^3] @ G` one axis at a time (cuBLAS picks another kernel for the
smaller product); CPU and larger batches are bit-identical. The integrated B-spline (2D) is not affected.

## Files SIMCODE writes

* psf.zip: torch-psf declares the format version per class (stage 4). What SIMCODE writes is unchanged: the calibration
  zips of `write_calib_zip` (the psf stage's `calib.psf.zip`) are 0.3.0 with an `integrated_spline` channel (a Gaussian
  calibration 0.2.0), the 4Pi zips (`FourPiPSF` / its `phase_spline` table) 0.2.0; their `psf.json` and arrays are
  byte-identical to those of the previous pin (only the text of `fit_report.pdf` differs), so older SIMCODE / torch-psf
  versions read them. torch-psf now also writes 0.4.0 (non-box 4Pi pixel integration, dipole component tables) and 0.5.0
  (an explicit `spline_table` block: Catmull-Rom density, B-spline pixel or per-angle pixel tables); SIMCODE builds none
  of these, and readers before psf-unify refuse them. Reading: psf.zip 0.2 - 0.5.
* Localization tables (`locs_p{1,2}_{net,solve}.hdf5`, `Dataset.save`): torch-psf dataset format_version 2 (the version
  attribute; the layout of SIMCODE's tables is unchanged, older readers do not check the version). A `Dataset` with a
  drift is saved with the `/drift` dataset (attributes `applied`, `info`) as before; torch-psf's own reader takes the
  record of an applied drift and warns about an unapplied curve (`torch_psf.Drift.load(path)` reads it).
