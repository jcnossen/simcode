# SIMCODE and torch-psf

SIMCODE gets its PSF models, PSF / calibration files, in-situ PSF fits, movie readers and localization `Dataset` from
[torch-psf](https://github.com/jcnossen/torch-psf). This page states what SIMCODE relies on, so that a torch-psf change
can be checked against it, and lists the pending migration to torch-psf's unified PSF classes.

## One import point

Only `simcode/psf/__init__.py` imports `torch_psf` (checked 2026-11-19: no other module in `simcode/` does). Every other
module uses the names re-exported there; its docstring lists them. A change of the torch-psf API is therefore a change
of that one file. Keep it that way: new code imports from `simcode.psf`, not from `torch_psf`.

(`simcode/solve/pattern.py` has its own copy of the cubic B-spline kernel for the intensity map; it does not import
torch-psf.)

## The pin

`pyproject.toml` pins torch-psf to one commit. Current pin: `4b0ffc3` (2026-11-19). The models m1809 / m25 and the final
in-situ PSF files of the paper data (`<movie>/simcode_results/psf_v4/`) were made with this commit.

## What must not change: the rendered pixels

The networks are trained on images rendered by `calib_psf` (the integrated cubic B-spline of the calibration, 4 knots per
pixel, `SPLINE` in `simcode/psf/__init__.py`), and the pattern solve, the CRLB and the simulator render from the same
files at run time. A torch-psf update that changes those pixels makes a trained model inconsistent with the solve and
the simulator, without any error. So any new pin must give, for the final calibrations:

* reading the file: `PSF.load(<calib json | psf.zip>, angle=None / 0 / 1)` renders (values and x / y / z derivatives)
  identical to the current pin, bit for bit as the target and at most 1e-6 of the peak;
* building from the json: the integrated spline built with SIMCODE's grid (`SPLINE`) and the **cubic B-spline kernel**
  has the same coefficients and renders, under the same tolerance.

Reference files: `calib_1809_2_psf_v4_axis.{json,psf.zip}` and `calib_2p5nM_m2_psf_v4_axis.{json,psf.zip}` (per
excitation angle; psf.zip format 0.3.0 with an `integrated_spline` block).

## Pending: torch-psf's unified PSF classes (psf-unify)

torch-psf is being refactored into a unified PSF design (branch `psf-unify`; design note
`docs/notes/psf_unify_design.md` in torch-psf). It renames classes without aliases (e.g. `IntegratedSplinePSF` becomes
one `SplinePSF`; `convert_calibration` / `save_zip` may move), writes psf.zip 0.4.0 and may change the default spline for
newly built tables. Every rename gets a documented replacement, existing files keep reading with the same pixels (checked
on the files above at every refactor stage), and nothing requires a retrain.

When psf-unify is merged into torch-psf master, SIMCODE must be updated on its own branch before its pin moves:

1. Adapt `simcode/psf/__init__.py` (and only it) to the renamed torch-psf names; keep SIMCODE's own names
   (`calib_psf`, `integrated_spline`, `vectorial_psf`, `read_calib`, `write_calib_zip`, ...) unchanged.
2. When building splines, pass the kernel (`'bspline'`) and the `SPLINE` grid explicitly, so a new torch-psf default
   for new tables does not change SIMCODE's renders. Keep this until the next retrain.
3. Run the two pixel checks above against the current pin.
4. Run the full test suite (CPU and GPU) and one end-to-end `simcode.run` on a short simulated movie.
5. Then bump the pin in `pyproject.toml`. If step 3 fails, do not bump: report the difference.

Users: psf.zip files written after the bump will be format 0.4.0, which torch-psf versions before psf-unify refuse.
