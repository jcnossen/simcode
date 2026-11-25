"""Estimate a planar bead movie and export psf.zip, bead-fit.npz and modulation.json.

From a clone of this SIMCODE repository, with Python, Node and npm installed:

    git clone https://github.com/jcnossen/torch-psf ../torch-psf
    python -m pip install -e ../torch-psf
    npm ci --prefix ../torch-psf/web
    python examples/estimate_bead_slide.py movie.tif --output fit

The TIFF must be grayscale TYX, at least 96 x 96 pixels, with five or more
complete illumination cycles. Defaults match the web example (A0 A120 A240
B0 B120 B240). Set the optical and camera arguments for your acquisition.
Use --structured-beads for a ring or other spatially correlated sample.

TorchPSF fits the shared vectorial PSF, bead positions, frame photometry and
XYZ drift in Python. The modulation step calls the web app's own estimator
so both use identical field maps and the same optional image-power depth fit.
"""
import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
import torch_psf
from torch_psf import DenseBeadEstimator
from torch_psf.psfzip import save_zip


def estimate_modulation(movie, fit, optics, schedule, *, random_beads=True):
    """Estimate field maps from the fitted bead traces and optional random-field image power."""
    # Compile the shared TypeScript estimator using TorchPSF's web dependencies.
    package = Path(torch_psf.__file__).resolve().parents[1] / "web/node_modules/esbuild"
    if not package.is_dir():
        raise RuntimeError("Install TorchPSF in editable mode and run npm ci --prefix ../torch-psf/web.")
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        cli = work / "modulation.cjs"
        source = Path(__file__).with_name("bead_slide_modulation.ts")
        subprocess.run(["node", "-e",
                        "require(process.argv[1]).buildSync({entryPoints:[process.argv[2]],"
                        "platform:'node',bundle:true,outfile:process.argv[3]})",
                        str(package), str(source), str(cli)], check=True)
        snapshot = dict(width=movie.shape[2], height=movie.shape[1], xy=fit.xy.tolist(),
                        intensity=fit.photons.tolist(), drift=fit.drift.tolist(),
                        settings=dict(diameterNm=1000 * optics["diameter_um"], optics=optics))
        (work / "input.json").write_text(json.dumps(dict(snapshot=snapshot, schedule=schedule)))
        if random_beads:
            movie.astype("<f4", copy=False).tofile(work / "pixels.f32")
        subprocess.run(["node", str(cli), str(work), str(work / "modulation.json")], check=True)
        return json.loads((work / "modulation.json").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("movie", type=Path)
    p.add_argument("--output", type=Path, default=Path("bead-slide-fit"))
    p.add_argument("--pixel-nm", type=float, default=108.33)
    p.add_argument("--diameter-nm", type=float, default=40)
    p.add_argument("--wavelength-nm", type=float, default=700)
    p.add_argument("--na", type=float, default=1.49)
    p.add_argument("--immersion-index", type=float, default=1.518)
    p.add_argument("--sample-index", type=float, default=1.33)
    p.add_argument("--gain", type=float, default=1, help="ADU per photoelectron")
    p.add_argument("--offset", type=float, default=0, help="camera offset in ADU")
    p.add_argument("--axes", type=int, default=2)
    p.add_argument("--phases", type=int, default=3)
    p.add_argument("--interleaved", action="store_true")
    p.add_argument("--estimate-phases", action="store_true")
    p.add_argument("--structured-beads", action="store_true")
    args = p.parse_args()

    if args.gain <= 0 or args.axes < 1 or args.phases < 3:
        p.error("Use a positive gain, at least one axis and at least three phases for modulation.")
    movie = tifffile.imread(args.movie).astype(np.float32)
    period = args.axes * args.phases
    if movie.ndim != 3 or min(movie.shape[1:]) < 96 or len(movie) < 5 * period:
        p.error("Vectorial movie calibration requires TYX, at least 96 x 96 pixels and five cycles.")
    movie -= args.offset
    movie /= args.gain
    optics = dict(pixel_um=args.pixel_nm / 1000, diameter_um=args.diameter_nm / 1000,
                  wavelength_um=args.wavelength_nm / 1000, NA=args.na,
                  n_med=args.sample_index, n_imm=args.immersion_index, focus_um=0.1)
    args.output.mkdir(parents=True, exist_ok=True)
    fit = DenseBeadEstimator.fit_movie(movie, np.arange(len(movie)) % period, optics=optics, radius=6)
    cfg = getattr(getattr(fit.psf, "optic", None), "cfg", {})
    if cfg.get("calibrated_psf") is None:
        raise ValueError("The movie did not yield a calibrated pupil. Check focus diversity and phase order.")
    save_zip(str(args.output / "psf.zip"), cfg["calibrated_psf"])
    np.savez_compressed(args.output / "bead-fit.npz", xy_px=fit.xy, photons=fit.photons,
                        drift_xy_px=fit.drift, drift_z_um=fit.axial_drift_um, background=fit.background)
    schedule = dict(axes=args.axes, phases=args.phases, interleaved=args.interleaved,
                    estimatePhases=args.estimate_phases)
    result = estimate_modulation(movie, fit, optics, schedule, random_beads=not args.structured_beads)
    (args.output / "modulation.json").write_text(json.dumps(result, indent=2))



if __name__ == "__main__":
    main()
