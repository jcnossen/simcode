"""Simulate a planar bead movie, recover the PSF and illumination, and plot the fit.

Install TorchPSF from a source checkout (its reference simulator lives in study/):
    git clone https://github.com/jcnossen/torch-psf ../torch-psf
    python -m pip install -e ../torch-psf
    python -m pip install matplotlib
    python examples/estimate_bead_slide.py

Outputs are written to bead-slide-example/ under the current working directory.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch_psf
from torch_psf import DenseBeadEstimator
from torch_psf.psfzip import save_zip

from bead_slide_modulation import estimate_modulation

# Reuse the TorchPSF simulation that reproduces the browser's bead-slide example.
torch_repo = Path(torch_psf.__file__).resolve().parents[1]
sys.path.insert(0, str(torch_repo / "study/simcode_random_walk"))
from simcode_bead_movie import OPTICS, simulate



def main():
    out = Path("bead-slide-example")
    out.mkdir(exist_ok=True)
    seed, n_frames, n_beads = 17, 40, 1200
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate uniformly distributed finite beads, two illumination axes and XYZ drift.
    simulate(out, seed=seed, frames=n_frames, device=device, bead_count=n_beads)
    movie = tifffile.imread(out / f"{seed}.tif")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    lo, hi = np.percentile(movie[0], [1, 99.7])
    for t, ax in enumerate(axes):
        ax.imshow(movie[t, :100, :100], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"Axis A, phase {120*t}°")
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out / "movie-frames.png", dpi=160)
    plt.close(fig)

    # Fit using only movie pixels, nominal optics and exposure labels.
    optics = dict(OPTICS, diameter_um=0.040, focus_um=0.1)
    exposure_slot = np.arange(len(movie)) % 6
    fit = DenseBeadEstimator.fit_movie(movie, exposure_slot, optics=optics, radius=6)
    pupil = fit.psf.optic.cfg["calibrated_psf"]
    result = estimate_modulation(movie, fit, optics)

    save_zip(str(out / "psf.zip"), pupil)
    (out / "modulation.json").write_text(json.dumps(result, indent=2))
    print(f"Recovered {len(fit.xy)} bead candidates from {len(movie)} frames.")

    # Score the blind estimates against simulation truth.
    truth = json.loads((out / f"{seed}-truth.json").read_text())
    print("Axis   pitch (nm)      angle (deg)      center depth")
    print("       true / fit      true / fit       true / fit")
    for i, pattern in enumerate(result["patterns"]):
        print(f"{'AB'[i]}      200 / {pattern['pitch_nm']:.1f}     "
              f"{[13, 104][i]:3} / {pattern['orientation_deg']:.1f}       "
              f"{[.84, .88][i]:.2f} / {pattern['depth'][0]:.3f}")
    print(f"Center A/B intensity ratio: true 1.12, fitted {result['ratio'][0]:.3f}")

    fig, (pax, dax) = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    modes = np.arange(5, 12)
    pax.bar(modes - .2, 1000 * np.array(truth["coef"]), .4, label="True")
    pax.bar(modes + .2, 1000 * pupil.zcoef.detach().cpu().numpy().ravel(), .4, label="Fitted")
    pax.set(xlabel="Noll index", ylabel="Zernike coefficient (mλ RMS)")
    pax.legend()

    u = np.linspace(-1, 1, 100)
    for i, pattern in enumerate(result["patterns"]):
        c = pattern["depth"]
        color = f"C{i}"
        x = (u + 1) / 2 * (movie.shape[2] - 1)
        true = .84 + .06*(2*x/movie.shape[2]-1) if i == 0 else np.full_like(u, .88-.05/movie.shape[1])
        dax.plot(x, true, "--", color=color, label=f"{'AB'[i]} true")
        dax.plot(x, c[0] + c[1]*u + c[3]*u**2, color=color, label=f"{'AB'[i]} fitted")
    dax.set(xlabel="Camera x (pixels), field midline", ylabel="Modulation depth")
    dax.legend(ncol=2)
    fig.savefig(out / "calibration.png", dpi=160)
    plt.close(fig)
    print(f"Saved figures and calibration to {out.resolve()}")


if __name__ == "__main__":
    main()
