"""
PSF-perturbed simulation (critic review 24 #1): build_lut renders a Simulator PSF lookup table from the calibrated
vectorial PSF with a perturbation (used by simulate/acquisition.py):
  width   pixel-size scale (PSF width in pixels x w)
  zern    Zernike coefficient deltas on top of the calibration
  focus   focus shift (um)
"""
import json
import torch
from simcode import psf as spsf
from simcode.psf import render

dev = 'cuda'


def build_lut(sim, width=1.0, zern_delta=None, focus_shift=0.0, zmax=None, angle=None):
    """returns (lut [nz, ls+1, ls+1, R, R], zmax) for a perturbed PSF, rendered like Simulator.__init__
    (2026-11-11) with the calib's opt-in PSF terms: torch_psf.PSF.load of the calib dict with the pixel size / width, the focus
    and the Zernikes changed (angle = that pattern angle's PSF, None = average).
    A Gaussian calibration ('psf_gauss_sigma_um') gets sigma x width; the Zernike / focus terms do not apply to it."""
    zmax = zmax or sim.zmax
    c = json.load(open(sim.calib_path))
    if c.get('psf_gauss_sigma_um') is not None:
        base = spsf.PSF.load(c, roisize=sim.R, device=dev)
        psf = spsf.GaussianPSF(sim.R, sigma_px=base.sigma[0] * width, device=dev)
    else:
        c['pixelsize_um'] = c['pixelsize_um'] / width
        c['focus_um'] = sim.focus + focus_shift
        if zern_delta is not None:
            z = torch.tensor(c['zernike_waves'], device=dev, dtype=torch.float32) + zern_delta
            c['zernike_waves'] = z.tolist()
        psf = spsf.PSF.load(c, roisize=sim.R, oversample=3, device=dev, angle=angle)
    zs = torch.linspace(getattr(sim, 'zmin', 0.0), zmax, sim.lz, device=dev)
    sub = (torch.arange(sim.ls + 1, device=dev) / sim.ls) - 0.5
    Z, SY, SX = torch.meshgrid(zs, sub, sub, indexing='ij')
    pos = torch.stack([SX.reshape(-1), SY.reshape(-1), Z.reshape(-1)], 1)
    with torch.no_grad():
        lut = torch.cat([render(psf, p) for p in pos.split(1024)])
    return lut.reshape(sim.lz, sim.ls + 1, sim.ls + 1, sim.R, sim.R), zmax


