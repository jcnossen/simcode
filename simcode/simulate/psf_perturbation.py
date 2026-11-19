"""
PSF-perturbed simulation (critic review 24 #1): build_psf = the calibrated PSF with a perturbation, rendered by the same model as
the calibration (simcode.psf.calib_psf: the integrated cubic B-spline, built from the perturbed vectorial model; used by
simulate/acquisition.py):
  width   pixel-size scale (PSF width in pixels x w)
  zern    Zernike coefficient deltas on top of the calibration
  focus   focus shift (um)
  blur    an extra anisotropic Gaussian image blur (sx, sy px), added to the calibration's blur covariance
"""
import torch
from simcode import psf as spsf

dev = 'cuda'


def build_psf(sim, width=1.0, zern_delta=None, focus_shift=0.0, blur_px=None):
    """the Simulator's PSF (sim.calib_path, ROI sim.R) perturbed: the calibration dict with the pixel size / width, the focus, the
    Zernikes and the blur changed -> simcode.psf.calib_psf of that dict: one model, or (angle 0, angle 1) for a per-pattern-angle
    calibration (as sim.psf / sim.psf_angle). A Gaussian calibration ('psf_gauss_sigma_um') gets sigma x width; the Zernike /
    focus / blur terms do not apply to it."""
    c = spsf.read_calib(sim.calib_path)
    if c.get('psf_gauss_sigma_um') is not None:
        base = spsf.PSF.load(c, roisize=sim.R, device=dev)
        return spsf.GaussianPSF(sim.R, sigma_px=base.sigma[0] * width, device=dev)
    c['pixelsize_um'] = c['pixelsize_um'] / width
    c['focus_um'] = sim.focus + focus_shift
    if zern_delta is not None:
        z = torch.tensor(c['zernike_waves'], device=dev, dtype=torch.float32) + zern_delta
        c['zernike_waves'] = z.tolist()
    if blur_px is not None:
        add = [float(blur_px[0]) ** 2, float(blur_px[1]) ** 2, 0.0]
        if c.get('psf_blur_angle_px2') is not None:
            c['psf_blur_angle_px2'] = [[b + a for a, b in zip(add, B)] for B in c['psf_blur_angle_px2']]
        else:
            c['psf_blur_px2'] = [b + a for a, b in zip(add, c.get('psf_blur_px2') or [0.0, 0.0, 0.0])]
    spl = spsf.calib_psf(c, sim.R, device=dev, z_range=(sim.zmin, sim.zmax))
    return tuple(spl.for_angle(k) for k in range(2)) if spsf.has_angle_psf(c) else spl
