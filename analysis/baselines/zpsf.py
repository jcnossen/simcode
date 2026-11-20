"""One PSF input for the classical / DECODE baselines of the paper rerun (SMLM, SIMFLUX, DECODE all use the SAME PSF).

spec (one string, the `--psf` argument of paper_pipeline.py and decode_paper.py):
  gauss:<sigma_px>        2D Gaussian (pixel-integrated, erf), e.g. gauss:1.3 = the published paper's PSF
  <calib>.json / .psf.zip in-situ calibration: its PSF as SIMCODE renders it, simcode.psf.calib_psf = the integrated cubic
                          B-spline of the calibration's vectorial model (torch_psf SplinePSF; the coefficients from
                          the psf.zip when it carries them, else built on load). The SAME function and pixel values as the
                          SIMCODE training simulator / templates (2026-11-19, one PSF renderer). A per-pattern-angle
                          calibration ('psf_dipole_M_angle' / 'psf_blur_angle_px2'): eval(..., angle=0 / 1) is that angle's
                          PSF (frame f: angle f % 6 // 3), angle=None the mean of the two angle images.

API: psf = load(spec); g, gx, gy = psf.eval(xc, yc, P[, z][, angle=]) with xc, yc [N] = emitter position relative to the ROI
centre (px; pixel j of a P-px ROI has its centre at j - (P-1)/2), returns [N,P,P] tensors (g sums to ~1 over the full plane;
gx = dg/dxc). psf.z_fit = the fit plane (um). psf.desc = a short description (stored in the outputs). psf.n_angles = 2 for a
per-angle calibration, else 0.
knot_spline(...): a Catmull-Rom pixel spline (torch_psf SplinePSF) on a stored knot stack (the 4Pi analysis, analysis/fourpi/refit.py)."""
import os, sys, json, math
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.psf import SplinePSF


class GaussPSF:
    def __init__(self, sigma, device='cuda'):
        self.sigma = float(sigma); self.dev = torch.device(device); self.z_fit = 0.0
        self.desc = f'gauss:{self.sigma}'

    n_angles = 0

    def eval(self, xc, yc, P, z=None, grad=True, angle=None):
        s = self.sigma * math.sqrt(2)
        o = torch.arange(P, device=xc.device, dtype=torch.float32) - (P - 1) / 2
        def ax(c):
            a = (o[None] + 0.5 - c[:, None]) / s; b = (o[None] - 0.5 - c[:, None]) / s
            e = 0.5 * (torch.erf(a) - torch.erf(b))
            de = -(torch.exp(-a ** 2) - torch.exp(-b ** 2)) / (math.sqrt(math.pi) * s)      # d e / d c
            return e, de
        ex, dex = ax(xc); ey, dey = ax(yc)
        g = ey[:, :, None] * ex[:, None, :]
        if not grad:
            return g
        return g, ey[:, :, None] * dex[:, None, :], dey[:, :, None] * ex[:, None, :]


def knot_spline(zstack, S, u0, z0, hz, P, xmax=None):
    """torch_psf SplinePSF (Catmull-Rom of pixel values, grouped-convolution evaluation) on a stored knot stack [nz, n, n] (knot origin u0 px, S knots per px, plane 0 at z0 um,
    spacing hz um) for P x P ROIs (also even); offset limit P / 2 + 1 px (or xmax), reduced to what the knots support"""
    return SplinePSF(P, (float(z0), float(z0) + (zstack.shape[0] - 1) * float(hz)), oversample=int(S), dz_um=float(hz),
                     zpad=0, xy_limit=P / 2 + 1 if xmax is None else xmax, device=zstack.device,
                     knots=zstack.contiguous(), knot_u0_px=float(u0), knot_z0_um=float(z0), kernel='catmull', value='pixel',
                     backend='conv')


class CalibPSF:
    """a calibration's PSF: simcode.psf.calib_psf models per (ROI size, angle) (grid: a taller / other knot grid built with
    simcode.psf.integrated_spline instead, e.g. for figures; not the fitted PSF of the baselines)"""
    def __init__(self, path, z_fit=None, device='cuda', grid=None):
        from simcode import psf as spsf
        from simcode.config import resolve
        self.path = resolve(path); self.dev = torch.device(device); c = spsf.read_calib(self.path)
        self.n_angles = 2 if spsf.has_angle_psf(c) else 0
        self.z_fit = float(c.get('emitter_z_median_um', 0.055) if z_fit is None else z_fit)
        self.focus_um = float(c['focus_um'])
        self._base = spsf.integrated_spline(self.path, device, **grid) if grid else None
        g = self._base if grid else spsf.calib_psf(self.path, device=device)
        self.z_limits = g.z_limits
        self.desc = (f'calib:{os.path.basename(self.path)} integrated spline (simcode.psf.calib_psf, S {g.S}, dz {g.hz} um, '
                     f'z {g.z_range[0]}-{g.z_range[1]} um) angles={self.n_angles or "None"}')
        self._m = {}

    def model(self, P, angle=None):
        k = (int(P), angle)
        if k not in self._m:
            from simcode import psf as spsf
            if self._base is not None:
                m = self._base.for_angle(angle) if angle is not None and self._base.n_angles else self._base
                self._m[k] = m if m.roisize == P else m.resized(P)
            else:
                self._m[k] = spsf.calib_psf(self.path, P, angle=angle, device=self.dev)
        return self._m[k]

    def eval(self, xc, yc, P, z=None, grad=True, angle=None):
        zz = torch.full_like(xc, self.z_fit) if z is None else z
        pos = torch.stack([xc, yc, zz], 1).float()
        m = self.model(P, angle)
        if not grad:
            return m.render(pos)
        g, gx, gy, gz = m.render(pos, grad=True)
        return g, gx, gy


def load(spec, device='cuda', z_fit=None):
    if spec.startswith('gauss:'):
        return GaussPSF(float(spec.split(':', 1)[1]), device)
    if spec.endswith('.json') or spec.endswith('.psf.zip'):
        return CalibPSF(spec, z_fit, device)
    raise ValueError(f'unknown PSF spec {spec} (gauss:<sigma px>, a calib .json or .psf.zip)')
