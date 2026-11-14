"""The one place where SIMCODE uses torch_psf (github.com/jcnossen/torch-psf; pinned in pyproject.toml). Every other module
imports the PSF models, the calibration json reader, the fits and the TIFF reader from here, so a change of the torch_psf
API is a change of this file only.

From torch_psf (names as there): VectorialPSF, ConvSplinePSF, GaussianPSF, load_psf (calibration json -> model; angle =
per-pattern-angle PSF), Optics, Camera, TiffMovie, ArrayMovie, open_movie, solve_linear (closed-form I / bg of the
variable projection), Tracks, fit_tracks (in-situ PSF fit on tracks), estimate_psf_beads / fit_beads (bead z-stacks),
poisson_nll, photosel_tensor, gauss_blur_os, dipole_tensor, noll_to_nm, emitter_height_pos_fn, geometry_matrix.

SIMCODE-specific:

  has_angle_psf(c), angle_kwargs(c, dev)   per-pattern-angle PSF terms of a calibration dict (SIM illumination angle =
                                           step // 3 = frame % 6 // 3): 'psf_dipole_M_angle' [2][3][3], 'psf_blur_angle_px2'
                                           [2][3]. torch_psf.load_psf(path, angle=0 / 1 / None) gives that angle's model
                                           (None = the angle average: mean dipole tensor, mean blur covariance).
  render(psf, pos, **kw)                   normalized PSF at the model's focus for a VectorialPSF (kw = per-call dipM / blur)
                                           or a GaussianPSF (calib key 'psf_gauss_sigma_um'; z / focus / kw do not apply).
  spline(psf, zmax, roisize)               the PSF spline of the solve and the head template: knots 6 x oversampled, 10 nm
                                           in z over emitter heights 0 .. zmax plus 3 planes each side, offset limit
                                           P / 2 + 1 px, at the model's focus (or focus_um).
"""
import torch
import torch_psf as tp
from torch_psf import VectorialPSF, ConvSplinePSF, GaussianPSF, load_psf, Optics, Camera, estimate_psf_beads
from torch_psf.movie import TiffMovie, ArrayMovie, open_movie
from torch_psf.varpro import solve_linear
from torch_psf.spots import Tracks
from torch_psf.insitu import fit_tracks, fit_beads
from torch_psf.fit import poisson_nll
from torch_psf.vectorial import photosel_tensor, gauss_blur_os, dipole_tensor
from torch_psf.zernike import noll_to_nm
from torch_psf.spline import emitter_height_pos_fn, geometry_matrix

ANGLE_KEYS = ('psf_dipole_M_angle', 'psf_blur_angle_px2')


def has_angle_psf(c):
    """True if the calib dict c has a per-pattern-angle PSF (angle = step // 3 = frame % 6 // 3)"""
    return any(c.get(k) is not None for k in ANGLE_KEYS)


def angle_kwargs(c, dev):
    """per-pattern-angle VectorialPSF.forward kwargs [kw0, kw1] (dipM / blur of that angle) of a calib dict, or None"""
    if not has_angle_psf(c):
        return None
    out = []
    for k in range(2):
        kw = {}
        if c.get('psf_dipole_M_angle') is not None: kw['dipM'] = torch.tensor(c['psf_dipole_M_angle'][k], device=dev, dtype=torch.float32)
        if c.get('psf_blur_angle_px2') is not None: kw['blur'] = torch.tensor(c['psf_blur_angle_px2'][k], device=dev, dtype=torch.float32)
        out.append(kw)
    return out


def render(psf, pos, **kw):
    """normalized PSF [N, P, P] at pos [N, 3] (x, y px rel. to the ROI centre, z um) at psf.focus_um"""
    if isinstance(psf, tp.VectorialPSF):
        return psf(pos, psf.focus_um, **kw)
    return psf.render(pos)


class _GaussKnots(torch.nn.Module):
    """A GaussianPSF in the form ConvSplinePSF.knots_from_vectorial evaluates (a module with roisize, focus_um and
    forward(pos, focus)); torch_psf builds spline knots from a VectorialPSF only."""
    def __init__(self, g, roisize):
        super().__init__()
        self.g = tp.GaussianPSF(roisize, sigma_px=g.sigma, gamma_um=g.gamma, d_um=g.d,
                                z_range=g.z_range or (-0.5, 0.5), device=g.dev)
        self.roisize, self.focus_um = roisize, 0.0

    def forward(self, pos, focus):
        return self.g.render(pos)


def spline(psf, zmax=0.2, roisize=None, focus_um=None, oversample=6, dz_um=0.01, zpad=3, xy_limit=None, chunk=4):
    """ConvSplinePSF of psf (VectorialPSF or GaussianPSF) for roisize x roisize patches (default psf.roisize) and emitter
    heights 0 .. zmax um at the focus psf.focus_um (or focus_um)"""
    P = psf.roisize if roisize is None else roisize
    kw = dict(oversample=oversample, dz_um=dz_um, zpad=zpad, xy_limit=P / 2 + 1 if xy_limit is None else xy_limit)
    if isinstance(psf, tp.VectorialPSF):
        pos_fn = None if focus_um is None else emitter_height_pos_fn(focus_um)
        return tp.ConvSplinePSF.from_vectorial(psf, P, (0.0, zmax), pos_fn=pos_fn, chunk=chunk, **kw)
    s = tp.ConvSplinePSF(P, (0.0, zmax), device=psf.dev, **kw)
    s.knots = s.knots_from_vectorial(_GaussKnots(psf, s.Pl), chunk=chunk)
    return s
