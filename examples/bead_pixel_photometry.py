"""Refine per-exposure bead photons against the camera pixels for planar SIM.

All bead footprints share pixels. Only bead centres persist across frames.
Brightness and background are fit separately in every exposure, before using
any illumination model. Positions, pupil and focus are in camera coordinates.
"""
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from torch_psf.estimation.bead_movie import FiniteBeadRenderer


def _phase_bank(pupil, diameter_um, focus_um):
    # A wide vectorial ROI retains the PSF tails and its photon normalization.
    renderer = FiniteBeadRenderer(dict(calibrated_psf=pupil, diameter_um=diameter_um,
                                       pixel_um=pupil.pixel_um), 16)
    offset = np.linspace(-.5, .5, 7)
    y, x = np.meshgrid(offset, offset, indexing="ij")
    centres = renderer.tensor(np.c_[x.ravel(), y.ravel(), np.full(x.size, diameter_um/2)])
    nodes = (centres[:, None] + renderer.nodes).flatten(0, 1)
    bank = []
    for z in focus_um:
        images = torch.cat([renderer.model(chunk, float(z)) for chunk in nodes.split(48)])
        # Keep 27 x 27 pixels per bead after optical rendering.
        bank.append(images.reshape(7, 7, 6, 33, 33).mean(2)[:, :, 3:-3, 3:-3].contiguous().cpu())
    return bank


def _exposure(image, xy, bank, initial, axial=None):
    device = image.device
    h, w = image.shape
    side = bank.shape[-1]
    anchor = xy.round().long()
    cell = ((xy - anchor + .5)*6).clamp(0, 6)
    ix, iy = cell.long().clamp_max(5).T
    dx, dy = (cell - torch.stack([ix, iy], dim=1)).T[:, :, None, None]
    stamp = torch.lerp(torch.lerp(bank[iy, ix], bank[iy, ix+1], dx),
                       torch.lerp(bank[iy+1, ix], bank[iy+1, ix+1], dx), dy)
    axis = torch.arange(side, device=device) - side//2
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    x = x + anchor[:, 0, None, None]
    y = y + anchor[:, 1, None, None]
    valid = ((x >= 0) & (x < w) & (y >= 0) & (y < h)).flatten(1)
    pixel = (y*w+x).clamp(0, h*w-1).flatten(1)
    kernel = stamp.flatten(1)*valid
    observed = image.flatten()
    flux = initial.clamp_min(1)
    background = torch.quantile(observed[::max(1, len(observed)//16000)], .01).clamp_min(0)

    def project(values):
        return torch.zeros_like(observed).scatter_add_(0, pixel.flatten(), values.flatten())

    for iteration in range(25):
        mu = (background + project(flux[:, None]*kernel)).clamp_min(1)
        residual = observed - mu
        weight = (mu+4).reciprocal()
        if iteration >= 6:
            weight *= (4*(mu+4).sqrt()/residual.abs().clamp_min(1)).clamp_max(1)
        wr = (weight*residual)[pixel]
        if iteration == 24:
            break
        denom = (kernel.square()*weight[pixel]).sum(1).clamp_min(1e-8)
        step = (kernel*wr).sum(1)/denom
        bg_step = (weight*residual).sum()/weight.sum() if iteration % 3 == 0 else 0.
        # Limit the combined Jacobi step when many nearby PSFs move together.
        step = torch.maximum(step, -flux+1)
        direction = project(step[:, None]*kernel) + bg_step
        scale = ((weight*residual*direction).sum() /
                 (weight*direction.square()).sum().clamp_min(1e-8)).clamp(0, .7)
        flux = flux + scale*step
        background = (background + scale*bg_step).clamp_min(0)
    gy, gx = torch.gradient(stamp, dim=(1, 2))
    gx, gy = [-g.flatten(1)*valid for g in (gx, gy)]
    rhs = torch.stack([(g*wr).sum(1)*flux for g in (gx, gy)], 1)
    gram = torch.stack([(g*l*weight[pixel]).sum(1)*flux.square()
                        for g, l in ((gx, gx), (gx, gy), (gy, gy))], 1)
    step = None
    if axial is not None:
        dz = torch.lerp(torch.lerp(axial[iy, ix], axial[iy, ix+1], dx),
                        torch.lerp(axial[iy+1, ix], axial[iy+1, ix+1], dx), dy)
        jac = torch.stack([project(flux[:, None]*g) for g in (gx, gy, dz.flatten(1)*valid)])
        normal = (jac*weight)@jac.T + torch.eye(3, device=device)
        step = torch.linalg.solve(normal, jac@(weight*residual))
    return flux, rhs, gram, residual.reshape(h, w), step


@torch.no_grad()
def refine_photometry(movie, fit, pupil, focus_um, diameter_um, *, refine_motion=False):
    """Optionally return refined XY drift and absolute focus with the catalog and photons."""
    device = pupil.dev
    xy = torch.as_tensor(fit.xy, device=device, dtype=torch.float32).clone()
    flux = torch.as_tensor(fit.photons, device=device, dtype=torch.float32).clone()
    drift = torch.as_tensor(fit.drift, device=device, dtype=torch.float32)
    drift = drift.clone()
    focus = np.array(focus_um, dtype=float, copy=True)
    planes = {}
    def plane(z):
        if z not in planes:
            planes[z] = _phase_bank(pupil, diameter_um, [z])[0]
        return planes[z]
    passes = 12 if refine_motion else 6
    detect_pass = 5 if refine_motion else 1
    for iteration in range(passes):
        rhs = torch.zeros_like(xy)
        gram = xy.new_zeros(len(xy), 3)
        if iteration == detect_pass:
            mean_residual = np.zeros(tuple(movie.shape[1:]), dtype=np.float32)
        for t, frame in enumerate(movie):
            image = torch.as_tensor(frame, device=device, dtype=torch.float32)
            axial = None
            if refine_motion:
                z = focus[t]/.02
                lo = int(np.floor(z))
                low, high = plane(.02*lo).to(device), plane(.02*(lo+1)).to(device)
                kernel = torch.lerp(low, high, z-lo)
                if iteration < passes-1:
                    axial = high-low
            else:
                kernel = plane(focus[t]).to(device)
            flux[t], gradient, curvature, residual, step = _exposure(
                image, xy+drift[t], kernel, flux[t], axial)
            rhs += gradient
            gram += curvature
            if iteration == detect_pass:
                mean_residual += ndimage.shift(residual.cpu().numpy(),
                    -drift[t].flip(0).cpu().numpy(), order=1)/len(movie)
            if step is not None:
                drift[t] += .8*step[:2].clamp(-.4,.4)
                focus[t] += .016*float(step[2].clamp(-2,2))
        if iteration == passes-1:
            break
        a, b, c = gram.T
        det = (a*c-b*b).clamp_min(1e-8)
        delta = torch.stack([(c*rhs[:, 0]-b*rhs[:, 1])/det,
                             (a*rhs[:, 1]-b*rhs[:, 0])/det], 1)
        xy += delta.clamp(-.5, .5)
        if iteration == detect_pass:
            score = ndimage.gaussian_filter(mean_residual, .9) - ndimage.gaussian_filter(mean_residual, 3)
            threshold = max(3., 5*np.median(abs(score-np.median(score))))
            y, x = np.where((score == ndimage.maximum_filter(score, 3)) & (score > threshold))
            peaks = np.c_[x, y]
            peaks = peaks[cKDTree(xy.cpu().numpy()).query(peaks)[0] > 1.65]
            if len(peaks):
                xy = torch.cat([xy, xy.new_tensor(peaks)])
                initial = flux.median().clamp_min(1)*.25
                flux = torch.cat([flux, initial.expand(len(movie), len(peaks))], 1)
    result = xy.cpu().double(), flux.cpu().double()
    if refine_motion:
        return (*result, drift.cpu().double(), focus)
    return result


def exposure_phases(flux, nominal):
    """Fit common exposure phases from bead traces, with independent DC and quadratures."""
    if len(nominal) < 6:
        return nominal
    phases = nominal.clone()
    y = flux/flux.mean(0).clamp_min(1)
    for _ in range(16):
        design = torch.stack([torch.ones_like(phases), phases.cos(), -phases.sin()], 1)
        if torch.linalg.matrix_rank(design) < 3:
            return nominal
        coef = torch.linalg.lstsq(design, y).solution
        residual = y-design@coef
        deriv = -phases.sin()[:, None]*coef[1]-phases.cos()[:, None]*coef[2]
        weights = (.08/residual.abs().clamp_min(.0001)).clamp_max(1)
        weights *= (.14/residual.square().mean(0).clamp_min(1e-6).sqrt()).clamp_max(1)
        step = ((residual*deriv*weights).sum(1) /
                (deriv.square()*weights).sum(1).clamp_min(1)).clamp(-.3, .3)
        phases += step-step.mean()
    return phases
