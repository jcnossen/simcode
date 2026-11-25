"""Planar two-axis, three-phase modulation estimation using PyTorch.

Inputs are camera frames in photoelectrons, bead photometry and XY drift.
The optional image-power depth estimator assumes uniformly random beads.
For structured samples use random_beads=False. A calibrated pupil and the
frame focus trace from DenseBeadEstimator.fit_movie enable raw-pixel
photometry refinement. Evaluate spatial maps where beads sample the field.
"""
import math

import torch
import torch.nn.functional as F


def basis(xy, width, height):
    u, v = (2 * xy / xy.new_tensor([width - 1, height - 1]) - 1).T
    return torch.stack([torch.ones_like(u), u, v, u*u, u*v, v*v], dim=1)


def surface(x, y):
    if len(y) < 6 or torch.linalg.matrix_rank(x) < 6:
        raise ValueError("Too few independent samples for a quadratic field")
    weights = torch.ones_like(y)
    for _ in range(4):
        c = torch.linalg.lstsq(x * weights[:, None], y * weights).solution
        error = (x @ c - y).abs()
        weights = ((2 * error.kthvalue(len(error)//2 + 1).values + 1e-4) / error.clamp_min(1e-10)).clamp_max(1).sqrt()
    return c


def carrier(xy, harmonic, width, height):
    z = torch.complex(harmonic[1], harmonic[2])
    z = z / z.abs().clamp_min(z.abs().kthvalue(int(.85 * len(z)) + 1).values.clamp_min(1))
    z = z - z.mean()
    n = 1 << (2 * max(width, height) - 1).bit_length()
    x, y = (2 * xy + .5).floor().long().T
    grid = z.new_zeros(n * n)
    grid.index_add_(0, (y % n) * n + x % n, z)
    freq = torch.fft.fftfreq(n, d=.5, dtype=xy.dtype)
    power = torch.fft.fft2(grid.view(n, n)).abs()
    power[torch.hypot(freq[:, None], freq) < 2 / max(width, height)] = 0
    iy, ix = divmod(int(power.argmax()), n)
    k = 2 * math.pi * freq[[ix, iy]]
    offsets = torch.cartesian_prod(xy.new_tensor([-1, 0, 1]), xy.new_tensor([-1, 0, 1]))
    for j in range(1, 6):
        trials = k + (4 * math.pi / n / 2**j) * offsets
        score = (z[:, None] * torch.exp(-1j * (xy @ trials.T))).sum(0).abs()
        k = trials[score.argmax()]
    return k


def image_depth(movie, axis, k, transfer, bead_count):
    """Shot-noise-corrected harmonic/DC power in tiles of random bead texture."""
    h, w = movie.shape[-2:]
    margin = 8
    nx, ny = [min(16, (a - 2 * margin) // 40) for a in (w, h)]
    f = k / (2 * math.pi)
    if min(nx, ny) < 3 or bead_count < 6 * nx * ny or (f - f.round()).norm() < .35:
        return None
    # Use complete phase triplets for this axis, including the final partial cycle.
    triplets = movie[torch.arange(len(movie)) // 3 % 2 == axis]
    triplets = triplets[:len(triplets) // 3 * 3]
    t = torch.arange(-margin, margin + 1, dtype=torch.float32)
    def gaussian(sigma):
        g = torch.exp(-.5 * (t / sigma)**2) * (t.abs() <= math.ceil(3*sigma))
        g = g / g.sum()
        return g[:, None] * g
    broad = gaussian(2.6)
    band = gaussian(1.15) - broad
    stride = max(2, math.ceil(max(h, w) / 300))
    # Valid convolution removes the edge margin; both phase and DC use the same pixels.
    filtered = F.conv2d(triplets[:, None], torch.stack([band, broad])[:, None], stride=stride)
    signal, level = filtered.unbind(1)
    signal = signal.reshape(-1, 3, *signal.shape[-2:])
    level = level.reshape_as(signal)
    phases = torch.arange(3) * (2 * math.pi / 3)
    weights = torch.stack([torch.ones(3)/3, 2*phases.cos()/3, -2*phases.sin()/3])
    power = (torch.einsum('ap,cpyx->cayx', weights, signal).square()
             - band.square().sum() * torch.einsum('ap,cpyx->cayx', weights.square(), level)).sum(0)
    iy = torch.arange(power.shape[1]) * stride * ny // (h - 2*margin)
    ix = torch.arange(power.shape[2]) * stride * nx // (w - 2*margin)
    tile = (iy[:, None] * nx + ix).flatten()
    sums = power.new_zeros(3, nx*ny).index_add_(1, tile, power.flatten(1))
    keep = sums[0] > 0
    if keep.sum() < 9:
        return None
    idx = torch.arange(nx*ny)[keep]
    xy = torch.stack([margin + (idx % nx + .5)*(w-2*margin)/nx,
                      margin + (idx // nx + .5)*(h-2*margin)/ny], dim=1).double()
    ac = sums[1:, keep].sum(0)
    return surface(basis(xy, w, h), (ac.sign() * (ac.abs()/sums[0, keep]).sqrt()).double() / transfer)


def estimate_modulation(movie, fit, optics, random_beads=False):
    """Fit a carrier, phase and depth surface per axis, plus the A/B intensity ratio."""
    movie = torch.as_tensor(movie, dtype=torch.float32)
    xy, flux, drift = [torch.as_tensor(a, dtype=torch.float64) for a in
                       (fit.xy, fit.photons, fit.drift)]
    _, h, w = movie.shape
    optic = getattr(getattr(fit, "psf", None), "optic", None)
    cfg = getattr(optic, "cfg", {})
    axial = getattr(fit, "axial_drift_um", None)
    refit = not random_beads and cfg.get("calibrated_psf") is not None and axial is not None
    if refit:
        from bead_pixel_photometry import exposure_phases, refine_photometry
        xy, flux = refine_photometry(movie, fit, cfg["calibrated_psf"],
                                     cfg["focus_um"]+axial, optics["diameter_um"])
    field = basis(xy, w, h)
    frames = torch.arange(len(movie))
    patterns, means = [], []
    def demodulate(k, sel, steps):
        angles = steps + drift[sel] @ k
        design = torch.stack([torch.ones_like(angles), angles.cos(), -angles.sin()], dim=1)
        return torch.linalg.lstsq(design, flux[sel]).solution

    waves = []
    for axis in range(2):
        sel = frames // 3 % 2 == axis
        steps = (frames[sel] % 3).double() * (2 * math.pi / 3)
        k = xy.new_zeros(2)
        for _ in range(2):
            k = carrier(xy, demodulate(k, sel, steps), w, h)
        waves.append(k)
    fields = None
    if (random_beads and len(movie) >= 30 and min(h,w) >= 224 and
            cfg.get("calibrated_psf") is not None and axial is not None and len(xy) >= 6*((w-16)//40)*((h-16)//40)):
        from dense_slide_depth import dense_slide_fields
        fields = dense_slide_fields(movie, drift, cfg["focus_um"]+axial, cfg["calibrated_psf"],
                                    torch.stack(waves), optics)
    for axis in range(2):
        sel = frames // 3 % 2 == axis
        steps = (frames[sel] % 3).double() * (2 * math.pi / 3)
        k = fields[axis][1] if fields is not None else waves[axis]
        if refit:
            angles = exposure_phases(flux[sel], steps + drift[sel] @ k)
            design = torch.stack([torch.ones_like(angles), angles.cos(), -angles.sin()], dim=1)
            harmonic = torch.linalg.lstsq(design, flux[sel]).solution
        elif fields is None:
            harmonic = demodulate(k, sel, steps)
        else:
            angles = fields[axis][2]
            design = torch.stack([torch.ones_like(angles), angles.cos(), -angles.sin()], dim=1)
            harmonic = torch.linalg.lstsq(design, flux[sel]).solution
        dc, re, im = harmonic
        amp = torch.hypot(re, im)
        valid = (dc > 1) & (amp > .01*dc)
        if refit:
            # Residual detection can add weak candidates supported by few photons.
            # Require a measured DC above its harmonic regression uncertainty.
            residual = flux[sel] - design @ harmonic
            sigma = residual.square().sum(0).div(max(1, len(angles)-3)).sqrt()
            dc_se = torch.linalg.pinv(design)[0].norm()*sigma
            valid &= dc > 5*dc_se
        q = k.norm() * optics["diameter_um"] / (2*optics["pixel_um"])
        transfer = 1. if q < .001 else 3*(q.sin()-q*q.cos())/q**3
        if transfer < .1:
            raise ValueError("Carrier is too strongly attenuated by the bead diameter")
        residual = torch.atan2(im[valid], re[valid]) - xy[valid] @ k
        center = torch.exp(1j*residual).mean().angle()
        phase = surface(field[valid], center + torch.exp(1j*(residual-center)).angle())
        if fields is not None:
            depth, method = fields[axis][0], "drift- and transfer-corrected random-bead image power"
        else:
            depth = image_depth(movie, axis, k, transfer, len(xy)) if random_beads else None
            method = "random-bead image power" if depth is not None else "bead photometry"
            if refit:
                method = "overlap-refined bead photometry"
        if depth is None:
            depth = surface(field[valid], amp[valid]/dc[valid]/transfer)
        pattern = dict(k_rad_px=k.tolist(), phase=phase.tolist(), depth=depth.tolist(),
                             pitch_nm=float(2000*math.pi*optics["pixel_um"]/k.norm()),
                             orientation_deg=math.degrees(math.atan2(k[1], k[0])) % 360,
                             depth_method=method)
        if refit:
            pattern.update(beads=int(valid.sum()), exposure_frames=frames[sel].tolist(),
                           exposure_phases_rad=angles.tolist())
        patterns.append(pattern)
        means.append(dc)
    valid = (means[0] > 1) & (means[1] > 1)
    ratio = surface(field[valid], means[0][valid]/means[1][valid])
    return dict(patterns=patterns, ratio=ratio.tolist(), basis="[1,u,v,u²,uv,v²]; u,v in [-1,1]")
