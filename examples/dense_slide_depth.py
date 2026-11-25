"""Dense random slide calibration from noise-corrected Fourier-order powers.

Registered patches of a spatially random bead field have the same expected
object power in the central and shifted bands. Unmix all exposures at their
actual phases, estimate changes of optical transfer, and equate those powers
locally to fit a quadratic depth field. All spatial frequencies are in camera
coordinates and all inputs come from the movie fit.
"""
import math

import torch
import torch.nn.functional as F


def pattern_candidates(movie):
    """Candidate frequencies in radians per camera pixel from phase-tagged image power."""
    n, h, w = movie.shape
    device = movie.device
    wy = torch.hann_window(h, periodic=False, device=device)
    wx = torch.hann_window(w, periodic=False, device=device)
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.fftfreq(w, device=device)[None, :]
    r2 = fx.square() + fy.square()
    filt = torch.exp(-2 * math.pi**2 * 0.6**2 * r2) - torch.exp(
        -2 * math.pi**2 * 3**2 * r2
    )
    acc = torch.zeros((2, 2 * h, 2 * w), dtype=torch.complex64, device=device)
    for t in range(n):
        texture = (
            torch.fft.ifft2(torch.fft.fft2(movie[t]) * filt).real * wy[:, None] * wx
        )
        spectrum = torch.fft.fftshift(torch.fft.fft2(texture))
        pad = F.pad(spectrum, (w // 2, w - w // 2, h // 2, h - h // 2))
        signal = torch.fft.ifft2(torch.fft.ifftshift(pad)).real
        acc[t // 3 % 2] += (
            signal.square()
            / signal.square().mean().clamp_min(1e-9)
            * complex(
                math.cos(2 * math.pi * (t % 3) / 3),
                -math.sin(2 * math.pi * (t % 3) / 3),
            )
        )
    nn = 2 ** math.ceil(math.log2(max(2 * h, 2 * w)))
    freq = torch.fft.fftfreq(nn, d=0.5, device=device)
    r = freq[:, None].hypot(freq[None, :])
    output = []
    for image in acc:
        mag = torch.fft.fft2(image, s=(nn, nn)).abs()
        floor = F.avg_pool2d(
            F.pad(mag[None, None], (7,) * 4, mode="circular"), 15, stride=1
        )[0, 0]
        score = mag / (floor + 0.001 * floor.max())
        score[(r < 0.05) | (r > 0.91)] = 0
        peaks = score == F.max_pool2d(score[None, None], 9, stride=1, padding=4)[0, 0]
        idx = (score * peaks).flatten().topk(4).indices
        ks = []
        log = mag.clamp_min(1e-9).log()
        for i in idx:
            iy, ix = i // nn, i % nn
            k = 2 * math.pi * torch.stack([freq[ix], freq[iy]])
            # Sub-bin parabolic interpolation of the log-magnitude.
            for axis, shift in enumerate([(0, 1), (1, 0)]):
                dy, dx = shift
                l = log[(iy - dy) % nn, (ix - dx) % nn]
                c = log[iy, ix]
                rr = log[(iy + dy) % nn, (ix + dx) % nn]
                delta = 0.5 * (l - rr) / (l - 2 * c + rr).clamp_max(-1e-8)
                k[axis] += delta.clamp(-0.5, 0.5) * 4 * math.pi / nn
            ks.append(k)
        output.append(torch.stack(ks))
    return output


def temporal_phases(movie, drift, frames):
    """Recover the phase of each exposure from the ellipse traced by low-frequency image modes."""
    _, h, w = movie.shape
    dev = movie.device
    fy = torch.fft.fftfreq(h, device=dev)[:, None]
    fx = torch.fft.rfftfreq(w, device=dev)[None, :]
    band = (fx.hypot(fy) > 0.014) & (fx.hypot(fy) < 0.047)

    # Taper only the image boundary, retaining independent spatial modes across the field.
    def taper(n):
        x = torch.arange(n, device=dev)
        edge = torch.minimum(x, (n - 1) - x) / max(6, n // 20)
        return 0.5 - 0.5 * torch.cos(math.pi * edge.clamp(0, 1))

    win = taper(h)[:, None] * taper(w)
    data = movie[frames]
    mean = (data * win).sum((-1, -2), keepdim=True) / win.sum()
    y = torch.fft.rfft2((data - mean) * win)[:, band]
    wave_x = fx.expand(h, -1)[band]
    wave_y = fy.expand(-1, fx.shape[1])[band]
    y *= torch.exp(
        2j
        * math.pi
        * (drift[frames, 0, None] * wave_x + drift[frames, 1, None] * wave_y)
    )
    # Modestly balance object channels; conditioning on a single large DC component
    # would align the phase estimate to that particular bead realization.
    scale = y.abs().square().mean(0)
    y = y / (scale + scale.median()).sqrt()
    centered = y - y.mean(0)
    sample = torch.cat([centered.real, centered.imag], 1)
    u, _, _ = torch.linalg.svd(sample, full_matrices=False)
    x, v = (u[:, :2] * math.sqrt(len(u))).T
    D = torch.stack([x * x, 2 * x * v, v * v, x, v], 1)
    a, b, c, d, e = torch.linalg.lstsq(D, torch.ones_like(x)).solution
    Q = torch.stack([a, b, b, c]).reshape(2, 2)
    vals, vecs = torch.linalg.eigh(Q)
    if vals.min() > 0:
        center = -0.5 * torch.linalg.solve(Q, torch.stack([d, e]))
        uv = (torch.stack([x, v], 1) - center) @ (vecs * vals.sqrt())
    else:
        uv = torch.stack([x, v], 1)
    return torch.atan2(uv[:, 1], uv[:, 0])


def carrier_coherence(
    patches, spectra, origins, qx, qy, proposals, phase_options, cutoff
):
    """A real carrier makes the harmonics in adjacent spectral orders share bead texture."""
    n = patches.shape[-1]
    grid = torch.arange(n, device=patches.device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    scores = []
    for k, phases in zip(proposals, phase_options):
        kx, ky = k / (2 * math.pi)
        a, b = qx.hypot(qy), (qx + kx).hypot(qy + ky)
        band = (a > 0.08) & (b > 0.08) & (a < 0.9 * cutoff) & (b < 0.9 * cutoff)
        if int(band.sum()) < 20 or k.norm() < 0.3:
            scores.append(0.0)
            continue
        shifted = torch.fft.fft2(patches * torch.exp(-1j * (xx * k[0] + yy * k[1])))
        design = torch.stack(
            [torch.ones_like(phases), torch.exp(1j * phases), torch.exp(-1j * phases)],
            1,
        )
        inverse = torch.linalg.pinv(design)
        left = torch.einsum("ct,tpf->cpf", inverse, spectra[:, :, band])
        right = torch.einsum("ct,tpf->cpf", inverse, shifted[:, :, band])

        phase_origin = torch.exp(1j * (origins @ k))

        def correlation(first, second):
            numerator = (first * second.conj()).mean(-1) * phase_origin
            denominator = (
                (first.abs().square().mean(-1) * second.abs().square().mean(-1))
                .clamp_min(1e-10)
                .sqrt()
            )
            return (numerator / denominator).mean().abs()

        scores.append(
            float((correlation(left[0], right[1]) + correlation(left[2], right[0])) / 2)
        )
    return scores


@torch.enable_grad()
def calibrate_transfer(images, angles, qx, qy):
    # Subsample windows to limit the weight of overlapping pixels.
    images = images[:, ::3]
    dev = images.device
    radius = qx.hypot(qy)
    usable = (radius > 0.035) & (radius < 0.295) & (qx >= 0)
    # Every Fourier channel has its own (unknown) object spectrum. A smooth relative
    # optical transfer across frames and a global phase per exposure are shared by every patch.
    x = qx / 0.18
    y = qy / 0.18
    basis = torch.stack(
        [x * x, y * y, x * y, x**4, y**4, x * x * y * y, x**3 * y, x * y**3]
    )
    odd = torch.stack([x, y, x**3, x * x * y, x * y * y, y**3])
    sel = torch.where(usable.flatten())[0]
    B = basis.flatten(1)[:, sel]
    P = odd.flatten(1)[:, sel]
    Y = images.flatten(2)[:, :, sel].permute(2, 0, 1).contiguous()
    n = Y.shape[1]
    scale = Y.abs().square().mean((1, 2), keepdim=True).clamp_min(4).sqrt()
    Y = Y / scale
    # Use measured local spectral power for the initial relative optical attenuation.
    power = Y.abs().square().mean(2).clamp_min(0.001).log() * 0.5
    init = torch.linalg.lstsq(B.T, power - power.mean(1, keepdim=True)).solution.T
    even = init.detach().clone().requires_grad_()
    tilt = torch.zeros((n, len(P)), device=dev, requires_grad=True)

    def transfer(even, tilt, B, P):
        return torch.exp(
            ((even - even.mean(0)) @ B).clamp(-3, 3) + 1j * ((tilt - tilt.mean(0)) @ P)
        )

    def design(theta):
        return torch.stack(
            [torch.ones_like(theta), torch.exp(1j * theta), torch.exp(-1j * theta)], 1
        )

    opt = torch.optim.Adam([even, tilt], lr=0.022)
    for i in range(95):
        if i == 60:
            for g in opt.param_groups:
                g["lr"] *= 0.35
        opt.zero_grad()
        H = transfer(even, tilt, B, P)
        A = H.T[:, :, None] * design(angles)[None]
        with torch.no_grad():
            gram = A.mH @ A + 1e-5 * torch.eye(3, device=dev)
            obj = torch.linalg.solve(gram, A.mH @ Y)
        residual = A @ obj - Y
        loss = residual.abs().square().mean() + 0.00015 * (
            (even - init).square().mean() + tilt.square().mean()
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = transfer(even, tilt, basis.flatten(1), odd.flatten(1)).reshape(n, *qx.shape)
        return h


def _field_patches(aligned, side):
    """Sample the whole field, including partial patches at the camera boundary."""
    _, height, width = aligned.shape
    device = aligned.device
    offset = side // 4

    def starts(length):
        intervals = min(20, math.ceil((length - side / 2) / 28))
        return (
            torch.linspace(-offset, length - 3 * offset, intervals + 1)
            .round()
            .int()
            .tolist()
        )

    coords = [(x, y) for y in starts(height) for x in starts(width)]
    origins = aligned.new_tensor(coords)
    padded = F.pad(aligned, (offset,) * 4)
    patches = torch.stack(
        [
            padded[:, y + offset : y + offset + side, x + offset : x + offset + side]
            for x, y in coords
        ],
        1,
    )
    line = torch.arange(side, device=device)
    hann = torch.hann_window(side, periodic=False, device=device)

    def aperture(origin, length):
        pixel = origin[:, None] + line
        distance = torch.minimum(pixel, length - 1 - pixel)
        # Omit the first three camera pixels and taper over the next eight.
        # No reflected specimen intensity contributes to a partial window.
        boundary = 0.5 - 0.5 * torch.cos(math.pi * ((distance - 3) / 8).clamp(0, 1))
        window = hann * boundary
        position = 2 * pixel / (length - 1) - 1
        weight = window.square()
        weight = weight / weight.sum(1, keepdim=True)
        return window, (position * weight).sum(1), (position.square() * weight).sum(1)

    wx, u, uu = aperture(origins[:, 0], width)
    wy, v, vv = aperture(origins[:, 1], height)
    window = wy[:, :, None] * wx[:, None, :]
    local_mean = (patches * window).sum((-2, -1), keepdim=True) / window.sum(
        (-2, -1), keepdim=True
    )
    # Power integrates against window squared. A partial patch has a shifted centroid
    # and different second moments, which enter the quadratic design directly.
    spatial = torch.stack([torch.ones_like(u), u, v, uu, u * v, vv], 1)
    return (patches - local_mean) * window, spatial, origins


def _texture_weights(radial, qx, qy, band, central_order, transfer_power):
    """Balance the optical power envelope while downweighting noisy weak-transfer modes."""
    side = len(qx)
    azimuth = torch.remainder(torch.atan2(qy[band], qx[band]) / math.pi, 1)
    sector = 6 * (radial[band] * side).round().long() + (6 * azimuth).long().clamp_max(
        5
    )
    dc_power = central_order.abs().square().mean(0)
    sums = dc_power.new_zeros(6 * side).scatter_add_(0, sector, dc_power)
    counts = torch.bincount(sector, minlength=len(sums))
    # The envelope is shared across patches. Using a local denominator here
    # would correlate the noise in each patch with its regression weight.
    envelope = (sums[sector] / counts[sector]).clamp_min(1)
    return (transfer_power / (transfer_power + 0.015)).square() / envelope


def _consistent_band_weights(weights, radii, signal, spatial):
    """Reject frequency bands whose modulation disagrees across independent spatial blocks."""
    # A single depth applies across Fourier radii. In strongly defocused movies,
    # a relative transfer model can leave appreciable DC leakage at high radius.
    # Use paired block jackknifes so real spatial structure contributes to both estimates.
    block_xy = ((spatial[:, 1:3] + 1) * 2).long().clamp(0, 3)
    blocks = block_xy[:, 0] + 4 * block_xy[:, 1]
    cuts = (0.11, 0.155, 0.20, 0.245, float("inf"))
    masks = [(radii > low) & (radii <= high) for low, high in zip((0., *cuts[:-1]), cuts)]
    def block_ratios(mask):
        powers = (signal[:, :, mask] * weights[mask]).sum(-1)
        totals = powers.new_zeros((3, 16)).index_add_(1, blocks, powers)
        sums = totals.sum(1, keepdim=True)
        sample = torch.cat([sums, sums - totals], 1)
        return 2 * (sample[1] + sample[2]) / sample[0].clamp_min(1e-12)
    if not masks[0].any():
        return weights
    reference = block_ratios(masks[0])
    accepted = masks[0].clone()
    for mask in masks[1:]:
        if not mask.any():
            continue
        difference = block_ratios(mask) - reference
        leave_one_out = difference[1:]
        se = (15 / 16 * (leave_one_out - leave_one_out.mean()).square().sum()).sqrt()
        if difference[0].abs() > 3 * se:
            break
        accepted |= mask
    return weights * accepted


@torch.no_grad()
def dense_slide_fields(movie, drift, focus, pupil, initial_k, optics):
    """Return depth coefficients, carrier (rad/pixel) and exposure phases for each axis.

    Expects at least five A0/A120/A240/B0/B120/B240 cycles in photoelectrons.
    The depth surface uses [1, u, v, u², uv, v²] over normalized camera coordinates.
    """
    device = pupil.dev
    images = torch.as_tensor(movie, dtype=torch.float32, device=device)
    drift = torch.as_tensor(drift, dtype=torch.float32, device=device)
    initial_k = torch.as_tensor(initial_k, dtype=torch.float32, device=device)
    nt, height, width = images.shape
    side = 64
    if nt < 30 or min(height, width) < 224:
        raise ValueError(
            "Dense-slide field fitting needs five complete cycles and a field at least 224 pixels wide and high."
        )

    margin = 12
    padded = F.pad(images[:, None], (margin,) * 4, mode="reflect")[:, 0]
    fy = torch.fft.fftfreq(padded.shape[1], device=device)[:, None]
    fx = torch.fft.rfftfreq(padded.shape[2], device=device)[None, :]
    phase_ramp = (
        2j * math.pi * (drift[:, 0, None, None] * fx + drift[:, 1, None, None] * fy)
    )
    aligned = torch.fft.irfft2(
        torch.fft.rfft2(padded) * phase_ramp.exp(), s=padded.shape[-2:]
    )
    aligned = aligned[:, margin : margin + height, margin : margin + width]

    # The fitted 33-pixel footprint selects frequencies that carry usable signal.
    # Within that band, the relative complex transfer is estimated from the data.
    model = pupil.resized(33)
    radius = optics["diameter_um"] / 2
    nodes = torch.cat([torch.eye(3, device=device), -torch.eye(3, device=device)]) * (
        radius * math.sqrt(0.6)
    )
    nodes[:, :2] /= optics["pixel_um"]
    nodes[:, 2] += radius
    stamps = torch.stack([model(nodes, float(z)).mean(0) for z in focus])
    otf = torch.fft.fft2(
        F.pad(stamps, (0, side - 33, 0, side - 33)).roll((-16, -16), (-2, -1))
    )

    filtered, spatial, origins = _field_patches(aligned, side)
    spectra = torch.fft.fft2(filtered)
    freq = torch.fft.fftfreq(side, device=device)
    qy, qx = torch.meshgrid(freq, freq, indexing="ij")
    radial = qx.hypot(qy)
    times = torch.arange(nt, device=device)
    candidates = pattern_candidates(images)
    result = []
    for axis in range(2):
        t = times[times // 3 % 2 == axis]
        proposals = torch.cat(
            [initial_k[axis, None], candidates[axis], -initial_k[axis, None]]
        )
        nominal = (t % 3) * (2 * math.pi / 3) + (drift[t] @ proposals.T).T
        measured = temporal_phases(images, drift, t)
        choices = torch.stack([measured, -measured])
        correlation = torch.exp(1j * (nominal[:, None, :] - choices)).mean(-1)
        consistency, direction = correlation.abs().max(1)
        offsets = correlation[
            torch.arange(len(proposals), device=device), direction
        ].angle()
        phase_options = choices[direction] + offsets[:, None]
        coherence = carrier_coherence(
            filtered[t, ::2],
            spectra[t, ::2],
            origins[::2],
            qx,
            qy,
            proposals,
            phase_options,
            2 * optics["NA"] * optics["pixel_um"] / optics["wavelength_um"],
        )
        score = (torch.as_tensor(coherence, device=device) + 0.01) * (
            0.1 + consistency
        ).square()
        best = int(score.argmax())
        k, angles = proposals[best], phase_options[best]
        H = calibrate_transfer(spectra[t], angles, qx, qy)
        h2 = otf[t].abs().square().mean(0)
        band = (radial > 0.065) & (radial < 0.29) & (h2 > 0.0015)
        if int(band.sum()) < 20:
            raise ValueError(
                "Too little optical bandwidth to estimate modulation depth."
            )
        phases = torch.stack(
            [torch.ones_like(angles), torch.exp(1j * angles), torch.exp(-1j * angles)],
            1,
        )
        A = H[:, band].T[:, :, None] * phases
        inverse = torch.linalg.solve(
            A.mH @ A + 1e-7 * torch.eye(3, device=device), A.mH
        )
        orders = torch.einsum("fct,tpf->cpf", inverse, spectra[t][:, :, band])
        fitted = torch.einsum("ftc,cpf->tpf", A, orders)
        residual = spectra[t][:, :, band] - fitted
        # Estimate demodulated power variance from frame residuals (HC2 leverage correction).
        leverage = torch.einsum("ftc,fct->tf", A, inverse).real
        variance = residual.abs().square() / (1 - leverage[:, None, :]).clamp_min(0.05)
        noise = torch.einsum("fct,tpf->cpf", inverse.abs().square(), variance)
        weights = _texture_weights(radial, qx, qy, band, orders[0], h2[band])
        signal = orders.abs().square() - noise
        weights = _consistent_band_weights(weights, radial[band], signal, spatial)
        powers = (signal * weights).sum(-1)
        q = k.norm() * radius / optics["pixel_um"]
        transfer = 1.0 if q < 0.001 else 3 * (q.sin() - q * q.cos()) / q**3
        if transfer < 0.1:
            raise ValueError(
                "The bead diameter suppresses the measured carrier too strongly."
            )
        ac = 2 * (powers[1] + powers[2]) / transfer**2
        dc = powers[0]
        if dc.mean() <= 0 or ac.sum() <= 0:
            raise ValueError("Insufficient modulation power above photon noise.")
        coeff = torch.zeros(6, device=device)
        coeff[0] = (ac.sum() / dc.sum()).sqrt()
        # Fit the signed moments before division or square root. Taking magnitudes
        # per tile would give positive depth even for noise-only modulation.
        for _ in range(4):
            depth = (spatial @ coeff).clamp_min(0.02)
            lhs = spatial.T @ ((2 * depth * dc / dc.mean())[:, None] * spatial)
            rhs = spatial.T @ ((ac - depth.square() * dc) / dc.mean())
            coeff += torch.linalg.solve(lhs, rhs)
        result.append((coeff.cpu().double(), k.cpu().double(), angles.cpu().double()))
    return result
