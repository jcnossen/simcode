"""
Training data for the pattern-free detector: crlb_dense.make_scene (calibrated model: vectorial PSF LUT, SIMFLUX
pattern randomised per FOV, smooth structured background x per-angle factor, sCMOS read-noise map, ADU rounding) with
an explicit emitter list:
  - DNA-PAINT kinetics: Poisson binding events, exponential on-times, mean on-time log-uniform 2-60 frames
  - density log-uniform 0.0005-0.04 ON/frame/px per FOV; half of the FOVs on random filaments (as Simulator.sample)
  - oversampled close pairs: a fraction of the events gets a partner at 0.5-3 px (random direction, own kinetics,
    started so that it overlaps the window)
  - per-frame photons (mean illumination) log-uniform 40-6000: half of the FOVs per emitter over the full range,
    half with a per-FOV median (log-uniform) x lognormal(0.5) per emitter
  - bg log-uniform 1-30 ph/px/frame per FOV, modulation depth U(0.5, 1)
Targets (assign): GT emitters with >= 1 on-frame (sum of on-fractions) and >= min_phot window photons take their
nearest pixel; when two fall in the same pixel the brighter keeps it and the other takes the free 8-neighbour pixel
nearest to it (offset target up to ~1.2 px), so close pairs get two positives; still-colliding ones are ignored.
Emitters with 0 < on-frames < 1 or fewer photons are 'don't care' (their pixel is excluded from the negatives).
"""
import math
import torch
import simcode.sim.crlb as CD

dev = 'cuda'


def lu(lo, hi, *n):
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(*n, device=dev))


def emitters(nb, H, W, T, zmax, dens=(0.0005, 0.04), ontime=(2.0, 60.0), photons=(40.0, 6000.0),
             pair_frac=0.25, pair_sep=(0.5, 3.0), structured_frac=0.5, margin=6, carpet=0.0):
    g = lambda *s: torch.rand(*s, device=dev)
    d = lu(*dens, nb); tau = lu(*ontime, nb)
    area = (H + 2 * margin) * (W + 2 * margin); t_lo = -4 * tau
    n_ev = torch.poisson(d * area * (T - t_lo) / tau).long()
    fov = torch.repeat_interleave(torch.arange(nb, device=dev), n_ev); N = len(fov)
    t0 = t_lo[fov] + (T - t_lo[fov]) * g(N); t1 = t0 - torch.log(g(N).clamp(min=1e-9)) * tau[fov]
    keep = t1 > 0; fov, t0, t1 = fov[keep], t0[keep], t1[keep]; N = len(fov)
    x = -margin + (W + 2 * margin) * g(N); y = -margin + (H + 2 * margin) * g(N)
    structured = g(nb) < structured_frac
    if structured.any() and N:
        nfil = 6
        p0 = torch.stack([-margin + (W + 2 * margin) * g(nb, nfil), -margin + (H + 2 * margin) * g(nb, nfil)], -1)
        ang = g(nb, nfil) * math.pi
        fi = torch.randint(0, nfil, (N,), device=dev)
        s = (g(N) - 0.5) * 2 * (W + H); a_ = ang[fov, fi]
        xs = p0[fov, fi, 0] + s * torch.cos(a_) + torch.randn(N, device=dev) * 0.1
        ys = p0[fov, fi, 1] + s * torch.sin(a_) + torch.randn(N, device=dev) * 0.1
        inside = (xs > -margin) & (xs < W + margin) & (ys > -margin) & (ys < H + margin)
        use = structured[fov] & inside & (g(N) < 0.8)
        x = torch.where(use, xs, x); y = torch.where(use, ys, y)
    # close-pair partners of events active in the window
    act = (t1 > 0) & (t0 < T)
    pi = torch.nonzero(act & (g(N) < pair_frac))[:, 0]
    if len(pi):
        M = len(pi)
        sep = pair_sep[0] + (pair_sep[1] - pair_sep[0]) * g(M); th = g(M) * 2 * math.pi
        tp = tau[fov[pi]]
        pt0 = -tp + (T - 0.5 + tp) * g(M)                       # start in [-tau, T - 0.5]
        pt1 = pt0 + 0.5 - torch.log(g(M).clamp(min=1e-9)) * tp
        fov = torch.cat([fov, fov[pi]]); t0 = torch.cat([t0, pt0]); t1 = torch.cat([t1, pt1])
        x = torch.cat([x, x[pi] + sep * torch.cos(th)]); y = torch.cat([y, y[pi] + sep * torch.sin(th)])
        N = len(fov)
        ok = (x > -margin) & (x < W + margin) & (y > -margin) & (y < H + margin)   # stay inside the render canvas
        fov, t0, t1, x, y = fov[ok], t0[ok], t1[ok], x[ok], y[ok]; N = len(fov)
    z = zmax * g(N) ** 1.5
    per_em = g(nb) < 0.5
    b_em = lu(*photons, N)
    med = lu(*photons, nb)[fov] * torch.exp(0.5 * torch.randn(N, device=dev))
    bright = torch.where(per_em[fov], b_em, med.clamp(photons[0] * 0.5, photons[1] * 2))
    if carpet > 0:
        # dim flickering carpet (sub-threshold / out-of-focus binding, real-data-like background texture): density
        # U(0, carpet) ON/px/frame, per-frame photons log-uniform 5-60, on-time log-uniform 0.5-3 frames, z U(0, zmax)
        cd = torch.rand(nb, device=dev) * carpet; ct = lu(0.5, 3.0, nb)
        area = (H + 2 * margin) * (W + 2 * margin)
        n_c = torch.poisson(cd * area * (T + 4 * ct) / ct).long()
        fc = torch.repeat_interleave(torch.arange(nb, device=dev), n_c); M_ = len(fc)
        c0 = -4 * ct[fc] + (T + 4 * ct[fc]) * g(M_); c1 = c0 - torch.log(g(M_).clamp(min=1e-9)) * ct[fc]
        fov = torch.cat([fov, fc]); t0 = torch.cat([t0, c0]); t1 = torch.cat([t1, c1])
        x = torch.cat([x, -margin + (W + 2 * margin) * g(M_)]); y = torch.cat([y, -margin + (H + 2 * margin) * g(M_)])
        z = torch.cat([z, zmax * g(M_)]); bright = torch.cat([bright, lu(5.0, 60.0, M_)])
        is_c = torch.cat([torch.zeros(N, dtype=torch.bool, device=dev), torch.ones(M_, dtype=torch.bool, device=dev)])
        return dict(fov=fov, x=x, y=y, z=z, bright=bright, t0=t0, t1=t1, carpet=is_c)
    return dict(fov=fov, x=x, y=y, z=z, bright=bright, t0=t0, t1=t1)


def sample(sim, nb, **kw):
    em = emitters(nb, sim.H, sim.W, sim.T, sim.zmax, **kw)
    bg = lu(1.0, 30.0, nb)[:, None, None, None]
    depth = (0.5 + 0.5 * torch.rand(nb, 1, device=dev))
    o = CD.make_scene(sim, nb, em=em, bg=bg, depth=depth)
    if 'carpet' in em: o['carpet'] = em['carpet']
    return o


def assign(gt, B, H, W, min_on=1.0, min_phot=60.0, dontcare=None):
    """-> label [B,H,W] (1 pos, 0 neg), ignore [B,H,W] bool, tidx [B,H,W] (GT index, -1), off [B,2,H,W] targets."""
    tot = gt['phot'].sum(1); on = gt['onf'].sum(1)
    px, py = torch.round(gt['x']).long(), torch.round(gt['y']).long()
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    pos = inside & (on >= min_on) & (tot >= min_phot)
    if dontcare is not None:     # e.g. the dim carpet: never a positive, its pixel is excluded from the negatives
        pos = pos & ~dontcare
    dc = inside & (on > 0) & ~pos
    HW = H * W
    ign = torch.zeros(B * HW, dtype=torch.bool, device=dev)
    ign[(gt['fov'] * HW + py * W + px)[dc]] = True
    tid = torch.full((B * HW,), -1, dtype=torch.long, device=dev)
    idx = torch.nonzero(pos)[:, 0]
    key = tot[idx] + 1e-6 * torch.arange(len(idx), device=dev)          # unique keys: deterministic winner
    lin = gt['fov'][idx] * HW + py[idx] * W + px[idx]
    best = torch.full((B * HW,), -1.0, device=dev).scatter_reduce(0, lin, key, 'amax')
    win = key == best[lin]
    tid[lin[win]] = idx[win]
    lose = idx[~win]
    if len(lose):
        o = torch.tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)], device=dev)
        cy = py[lose][:, None] + o[None, :, 0]; cx = px[lose][:, None] + o[None, :, 1]
        okc = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
        cl = gt['fov'][lose][:, None] * HW + cy.clamp(0, H - 1) * W + cx.clamp(0, W - 1)
        okc &= tid[cl] < 0
        dd = (gt['x'][lose][:, None] - cx) ** 2 + (gt['y'][lose][:, None] - cy) ** 2
        dd = torch.where(okc, dd, torch.full_like(dd, 1e9))
        j = dd.argmin(1); ok = dd.min(1).values < 1e8
        lin2 = cl[torch.arange(len(lose), device=dev), j][ok]; l2 = lose[ok]
        k2 = tot[l2] + 1e-6 * torch.arange(len(l2), device=dev)
        best2 = torch.full((B * HW,), -1.0, device=dev).scatter_reduce(0, lin2, k2, 'amax')
        w2 = k2 == best2[lin2]
        tid[lin2[w2]] = l2[w2]
    tid = tid.reshape(B, H, W)
    label = (tid >= 0).float()
    ign = ign.reshape(B, H, W) & (tid < 0)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    tc = tid.clamp(min=0)
    off = torch.stack([gt['x'][tc] - xx, gt['y'][tc] - yy], 1)
    return label, ign, tid, off
