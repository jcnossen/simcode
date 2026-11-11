"""
Scene generator and multi-emitter Cramer-Rao bounds for the benchmark scenes (used by the gate suite simcode2/hd/gate.py,
the nuisance CRLB simcode2/eff/crlb_nuis.py, the detector training data simcode2/det/data.py).

Multi-emitter CRLB (exact vectorial PSF, true pattern, per-pixel read noise, Poisson + Gaussian variance) on the fixed
15 x 15 region around the target pixel; expected image = bg + all emitters; free parameters = every active emitter whose
pixel is within 9 px (Chebyshev) of the target pixel, plus one bg per angle:
  SF   : (x, y, z, B, t_on, t_off) per emitter with the true pattern illumination (kinetics as nuisance)
  SM   : same, illumination constant (pattern information removed)       -> bound for PSF-domain estimators
  free : (x, y, z, I_1..I_T) per emitter, free per-frame intensities       -> bound for per-frame photon estimates
  each also 'isolated' (same emitter, no neighbours in the image or the parameter set).
The PSF / camera model comes from the in-situ calibration json `a.calib` (default: models/insitu/calib_0.5nM.json; set
`crlb.a.calib = <path>` before models()).
"""
import math, types
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from simcode2.sim.simulator import Simulator
from simcode2.psf.lut import PSFLUT
from simcode2.nn.blocks import extract_patches
from simcode2 import config as _cfg

a = types.SimpleNamespace(calib=_cfg.model(_cfg.INSITU_05))
import os as _os
_CDIV = int(_os.environ.get('GATE_CRLB_DIV', '3'))   # (2026-11-11) per-angle PSF CRLB batch divisor (memory only; same results)
dev = 'cuda'; S = 64; P = 11; h = P // 2; R = 15; HR = R // 2
BOXP, BOXM = 9, 14          # parameter set / expected-image set (Chebyshev px from the target pixel)


# ----------------------------------------------------------------------------------------------------------- scenes
def make_scene(sim, nb, em=None, dens=0.01, tau=50.0, photons=(150, 500), bg=5.0, depth=0.93):
    """Simulator.sample with the true pattern (no phase/depth error), uniform positions, constant bg level, and an
    optional explicit emitter list em = dict(fov, x, y, z, bright, t0, t1). Same pattern/bg/camera model."""
    H, W, T = sim.H, sim.W, sim.T
    g = lambda *s: torch.rand(*s, device=dev)
    margin = 6
    lu = lambda lo, hi, n: torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * g(n))
    if em is None:
        area = (H + 2 * margin) * (W + 2 * margin); t_lo = -4 * tau
        n_ev = torch.poisson(torch.full((nb,), dens * area * (T - t_lo) / tau, device=dev)).long()
        fov = torch.repeat_interleave(torch.arange(nb, device=dev), n_ev); N = len(fov)
        t0 = t_lo + (T - t_lo) * g(N); t1 = t0 - torch.log(g(N).clamp(min=1e-9)) * tau
        keep = t1 > 0; fov, t0, t1 = fov[keep], t0[keep], t1[keep]; N = len(fov)
        x = -margin + (W + 2 * margin) * g(N); y = -margin + (H + 2 * margin) * g(N)
        z = sim.zmax * g(N) ** 1.5
        Bmed = lu(*photons, nb)
        bright = Bmed[fov] * torch.exp(0.35 * torch.randn(N, device=dev)) * torch.exp(-z / sim.tirf)
    else:
        fov, x, y, z, bright, t0, t1 = (em[k] for k in ('fov', 'x', 'y', 'z', 'bright', 't0', 't1')); N = len(fov)
    fr = torch.arange(T, device=dev, dtype=torch.float32)
    onf = (torch.minimum(t1[:, None], fr[None] + 1) - torch.maximum(t0[:, None], fr[None])).clamp(0, 1)
    step0 = torch.randint(0, 6, (nb,), device=dev)
    steps = (step0[:, None] + torch.arange(T, device=dev)[None]) % 6
    rot = (g(nb, 2) - 0.5) * math.radians(10); scale = 1 + (g(nb, 2) - 0.5) * 0.06
    k = torch.stack([sim.k0[:, 0] * torch.cos(rot) - sim.k0[:, 1] * torch.sin(rot),
                     sim.k0[:, 0] * torch.sin(rot) + sim.k0[:, 1] * torch.cos(rot)], -1) * scale[..., None]
    phi = ((g(nb, 2) * 2 * math.pi)[:, :, None] + torch.tensor([0, 2, 4], device=dev) * math.pi / 3
           + (g(nb, 2, 3) - 0.5) * 0.4).reshape(nb, 6)
    m = (depth + (g(nb, 6) - 0.5) * 0.1).clamp(0, 1)
    rel = 1 + (g(nb, 6) - 0.5) * 0.2
    angle = steps // 3
    kk = k[torch.arange(nb, device=dev)[:, None], angle]
    ph = phi.gather(1, steps); mm = m.gather(1, steps); rr = rel.gather(1, steps)
    kre = (kk[fov] * torch.stack([x, y], -1)[:, None]).sum(-1)
    phot = bright[:, None] * rr[fov] * (1 + mm[fov] * torch.cos(kre - ph[fov])) * onf
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    illum = rr[..., None, None] * (1 + mm[..., None, None] * torch.cos(kk[..., 0, None, None] * xx + kk[..., 1, None, None] * yy - ph[..., None, None]))
    patt = torch.stack([kk[..., 0], kk[..., 1], ph, mm, rr], -1)
    Rr = sim.R; hr = Rr // 2
    img = torch.zeros(nb, T, H + 2 * (margin + hr) + 2, W + 2 * (margin + hr) + 2, device=dev)
    if N > 0:
        psf = sim.render_psf(x, y, z)
        cx = torch.round(x).long() + margin + 1; cy = torch.round(y).long() + margin + 1
        oy, ox = torch.meshgrid(torch.arange(Rr, device=dev), torch.arange(Rr, device=dev), indexing='ij')
        iy = (cy[:, None, None] + oy).reshape(N, -1); ix = (cx[:, None, None] + ox).reshape(N, -1)
        Wp, Hp = img.shape[-1], img.shape[-2]
        lin = fov[:, None, None] * T * Hp * Wp + torch.arange(T, device=dev)[None, :, None] * Hp * Wp + (iy * Wp + ix)[:, None, :]
        img.view(-1).index_add_(0, lin.reshape(-1), (phot[:, :, None] * psf.reshape(N, 1, -1)).reshape(-1))
    off = margin + hr + 1
    img = img[:, :, off:off + H, off:off + W]
    low = torch.nn.functional.interpolate(g(nb, 1, 4, 4), size=(H, W), mode='bicubic', align_corners=False)
    bgmap = bg * (0.7 + 0.6 * low).clamp(min=0.2)
    bgf = (1 + (g(nb, 2) - 0.5) * 0.3).gather(1, angle)[..., None, None]
    mu = img + bgmap * bgf
    Hm, Wm = sim.rn_map.shape
    oy_ = torch.randint(0, Hm - H + 1, (nb,)).tolist(); ox_ = torch.randint(0, Wm - W + 1, (nb,)).tolist()
    rn = torch.stack([sim.rn_map[y0:y0 + H, x0:x0 + W] for y0, x0 in zip(oy_, ox_)])
    flip = torch.rand(nb, device=dev) < 0.5
    rn = torch.where(flip[:, None, None], rn.flip(-1), rn)[:, None]
    e = torch.poisson(mu) + torch.randn_like(mu) * rn
    frames = (torch.round(e / sim.gain + sim.offset).clamp(0, 65535) - sim.offset) * sim.gain
    gt = dict(fov=fov, x=x, y=y, z=z, bright=bright, t0=t0, t1=t1, onf=onf, phot=phot)
    return dict(frames=frames, illum=illum, patt=patt, rn=rn, bg=bgmap * bgf, steps=steps, gt=gt, nb=nb)


def pair_emitters(nb, sep, ratio, kin, T, zmax, tau=6.0):
    """9 pairs per FOV on an 18 px grid; returns em dict and the pair role (0: A, 1: B) of every emitter."""
    g = lambda *s: torch.rand(*s, device=dev)
    c = torch.tensor([14.0, 32.0, 50.0], device=dev)
    cy, cx = torch.meshgrid(c, c, indexing='ij')
    n = nb * 9
    ctr = torch.stack([cx.reshape(-1), cy.reshape(-1)], 1).repeat(nb, 1) + (g(n, 2) - 0.5)
    th = g(n) * 2 * math.pi; dv = torch.stack([torch.cos(th), torch.sin(th)], 1) * sep / 2
    pos = torch.cat([ctr - dv, ctr + dv])                                  # A then B
    fov = torch.arange(nb, device=dev).repeat_interleave(9).repeat(2)
    BA, BB = (150.0, 150.0) if ratio == '1:1' else (100.0, 300.0)
    bright = torch.cat([torch.full((n,), BA, device=dev), torch.full((n,), BB, device=dev)])
    z = zmax * g(2 * n) ** 1.5
    if kin == 'always-on':
        t0 = torch.full((2 * n,), -1e4, device=dev); t1 = torch.full((2 * n,), 1e4, device=dev)
    else:
        fr = torch.arange(T, device=dev, dtype=torch.float32)
        t0 = torch.zeros(2 * n, device=dev); t1 = torch.zeros(2 * n, device=dev); todo = torch.ones(2 * n, dtype=torch.bool, device=dev)
        for _ in range(200):
            if not todo.any(): break
            m_ = int(todo.sum()); s0 = -6 + (T + 6) * g(m_); s1 = s0 - torch.log(g(m_).clamp(min=1e-9)) * tau
            onfr = (torch.minimum(s1[:, None], fr + 1) - torch.maximum(s0[:, None], fr)).clamp(0, 1).sum(1)
            ok = onfr >= 3.0
            idx = torch.nonzero(todo)[:, 0][ok]
            t0[idx] = s0[ok]; t1[idx] = s1[ok]; todo[idx] = False
    role = torch.cat([torch.zeros(n, device=dev), torch.ones(n, device=dev)])
    return dict(fov=fov, x=pos[:, 0], y=pos[:, 1], z=z, bright=bright, t0=t0, t1=t1), role


# ------------------------------------------------------------------------------------------------- matching
def match(G, d, nb):
    """One-to-one GT <-> detection assignment (squared distance, gate 1 px) per FOV over all active GT emitters."""
    gx, gy, gf = (G[k].cpu().numpy() for k in ('x', 'y', 'fov'))
    dx, dy, db = d['x'].cpu().numpy(), d['y'].cpu().numpy(), d['b'].cpu().numpy()
    out = np.full(len(gx), -1)
    for b in range(nb):
        gi = np.nonzero(gf == b)[0]; di = np.nonzero(db == b)[0]
        if len(gi) == 0 or len(di) == 0: continue
        D = (gx[gi, None] - dx[None, di]) ** 2 + (gy[gi, None] - dy[None, di]) ** 2
        near = (D < 1).any(1); gi2 = gi[near]
        if len(gi2) == 0: continue
        C = np.where(D[near] < 1, D[near], 1e6)
        r, c = linear_sum_assignment(C)
        ok = C[r, c] < 1
        out[gi2[r[ok]]] = di[c[ok]]
    return torch.tensor(out, device=dev)


# ---------------------------------------------------------------------------------------------------------- helpers
def illum_at(pt, X, Y):
    """pt [n,T,5]; X, Y [n] absolute FOV px -> il, dil/dX, dil/dY [n,T]."""
    arg = pt[..., 0] * X[:, None] + pt[..., 1] * Y[:, None] - pt[..., 2]
    s = -pt[..., 4] * pt[..., 3] * torch.sin(arg)
    return pt[..., 4] * (1 + pt[..., 3] * torch.cos(arg)), s * pt[..., 0], s * pt[..., 1]


def onfrac(ton, toff, T):
    fr = torch.arange(T, device=ton.device, dtype=torch.float32)[None]
    a_ = (torch.minimum(toff[:, None], fr + 1) - torch.maximum(ton[:, None], fr)).clamp(0, 1)
    pos = a_ > 0
    d_on = -((ton[:, None] >= fr) & (ton[:, None] < fr + 1) & pos).float()
    d_off = ((toff[:, None] > fr) & (toff[:, None] <= fr + 1) & pos).float()
    return a_, d_on, d_off


def shift_region(img, ox, oy):
    """img [n,K,C,R,R] in each emitter's own pixel-centred ROI; (ox, oy) [n,K] emitter pixel - target pixel.
    Returns the images in target-ROI pixels (zero where not covered)."""
    n, K, C, Rr, _ = img.shape
    o = torch.arange(Rr, device=img.device)
    ys = o[None, None] - oy[..., None]; xs = o[None, None] - ox[..., None]
    vy = (ys >= 0) & (ys < Rr); vx = (xs >= 0) & (xs < Rr)
    img = img.gather(3, ys.clamp(0, Rr - 1)[:, :, None, :, None].expand(n, K, C, Rr, Rr))
    img = img.gather(4, xs.clamp(0, Rr - 1)[:, :, None, None, :].expand(n, K, C, Rr, Rr))
    return img * (vy[:, :, None, :, None] & vx[:, :, None, None, :]).float()


def psf_channels(rpsf):
    """4 (p, dx, dy, dz), or 8 = angle-0 then angle-1 stacks when rpsf carries a per-pattern-angle PSF (rpsf.angle_kw)"""
    return 8 if getattr(rpsf, 'angle_kw', None) else 4


@torch.no_grad()   # (2026-11-11) no autograd graph (memory; same values)
def psf_stack(rpsf, x, y, z, focus, chunk=1024):
    """Exact vectorial PSF on each emitter's own 15x15 ROI + central-difference derivatives: [N,4,R,R] (p, dx, dy, dz).
    (2026-11-11) [N,8,R,R] (angle 0 | angle 1) if rpsf has a per-angle PSF (models() of a per-angle calib)."""
    pos = torch.stack([x - torch.round(x), y - torch.round(y), z], 1)
    kws = getattr(rpsf, 'angle_kw', None) or [{}]
    if len(kws) > 1: chunk = min(chunk, int(_os.environ.get('GATE_PSF_INNER', chunk)))   # (2026-11-11) opt-in memory cap, per-angle PSF
    out = []
    for p in pos.split(chunk):
        f = torch.full((len(p),), focus, device=dev); im = []
        for kw in kws:
            im.append(rpsf(p, f, **kw))
            for i, hh in ((0, 1e-3), (1, 1e-3), (2, 1e-3)):
                dd = torch.zeros_like(p); dd[:, i] = hh
                im.append((rpsf(p + dd, f, **kw) - rpsf(p - dd, f, **kw)) / (2 * hh))
        out.append(torch.stack(im, 1))
    return torch.cat(out) if out else torch.zeros(0, 4 * len(kws), R, R, device=dev)


def scaled_solve(FI, cols):
    """Jacobi-scaled, ridge-regularized (1e-6, as crlb_plots) inverse restricted to columns `cols`: returns
    (FI^-1)[cols][:, cols] [n,c,c] (float64 Cholesky; LU fallback)."""
    FI = FI.double()
    dg = FI.diagonal(dim1=1, dim2=2).clamp(min=1e-12); s = dg.rsqrt()
    M = FI * s[:, :, None] * s[:, None, :] + 1e-6 * torch.eye(FI.shape[-1], device=dev, dtype=torch.float64)
    E = torch.zeros(FI.shape[0], FI.shape[1], len(cols), device=dev, dtype=torch.float64)
    E[:, cols, torch.arange(len(cols))] = 1
    L, info = torch.linalg.cholesky_ex(M)
    X = torch.cholesky_solve(E, L)
    bad = info != 0
    if bad.any():
        X[bad] = torch.linalg.solve(M[bad], E[bad])
    X = X[:, cols]
    sc = s[:, cols]
    X = (X * sc[:, :, None] * sc[:, None, :]).float()
    bad = ~(X.diagonal(dim1=1, dim2=2) > 0).all(1)          # numerical breakdown (practically unidentifiable) -> NaN
    X[bad] = float('nan')
    return X


# -------------------------------------------------------------------------------------------------- multi-emitter CRLB
def crlb_all(G, o, psS, tg, T, focus):
    """Multi-emitter and isolated CRLBs of targets tg (indices into the active-GT arrays G).
    Returns dict: sf_m, sf_i, sm_m, sm_i [n,2] (x, y px); fr_m, fr_i [n,T] (per-frame I, free model); cov_or [n,2+T,2+T]
    (target (x, y, I_1..I_T) block of the multi-emitter free-model inverse Fisher)."""
    nb = o['nb']; Np = len(G['x'])
    pxi, pyi = torch.round(G['x']).long(), torch.round(G['y']).long()
    patt = o['patt']; ang_f = (patt[..., 0].abs() < patt[..., 1].abs()).long()
    # per-emitter time factors (true parameters)
    pt_e = patt[G['fov']]
    if psS.shape[1] == 8:   # (2026-11-11) per-angle PSF: [Np, 4, T, R, R], each frame's PSF follows its pattern angle
        ae = ang_f[G['fov']][:, None, :, None, None]
        psS = torch.where(ae == 0, psS[:, None, :4].transpose(1, 2), psS[:, None, 4:].transpose(1, 2))
        PT = T
    else:
        PT = 1
    il, dilx, dily = illum_at(pt_e, G['x'], G['y'])
    a_, da0, da1 = onfrac(G['t0'], G['t1'], T)
    Bv = G['bright']
    Ie = Bv[:, None] * a_ * il                                         # true per-frame photons
    # expected image of the whole FOV with the exact PSF: bg + all emitters
    pad = BOXM
    Sp = S + 2 * pad
    canvas = torch.zeros(nb, T, Sp, Sp, device=dev)
    oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
    lin = (G['fov'][:, None, None] * T * Sp * Sp + torch.arange(T, device=dev)[None, :, None] * Sp * Sp
           + ((pyi[:, None, None] + pad - HR + oy) * Sp + pxi[:, None, None] + pad - HR + ox).reshape(Np, 1, -1))
    canvas.view(-1).index_add_(0, lin.reshape(-1), (Ie[:, :, None] * psS[:, 0].reshape(Np, PT, -1)).reshape(-1))
    mu_full = canvas[:, :, pad:pad + S, pad:pad + S] + o['bg']
    rn2_full = (o['rn'] ** 2).expand(nb, 1, S, S)
    n = len(tg)
    tb, tcx, tcy = G['fov'][tg], pxi[tg], pyi[tg]
    mu_m = extract_patches(mu_full, tb, tcy, tcx, R).reshape(n, T, -1)
    bg_r = extract_patches(o['bg'], tb, tcy, tcx, R).reshape(n, T, -1)
    rn2_r = extract_patches(rn2_full, tb, tcy, tcx, R).reshape(n, 1, -1)
    mu_i = bg_r + Ie[tg][:, :, None] * psS[tg, 0].reshape(n, PT, -1)
    # parameter sets: target first, then active emitters within BOXP px
    sets = []
    for i in range(n):
        j = tg[i]
        cand = torch.nonzero((G['fov'] == G['fov'][j]) & ((pxi - pxi[j]).abs() <= BOXP) & ((pyi - pyi[j]).abs() <= BOXP))[:, 0]
        cand = cand[cand != j]
        sets.append(torch.cat([j[None], cand]))
    Ks = torch.tensor([len(s_) for s_ in sets])
    res = {k: torch.full((n, 2), float('nan'), device=dev) for k in ('sf_m', 'sf_i', 'sm_m', 'sm_i')}
    res['fr_m'] = torch.full((n, T), float('nan'), device=dev); res['fr_i'] = res['fr_m'].clone()
    res['cov_or'] = torch.full((n, 2 + T, 2 + T), float('nan'), device=dev)
    angT = ang_f[tb]                                                   # [n,T]
    bgcol = torch.stack([(angT == 0).float(), (angT == 1).float()], 1)[..., None].expand(n, 2, T, R * R)   # [n,2,T,RR]

    def run(idx, K, iso):
        m_ = len(idx)
        if iso:
            S_ = torch.stack([sets[i][:1] for i in idx.tolist()])
        else:
            S_ = torch.stack([torch.cat([sets[i], torch.full((K - len(sets[i]),), -1, dtype=torch.long, device=dev)]) for i in idx.tolist()])
        valid = (S_ >= 0).float(); Sc = S_.clamp(min=0); Kk = S_.shape[1]
        if PT == 1:
            ps = shift_region(psS[Sc], pxi[Sc] - tcx[idx][:, None], pyi[Sc] - tcy[idx][:, None]) * valid[:, :, None, None, None]
        else:
            ps = shift_region(psS[Sc].reshape(m_, Kk, 4 * T, R, R), pxi[Sc] - tcx[idx][:, None], pyi[Sc] - tcy[idx][:, None]) \
                * valid[:, :, None, None, None]
        ps = ps.reshape(m_, Kk, 4, PT, R * R)
        B_ = Bv[Sc][..., None]; A_ = a_[Sc]; IL = il[Sc]
        w = 1.0 / ((mu_i[idx] if iso else mu_m[idx]) + rn2_r[idx])         # [m,T,RR]
        # SMLM model: Fisher weights must come from the flat-illumination expected image (critic review 22 #1:
        # using the modulated mu with the flat Jacobian inflated the SMLM information). Replace the modulated
        # contributions of the fitted emitters by flat ones (emitters outside the set stay as they are; minor).
        ilflat = pt_e[Sc][..., 4]
        c_mod = ((B_ * A_ * IL)[..., None] * ps[:, :, 0]).sum(1)          # [m,T,RR]
        c_flat = ((B_ * A_ * ilflat)[..., None] * ps[:, :, 0]).sum(1)
        w_sm = 1.0 / (((mu_i[idx] if iso else mu_m[idx]) - c_mod + c_flat).clamp(min=1e-3) + rn2_r[idx])
        out = {}
        for model in ('sf', 'sm', 'fr'):
            wm = w_sm if model == 'sm' else w
            if model == 'fr':
                I_ = (B_ * A_ * IL)[..., None]                             # [m,K,T,1]
                cols = [I_ * ps[:, :, 1], I_ * ps[:, :, 2], I_ * ps[:, :, 3]]
                eye = torch.eye(T, device=dev)[None, None, :, :, None]     # [1,1,T(s),T(t),1]
                JI = eye * ps[:, :, 0][:, :, None]                         # [m,K,T,T,RR]
                J = torch.cat([torch.stack(cols, 2), JI], 2)               # [m,K,3+T,T,RR]
                q = 3 + T
            else:
                ilm = IL if model == 'sf' else pt_e[Sc][..., 4]
                dx_ = dilx[Sc] if model == 'sf' else torch.zeros_like(IL)
                dy_ = dily[Sc] if model == 'sf' else torch.zeros_like(IL)
                # SM: constant illumination with the same photons (rel only), as crlb_plots
                BA = B_ * A_
                c = [(BA * dx_)[..., None] * ps[:, :, 0] + (BA * ilm)[..., None] * ps[:, :, 1],
                     (BA * dy_)[..., None] * ps[:, :, 0] + (BA * ilm)[..., None] * ps[:, :, 2],
                     (BA * ilm)[..., None] * ps[:, :, 3],
                     (A_ * ilm)[..., None] * ps[:, :, 0],
                     (B_ * da0[Sc] * ilm)[..., None] * ps[:, :, 0],
                     (B_ * da1[Sc] * ilm)[..., None] * ps[:, :, 0]]
                J = torch.stack(c, 2); q = 6
            J = J * valid[:, :, None, None, None]
            J = torch.cat([J.reshape(m_, Kk * q, T, R * R), bgcol[idx]], 1).reshape(m_, Kk * q + 2, -1)
            J = J.double()
            FI = torch.einsum('npk,nqk->npq', J * wm.reshape(m_, 1, -1).double(), J)
            del J
            if model == 'fr':
                cols = [0, 1] + list(range(3, 3 + T))
                C = scaled_solve(FI, cols)
                out['fr'] = C
            else:
                out[model] = scaled_solve(FI, [0, 1])
            del FI
        return out

    for iso in (False, True):
        order = torch.argsort(Ks) if not iso else torch.arange(n)
        i0 = 0
        while i0 < n:
            K = 1 if iso else int(Ks[order[i0]])
            npar = K * (3 + T) + 2
            nbat = max(1, min(512, int(3e7 // (npar * T * R * R * (_CDIV if PT > 1 else 1)))))   # (2026-11-11) per-angle PSF: a third of the batch (memory; same results)
            idx = order[i0:i0 + nbat]
            if not iso:
                K = int(Ks[idx].max()); npar = K * (3 + T) + 2
                nbat = max(1, min(len(idx), int(3e7 // (npar * T * R * R * (_CDIV if PT > 1 else 1))))); idx = idx[:nbat]
            idx = idx.to(dev)
            out = run(idx, K, iso)
            sfx = '_i' if iso else '_m'
            res['sf' + sfx][idx] = out['sf'].diagonal(dim1=1, dim2=2).clamp(min=0).sqrt()
            res['sm' + sfx][idx] = out['sm'].diagonal(dim1=1, dim2=2).clamp(min=0).sqrt()
            res['fr' + sfx][idx] = out['fr'].diagonal(dim1=1, dim2=2)[:, 2:].clamp(min=0).sqrt()
            if not iso: res['cov_or'][idx] = out['fr']
            i0 += len(idx)
    res['K'] = Ks.to(dev)
    return res


def two_stage_bounds(Cb, G, tg, o, T):
    """Analytic x/y bounds of the PER-EMITTER explicit-pattern scheme with ideal inputs: v = (x, y, I_1..I_T) of the
    target ~ N(truth, Cb) (Cb = multi-emitter free-model CRLB block), fitted with the target's SIMFLUX model
    (x, y, B, t_on, t_off). full: (G^T Cb^-1 G)^-1 (best per-emitter combination); diag: sandwich covariance of the
    diagonal-weight fit (what iv_joint does). Their ratio to CRLB_SF_multi is the information lost by combining per
    emitter (the pattern constraint on the neighbours' intensities is not used)."""
    pt = o['patt'][G['fov'][tg]]
    il, dx, dy = illum_at(pt, G['x'][tg], G['y'][tg])
    a_, d0, d1 = onfrac(G['t0'][tg], G['t1'][tg], T)
    B = G['bright'][tg][:, None]
    n = len(tg)
    Gm = torch.zeros(n, 2 + T, 5, device=dev)
    Gm[:, 0, 0] = 1; Gm[:, 1, 1] = 1
    Gm[:, 2:, 0] = B * a_ * dx; Gm[:, 2:, 1] = B * a_ * dy; Gm[:, 2:, 2] = a_ * il; Gm[:, 2:, 3] = B * il * d0; Gm[:, 2:, 4] = B * il * d1
    Cd = Cb.double(); ok = torch.isfinite(Cd).all(2).all(1)
    Cd = torch.where(ok[:, None, None], Cd, torch.eye(2 + T, device=dev, dtype=torch.float64))
    Cd = 0.5 * (Cd + Cd.transpose(1, 2)) + 1e-9 * Cd.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(2 + T, device=dev, dtype=torch.float64)
    Gd = Gm.double()
    M = (Gd.transpose(1, 2) @ torch.linalg.solve(Cd, Gd)).float()
    full = scaled_solve(M, [0, 1]).diagonal(dim1=1, dim2=2).clamp(min=0).sqrt()
    W = torch.diag_embed(1.0 / Cd.diagonal(dim1=1, dim2=2))
    A = (Gd.transpose(1, 2) @ W @ Gd).float()
    Ai = scaled_solve(A, [0, 1, 2, 3, 4]).double()
    Bm = Gd.transpose(1, 2) @ W @ Cd @ W @ Gd
    diag = (Ai @ Bm @ Ai).float().diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt()
    nanm = torch.full_like(full, float('nan'))
    return torch.where(ok[:, None], full, nanm), torch.where(ok[:, None], diag, nanm)


# ------------------------------------------------------------------------------------------------ models
def models(T):
    """calibrated simulator (64 x 64), exact 15 x 15 vectorial PSF for the CRLB, 11 x 11 PSF LUT"""
    sim = Simulator(a.calib, H=S, W=S, T=T, device=dev)
    p0 = sim.psf
    mk = lambda Pp: type(p0)(Pp, pixelsize=p0.pixelsize, NA=float(p0.NA), n_imm=p0.n_imm, n_med=float(p0.n_med),
                             wavelength=p0.wavelength, oversample=3, device=dev)
    rpsf = mk(R); rp11 = mk(P)
    with torch.no_grad():
        for q in (rpsf, rp11): q.zcoef.copy_(p0.zcoef); q.apod.copy_(p0.apod)
    from simcode2.psf.vectorial import copy_psf_extras, angle_kwargs
    for q in (rpsf, rp11): copy_psf_extras(q, p0)       # opt-in calib terms (angle average); absent = unchanged
    rpsf.angle_kw = angle_kwargs(sim.c, dev)             # per-angle PSF for psf_stack / crlb_all (None = off)
    return sim, rpsf, PSFLUT(rp11, sim.focus, sim.zmax)
