"""Experiment-matched full-scene 4Pi-SIMFLUX simulator for network training (analogue of simcode/hd/realsim.py).

A batch = B scenes of H x W px (ch0 grid), 6 frames (2 angles x 3 steps), 4 interference channels. Each channel is
rendered in its OWN crop coordinates through the real in-situ registration (out/4pi/smlm/reg/reg_insitu_<cell>.json),
for a scene origin drawn inside the illuminated field (ch0 x 60-300, y 0-200), so the front end's code path (registered
sum by bilinear warp for the detector; unwarped channel patches at mapped positions for the head) is the real one.

Emitters (per scene): density log-uniform `dens` ON emitters / um^2 / frame; a mix of uniform random emitters,
filament-like lines (label radius 20 nm, tilted in z), NPC-like rings (8 corners, two rings 50 nm apart in z) and close
pairs (0.3-3 px, dz up to +-0.3 um). z in `zr`. DNA-PAINT kinetics inside the window (on-through, start / end in the
window), per-event photon rate log-normal around `F_med` e/frame (sd log 0.6, clipped to `F_rng`).
Pattern: k from the measured value +- 0.3 %, step phases 120 deg +- 3 deg, the phase at every position arbitrary (a
per-scene offset uniform over 2 pi: the v3 lesson), depth 0.6-0.95.
4Pi phase: smooth random field over the scene (+- 60 deg) + per-scene offset uniform over 2 pi, per-angle offset delta_a in
[-16, -8] deg.
PSF: a bank of LUTs (the fitted model + perturbations: blur, Noll 6, inter-pair phase, tau, coherence), one per scene.
Background per channel: measured levels x log-uniform(0.5, 2), smooth structure +-30 %, slow per-frame fluctuation.
Camera: Poisson + Gaussian read noise (1.14 e) + per-pixel offset error (0.2 e), output in e (offset-subtracted, x gain).
Returns dict:
  img [B,6,4,H,W] channel crops (e), origin [B,4,2] (crop origin of each channel, channel px), det_in [B,6,H,W]
  (registered 4-channel sum per frame), and per emitter-window GT lists: b, x, y (ch0 px in the scene), z, Nt [6],
  q [6], Phi (4Pi phase at the emitter), T [6,4], C [6,4] (read-out targets as in sim4pi).
"""
import json, math
import numpy as np
import torch
import torch.nn.functional as F
from simcode.fourpi.lut4pi import LUT4Pi

from simcode.config import REPO
from simcode.fourpi.setup import P4
BG = {'MT': (0.42, 1.06, 0.54, 1.17), 'NPC': (0.41, 0.93, 0.51, 1.15)}
K0 = {'MT': ((3.839, -0.024), (0.003, 3.851)), 'NPC': ((3.826, -0.028), (0.013, 3.845))}


def _feats(x, y, order):
    u, v = (x - 180.0) / 180.0, (y - 100.0) / 100.0
    return torch.stack([u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i)], -1)


class RegMap:
    """the in-situ polynomial map (register.PolyMap) in torch: ch0 crop coords -> channel c crop coords"""
    def __init__(self, cell, dev):
        d = json.load(open(f'{P4}/smlm/reg/reg_insitu_{cell}.json'))
        self.order = d['order']
        self.ci = {int(c): torch.tensor(v, device=dev, dtype=torch.float32) for c, v in d['inv'].items()}

    def __call__(self, c, x, y):
        if c == 0: return x, y
        Fm = _feats(x, y, self.order) @ self.ci[c]
        return Fm[..., 0], Fm[..., 1]


def perturbed_states(st, n, rng):
    """PSF jitter bank: the fitted state and n - 1 perturbations with the measured field-dependence magnitudes"""
    out = [st]
    for i in range(n - 1):
        s = json.loads(json.dumps(st))
        s['blur_px2'] = max(0.05, s['blur_px2'] + rng.uniform(-0.08, 0.08))
        zt, zb = list(s['zt']), list(s['zb']); i6 = s['meta']['zidx'].index(6)
        dn = rng.uniform(-0.012, 0.012); zt[i6] += dn; zb[i6] -= dn; s['zt'], s['zb'] = zt, zb
        e = math.radians(rng.uniform(-8, 8)); s['phi'] = [s['phi'][0], s['phi'][1] + e, s['phi'][2], s['phi'][3] + e]
        t = np.array(s['tau']) * np.exp(rng.uniform(-0.08, 0.08, 4)); s['tau'] = (t / t.sum()).tolist()
        s['gamma'] = float(np.clip(s['gamma'] * rng.uniform(0.75, 1.1), 0.05, 0.98))
        out.append(s)
    return out


class RealSim4Pi:
    def __init__(self, psf_state, H=64, W=64, cells=('MT', 'NPC'), n_psf=6, P=17, dev='cuda', seed=0, wsd=1.0, sub=2):
        self.H, self.W, self.dev, self.P, self.wsd = H, W, dev, P, wsd
        self.g = torch.Generator(device=dev).manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.luts = [LUT4Pi(s, P=P, dev=dev, sub=sub) for s in perturbed_states(psf_state, n_psf, rng)]
        self.cells = cells
        self.reg = {c: RegMap(c, dev) for c in cells}

    # ---- random helpers
    def u(self, n, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(n, device=self.dev, generator=self.g)

    def lu(self, n, lo, hi):
        return torch.exp(self.u(n, math.log(lo), math.log(hi)))

    def emitters(self, n_scene, dens, zr, kind=None):
        """emitter positions per scene: list of (x, y, z) tensors (ch0 px within the scene, um)"""
        H, W = self.H, self.W
        area = H * W * 0.128 ** 2
        out = []
        for b in range(n_scene):
            n = int(torch.poisson(torch.tensor(float(dens[b]) * area * 1.6, device=self.dev), generator=self.g).item()) + 1
            kd = {'uniform': 0.1, 'filament': 0.5, 'ring': 0.75, 'pair': 0.9}.get(kind, None) if kind else None
            kind_u = self.u(1).item() if kd is None else kd
            if kind_u < 0.35:      # uniform
                x = self.u(n, -4, W + 4); y = self.u(n, -4, H + 4); z = self.u(n, *zr)
            elif kind_u < 0.65:    # filaments
                nf = int(self.u(1, 1, 5).item())
                xs, ys, zs = [], [], []
                for f in range(nf):
                    k = max(1, n // nf)
                    x0, y0 = self.u(1, 0, W).item(), self.u(1, 0, H).item(); th = self.u(1, 0, math.pi).item()
                    s = self.u(k, -1.0, 1.0) * max(H, W)
                    za, zb_ = self.u(2, *zr).tolist()
                    ang = self.u(k, 0, 2 * math.pi); rr = 0.020 / 0.128
                    xs.append(x0 + s * math.cos(th) - rr * torch.cos(ang) * math.sin(th))
                    ys.append(y0 + s * math.sin(th) + rr * torch.cos(ang) * math.cos(th))
                    zs.append(za + (zb_ - za) * (s / max(H, W) + 1) / 2 + 0.020 * torch.sin(ang))
                x, y, z = torch.cat(xs), torch.cat(ys), torch.cat(zs)
            elif kind_u < 0.85:    # NPC-like rings
                npore = max(1, n // 6)
                cx, cy, cz = self.u(npore, 0, W), self.u(npore, 0, H), self.u(npore, zr[0] + 0.03, zr[1] - 0.03)
                k = torch.randint(0, 16, (n,), device=self.dev, generator=self.g); p = torch.randint(0, npore, (n,), device=self.dev, generator=self.g)
                a = 2 * math.pi * (k % 8) / 8 + (k // 8) * 0.12 + 0.05 * torch.randn(n, device=self.dev, generator=self.g)
                r = 0.0537 / 0.128
                x = cx[p] + r * torch.cos(a); y = cy[p] + r * torch.sin(a); z = cz[p] + torch.where(k // 8 == 0, -0.025, 0.025)
            else:                # close pairs
                m = max(1, n // 2)
                x0, y0, z0 = self.u(m, 0, W), self.u(m, 0, H), self.u(m, *zr)
                d = self.lu(m, 0.3, 3.0); th = self.u(m, 0, 2 * math.pi)
                dz = self.u(m, -0.3, 0.3)
                x = torch.cat([x0, x0 + d * torch.cos(th)]); y = torch.cat([y0, y0 + d * torch.sin(th)])
                z = torch.cat([z0, (z0 + dz).clamp(*zr)])
            out.append((x, y, z.clamp(*zr)))
        return out

    def kinetics(self, n, p_through):
        t = torch.arange(6, device=self.dev, dtype=torch.float32)
        mode = self.u(n)
        ton = torch.where(mode < p_through, torch.full((n,), -10.0, device=self.dev), self.u(n, -1, 6))
        toff = torch.where(mode < p_through, torch.full((n,), 20.0, device=self.dev), self.u(n, 0, 7))
        half = (mode >= p_through) & (mode < p_through + (1 - p_through) / 2)
        toff = torch.where(half, torch.full((n,), 20.0, device=self.dev), toff)
        ton = torch.where(mode >= p_through + (1 - p_through) / 2, torch.full((n,), -10.0, device=self.dev), ton)
        return (torch.minimum(t[None] + 1, toff[:, None]) - torch.maximum(t[None], ton[:, None])).clamp(0, 1)

    def sample(self, B=8, dens=(0.01, 2.0), zr=(-0.4, 0.4), F_med=450.0, F_rng=(60.0, 6000.0), p_through=0.5,
               rn=1.14, offerr=0.2, kind=None, delta_deg=None):
        """vectorised over scenes: one LUT evaluation per (PSF variant, channel, angle), one scatter per (channel, frame)"""
        dev, H, W, P = self.dev, self.H, self.W, self.P
        h = P // 2; pad = h + 2
        dsc = self.lu(B, *dens)
        ci = (self.u(B) * len(self.cells)).long().clamp(max=len(self.cells) - 1)              # cell per scene
        cell = [self.cells[int(c)] for c in ci]
        ox = self.u(B, 60, 300 - W).floor(); oy = self.u(B, 0, 200 - H).floor()
        ems = self.emitters(B, dsc, zr, kind)
        x = torch.cat([e[0] for e in ems]); y = torch.cat([e[1] for e in ems]); z = torch.cat([e[2] for e in ems])
        sb = torch.cat([torch.full((len(e[0]),), b, device=dev, dtype=torch.long) for b, e in enumerate(ems)])
        n = len(x)
        # per-scene parameters
        k0 = torch.stack([torch.tensor(K0[c], device=dev) for c in cell]) * (1 + self.u(B * 2, -0.003, 0.003)).reshape(B, 2, 1)
        psi = (2 * math.pi * torch.arange(3, device=dev) / 3)[None, None] + self.u(B * 2, 0, 2 * math.pi).reshape(B, 2, 1) \
            + torch.deg2rad(self.u(B * 6, -3, 3)).reshape(B, 2, 3)
        mdep = self.u(B, 0.6, 0.95)
        lut_i = (self.u(B) * len(self.luts)).long().clamp(max=len(self.luts) - 1)
        cxy = self.u(B * 4, -1, 1).reshape(B, 4); phi_off = self.u(B, 0, 2 * math.pi)
        delta = torch.deg2rad(self.u(B, -16, -8)) if delta_deg is None else torch.full((B,), math.radians(delta_deg), device=dev)
        bgl = torch.stack([torch.tensor(BG[c], device=dev) for c in cell]) * self.lu(B * 4, 0.5, 2.0).reshape(B, 4)
        # per emitter
        X, Y = x + ox[sb], y + oy[sb]
        ang = torch.tensor([0, 0, 0, 1, 1, 1], device=dev); st = torch.tensor([0, 1, 2, 0, 1, 2], device=dev)
        kk = k0[sb][:, ang]                                                                    # [n,6,2]
        Pt = 1 + mdep[sb, None] * torch.cos(kk[..., 0] * X[:, None] + kk[..., 1] * Y[:, None] - psi[sb][:, ang, st])
        q = self.kinetics(n, p_through)
        Fr = (F_med * torch.exp(0.6 * torch.randn(n, device=dev, generator=self.g))).clamp(*F_rng)
        Nt = Fr[:, None] * q * Pt
        c4 = cxy[sb]
        Phi = phi_off[sb] + math.radians(60) * (c4[:, 0] * x / W + c4[:, 1] * y / H + c4[:, 2] * (x / W) * (y / H))
        # channel positions and crop origins (registration of the scene's cell)
        origin = torch.zeros(B, 4, 2, device=dev)
        lx = torch.zeros(n, 4, device=dev); ly = torch.zeros(n, 4, device=dev)
        for cname in set(cell):
            msc = torch.tensor([c == cname for c in cell], device=dev); me = msc[sb]
            reg = self.reg[cname]
            for c in range(4):
                xc, yc = reg(c, X[me], Y[me])
                cx0, cy0 = reg(c, ox[msc], oy[msc])
                origin[msc, c, 0] = torch.round(cx0); origin[msc, c, 1] = torch.round(cy0)
                lx[me, c] = xc - origin[sb[me], c, 0]; ly[me, c] = yc - origin[sb[me], c, 1]
        img = torch.zeros(B, 6, 4, H + 2 * pad, W + 2 * pad, device=dev)
        T = torch.zeros(n, 6, 4, device=dev); C = torch.zeros(n, 6, 4, device=dev)
        o_ = torch.arange(P, device=dev)
        for li in range(len(self.luts)):
            me = lut_i[sb] == li
            if not me.any(): continue
            idx = torch.nonzero(me)[:, 0]
            lut = self.luts[li]
            for c in range(4):
                ix, iy = torch.round(lx[idx, c]), torch.round(ly[idx, c])
                pos = torch.stack([lx[idx, c] - ix, ly[idx, c] - iy, z[idx]], 1)
                ex = (o_[None, None, :] - h).float() - pos[:, 0, None, None]; ey = (o_[None, :, None] - h).float() - pos[:, 1, None, None]
                w = torch.exp(-(ex ** 2 + ey ** 2) / (2 * self.wsd ** 2))
                Yi = ((iy.long() + pad)[:, None, None] + o_[None, :, None] - h).expand(-1, P, P)
                Xi = ((ix.long() + pad)[:, None, None] + o_[None, None, :] - h).expand(-1, P, P)
                ok = (Yi >= 0) & (Yi < img.shape[-2]) & (Xi >= 0) & (Xi < img.shape[-1])
                bi = sb[idx][:, None, None].expand(-1, P, P)
                inc, cr, cim = (row[c].evaluate(pos) for row in lut.luts)                         # 3 components, once
                for a in range(2):
                    ph = (Phi[idx] + (delta[sb[idx]] if a else 0.0))[:, None, None]
                    g = inc + torch.cos(ph) * cr - torch.sin(ph) * cim                                 # [m,P,P]
                    gs = g.sum((-1, -2)); gw = (g * w).sum((-1, -2))
                    for t in range(3 * a, 3 * a + 3):
                        img[:, t, c].index_put_((bi[ok], Yi[ok], Xi[ok]), (g * Nt[idx, t, None, None])[ok], accumulate=True)
                        T[idx, t, c] = Nt[idx, t] * gs; C[idx, t, c] = Nt[idx, t] * gw
        img = img[..., pad:pad + H, pad:pad + W]
        yy, xx = torch.meshgrid(torch.arange(H, device=dev) / H, torch.arange(W, device=dev) / W, indexing='ij')
        a1, a2, a3 = self.u(B * 4, -0.3, 0.3).reshape(B, 1, 4, 1, 1), self.u(B * 4, 0, 2 * math.pi).reshape(B, 1, 4, 1, 1), self.u(B * 4, 0.3, 1.5).reshape(B, 1, 4, 1, 1)
        struct = 1 + a1 * torch.sin(2 * math.pi * a3 * xx + a2) * torch.cos(2 * math.pi * a3 * yy)
        fl = 1 + 0.05 * torch.randn(B, 6, 4, 1, 1, device=dev, generator=self.g)
        lam = img + bgl[:, None, :, None, None] * struct * fl
        obs = torch.poisson(lam.clamp(min=0), generator=self.g) + rn * torch.randn(lam.shape, device=dev, generator=self.g) \
            + offerr * torch.randn(B, 1, 4, H, W, device=dev, generator=self.g)
        det_in = self.register_sum(obs, origin, ox, oy, cell)
        inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        G = dict(b=sb[inside], x=x[inside], y=y[inside], z=z[inside], Nt=Nt[inside], q=q[inside], Phi=Phi[inside], T=T[inside],
                 C=C[inside], lx=lx[inside], ly=ly[inside], delta=delta[sb[inside]], psi=psi[sb[inside]], k=k0[sb[inside]],
                 m=mdep[sb[inside]], X=X[inside], Y=Y[inside])
        scene = dict(k=k0, psi=psi, m=mdep, cxy=cxy, phi_off=phi_off, delta=delta)
        return dict(img=obs, mu=lam, origin=origin, ox=ox, oy=oy, cell=cell, det_in=det_in, gt=G, dens=dsc, bg=bgl, scene=scene)

    @staticmethod
    def phase_at(scene, b, x, y, H, W):
        """4Pi phase field value of scene b at scene ch0 position (x, y) (as rendered)"""
        c4 = scene['cxy'][b]
        return scene['phi_off'][b] + math.radians(60) * (c4[:, 0] * x / W + c4[:, 1] * y / H + c4[:, 2] * (x / W) * (y / H))

    def register_sum(self, obs, origin, ox, oy, cell):
        """registered 4-channel sum per frame on the ch0 grid (bilinear warp, as front4pi.Front.detect)"""
        B, T_, C_, H, W = obs.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=self.dev, dtype=torch.float32), torch.arange(W, device=self.dev, dtype=torch.float32), indexing='ij')
        out = obs[:, :, 0].clone()
        for b in range(B):
            reg = self.reg[cell[b]]
            for c in (1, 2, 3):
                xc, yc = reg(c, xx + ox[b], yy + oy[b])
                gx = (xc - origin[b, c, 0]) / (W - 1) * 2 - 1; gy = (yc - origin[b, c, 1]) / (H - 1) * 2 - 1
                grid = torch.stack([gx, gy], -1)[None].expand(T_, -1, -1, -1)
                out[b] += F.grid_sample(obs[b, :, c:c + 1], grid, mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
        return out
