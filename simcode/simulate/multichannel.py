"""MultiChannelAcquisitionSim: experiment-matched full-scene simulator of a multi-channel acquisition (the 4Pi-SIMFLUX
set-up: 4 interference channels; the analogue of simcode/simulate/acquisition.py for one channel), for the training of
the 4Pi detector and the simulated 4Pi movies.

A batch = B scenes of H x W px (ch0 grid), T frames (patterns=True: T = 6 = 2 angles x 3 steps; patterns=False, plain
4Pi: any T, one 4Pi phase for all frames, e.g. T = 1), C channels (C = the channels of the PSF table, 4 for the 4Pi LUT).
Each channel is rendered in its OWN crop coordinates through the registration map of the scene's cell (a CellSpec: map
json (a PolyMap: simcode/channels/registration.py) with its crop size, the field the scene origin is drawn in, the
background level per channel, the SIM wave vectors; the 4Pi-SIMFLUX cells MT / NPC: fourpi_sf_cells(), the default), so
the front end's code path (registered sum by bilinear warp for the detector; unwarped channel patches at mapped
positions for the head) is the real one.

Emitters (per scene): density log-uniform `dens` ON emitters / um^2 / frame; a mix of uniform random emitters,
filament-like lines (label radius 20 nm, tilted in z), NPC-like rings (8 corners, two rings 50 nm apart in z) and close
pairs (0.3-3 px, dz up to +-0.3 um). z in `zr`. Kinetics inside the window (on-through, start / end in the window; T = 1:
the on fraction of the frame), per-event photon rate log-normal around `F_med` e/frame (sd log `F_sd`, clipped to `F_rng`).
Pattern: k from the measured value +- 0.3 %, step phases 120 deg +- 3 deg, the phase at every position arbitrary (a
per-scene offset uniform over 2 pi: the v3 lesson), depth 0.6-0.95. patterns=False: no SIM modulation (plain 4Pi; the
same random draws, so the scenes are otherwise those of patterns=True).
4Pi phase field: smooth random field over the scene (+- phase_field_deg, 60) + per-scene offset uniform over 2 pi,
per-angle offset delta_a in [-16, -8] deg (patterns only).
Registration error (reg_shift_sd px, reg_lin_sd; default 0 = none): per scene and channel c >= 1 a random shift and
linear distortion of the TRUE registration, drawn from a separate generator (the other draws are unchanged); the
channels are rendered through the true map, the detector input (registered sum) and the returned crop origins use the
nominal map, as a front end with an estimated map does.
PSF: a bank of LUTs (the fitted model + perturbations: blur, Noll 6, inter-pair phase, tau, coherence), one per scene.
Background per channel: measured levels x log-uniform(0.5, 2), smooth structure +-30 %, slow per-frame fluctuation.
Camera: Poisson + Gaussian read noise (rn, 1.14 e) + per-pixel offset error (offerr, 0.2 e), output in e (offset-subtracted, x gain).
Returns dict:
  img [B,T,4,H,W] channel crops (e), origin [B,4,2] (crop origin of each channel, channel px), det_in [B,T,H,W]
  (registered 4-channel sum per frame; det=False: None), and per emitter-window GT lists: b, x, y (ch0 px in the scene),
  z, Nt [T], q [T], Phi (4Pi phase at the emitter), T [T,C], C [T,C] (read-out targets as in simulate/fourpi_patches.py).
candidate_patches(S, ...): emitter-network training samples from a scene batch (the GT emitters as candidates, detection
jitter, the real neighbours and backgrounds), in the format of FourPiPatchSim.sample.
"""
import json, math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from simcode.fourpi.spline_table import LUT4Pi

from simcode.config import root
P4 = root('fourpi')


@dataclass
class CellSpec:
    """one acquisition (cell) of the scene simulator: registration map json and the crop size it was made for, the field
    [x0, x1) x [y0, y1) (ch0 crop px) the scene origin is drawn in, the measured background per channel (e/px/frame),
    the SIM wave vectors per angle (rad/px; None without patterns)"""
    name: str
    regmap: str
    W: int
    H: int
    xlim: tuple
    ylim: tuple
    bg: tuple
    k0: tuple = None


def fourpi_sf_cells():
    """the 4Pi-SIMFLUX cells (MT = Cell03, NPC = Cell07): in-situ maps of the 4Pi work dir, 360 x 200 crops"""
    BG = {'MT': (0.42, 1.06, 0.54, 1.17), 'NPC': (0.41, 0.93, 0.51, 1.15)}
    K0 = {'MT': ((3.839, -0.024), (0.003, 3.851)), 'NPC': ((3.826, -0.028), (0.013, 3.845))}
    return [CellSpec(c, f'{P4}/smlm/reg/reg_insitu_{c}.json', 360, 200, (60, 300), (0, 200), BG[c], K0[c]) for c in ('MT', 'NPC')]


class TorchMap:
    """the inverse of a registration map (PolyMap) in torch: ch0 crop coords -> channel c crop coords"""
    def __init__(self, pm, dev):
        self.order, self.norm = pm.order, pm.norm
        self.ci = {int(c): torch.tensor(v, device=dev, dtype=torch.float32) for c, v in pm.ci.items()}

    def feats(self, x, y):
        cx, cy, sx, sy = self.norm
        u, v = (x - cx) / sx, (y - cy) / sy
        return torch.stack([u ** i * v ** j for i in range(self.order + 1) for j in range(self.order + 1 - i)], -1)

    def __call__(self, c, x, y):
        if c == 0: return x, y
        Fm = self.feats(x, y) @ self.ci[c]
        return Fm[..., 0], Fm[..., 1]


def cell_map(spec, dev):
    from simcode.channels.registration import PolyMap
    return TorchMap(PolyMap.load(spec.regmap, size=(spec.W, spec.H)), dev)


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


class MultiChannelAcquisitionSim:
    def __init__(self, psf_state, H=64, W=64, cells=None, n_psf=6, P=17, dev='cuda', seed=0, wsd=1.0, sub=2,
                 patterns=True, phase_field_deg=60.0, reg_shift_sd=0.0, reg_lin_sd=0.0, T=6, zlut=(-0.6, 0.6)):
        """cells: list of CellSpec (default: the 4Pi-SIMFLUX cells); T frames per window (patterns: 6); zlut: z range of the PSF tables"""
        if patterns and T != 6: raise ValueError('MultiChannelAcquisitionSim: the SIM pattern needs T = 6 (2 angles x 3 steps)')
        self.H, self.W, self.dev, self.P, self.wsd, self.T = H, W, dev, P, wsd, T
        self.g = torch.Generator(device=dev).manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.luts = [LUT4Pi(s, P=P, dev=dev, sub=sub, zmin=zlut[0], zmax=zlut[1]) for s in perturbed_states(psf_state, n_psf, rng)]
        cells = fourpi_sf_cells() if cells is None else list(cells)
        self.spec = {c.name: c for c in cells}
        self.cells = [c.name for c in cells]
        self.reg = {c.name: cell_map(c, dev) for c in cells}
        self.C = len(self.luts[0].luts[0])
        self.ang = (0, 0, 0, 1, 1, 1) if patterns else (0,) * T        # angle index (pattern orientation, 4Pi offset delta_a) of every frame
        self.patterns, self.phase_field_deg, self.reg_shift_sd, self.reg_lin_sd = patterns, phase_field_deg, reg_shift_sd, reg_lin_sd
        self.g_reg = torch.Generator(device=dev).manual_seed(seed + 7919)

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
        T = self.T
        t = torch.arange(T, device=self.dev, dtype=torch.float32)
        mode = self.u(n)
        ton = torch.where(mode < p_through, torch.full((n,), -10.0, device=self.dev), self.u(n, -1, T))
        toff = torch.where(mode < p_through, torch.full((n,), 20.0, device=self.dev), self.u(n, 0, T + 1))
        half = (mode >= p_through) & (mode < p_through + (1 - p_through) / 2)
        toff = torch.where(half, torch.full((n,), 20.0, device=self.dev), toff)
        ton = torch.where(mode >= p_through + (1 - p_through) / 2, torch.full((n,), -10.0, device=self.dev), ton)
        return (torch.minimum(t[None] + 1, toff[:, None]) - torch.maximum(t[None], ton[:, None])).clamp(0, 1)

    def sample(self, B=8, dens=(0.01, 2.0), zr=(-0.4, 0.4), F_med=450.0, F_rng=(60.0, 6000.0), p_through=0.5,
               rn=1.14, offerr=0.2, kind=None, delta_deg=None, ksum=1, F_sd=0.6, bg_rng=(0.5, 2.0), det=True, phase=None):
        """vectorised over scenes: one LUT evaluation per (PSF variant, channel, angle), one scatter per (channel, frame).
        bg_rng: range of the log-uniform factor on the cell's background levels; det=False: no detector input;
        phase = dict(phi_off (rad), cxy (4)): the same 4Pi phase field in every scene (simulated movies; drawn otherwise)"""
        dev, H, W, P, NC, T_ = self.dev, self.H, self.W, self.P, self.C, self.T
        h = P // 2; pad = h + 2
        dsc = self.lu(B, *dens)
        ci = (self.u(B) * len(self.cells)).long().clamp(max=len(self.cells) - 1)              # cell per scene
        cell = [self.cells[int(c)] for c in ci]
        sp = [self.spec[c] for c in cell]
        lo = torch.tensor([s.xlim[0] for s in sp], device=dev, dtype=torch.float32); hi = torch.tensor([s.xlim[1] - W for s in sp], device=dev, dtype=torch.float32)
        ox = (lo + (hi - lo) * torch.rand(B, device=dev, generator=self.g)).floor()
        lo = torch.tensor([s.ylim[0] for s in sp], device=dev, dtype=torch.float32); hi = torch.tensor([s.ylim[1] - H for s in sp], device=dev, dtype=torch.float32)
        oy = (lo + (hi - lo) * torch.rand(B, device=dev, generator=self.g)).floor()
        ems = self.emitters(B, dsc, zr, kind)
        x = torch.cat([e[0] for e in ems]); y = torch.cat([e[1] for e in ems]); z = torch.cat([e[2] for e in ems])
        sb = torch.cat([torch.full((len(e[0]),), b, device=dev, dtype=torch.long) for b, e in enumerate(ems)])
        n = len(x)
        # per-scene parameters
        k0 = torch.stack([torch.tensor(s.k0 if s.k0 is not None else ((0.0, 0.0), (0.0, 0.0)), device=dev) for s in sp]) \
            * (1 + self.u(B * 2, -0.003, 0.003)).reshape(B, 2, 1)
        psi = (2 * math.pi * torch.arange(3, device=dev) / 3)[None, None] + self.u(B * 2, 0, 2 * math.pi).reshape(B, 2, 1) \
            + torch.deg2rad(self.u(B * 6, -3, 3)).reshape(B, 2, 3)
        mdep = self.u(B, 0.6, 0.95)
        lut_i = (self.u(B) * len(self.luts)).long().clamp(max=len(self.luts) - 1)
        cxy = self.u(B * 4, -1, 1).reshape(B, 4); phi_off = self.u(B, 0, 2 * math.pi)
        if phase is not None:
            cxy = torch.tensor(phase['cxy'], device=dev, dtype=torch.float32)[None].expand(B, 4).contiguous()
            phi_off = torch.full((B,), float(phase['phi_off']), device=dev)
        delta = torch.deg2rad(self.u(B, -16, -8)) if delta_deg is None else torch.full((B,), math.radians(delta_deg), device=dev)
        bgl = torch.stack([torch.tensor(s.bg, device=dev) for s in sp]) * self.lu(B * NC, *bg_rng).reshape(B, NC)
        if ksum != 1:   # K summed windows (real dense data by summing): K x background, sqrt(K) read noise, K x offset error
            bgl = bgl * ksum; rn = rn * math.sqrt(ksum); offerr = offerr * ksum
        # per emitter
        X, Y = x + ox[sb], y + oy[sb]
        if self.patterns:
            ang = torch.tensor([0, 0, 0, 1, 1, 1], device=dev); st = torch.tensor([0, 1, 2, 0, 1, 2], device=dev)
            kk = k0[sb][:, ang]                                                                # [n,6,2]
            Pt = 1 + mdep[sb, None] * torch.cos(kk[..., 0] * X[:, None] + kk[..., 1] * Y[:, None] - psi[sb][:, ang, st])
        else:
            Pt = torch.ones(n, T_, device=dev)                                                 # plain 4Pi (no SIM)
        q = self.kinetics(n, p_through)
        Fr = (F_med * torch.exp(F_sd * torch.randn(n, device=dev, generator=self.g))).clamp(*F_rng)
        Nt = Fr[:, None] * q * Pt
        c4 = cxy[sb]
        Phi = phi_off[sb] + math.radians(self.phase_field_deg) * (c4[:, 0] * x / W + c4[:, 1] * y / H + c4[:, 2] * (x / W) * (y / H))
        # channel positions and crop origins (registration of the scene's cell)
        origin = torch.zeros(B, NC, 2, device=dev)
        lx = torch.zeros(n, NC, device=dev); ly = torch.zeros(n, NC, device=dev)
        rerr = self.registration_error(B)
        for cname in set(cell):
            msc = torch.tensor([c == cname for c in cell], device=dev); me = msc[sb]
            reg = self.reg[cname]
            for c in range(NC):
                xc, yc = reg(c, X[me], Y[me])
                if rerr is not None and c > 0: xc, yc = self.perturb(rerr, c, sb[me], xc, yc)
                cx0, cy0 = reg(c, ox[msc], oy[msc])
                origin[msc, c, 0] = torch.round(cx0); origin[msc, c, 1] = torch.round(cy0)
                lx[me, c] = xc - origin[sb[me], c, 0]; ly[me, c] = yc - origin[sb[me], c, 1]
        img = torch.zeros(B, T_, NC, H + 2 * pad, W + 2 * pad, device=dev)
        T = torch.zeros(n, T_, NC, device=dev); C = torch.zeros(n, T_, NC, device=dev)
        o_ = torch.arange(P, device=dev)
        for li in range(len(self.luts)):
            me = lut_i[sb] == li
            if not me.any(): continue
            idx = torch.nonzero(me)[:, 0]
            lut = self.luts[li]
            for c in range(NC):
                ix, iy = torch.round(lx[idx, c]), torch.round(ly[idx, c])
                pos = torch.stack([lx[idx, c] - ix, ly[idx, c] - iy, z[idx]], 1)
                ex = (o_[None, None, :] - h).float() - pos[:, 0, None, None]; ey = (o_[None, :, None] - h).float() - pos[:, 1, None, None]
                w = torch.exp(-(ex ** 2 + ey ** 2) / (2 * self.wsd ** 2))
                Yi = ((iy.long() + pad)[:, None, None] + o_[None, :, None] - h).expand(-1, P, P)
                Xi = ((ix.long() + pad)[:, None, None] + o_[None, None, :] - h).expand(-1, P, P)
                ok = (Yi >= 0) & (Yi < img.shape[-2]) & (Xi >= 0) & (Xi < img.shape[-1])
                bi = sb[idx][:, None, None].expand(-1, P, P)
                inc, cr, cim = (row[c].render(pos) for row in lut.luts)                         # 3 components, once
                for a in sorted(set(self.ang)):
                    ph = (Phi[idx] + (delta[sb[idx]] if a else 0.0))[:, None, None]
                    g = inc + torch.cos(ph) * cr - torch.sin(ph) * cim                                 # [m,P,P]
                    gs = g.sum((-1, -2)); gw = (g * w).sum((-1, -2))
                    for t in [t for t in range(T_) if self.ang[t] == a]:
                        img[:, t, c].index_put_((bi[ok], Yi[ok], Xi[ok]), (g * Nt[idx, t, None, None])[ok], accumulate=True)
                        T[idx, t, c] = Nt[idx, t] * gs; C[idx, t, c] = Nt[idx, t] * gw
        img = img[..., pad:pad + H, pad:pad + W]
        yy, xx = torch.meshgrid(torch.arange(H, device=dev) / H, torch.arange(W, device=dev) / W, indexing='ij')
        a1, a2, a3 = self.u(B * NC, -0.3, 0.3).reshape(B, 1, NC, 1, 1), self.u(B * NC, 0, 2 * math.pi).reshape(B, 1, NC, 1, 1), self.u(B * NC, 0.3, 1.5).reshape(B, 1, NC, 1, 1)
        struct = 1 + a1 * torch.sin(2 * math.pi * a3 * xx + a2) * torch.cos(2 * math.pi * a3 * yy)
        fl = 1 + 0.05 * torch.randn(B, T_, NC, 1, 1, device=dev, generator=self.g)
        lam = img + bgl[:, None, :, None, None] * struct * fl
        obs = torch.poisson(lam.clamp(min=0), generator=self.g) + rn * torch.randn(lam.shape, device=dev, generator=self.g) \
            + offerr * torch.randn(B, 1, NC, H, W, device=dev, generator=self.g)
        det_in = self.register_sum(obs, origin, ox, oy, cell) if det else None
        inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        G = dict(b=sb[inside], x=x[inside], y=y[inside], z=z[inside], Nt=Nt[inside], q=q[inside], Phi=Phi[inside], T=T[inside],
                 C=C[inside], lx=lx[inside], ly=ly[inside], delta=delta[sb[inside]], psi=psi[sb[inside]], k=k0[sb[inside]],
                 m=mdep[sb[inside]], X=X[inside], Y=Y[inside], F=Fr[inside])
        scene = dict(k=k0, psi=psi, m=mdep, cxy=cxy, phi_off=phi_off, delta=delta, lut_i=lut_i)
        if not self.patterns: G['m'] = torch.zeros_like(G['m'])
        return dict(img=obs, mu=lam, origin=origin, ox=ox, oy=oy, cell=cell, det_in=det_in, gt=G, dens=dsc, bg=bgl, scene=scene,
                    bgmap=bgl[:, None, :, None, None] * struct * fl)

    def candidate_patches(self, S, P=15, jit=0.35, min_photons=150.0, nb_r=7.0, max_n=None):
        """emitter-network training samples from a scene batch S (sample()): every GT emitter inside the scene with >=
        min_photons in the window is a candidate at the pixel nearest to its position + detection jitter (sd jit px);
        per channel the P x P patch of the scene at round(nominal map(candidate)) (as ChannelFront.patches). The format
        of FourPiPatchSim.sample: img [n,T,4,P,P], mu0 [n,T,4,P,P] (the candidate's own expected image, exactly as
        rendered), bg [n,T,4,P,P] (background), x, y (emitter - candidate pixel, ch0 px), z, Phi0, F (full-on photons /
        frame), q, Nt, m = eps = 0 (no pattern), doff [n,4,2], T, C, nnb (other emitters with >= min_photons / 3 within
        nb_r px). Plain 4Pi scenes only."""
        if self.patterns: raise NotImplementedError('candidate_patches: plain 4Pi scenes (patterns=False) only')
        dev, H, W, NC = self.dev, self.H, self.W, self.C
        G = S['gt']; h = P // 2; Pl = self.P; hl = Pl // 2
        b = G['b']; n0 = len(b)
        tot = G['Nt'].sum(1)
        g = self.g
        cx = torch.round(G['x'] + jit * torch.randn(n0, device=dev, generator=g)); cy = torch.round(G['y'] + jit * torch.randn(n0, device=dev, generator=g))
        ok = (tot >= min_photons)
        mxy = torch.zeros(n0, NC, 2, device=dev)
        for cname in set(S['cell']):
            me = torch.tensor([S['cell'][int(i)] == cname for i in range(len(S['cell']))], device=dev)[b]
            reg = self.reg[cname]
            for c in range(NC):
                xc, yc = reg(c, cx[me] + S['ox'][b[me]], cy[me] + S['oy'][b[me]])
                mxy[me, c, 0] = xc - S['origin'][b[me], c, 0]; mxy[me, c, 1] = yc - S['origin'][b[me], c, 1]
        ctr = torch.round(mxy); doff = mxy - ctr
        ok &= ((ctr[..., 0] >= h) & (ctr[..., 0] < W - h) & (ctr[..., 1] >= h) & (ctr[..., 1] < H - h)).all(1)
        idx = torch.nonzero(ok)[:, 0]
        if max_n is not None and len(idx) > max_n: idx = idx[torch.randperm(len(idx), device=dev, generator=g)[:max_n]]
        n = len(idx); bi = b[idx]; ctr = ctr[idx].long(); doff = doff[idx]
        o = torch.arange(P, device=dev) - h
        Yi = ctr[:, :, 1, None] + o[None, None]; Xi = ctr[:, :, 0, None] + o[None, None]           # [n,C,P]
        tt = torch.arange(self.T, device=dev); ch = torch.arange(NC, device=dev)
        ix = (bi[:, None, None, None, None], tt[None, :, None, None, None], ch[None, None, :, None, None], Yi[:, None, :, :, None], Xi[:, None, :, None, :])
        img = S['img'][ix]; bg = S['bgmap'][ix]
        # the candidate's own expected image: the scene's stamp (centred on round(lx)) shifted into the patch
        mu0 = torch.zeros(n, self.T, NC, P, P, device=dev)
        lx, ly, z = G['lx'][idx], G['ly'][idx], G['z'][idx]
        ang = torch.tensor(self.ang, device=dev)
        for li in range(len(self.luts)):
            me = S['scene']['lut_i'][bi] == li
            if not me.any(): continue
            k = torch.nonzero(me)[:, 0]; lut = self.luts[li]
            for c in range(NC):
                rx, ry = torch.round(lx[k, c]), torch.round(ly[k, c])
                pos = torch.stack([lx[k, c] - rx, ly[k, c] - ry, z[k]], 1)
                inc, cr, cim = (row[c].render(pos) for row in lut.luts)
                sx = (ctr[k, c, 0] - rx.long())[:, None] + o[None] + hl; sy = (ctr[k, c, 1] - ry.long())[:, None] + o[None] + hl   # stamp index of the patch pixels
                vx = (sx >= 0) & (sx < Pl); vy = (sy >= 0) & (sy < Pl)
                gi = (torch.arange(len(k), device=dev)[:, None, None], sy.clamp(0, Pl - 1)[:, :, None], sx.clamp(0, Pl - 1)[:, None, :])
                msk = (vy[:, :, None] & vx[:, None, :]).float()
                for a in sorted(set(self.ang)):
                    ph = (G['Phi'][idx][k] + (G['delta'][idx][k] if a else 0.0))[:, None, None]
                    gimg = (inc + torch.cos(ph) * cr - torch.sin(ph) * cim)[gi] * msk
                    for t in [t for t in range(self.T) if self.ang[t] == a]:
                        mu0[k, t, c] = G['Nt'][idx][k, t, None, None] * gimg
        # neighbours (other emitters of the scene within nb_r px, ch0)
        d2 = (G['x'][idx, None] - G['x'][None]) ** 2 + (G['y'][idx, None] - G['y'][None]) ** 2
        same = (b[idx, None] == b[None]) & (d2 < nb_r ** 2) & (d2 > 1e-8) & (tot[None] >= min_photons / 3)
        x = G['x'][idx] - cx[idx]; y = G['y'][idx] - cy[idx]
        return dict(img=img, mu0=mu0, bg=bg, x=x, y=y, z=z, Phi0=G['Phi'][idx], F=G['F'][idx], q=G['q'][idx], Nt=G['Nt'][idx],
                    m=torch.zeros(n, device=dev), eps=torch.zeros(n, 2, device=dev), doff=doff, T=G['T'][idx], C=G['C'][idx],
                    nnb=same.sum(1), b=bi)

    @staticmethod
    def phase_at(scene, b, x, y, H, W, phase_field_deg=60.0):
        """4Pi phase field value of scene b at scene ch0 position (x, y) (as rendered)"""
        c4 = scene['cxy'][b]
        return scene['phi_off'][b] + math.radians(phase_field_deg) * (c4[:, 0] * x / W + c4[:, 1] * y / H + c4[:, 2] * (x / W) * (y / H))

    def registration_error(self, B):
        """per scene and channel: shift (2) and linear terms (4) of the registration error, or None"""
        if self.reg_shift_sd <= 0 and self.reg_lin_sd <= 0: return None
        g, dev = self.g_reg, self.dev
        return dict(shift=self.reg_shift_sd * torch.randn(B, self.C, 2, device=dev, generator=g),
                    lin=self.reg_lin_sd * torch.randn(B, self.C, 2, 2, device=dev, generator=g))

    def perturb(self, e, c, b, x, y):
        """true channel-c coordinates = nominal + shift + lin . ((x, y) - crop centre) / crop half size"""
        cx, cy, sx, sy = next(iter(self.reg.values())).norm
        u, v = (x - cx) / sx, (y - cy) / sy
        L = e['lin'][b, c]
        return (x + e['shift'][b, c, 0] + L[:, 0, 0] * u + L[:, 0, 1] * v,
                y + e['shift'][b, c, 1] + L[:, 1, 0] * u + L[:, 1, 1] * v)

    def register_sum(self, obs, origin, ox, oy, cell):
        """registered channel sum per frame on the ch0 grid (bilinear warp with the nominal map, as ChannelFront.registered_sum)"""
        B, T_, C_, H, W = obs.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=self.dev, dtype=torch.float32), torch.arange(W, device=self.dev, dtype=torch.float32), indexing='ij')
        out = obs[:, :, 0].clone()
        for b in range(B):
            reg = self.reg[cell[b]]
            for c in range(1, C_):
                xc, yc = reg(c, xx + ox[b], yy + oy[b])
                gx = (xc - origin[b, c, 0]) / (W - 1) * 2 - 1; gy = (yc - origin[b, c, 1]) / (H - 1) * 2 - 1
                grid = torch.stack([gx, gy], -1)[None].expand(T_, -1, -1, -1)
                out[b] += F.grid_sample(obs[b, :, c:c + 1], grid, mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
        return out
