"""FourPiPatchSim: candidate-patch simulator for 4Pi windows of T frames, 4 interference channels: 4Pi-SIMFLUX (pattern=True,
T = 6 = 2 angles x 3 steps) or plain 4Pi (pattern=False, T frames without SIM modulation and without per-angle 4Pi phase
offsets; T = 1: single frames).

One sample = the T x 4 unwarped channel patches (P x P, e) around a candidate, i.e. what the 4Pi emitter network sees
(simcode/emitter/fourpi.py), plus GT. The training data of the 4Pi emitter networks and the gates of the 4Pi solve.
Geometry: the main emitter is at (x, y) px relative to the candidate (detection jitter sd `jit`); channel c patch is
centred on round(x_cand + d_c) with a per-sample fractional channel offset d_c (the registration map's sub-pixel
part), so the emitter sits at (x, y) + o_c in that patch, o_c = d_c - round(d_c) in x and y ([n,4,2]); the head gets o_c.
Photons: per-frame full-on photons F (log-uniform), kinetics (on fraction q_t per frame: on-through, starts or ends in
the window), SIM pattern (k along x for angle 0, y for angle 1; steps 0/120/240 deg + per-window phase drift, depth m),
4Pi: phase Phi = Phi0 + delta_a (Phi0 uniform per sample; neighbours within the patch share Phi0 up to a small
field gradient), per-channel background with a linear gradient, Poisson + Gaussian read noise (rn, 1.14 e) + a residual
per-pixel offset error (offerr). Plain 4Pi: Phi = Phi0 in every frame, m = eps = 0 in the returned sample (the solve's
pattern factor is then 1), photons F x q_t; the random draws of the pattern are skipped.
GT targets per (t, c): T_tc = expected photons of the main emitter in the patch; C_tc = expected photons weighted by a
Gaussian window (sd `wsd` px) centred at the true position (the second, GT-definable read-out that carries the 4Pi
centre contrast).
"""
import math
import torch

K_PAT = (3.839, 3.851)            # rad/px (MT; NPC 3.826 / 3.845)
DELTA_A = (0.0, math.radians(-12.0))


class FourPiPatchSim:
    def __init__(self, lut, P=15, wsd=1.0, dev='cuda', seed=0, T=6, pattern=True):
        if pattern and T != 6: raise ValueError('FourPiPatchSim: the SIM pattern needs T = 6 (2 angles x 3 steps)')
        self.lut, self.P, self.wsd, self.dev, self.T, self.with_pattern = lut, P, wsd, dev, T, pattern
        self.ang = (0, 0, 0, 1, 1, 1) if pattern else (0,) * T          # angle index of every frame
        self.delta = DELTA_A if pattern else (0.0,)                       # 4Pi phase offset per angle
        self.g = torch.Generator(device=dev).manual_seed(seed)
        o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
        self.oy, self.ox = torch.meshgrid(o, o, indexing='ij')

    def u(self, n, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(n, device=self.dev, generator=self.g)

    def lu(self, n, lo, hi):
        return torch.exp(self.u(n, math.log(lo), math.log(hi)))

    def kinetics(self, n, p_on=0.6):
        """q [n,T]: on fraction per frame; on-through with prob p_on, else a start or end inside the window"""
        T = self.T
        t = torch.arange(T, device=self.dev, dtype=torch.float32)
        mode = self.u(n)
        ton = torch.where(mode < p_on, torch.full((n,), -10.0, device=self.dev), self.u(n, -1, T))
        toff = torch.where(mode < p_on, torch.full((n,), 20.0, device=self.dev), self.u(n, 0, T + 1))
        sw = (mode >= p_on) & (mode < p_on + (1 - p_on) / 2)      # half of the rest: only starts / only ends
        toff = torch.where(sw, torch.full((n,), 20.0, device=self.dev), toff)
        ton = torch.where((mode >= p_on + (1 - p_on) / 2), torch.full((n,), -10.0, device=self.dev), ton)
        q = (torch.minimum(t[None] + 1, toff[:, None]) - torch.maximum(t[None], ton[:, None])).clamp(0, 1)
        return q

    def pattern(self, x, y, m, eps):
        """[n,T] modulation factor 1 + m cos(k r - psi_s - eps_a) (plain 4Pi: 1)"""
        if not self.with_pattern: return torch.ones(len(x), self.T, device=self.dev)
        out = []
        for a in range(2):
            r = x if a == 0 else y
            for s in range(3):
                out.append(1 + m * torch.cos(K_PAT[a] * r - 2 * math.pi * s / 3 - eps[:, a]))
        return torch.stack(out, 1)

    def render_ch(self, pos, Phi, c):
        """[n,A,P,P]: channel c for the A angle offsets delta_a"""
        return torch.stack([self.lut.evaluate_ch(pos, Phi + self.delta[a], c) for a in range(len(self.delta))], 1)

    def render(self, pos, Phi):
        """expected per-channel image per frame-angle: [n,A,4,P,P] for angle offsets delta_a"""
        return torch.stack([self.lut.evaluate(pos, Phi + self.delta[a]) for a in range(len(self.delta))], 1)

    def sample(self, n, F=(60.0, 2500.0), zr=(-0.45, 0.45), bg=(0.2, 2.0), p_nb=0.5, nb_max=3, jit=0.35,
               m=(0.6, 0.95), rn=1.14, offerr=0.2, p_on=0.6, eps_range=0.3):
        d, P, T = self.dev, self.P, self.T
        x = torch.randn(n, device=d, generator=self.g) * jit
        y = torch.randn(n, device=d, generator=self.g) * jit
        z = self.u(n, *zr)
        Phi0 = self.u(n, 0, 2 * math.pi)
        Fm = self.lu(n, *F)
        q = self.kinetics(n, p_on)
        mm = self.u(n, *m) if self.with_pattern else torch.zeros(n, device=d)
        # per-window pattern phase offset at the patch centre. eps_range = pi makes it arbitrary, as in a movie (the
        # candidate pixel's pattern phase k . r is arbitrary); the default 0.3 rad (v1/v2 training) lets the head learn a
        # pattern-phase position cue that breaks on real movies (2026-11-11, see out/4pi/PROCESSING_STATUS.md)
        eps = self.u(n * 2, -eps_range, eps_range).reshape(n, 2) if self.with_pattern else torch.zeros(n, 2, device=d)
        doff = self.u(n * 8, -0.5, 0.5).reshape(n, 4, 2)        # fractional channel offsets (x, y)
        doff[:, 0] = 0.0
        Nt = Fm[:, None] * q * self.pattern(x, y, mm, eps)       # [n,T] photons per frame
        ang = torch.tensor(self.ang, device=d)
        mu = torch.zeros(n, T, 4, P, P, device=d)
        Tt = torch.zeros(n, T, 4, device=d); Ct = torch.zeros(n, T, 4, device=d)
        for c in range(4):
            pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
            g = self.render_ch(pc, Phi0, c)                      # [n,A,P,P]
            gt = g[:, ang]                                       # [n,T,P,P]
            w = torch.exp(-((self.ox[None] - pc[:, 0, None, None]) ** 2 + (self.oy[None] - pc[:, 1, None, None]) ** 2)
                          / (2 * self.wsd ** 2))
            mu[:, :, c] = Nt[:, :, None, None] * gt
            Tt[:, :, c] = Nt * gt.sum((-1, -2))
            Ct[:, :, c] = Nt * (gt * w[:, None]).sum((-1, -2))
        mu0 = mu.clone()                                         # main emitter only (targets of the window read-outs)
        # neighbours
        nnb = (self.u(n) < p_nb).long() * torch.randint(1, nb_max + 1, (n,), device=d, generator=self.g)
        nbinfo = []
        for j in range(nb_max):
            sel = nnb > j
            if not sel.any():
                break
            k = int(sel.sum())
            r = self.lu(k, 0.6, 7.0); th = self.u(k, 0, 2 * math.pi)
            xn = x[sel] + r * torch.cos(th); yn = y[sel] + r * torch.sin(th)
            zn = (z[sel] + 0.15 * torch.randn(k, device=d, generator=self.g)).clamp(*zr)
            Fn = self.lu(k, *F); qn = self.kinetics(k, p_on)
            Ntn = Fn[:, None] * qn * self.pattern(xn, yn, mm[sel], eps[sel])
            Phin = Phi0[sel] + 0.05 * torch.randn(k, device=d, generator=self.g)
            for c in range(4):
                pc = torch.stack([xn + doff[sel, c, 0], yn + doff[sel, c, 1], zn], 1)
                g = self.render_ch(pc, Phin, c)[:, ang]
                mu[sel, :, c] += Ntn[:, :, None, None] * g
            nbinfo.append(dict(sel=sel, x=xn, y=yn, z=zn, N=Ntn))
        b = self.lu(n * 4, *bg).reshape(n, 1, 4, 1, 1)
        gx = self.u(n * 4, -0.03, 0.03).reshape(n, 1, 4, 1, 1); gy = self.u(n * 4, -0.03, 0.03).reshape(n, 1, 4, 1, 1)
        bgim = (b * (1 + gx * self.ox + gy * self.oy)).clamp(min=0.02).expand(n, T, 4, P, P)
        lam = mu + bgim
        img = torch.poisson(lam, generator=self.g) + rn * torch.randn(lam.shape, device=d, generator=self.g) \
            + offerr * torch.randn(n, 1, 4, P, P, device=d, generator=self.g)
        return dict(img=img, mu=mu, mu0=mu0, bg=bgim, x=x, y=y, z=z, Phi0=Phi0, Nt=Nt, q=q, F=Fm, m=mm, eps=eps, doff=doff,
                    T=Tt, C=Ct, nnb=nnb, nb=nbinfo)
