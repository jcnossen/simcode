"""Explicit 4Pi solve on the K read-outs per (frame, channel) of the 4Pi emitter network (no pixel fit): 4Pi-SIMFLUX
(T = 6 frames = 2 angles x 3 steps, the SIM pattern) or plain 4Pi (k = None: no pattern, one angle, delta = (0,), T
frames, e.g. T = 1).

Data per candidate: Y [n,T,4,K] (frame t, channel c, read-out window k; windows of simcode/emitter/fourpi.py) with a
K x K Cholesky factor per (t, c) Lb [n,T,4,K,K]; the network prior (dx, dy, z) +- sigma. Known: the SIM pattern per frame
(k_a, phase psi_t incl. drift and the candidate's absolute position, depth m_t), the 4Pi phase field value Phi0 at the
candidate (+ per-angle offset delta_a; ang[t] = the angle of frame t), the channel sub-pixel offsets doff [n,4,2], the
4Pi PSF spline table (LUT).
Model:
    E[Y_tck] = N_t sum_px w_k(px; theta) g_c(px; (x, y) + doff_c, z, Phi0 + delta_a(t))          (tables over (x, y, z))
Kinetic models:
    'on'   : N_t = F (1 + m_t cos(k_a r - psi_t))   (no pattern: N_t = F)               params x, y, z, F
    'kin'  : N_t = F q_t (1 + ...), q_t = overlap of frame t with [t_on, t_off]           params x, y, z, F, t_on, t_off
    'free' : N_t free per frame (z from the channel profile; x, y from the prior)       params x, y, z, N_1..T
  (T = 1: 'on' = 'free'; 'kin' is not defined)
fit_phi=True appends Phi (the 4Pi phase offset at the candidate) as a free parameter (start Phi0): the phase field is
measured from the population with z held by the prior. LM on the Gaussian NLL (whitened residuals) + the prior as
pseudo-observations, z starts around the prior (spacing Lambda / 2), best objective kept.
Window placement: the network places the windows at its prior p0 = (x, y, z), not at the unknown emitter position.
  * o['W'] [n,4,K,P,P] (the network's actual windows): exact model E[Y] = N sum_px W(px) g(px; theta) from the LUT;
  * o['p0'] [n,3] without W: first order in the displacement d = p0 - theta from the tables,
        E[Y] = N [A(theta) + (dA/dtheta - G(theta)) . d],   G = sum_px w(px; theta) dg/dtheta (PSF moves, window fixed);
  * neither: windows co-moving with theta (tables).
Background nuisance: with o['Wsum'] [n,4,K] (window sums) and o['sb'] [n,4] (prior sd, e/px) a residual background per
channel b_c, common to the T frames, is fitted: E[Y] += b_c Wsum[c, k], prior b_c ~ N(0, sb^2).
The pattern is a 1D wave per angle (k along x / y) or, with self.k2 = ((kx, ky), (kx, ky)), the full 2D wave vectors.
self.coh_scale scales the interference terms of the PSF model (real data: in-situ contrast).

  FourPiSolve(lut, wins, wsd, P, k, delta, dev, ang, zr)   .build_table(), .solve(o, Phi0, doff, m, psi, mode, ...)
      plain 4Pi: FourPiSolve(lut, wins, k=None, delta=(0.0,), ang=(0,) * T, zr=...); m, psi are then ignored (None)
  psi_from_eps(eps), kin_start / kin_starts (kinetics starts), best_of (lower objective of two solves)
"""
import math
import torch
from simcode.simulate.fourpi_patches import DELTA_A, K_PAT
from simcode.emitter.fourpi import window, WINDOWS_TC
from simcode.psf import fourpi_component

ANG = (0, 0, 0, 1, 1, 1)


def psi_from_eps(eps):
    """sim convention -> per-frame pattern phases [n,6]: psi_t = 2 pi s / 3 + eps_a"""
    return torch.stack([2 * math.pi * s / 3 + eps[:, a] for a in range(2) for s in range(3)], 1)


class FourPiSolve:
    def __init__(self, lut, wins=WINDOWS_TC, wsd=1.0, P=15, k=K_PAT, delta=DELTA_A, dev='cuda', ang=ANG, zr=(-0.6, 0.6)):
        """k: SIM wave numbers per angle (None: no pattern); delta: 4Pi phase offset per angle; ang: angle of every frame
        (T = len(ang)); zr: z range of the solve (um)"""
        self.lut, self.wsd, self.P, self.dev, self.delta = lut, wsd, P, dev, tuple(delta)
        self.k = tuple(k) if k is not None else None
        self.ang, self.T, self.zr = tuple(ang), len(ang), tuple(zr)
        if max(self.ang) >= len(self.delta): raise ValueError('FourPiSolve: an angle index of ang has no delta')
        o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
        self.oy, self.ox = torch.meshgrid(o, o, indexing='ij')
        self.wins = tuple(wins); self.K = len(self.wins)

    def build_table(self, xr=3.5, hx=0.1, zr=(-0.6, 0.6), hz=0.005, chunk=8192):
        """Tabulate the read-out sums [c, comp (inc, cr, ci), k] and their derivatives on an (x, y, z) grid"""
        dev = self.dev; K = self.K
        xs = torch.arange(-xr, xr + 1e-6, hx, device=dev); zs = torch.arange(zr[0], zr[1] + 1e-6, hz, device=dev)
        Z, Y, X = torch.meshgrid(zs, xs, xs, indexing='ij')
        pos = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
        vals = []
        for i in range(0, len(pos), chunk):
            p = pos[i:i + chunk]
            ex = self.ox[None] - p[:, 0, None, None]; ey = self.oy[None] - p[:, 1, None, None]
            row = []; grow = [[], [], []]
            for c in range(4):
                ev = [fourpi_component(self.lut, k, c, p, True) for k in range(3)]
                inc = ev[0][0].clamp(min=1e-7)
                ws = [window(k, par, inc, ex, ey, (ev[1][0], ev[2][0]), ev[0][1:]) for k, par in self.wins]
                for v, gx, gy, gz in ev:
                    row += [(w * v).sum((-1, -2)) for w in ws]
                    for j, gg in enumerate((gx, gy, gz)):
                        grow[j] += [(w * gg).sum((-1, -2)) for w in ws]
            vals.append(torch.stack(row + grow[0] + grow[1] + grow[2], 1))
        VG = torch.cat(vals).T.reshape(48 * K, len(zs), len(xs), len(xs))
        V = VG[:12 * K]
        dX = torch.gradient(V, spacing=hx, dim=3)[0]; dY = torch.gradient(V, spacing=hx, dim=2)[0]
        dZ = torch.gradient(V, spacing=hz, dim=1)[0]
        self.tab = torch.cat([V, dX, dY, dZ, VG[12 * K:]])[None].contiguous(); self.tab_lim = (xr, zr)   # [1, 84 K, z, y, x]

    def readout_k(self, x, y, z, Phi, doff):
        """A [n,A (angle),4,K,5]: value, d/dx, d/dy, d/dz, d/dPhi (window co-moving); G [n,A,4,K,3]: d/d(x, y, z) of the
        PSF under a fixed window"""
        K = self.K; nA = len(self.delta); out = [[] for _ in range(nA)]; outg = [[] for _ in range(nA)]
        W = (getattr(self, '_ctx', None) or {}).get('W')
        if W is not None:   # exact: the network's actual windows W [n,4,K,P,P] (fixed), PSF components from the LUT at theta
            for c in range(4):
                pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
                ev = [fourpi_component(self.lut, k, c, pc, True) for k in range(3)]    # 3 x (v, gx, gy, gz) [n,P,P]
                Kt = torch.stack([torch.stack([torch.einsum('nkuv,nuv->nk', W[:, c], e_) for e_ in ev_]) for ev_ in ev])  # [comp, 4, n, K]
                Kt = Kt.permute(2, 1, 0, 3)                                           # [n, 4 (val, dx, dy, dz), comp, K]
                kc = getattr(self, 'coh_scale', 1.0)       # interference-contrast scale of the PSF model (real data: in-situ value)
                if kc != 1.0:
                    Kt = torch.cat([Kt[:, :, :1], Kt[:, :, 1:] * kc], 2)
                for a in range(nA):
                    ph = Phi + self.delta[a]
                    cP, sP = torch.cos(ph)[:, None, None], torch.sin(ph)[:, None, None]
                    val = Kt[:, :, 0] + cP * Kt[:, :, 1] - sP * Kt[:, :, 2]
                    dp = -sP * Kt[:, :1, 1] - cP * Kt[:, :1, 2]
                    out[a].append(torch.cat([val, dp], 1).transpose(1, 2))
            return torch.stack([torch.stack(out[a], 1) for a in range(nA)], 1), None
        xr, zr = self.tab_lim
        for c in range(4):
            pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
            g = torch.stack([pc[:, 0] / xr, pc[:, 1] / xr, (pc[:, 2] - zr[0]) / (zr[1] - zr[0]) * 2 - 1], -1)
            o = torch.nn.functional.grid_sample(self.tab, g[None, :, None, None, :], mode='bilinear', align_corners=True,
                                                padding_mode='border')
            Kt = o[0, :, :, 0, 0].T.reshape(-1, 7, 4, 3, K)[:, :, c]                  # [n, (val, dx, dy, dz, gx, gy, gz), comp, K]
            for a in range(nA):
                ph = Phi + self.delta[a]
                cP, sP = torch.cos(ph)[:, None, None], torch.sin(ph)[:, None, None]
                val = Kt[:, :, 0] + cP * Kt[:, :, 1] - sP * Kt[:, :, 2]              # [n, 7, K]
                dp = -sP * Kt[:, :1, 1] - cP * Kt[:, :1, 2]
                out[a].append(torch.cat([val[:, :4], dp], 1).transpose(1, 2))        # [n, K, 5]
                outg[a].append(val[:, 4:].transpose(1, 2))                           # [n, K, 3]
        return (torch.stack([torch.stack(out[a], 1) for a in range(nA)], 1),
                torch.stack([torch.stack(outg[a], 1) for a in range(nA)], 1))

    def pattern(self, x, y, m, psi):
        """m [n,T] or [n], psi [n,T] -> factor, d/dx, d/dy [n,T]; 1D wave per angle (self.k) or the full 2D wave vectors
        (self.k2): the estimated pattern is rotated by a few mrad, which the 1D form puts into psi at the candidate pixel only.
        No pattern (self.k None): factor 1, derivatives 0 (m, psi ignored)"""
        k2 = getattr(self, 'k2', None)
        if self.k is None and k2 is None:
            one = torch.ones(len(x), self.T, device=x.device, dtype=x.dtype)
            return one, torch.zeros_like(one), torch.zeros_like(one)
        m = m if m.dim() == 2 else m[:, None].expand(-1, self.T)
        P = []; dPx = []; dPy = []
        for t in range(self.T):
            a = self.ang[t]
            if k2 is None:
                r = x if a == 0 else y
                arg = self.k[a] * r - psi[:, t]
                P.append(1 + m[:, t] * torch.cos(arg))
                d = -m[:, t] * self.k[a] * torch.sin(arg)
                dPx.append(d if a == 0 else torch.zeros_like(d)); dPy.append(d if a == 1 else torch.zeros_like(d))
            else:
                arg = k2[a][0] * x + k2[a][1] * y - psi[:, t]
                P.append(1 + m[:, t] * torch.cos(arg))
                d = -m[:, t] * torch.sin(arg)
                dPx.append(d * k2[a][0]); dPy.append(d * k2[a][1])
        return torch.stack(P, 1), torch.stack(dPx, 1), torch.stack(dPy, 1)

    def model(self, th, mode, Phi0, doff, m, psi, fit_phi=False):
        """-> pred [n,T,4,K] and J [n,T,4,K,p]; parameter order: x, y, z, kinetics..., [Phi], [b_0..b_3]"""
        ctx = getattr(self, '_ctx', None) or {}
        p0, Wsum = ctx.get('p0'), ctx.get('Wsum')
        nb = 4 if Wsum is not None else 0
        x, y, z = th[:, 0], th[:, 1], th[:, 2]
        Phi = th[:, th.shape[1] - 1 - nb] if fit_phi else Phi0
        T = self.T
        ang = torch.tensor(self.ang, device=th.device)
        A, G = self.readout_k(x, y, z, Phi, doff)
        A = A[:, ang]                                                               # [n,T,4,K,5]
        if G is not None and p0 is not None:
            G = G[:, ang]                                                           # [n,T,4,K,3]   # windows fixed at p0: first order in the displacement; Jacobian = PSF derivative under a fixed window
            d = (p0 - th[:, :3])[:, None, None, None, :]
            A0 = A[..., 0] + ((A[..., 1:4] - G) * d).sum(-1)
            A = torch.cat([A0[..., None], G, A[..., 4:5]], -1)
        n, p = th.shape
        nk = {'on': 1, 'kin': 3, 'free': T}[mode]
        if mode == 'kin':
            F, ton, toff = th[:, 3], th[:, 4], th[:, 5]
            t = torch.arange(T, device=th.device, dtype=th.dtype)[None]
            a1 = torch.minimum(t + 1, toff[:, None]); a0 = torch.maximum(t, ton[:, None])
            q_raw = a1 - a0; q = q_raw.clamp(0, 1)
            dq_off = ((toff[:, None] < t + 1) & (toff[:, None] > t) & (q_raw > 0)).float()
            dq_on = -((ton[:, None] > t) & (ton[:, None] < t + 1) & (q_raw > 0)).float()
            Pt, dPx, dPy = self.pattern(x, y, m, psi)
            Nt = F[:, None] * q * Pt
            dN = torch.stack([F[:, None] * q * dPx, F[:, None] * q * dPy, torch.zeros_like(Pt), q * Pt,
                              F[:, None] * Pt * dq_on, F[:, None] * Pt * dq_off], -1)
        elif mode == 'on':
            F = th[:, 3]
            Pt, dPx, dPy = self.pattern(x, y, m, psi)
            Nt = F[:, None] * Pt
            dN = torch.stack([F[:, None] * dPx, F[:, None] * dPy, torch.zeros_like(Pt), Pt], -1)
        else:
            Nt = th[:, 3:3 + T]
            dN = torch.cat([torch.zeros(n, T, 3, device=th.device), torch.eye(T, device=th.device)[None].expand(n, T, T)], -1)
        npad = (1 if fit_phi else 0) + nb
        if npad:
            dN = torch.cat([dN, torch.zeros(n, T, npad, device=th.device)], -1)
        pred = Nt[:, :, None, None] * A[..., 0]
        geo = [A[..., 1:4], torch.zeros(*A.shape[:-1], nk, device=th.device)]
        if fit_phi:
            geo.append(A[..., 4:5])
        if nb:
            geo.append(torch.zeros(*A.shape[:-1], nb, device=th.device))
        J = Nt[:, :, None, None, None] * torch.cat(geo, -1) + dN[:, :, None, None, :] * A[..., 0:1]
        if nb:   # residual background per channel, common to the frames
            b = th[:, p - 4:]
            pred = pred + (b[:, :, None] * Wsum)[:, None]
            for c in range(4):
                J[:, :, c, :, p - 4 + c] = Wsum[:, None, c, :]
        return pred, J

    def solve(self, o, Phi0, doff, m, psi, mode='on', iters=20, Lam=0.257, zstarts=(-0.5, 0.0, 0.5), prior_w=1.0,
              fit_phi=False, phistarts=None, crel=0.0, kin0=None):
        """o: Y [n,T,4,K], Lb [n,T,4,K,K] (Cholesky blocks), dxy, z, ls; optional p0 [n,3], W, Wsum [n,4,K] + sb [n,4],
        z0 / xy0 (start values; the prior stays o['z'] / o['dxy']). Returns dict(th, obj, chi2, sig)"""
        dev = o['Y'].device; n = o['Y'].shape[0]; T = self.T; z_lo, z_hi = self.zr
        if mode == 'kin' and T < 2: raise ValueError("FourPiSolve: mode 'kin' needs T >= 2 frames")
        Y = o['Y']; Lb = o['Lb']
        self._ctx = dict(p0=o.get('p0'), Wsum=o.get('Wsum'), W=o.get('W'))
        nb = 4 if o.get('Wsum') is not None else 0
        sp = torch.exp(o['ls'])
        mu_p = torch.stack([o['dxy'][:, 0], o['dxy'][:, 1], o['z']], 1)

        def resid(th):
            pred, J = self.model(th, mode, Phi0, doff, m, psi, fit_phi)
            r = Y - pred
            ru = torch.linalg.solve_triangular(Lb, r[..., None], upper=False)[..., 0].reshape(n, -1)
            Ju = torch.linalg.solve_triangular(Lb, J, upper=False).reshape(n, -1, th.shape[1])
            rp = (mu_p - th[:, :3]) / sp * prior_w
            Jp = torch.zeros(n, 3 + nb, th.shape[1], device=dev)
            for i in range(3): Jp[:, i, i] = prior_w / sp[:, i]
            if nb:
                pb = th.shape[1] - 4
                rp = torch.cat([rp, -th[:, pb:] / o['sb']], 1)
                for c in range(4): Jp[:, 3 + c, pb + c] = 1.0 / o['sb'][:, c]
            return torch.cat([ru, rp], 1), torch.cat([Ju, Jp], 1), (ru ** 2).sum(1)

        Fi = Y[..., 0].clamp(min=0).sum(2)        # read-out 0 is the photon-normalised one
        best = None
        starts = [(js, 0.0) for js in zstarts] if not fit_phi else \
            [(0.0, ph) for ph in (phistarts if phistarts is not None else (0.0, 2.1, 4.2))]
        for js, ph in starts:
            z0 = (o.get('z0', o['z']) + js * Lam).clamp(z_lo + 0.02, z_hi - 0.02)
            if mode == 'kin':
                th = torch.cat([o.get('xy0', o['dxy']), z0[:, None], kin0], 1)
            elif mode == 'on':
                th = torch.stack([o['dxy'][:, 0], o['dxy'][:, 1], z0, (Fi.mean(1) / 0.85).clamp(min=5)], 1)
            else:
                th = torch.cat([o.get('xy0', o['dxy']), z0[:, None], Fi / 0.85], 1)
            if fit_phi:
                th = torch.cat([th, (Phi0 + ph)[:, None]], 1)
            if nb:
                th = torch.cat([th, torch.zeros(n, 4, device=dev)], 1)
            lam = torch.full((n,), 1e-2, device=dev)
            r, J, _ = resid(th); obj = (r ** 2).sum(1)
            eye = torch.eye(th.shape[1], device=dev)
            for _ in range(iters):
                g = torch.einsum('nmp,nm->np', J, r)
                H = torch.einsum('nmp,nmq->npq', J, J)
                Hd = H + lam[:, None, None] * (torch.diagonal(H, dim1=1, dim2=2)[:, :, None] * eye + 1e-6 * eye)
                st = torch.linalg.solve(Hd, g[..., None])[..., 0]
                tn = th + st
                tn[:, 2] = tn[:, 2].clamp(z_lo, z_hi); tn[:, :2] = tn[:, :2].clamp(-3, 3)
                if mode == 'kin':
                    tn[:, 3] = tn[:, 3].clamp(min=1.0); tn[:, 4] = tn[:, 4].clamp(-1.0, T - 0.1)
                    tn[:, 5] = torch.maximum(tn[:, 5].clamp(0.1, T + 1.0), tn[:, 4] + 0.1)
                rn, Jn, _ = resid(tn); on = (rn ** 2).sum(1)
                ok = on < obj
                th = torch.where(ok[:, None], tn, th); r = torch.where(ok[:, None], rn, r)
                J = torch.where(ok[:, None, None], Jn, J); obj = torch.where(ok, on, obj)
                lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-6, 1e6)
            if best is None:
                best = [th, obj]
            else:
                bb = obj < best[1]
                best = [torch.where(bb[:, None], th, best[0]), torch.where(bb, obj, best[1])]
        th, obj = best
        _, J, chi2 = resid(th)
        H = torch.einsum('nmp,nmq->npq', J, J)
        cov = torch.linalg.pinv(H)
        return dict(th=th, obj=obj, chi2=chi2, sig=torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0)))


def kin_start(rf, m, psi, sv):
    """t_on / t_off / F starts from a 'free' solve: per-frame photons divided by the pattern factor at the solved
    position; frames above 40 % of the maximum are 'on'."""
    th = rf['th']; x, y = th[:, 0], th[:, 1]; T = sv.T
    Pt, _, _ = sv.pattern(x, y, m, psi)
    q = (th[:, 3:3 + T] / Pt.clamp(min=0.05)).clamp(min=0)
    F = q.max(1).values.clamp(min=1.0)
    on = q > 0.4 * F[:, None]
    t = torch.arange(T, device=th.device, dtype=th.dtype)[None].expand_as(q)
    ton = torch.where(on, t, torch.full_like(t, 9)).min(1).values
    toff = torch.where(on, t + 1, torch.full_like(t, -9)).max(1).values
    ton = torch.where(ton <= 0, torch.full_like(ton, -0.5), ton)
    toff = torch.where(toff >= T, torch.full_like(toff, T + 0.5), toff)
    Fm = (q * on).sum(1) / on.sum(1).clamp(min=1)
    return torch.stack([Fm.clamp(min=1.0), ton, toff], 1)


def kin_starts(rf, m, psi, sv):
    """two kinetics starts: the thresholded one (kin_start) and on-through (t_on -0.5, t_off 6.5)"""
    k1 = kin_start(rf, m, psi, sv); T = sv.T
    k2 = k1.clone(); k2[:, 1] = -0.5; k2[:, 2] = T + 0.5
    Pt, _, _ = sv.pattern(rf['th'][:, 0], rf['th'][:, 1], m, psi)
    k2[:, 0] = (rf['th'][:, 3:3 + T].sum(1) / Pt.sum(1).clamp(min=0.1)).clamp(min=1.0)
    return k1, k2


def best_of(r1, r2):
    b = r2['obj'] < r1['obj']
    return {k: torch.where(b.reshape(-1, *([1] * (v.dim() - 1))), r2[k], v) for k, v in r1.items()}
