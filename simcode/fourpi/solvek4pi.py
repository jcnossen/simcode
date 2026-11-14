"""Explicit 4Pi-SIMFLUX solve on K fixed-window read-outs per (frame, channel) (simcode/fourpi/windows4pi.py).

Same model, kinetics modes, starts, prior handling and LM as Solve4Pi (solve4pi.py, unchanged); differences:
  * K read-outs with canonical windows instead of (T = patch sum, C = Gaussian window): tables over (x, y, z) per channel,
    component and read-out;
  * the data are o['Y'] [n,6,4,K] with a K x K Cholesky factor per (frame, channel) o['Lb'] [n,6,4,K,K];
  * window placement: the head places the windows at its prior p0 = (x, y, z), not at the unknown emitter position. With
    o['p0'] [n,3] the model is first order in the displacement d = p0 - theta:
        E[Y] = N [A(theta) + (dA/dtheta - G(theta)) . d],   G = sum_px w(px; theta) dg/dtheta  (PSF moves, window fixed),
    and the Jacobian is N G (the window stays where the head put it while theta moves);
    With o['W'] [n,4,K,P,P] (the head's actual windows) the model is exact instead: E[Y] = N sum_px W(px) g(px; theta)
    from the LUT (slower than the tables; needed for windows that change fast with the placement, e.g. 'cr' / 'ci' in z);
  * background nuisance (the main solve's B): with o['Wsum'] [n,4,K] (window sums) and o['sb'] [n,4] (prior sd, e/px) a
    residual background per channel b_c, common to the 6 frames, is fitted: E[Y] += b_c Wsum[c, k], prior b_c ~ N(0, sb^2).
Network intensities -> explicit solve; no pixel fit.
"""
import torch
from simcode.fourpi.solve4pi import Solve4Pi, ANG
from simcode.fourpi.windows4pi import window, DEFAULT


class SolveK(Solve4Pi):
    def __init__(self, lut, wins=DEFAULT, **kw):
        super().__init__(lut, **kw)
        self.wins = tuple(wins); self.K = len(self.wins)

    def build_table(self, xr=3.5, hx=0.1, zr=(-0.6, 0.6), hz=0.005, chunk=8192):
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
                ev = [comp[c].render(p, True) for comp in self.lut.luts]
                inc = ev[0][0].clamp(min=1e-7)
                ws = [window(k, par, inc, ex, ey, (ev[1][0], ev[2][0])) for k, par in self.wins]
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
        """A [n,2(angle),4,K,5]: value, d/dx, d/dy, d/dz, d/dPhi (window co-moving); G [n,2,4,K,3]: d/d(x, y, z) of the
        PSF under a fixed window"""
        K = self.K; out = [[], []]; outg = [[], []]
        W = (getattr(self, '_ctx', None) or {}).get('W')
        if W is not None:   # exact: the head's actual windows W [n,4,K,P,P] (fixed), PSF components from the LUT at theta
            for c in range(4):
                pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
                ev = [comp[c].render(pc, True) for comp in self.lut.luts]          # 3 x (v, gx, gy, gz) [n,P,P]
                Kt = torch.stack([torch.stack([torch.einsum('nkuv,nuv->nk', W[:, c], e_) for e_ in ev_]) for ev_ in ev])  # [comp, 4, n, K]
                Kt = Kt.permute(2, 1, 0, 3)                                           # [n, 4 (val, dx, dy, dz), comp, K]
                kc = getattr(self, 'coh_scale', 1.0)       # interference-contrast scale of the PSF model (real data: in-situ value)
                if kc != 1.0:
                    Kt = torch.cat([Kt[:, :, :1], Kt[:, :, 1:] * kc], 2)
                for a in range(2):
                    ph = Phi + self.delta[a]
                    cP, sP = torch.cos(ph)[:, None, None], torch.sin(ph)[:, None, None]
                    val = Kt[:, :, 0] + cP * Kt[:, :, 1] - sP * Kt[:, :, 2]
                    dp = -sP * Kt[:, :1, 1] - cP * Kt[:, :1, 2]
                    out[a].append(torch.cat([val, dp], 1).transpose(1, 2))
            return torch.stack([torch.stack(out[a], 1) for a in range(2)], 1), None
        xr, zr = self.tab_lim
        for c in range(4):
            pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
            g = torch.stack([pc[:, 0] / xr, pc[:, 1] / xr, (pc[:, 2] - zr[0]) / (zr[1] - zr[0]) * 2 - 1], -1)
            o = torch.nn.functional.grid_sample(self.tab, g[None, :, None, None, :], mode='bilinear', align_corners=True,
                                                padding_mode='border')
            Kt = o[0, :, :, 0, 0].T.reshape(-1, 7, 4, 3, K)[:, :, c]                  # [n, (val, dx, dy, dz, gx, gy, gz), comp, K]
            for a in range(2):
                ph = Phi + self.delta[a]
                cP, sP = torch.cos(ph)[:, None, None], torch.sin(ph)[:, None, None]
                val = Kt[:, :, 0] + cP * Kt[:, :, 1] - sP * Kt[:, :, 2]              # [n, 7, K]
                dp = -sP * Kt[:, :1, 1] - cP * Kt[:, :1, 2]
                out[a].append(torch.cat([val[:, :4], dp], 1).transpose(1, 2))        # [n, K, 5]
                outg[a].append(val[:, 4:].transpose(1, 2))                           # [n, K, 3]
        return (torch.stack([torch.stack(out[a], 1) for a in range(2)], 1),
                torch.stack([torch.stack(outg[a], 1) for a in range(2)], 1))

    def pattern(self, x, y, m, psi):
        """as Solve4Pi.pattern, with an optional full 2D wave vector per angle (self.k2 = ((kx, ky), (kx, ky))): the
        estimated pattern is rotated by a few mrad, which the 1D form puts into psi at the candidate pixel only"""
        k2 = getattr(self, 'k2', None)
        if k2 is None:
            return super().pattern(x, y, m, psi)
        m = m if m.dim() == 2 else m[:, None].expand(-1, 6)
        P = []; dPx = []; dPy = []
        for t in range(6):
            a = ANG[t]
            arg = k2[a][0] * x + k2[a][1] * y - psi[:, t]
            P.append(1 + m[:, t] * torch.cos(arg))
            d = -m[:, t] * torch.sin(arg)
            dPx.append(d * k2[a][0]); dPy.append(d * k2[a][1])
        return torch.stack(P, 1), torch.stack(dPx, 1), torch.stack(dPy, 1)

    def model(self, th, mode, Phi0, doff, m, psi, fit_phi=False):
        """-> pred [n,6,4,K] and J [n,6,4,K,p]; parameter order: x, y, z, kinetics..., [Phi], [b_0..b_3]"""
        ctx = getattr(self, '_ctx', None) or {}
        p0, Wsum = ctx.get('p0'), ctx.get('Wsum')
        nb = 4 if Wsum is not None else 0
        x, y, z = th[:, 0], th[:, 1], th[:, 2]
        Phi = th[:, th.shape[1] - 1 - nb] if fit_phi else Phi0
        ang = torch.tensor(ANG, device=th.device)
        A, G = self.readout_k(x, y, z, Phi, doff)
        A = A[:, ang]                                                               # [n,6,4,K,5]
        if G is not None and p0 is not None:
            G = G[:, ang]                                                           # [n,6,4,K,3]   # windows fixed at p0: first order in the displacement; Jacobian = PSF derivative under a fixed window
            d = (p0 - th[:, :3])[:, None, None, None, :]
            A0 = A[..., 0] + ((A[..., 1:4] - G) * d).sum(-1)
            A = torch.cat([A0[..., None], G, A[..., 4:5]], -1)
        n, p = th.shape
        nk = {'on': 1, 'kin': 3, 'free': 6}[mode]
        if mode == 'kin':
            F, ton, toff = th[:, 3], th[:, 4], th[:, 5]
            t = torch.arange(6, device=th.device, dtype=th.dtype)[None]
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
            Nt = th[:, 3:9]
            dN = torch.cat([torch.zeros(n, 6, 3, device=th.device), torch.eye(6, device=th.device)[None].expand(n, 6, 6)], -1)
        npad = (1 if fit_phi else 0) + nb
        if npad:
            dN = torch.cat([dN, torch.zeros(n, 6, npad, device=th.device)], -1)
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
        """o: Y [n,6,4,K], Lb [n,6,4,K,K] (Cholesky blocks), dxy, z, ls; optional p0 [n,3], Wsum [n,4,K] + sb [n,4].
        Returns dict(th, obj, chi2, sig)"""
        dev = o['Y'].device; n = o['Y'].shape[0]
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
            z0 = (o.get('z0', o['z']) + js * Lam).clamp(-0.58, 0.58)       # 'z0': optional start value (the prior stays o['z'])
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
                tn[:, 2] = tn[:, 2].clamp(-0.6, 0.6); tn[:, :2] = tn[:, :2].clamp(-3, 3)
                if mode == 'kin':
                    tn[:, 3] = tn[:, 3].clamp(min=1.0); tn[:, 4] = tn[:, 4].clamp(-1.0, 5.9)
                    tn[:, 5] = torch.maximum(tn[:, 5].clamp(0.1, 7.0), tn[:, 4] + 0.1)
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
