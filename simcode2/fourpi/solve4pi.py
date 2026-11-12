"""Explicit 4Pi-SIMFLUX solve on the head's network intensities (no pixel fit).

Data per candidate: T_tc, C_tc (t = 6 frames, c = 4 channels) with a 2 x 2 Cholesky factor per (t, c); the network
prior (dx, dy, z) +- sigma. Known: SIM pattern per frame (k_a, phase psi_t incl. drift and the candidate's absolute
position, depth m_t), the 4Pi phase field value Phi0 at the candidate (+ per-angle offset delta_a), channel sub-pixel
offsets doff [n,4,2], the 4Pi PSF LUT.
Model:
    E[T_tc] = N_t A_T(c, a(t); x, y, z, Phi),   E[C_tc] = N_t A_C(c, a(t); x, y, z, Phi)
    A_T = sum_q g_c(q; (x, y) + doff_c, z, Phi + delta_a),  A_C = sum_q w(q; (x, y) + doff_c) g_c(q; ...),
    w = Gaussian window (sd wsd) at the position (the read-out-2 definition used in training).
Kinetic models:
    'on'   : N_t = F (1 + m_t cos(k_a r - psi_t))                            params x, y, z, F
    'kin'  : N_t = F q_t (1 + ...), q_t = overlap of frame t with [t_on, t_off]   params x, y, z, F, t_on, t_off
    'free' : N_t free per frame (z from the channel profile; x, y from the prior / read-out shape)
                                                                             params x, y, z, N_1..6
fit_phi=True appends Phi (the 4Pi phase offset at the candidate) as a free parameter (start Phi0): used to measure the
phase field from the population with z held by the prior.
LM on the Gaussian NLL (whitened residuals) + the prior as pseudo-observations, z starts around the prior (spacing
Lambda / 2), best objective kept. Jacobians analytic from the LUT derivatives.
"""
import math
import torch
from simcode2.fourpi.sim4pi import DELTA_A, K_PAT

ANG = (0, 0, 0, 1, 1, 1)


def psi_from_eps(eps):
    """sim convention -> per-frame pattern phases [n,6]: psi_t = 2 pi s / 3 + eps_a"""
    return torch.stack([2 * math.pi * s / 3 + eps[:, a] for a in range(2) for s in range(3)], 1)


class Solve4Pi:
    def __init__(self, lut, wsd=1.0, P=15, k=K_PAT, delta=DELTA_A, dev='cuda'):
        self.lut, self.wsd, self.P, self.dev, self.k, self.delta = lut, wsd, P, dev, tuple(k), tuple(delta)
        o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
        self.oy, self.ox = torch.meshgrid(o, o, indexing='ij')

    def build_table(self, xr=3.5, hx=0.1, zr=(-0.6, 0.6), hz=0.005, chunk=8192):
        """Tabulate the read-out sums K[c, comp (inc, cr, ci), ro (T, C)] on an (x, y, z) grid; readout() then uses
        trilinear interpolation (values) and central-difference tables (derivatives) instead of LUT images."""
        dev = self.dev
        xs = torch.arange(-xr, xr + 1e-6, hx, device=dev); zs = torch.arange(zr[0], zr[1] + 1e-6, hz, device=dev)
        Z, Y, X = torch.meshgrid(zs, xs, xs, indexing='ij')
        pos = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
        s = lambda t: t.sum((-1, -2))
        vals = []
        for i in range(0, len(pos), chunk):
            p = pos[i:i + chunk]
            ex = self.ox[None] - p[:, 0, None, None]; ey = self.oy[None] - p[:, 1, None, None]
            w = torch.exp(-(ex ** 2 + ey ** 2) / (2 * self.wsd ** 2))
            row = []
            for c in range(4):
                for comp in self.lut.luts:
                    v = comp[c].evaluate(p)
                    row += [s(v), s(w * v)]
            vals.append(torch.stack(row, 1))
        V = torch.cat(vals).T.reshape(24, len(zs), len(xs), len(xs))                 # [24, z, y, x]
        dX = torch.gradient(V, spacing=hx, dim=3)[0]; dY = torch.gradient(V, spacing=hx, dim=2)[0]
        dZ = torch.gradient(V, spacing=hz, dim=1)[0]
        self.tab = torch.cat([V, dX, dY, dZ])[None].contiguous()                     # [1, 96, z, y, x]
        self.tab_lim = (xr, zr)

    def _tab_eval(self, pc):
        xr, zr = self.tab_lim
        g = torch.stack([pc[:, 0] / xr, pc[:, 1] / xr, (pc[:, 2] - zr[0]) / (zr[1] - zr[0]) * 2 - 1], -1)
        o = torch.nn.functional.grid_sample(self.tab, g[None, :, None, None, :], mode='bilinear', align_corners=True,
                                            padding_mode='border')
        return o[0, :, :, 0, 0].T.reshape(-1, 4, 24)                                  # [n, (val, dx, dy, dz), 24]

    def readout(self, x, y, z, Phi, doff):
        """A_T, A_C [n,2(angle),4,5]: value, d/dx, d/dy, d/dz, d/dPhi"""
        if getattr(self, 'tab', None) is not None:
            rT = [[], []]; rC = [[], []]
            for c in range(4):
                pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
                K = self._tab_eval(pc).reshape(-1, 4, 4, 3, 2)[:, :, c]              # [n, deriv, comp, ro]
                for a in range(2):
                    ph = Phi + self.delta[a]
                    cP, sP = torch.cos(ph)[:, None], torch.sin(ph)[:, None]
                    for ro, out in ((0, rT[a]), (1, rC[a])):
                        val = K[:, :, 0, ro] + cP * K[:, :, 1, ro] - sP * K[:, :, 2, ro]
                        dp = -sP * K[:, :1, 1, ro] - cP * K[:, :1, 2, ro]
                        out.append(torch.cat([val, dp], -1))
            return (torch.stack([torch.stack(rT[a], 1) for a in range(2)], 1),
                    torch.stack([torch.stack(rC[a], 1) for a in range(2)], 1))
        # the read-outs are linear in the LUT components: reduce each component (inc, cr, ci) x (value, dx, dy, dz)
        # to its two read-out sums once per channel, then combine for both angles (only the phase differs)
        s = lambda t: t.sum((-1, -2))
        rT = [[], []]; rC = [[], []]
        for c in range(4):
            pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
            ex = self.ox[None] - pc[:, 0, None, None]; ey = self.oy[None] - pc[:, 1, None, None]
            w = torch.exp(-(ex ** 2 + ey ** 2) / (2 * self.wsd ** 2))
            wx = w * ex / self.wsd ** 2; wy = w * ey / self.wsd ** 2
            KT, KC = [], []
            for row in self.lut.luts:
                v, gx, gy, gz = row[c].evaluate(pc, True)
                KT.append(torch.stack([s(v), s(gx), s(gy), s(gz)], -1))
                KC.append(torch.stack([s(w * v), s(w * gx + wx * v), s(w * gy + wy * v), s(w * gz)], -1))
            for a in range(2):
                ph = Phi + self.delta[a]
                cP, sP = torch.cos(ph)[:, None], torch.sin(ph)[:, None]
                for K, out in ((KT, rT[a]), (KC, rC[a])):
                    val = K[0] + cP * K[1] - sP * K[2]                             # [n,4]
                    dp = -sP * K[1][:, :1] - cP * K[2][:, :1]
                    out.append(torch.cat([val, dp], -1))
        AT = torch.stack([torch.stack(rT[a], 1) for a in range(2)], 1)
        AC = torch.stack([torch.stack(rC[a], 1) for a in range(2)], 1)
        return AT, AC

    def pattern(self, x, y, m, psi):
        """m [n,6] or [n], psi [n,6] -> factor, d/dx, d/dy [n,6]"""
        m = m if m.dim() == 2 else m[:, None].expand(-1, 6)
        P = []; dPx = []; dPy = []
        for t in range(6):
            a = ANG[t]; r = x if a == 0 else y
            arg = self.k[a] * r - psi[:, t]
            P.append(1 + m[:, t] * torch.cos(arg))
            d = -m[:, t] * self.k[a] * torch.sin(arg)
            dPx.append(d if a == 0 else torch.zeros_like(d)); dPy.append(d if a == 1 else torch.zeros_like(d))
        return torch.stack(P, 1), torch.stack(dPx, 1), torch.stack(dPy, 1)

    def model(self, th, mode, Phi0, doff, m, psi, fit_phi=False):
        """-> pred [n,6,4,2] and J [n,6,4,2,p]"""
        x, y, z = th[:, 0], th[:, 1], th[:, 2]
        Phi = th[:, -1] if fit_phi else Phi0
        AT, AC = self.readout(x, y, z, Phi, doff)
        ang = torch.tensor(ANG, device=th.device)
        A = torch.stack([AT, AC], -2)[:, ang]                                       # [n,6,4,2,5]
        n, p = th.shape
        nk = {'on': 1, 'kin': 3, 'free': 6}[mode]
        if mode == 'kin':   # N_t = F q_t P_t, q_t = overlap of frame t with [t_on, t_off] (SIMCODE v2 kinetics)
            F, ton, toff = th[:, 3], th[:, 4], th[:, 5]
            t = torch.arange(6, device=th.device, dtype=th.dtype)[None]
            a1 = torch.minimum(t + 1, toff[:, None]); a0 = torch.maximum(t, ton[:, None])
            q_raw = a1 - a0; inner = (q_raw > 0) & (q_raw < 1)
            q = q_raw.clamp(0, 1)
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
            dN = torch.stack([F[:, None] * dPx, F[:, None] * dPy, torch.zeros_like(Pt), Pt], -1)   # [n,6,4]
        else:
            Nt = th[:, 3:9]
            dN = torch.cat([torch.zeros(n, 6, 3, device=th.device), torch.eye(6, device=th.device)[None].expand(n, 6, 6)], -1)
        if fit_phi:
            dN = torch.cat([dN, torch.zeros(n, 6, 1, device=th.device)], -1)
        pred = Nt[:, :, None, None] * A[..., 0]
        geo = [A[..., 1:4], torch.zeros(*A.shape[:-1], nk, device=th.device)]
        if fit_phi:
            geo.append(A[..., 4:5])
        J = Nt[:, :, None, None, None] * torch.cat(geo, -1) + dN[:, :, None, None, :] * A[..., 0:1]
        return pred, J

    def solve(self, o, Phi0, doff, m, psi, mode='on', iters=20, Lam=0.246, zstarts=(-0.5, 0.0, 0.5), prior_w=1.0,
              fit_phi=False, phistarts=None, crel=0.0, kin0=None):
        """o: head outputs (T, C, L, dxy, z, ls). Returns dict(th, obj, chi2, sig)"""
        dev = o['T'].device; n = o['T'].shape[0]
        Y = torch.stack([o['T'], o['C']], -1)
        L = o['L']
        a = L[..., 0, 0]; b = L[..., 1, 0]; d = L[..., 1, 1]
        if crel > 0:   # relative nuisance error (read-out / model mismatch) added in quadrature to the diagonal
            a = torch.sqrt(a ** 2 + (crel * o['T'].clamp(min=0)) ** 2)
            d = torch.sqrt(d ** 2 + (crel * o['C'].clamp(min=0)) ** 2)
        sp = torch.exp(o['ls'])
        mu_p = torch.stack([o['dxy'][:, 0], o['dxy'][:, 1], o['z']], 1)

        Lfull = o.get('Lfull')   # (2026-11-12) optional explicit Cholesky factor [n,48,48] of the full read-out covariance

        def resid(th):
            pred, J = self.model(th, mode, Phi0, doff, m, psi, fit_phi)
            r = Y - pred
            if Lfull is not None:
                ru = torch.linalg.solve_triangular(Lfull, r.reshape(n, 48, 1), upper=False)[..., 0]
                Ju = torch.linalg.solve_triangular(Lfull, J.reshape(n, 48, th.shape[1]), upper=False)
            else:
                u1 = r[..., 0] / a; u2 = (r[..., 1] - b * u1) / d
                J1 = J[..., 0, :] / a[..., None]; J2 = (J[..., 1, :] - b[..., None] * J1) / d[..., None]
                ru = torch.stack([u1, u2], -1).reshape(n, -1)
                Ju = torch.stack([J1, J2], -2).reshape(n, -1, th.shape[1])
            rp = (mu_p - th[:, :3]) / sp * prior_w
            Jp = torch.zeros(n, 3, th.shape[1], device=dev)
            for i in range(3): Jp[:, i, i] = prior_w / sp[:, i]
            return torch.cat([ru, rp], 1), torch.cat([Ju, Jp], 1), (ru ** 2).sum(1)

        Fi = o['T'].clamp(min=0).sum(2)
        best = None
        starts = [(js, 0.0) for js in zstarts] if not fit_phi else \
            [(0.0, ph) for ph in (phistarts if phistarts is not None else (0.0, 2.1, 4.2))]
        for js, ph in starts:
            z0 = (o['z'] + js * Lam).clamp(-0.58, 0.58)
            if mode == 'kin':   # kin0 [n,3]: F, t_on, t_off starts
                th = torch.cat([o['dxy'], z0[:, None], kin0], 1)
            elif mode == 'on':
                th = torch.stack([o['dxy'][:, 0], o['dxy'][:, 1], z0, (Fi.mean(1) / 0.85).clamp(min=5)], 1)
            else:
                th = torch.cat([o['dxy'], z0[:, None], Fi / 0.85], 1)
            if fit_phi:
                th = torch.cat([th, (Phi0 + ph)[:, None]], 1)
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


def kin_start(rf, m, psi, sv):
    """t_on / t_off / F starts from a 'free' solve: per-frame photons divided by the pattern factor at the solved
    position; frames above 40 % of the maximum are 'on'."""
    th = rf['th']; x, y = th[:, 0], th[:, 1]
    Pt, _, _ = sv.pattern(x, y, m, psi)
    q = (th[:, 3:9] / Pt.clamp(min=0.05)).clamp(min=0)
    F = q.max(1).values.clamp(min=1.0)
    on = q > 0.4 * F[:, None]
    t = torch.arange(6, device=th.device, dtype=th.dtype)[None].expand_as(q)
    ton = torch.where(on, t, torch.full_like(t, 9)).min(1).values
    toff = torch.where(on, t + 1, torch.full_like(t, -9)).max(1).values
    ton = torch.where(ton <= 0, torch.full_like(ton, -0.5), ton)
    toff = torch.where(toff >= 6, torch.full_like(toff, 6.5), toff)
    Fm = (q * on).sum(1) / on.sum(1).clamp(min=1)
    return torch.stack([Fm.clamp(min=1.0), ton, toff], 1)


def kin_starts(rf, m, psi, sv):
    """two kinetics starts: the thresholded one (kin_start) and on-through (t_on -0.5, t_off 6.5)"""
    k1 = kin_start(rf, m, psi, sv)
    k2 = k1.clone(); k2[:, 1] = -0.5; k2[:, 2] = 6.5
    Pt, _, _ = sv.pattern(rf['th'][:, 0], rf['th'][:, 1], m, psi)
    k2[:, 0] = (rf['th'][:, 3:9].sum(1) / Pt.sum(1).clamp(min=0.1)).clamp(min=1.0)
    return k1, k2


def best_of(r1, r2):
    b = r2['obj'] < r1['obj']
    return {k: torch.where(b.reshape(-1, *([1] * (v.dim() - 1))), r2[k], v) for k, v in r1.items()}
