"""
Evaluation-only REFERENCE (user request 2026-11-09): a basic single-emitter Levenberg-Marquardt pixel MLE. NOT part of the
method (the method stays network-only); used only as a reference line in the CRLB plots (analysis/figures/crlb_all_v2.py).
Model on a P x P ROI (default 11 x 11) centred on a pixel (cx, cy), all T frames:
    mu_t(px) = I_t * psf(px; x, y, z0) + bg_t
  psf = the calibrated vectorial PSF (calib_0.5nM, simcode.psf.spline on P x P; simulator normalisation, so I_t = photons),
  z FIXED at z0 = 0.05 um (the emitter median height, = the head's PSF template PSFCol) or free (zfree=True, clamped to
  [0, zmax], start z0); parameters theta = (x, y, [z,] I_1..I_T, bg_1..bg_T) (14 / 15), no pattern, no neighbours (single emitter: neighbours contaminate the fit by design).
Likelihood: sCMOS Poisson (Huang et al. 2013): y + rn^2 ~ Poisson(mu + rn^2) per pixel, rn = the per-pixel read-noise
map; LM with Fisher scoring (F = J^T diag(1 / (mu + rn^2)) J), multiplicative damping on diag(F), accept / reject.
Returns the estimate and the inverse Fisher at the estimate (I-block -> covariance for the explicit pattern solve).
"""
import torch
from simcode.psf import spline                 # = the development hd/solve.make_lut
from simcode.layers.blocks import extract_patches


class LMRef:
    def __init__(self, sim, P=11, z0=0.05, zfree=False, bg='frame'):
        self.lut = spline(sim.psf, sim.zmax, roisize=P); self.P = P; self.z0 = z0; self.zfree = zfree; self.zmax = sim.zmax
        self.k = 3 if zfree else 2                                              # index of I_1 in theta
        # bg='frame' (default, unchanged): one bg_t per frame; bg='angle' (critic 34 #4 diagnostic): one bg per pattern
        # angle (fit(..., ang=[n,T] angle index) required), as the CRLB_SF model
        self.bgm = bg
        self.bmap = None

    def model(self, th, T):
        k = self.k
        z = th[:, 2] if self.zfree else torch.full_like(th[:, 0], self.z0)
        g, gx, gy, gz = self.lut.render(torch.stack([th[:, 0], th[:, 1], z], 1), grad=True)   # [n,P,P]
        I = th[:, k:k + T]; bg = th[:, k + T:]
        if self.bmap is not None: bg = torch.einsum('nj,njt->nt', bg, self.bmap)
        mu = I[..., None, None] * g[:, None] + bg[..., None, None]
        return mu, g, (gx, gy, gz)

    def jac(self, th, g, gd, T):
        n, P, k = th.shape[0], self.P, self.k
        I = th[:, k:k + T]
        nbg = th.shape[1] - k - T
        J = torch.zeros(n, k + T + nbg, T, P, P, device=th.device)
        for i in range(k):
            J[:, i] = I[..., None, None] * gd[i][:, None]
        ar = torch.arange(T, device=th.device)
        J[:, k + ar, ar] = g[:, None]
        if self.bmap is None:
            J[:, k + T + ar, ar] = 1.0
        else:
            J[:, k + T:] = self.bmap[..., None, None]
        return J.reshape(n, k + T + nbg, -1)

    @torch.no_grad()
    def fit(self, frames, rn, b, cx, cy, x0, y0, iters=40, xy_box=3.0, ang=None):
        """frames [B,T,H,W] photons; rn [B,1,H,W] read-noise sd (photons); b, cx, cy [n] ROI centre pixel (long);
        x0, y0 [n] start (absolute px). Returns dict x, y, (z,) I [n,T], bg [n,T], cov [n,NP,NP] (theta order x, y, (z,)
        I_1..T, bg_1..T), nll, edge."""
        n = len(b); T = frames.shape[1]; P = self.P; k = self.k
        if n == 0: return None
        if self.bgm == 'angle':
            self.bmap = torch.stack([(ang == 0).float(), (ang == 1).float()], 1)          # [n,2,T]
        NP = k + T + (2 if self.bgm == 'angle' else T)
        y = extract_patches(frames, b, cy, cx, P)                                     # [n,T,P,P]
        rn2 = extract_patches(rn ** 2, b, cy, cx, P)                                  # [n,1,P,P]
        rn2 = rn2.expand(n, T, P, P).reshape(n, -1)
        yv = (y.reshape(n, -1) + rn2).clamp(min=0)
        border = torch.cat([y[:, :, 0], y[:, :, -1], y[:, :, 1:-1, 0], y[:, :, 1:-1, -1]], -1)
        bg0 = border.median(-1).values.clamp(min=0.1)
        dx0, dy0 = x0 - cx.float(), y0 - cy.float()
        g0 = self.lut.render(torch.stack([dx0, dy0, torch.full_like(dx0, self.z0)], 1))
        I0 = ((y - bg0[..., None, None]).sum((2, 3)) / g0.sum((1, 2))[:, None]).clamp(min=1.0)
        zc = [torch.full_like(dx0, self.z0)[:, None]] if self.zfree else []
        if self.bmap is not None:
            bg0 = torch.einsum('nt,njt->nj', bg0, self.bmap / self.bmap.sum(2, keepdim=True).clamp(min=1))
        th = torch.cat([dx0[:, None], dy0[:, None]] + zc + [I0, bg0], 1)

        def nll(mu):
            m = mu.reshape(n, -1).clamp(min=1e-3) + rn2
            return (m - yv * torch.log(m)).sum(1)

        def clampth(t):
            t = t.clone()
            t[:, 0:2] = t[:, 0:2].clamp(-xy_box, xy_box)
            if self.zfree: t[:, 2] = t[:, 2].clamp(0.0, self.zmax)
            t[:, k:k + T] = t[:, k:k + T].clamp(min=-50.0)
            t[:, k + T:] = t[:, k + T:].clamp(min=0.01)
            return t
        mu, g, gd = self.model(th, T); L = nll(mu)
        lam = torch.full((n,), 1e-2, device=frames.device)
        eye = torch.eye(NP, device=frames.device)
        for _ in range(iters):
            J = self.jac(th, g, gd, T)
            m = mu.reshape(n, -1).clamp(min=1e-3) + rn2
            w = 1.0 / m
            F = torch.einsum('npk,nqk,nk->npq', J, J, w)
            gr = torch.einsum('npk,nk->np', J, yv / m - 1.0)
            Fd = F + lam[:, None, None] * torch.diag_embed(F.diagonal(dim1=1, dim2=2)) + 1e-6 * eye
            step = torch.linalg.solve(Fd, gr[..., None])[..., 0]
            cand = clampth(th + step)
            mu_c, g_c, gd_c = self.model(cand, T); Lc = nll(mu_c)
            ok = Lc < L
            th = torch.where(ok[:, None], cand, th)
            mu = torch.where(ok[:, None, None, None], mu_c, mu)
            g = torch.where(ok[:, None, None], g_c, g); gd = tuple(torch.where(ok[:, None, None], a_, b_) for a_, b_ in zip(gd_c, gd))
            L = torch.where(ok, Lc, L)
            lam = torch.where(ok, lam * 0.3, lam * 8).clamp(1e-7, 1e7)
        J = self.jac(th, g, gd, T)
        m = mu.reshape(n, -1).clamp(min=1e-3) + rn2
        F = torch.einsum('npk,nqk,nk->npq', J, J, 1.0 / m).double()
        d = F.diagonal(dim1=1, dim2=2).clamp(min=1e-30).rsqrt()
        M = F * d[:, :, None] * d[:, None, :]
        cov = (torch.linalg.pinv(M + 1e-10 * torch.eye(NP, device=F.device, dtype=F.dtype), rtol=1e-12) * d[:, :, None] * d[:, None, :]).float()
        bgo = th[:, k + T:] if self.bmap is None else torch.einsum('nj,njt->nt', th[:, k + T:], self.bmap)
        self.bmap = None
        return dict(x=th[:, 0] + cx.float(), y=th[:, 1] + cy.float(), z=th[:, 2] if self.zfree else None, I=th[:, k:k + T], bg=bgo,
                    cov=cov, nll=L, edge=(th[:, :2].abs() >= xy_box - 1e-3).any(1), k=k)


def solve_input(r, x_det, y_det):
    """LM output -> the dict solve.solve_emitters expects (I, Cholesky of the I-block covariance, pattern-free position prior)"""
    T = r['I'].shape[1]; k = r['k']
    CI = r['cov'][:, k:k + T, k:k + T]
    CI = 0.5 * (CI + CI.transpose(1, 2)) + 1e-3 * torch.eye(T, device=CI.device)
    L, info = torch.linalg.cholesky_ex(CI)
    bad = info != 0
    if bad.any():
        L[bad] = torch.diag_embed(CI[bad].diagonal(dim1=1, dim2=2).clamp(min=1e-3).sqrt())
    sxy = torch.stack([r['cov'][:, 0, 0], r['cov'][:, 1, 1]], 1).clamp(min=1e-6).sqrt()
    return dict(I=r['I'], L=L, x=r['x'], y=r['y'], lsxy=sxy.log(), x_det=x_det, y_det=y_det)


# explicit pattern solve settings for the LM reference: the same fringe-start posterior mean (PM, T = 1, box 1.5) as
# the network path, with the LM's own Fisher covariance unscaled (kap = 1) and no chi2 sigma inflation
_ONE = dict(ph=[1.0, 2.0], kap_p=[1.0, 1.0], kap_I2=[1.0, 1.0], f=[1.0, 1.0])     # constant 1 (all solve calibration keys are required)
LM_CAL = dict(kap_bins=_ONE, sig_ph=_ONE, prior_off=[0.0, 0.0], cmult=0.0, cmult_ang=0.0, sig_a=1.0, sig_g=0.0, box=1.5, mix_T=1.0)


# ------------------------------------------------------------------------------------------------------------------------
# Joint SIMFLUX MLE reference (critic 34 #4, 2026-11-10). EVALUATION ONLY, like LMRef above; not part of the method.
# One-stage pixel likelihood with the pattern KNOWN (the scene's patt_est = the true pattern on the exact-model scenes):
#     mu_t(px) = B * a_t * il_t(X, Y) * psf(px; x, y, z) + bg_t
#   il_t(X, Y) = rel_t (1 + m_t cos(kx_t X + ky_t Y - phi_t)) at the absolute FOV position (X, Y) = (cx + x, cy + y)
#   a_t = 1 (kin=False: always on, the iso model) or the on-fraction of frame t for an on-interval [t0, t1]
#         (kin=True: the CRLB_SF kinetics nuisance, simcode.simulate.crlb onfrac)
#   bg: 'frame' (6 bg_t), 'angle' (one per pattern angle, = the CRLB_SF model) or 'one'
# theta = (x, y, [z,] B, [t0, t1,] bg...). Same ROI / candidates / sCMOS-Poisson likelihood / Fisher-scoring LM as LMRef.
# Start: the pattern-free LMRef fit (z free) from the detector point; then the SIMFLUX LM from the 9 fringe-shifted starts
# (0, +-1 pitch along each angle's k) and the start with the lowest final NLL is kept (the SIMFLUX likelihood is multimodal
# along k with period = the pitch).
class SFRef:
    def __init__(self, sim, P=11, zfree=True, bg='angle', kin=False, z0=0.05):
        self.lm = LMRef(sim, P, z0=z0, zfree=True)
        self.lut = self.lm.lut; self.P = P; self.zfree = zfree; self.z0 = z0; self.zmax = sim.zmax
        self.bgm = bg; self.kin = kin

    def _layout(self, T):
        k = 3 if self.zfree else 2                     # index of B
        kk = k + 1 + (2 if self.kin else 0)            # index of the first bg
        nbg = {'frame': T, 'angle': 2, 'one': 1}[self.bgm]
        return k, kk, nbg

    def _bgmap(self, ang, T):
        """[n,nbg,T] 0/1 map bg parameter -> frame"""
        n = ang.shape[0]
        if self.bgm == 'frame': return torch.eye(T, device=ang.device)[None].expand(n, T, T)
        if self.bgm == 'one': return torch.ones(n, 1, T, device=ang.device)
        return torch.stack([(ang == 0).float(), (ang == 1).float()], 1)

    def model(self, th, pt, cx, cy, bmap, T, grad=True):
        k, kk, nbg = self._layout(T)
        z = th[:, 2] if self.zfree else torch.full_like(th[:, 0], self.z0)
        g, gx, gy, gz = self.lut.render(torch.stack([th[:, 0], th[:, 1], z], 1), grad=True)
        X = cx.float() + th[:, 0]; Y = cy.float() + th[:, 1]
        arg = pt[..., 0] * X[:, None] + pt[..., 1] * Y[:, None] - pt[..., 2]
        il = pt[..., 4] * (1 + pt[..., 3] * torch.cos(arg))
        s = -pt[..., 4] * pt[..., 3] * torch.sin(arg)
        dilx, dily = s * pt[..., 0], s * pt[..., 1]
        if self.kin:
            fr = torch.arange(T, device=th.device, dtype=th.dtype)[None]
            t0, t1 = th[:, k + 1, None], th[:, k + 2, None]
            a = (torch.minimum(t1, fr + 1) - torch.maximum(t0, fr)).clamp(0, 1)
            pos = a > 0
            da0 = -((t0 >= fr) & (t0 < fr + 1) & pos).float(); da1 = ((t1 > fr) & (t1 <= fr + 1) & pos).float()
        else:
            a = torch.ones_like(il); da0 = da1 = None
        B = th[:, k, None]
        bgt = torch.einsum('nj,njt->nt', th[:, kk:kk + nbg], bmap)
        amp = B * a * il                                                            # [n,T] photons per frame
        mu = amp[..., None, None] * g[:, None] + bgt[..., None, None]
        if not grad: return mu, None
        cols = [(B * a * dilx)[..., None, None] * g[:, None] + amp[..., None, None] * gx[:, None],
                (B * a * dily)[..., None, None] * g[:, None] + amp[..., None, None] * gy[:, None]]
        if self.zfree: cols.append(amp[..., None, None] * gz[:, None])
        cols.append((a * il)[..., None, None] * g[:, None])
        if self.kin:
            cols += [(B * da0 * il)[..., None, None] * g[:, None], (B * da1 * il)[..., None, None] * g[:, None]]
        P = self.P
        for j in range(nbg):
            cols.append(bmap[:, j, :, None, None].expand(-1, T, P, P))
        J = torch.stack(cols, 1).reshape(th.shape[0], len(cols), -1)
        return mu, J

    def _clamp(self, t, T, xy_box):
        k, kk, nbg = self._layout(T)
        t = t.clone()
        t[:, 0:2] = t[:, 0:2].clamp(-xy_box, xy_box)
        if self.zfree: t[:, 2] = t[:, 2].clamp(0.0, self.zmax)
        t[:, k] = t[:, k].clamp(min=1.0)
        if self.kin:
            t[:, k + 1] = t[:, k + 1].clamp(-0.5, T - 0.05); t[:, k + 2] = torch.maximum(t[:, k + 2].clamp(0.05, T + 0.5), t[:, k + 1] + 0.05)
        t[:, kk:] = t[:, kk:].clamp(min=0.01)
        return t

    def _lm(self, th, yv, rn2, pt, cx, cy, bmap, T, iters, xy_box):
        n = th.shape[0]; NP = th.shape[1]
        def nll(mu):
            m = mu.reshape(n, -1).clamp(min=1e-3) + rn2
            return (m - yv * torch.log(m)).sum(1)
        mu, J = self.model(th, pt, cx, cy, bmap, T); L = nll(mu)
        lam = torch.full((n,), 1e-2, device=th.device); eye = torch.eye(NP, device=th.device)
        for _ in range(iters):
            m = mu.reshape(n, -1).clamp(min=1e-3) + rn2
            F = torch.einsum('npk,nqk,nk->npq', J, J, 1.0 / m)
            gr = torch.einsum('npk,nk->np', J, yv / m - 1.0)
            Fd = F + lam[:, None, None] * torch.diag_embed(F.diagonal(dim1=1, dim2=2)) + 1e-6 * eye
            step = torch.linalg.solve(Fd, gr[..., None])[..., 0]
            cand = self._clamp(th + step, T, xy_box)
            mu_c, J_c = self.model(cand, pt, cx, cy, bmap, T); Lc = nll(mu_c)
            ok = Lc < L
            th = torch.where(ok[:, None], cand, th); mu = torch.where(ok[:, None, None, None], mu_c, mu)
            J = torch.where(ok[:, None, None], J_c, J); L = torch.where(ok, Lc, L)
            lam = torch.where(ok, lam * 0.3, lam * 8).clamp(1e-7, 1e7)
        m = mu.reshape(n, -1).clamp(min=1e-3) + rn2
        F = torch.einsum('npk,nqk,nk->npq', J, J, 1.0 / m).double()
        return th, L, F

    @torch.no_grad()
    def fit(self, frames, rn, patt, b, cx, cy, x0, y0, iters=40, iters0=40, xy_box=3.0, nshift=1):
        """frames [B,T,H,W] photons; rn [B,1,H,W]; patt [B,T,5] (kx, ky, phi, m, rel; the KNOWN pattern); b, cx, cy [n] ROI
        centre pixel; x0, y0 [n] start (absolute px; the pattern-free LM is run from here first). Returns dict x, y, z, B,
        bg, (t0, t1), cov [n,NP,NP] (inverse Fisher at the estimate, theta order), nll, x_lm / y_lm (pattern-free LM), edge,
        ishift (index of the chosen fringe start; 4 = no shift for nshift 1)."""
        n = len(b); T = frames.shape[1]; P = self.P
        if n == 0: return None
        r0 = self.lm.fit(frames, rn, b, cx, cy, x0, y0, iters=iters0, xy_box=xy_box)
        k, kk, nbg = self._layout(T)
        y = extract_patches(frames, b, cy, cx, P)
        rn2 = extract_patches(rn ** 2, b, cy, cx, P).expand(n, T, P, P).reshape(n, -1)
        yv = (y.reshape(n, -1) + rn2).clamp(min=0)
        pt = patt[b].float()
        ang = (pt[..., 0].abs() < pt[..., 1].abs()).long()
        bmap = self._bgmap(ang, T)
        # start: LM position / z; B from the LM I_t projected on the pattern (least squares on il_t); bg from the LM bg_t
        dx0, dy0 = r0['x'] - cx.float(), r0['y'] - cy.float()
        z0 = r0['z'].clamp(0, self.zmax)
        X = cx.float() + dx0; Y = cy.float() + dy0
        il = pt[..., 4] * (1 + pt[..., 3] * torch.cos(pt[..., 0] * X[:, None] + pt[..., 1] * Y[:, None] - pt[..., 2]))
        It = r0['I'].clamp(min=0)
        if self.kin:
            q = It / il.clamp(min=1e-3)
            B0 = q.topk(min(3, T), 1).values.median(1).values.clamp(min=1.0)
            a0 = (q / B0[:, None]).clamp(0, 1); on = a0 > 0.3
            fi = torch.where(on.any(1), on.float().argmax(1), torch.zeros_like(B0, dtype=torch.long))
            la = torch.where(on.any(1), T - 1 - on.flip(1).float().argmax(1), torch.full_like(fi, T - 1))
            t0 = fi.float() + 1 - a0.gather(1, fi[:, None])[:, 0].clamp(min=0.05)
            t1 = la.float() + a0.gather(1, la[:, None])[:, 0].clamp(min=0.05)
            kin0 = [t0[:, None], t1[:, None]]
        else:
            B0 = ((It * il).sum(1) / (il * il).sum(1).clamp(min=1e-6)).clamp(min=1.0); kin0 = []
        bgw = bmap / bmap.sum(2, keepdim=True).clamp(min=1)
        bg0 = torch.einsum('nt,njt->nj', r0['bg'], bgw).clamp(min=0.01)
        zc = [z0[:, None]] if self.zfree else []
        th0 = torch.cat([dx0[:, None], dy0[:, None]] + zc + [B0[:, None]] + kin0 + [bg0], 1)
        # fringe-shifted starts: one pitch vector per angle (mean k over that angle's frames)
        kv = torch.stack([(pt[..., :2] * (ang == a_)[..., None]).sum(1) / (ang == a_).sum(1, keepdim=True).clamp(min=1) for a_ in (0, 1)], 1)   # [n,2,2]
        pv = 2 * torch.pi * kv / (kv ** 2).sum(-1, keepdim=True).clamp(min=1e-9)
        best = None
        rng = range(-nshift, nshift + 1); si = 0
        for i in rng:
            for j in rng:
                th = th0.clone(); th[:, 0:2] = th[:, 0:2] + i * pv[:, 0] + j * pv[:, 1]
                th = self._clamp(th, T, xy_box)
                th, L, F = self._lm(th, yv, rn2, pt, cx, cy, bmap, T, iters, xy_box)
                if best is None:
                    best = [th, L, F, torch.full((n,), si, device=th.device)]
                else:
                    ok = L < best[1]
                    best = [torch.where(ok[:, None], th, best[0]), torch.where(ok, L, best[1]), torch.where(ok[:, None, None], F, best[2]),
                            torch.where(ok, torch.full_like(best[3], si), best[3])]
                si += 1
        th, L, F, ish = best
        NP = th.shape[1]
        d = F.diagonal(dim1=1, dim2=2).clamp(min=1e-30).rsqrt()
        M = F * d[:, :, None] * d[:, None, :]
        cov = (torch.linalg.pinv(M + 1e-10 * torch.eye(NP, device=F.device, dtype=F.dtype), rtol=1e-12) * d[:, :, None] * d[:, None, :]).float()
        out = dict(x=th[:, 0] + cx.float(), y=th[:, 1] + cy.float(), z=th[:, 2] if self.zfree else None, B=th[:, k],
                   bg=th[:, kk:], cov=cov, nll=L, edge=(th[:, :2].abs() >= xy_box - 1e-3).any(1), ishift=ish,
                   x_lm=r0['x'], y_lm=r0['y'], s_lm=torch.stack([r0['cov'][:, 0, 0], r0['cov'][:, 1, 1]], 1).clamp(min=0).sqrt(),
                   I_lm=r0['I'], lm=r0)
        if self.kin: out.update(t0=th[:, k + 1], t1=th[:, k + 2])
        return out
