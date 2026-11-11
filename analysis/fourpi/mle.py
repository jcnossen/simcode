"""Batched Poisson MLE (Levenberg-Marquardt, per-sample damping) on CPU/GPU with torch, plus CRLB.

Noise model: sCMOS approximation, data d (e) and model mu get + v (v = read-noise variance in e^2 of the summed pixel),
log L = sum (d + v) log(mu + v) - (mu + v). Fisher I_ij = sum dmu_i dmu_j / (mu + v).

Models:
  GaussAstig: theta = (x, y, N, bg, sx, sy), pixel-integrated (erf) elliptical Gaussian, x/y relative to ROI centre (px).
  PSFModel(psf): theta = (x, y, z, N, bg) with psf.eval3(x, y, z, P) -> g, gx, gy, gz (normalised PSF and derivatives).
ROI convention: pixel j of a P-px ROI has its centre at j - (P - 1) / 2.
"""
import math
import torch


class GaussAstig:
    npar = 6
    names = ('x', 'y', 'N', 'bg', 'sx', 'sy')

    def __init__(self, P, smin=0.6, smax=4.5):
        self.P = P; self.smin, self.smax = smin, smax

    def __call__(self, th):
        P = self.P
        o = torch.arange(P, dtype=th.dtype, device=th.device) - (P - 1) / 2
        def ax(c, s):
            s2 = math.sqrt(2) * s[:, None]
            a = (o[None] + 0.5 - c[:, None]) / s2; b = (o[None] - 0.5 - c[:, None]) / s2
            e = 0.5 * (torch.erf(a) - torch.erf(b))
            ga, gb = torch.exp(-a ** 2) / math.sqrt(math.pi), torch.exp(-b ** 2) / math.sqrt(math.pi)
            dc = -(ga - gb) / s2                       # d e / d c
            ds = -(ga * a - gb * b) / s[:, None]       # d e / d s
            return e, dc, ds
        x, y, N, bg, sx, sy = th.unbind(1)
        ex, dex, dsx = ax(x, sx); ey, dey, dsy = ax(y, sy)
        g = ey[:, :, None] * ex[:, None, :]
        mu = N[:, None, None] * g + bg[:, None, None]
        J = torch.stack([N[:, None, None] * ey[:, :, None] * dex[:, None, :],
                         N[:, None, None] * dey[:, :, None] * ex[:, None, :],
                         g, torch.ones_like(g),
                         N[:, None, None] * ey[:, :, None] * dsx[:, None, :],
                         N[:, None, None] * dsy[:, :, None] * ex[:, None, :]], 1)
        return mu, J

    def clamp(self, th):
        P = self.P
        th[:, 0:2] = th[:, 0:2].clamp(-P / 2 + 1, P / 2 - 1)
        th[:, 2] = th[:, 2].clamp(min=1.0)
        th[:, 3] = th[:, 3].clamp(min=1e-3)
        th[:, 4:6] = th[:, 4:6].clamp(self.smin, self.smax)
        return th

    def init(self, roi, bg0=None):
        B, P, _ = roi.shape
        o = torch.arange(P, dtype=roi.dtype) - (P - 1) / 2
        edge = torch.cat([roi[:, 0], roi[:, -1], roi[:, :, 0], roi[:, :, -1]], 1)
        bg = edge.median(1).values.clamp(min=0.01) if bg0 is None else bg0
        s = (roi - bg[:, None, None]).clamp(min=0)
        tot = s.sum((1, 2)).clamp(min=1)
        cx = (s.sum(1) * o).sum(1) / tot; cy = (s.sum(2) * o).sum(1) / tot
        return torch.stack([cx.clamp(-1.5, 1.5), cy.clamp(-1.5, 1.5), tot * 1.1, bg, torch.full_like(bg, 1.5), torch.full_like(bg, 1.5)], 1)


class PSFModel:
    """theta = (x, y, z, N, bg); psf.eval3(x, y, z, P) -> g, gx, gy, gz"""
    npar = 5
    names = ('x', 'y', 'z', 'N', 'bg')

    def __init__(self, psf, P, zlim):
        self.psf, self.P, self.zlim = psf, P, zlim

    def __call__(self, th):
        x, y, z, N, bg = th.unbind(1)
        g, gx, gy, gz = self.psf.eval3(x, y, z, self.P)
        mu = N[:, None, None] * g + bg[:, None, None]
        J = torch.stack([N[:, None, None] * gx, N[:, None, None] * gy, N[:, None, None] * gz, g, torch.ones_like(g)], 1)
        return mu, J

    def clamp(self, th):
        P = self.P
        th[:, 0:2] = th[:, 0:2].clamp(-P / 2 + 1.5, P / 2 - 1.5)
        th[:, 2] = th[:, 2].clamp(*self.zlim)
        th[:, 3] = th[:, 3].clamp(min=1.0)
        th[:, 4] = th[:, 4].clamp(min=1e-3)
        return th


def loglik(d, mu, v):
    mu = mu.clamp(min=1e-6) + v
    dd = (d + v).clamp(min=1e-6)
    return (dd * torch.log(mu) - mu).sum((-1, -2)), (dd * torch.log(dd) - dd).sum((-1, -2))


def lm_fit(model, d, th0, v, iters=40, lam0=1e-2, tol=1e-6):
    """d [B,P,P] (e), th0 [B,np], v scalar or [B,1,1] read-noise variance. Returns th, ll, crlb, chi2, iterations used"""
    th = model.clamp(th0.clone()); B = th.shape[0]
    lam = torch.full((B,), lam0, dtype=d.dtype, device=d.device)
    mu, J = model(th); ll = loglik(d, mu, v)[0]
    I = torch.eye(th.shape[1], dtype=d.dtype, device=d.device)
    for it in range(iters):
        w = 1.0 / (mu.clamp(min=1e-6) + v)
        Jf = J.flatten(2)                                            # [B, np, pix]
        r = ((d - mu) * w).flatten(1)                                 # gradient of LL: sum (d - mu)/(mu+v) dmu
        grad = (Jf * r[:, None]).sum(-1)
        H = torch.bmm(Jf * w.flatten(1)[:, None], Jf.transpose(1, 2))
        Hd = H + lam[:, None, None] * torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2) + 1e-9)
        step = torch.linalg.solve(Hd, grad[..., None])[..., 0]
        thn = model.clamp(th + step)
        mun, Jn = model(thn); lln = loglik(d, mun, v)[0]
        acc = lln >= ll
        th = torch.where(acc[:, None], thn, th); mu = torch.where(acc[:, None, None], mun, mu)
        J = torch.where(acc[:, None, None, None], Jn, J)
        dll = (lln - ll).abs(); ll = torch.where(acc, lln, ll)
        lam = torch.where(acc, lam * 0.3, lam * 10).clamp(1e-6, 1e6)
        if it > 5 and bool(((dll < tol * (1 + ll.abs())) | ~acc).all()) and bool((lam < 1).all()):
            break
    w = 1.0 / (mu.clamp(min=1e-6) + v)
    Jf = J.flatten(2)
    Fi = torch.bmm(Jf * w.flatten(1)[:, None], Jf.transpose(1, 2))
    try:
        cov = torch.linalg.inv(Fi + 1e-9 * I)
        crlb = torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0).sqrt()
    except Exception:
        crlb = torch.full_like(th, float('nan'))
    ll, llsat = loglik(d, mu, v)
    chi2 = (((d - mu) ** 2) / (mu.clamp(min=1e-6) + v)).sum((-1, -2))
    return th, ll, crlb, chi2, ll - llsat, mu
