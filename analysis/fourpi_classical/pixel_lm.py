"""(analysis code of the classical 4Pi baselines; moved out of the package 2026-11-15: the network pipeline has no pixel fit.
torch_psf.fit.lm_poisson is the same scheme but not numerically identical: damping start 1.0 vs 1e-2, lambda floor 1e-4
vs 1e-6, double-precision step, no clamp of d + v; the classical tables were made with this one.)
Batched Levenberg-Marquardt Poisson MLE (sCMOS read-noise shift v: mu + v ~ Poisson) with a central-difference
Jacobian; generic model f(theta [n,p]) -> mu [n,M]. Also the Fisher matrix / CRLB at theta. Used for the classical
4Pi baselines (pixel MLE on the 24 sub-images, summed astigmatic SMLM) and the explicit intensity solve (Gaussian
NLL variant `lm_gauss`)."""
import torch


def jac(f, th, eps):
    cols = []
    for p in range(th.shape[1]):
        tp = th.clone(); tm = th.clone()
        tp[:, p] += eps[p]; tm[:, p] -= eps[p]
        cols.append((f(tp) - f(tm)) / (2 * eps[p]))
    return torch.stack(cols, -1)                                 # [n,M,p]


def loglik(d, mu, v):
    mv = (mu + v).clamp(min=1e-6)
    return ((d + v).clamp(min=0) * torch.log(mv) - mv).sum(-1)


def lm_poisson(f, d, th0, v, eps, lo=None, hi=None, iters=30, lam0=1e-2):
    th = th0.clone(); n, p = th.shape
    lam = torch.full((n,), lam0, device=th.device)
    mu = f(th); ll = loglik(d, mu, v)
    eye = torch.eye(p, device=th.device)
    for _ in range(iters):
        J = jac(f, th, eps)
        w = 1.0 / (mu + v).clamp(min=1e-3)
        r = ((d + v).clamp(min=0) - (mu + v)) * w                 # score weights
        g = torch.einsum('nmp,nm->np', J, r)
        H = torch.einsum('nmp,nm,nmq->npq', J, w, J)
        Hd = H + lam[:, None, None] * (torch.diagonal(H, dim1=1, dim2=2)[:, :, None] * eye + 1e-6 * eye)
        step = torch.linalg.solve(Hd, g[..., None])[..., 0]
        tn = th + step
        if lo is not None: tn = torch.maximum(tn, lo)
        if hi is not None: tn = torch.minimum(tn, hi)
        mun = f(tn); lln = loglik(d, mun, v)
        ok = lln > ll
        th = torch.where(ok[:, None], tn, th); mu = torch.where(ok[:, None], mun, mu); ll = torch.where(ok, lln, ll)
        lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-6, 1e6)
    return th, ll, mu


def fisher(f, th, v, eps):
    J = jac(f, th, eps); mu = f(th)
    return torch.einsum('nmp,nm,nmq->npq', J, 1.0 / (mu + v).clamp(min=1e-3), J)


def crlb(F):
    return torch.sqrt(torch.diagonal(torch.linalg.pinv(F), dim1=1, dim2=2).clamp(min=0))


def lm_poisson_j(fj, d, th0, v, lo=None, hi=None, iters=30, lam0=1e-2):
    """as lm_poisson with an analytic model: fj(theta) -> (mu [n,M], J [n,M,p])"""
    th = th0.clone(); n, p = th.shape
    lam = torch.full((n,), lam0, device=th.device)
    mu, J = fj(th); ll = loglik(d, mu, v)
    eye = torch.eye(p, device=th.device)
    for _ in range(iters):
        w = 1.0 / (mu + v).clamp(min=1e-3)
        r = ((d + v).clamp(min=0) - (mu + v)) * w
        g = torch.einsum('nmp,nm->np', J, r)
        H = torch.einsum('nmp,nm,nmq->npq', J, w, J)
        Hd = H + lam[:, None, None] * (torch.diagonal(H, dim1=1, dim2=2)[:, :, None] * eye + 1e-6 * eye)
        step = torch.linalg.solve(Hd, g[..., None])[..., 0]
        tn = th + step
        if lo is not None: tn = torch.maximum(tn, lo)
        if hi is not None: tn = torch.minimum(tn, hi)
        mun, Jn = fj(tn); lln = loglik(d, mun, v)
        ok = lln > ll
        th = torch.where(ok[:, None], tn, th); mu = torch.where(ok[:, None], mun, mu)
        J = torch.where(ok[:, None, None], Jn, J); ll = torch.where(ok, lln, ll)
        lam = torch.where(ok, lam * 0.3, lam * 5).clamp(1e-6, 1e6)
    return th, ll, mu


def fisher_j(fj, th, v):
    mu, J = fj(th)
    return torch.einsum('nmp,nm,nmq->npq', J, 1.0 / (mu + v).clamp(min=1e-3), J)
