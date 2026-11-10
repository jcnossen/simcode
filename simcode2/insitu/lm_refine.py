"""
Batched Levenberg-Marquardt fit of the full modulated pixel model (MAP with an optional Gaussian position prior).
CALIBRATION / DIAGNOSTICS ONLY: used by insitu/coma_anchor_scan.py (odd-aberration profile scan). It is not part of the
SIMCODE v2 localization method, which uses no pixel-level fit (the networks give the per-frame intensities).
Ported unchanged (lm_refine + scaled_cov_diag) from the development LM refinement module.

Model per detection (ROI PxP centred on its candidate pixel, all T frames):
    mu_t(px) = B * illum_t(X, Y) * a_t * psf(px; x, y, z) + bg[angle(t)]
theta = (x, y, z, B, bg0, bg1, bg_slope_x, bg_slope_y[, t_on, t_off]).
Kinetics mode (a_t=None): a_t = overlap of the binding interval [t_on, t_off] with frame [t, t+1] (the simulator's
model), which keeps the modulation information while handling bind/unbind frames.
Prior: Gaussian on (x, y, z) centred on theta0 with sigmas prior_sigma (prior_scale inflates them). Gate: if the MAP
estimate moves more than `gate` prior-sigmas in x or y, the initial estimate is kept (flag returned).
"""
import torch


def scaled_cov_diag(FI):
    """Diagonal of the inverse Fisher matrix with Jacobi scaling: pinv's default rtol truncated the
    x/y directions when a tight prior (e.g. z fixed with sigma 1e-4) made the matrix badly scaled. Unidentifiable
    directions (zero diagonal) get a huge variance instead of 0."""
    d = FI.diagonal(dim1=1, dim2=2).clamp(min=1e-30)
    s = d.rsqrt()
    M = FI * s[:, :, None] * s[:, None, :]
    eye = torch.eye(FI.shape[-1], device=FI.device, dtype=FI.dtype)
    Minv = torch.linalg.pinv(M + 1e-9 * eye, rtol=1e-10)
    var = Minv.diagonal(dim1=1, dim2=2) * s ** 2
    return var.clamp(min=0).sqrt()



def lm_refine(frames_roi, illum_fn, psf_model, focus, theta0, a_t, rn2, angle, prior_sigma=None, prior_scale=1.5,
              gate=3.0, iters=25, zmax=0.2, lam0=1e-2, ton_max=None, toff_min=None):
    """
    frames_roi [N,T,P,P] photons; theta0 [N,6] or [N,8] (slopes default 0); illum_fn(x,y)->[N,T]; a_t [N,T]; angle [N,T] in {0,1};
    prior_sigma [N,3] (x,y px; z um) or None. Returns theta [N,6], std [N,6], accepted [N] bool.
    """
    N, T, P, _ = frames_roi.shape
    dev = frames_roi.device
    kin = a_t is None
    if theta0.shape[1] == 6:
        theta0 = torch.cat([theta0, torch.zeros(N, 2, device=dev)], 1)
    NP = theta0.shape[1]
    assert NP == (10 if kin else 8)
    fr = torch.arange(T, device=dev, dtype=torch.float32)
    # read noise: scalar or per-pixel (sCMOS) map broadcastable to [N,T,P,P] -> flattened [N, T*P*P]
    if torch.is_tensor(rn2) and rn2.dim() > 0:
        rn2 = rn2.expand(N, T, P, P).reshape(N, -1)
    fvec = torch.full((N,), focus, device=dev)
    A1 = angle.float()[..., None, None]
    o = torch.arange(P, device=dev, dtype=torch.float32) - (P - 1) / 2
    U = o[None, None, None, :]; V = o[None, None, :, None]

    def model(th):
        p = psf_model(th[:, :3], fvec)                                   # [N,P,P]
        at = ((torch.minimum(th[:, 9:10], fr + 1) - torch.maximum(th[:, 8:9], fr)).clamp(0, 1) if kin else a_t)
        il = illum_fn(th[:, 0], th[:, 1]) * at                           # [N,T]
        bg = th[:, 4, None, None, None] * (1 - A1) + th[:, 5, None, None, None] * A1 \
            + th[:, 6, None, None, None] * U + th[:, 7, None, None, None] * V
        return th[:, 3, None, None, None] * il[..., None, None] * p[:, None] + bg

    def jac(th, mu):
        J = []
        for i in range(NP):
            base = [1e-3, 1e-3, 1e-3, 1e-2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2][i]
            h = base * (1 + th[:, i].abs()) if 3 <= i <= 5 else torch.full((N,), base, device=dev)
            d = torch.zeros_like(th); d[:, i] = h
            J.append(((model(th + d) - mu) / h[:, None, None, None]).reshape(N, -1))
        return torch.stack(J, 1)                                          # [N,NP,K]

    y = frames_roi.reshape(N, -1)
    mu0 = theta0[:, :3].clone()
    if prior_sigma is not None:
        pinv = torch.zeros(N, NP, device=dev)
        pinv[:, :3] = 1.0 / (prior_sigma * prior_scale).clamp(min=1e-4) ** 2
    else:
        pinv = torch.zeros(N, NP, device=dev)

    def objective(th, mu):
        mu = mu.reshape(N, -1)
        var = mu.clamp(min=1e-3) + rn2
        L = (0.5 * (y - mu) ** 2 / var + 0.5 * torch.log(var)).sum(1)
        return L + 0.5 * (pinv[:, :3] * (th[:, :3] - mu0) ** 2).sum(1)

    th = theta0.clone()
    lam = torch.full((N,), lam0, device=dev)
    eye = torch.eye(NP, device=dev)
    with torch.no_grad():
        mu = model(th); L = objective(th, mu)
        for _ in range(iters):
            J = jac(th, mu)
            w = 1.0 / (mu.reshape(N, -1).clamp(min=1e-3) + rn2)
            r = (y - mu.reshape(N, -1))
            A = torch.einsum('npk,nqk,nk->npq', J, J, w) + torch.diag_embed(pinv)
            g = torch.einsum('npk,nk->np', J, w * r)
            g[:, :3] -= pinv[:, :3] * (th[:, :3] - mu0)
            Ad = A + lam[:, None, None] * torch.diag_embed(A.diagonal(dim1=1, dim2=2))
            step = torch.linalg.solve(Ad + 1e-9 * eye, g[..., None])[..., 0]
            cand = th + step
            cand[:, 0:2] = cand[:, 0:2].clamp(-P / 2 + 1, P / 2 - 1)
            cand[:, 2] = cand[:, 2].clamp(0, zmax)
            cand[:, 3] = cand[:, 3].clamp(min=1.0)
            cand[:, 4:6] = cand[:, 4:6].clamp(min=0.0)
            if kin:
                cand[:, 8] = cand[:, 8].clamp(-1.0, T - 0.05)
                cand[:, 9] = torch.maximum(cand[:, 9], cand[:, 8] + 0.05).clamp(max=T + 1.0)
                if ton_max is not None: cand[:, 8] = torch.minimum(cand[:, 8], ton_max)
                if toff_min is not None: cand[:, 9] = torch.maximum(cand[:, 9], toff_min)
            mu_c = model(cand); Lc = objective(cand, mu_c)
            ok = Lc < L
            th = torch.where(ok[:, None], cand, th)
            mu = torch.where(ok[:, None, None, None], mu_c, mu)
            L = torch.where(ok, Lc, L)
            lam = torch.where(ok, lam * 0.3, lam * 10).clamp(1e-6, 1e6)
        J = jac(th, mu)
        w = 1.0 / (mu.reshape(N, -1).clamp(min=1e-3) + rn2)
        FI = torch.einsum('npk,nqk,nk->npq', J, J, w) + torch.diag_embed(pinv)
        std = scaled_cov_diag(FI)
        if prior_sigma is not None:
            dev_sig = ((th[:, :2] - mu0[:, :2]).abs() / prior_sigma[:, :2].clamp(min=1e-4)).max(1).values
            accepted = dev_sig <= gate
        else:
            accepted = torch.ones(N, dtype=torch.bool, device=dev)
        th = torch.where(accepted[:, None], th, torch.cat([theta0[:, :3], th[:, 3:]], 1))
    return th, std, accepted
