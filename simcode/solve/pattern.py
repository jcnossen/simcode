"""
Pattern geometry helpers of the explicit intensity-domain pattern solve (simcode/hd/msolve.py, msolve_fast.py).
Pattern per frame pt [T,5] = (kx, ky, phase, depth m, relint): ill_t(r) = rel_t (1 + m_t cos(k_t . r - phase_t)).
"""
import math
import torch


# ------------------------------------------------------------------------------------------------ pattern helpers
def illum_at(pt, X, Y):
    """pt [n,T,5]; X, Y [n] -> ill, d ill/dX, d ill/dY [n,T]."""
    arg = pt[..., 0] * X[:, None] + pt[..., 1] * Y[:, None] - pt[..., 2]
    s = -pt[..., 4] * pt[..., 3] * torch.sin(arg)
    return pt[..., 4] * (1 + pt[..., 3] * torch.cos(arg)), s * pt[..., 0], s * pt[..., 1]


def onfrac(ton, toff, T):
    """on-fraction of frames [t, t+1) for an on-interval [ton, toff] and its derivatives (piecewise linear)."""
    fr = torch.arange(T, device=ton.device, dtype=torch.float32)[None]
    a_ = (torch.minimum(toff[:, None], fr + 1) - torch.maximum(ton[:, None], fr)).clamp(0, 1)
    pos = a_ > 0
    d_on = -((ton[:, None] >= fr) & (ton[:, None] < fr + 1) & pos).float()
    d_off = ((toff[:, None] > fr) & (toff[:, None] <= fr + 1) & pos).float()
    return a_, d_on, d_off


def angle_index(pt):
    return (pt[..., 0].abs() < pt[..., 1].abs()).long()


def angle_k(pt):
    """mean k vector per angle [n,2(angle),2(kx,ky)] and the angle index of every frame [n,T]."""
    ang = angle_index(pt)
    ks = []
    for a_ in (0, 1):
        m_ = (ang == a_).float()
        ks.append((pt[..., :2] * m_[..., None]).sum(1) / m_.sum(1).clamp(min=1)[:, None])
    return torch.stack(ks, 1), ang


def fringe_shifts(Km, nf=1):
    """position shifts of the fringe hypotheses: dr with k_a . dr = 2 pi n_a, n_a in {-nf..nf}; [n,H,2], H=(2nf+1)^2.
    Index H//2 is (0, 0)."""
    n = Km.shape[0]
    nn_ = torch.tensor([[i, j] for i in range(-nf, nf + 1) for j in range(-nf, nf + 1)], device=Km.device,
                       dtype=torch.float32) * 2 * math.pi
    H = len(nn_)
    return torch.linalg.solve(Km[:, None].expand(n, H, 2, 2), nn_[None].expand(n, H, 2)[..., None])[..., 0]


def fringe_error(r_est, r_true, pt):
    """True where the estimate is in a wrong fringe along either pattern direction (|k_a . dr| > pi)."""
    Km, _ = angle_k(pt)
    ph = (Km * (r_est - r_true)[:, None, :]).sum(-1)
    return (ph.abs() > math.pi).any(1)


def kinetics_init(I, il0, T):
    """(B, t_on, t_off) start from per-frame intensities I [n,T] and the illumination il0 [n,T] at the start position
    . Boundaries are placed
    inside a frame unless that frame is clearly fully on (> 0.95); a dark neighbouring frame (illumination < 0.3 of
    the mean) carries no on/off information, so the interval is extended through it."""
    ratio = I.clamp(min=0) / il0.clamp(min=0.05)
    B0 = torch.quantile(ratio, 0.8, dim=1).clamp(min=10)
    ah = (ratio / B0[:, None]).clamp(0, 1); on = ah > 0.5
    fi = torch.where(on.any(1), on.float().argmax(1), torch.zeros_like(B0, dtype=torch.long))
    la = torch.where(on.any(1), T - 1 - on.flip(1).float().argmax(1), torch.full_like(B0, T - 1, dtype=torch.long))
    af, al = ah.gather(1, fi[:, None])[:, 0], ah.gather(1, la[:, None])[:, 0]
    t_on = torch.where((fi == 0) & (af > 0.95), torch.full_like(B0, -0.5), fi.float() + 1 - af.clamp(max=0.95))
    t_off = torch.where((la == T - 1) & (al > 0.95), torch.full_like(B0, T + 0.5), la.float() + al.clamp(max=0.95))
    rel_il = il0 / il0.mean(1, keepdim=True)
    prev_dark = (fi > 0) & (rel_il.gather(1, (fi - 1).clamp(min=0)[:, None])[:, 0] < 0.3)
    next_dark = (la < T - 1) & (rel_il.gather(1, (la + 1).clamp(max=T - 1)[:, None])[:, 0] < 0.3)
    t_on = torch.where(prev_dark, torch.where(fi - 1 == 0, torch.full_like(B0, -0.5), (fi - 1).float()), t_on)
    t_off = torch.where(next_dark, torch.where(la + 1 == T - 1, torch.full_like(B0, T + 0.5), (la + 2).float()), t_off)
    return B0, t_on, t_off




def chi2_sf(q, dof):
    """chi^2 survival function P(chi2_dof > q) (regularised upper incomplete gamma)"""
    return torch.special.gammaincc(dof / 2, q.clamp(min=0) / 2)
