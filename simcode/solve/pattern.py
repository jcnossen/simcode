"""
Pattern geometry helpers of the explicit intensity-domain pattern solve (simcode/solve/solve.py, solve_fast.py).
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


# (2026-11-19) per-orientation intensity map of the pattern (pattern dict 'relmap', processing/pattern_estimation.estimate_relmap):
# the relative intensity of each orientation varies over the field of view (beam envelope); the solves take it per row as o['relg'].
# The map is the angle contrast h(x, y) = (A_0 - A_1) / mean A as a 2D tensor-product cubic B-spline on a uniform knot grid over the
# map's box (the cubic B-spline kernel of the PSF splines, torch-psf pixint 'bspline'; coefficients fitted, not interpolated).
# Normalisation: the two orientations get the factors g_0 = e^{delta/2}, g_1 = e^{-delta/2}, delta = log((1 + h/2) / (1 - h/2)), so
# g_0 / g_1 = A_0 / A_1 and g_0 g_1 = 1 at every position (geometric mean 1: the map only redistributes intensity between the
# orientations; the common envelope stays in the per-window brightness B, which keeps its meaning); the map has weighted mean 0 over
# the clean windows (the global orientation ratio stays in the per-step rel_s). 2D only: the map has no height dependence (TIRF).
RELMAP_KIND = 'bspline_angle_contrast'


def bspline_kernel(s):
    """the cubic B-spline kernel (C2, support [-2, 2]; = torch_psf.pixint._kernel(s, 'bspline'))"""
    a = s.abs()
    return torch.where(a <= 1, 2 / 3 - a ** 2 + 0.5 * a ** 3, torch.where(a < 2, (2 - a) ** 3 / 6, torch.zeros_like(a)))


def bspline_basis(t, n):
    """the n + 3 uniform cubic B-splines of n knot intervals at t [m] in knot units (clamped to [0, n]): [m, n + 3]; basis k is
    centred on knot k - 1 (knots -1 .. n + 1), and the basis sums to 1 on [0, n] (partition of unity)"""
    t = t.clamp(0, n)
    return bspline_kernel(t[:, None] - (torch.arange(n + 3, device=t.device, dtype=t.dtype)[None] - 1))


def relmap_knot_coords(relmap, x, y):
    """movie positions x, y [m] -> knot coordinates (tx, ty) of the map's grid (float64; clamping is done by bspline_basis)"""
    x0, x1, y0, y1 = (float(v) for v in relmap['box'])
    nx, ny = (int(v) for v in relmap['n'])
    return (x.double() - x0) / (x1 - x0) * nx, (y.double() - y0) / (y1 - y0) * ny


def relmap_design(relmap, x, y):
    """the design matrix [m, (ny + 3) (nx + 3)] of the map at x, y (row-major over the (ny + 3) x (nx + 3) coefficient grid)"""
    nx, ny = (int(v) for v in relmap['n'])
    tx, ty = relmap_knot_coords(relmap, x, y)
    Bx, By = bspline_basis(tx, nx), bspline_basis(ty, ny)
    return (By[:, :, None] * Bx[:, None, :]).reshape(len(x), -1)


def relmap_contrast(relmap, x, y):
    """the angle contrast h(x, y) = (A_0 - A_1) / mean A of a pattern intensity map at movie positions x, y [m] (torch; positions
    outside the map's box are clamped to its edge; h clamped to +-1.5)"""
    if relmap.get('kind') != RELMAP_KIND:
        raise ValueError(f'pattern intensity map of kind {relmap.get("kind")!r}: only {RELMAP_KIND!r} is supported (the Legendre map of '
                         'the first 2026-11-19 version was replaced); re-run the pattern stage (--redo 1)')
    nx, ny = (int(v) for v in relmap['n'])
    tx, ty = relmap_knot_coords(relmap, x, y)
    c = torch.as_tensor(relmap['coef'], device=x.device, dtype=torch.float64).reshape(ny + 3, nx + 3)
    return ((bspline_basis(ty, ny) @ c) * bspline_basis(tx, nx)).sum(1).clamp(-1.5, 1.5).float()


def relmap_factors(relmap, x, y):
    """per-row factors g [n, 2] of the relative intensity of the angle-0 / angle-1 frames at movie positions x, y [n]:
    exp(+- delta / 2), delta = log((1 + h / 2) / (1 - h / 2)) (A_0 / A_1 = e^delta; g_0 g_1 = 1, geometric mean 1, the brightness B
    keeps its meaning). relmap None -> None"""
    if relmap is None: return None
    h = relmap_contrast(relmap, x, y)
    d = torch.log((1 + h / 2) / (1 - h / 2))
    return torch.stack([torch.exp(d / 2), torch.exp(-d / 2)], -1)


def scale_rel(pt, g):
    """per-row pattern pt [n,T,5] with the relint of every frame multiplied by the factor g [n,2] of its angle (g None: pt itself)"""
    if g is None: return pt
    f = torch.gather(g.to(pt.dtype), 1, angle_index(pt))
    return torch.cat([pt[..., :4], (pt[..., 4] * f)[..., None]], -1)


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
