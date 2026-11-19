"""
The per-orientation intensity map of the pattern (2026-11-19, always on; processing/pattern_estimation.estimate_relmap,
solve/pattern.relmap_factors / scale_rel, the solves' o['relg']). Run:  python tests/test_relmap.py   (or: pytest tests)
CPU or GPU, a few seconds. Checks:
  - the 2D cubic B-spline map (solve/pattern.relmap_contrast / relmap_design: partition of unity, linear maps reproduced exactly,
    clamping at the box edge, no other map kind accepted)
  - relmap_factors: A_0 / A_1 = (1 + h/2) / (1 - h/2), g_0 g_1 = 1 (geometric mean 1), no map -> None; scale_rel scales each frame by its angle's
    factor and returns the pattern itself without factors (the default path is unchanged)
  - estimate_relmap recovers a known field-dependent angle contrast from simulated clean emitter-windows (3-step pattern, Poisson
    noise, per-window brightness, a static per-emitter contrast scatter) to < 0.02 rms over the field, with a held-out gain, mean 0
  - the solve with o['relg'] = the solve with the explicitly scaled per-row pattern (reference and fast solve), and it differs
    from the solve without the map
  - config: pattern.intensity_map (deprecated) defaults to 1; 0 is accepted with a warning and set to 1; other values are refused
"""
import math, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simcode.solve import pattern as C

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 6
K = np.array([[3.034, 0.064], [-0.072, 3.033]])
PSI = np.array([-0.46, 1.64, -2.55, 1.74, -2.45, -0.35])
REL = np.array([1.05, 1.05, 1.05, 0.95, 0.95, 0.95])
BOX = (5.0, 115.0, 5.0, 115.0)


def true_h(x, y):
    u = 2 * (x - BOX[0]) / (BOX[1] - BOX[0]) - 1; v = 2 * (y - BOX[2]) / (BOX[3] - BOX[2]) - 1
    return 0.15 * u - 0.08 * v + 0.06 * (u * u - v * v) + 0.05 * u * v


def lin_map(ax=0.3, ay=0.0, h0=0.0, box=BOX, n=(2, 3)):
    """a B-spline map h = h0 + ax u + ay v (u, v in [-1, 1] over the box): cubic B-splines reproduce linear functions exactly with
    the coefficients at the knots (basis k centred on knot k - 1)"""
    nx, ny = n
    u = 2 * (np.arange(nx + 3) - 1) / nx - 1; v = 2 * (np.arange(ny + 3) - 1) / ny - 1
    coef = h0 + ax * u[None, :] + ay * v[:, None]                       # [ny + 3, nx + 3], row = y
    return dict(kind=C.RELMAP_KIND, knot_px=27.5, n=[nx, ny], box=list(box), coef=coef.ravel().tolist())


def test_factors():
    rm = lin_map(0.0, 0.0, 0.2)
    x = torch.tensor([10.0, 60.0, 500.0]); y = torch.tensor([10.0, 60.0, -3.0])
    g = C.relmap_factors(rm, x, y)
    assert torch.allclose(g[:, 0] / g[:, 1], torch.full((3,), 1.1 / 0.9), atol=1e-5)
    assert torch.allclose(g[:, 0] * g[:, 1], torch.ones(3), atol=1e-6)            # geometric mean 1 at every position
    assert C.relmap_factors(None, x, y) is None
    # basis: partition of unity on the box, C2 kernel = the cubic B-spline of torch-psf pixint
    t = torch.linspace(0, 3, 301, dtype=torch.float64)
    assert torch.allclose(C.bspline_basis(t, 3).sum(1), torch.ones_like(t), atol=1e-12)
    assert abs(float(C.bspline_kernel(torch.tensor(0.0))) - 2 / 3) < 1e-7 and abs(float(C.bspline_kernel(torch.tensor(1.0))) - 1 / 6) < 1e-7
    # a linear map h = 0.3 u (u = 2 (x - x0) / (x1 - x0) - 1); positions outside the box are clamped to the edge
    rm1 = lin_map(0.3, 0.0)
    h = C.relmap_contrast(rm1, torch.tensor([5.0, 60.0, 115.0, 1e4, 32.5]), torch.tensor([60.0, 60.0, 60.0, 60.0, -50.0]))
    assert torch.allclose(h, torch.tensor([-0.3, 0.0, 0.3, 0.3, -0.15]), atol=1e-5), h
    rm2 = lin_map(0.0, -0.2)
    assert torch.allclose(C.relmap_contrast(rm2, torch.tensor([60.0, 60.0]), torch.tensor([5.0, 115.0])), torch.tensor([0.2, -0.2]), atol=1e-5)
    # the design matrix gives the same map
    xx = torch.tensor([7.0, 33.0, 90.0], dtype=torch.float64); yy = torch.tensor([100.0, 12.0, 64.0], dtype=torch.float64)
    hd = C.relmap_design(rm1, xx, yy) @ torch.tensor(rm1['coef'], dtype=torch.float64)
    assert torch.allclose(hd.float(), C.relmap_contrast(rm1, xx, yy), atol=1e-6)
    # only the B-spline map exists (no alias for the Legendre map of the first version)
    try:
        C.relmap_contrast(dict(kind='legendre_angle_contrast', deg=1, box=list(BOX), coef=[0.2, 0.0, 0.0]), x, y)
    except ValueError:
        pass
    else:
        raise AssertionError('a Legendre map was accepted')
    pt = torch.zeros(2, T, 5); pt[..., 0] = torch.tensor([3.0, 3.0, 3.0, 0.1, 0.1, 0.1]); pt[..., 1] = torch.tensor([0.1] * 3 + [3.0] * 3)
    pt[..., 4] = 1.0
    gg = torch.tensor([[2.0, 0.5], [1.0, 1.0]])
    ps = C.scale_rel(pt, gg)
    assert torch.equal(ps[0, :, 4], torch.tensor([2.0, 2.0, 2.0, 0.5, 0.5, 0.5])) and torch.equal(ps[1, :, 4], torch.ones(T))
    assert torch.equal(ps[..., :4], pt[..., :4])
    assert C.scale_rel(pt, None) is pt


def sim_windows(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(5, 115, n); y = rng.uniform(5, 115, n); w = rng.integers(0, 30000, n)
    B = rng.lognormal(math.log(250), 0.5, n)
    hs = true_h(x, y) + 0.08 * rng.standard_normal(n)                # static per-emitter contrast scatter (photoselection)
    d = np.log((1 + hs / 2) / (1 - hs / 2))
    f = w[:, None] + np.arange(T)[None]; st = f % 6; ang = st // 3
    th = K[ang, 0] * x[:, None] + K[ang, 1] * y[:, None] - PSI[st]
    g = np.where(ang == 0, np.exp(d / 2)[:, None], np.exp(-d / 2)[:, None])
    mu = B[:, None] * REL[st] * g * (1 + 0.9 * np.cos(th))
    I = rng.poisson(mu).astype(float) + rng.normal(0, 3.0, mu.shape)
    return dict(w=w, x=x, y=y, x1=x, y1=y, I=I, phot=I.sum(1), on=np.ones(n, bool), clean=(I > 0).all(1))


def test_estimate_relmap():
    from simcode.processing import pattern_estimation as PE
    D = sim_windows()
    rm, info = PE.estimate_relmap(D, D['clean'], PSI, REL, verbose=lambda *a: None)
    gx, gy = np.meshgrid(np.linspace(10, 110, 21), np.linspace(10, 110, 21))
    he = C.relmap_contrast(rm, torch.tensor(gx.ravel()), torch.tensor(gy.ravel())).numpy()
    ht = true_h(gx.ravel(), gy.ravel())
    err = (he - he.mean()) - (ht - ht.mean())                        # the map is mean-free (the global ratio stays in rel)
    assert np.sqrt(np.mean(err ** 2)) < 0.02, np.sqrt(np.mean(err ** 2))
    assert info['heldout_gain'] > 0.2, info
    nx, ny = rm['n']
    assert rm['kind'] == C.RELMAP_KIND and nx == ny == PE.relmap_grid(rm['box'])[0] and len(rm['coef']) == (nx + 3) * (ny + 3)
    # mean zero over the windows (the global orientation ratio stays in rel)
    hw = C.relmap_contrast(rm, torch.tensor(D['x']), torch.tensor(D['y'])).numpy()
    assert abs(hw.mean()) < 0.01, hw.mean()


def _rows(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(10, 50, n); y = rng.uniform(10, 50, n)
    PW = torch.zeros(1, T, 5)
    for t in range(T):
        a = t // 3; PW[0, t] = torch.tensor([K[a, 0], K[a, 1], PSI[t], 0.9, REL[t]])
    th = K[np.arange(T) // 3, 0][None] * x[:, None] + K[np.arange(T) // 3, 1][None] * y[:, None] - PSI[None]
    I = 300 * REL[None] * (1 + 0.9 * np.cos(th)) * (1 + 0.1 * rng.standard_normal((n, T)))
    tt = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    L = torch.diag_embed(tt(np.sqrt(np.maximum(I, 1.0)) + 5.0))
    o = dict(I=tt(I), L=L, C=torch.zeros(n, T, 2, device=DEV), x=tt(x + 0.02 * rng.standard_normal(n)), y=tt(y + 0.02 * rng.standard_normal(n)),
             lsxy=torch.log(torch.full((n, 2), 0.1, device=DEV)))
    return o, PW.to(DEV), torch.zeros(n, dtype=torch.long, device=DEV)


def test_solve_relg():
    from simcode.solve import solve as SV, solve_fast as MF
    o, PW, b = _rows()
    cal = SV.solve_cal(dict(prior_off=[0.0, 0.0], box=0.5, mix_T=1.0), 0.05)
    rm = lin_map(0.25, -0.1, 0.05, box=(10.0, 50.0, 10.0, 50.0))
    g = C.relmap_factors(rm, o['x'], o['y'])
    ptr = C.scale_rel(PW[b], g); bi = torch.arange(len(b), device=DEV)
    for fn in (SV.solve_emitters, lambda *a: MF.solve_emitters_fast(*a, compile='off')):
        a = fn(dict(o, relg=g), PW, b, cal)
        r = fn(o, ptr, bi, cal)
        n0 = fn(o, PW, b, cal)
        for k in ('r', 's_cal', 'q'):
            assert torch.allclose(a[k], r[k], atol=1e-6, rtol=1e-6), k
        assert not torch.allclose(a['q'], n0['q']), 'the map has no effect'


def test_config():
    from simcode.pipeline import acquisition_config as AC
    import warnings
    from simcode.pipeline import localization_pipeline as LP
    assert AC.DEFAULTS['pattern']['intensity_map'] == 1 and 'pattern.intensity_map' not in LP.LATE_DEFAULTS
    c = AC.load(dict(name='t', movie='x.tif', out='o', pattern=dict(intensity_map=1)))
    assert c['pattern']['intensity_map'] == 1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        c = AC.load(dict(name='t', movie='x.tif', out='o', pattern=dict(intensity_map=0)))
    assert c['pattern']['intensity_map'] == 1 and any('intensity_map' in str(x.message) for x in w)
    for bad in (-1, 2.5, True, 4, 11):
        try:
            AC.load(dict(name='t', movie='x.tif', out='o', pattern=dict(intensity_map=bad)))
        except ValueError:
            continue
        raise AssertionError(f'intensity_map {bad!r} accepted')


if __name__ == '__main__':
    for f in (test_factors, test_estimate_relmap, test_solve_relg, test_config):
        f(); print(f.__name__, 'ok', flush=True)
