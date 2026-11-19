"""
Pattern estimation from pass-1 (pattern-free) localizations and per-frame intensities alone.

Model (frame t, step s = t % 6, angle a = s // 3):
    I_t = B rel_s (1 + m_a cos(k_a.r - psi_s - eps_a(t)))
Unknowns: k_a (angle + pitch), per-step phases psi_s, depth m_a, per-step relative intensity rel_s (incl. the
angle ratio; geometric mean 1), and a shared per-angle phase drift eps_a(t) (linear interpolation on 150-frame knots;
mean removed = gauge, absorbed in psi). Coordinates: movie pixels. Output = the common pattern file format
(k, phi = psi, m, logrel, dphi = eps, knot_frames), read by simcode.processing.common.pattern_frames.

Inputs per localization (pass-1 'out' rows): window start w, position r (pattern-free), per-frame linear intensities
I_j (frame w + j), photons, sigma; plus all detections of each window for the neighbour distance.

Selection ('clean' emitter-windows, pattern-free):
  - on through the window: the same emitter (< 1 px) is detected in the windows w - 6 and w + 6 (photon-disjoint
    neighbours, so the event spans the whole window) and all I_j > 0;
  - isolated: no other detection of the same window within `nn_min` px (default 3; crowding dilutes depth);
  - photons >= `min_phot`.
Positions are averaged over the linked windows w - 6, w, w + 6 (static site within 12 frames) to reduce the
position-noise phase scatter (affects only the k / phase / eps statistics, not any output localization).

Steps (as SIMCODE's pattern_estimator: DFT of localizations weighted by intensities, then phases, then drift):
  1. k_a: per step s, weights y_s - <y_s> with y_s = I_s / mean_6(I) (relative modulation, structure removed);
     power sum_{s in a} |sum_n w_ns exp(-i q.r_n)|^2, incoherent over 3000-frame blocks (tolerates eps drift);
     coarse peak from the FFT of an 8x super-resolved render inside the pitch annulus, then L-BFGS refinement of the
     direct DFT power over all clean localizations. Sign: largest |component| positive.
  2. psi_s, rel_s: weighted regression per step y_s = r_s + U_s cos th + V_s sin th, th = k_a.r - eps_a(t),
     psi_s = atan2(V_s, U_s) (position noise shrinks (U, V) but does not rotate it); rel from r_s.
  3. eps_a(t): per 150-frame block, weighted circular mean of Z_n exp(-i k_a.r_n), Z_n = exact 3-step solve of
     the angle's intensities with the current psi / rel (Z = A m exp(i th)); weights 1 / var(phase);
     3-block running mean, mean removed (into psi).
  4. m_a: unbiased cross-cycle estimator on pairs of photon-disjoint consecutive windows (w, w + 6) of the same clean
     emitter: m^2 = sum Re(Z_w conj(Z_w+6) exp(i (eps diff))) / sum A_w A_w+6 (independent noise, position-error free).
  2-4 are iterated (`n_iter`).
  5. (2026-11-19; always on) the per-orientation intensity map over the field of view (estimate_relmap): the
     relative intensity of the two orientations varies over the field (illumination envelope per orientation), seen as a smooth
     map of the angle contrast of the clean windows; stored as P['relmap'] (2D cubic B-spline, knot spacing RELMAP_KNOT_PX) and
     applied per row by every solve (o['relg'], simcode/solve/pattern.relmap_factors; g_0 g_1 = 1).
  6. brightness noise (2026-11-17; estimate_brightness_noise): c (the relative per-frame scatter of the emitter brightness) of
     the solve's intensity covariance Sigma = Sigma_net + c^2 diag(mu^2), from the residuals of the network intensities of the
     same clean emitter-windows against the estimated pattern with its intensity map (simcode/solve/brightness_noise_fit.py:
     the plain residual fit of the solve's noise model); stored in the pattern (cmult) and used by every solve of the run.
     (Removed 2026-11-19: the per-orientation term c_a (cmult_ang) and the component estimator with per-model network-transfer
     tables; the map + one c + the full Sigma_net give the same precision and FRC, brightness_noise_fit.)
"""
import math
import numpy as np, torch
from scipy.spatial import cKDTree
from simcode.util import spatial_hash as SH

# (2026-11-11, speed branch) neighbour queries: 'hash' = simcode/util/spatial_hash (torch, CUDA if available, else CPU;
# exact distances, same selections as cKDTree up to the choice among equidistant points), 'kdtree' = the original serial
# scipy cKDTree (reference). NB_DEVICE: None = cuda if available.
NB_BACKEND = 'hash'
NB_DEVICE = None
KD_WORKERS = 1       # 'kdtree' backend only (1 = the original serial queries; -1 = all cores, identical results)
NN_CAP = 5.0         # 'hash': nn beyond max(NN_CAP, nn_min) is returned as inf; every use of nn is a test nn >= t with t <= 5
NB_STATS = {}

T = 6


def _interp_eps(eps_knots, kf, f):
    """eps_knots [2,K] on frames j*kf -> eps at frames f [...] per angle (linear, last value held)."""
    tt = f / kf; i0 = np.clip(np.floor(tt).astype(int), 0, eps_knots.shape[1] - 2); w = np.clip(tt - i0, 0, 1)
    return np.stack([eps_knots[a, i0] * (1 - w) + eps_knots[a, i0 + 1] * w for a in (0, 1)])


def select(R, nn_min=3.0, min_phot=300.0, link_r=1.0):
    """clean emitter-windows from pass-1 rows R (all detections, flag out). Returns dict of arrays."""
    ws = R['wstart'].astype(np.int64); x, y = R['x'].astype(np.float64), R['y'].astype(np.float64)
    out = R['out'].astype(bool)
    io = np.nonzero(out)[0]
    if NB_BACKEND == 'hash':
        # nearest OTHER detection of the same window, capped (see NN_CAP), and the nearest detection < link_r in w -/+ 6
        nn, _ = SH.nearest(x, y, ws, x[io], y[io], ws[io], max(NN_CAP, nn_min), qself=io, device=NB_DEVICE, stats=NB_STATS)
        q = lambda dt: SH.nearest(x, y, ws, x[io], y[io], ws[io] + dt, link_r, device=NB_DEVICE, stats=NB_STATS)
    else:
        # neighbour distance: all detections of the same window (window starts are >= 3 frames apart -> scale 1e4)
        P_all = np.c_[x, y, ws * 1e4]
        tr = cKDTree(P_all)
        dd, _ = tr.query(P_all[io], k=2, workers=KD_WORKERS)
        nn = dd[:, 1]
        # linked detections in windows w - 6 and w + 6 (any detection, nearest < link_r)
        q = lambda dt: tr.query(np.c_[x[io], y[io], (ws[io] + dt) * 1e4], distance_upper_bound=link_r, workers=KD_WORKERS)
    dm, jm = q(-6); dp, jp = q(6)
    on = np.isfinite(dm) & np.isfinite(dp)
    jm = np.where(on, jm, 0); jp = np.where(on, jp, 0)
    I = R['I'][io].astype(np.float64)
    ok = on & (I > 0).all(1) & (R['phot'][io] >= min_phot)
    xm = np.where(on, (x[io] + x[jm] + x[jp]) / 3, x[io]); ym = np.where(on, (y[io] + y[jm] + y[jp]) / 3, y[io])
    s_r = 0.5 * (R['sx'][io] + R['sy'][io]).astype(np.float64)
    s_m = np.where(on, s_r / math.sqrt(3), s_r)
    D = dict(w=ws[io], x=xm, y=ym, x1=x[io], y1=y[io], I=I, phot=R['phot'][io].astype(np.float64), nn=nn, s=s_m, on=ok)
    D['clean'] = ok & (nn >= nn_min)
    return D


# ------------------------------------------------------------------------------------------------- step 1: k
def _render_fft_peak(x, y, w, H, W, qmin, qmax, zoom=8):
    """coarse peak of |FFT|^2 of the weighted render (summed over the weight columns) in the annulus qmin..qmax."""
    dev = 'cuda'
    Hs, Ws = H * zoom, W * zoom
    ix = np.clip((x * zoom).astype(int), 0, Ws - 1); iy = np.clip((y * zoom).astype(int), 0, Hs - 1)
    pw = torch.zeros(Hs, Ws, device=dev)
    for c in range(w.shape[1]):
        img = torch.zeros(Hs * Ws, device=dev)
        img.index_add_(0, torch.tensor(iy * Ws + ix, device=dev), torch.tensor(w[:, c], device=dev).float())
        pw += torch.fft.fft2(img.view(Hs, Ws)).abs() ** 2
    fy = torch.fft.fftfreq(Hs, d=1 / zoom, device=dev) * 2 * math.pi; fx = torch.fft.fftfreq(Ws, d=1 / zoom, device=dev) * 2 * math.pi
    QY, QX = torch.meshgrid(fy, fx, indexing='ij'); Q = torch.hypot(QX, QY)
    pw = torch.where((Q > qmin) & (Q < qmax) & (QX >= 0), pw, torch.zeros_like(pw))
    i = int(pw.argmax())
    return float(QX.reshape(-1)[i]), float(QY.reshape(-1)[i])


def _dft_power(q, X, Y, Wt, blk, nblk):
    """sum over blocks and columns of |sum_n Wt_n exp(-i q.r_n)|^2 (torch, differentiable in q)."""
    ph = q[0] * X + q[1] * Y
    c, s = torch.cos(ph), torch.sin(ph)
    re = torch.zeros(nblk, Wt.shape[1], device=X.device, dtype=X.dtype).index_add_(0, blk, Wt * c[:, None])
    im = torch.zeros(nblk, Wt.shape[1], device=X.device, dtype=X.dtype).index_add_(0, blk, Wt * s[:, None])
    return (re ** 2 + im ** 2).sum()


def estimate_k(D, shape, pitch_px=(1.6, 2.8), block=3000, verbose=print):
    sel = D['on'] & (D['nn'] >= 2.0)
    x, y, w = D['x'][sel], D['y'][sel], D['w'][sel]
    I = D['I'][sel]; yn = I / I.mean(1, keepdims=True)
    st = (w[:, None] + np.arange(T)[None]) % 6
    K = np.zeros((2, 2)); info = {}
    qmin, qmax = 2 * math.pi / pitch_px[1], 2 * math.pi / pitch_px[0]
    blk = (w // block).astype(np.int64); nblk = int(blk.max()) + 1
    for a in (0, 1):
        Wc = np.stack([np.take_along_axis(yn, np.argmax(st == s, 1)[:, None], 1)[:, 0] for s in range(3 * a, 3 * a + 3)], 1)
        Wc = Wc - Wc.mean(0, keepdims=True)
        qx, qy = _render_fft_peak(x, y, Wc, shape[0], shape[1], qmin, qmax)
        dev = 'cuda'
        X = torch.tensor(x, device=dev); Y = torch.tensor(y, device=dev); Wt = torch.tensor(Wc, device=dev)
        bt = torch.tensor(blk, device=dev)
        # local grid around the FFT peak (FFT bin ~ 2 pi / FOV), then L-BFGS on the direct DFT power
        g = np.linspace(-0.06, 0.06, 25)
        best = (-1, qx, qy)
        with torch.no_grad():
            for dx_ in g:
                for dy_ in g:
                    v = float(_dft_power(torch.tensor([qx + dx_, qy + dy_], device=dev, dtype=torch.float64), X, Y, Wt, bt, nblk))
                    if v > best[0]: best = (v, qx + dx_, qy + dy_)
        q = torch.tensor([best[1], best[2]], device=dev, dtype=torch.float64, requires_grad=True)
        p0 = float(_dft_power(q.detach(), X, Y, Wt, bt, nblk))
        opt = torch.optim.LBFGS([q], lr=1.0, max_iter=100, tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad(); v = -_dft_power(q, X, Y, Wt, bt, nblk) / p0; v.backward(); return v
        opt.step(closure)
        kq = q.detach().cpu().numpy()
        if kq[np.abs(kq).argmax()] < 0: kq = -kq
        K[a] = kq
        info[a] = dict(n=int(len(x)), fft_peak=[qx, qy], pitch_px=float(2 * math.pi / np.hypot(*kq)),
                       angle_deg=float(np.degrees(np.arctan2(kq[1], kq[0]))))
        verbose(f'  k angle {a}: ({kq[0]:.5f}, {kq[1]:.5f}) rad/px, pitch {info[a]["pitch_px"]:.4f} px, '
                f'angle {info[a]["angle_deg"]:.3f} deg  (FFT coarse {qx:.3f}, {qy:.3f}; n {len(x)})')
    return K, info


# ------------------------------------------------------------------------------------ steps 2-4: phases etc.
def _design(D, sel, K, eps_knots, kf):
    w = D['w'][sel]; f = w[:, None] + np.arange(T)[None]; st = f % 6; ang = st // 3
    E = _interp_eps(eps_knots, kf, f.astype(np.float64))                  # [2,n,T]
    eps = np.where(ang == 0, E[0], E[1])
    th = K[ang, 0] * D['x'][sel][:, None] + K[ang, 1] * D['y'][sel][:, None] - eps
    return st, ang, th


def fit_steps(D, sel, K, eps_knots, kf):
    """per-step weighted regression y_s = r_s + U cos th + V sin th -> psi_s, rel_s, amp_s (diluted depth)."""
    st, ang, th = _design(D, sel, K, eps_knots, kf)
    I = D['I'][sel]; yn = I / I.mean(1, keepdims=True)
    wt = np.clip(D['phot'][sel], 0, 4000)[:, None].repeat(T, 1)
    psi = np.zeros(6); rel = np.zeros(6); amp = np.zeros(6)
    for s in range(6):
        m_ = st == s
        Y = yn[m_]; th_ = th[m_]; ww = wt[m_]
        X = np.stack([np.ones_like(th_), np.cos(th_), np.sin(th_)], 1)
        for it in range(2):
            sw = np.sqrt(ww)
            c, *_ = np.linalg.lstsq(X * sw[:, None], Y * sw, rcond=None)
            res = Y - X @ c
            rs = 1.4826 * np.median(np.abs(res))
            ww = np.where(np.abs(res) < 4 * rs, wt[m_], 0.0)
        rel[s], psi[s], amp[s] = c[0], math.atan2(c[2], c[1]), math.hypot(c[1], c[2]) / c[0]
    rel = rel / np.exp(np.log(rel).mean())
    return psi, rel, amp


def angle_Z(I, st, psi, rel, a):
    """exact 3-step solve per row for angle a: I_j / rel_j = A + U cos psi_j + V sin psi_j -> A, Z = U + iV."""
    cols = np.stack([np.argmax(st == s, 1) for s in range(3 * a, 3 * a + 3)], 1)
    Ic = np.take_along_axis(I, cols, 1) / rel[3 * a:3 * a + 3][None]
    ps = psi[3 * a:3 * a + 3]
    M = np.stack([np.ones(3), np.cos(ps), np.sin(ps)], 1)
    coef = np.linalg.solve(M, Ic.T).T
    return coef[:, 0], coef[:, 1] + 1j * coef[:, 2], cols


def fit_eps(D, sel, K, psi, rel, m, kf=150, n_knots=None, smooth=3, verbose=print):
    """per-angle eps at 150-frame block centres -> knots (frames j*kf), mean removed. Returns eps_knots [2,K], se."""
    w = D['w'][sel]; f = w[:, None] + np.arange(T)[None]; st = f % 6
    I = D['I'][sel]
    nk = n_knots or int(f.max() // kf) + 2
    eps_k = np.zeros((2, nk)); se_b = np.zeros((2, nk - 1)); nb_ = np.zeros((2, nk - 1))
    knots = np.arange(nk) * kf; cent = (np.arange(nk - 1) + 0.5) * kf
    for a in (0, 1):
        A, Z, cols = angle_Z(I, st, psi, rel, a)
        ft = np.take_along_axis(f, cols, 1).mean(1)
        kn = np.hypot(*K[a])
        u = Z * np.exp(-1j * (K[a, 0] * D['x'][sel] + K[a, 1] * D['y'][sel]))            # ~ A m exp(-i eps)
        u = u / np.maximum(np.abs(u), 1e-9)
        var = kn ** 2 * D['s'][sel] ** 2 + 2.0 / (max(m[a], 0.3) ** 2 * np.maximum(A, 1.0))
        wt = np.where(A > 0, 1.0 / var, 0.0)
        b = np.clip((ft // kf).astype(int), 0, nk - 2)
        S = np.bincount(b, weights=wt * u.real, minlength=nk - 1) + 1j * np.bincount(b, weights=wt * u.imag, minlength=nk - 1)
        Wb = np.bincount(b, weights=wt, minlength=nk - 1); W2 = np.bincount(b, weights=wt ** 2, minlength=nk - 1)
        cnt = np.bincount(b, minlength=nk - 1)
        R_ = np.abs(S) / np.maximum(Wb, 1e-12)
        e = -np.angle(S)
        good = cnt >= 20
        e = np.unwrap(np.where(good, e, np.interp(np.arange(nk - 1), np.nonzero(good)[0], e[good])))
        # circular se of the weighted mean (approx.): sqrt(sum w^2 (1 - R^2)/2) / (R W)
        se = np.sqrt(np.maximum(W2 * (1 - np.minimum(R_, 0.999) ** 2) / 2, 0)) / np.maximum(R_ * Wb, 1e-12)
        if smooth > 1:
            e = np.convolve(np.pad(e, smooth // 2, mode='edge'), np.ones(smooth) / smooth, mode='valid')
        e = e - np.average(e, weights=np.maximum(cnt, 1))
        eps_k[a] = np.interp(knots, cent, e)
        se_b[a] = se; nb_[a] = cnt
        ok = good
        verbose(f'  eps angle {a}: blocks {ok.sum()}/{nk - 1}, median n {np.median(cnt[ok]):.0f}, rms {np.std(e[ok]):.3f} rad '
                f'({np.std(e[ok]) / kn * 108.33:.2f} nm), median se {np.median(se[ok]):.3f} rad')
    return eps_k, dict(se=se_b, n=nb_, block_centres=cent)


def depth_pair_query(D, idx):
    """for each clean emitter-window idx: the nearest clean detection (< 1 px) of the same site in window w + 6 -> (d, j)
    with j indexing idx (len(idx) if none)"""
    if NB_BACKEND == 'hash':
        x, y, w = D['x1'][idx], D['y1'][idx], D['w'][idx]
        return SH.nearest(x, y, w, x, y, w + 6, 1.0, device=NB_DEVICE, stats=NB_STATS)
    P = np.c_[D['x1'][idx], D['y1'][idx], D['w'][idx] * 1e4]
    return cKDTree(P).query(np.c_[D['x1'][idx], D['y1'][idx], (D['w'][idx] + 6) * 1e4], distance_upper_bound=1.0, workers=KD_WORKERS)


def fit_depth(D, sel, K, psi, rel, eps_knots, kf, verbose=print, pbins=(0, 1000, 2000, 4000, 1e9)):
    """cross-cycle depth per angle on photon-disjoint consecutive windows (w, w + 6) of the same clean emitter."""
    idx = np.nonzero(sel)[0]
    d, j = depth_pair_query(D, idx)
    ok = np.isfinite(d)
    i1, i2 = idx[ok], idx[j[ok]]
    res = {}
    for a in (0, 1):
        out = []
        for ii in (i1, i2):
            w = D['w'][ii]; f = w[:, None] + np.arange(T)[None]; st = f % 6
            A, Z, cols = angle_Z(D['I'][ii], st, psi, rel, a)
            ft = np.take_along_axis(f, cols, 1).mean(1)
            e = _interp_eps(eps_knots, kf, ft)[a]
            out.append((A, Z * np.exp(1j * e)))                                      # remove the drift rotation
        (A1, Z1), (A2, Z2) = out
        cross = np.real(Z1 * np.conj(Z2)); aa = A1 * A2
        g = (A1 > 0) & (A2 > 0)
        m2 = cross[g].sum() / aa[g].sum()
        res[a] = dict(m=float(np.sqrt(max(m2, 0))), n=int(g.sum()))
        ph = D['phot'][i1]
        for lo, hi in zip(pbins[:-1], pbins[1:]):
            b = g & (ph >= lo) & (ph < hi)
            if b.sum() > 50:
                res[a][f'm_{int(lo)}_{int(min(hi, 1e6))}'] = float(np.sqrt(max(cross[b].sum() / aa[b].sum(), 0)))
        # block bootstrap se (blocks of 1500 frames)
        bl = (D['w'][i1][g] // 1500).astype(int); ub = np.unique(bl); rng = np.random.default_rng(0)
        cb = np.bincount(np.searchsorted(ub, bl), weights=cross[g]); ab = np.bincount(np.searchsorted(ub, bl), weights=aa[g])
        bs = [np.sqrt(max(cb[k_].sum() / ab[k_].sum(), 0)) for k_ in (rng.integers(0, len(ub), len(ub)) for _ in range(200))]
        res[a]['se'] = float(np.std(bs))
        verbose(f'  depth angle {a}: {res[a]["m"]:.3f} +- {res[a]["se"]:.3f} (n pairs {res[a]["n"]})  ' +
                ' '.join(f'{k_[2:]}:{v:.3f}' for k_, v in res[a].items() if k_.startswith('m_')))
    return np.array([res[0]['m'], res[1]['m']]), res


RELMAP_PHB = (0, 450, 600, 800, 1050, 1400, 1900, 2600, 3600, 1e9)    # window-photon bins of the contrast centring
RELMAP_KNOT_PX = 13.75   # knot spacing of the intensity-map B-spline (movie px; n = round(box / spacing) intervals per axis)
RELMAP_SMOOTH = 1e-2     # second-difference penalty of the B-spline coefficients (P-spline), relative to the mean diagonal of the
                         # weighted normal matrix (independent of the number of windows); both by tile cross-validation


def relmap_grid(box, knot_px=RELMAP_KNOT_PX):
    """knot intervals (nx, ny) of the intensity-map B-spline over box (x0, x1, y0, y1): round(side / knot_px), at least 1"""
    return [max(1, int(round((box[1] - box[0]) / knot_px))), max(1, int(round((box[3] - box[2]) / knot_px)))]


def relmap_penalty(n):
    """D^T D of the second differences of the (ny + 3) x (nx + 3) coefficient grid (row-major, row = y) along x and along y"""
    mx, my = n[0] + 3, n[1] + 3
    dx = np.diff(np.eye(mx), 2, axis=0); dy = np.diff(np.eye(my), 2, axis=0)
    Dm = np.r_[np.kron(np.eye(my), dx), np.kron(dy, np.eye(mx))]
    return Dm.T @ Dm


def _relmap_X(box, n, x, y):
    from simcode.solve import pattern as C
    rm = dict(box=list(box), n=list(n))
    return C.relmap_design(rm, torch.as_tensor(np.asarray(x, np.float64)), torch.as_tensor(np.asarray(y, np.float64))).numpy()


def _robust_wls(X, y, w, n_irls=4, huber=2.5, pen=None):
    """robust (Huber IRLS) weighted LS; pen: optional penalty matrix, scaled by the mean diagonal of X^T W X"""
    ww = w.copy()
    for _ in range(n_irls):
        A = X.T @ (X * ww[:, None]); tr = np.trace(A) / len(A); A += 1e-9 * tr * np.eye(len(A))
        if pen is not None: A += pen * tr
        c = np.linalg.solve(A, X.T @ (ww * y))
        r = (y - X @ c) * np.sqrt(w); s = 1.4826 * np.median(np.abs(r)); a = np.abs(r) / max(s, 1e-12)
        ww = w * np.where(a > huber, huber / a, 1.0)
    return c


def estimate_relmap(D, sel, psi, rel, knot_px=RELMAP_KNOT_PX, smooth=RELMAP_SMOOTH, box=None, verbose=print):
    """(2026-11-19) per-orientation intensity map of the pattern: the angle contrast h = (A_0 - A_1) / mean A of the clean
    emitter-windows (A_a = the exact 3-step solve of angle a with the global psi / rel; position-free) as a 2D cubic B-spline
    (simcode/solve/pattern.relmap_design: uniform knots, spacing ~knot_px, relmap_grid) over the box (x0, x1, y0, y1) (movie px;
    default the extent of the windows), robust weighted LS with a second-difference penalty `smooth` on the coefficients
    (P-spline; relmap_penalty), h centred per window-photon bin (the networks' photon-dependent
    contrast bias), weights 1 / (a / photons + v_s) (fitted per bin), map weighted mean zero (the global orientation ratio stays in
    rel). The solves use rel_s x g_a, g_0 = exp(delta / 2), g_1 = exp(-delta / 2), delta = log((1 + h / 2) / (1 - h / 2))
    (simcode/solve/pattern.relmap_factors): g_0 g_1 = 1, so the per-window brightness B keeps its meaning.
    Measured (out/pattern_map, 2026-11-19; 1809_2 / 2.5 nM_2, v3 models; with the first, Legendre degree-4 version of the map): the
    map holds ~40 % of the h variance (held-out tiles), is stable over the movie (time thirds: map correlation >= 0.988) and
    matches the contrast map of the networks' per-frame background (correlation 0.93 / 0.94, an emitter-free measurement of the
    illumination); with it c_a 0.219 -> 0.109 / 0.230 -> 0.077 (component estimator); the positions of the clean windows move by
    0.5 / 3 nm rms, the photon-disjoint pair precision changes by < 0.7 %. Held-out v3 sims (no envelope): flat map, no gain.
    The B-spline replaced the Legendre polynomial (out/model_studies/relmap, 2026-11-19; held-out share of the h variance, mean of
    the checkerboard 11 / 22 px and random 5-fold tile cross-validations, 1809_2 / 2.5 nM_2): Legendre degree 4 0.394 / 0.405
    (best degree 0.405 / 0.405); unpenalised B-splines at best 0.398 / 0.406 (3 / 2 intervals) and unstable on finer grids; the
    P-spline at 13.75 px, smooth 1e-2 0.423 / 0.429 (flat optimum: 9-28 px with smooth 1e-3 - 1e-1 within 0.01). Pipeline
    check (pattern stage, component estimator): c_a 0.109 -> 0.099 / 0.077 -> 0.059 (no map: 0.219 / 0.230), the code's own
    checkerboard gain 0.402 -> 0.428 / 0.416 -> 0.437. Maps fitted on the windows below / above the median network z agree
    (correlation 0.997 / 0.993): 2D, no height dependence (the two orientations' TIRF decay lengths differ by < 1 nm).
    One map per orientation (per-step maps have no further held-out gain).
    Returns (relmap dict for the pattern file, info)."""
    w0 = D['w'][sel]; f = w0[:, None] + np.arange(T)[None]; st = f % 6
    A0, _, _ = angle_Z(D['I'][sel], st, psi, rel, 0); A1, _, _ = angle_Z(D['I'][sel], st, psi, rel, 1)
    Am = 0.5 * (A0 + A1); x, y, ph = D['x'][sel], D['y'][sel], D['phot'][sel]
    ok = (Am > 0) & np.isfinite(A0) & np.isfinite(A1)
    h = np.where(ok, (A0 - A1) / np.where(Am > 0, Am, 1.0), 0.0)
    ok &= np.abs(h) < 2
    pb = np.clip(np.searchsorted(RELMAP_PHB, ph) - 1, 0, len(RELMAP_PHB) - 2)
    hc = h.copy(); var_b = []; ph_b = []
    for b in range(len(RELMAP_PHB) - 1):
        m = ok & (pb == b)
        if m.sum() >= 50:
            hc[m] -= np.median(h[m]); var_b.append((1.4826 * np.median(np.abs(hc[m]))) ** 2); ph_b.append(np.median(ph[m]))
    if len(var_b) >= 2:          # var(h) ~ a / photons + v_s over the photon bins
        Xv = np.c_[1.0 / np.array(ph_b), np.ones(len(ph_b))]
        a_, vs = np.linalg.lstsq(Xv, np.array(var_b), rcond=None)[0]; a_ = max(a_, 1e-6); vs = max(vs, 1e-3)
    else:
        a_, vs = 1.0, 1e-2
    w = 1.0 / (a_ / np.maximum(ph, 1.0) + vs)
    if box is None: box = (float(x[ok].min()), float(x[ok].max()), float(y[ok].min()), float(y[ok].max()))
    n = relmap_grid(box, knot_px)
    X = _relmap_X(box, n, x, y); pen = smooth * relmap_penalty(n) if smooth > 0 else None
    c = _robust_wls(X[ok], hc[ok], w[ok], pen=pen)
    c -= np.average(X[ok] @ c, weights=w[ok])                                 # mean zero (the basis sums to 1)
    # held-out check: 2-fold checkerboard of 11-px tiles (weighted MSE, clipped at 5 robust sd) vs a constant
    tile = ((np.floor((x - box[0]) / 11.0) + np.floor((y - box[2]) / 11.0)) % 2).astype(int)
    def wmse(r, ww):
        s = 1.4826 * np.median(np.abs(r * np.sqrt(ww))); return float(np.mean(np.clip(r * np.sqrt(ww), -5 * s, 5 * s) ** 2))
    rm, r0 = np.zeros(ok.sum()), np.zeros(ok.sum()); Xo, ho, wo, to = X[ok], hc[ok], w[ok], tile[ok]
    for k in (0, 1):
        tr, te = to != k, to == k
        if tr.sum() < 10 * X.shape[1] or te.sum() < 10: rm[te] = ho[te]; r0[te] = ho[te]; continue
        rm[te] = ho[te] - Xo[te] @ _robust_wls(Xo[tr], ho[tr], wo[tr], pen=pen); r0[te] = ho[te] - np.average(ho[tr], weights=wo[tr])
    gain = 1 - wmse(rm, wo) / wmse(r0, wo)
    gx, gy = np.meshgrid(np.linspace(box[0], box[1], 41), np.linspace(box[2], box[3], 41)); M = _relmap_X(box, n, gx.ravel(), gy.ravel()) @ c
    from simcode.solve.pattern import RELMAP_KIND
    relmap = dict(kind=RELMAP_KIND, knot_px=float(knot_px), smooth=float(smooth), n=n, box=[float(b_) for b_ in box], coef=c.tolist(),
                  note='angle contrast h(x, y) = (A0 - A1) / mean A over the box (movie px) as a cubic B-spline with n = (nx, ny) uniform '
                       'knot intervals, coef row-major over the (ny + 3) x (nx + 3) grid (simcode/solve/pattern.relmap_design); '
                       'relint x g_a, g_0 = exp(delta / 2), g_1 = exp(-delta / 2), delta = log((1 + h / 2) / (1 - h / 2)), g_0 g_1 = 1 '
                       '(simcode/solve/pattern.relmap_factors); processing/pattern_estimation.estimate_relmap')
    info = dict(knot_px=float(knot_px), smooth=float(smooth), grid=n, n=int(ok.sum()), heldout_gain=float(gain), h_p1=float(np.percentile(M, 1)),
                h_p99=float(np.percentile(M, 99)), h_rms=float(M.std()), var_a=float(a_), var_static=float(vs), box=relmap['box'])
    verbose(f'  intensity map (cubic B-spline, {n[0]} x {n[1]} knot intervals of ~{knot_px:g} px, {X.shape[1]} parameters, {ok.sum()} '
            f'clean windows): angle contrast over the field p1 / p99 {info["h_p1"]:+.3f} / {info["h_p99"]:+.3f} (rms {info["h_rms"]:.3f}); '
            f'held-out share of the contrast variance (checkerboard 11-px tiles) {gain:.3f}')
    if gain <= 0:
        verbose('  intensity map: no held-out gain (too few clean windows or no field dependence); the map is kept (it is then '
                'close to flat)')
    return relmap, info


def estimate_pattern(R, shape, nn_min=3.0, min_phot=300.0, kf=150, n_iter=3, verbose=print, clean_rows=False):
    """full estimation from pass-1 rows R (dict of concatenated arrays), steps 1-5 (incl. the per-orientation intensity map
    P['relmap'], estimate_relmap; always on since 2026-11-19). Returns (pattern dict, info) (+ the indices into R of the clean
    emitter-windows with clean_rows)."""
    D = select(R, nn_min=nn_min, min_phot=min_phot)
    verbose(f'pattern estimation: {len(D["w"])} localizations, on-through {D["on"].sum()}, clean (nn >= {nn_min}, '
            f'>= {min_phot:.0f} ph) {D["clean"].sum()}')
    K, kinfo = estimate_k(D, shape, verbose=verbose)
    sel = D['clean']
    nk = int((D['w'].max() + T) // kf) + 2
    eps = np.zeros((2, nk)); m = np.array([0.9, 0.9])
    hist = []
    for it in range(n_iter):
        psi, rel, amp = fit_steps(D, sel, K, eps, kf)
        eps_new, einfo = fit_eps(D, sel, K, psi, rel, m, kf=kf, n_knots=nk, verbose=verbose)
        eps = eps_new - eps_new.mean(1, keepdims=True)                     # total drift (Z is solved without eps)
        psi, rel, amp = fit_steps(D, sel, K, eps, kf)
        m, dinfo = fit_depth(D, sel, K, psi, rel, eps, kf, verbose=verbose)
        spacing = [float((psi[3 * a + j + 1] - psi[3 * a + j]) % (2 * math.pi)) for a in (0, 1) for j in (0, 1)]
        verbose(f' iter {it}: psi {np.round(psi, 3).tolist()} spacing {np.round(spacing, 3).tolist()} rel {np.round(rel, 3).tolist()} '
                f'amp(diluted) {np.round(amp, 3).tolist()} depth {np.round(m, 3).tolist()}')
        hist.append(dict(psi=psi.tolist(), rel=rel.tolist(), amp=amp.tolist(), m=m.tolist(), spacing=spacing))
    # depth vs isolation (crowding-dilution diagnostic)
    nn_diag = {}
    for nnm in (2.0, 3.0, 4.0, 5.0):
        s_ = D['on'] & (D['nn'] >= nnm) & (D['phot'] >= min_phot)
        mm, _ = fit_depth(D, s_, K, psi, rel, eps, kf, verbose=lambda *a: None)
        nn_diag[str(nnm)] = mm.tolist()
    verbose(f'  depth vs nn_min: {nn_diag}')
    P = dict(k=torch.tensor(K, dtype=torch.float32), phi=torch.tensor(psi, dtype=torch.float32),
             m=torch.tensor(np.repeat(m, 3), dtype=torch.float32), logrel=torch.tensor(np.log(rel), dtype=torch.float32),
             dphi=torch.tensor(eps, dtype=torch.float32), knot_frames=kf,
             note='estimated from pass-1 pattern-free localizations (simcode/processing/pattern_estimation.py)')
    info = dict(k=kinfo, hist=hist, depth=dinfo, depth_vs_nn=nn_diag, eps_se=einfo['se'].tolist(), eps_n=einfo['n'].tolist(),
                n_clean=int(sel.sum()), n_on=int(D['on'].sum()), n=int(len(D['w'])))
    P['relmap'], info['relmap'] = estimate_relmap(D, sel, psi, rel, verbose=verbose)
    if clean_rows: return P, info, np.nonzero(R['out'].astype(bool))[0][sel]
    return P, info


BN_NMAX = 120000      # clean emitter-windows used for the brightness-noise fit (evenly subsampled over the movie)
BN_NMIN = 2000        # fewer clean emitter-windows: a warning (the estimate gets fewer SNR bins, it is noisier)
BN_NMIN_HARD = 300    # fewer: no estimate; c = BN_FALLBACK_C, with a warning
BN_FALLBACK_C = 0.1   # (the fixed c of the model calibrations before 2026-11-17)


def estimate_brightness_noise(out, R, clean, P, cal, nmax=BN_NMAX, n_iter=3, verbose=print):
    """step 6: brightness noise c of the run from its clean emitter-windows (indices `clean` into the pass-1 rows R of
    load_pass1_rows, which carry each row's chunk / row; at most nmax, evenly over the movie), the estimated pattern P (with
    its intensity map) and the model's pass-1 solve calibration cal (path or dict, without c): the residual fit of the solve's
    noise model (brightness_noise_fit.estimate). -> (c, info)"""
    import json, warnings
    from simcode.solve import brightness_noise_fit as BN
    cal = json.load(open(cal)) if isinstance(cal, str) else dict(cal)
    idx = np.asarray(clean)
    if len(idx) > nmax: idx = idx[np.linspace(0, len(idx) - 1, nmax).round().astype(np.int64)]
    rule = ('clean emitter-windows of the pattern estimation (on through the window, isolated, >= min_phot), non-kinetic solve with '
            'the estimated pattern and its intensity map; residual fit of Sigma_net + c^2 diag(mu^2), simcode/solve/brightness_noise_fit.py')
    info = dict(n_rows=int(len(idx)), n_clean=int(len(clean)), rule=rule)
    if len(idx) < BN_NMIN_HARD:
        msg = (f'brightness noise: only {len(idx)} clean emitter-windows (< {BN_NMIN_HARD}): not estimated, using c {BN_FALLBACK_C:.3f}')
        warnings.warn(msg); verbose('  WARNING ' + msg)
        info.update(c=BN_FALLBACK_C, fallback=True)
        return BN_FALLBACK_C, info
    if len(idx) < BN_NMIN:
        msg = f'brightness noise: only {len(idx)} clean emitter-windows (< {BN_NMIN}): the estimate is noisier'
        warnings.warn(msg); verbose('  WARNING ' + msg)
    by = {int(f0): R['row'][idx[R['chunk'][idx] == f0]] for f0 in np.unique(R['chunk'][idx])}
    rows = BN.chunk_rows(out, P, by)
    c, hist, _ = BN.estimate(rows, cal, n_iter=n_iter, log=verbose)
    q50 = BN.q_table(rows, BN.with_c(cal, c))
    verbose(f'  brightness noise: c {c:.4f} ({len(idx)} clean emitter-windows); q/dof p50 per photon bin '
            + ' '.join(f'{lo:g}:{q:.2f}' for lo, q in q50))
    info.update(c=c, hist=hist, q50_by_photons=q50)
    return c, info
