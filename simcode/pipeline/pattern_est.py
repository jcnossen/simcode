"""
Pattern estimation from pass-1 (pattern-free) localizations and per-frame intensities alone.

Model (frame t, step s = t % 6, angle a = s // 3):
    I_t = B rel_s (1 + m_a cos(k_a.r - psi_s - eps_a(t)))
Unknowns: k_a (angle + pitch), per-step phases psi_s, depth m_a, per-step relative intensity rel_s (incl. the
angle ratio; geometric mean 1), and a shared per-angle phase drift eps_a(t) (linear interpolation on 150-frame knots;
mean removed = gauge, absorbed in psi). Coordinates: movie pixels. Output = the common pattern file format
(k, phi = psi, m, logrel, dphi = eps, knot_frames), read by simcode.pipeline.common.pattern_frames.

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


def estimate_pattern(R, shape, nn_min=3.0, min_phot=300.0, kf=150, n_iter=3, verbose=print):
    """full estimation from pass-1 rows R (dict of concatenated arrays). Returns (pattern dict, info)."""
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
             note='estimated from pass-1 pattern-free localizations (simcode/pipeline/pattern_est.py)')
    info = dict(k=kinfo, hist=hist, depth=dinfo, depth_vs_nn=nn_diag, eps_se=einfo['se'].tolist(), eps_n=einfo['n'].tolist(),
                n_clean=int(sel.sum()), n_on=int(D['on'].sum()), n=int(len(D['w'])))
    return P, info
