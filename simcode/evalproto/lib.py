"""Evaluation protocol library (GT-free metrics and their GT validation on punctate sims; docs/EVALUATION.md). CPU only.
Revisit statistic, block-halves FRC variants (std / adj300 / gap300 / gap600), identical-row pairing, NN proxies (own and
doubly-shared; lag-averaged over 12-36 frames with an event-block bootstrap), event linking + event-revisit proxy, GT truth
decomposition (dyn / event / site), fast GT window table / matching for long sims.
Coordinates in px (PX nm / px), f = window start (frames)."""
import numpy as np
from scipy.spatial import cKDTree
PX = 108.33
CROP = (5, 5, 110)


# ------------------------------------------------------------------ FRC (1/7 threshold, 5-nm render)
from simcode.evalproto.frc import frc  # noqa (the one FRC implementation: simcode/evalproto/frc.py)


def frc_split(x, y, fr, blk=300):
    """FRC with alternating 300-frame block halves"""
    h = (fr // blk) % 2 == 0
    return frc(x[h], y[h], x[~h], y[~h])


BIG = 1e4


# ------------------------------------------------------------------ revisit statistic (critic 36 definition, end-guarded)
def revisit(x, y, f, lags=(6, 60, 120, 300, 1200), r=0.5, grid=6, nq=40000, seed=36, fmax=None):
    """P(another row of the same set within r px at window start f + lag), queries = random rows on the grid (f % grid == 0)
    whose f + lag is inside the movie (end guard)."""
    rng = np.random.default_rng(seed)
    g = f % grid == 0; x, y, f = x[g], y[g], f[g]
    fmax = f.max() if fmax is None else fmax
    T = cKDTree(np.c_[x, y, f * BIG])
    out = {}
    for lag in lags:
        cand = np.nonzero(f + lag <= fmax)[0]
        if len(cand) < 100: out[lag] = np.nan; continue
        q = rng.choice(cand, min(nq, len(cand)), replace=False)
        d, _ = T.query(np.c_[x[q], y[q], (f[q] + lag) * BIG], k=1, distance_upper_bound=r)
        out[lag] = float(np.mean(np.isfinite(d)))
    return out


# ------------------------------------------------------------------ FRC variants
def frc_std(x, y, f):
    """standard: alternating 300-frame blocks, all data"""
    return frc_split(x, y, f)


def frc_blocks(x, y, f, period, pairs, blk=300):
    """mean FRC over block pairs (o, o2) of each `period`-block cycle: half A = blocks b % period == o, half B = o2"""
    b = f // blk; r = []
    for o, o2 in pairs:
        a = b % period == o; c = b % period == o2
        if a.sum() < 50 or c.sum() < 50: r.append(np.nan); continue
        r.append(frc(x[a], y[a], x[c], y[c]))
    return float(np.nanmean(r))


def frc_adj300(x, y, f):   # control at the gap300 data volume: adjacent blocks (0 vs 1), (2 vs 3) of each 1200 frames
    return frc_blocks(x, y, f, 4, ((0, 1), (2, 3)))


def frc_gap300(x, y, f):   # blocks 300 frames apart: (0 vs 2), (1 vs 3) of each 1200
    return frc_blocks(x, y, f, 4, ((0, 2), (1, 3)))


def frc_adj600c(x, y, f):  # control at the gap600 data volume (1/3): adjacent blocks of each 1800
    return frc_blocks(x, y, f, 6, ((0, 1), (2, 3), (4, 5)))


def frc_gap600(x, y, f):   # blocks 600 frames apart: (0 vs 3), (1 vs 4), (2 vs 5) of each 1800
    return frc_blocks(x, y, f, 6, ((0, 3), (1, 4), (2, 5)))


FRCS = dict(std=frc_std, adj300=frc_adj300, gap300=frc_gap300, gap600=frc_gap600)


# ------------------------------------------------------------------ pairing / NN proxy
def pair(xa, ya, fa, xb, yb, fb, r=0.5):
    """exact-window-start mutual nearest pairs within r px -> (ia, ib)"""
    ta = cKDTree(np.c_[xa, ya, fa * BIG]); tb = cKDTree(np.c_[xb, yb, fb * BIG])
    d1, j = tb.query(np.c_[xa, ya, fa * BIG], distance_upper_bound=r, workers=-1)
    d2, i = ta.query(np.c_[xb, yb, fb * BIG], distance_upper_bound=r, workers=-1)
    ok = np.isfinite(d1); ia = np.nonzero(ok)[0]; ib = j[ok]
    mu = i[ib] == ia
    return ia[mu], ib[mu]


def nn_next(x, y, f, dt=6, R=50.0):
    """index of the nearest row (same set) at window start f + dt within R nm (-1 if none)"""
    T = cKDTree(np.c_[x, y, f * BIG])
    d, j = T.query(np.c_[x, y, (f + dt) * BIG], distance_upper_bound=R / PX, workers=-1)
    return np.where(np.isfinite(d), j, -1)


def sig1d(v, R=50.0):
    """per-axis sigma from paired 1-D differences v (nm): Gaussian(sqrt2 s) + uniform on |v| < R (ML, binned)"""
    v = v[np.isfinite(v) & (np.abs(v) < R)]
    if len(v) < 50: return np.nan
    h, e = np.histogram(v, bins=np.arange(-R, R + 1e-9, 0.1)); r = 0.5 * (e[1:] + e[:-1])
    sig = np.arange(0.5, 30.0, 0.05); ss = np.sqrt(2) * sig[:, None]
    g = np.exp(-r[None] ** 2 / (2 * ss ** 2)) / (np.sqrt(2 * np.pi) * ss)
    best = (-np.inf, np.nan)
    for w in np.arange(0.3, 1.0001, 0.02):
        ll = (h[None] * np.log(np.minimum(w, 1.0) * g + (1 - min(w, 1.0)) / (2 * R) + 1e-300)).sum(1); k = ll.argmax()
        if ll[k] > best[0]: best = (ll[k], sig[k])
    return float(best[1])


def nn_proxy_shared(A, B, ia, ib, dt=6):
    """doubly-shared NN proxy (realgap decomp protocol): pairs (i, i') of shared rows where both methods have the shared row at
    f + dt that is each one's nearest; sigma per axis from the 1-D differences of each method. Returns (sig_A, sig_B, n)."""
    ka = -np.ones(len(A['x']), np.int64); ka[ia] = np.arange(len(ia))
    kb = -np.ones(len(B['x']), np.int64); kb[ib] = np.arange(len(ib))
    na = nn_next(A['x'], A['y'], A['f'], dt); nb = nn_next(B['x'], B['y'], B['f'], dt)
    pa = na[ia]; pb = nb[ib]
    ok = (pa >= 0) & (pb >= 0)
    ok[ok] &= (ka[pa[ok]] >= 0) & (ka[pa[ok]] == kb[pb[ok]])
    i0 = np.nonzero(ok)[0]
    va = np.r_[(A['x'][pa[i0]] - A['x'][ia[i0]]), (A['y'][pa[i0]] - A['y'][ia[i0]])] * PX
    vb = np.r_[(B['x'][pb[i0]] - B['x'][ib[i0]]), (B['y'][pb[i0]] - B['y'][ib[i0]])] * PX
    return sig1d(va), sig1d(vb), int(len(i0))


# ------------------------------------------------------------------ event linking + event-revisit proxy
def link_events(x, y, f, grid=6, r=0.75):
    """link rows on the grid (f % grid == 0) into events: row -> nearest row at f + grid within r px (mutual), chains.
    Returns event id per row (-1 off-grid)."""
    ev = -np.ones(len(x), np.int64)
    g = np.nonzero(f % grid == 0)[0]
    xg, yg, fg = x[g], y[g], f[g]
    T = cKDTree(np.c_[xg, yg, fg * BIG])
    d, j = T.query(np.c_[xg, yg, (fg + grid) * BIG], distance_upper_bound=r, workers=-1)
    nxt = np.where(np.isfinite(d), j, -1)
    d2, jp = T.query(np.c_[xg, yg, (fg - grid) * BIG], distance_upper_bound=r, workers=-1)
    prv = np.where(np.isfinite(d2), jp, -1)
    ok = nxt >= 0; ok[ok] &= prv[nxt[ok]] == np.nonzero(ok)[0]
    nxt = np.where(ok, nxt, -1)
    has_prev = np.zeros(len(g), bool); has_prev[nxt[nxt >= 0]] = True
    lab = -np.ones(len(g), np.int64)
    heads = np.nonzero(~has_prev)[0]
    lab[heads] = np.arange(len(heads))
    cur = heads.copy()
    while len(cur):
        n_ = nxt[cur]; k = n_ >= 0
        lab[n_[k]] = lab[cur[k]]; cur = n_[k]
    ev[g] = lab
    return ev


def event_means(x, y, f, ev, nmin=2):
    """per linked event: mean x, y, first / last window start, n rows (events with >= nmin rows)"""
    k = ev >= 0
    u, inv, cnt = np.unique(ev[k], return_inverse=True, return_counts=True)
    mx = np.bincount(inv, x[k]) / cnt; my = np.bincount(inv, y[k]) / cnt
    f0 = np.full(len(u), np.iinfo(np.int64).max); np.minimum.at(f0, inv, f[k])
    f1 = np.full(len(u), -1); np.maximum.at(f1, inv, f[k])
    s = cnt >= nmin
    return dict(id=u[s], x=mx[s], y=my[s], f0=f0[s], f1=f1[s], n=cnt[s])


def event_revisit(E, gap=300, H=3000, R=50.0, k=48, chunk=200000, ret_pairs=False):
    """event-revisit proxy: for each linked event (end f1), the nearest OTHER event mean that starts in [f1 + gap, f1 + gap + H]
    within R nm -> 1-D differences (nm) -> sig1d (Gaussian sqrt2 s + uniform). Returns (s = sigma of the event-mean difference
    per axis, n pairs[, (i, j) pair indices]). Vectorised: the k spatial nearest event means, first one meeting the time window."""
    X, Y, F0, F1 = E['x'], E['y'], E['f0'], E['f1']
    T = cKDTree(np.c_[X, Y])
    I, J = [], []
    for a in range(0, len(X), chunk):
        q = np.arange(a, min(a + chunk, len(X)))
        d, j = T.query(np.c_[X[q], Y[q]], k=k, distance_upper_bound=R / PX, workers=-1)
        ok = np.isfinite(d); jj = np.where(ok, j, 0)
        ok &= (F0[jj] >= F1[q, None] + gap) & (F0[jj] < F1[q, None] + gap + H)
        has = ok.any(1); first = ok.argmax(1)
        I.append(q[has]); J.append(jj[has, first[has]])
    I = np.concatenate(I); J = np.concatenate(J)
    v = np.r_[X[J] - X[I], Y[J] - Y[I]] * PX
    s = sig1d(v, R)
    return (s, len(I), (I, J)) if ret_pairs else (s, len(I))


# ------------------------------------------------------------------ sim GT: window table + matching
def gt_windows_fast(G, grid=3, nf=None):
    """(event, window start w % grid == 0) rows with >= 1 on-frame: columns w, ev, x, y, P (pattern-weighted window photons
    without frame_sd, as compare.gt_windows), target flag (photon time centroid in [1, 4), inside the crop)."""
    x, y, br, t0, t1, ea, es, patt = (G[k] for k in ('x', 'y', 'bright', 't0', 't1', 'ea', 'es', 'patt'))
    NF = int(G['nf']) if nf is None else nf
    wlo = np.maximum(np.ceil((t0 - 5) / grid), 0).astype(np.int64) * grid      # first window start with w + 6 > t0
    whi = np.minimum(np.floor(t1 / grid - 1e-9), (NF - 6) // grid).astype(np.int64) * grid
    n = np.maximum((whi - wlo) // grid + 1, 0)
    ev = np.repeat(np.arange(len(x)), n)
    w = wlo[ev] + grid * (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    fr = w[:, None] + np.arange(6)[None]
    onf = np.clip(np.minimum(t1[ev, None], fr + 1) - np.maximum(t0[ev, None], fr), 0, 1)
    pt = patt[fr]
    il = pt[..., 4] * (1 + pt[..., 3] * np.cos(pt[..., 0] * x[ev, None] + pt[..., 1] * y[ev, None] - pt[..., 2]))
    s = fr % 6
    ph = br[ev, None] * il * onf * np.take_along_axis(ea[ev], s // 3, 1) * np.take_along_axis(es[ev], s, 1)
    P = ph.sum(1); tc = (ph * np.arange(6)).sum(1) / np.maximum(P, 1e-6)
    ins = (x[ev] >= CROP[0]) & (x[ev] < CROP[0] + CROP[2]) & (y[ev] >= CROP[1]) & (y[ev] < CROP[1] + CROP[2])
    tgt = ins & (tc >= 1.0) & (tc < 4.0) & (onf.sum(1) >= 1)
    k = onf.sum(1) > 0
    return np.c_[w, ev, x[ev], y[ev], P, tgt][k]


def match_fast(GW, wo, xo, yo, r=1.0, rounds=4):
    """one-to-one matching within r px per window by iterated mutual nearest neighbours (~ greedy); -> gi per output (-1)"""
    gi = -np.ones(len(xo), np.int64)
    go = np.ones(len(GW), bool); oo = np.ones(len(xo), bool)
    for _ in range(rounds):
        io = np.nonzero(oo)[0]; ig = np.nonzero(go)[0]
        if len(io) == 0 or len(ig) == 0: break
        Tg = cKDTree(np.c_[GW[ig, 2], GW[ig, 3], GW[ig, 0] * BIG]); To = cKDTree(np.c_[xo[io], yo[io], wo[io] * BIG])
        d1, j = Tg.query(np.c_[xo[io], yo[io], wo[io] * BIG], distance_upper_bound=r, workers=-1)
        d2, i = To.query(np.c_[GW[ig, 2], GW[ig, 3], GW[ig, 0] * BIG], distance_upper_bound=r, workers=-1)
        ok = np.isfinite(d1); a = np.nonzero(ok)[0]; b = j[ok]
        mu = i[b] == a; a, b = a[mu], b[mu]
        gi[io[a]] = ig[b]; oo[io[a]] = False; go[ig[b]] = False
    return gi


# ------------------------------------------------------------------ GT-free metric bundle (shared by valid.py and realproto.py)
def nn_own(x, y, f, dt=6):
    j = nn_next(x, y, f, dt)
    k = j >= 0
    return sig1d(np.r_[x[j[k]] - x[k], y[j[k]] - y[k]] * PX)


def revisit_proxy(x, y, f):
    ev = link_events(x, y, f, grid=6, r=0.75)
    Em = event_means(x, y, f, ev, nmin=2)
    s, n, (I, J) = event_revisit(Em, gap=300, ret_pairs=True)
    return s / np.sqrt(2), n, float(np.mean(0.5 * (1 / Em['n'][I] + 1 / Em['n'][J]))) if n else np.nan


def metrics(x, y, f, frcs=('std', 'adj300', 'gap300', 'gap600'), extra=True):
    o = {k: FRCS[k](x, y, f) for k in frcs}
    if extra:
        o['nn'] = nn_own(x, y, f)
        o['nn12'] = nn_own(x, y, f, 12)     # context-robust (skips one window): window-to-window error without the lag-6 context term
        r, n, inv_n = revisit_proxy(x, y, f)
        o['rev'] = r; o['n_rev'] = n
        o['rev_st'] = float(np.sign(r ** 2 - o['nn'] ** 2 * inv_n) * np.sqrt(abs(r ** 2 - o['nn'] ** 2 * inv_n)))
    return o


def perturb(x, y, f, a, mode, grid=3):
    if a == 0: return x, y
    if mode == 'iid':
        return x + RNG.normal(0, a / PX, len(x)), y + RNG.normal(0, a / PX, len(y))
    ev = link_events(x, y, f, grid=grid, r=0.75)     # GT-free per-event static: linked events on the set's grid
    u, inv = np.unique(ev, return_inverse=True)
    d = RNG.normal(0, a / PX, (len(u), 2))[inv]
    return x + d[:, 0], y + d[:, 1]



RNG = np.random.default_rng(11)


# ------------------------------------------------------------------ GT truth decomposition (sim)
def truth(M, idx, e=None):
    e = M['e'][idx] if e is None else e
    ev, st, f = M['ev'][idx], M['site'][idx], M['f'][idx]
    g = np.isfinite(e).all(1) & (ev >= 0)
    rms_all = float(np.sqrt(np.nanmean(e[g] ** 2)))
    g &= (np.hypot(e[:, 0], e[:, 1]) < 50)
    e, ev, st, f = e[g], ev[g], st[g], f[g]
    out = dict(n=int(len(e)), rms=float(np.sqrt((e ** 2).mean())), rms_all=rms_all)
    # G6 sub-grid for the covariance decomposition (disjoint windows)
    k6 = f % 6 == 0; e6, ev6, st6, f6 = e[k6], ev[k6], st[k6], f[k6]
    var = float((e6 ** 2).mean())
    u, inv = np.unique(ev6, return_inverse=True)
    S = np.stack([np.bincount(inv, e6[:, a], len(u)) for a in (0, 1)], 1); Q = np.stack([np.bincount(inv, e6[:, a] ** 2, len(u)) for a in (0, 1)], 1)
    n = np.bincount(inv, minlength=len(u)).astype(float)
    # lag-6 pairs explicitly
    o = np.lexsort((f6, ev6)); evs, fs, es = ev6[o], f6[o], e6[o]
    adj = (evs[1:] == evs[:-1]) & (fs[1:] - fs[:-1] == 6)
    c6s = (es[1:][adj] * es[:-1][adj]).sum(0); n6 = adj.sum()
    num = (S ** 2 - Q).sum(0) - 2 * c6s; den = (n * (n - 1)).sum() - 2 * n6
    cov_ev = float(np.mean(num / max(den, 1)))
    cov6 = float(np.mean(c6s / max(n6, 1)))
    # site: pairs of rows of different events at the same site
    ks = st6 >= 0
    us, invs = np.unique(st6[ks], return_inverse=True)
    Ss = np.stack([np.bincount(invs, e6[ks, a], len(us)) for a in (0, 1)], 1); ns = np.bincount(invs, minlength=len(us)).astype(float)
    evsite = st6[ks]; ue, ie = np.unique(ev6[ks], return_inverse=True)
    Se = np.stack([np.bincount(ie, e6[ks, a], len(ue)) for a in (0, 1)], 1); ne = np.bincount(ie, minlength=len(ue)).astype(float)
    num_s = (Ss ** 2).sum(0) - (Se ** 2).sum(0); den_s = (ns ** 2).sum() - (ne ** 2).sum()
    cov_site = float(np.mean(num_s / max(den_s, 1))) if den_s > 100 else np.nan
    sq = lambda v: float(np.sign(v) * np.sqrt(abs(v))) if np.isfinite(v) else np.nan
    out.update(site=sq(cov_site), event=sq(cov_ev - (cov_site if np.isfinite(cov_site) else 0)), dyn=sq(var - cov_ev), ctx=sq(cov6 - cov_ev),
               n_site_pairs=float(den_s))
    return out




# ------------------------------------------------------------------ (critic 37 #4) lag-averaged NN with event-block bootstrap, 0.01-nm grid
LAGS = (12, 18, 24, 30, 36)
_HB = 0.1; _HR = 50.0
_HE = np.arange(-_HR, _HR + 1e-9, _HB); _HC = 0.5 * (_HE[1:] + _HE[:-1])


def _ll(h, sig, w):
    """log-lik of histogram h under w N(0, 2 sig^2) + (1 - w) U(-R, R); sig [ns], w [nw] -> [ns, nw]"""
    ss = np.sqrt(2) * sig[:, None]
    g = np.exp(-_HC[None] ** 2 / (2 * ss ** 2)) / (np.sqrt(2 * np.pi) * ss) * _HB      # per-bin prob mass (Gaussian)
    u = _HB / (2 * _HR)
    nz = h > 0; hn = h[nz]; g = g[:, nz]
    return np.stack([(hn[None] * np.log(wv * g + (1 - wv) * u + 1e-300)).sum(1) for wv in w], 1)


def sig1d_fine(v=None, h=None, s0=None, w0=None):
    """per-axis sigma (0.01-nm grid) and Gaussian weight w (uniform weight 1 - w = tail indicator) from paired 1-D differences
    (or their 0.1-nm histogram h on |v| < 50 nm). Coarse 0.05 nm / 0.02 search, then 0.01 nm / 0.005 refinement; with a start
    (s0, w0) only the local refinement (+-1 nm, +-0.08) is searched (bootstrap replicates)."""
    if h is None:
        v = v[np.isfinite(v) & (np.abs(v) < _HR)]
        h = np.histogram(v, _HE)[0].astype(float)
    if h.sum() < 50: return np.nan, np.nan
    if s0 is None:
        sg = np.arange(0.5, 30.0, 0.05); wg = np.minimum(np.arange(0.3, 1.0001, 0.02), 1.0)
        L_ = _ll(h, sg, wg); k = np.unravel_index(L_.argmax(), L_.shape); s0, w0 = sg[k[0]], wg[k[1]]
        sg = np.arange(max(0.5, s0 - 0.1), s0 + 0.1 + 1e-9, 0.01); wg = np.clip(np.arange(w0 - 0.03, w0 + 0.03 + 1e-9, 0.005), 0.05, 1.0)
    else:
        sg = np.arange(max(0.5, s0 - 1.0), s0 + 1.0 + 1e-9, 0.01); wg = np.unique(np.clip(np.arange(w0 - 0.08, w0 + 0.08 + 1e-9, 0.01), 0.05, 1.0))
    L_ = _ll(h, sg, wg); k = np.unravel_index(L_.argmax(), L_.shape)
    return float(sg[k[0]]), float(wg[k[1]])


def nn_shared_diffs(A, B, ia, ib, dt):
    """doubly-shared NN pairs at lag dt (as nn_proxy_shared) -> (va, vb, rows of A of the pair starts; x and y stacked)"""
    ka = -np.ones(len(A['x']), np.int64); ka[ia] = np.arange(len(ia))
    kb = -np.ones(len(B['x']), np.int64); kb[ib] = np.arange(len(ib))
    na = nn_next(A['x'], A['y'], A['f'], dt); nb = nn_next(B['x'], B['y'], B['f'], dt)
    pa = na[ia]; pb = nb[ib]
    ok = (pa >= 0) & (pb >= 0)
    ok[ok] &= (ka[pa[ok]] >= 0) & (ka[pa[ok]] == kb[pb[ok]])
    i0 = np.nonzero(ok)[0]
    va = np.r_[(A['x'][pa[i0]] - A['x'][ia[i0]]), (A['y'][pa[i0]] - A['y'][ia[i0]])] * PX
    vb = np.r_[(B['x'][pb[i0]] - B['x'][ib[i0]]), (B['y'][pb[i0]] - B['y'][ib[i0]])] * PX
    return va, vb, np.r_[ia[i0], ia[i0]]


def nn_own_diffs(x, y, f, dt, sel=None):
    """own-set NN pairs at lag dt -> (v (x and y stacked, nm), start row index stacked). sel: bool mask, pairs with both rows in sel"""
    j = nn_next(x, y, f, dt); k = j >= 0
    if sel is not None: k &= sel; k[k] &= sel[j[k]]
    i = np.nonzero(k)[0]
    return np.r_[x[j[i]] - x[i], y[j[i]] - y[i]] * PX, np.r_[i, i]


def _bins(v):
    b = np.floor((v + _HR) / _HB).astype(np.int64); ok = np.isfinite(v) & (b >= 0) & (b < len(_HC))
    return np.where(ok, b, -1)


def _hist_w(b, ev_of_pair, wev):
    k = b >= 0
    return np.bincount(b[k], weights=wev[ev_of_pair[k]], minlength=len(_HC))


def lagavg_shared(A, B, ia, ib, grid, lags=LAGS, nboot=200, seed=37):
    """mean over lags of sig_A / sig_B (doubly-shared NN, identical rows) with an event-block (linked events of A's identical
    rows) Poisson bootstrap CI. -> dict ratio, lo, hi, sA, sB (lag means), uA, uB (mean uniform weight 1 - w), per-lag values"""
    rng = np.random.default_rng(seed)
    Ai = {k: A[k][ia] for k in ('x', 'y', 'f')}
    ev_i = link_events(Ai['x'], Ai['y'], Ai['f'], grid=grid, r=0.75)
    ev = -np.ones(len(A['x']), np.int64); ev[ia] = ev_i
    u, evc = np.unique(ev[ia], return_inverse=True); evmap = -np.ones(len(A['x']), np.int64); evmap[ia] = evc
    per = []
    for dt in lags:
        va, vb, ra = nn_shared_diffs(A, B, ia, ib, dt)
        ba, bb = _bins(va), _bins(vb); e = evmap[ra]
        one = np.ones(len(u))
        sa, wa = sig1d_fine(h=_hist_w(ba, e, one)); sb, wb = sig1d_fine(h=_hist_w(bb, e, one))
        per.append(dict(dt=dt, sa=sa, sb=sb, wa=wa, wb=wb, n=len(va) // 2, ba=ba, bb=bb, e=e))
    W = rng.poisson(1.0, (nboot, len(u))).astype(float)
    R = []
    for b in range(nboot):
        rr = []
        for P in per:
            sa, _ = sig1d_fine(h=_hist_w(P['ba'], P['e'], W[b]), s0=P['sa'], w0=P['wa'])
            sb, _ = sig1d_fine(h=_hist_w(P['bb'], P['e'], W[b]), s0=P['sb'], w0=P['wb'])
            rr.append(sa / sb)
        R.append(np.mean(rr))
    R = np.array(R)
    rat = float(np.mean([P['sa'] / P['sb'] for P in per]))
    return dict(ratio=rat, lo=float(np.percentile(R, 2.5)), hi=float(np.percentile(R, 97.5)), sA=float(np.mean([P['sa'] for P in per])),
                sB=float(np.mean([P['sb'] for P in per])), uA=float(1 - np.mean([P['wa'] for P in per])), uB=float(1 - np.mean([P['wb'] for P in per])),
                n=int(np.mean([P['n'] for P in per])), per=[{k: P[k] for k in ('dt', 'sa', 'sb', 'wa', 'wb', 'n')} for P in per], boot=R.tolist())


def lagavg_own(x, y, f, grid, lags=LAGS, nboot=200, seed=37, sel=None):
    """own-set NN, mean over lags of sigma (0.01-nm grid) with an event-block bootstrap (linked events of the set);
    returns dict s (lag mean), u (uniform weight), boot (replicate lag means), per-lag values"""
    rng = np.random.default_rng(seed)
    ev = link_events(x, y, f, grid=grid, r=0.75)
    u_, evc = np.unique(ev, return_inverse=True)
    per = []
    for dt in lags:
        v, r = nn_own_diffs(x, y, f, dt, sel)
        b = _bins(v); e = evc[r]; s, w = sig1d_fine(h=_hist_w(b, e, np.ones(len(u_))))
        per.append(dict(dt=dt, s=s, w=w, n=len(v) // 2, b=b, e=e))
    W = rng.poisson(1.0, (nboot, len(u_))).astype(float)
    Bs = np.array([np.mean([sig1d_fine(h=_hist_w(P['b'], P['e'], W[k]), s0=P['s'], w0=P['w'])[0] for P in per]) for k in range(nboot)])
    return dict(s=float(np.mean([P['s'] for P in per])), u=float(1 - np.mean([P['w'] for P in per])), n=int(np.mean([P['n'] for P in per])),
                per=[{k: P[k] for k in ('dt', 's', 'w', 'n')} for P in per], boot=Bs.tolist())


def ratio_ci(a, b):
    """ratio of two independent lagavg_own results with bootstrap CI"""
    R = np.array(a['boot']) / np.array(b['boot'])
    return dict(ratio=a['s'] / b['s'], lo=float(np.percentile(R, 2.5)), hi=float(np.percentile(R, 97.5)))
