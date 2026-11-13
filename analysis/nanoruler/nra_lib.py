"""Nanoruler precision analysis library (CPU only).

Method tables: npz with frame, x_nm, y_nm, sx_nm, sy_nm (reported uncertainty per localization), photons and optionally
crlb_x_nm, crlb_y_nm. All coordinates in nm in one common frame (the frame of the per-raw-frame quick fit: pixel centres at
integer pixel coordinates, 65 nm px).

Pipeline: find_sites (binding-site clusters of a reference table) -> build_rulers (80 nm pairs / 3-site lines, crowding,
edge and count rejections; one list for every method) -> eval_method (global shift, fixed-radius assignment, one
re-centring on the method's own median centre, residuals) -> summarize / bootstrap (resampling rulers) -> report / figures.

Conventions used everywhere:
- robust sd (rsd) = 1.4826 x MAD about the median; clipped sd (csd) = plain sd (ddof 1, about the mean) of the values within
  3 rsd of the median (a 3-sigma clip deflates a Gaussian sd by 1.3 %, not corrected).
- residuals are taken about the per-site median of the method's own localizations; a centre estimated from n values
  deflates the residual spread by sqrt(1 - k / n), k = 2 - pi / 2 = 0.43 for a median centre (k = 1 for a mean); residuals
  and per-site robust sds are divided by this factor, and only sites with n >= min_n (default 10) are used.
- "site_*_median" = median over sites of the per-site spread; "pooled_*" = spread of all residuals of all sites pooled.
- bootstrap unit = ruler (its sites and localizations are resampled together), percentile 95 % intervals.
"""
import json, sys, numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter

KMED = 2 - np.pi / 2
FIELDS = ('frame', 'x_nm', 'y_nm', 'sx_nm', 'sy_nm', 'photons')
OPT = ('crlb_x_nm', 'crlb_y_nm')
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


# ------------------------------------------------------------------ basics
def rsd(v):
    v = np.asarray(v, float)
    if len(v) < 2: return np.nan
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def csd(v, k=3.0):
    v = np.asarray(v, float)
    if len(v) < 3: return np.nan
    s = rsd(v); g = np.abs(v - np.median(v)) <= k * s
    return float(np.std(v[g], ddof=1)) if g.sum() > 2 else np.nan


def load_table(fn):
    d = np.load(fn)
    t = {k: np.asarray(d[k], float) for k in FIELDS}
    for k in OPT:
        if k in d.files: t[k] = np.asarray(d[k], float)
    g = np.isfinite(t['x_nm']) & np.isfinite(t['y_nm']) & np.isfinite(t['sx_nm']) & np.isfinite(t['sy_nm'])
    return {k: v[g] for k, v in t.items()}


def save_table(fn, **kw):
    for k in FIELDS: assert k in kw, k
    np.savez_compressed(fn, **{k: np.asarray(v) for k, v in kw.items()})


def merge_tables(tabs):
    keys = set(tabs[0]);
    for t in tabs[1:]: keys &= set(t)
    return {k: np.concatenate([t[k] for t in tabs]) for k in keys}


# ------------------------------------------------------------------ sites
def find_sites(x, y, radius=30.0, px=4.0, blur=8.0, weak_min=5, fov=None):
    """Cluster candidates: local maxima of a blurred px-nm histogram, refined by 3 median-shift iterations within radius;
    duplicates (centres closer than 20 nm) merged. Returns centres [n, 2], n locs within radius, robust sd (mean of x, y)."""
    if fov is None: fov = (min(x.min(), y.min()) - 100, max(x.max(), y.max()) + 100)
    lo, hi = fov; n = int(np.ceil((hi - lo) / px))
    h = np.histogram2d(y, x, bins=n, range=[[lo, lo + n * px], [lo, lo + n * px]])[0]
    hb = gaussian_filter(h, blur / px) * (2 * np.pi * (blur / px) ** 2)       # ~ locs within the blur kernel
    w = int(2 * round(radius / px) + 1)
    pk = (hb == maximum_filter(hb, w)) & (hb >= weak_min * 0.5)
    iy, ix = np.nonzero(pk)
    c = np.c_[lo + (ix + 0.5) * px, lo + (iy + 0.5) * px]
    T = cKDTree(np.c_[x, y]); P = np.c_[x, y]
    for it in range(3):
        nb = T.query_ball_point(c, radius)
        c = np.array([np.median(P[i], 0) if len(i) else c[j] for j, i in enumerate(nb)])
    nb = T.query_ball_point(c, radius); cnt = np.array([len(i) for i in nb])
    o = np.argsort(-cnt); keep = []; Tc = cKDTree(c); dead = np.zeros(len(c), bool)
    for j in o:
        if dead[j] or cnt[j] < weak_min: continue
        keep.append(j)
        for k in Tc.query_ball_point(c[j], 20.0):
            if k != j: dead[k] = True
    keep = np.array(keep, int)
    sd = np.array([0.5 * (rsd(P[nb[j], 0]) + rsd(P[nb[j], 1])) for j in keep])
    return c[keep], cnt[keep], sd


def build_rulers(c, cnt, sd, min_locs=30, dist=80.0, tol=15.0, crowd=250.0, edge=500.0, fov=(0.0, 26000.0), sd_factor=2.0,
                 radius=30.0):
    """Pair strong sites (cnt >= min_locs, spread <= sd_factor x median) into rulers. Crowding is tested against all
    candidates (strong and weak). A weak candidate at the position of a third site (80 nm further along the axis, within
    radius) does not reject a 2-site ruler; it is flagged (weak_third). Returns (sites, rulers, stats)."""
    strong = cnt >= min_locs
    sdmax = sd_factor * np.median(sd[strong]) if strong.any() else np.inf
    broad = strong & (sd > sdmax); strong &= ~broad
    T = cKDTree(c); S = np.nonzero(strong)[0]
    links = {i: [] for i in S}
    for i, j in T.query_pairs(dist + tol):
        if strong[i] and strong[j] and np.hypot(*(c[i] - c[j])) >= dist - tol: links[i].append(j); links[j].append(i)
    seen = set(); rulers = []; st = dict(n_candidates=len(c), n_strong=int(strong.sum()), n_broad=int(broad.sum()), isolated=0,
                                         bad_topology=0, crowded=0, edge=0, not_linear=0)
    for i in S:
        if i in seen: continue
        comp = [i]; stack = [i]; seen.add(i)
        while stack:
            for k in links[stack.pop()]:
                if k not in seen: seen.add(k); comp.append(k); stack.append(k)
        if len(comp) == 1: st['isolated'] += 1; continue
        nl = sorted(len(links[k]) for k in comp)
        if not ((len(comp) == 2) or (len(comp) == 3 and nl == [1, 1, 2])): st['bad_topology'] += 1; continue
        if len(comp) == 3:
            mid = [k for k in comp if len(links[k]) == 2][0]; ends = [k for k in comp if k != mid]
            comp = [ends[0], mid, ends[1]]
            dee = np.hypot(*(c[ends[0]] - c[ends[1]]))
            if abs(dee - 2 * dist) > 1.5 * tol: st['not_linear'] += 1; continue
        u = c[comp[-1]] - c[comp[0]]; u = u / np.hypot(*u)
        if u[0] < 0 or (u[0] == 0 and u[1] < 0): u = -u; comp = comp[::-1]
        ctr = c[comp].mean(0)
        others = set()
        for k in comp: others |= set(T.query_ball_point(c[k], crowd))
        others -= set(comp); weak3 = False; bad = False
        for k in others:
            if len(comp) == 2 and not strong[k] and not broad[k]:
                e = [c[comp[0]] - dist * u, c[comp[1]] + dist * u]
                if min(np.hypot(*(c[k] - e[0])), np.hypot(*(c[k] - e[1]))) <= radius: weak3 = True; continue
            bad = True
        if bad: st['crowded'] += 1; continue
        if (c[comp].min() < fov[0] + edge) or (c[comp].max() > fov[1] - edge): st['edge'] += 1; continue
        rulers.append(dict(sites=comp, axis=u, centre=ctr, weak_third=weak3))
    # compact site list
    sites = []; out = []
    for r, R in enumerate(rulers):
        ids = []
        for k in R['sites']:
            ids.append(len(sites)); sites.append(dict(id=len(sites), ruler=r, x=float(c[k, 0]), y=float(c[k, 1]), n_ref=int(cnt[k]),
                                                      sd_ref=float(sd[k])))
        out.append(dict(id=r, sites=ids, n_sites=len(ids), axis=[float(R['axis'][0]), float(R['axis'][1])],
                        centre=[float(R['centre'][0]), float(R['centre'][1])], weak_third=bool(R['weak_third'])))
    st.update(n_rulers=len(out), n_rulers_2=sum(r['n_sites'] == 2 for r in out), n_rulers_3=sum(r['n_sites'] == 3 for r in out),
              n_weak_third=sum(r['weak_third'] for r in out), sd_max=float(sdmax))
    return sites, out, st


def save_sites(fn, sites, rulers, meta):
    json.dump(dict(meta=meta, sites=sites, rulers=rulers), open(fn, 'w'), indent=1)


def load_sites(fn):
    d = json.load(open(fn)); return d['sites'], d['rulers'], d['meta']


def pick_overlap(px, py, sites, rulers, cand=None, tol=15.0):
    """How many external picks (site-level centres) coincide (within tol nm) with sites of the automatic rulers."""
    sc = np.array([[s['x'], s['y']] for s in sites]); d, j = cKDTree(sc).query(np.c_[px, py])
    hit = d <= tol; sr = np.array([s['ruler'] for s in sites]); n3 = np.array([r['n_sites'] for r in rulers])
    o = dict(n_picks=len(px), n_on_auto_ruler_site=int(hit.sum()), n_on_2site=int((hit & (n3[sr[j]] == 2)).sum()),
             n_on_3site=int((hit & (n3[sr[j]] == 3)).sum()), n_auto_sites=len(sites),
             n_auto_sites_with_pick=int(len(np.unique(j[hit]))), n_auto_rulers_with_pick=int(len(np.unique(sr[j[hit]]))), tol_nm=tol)
    if cand is not None:
        dc, _ = cKDTree(cand).query(np.c_[px, py]); o['n_on_any_candidate'] = int((dc <= tol).sum())
    return o


def ruler_pick_overlap(px, py, rulers, tol=60.0):
    """Ruler-level external picks vs automatic ruler centres."""
    rc = np.array([r['centre'] for r in rulers]); d, j = cKDTree(rc).query(np.c_[px, py]); hit = d <= tol
    n3 = np.array([r['n_sites'] for r in rulers])
    return dict(n_picks=len(px), n_on_auto_ruler=int(hit.sum()), n_on_2site=int((hit & (n3[j] == 2)).sum()),
                n_on_3site=int((hit & (n3[j] == 3)).sum()), tol_nm=tol)


# ------------------------------------------------------------------ assignment
def global_shift(x, y, sc, rmax=40.0, iters=4):
    """Robust constant offset (method - reference sites): median offset of the locs to the nearest site, iterated."""
    T = cKDTree(sc); s = np.zeros(2)
    for it in range(iters):
        d, j = T.query(np.c_[x - s[0], y - s[1]], distance_upper_bound=rmax if it == 0 else 0.75 * rmax)
        g = np.isfinite(d)
        if g.sum() < 20: break
        s = s + np.array([np.median(x[g] - s[0] - sc[j[g], 0]), np.median(y[g] - s[1] - sc[j[g], 1])])
    return s


def assign(x, y, c, radius):
    d, j = cKDTree(c).query(np.c_[x, y], distance_upper_bound=radius)
    g = np.nonzero(np.isfinite(d))[0]
    return g, j[g]


def eval_method(tab, sites, rulers, radius=30.0, min_n=10, shift=True, t_split=None, pair_dt=1, trace_bins=20):
    """Assign a method table to the sites; returns dict with per-loc (L_*), per-site (S_*) and per-ruler arrays."""
    assert radius < 40.0
    sc = np.array([[s['x'], s['y']] for s in sites]); sr = np.array([s['ruler'] for s in sites]); ns = len(sc)
    ax = np.array([rulers[r]['axis'] for r in sr])
    sh = global_shift(tab['x_nm'], tab['y_nm'], sc) if shift else np.zeros(2)
    x = tab['x_nm'] - sh[0]; y = tab['y_nm'] - sh[1]
    g, j = assign(x, y, sc, radius)
    c1 = sc.copy()
    for s in np.unique(j): c1[s] = [np.median(x[g[j == s]]), np.median(y[g[j == s]])]
    g, j = assign(x, y, c1, radius)
    o = np.lexsort((tab['frame'][g], j)); g, j = g[o], j[o]
    n = np.bincount(j, minlength=ns); st = np.r_[0, np.cumsum(n)]
    c2 = np.full((ns, 2), np.nan); S = {k: np.full(ns, np.nan) for k in
                                         ('rsd_x', 'rsd_y', 'rsd_along', 'rsd_across', 'csd_x', 'csd_y', 'csd_along', 'csd_across',
                                          'half_dx', 'half_dy', 'half_exp')}
    fr = tab['frame'][g]; ts = 0.5 * (tab['frame'].min() + tab['frame'].max()) if t_split is None else t_split
    rx = np.zeros(len(g)); ry = np.zeros(len(g)); defl = np.ones(len(g))
    for s in range(ns):
        if n[s] == 0: continue
        sl = slice(st[s], st[s + 1]); xs, ys = x[g[sl]], y[g[sl]]
        c2[s] = [np.median(xs), np.median(ys)]
        f = np.sqrt(max(1 - KMED / n[s], 0.1)); defl[sl] = f
        rx[sl] = xs - c2[s, 0]; ry[sl] = ys - c2[s, 1]
        if n[s] < min_n: continue
        ra = rx[sl] * ax[s, 0] + ry[sl] * ax[s, 1]; rc = -rx[sl] * ax[s, 1] + ry[sl] * ax[s, 0]
        for k, v in (('x', rx[sl]), ('y', ry[sl]), ('along', ra), ('across', rc)):
            S['rsd_' + k][s] = rsd(v) / f; S['csd_' + k][s] = csd(v)
        h = fr[sl] < ts
        if h.sum() >= 5 and (~h).sum() >= 5:
            S['half_dx'][s] = np.median(xs[~h]) - np.median(xs[h]); S['half_dy'][s] = np.median(ys[~h]) - np.median(ys[h])
            S['half_exp'][s] = 1.2533 * 0.5 * (S['rsd_x'][s] + S['rsd_y'][s]) * np.sqrt(1 / h.sum() + 1 / (~h).sum())
    M = dict(shift=sh, n=n, centre=c2, S=S, site_ruler=sr, axis=ax, L_site=j, L_idx=g, L_rx=rx / defl, L_ry=ry / defl,
             L_sx=tab['sx_nm'][g], L_sy=tab['sy_nm'][g], L_ph=tab['photons'][g], L_frame=fr, n_table=len(x),
             L_x=x[g], L_y=y[g])
    M['L_ra'] = M['L_rx'] * ax[j, 0] + M['L_ry'] * ax[j, 1]; M['L_rc'] = -M['L_rx'] * ax[j, 1] + M['L_ry'] * ax[j, 0]
    for k in OPT:
        if k in tab: M['L_' + k] = tab[k][g]
    # consecutive-frame pairs within a site (same binding event): difference / sqrt(2) = precision without site size / drift
    pp = np.nonzero((j[1:] == j[:-1]) & (fr[1:] - fr[:-1] == pair_dt) & (n[j[1:]] >= min_n))[0]
    M.update(P_site=j[pp], P_dx=(x[g[pp + 1]] - x[g[pp]]) / np.sqrt(2), P_dy=(y[g[pp + 1]] - y[g[pp]]) / np.sqrt(2),
             P_zx=(x[g[pp + 1]] - x[g[pp]]) / np.hypot(M['L_sx'][pp + 1], M['L_sx'][pp]),
             P_zy=(y[g[pp + 1]] - y[g[pp]]) / np.hypot(M['L_sy'][pp + 1], M['L_sy'][pp]),
             P_sx=np.hypot(M['L_sx'][pp + 1], M['L_sx'][pp]) / np.sqrt(2), P_sy=np.hypot(M['L_sy'][pp + 1], M['L_sy'][pp]) / np.sqrt(2))
    # global residual-drift trace: median residual of all sites (n >= min_n) in time bins
    te = np.linspace(tab['frame'].min(), tab['frame'].max() + 1, trace_bins + 1); tb = np.clip(np.searchsorted(te, fr, side='right') - 1, 0, trace_bins - 1)
    gg = n[j] >= min_n
    M['trace'] = dict(t=(0.5 * (te[1:] + te[:-1])).tolist(), n=[int((gg & (tb == i)).sum()) for i in range(trace_bins)],
                      x=[float(np.median(rx[gg & (tb == i)])) if (gg & (tb == i)).any() else float('nan') for i in range(trace_bins)],
                      y=[float(np.median(ry[gg & (tb == i)])) if (gg & (tb == i)).any() else float('nan') for i in range(trace_bins)])
    # per-ruler adjacent distances (method's own centres)
    nr = len(rulers); ok = np.zeros(nr, bool); pr = []; pd = []; er = []; ed = []
    for r, R in enumerate(rulers):
        ids = R['sites']; ok[r] = all(n[i] >= min_n for i in ids)
        if not ok[r]: continue
        for a, b in zip(ids[:-1], ids[1:]): pr.append(r); pd.append(np.hypot(*(c2[a] - c2[b])))
        if len(ids) == 3: er.append(r); ed.append(np.hypot(*(c2[ids[0]] - c2[ids[2]])))
    M.update(ruler_ok=ok, pair_ruler=np.array(pr, int), pair_dist=np.array(pd), end_ruler=np.array(er, int), end_dist=np.array(ed),
             ruler_nsites=np.array([R['n_sites'] for R in rulers]))
    return M


# ------------------------------------------------------------------ summaries
def photon_edges(M, rsel, nbins):
    g = rsel[M['site_ruler'][M['L_site']]] & (M['n'][M['L_site']] >= 1)
    e = np.quantile(M['L_ph'][g], np.linspace(0, 1, nbins + 1)); e[0] = -np.inf; e[-1] = np.inf
    return e


def summarize(M, w, edges):
    """Flat dict of statistics for ruler multiplicities w (int per ruler; 0 = not used)."""
    sr = M['site_ruler']; ws = w[sr]; wl = ws[M['L_site']]
    li = np.repeat(np.arange(len(wl)), wl); si = np.repeat(np.arange(len(ws)), ws)
    o = dict(n_rulers=int(w.sum()), n_sites=int(ws.sum()), n_locs=len(li),
             n_rulers_2=int(w[M['ruler_nsites'] == 2].sum()), n_rulers_3=int(w[M['ruler_nsites'] == 3].sum()))
    pd = np.repeat(M['pair_dist'], w[M['pair_ruler']]); p3 = np.repeat(M['ruler_nsites'][M['pair_ruler']] == 3, w[M['pair_ruler']])
    o.update(dist_mean=np.mean(pd), dist_median=np.median(pd), dist_sd=np.std(pd, ddof=1), dist_rsd=rsd(pd), n_pairs=len(pd))
    o.update(dist2_mean=np.mean(pd[~p3]) if (~p3).any() else np.nan, dist3_mean=np.mean(pd[p3]) if p3.any() else np.nan)
    ed = np.repeat(M['end_dist'], w[M['end_ruler']]); o['dist3_end_mean'] = np.mean(ed) if len(ed) else np.nan
    nn = M['n'][si]; o.update(locs_per_site_median=float(np.median(nn)), locs_per_site_mean=float(np.mean(nn)))
    for k in ('x', 'y', 'along', 'across'):
        o['site_rsd_%s_median' % k] = np.nanmedian(M['S']['rsd_' + k][si]); o['site_csd_%s_median' % k] = np.nanmedian(M['S']['csd_' + k][si])
    R = dict(x=M['L_rx'][li], y=M['L_ry'][li], along=M['L_ra'][li], across=M['L_rc'][li])
    for k, v in R.items(): o['pooled_rsd_' + k] = rsd(v); o['pooled_csd_' + k] = csd(v)
    ph = M['L_ph'][li]; b = np.clip(np.searchsorted(edges, ph, side='right') - 1, 0, len(edges) - 2)
    sig = dict(x=M['L_sx'][li], y=M['L_sy'][li])
    crl = {a: M['L_crlb_%s_nm' % a][li] for a in 'xy'} if 'L_crlb_x_nm' in M else None

    def cal(sel, pre):
        o[pre + 'n'] = int(sel.sum()) if sel is not None else len(li)
        q = (lambda v: v[sel]) if sel is not None else (lambda v: v)
        o[pre + 'phot_med'] = float(np.median(q(ph))) if o[pre + 'n'] else np.nan
        for a in 'xy':
            r = q(R[a]); s = q(sig[a])
            if len(r) < 5:
                for k in ('rsd_', 'sig_med_', 'sig_rms_', 'ratio_', 'ratio_rms_', 'zsd_', 'excess_'): o[pre + k + a] = np.nan
                continue
            m = rsd(r); sm = np.median(s); o[pre + 'rsd_' + a] = m; o[pre + 'sig_med_' + a] = sm
            o[pre + 'sig_rms_' + a] = np.sqrt(np.mean(s ** 2)); o[pre + 'ratio_' + a] = m / sm
            o[pre + 'ratio_rms_' + a] = m / o[pre + 'sig_rms_' + a]; o[pre + 'zsd_' + a] = rsd(r / s)
            o[pre + 'excess_' + a] = np.sign(m - sm) * np.sqrt(abs(m ** 2 - sm ** 2))
            if crl is not None: o[pre + 'crlb_med_' + a] = np.median(q(crl[a]))
    cal(None, 'all_')
    for i in range(len(edges) - 1): cal(b == i, 'bin%d_' % i)
    pw = ws[M['P_site']]; pi_ = np.repeat(np.arange(len(pw)), pw); o['n_consec_pairs'] = len(pi_)
    for a_ in 'xy':
        if len(pi_) >= 20:
            o['pair_rsd_' + a_] = rsd(M['P_d' + a_][pi_]); o['pair_zsd_' + a_] = rsd(M['P_z' + a_][pi_])
            o['pair_sig_med_' + a_] = float(np.median(M['P_s' + a_][pi_])); o['pair_ratio_' + a_] = o['pair_rsd_' + a_] / o['pair_sig_med_' + a_]
        else:
            for k_ in ('pair_rsd_', 'pair_zsd_', 'pair_sig_med_', 'pair_ratio_'): o[k_ + a_] = np.nan
    hx, hy, he = M['S']['half_dx'][si], M['S']['half_dy'][si], M['S']['half_exp'][si]; g = np.isfinite(hx)
    o.update(half_dx_median=np.median(hx[g]), half_dy_median=np.median(hy[g]), half_dx_rsd=rsd(hx[g]), half_dy_rsd=rsd(hy[g]),
             half_expected_sd=np.median(he[g]))
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in o.items()}


def bootstrap(M, rsel, edges, nboot=1000, seed=1):
    """Point estimate on the selected rulers + percentile 95 % CI resampling rulers."""
    w0 = rsel.astype(int); est = summarize(M, w0, edges)
    rng = np.random.default_rng(seed); idx = np.nonzero(rsel)[0]; acc = {k: [] for k in est}
    for b in range(nboot):
        w = np.bincount(rng.choice(idx, len(idx)), minlength=len(rsel))
        s = summarize(M, w, edges)
        for k in est: acc[k].append(s[k])
    ci = {}
    for k, v in acc.items():
        v = np.array(v, float); ci[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [np.nan, np.nan]
    return est, ci


def paired(MA, MB, rsel, nboot=1000, seed=1):
    """Per-site difference of spread A - B on the same sites (both with valid spread), CI resampling rulers."""
    sr = MA['site_ruler']; out = {}
    rng = np.random.default_rng(seed); idx = np.nonzero(rsel)[0]
    W = [rsel.astype(int)] + [np.bincount(rng.choice(idx, len(idx)), minlength=len(rsel)) for b in range(nboot)]
    for k in ('x', 'y', 'along', 'across', 'xy'):
        if k == 'xy':
            a = np.sqrt(0.5 * (MA['S']['rsd_x'] ** 2 + MA['S']['rsd_y'] ** 2)); b = np.sqrt(0.5 * (MB['S']['rsd_x'] ** 2 + MB['S']['rsd_y'] ** 2))
        else: a, b = MA['S']['rsd_' + k], MB['S']['rsd_' + k]
        g = np.isfinite(a) & np.isfinite(b)
        res = []
        for w in W:
            si = np.repeat(np.arange(len(sr)), w[sr] * g); d = a[si] - b[si]
            res.append([np.median(d), np.mean(d), np.median(a[si] / b[si]), np.mean(d < 0)])
        res = np.array(res); names = ('median_diff_nm', 'mean_diff_nm', 'median_ratio', 'frac_sites_A_smaller')
        out[k] = {nm: dict(est=float(res[0, i]), ci=[float(np.percentile(res[1:, i], 2.5)), float(np.percentile(res[1:, i], 97.5))])
                  for i, nm in enumerate(names)}
        out[k]['n_sites'] = int((g & rsel[sr]).sum())
    return out


# ------------------------------------------------------------------ FRC
def frc_nm(x, y, seed=1, px=2.5):
    """Random-half-split FRC (1/7 threshold, 5-nm render) with the repository implementation (which takes coordinates in
    units of its PX constant; nm coordinates are divided by that constant, so no pixel-size assumption enters)."""
    sys.path.insert(0, '/home/jc/simcode2')
    from simcode2.evalproto import lib as E
    if len(x) < 100: return float('nan')
    h = np.random.default_rng(seed).random(len(x)) < 0.5
    return E.frc(x[h] / E.PX, y[h] / E.PX, x[~h] / E.PX, y[~h] / E.PX, px=px)


# ------------------------------------------------------------------ report
def fmt(est, ci, k, d=2):
    if k not in est or not np.isfinite(est[k]): return '-'
    return ('%.{0}f [%.{0}f, %.{0}f]'.format(d)) % (est[k], ci[k][0], ci[k][1])


def markdown(res, edges_txt=''):
    L = ['## Nanoruler metrics (nm; value [95 % bootstrap CI over rulers])', '']
    cols = [('rulers (2/3-site)', None), ('sites', None), ('locs/site med (mean)', None), ('distance mean', 'dist_mean'),
            ('distance median', 'dist_median'), ('distance sd', 'dist_sd'),
            ('site spread x', 'site_rsd_x_median'), ('site spread y', 'site_rsd_y_median'), ('along', 'site_rsd_along_median'),
            ('across', 'site_rsd_across_median'), ('pooled x', 'pooled_rsd_x'), ('pooled y', 'pooled_rsd_y'),
            ('reported sigma x / y (median)', None), ('ratio x', 'all_ratio_x'), ('ratio y', 'all_ratio_y'),
            ('z sd x', 'all_zsd_x'), ('z sd y', 'all_zsd_y'), ('FRC rulers / all', None)]
    L.append('| method | ' + ' | '.join(c[0] for c in cols) + ' |'); L.append('|' + '---|' * (len(cols) + 1))
    for name, r in res.items():
        e, c = r['est'], r['ci']; row = [name]
        for t, k in cols:
            if k: row.append(fmt(e, c, k))
            elif t.startswith('rulers'): row.append('%d (%d/%d)' % (e['n_rulers'], e['n_rulers_2'], e['n_rulers_3']))
            elif t == 'sites': row.append('%d' % e['n_sites'])
            elif t.startswith('locs'): row.append('%.0f (%.1f)' % (e['locs_per_site_median'], e['locs_per_site_mean']))
            elif t.startswith('reported'): row.append('%.2f / %.2f' % (e['all_sig_med_x'], e['all_sig_med_y']))
            else: row.append('%.1f / %.1f' % (r['frc_rulers_nm'], r['frc_all_nm']))
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['', 'site spread = median over sites of the per-site robust sd (1.4826 MAD about the site median, small-n corrected); '
          'pooled = robust sd of all residuals pooled; ratio = pooled robust sd / median reported sigma; z sd = robust sd of '
          'residual / reported sigma (1 = calibrated). Residuals include the physical site size and residual drift.', '']
    L += ['### Clipped sd (plain sd after 3-robust-sd clipping), 3-site rulers, drift diagnostic', '',
          '| method | site csd x | site csd y | along | across | pooled csd x | pooled csd y | dist 2-site | dist 3-site adjacent | '
          '3-site end-to-end | consecutive-frame pair precision x | y | pair z sd x | y | n pairs | global shift x, y | half-movie centre shift median x, y | sd over sites x, y (noise expectation) |',
          '|' + '---|' * 18]
    for name, r in res.items():
        e, c = r['est'], r['ci']
        L.append('| ' + ' | '.join([name] + [fmt(e, c, k) for k in ('site_csd_x_median', 'site_csd_y_median', 'site_csd_along_median',
                 'site_csd_across_median', 'pooled_csd_x', 'pooled_csd_y', 'dist2_mean', 'dist3_mean', 'dist3_end_mean', 'pair_rsd_x', 'pair_rsd_y', 'pair_zsd_x', 'pair_zsd_y')] +
                 ['%d' % e['n_consec_pairs'], '%.2f, %.2f' % tuple(r['shift_nm']), '%.2f, %.2f' % (e['half_dx_median'], e['half_dy_median']),
                  '%.2f, %.2f (%.2f)' % (e['half_dx_rsd'], e['half_dy_rsd'], e['half_expected_sd'])]) + ' |')
    L += ['', 'consecutive-frame pair precision = robust sd of (position difference of two localizations of one site in consecutive '
          'frames) / sqrt(2): free of site size and slow drift; pair z sd = robust sd of difference / sqrt(sigma1^2 + sigma2^2).', '']
    for name, r in res.items():
        e, c = r['est'], r['ci']; nb = r['nbins']; has = 'bin0_crlb_med_x' in e
        L += ['### Calibration in photon bins: %s' % name, '',
              '| photons (median; range) | n | spread x | sigma x' + (' | CRLB x' if has else '') + ' | ratio x | z sd x | spread y | sigma y' +
              (' | CRLB y' if has else '') + ' | ratio y | z sd y |', '|' + '---|' * (12 if has else 10)]
        for i in range(nb):
            p = 'bin%d_' % i; ed = r['edges']
            row = ['%.0f; %s - %s' % (e[p + 'phot_med'], '%.0f' % ed[i] if np.isfinite(ed[i]) else 'min', '%.0f' % ed[i + 1] if np.isfinite(ed[i + 1]) else 'max'),
                   '%d' % e[p + 'n']]
            for a in 'xy':
                row += [fmt(e, c, p + 'rsd_' + a), fmt(e, c, p + 'sig_med_' + a)] + ([fmt(e, c, p + 'crlb_med_' + a)] if has else []) + \
                       [fmt(e, c, p + 'ratio_' + a), fmt(e, c, p + 'zsd_' + a)]
            L.append('| ' + ' | '.join(row) + ' |')
        L.append('')
    return '\n'.join(L)


# ------------------------------------------------------------------ figures
def _style():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                         'xtick.color': '#52514e', 'ytick.color': '#52514e', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                         'savefig.facecolor': '#fcfcfb', 'legend.frameon': False})
    return plt


def fig_gallery(fn, Ms, rulers, rsel, n=24, seed=1, box=280.0, px=2.5, blur=2.5, ncol=12):
    """Same rulers for every method; histogram at px nm blurred with a fixed Gaussian (blur nm); every panel is scaled to
    its own maximum (same rule for all methods). Scale bar 80 nm."""
    plt = _style(); rng = np.random.default_rng(seed); idx = np.nonzero(rsel)[0]
    pick = np.sort(rng.choice(idx, min(n, len(idx)), replace=False)); nrow = int(np.ceil(len(pick) / ncol)); nm = len(Ms)
    nb = int(box / px); fig, axs = plt.subplots(nm * nrow, ncol, figsize=(ncol * 1.05, nm * nrow * 1.05 + 0.35 * nm), squeeze=False)
    for m, (name, M) in enumerate(Ms.items()):
        for q in range(nrow * ncol):
            ax = axs[m * nrow + q // ncol, q % ncol]; ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values(): s.set_visible(False)
            if q >= len(pick): ax.set_visible(False); continue
            R = rulers[pick[q]]; cx, cy = R['centre']
            g = (np.abs(M['T_x'] - cx) < box / 2 + 3 * blur) & (np.abs(M['T_y'] - cy) < box / 2 + 3 * blur)
            h = np.histogram2d(M['T_y'][g], M['T_x'][g], bins=nb, range=[[cy - box / 2, cy + box / 2], [cx - box / 2, cx + box / 2]])[0]
            h = gaussian_filter(h, blur / px); ax.imshow(h, cmap='inferno', vmin=0, vmax=max(h.max(), 1e-9), origin='lower', interpolation='nearest')
            if q == 0:
                ax.plot([0.08 * nb, 0.08 * nb + 80 / px], [0.08 * nb] * 2, color='w', lw=2)
                ax.set_title('%s   (scale bar 80 nm, ruler ids %s ...)' % (name, ', '.join(str(i) for i in pick[:4])), loc='left', fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.04, hspace=0.12)
    fig.savefig(fn, dpi=170); plt.close(fig)
    return pick.tolist()


def fig_spread(fn, res):
    plt = _style(); fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for j, a in enumerate('xy'):
        ax = axs[j]
        for m, (name, r) in enumerate(res.items()):
            e, c = r['est'], r['ci']; nb = r['nbins']; col = COLORS[m % len(COLORS)]
            p = np.array([e['bin%d_phot_med' % i] for i in range(nb)]); v = np.array([e['bin%d_rsd_%s' % (i, a)] for i in range(nb)])
            lo = np.array([c['bin%d_rsd_%s' % (i, a)][0] for i in range(nb)]); hi = np.array([c['bin%d_rsd_%s' % (i, a)][1] for i in range(nb)])
            ax.errorbar(p, v, yerr=[v - lo, hi - v], color=col, lw=2, marker='o', ms=5, capsize=2, label='%s: measured spread' % name)
            ax.plot(p, [e['bin%d_sig_med_%s' % (i, a)] for i in range(nb)], color=col, lw=1.5, ls='--', label='%s: reported sigma' % name)
            if 'bin0_crlb_med_' + a in e:
                ax.plot(p, [e['bin%d_crlb_med_%s' % (i, a)] for i in range(nb)], color=col, lw=1.2, ls=':', label='%s: CRLB' % name)
        ax.set_xlabel('photons per 6-frame window (bin median)'); ax.set_title('%s axis' % a, loc='left')
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel('nm (robust sd of residuals about the site centre)'); axs[1].legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.suptitle('Cluster spread vs photons (bars: 95 % bootstrap CI over rulers)', x=0.01, ha='left', fontsize=10)
    fig.tight_layout(); fig.savefig(fn, dpi=160); plt.close(fig)


def fig_dist(fn, Ms, rsel, res):
    plt = _style(); nm = len(Ms); fig, axs = plt.subplots(nm, 1, figsize=(5.5, 1.9 * nm + 0.6), sharex=True, squeeze=False)
    bins = np.arange(50, 111, 2.0)
    for m, (name, M) in enumerate(Ms.items()):
        ax = axs[m, 0]; d = M['pair_dist'][rsel[M['pair_ruler']]]; e = res[name]['est']
        ax.hist(np.clip(d, bins[0], bins[-1] - 1e-6), bins=bins, color=COLORS[m % len(COLORS)], edgecolor='#fcfcfb', linewidth=0.6)
        ax.axvline(80, color='#52514e', lw=1, ls='--')
        ax.set_title('%s: mean %.2f, median %.2f, sd %.2f nm (n = %d)' % (name, e['dist_mean'], e['dist_median'], e['dist_sd'], len(d)), loc='left', fontsize=9)
        ax.set_ylabel('site pairs')
    axs[-1, 0].set_xlabel('site-to-site distance (nm); dashed: 80 nm')
    fig.tight_layout(); fig.savefig(fn, dpi=160); plt.close(fig)
