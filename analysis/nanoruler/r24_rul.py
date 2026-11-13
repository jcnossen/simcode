"""R24: SITE-PAIR analysis of closely spaced binding sites (40 nm, 20 nm), where the fixed-radius site assignment of nra_lib (80 nm rulers,
radius 30 nm) cannot work. CPU only. The same definitions for simulated and real data and for every method.

The nr40 / nr20 samples are NOT linear rulers but 2D origami structures (nr40: sites on the corners of a ~40 nm square; nr20: a ~20 nm grid),
so the unit of analysis is the PAIR of adjacent sites (centre, axis) and the unit of the bootstrap is the STRUCTURE (connected sites).

Tables: nra format (npz: frame = 6-frame cycle index, x_nm, y_nm, sx_nm, sy_nm, photons; optional further sigma columns sx_<lv>_nm / sy_<lv>_nm).

  pairs   --ref table.npz | --picks picks.npz | --truth sites.npz   --d0 20 --out pairs.json
      sites:  ref   = cluster maxima of the reference table (as nra_lib.find_sites: maxima of the blurred 2-nm histogram, 3 median-shift iterations
                      within the site radius rs = 0.4 d0, duplicates closer than 0.5 d0 merged [nra_lib merges at a fixed 20 nm], n >= min_locs)
              picks = external site centres (the paper's picks);   truth = the simulated sites of sim_nr.py.
      structures = connected components of the sites linked at <= 1.6 d0.  pairs = two sites at a distance within 35 % of d0 with no third site
      inside the pair window and >= edge nm from the field border. Pair centre = midpoint, axis = unit vector between the two sites.
  run     --pairs pairs.json --methods name=table.npz ... --d0 20 [--cmp A:B ...]
      per method: one global shift (median offset to the nearest site). PAIR WINDOW: |t| <= WA = d0 along the pair axis, |c| <= WC = d0 / 2 across
      (t, c relative to the pair centre). Truncated maximum likelihood on the window (L-BFGS-B from several starts):
        along:  K = 1: one Gaussian (mean, sd) + uniform background;  K = 2: two Gaussians of COMMON sd at m -+ d / 2 with free weights + uniform
                background (weight <= 0.3); every Gaussian is normalized on the window (truncation), BIC = -2 ln L + p ln n, p = 3 / 5.
        across: one Gaussian (mean, sd) truncated at -+WC + uniform background -> spread across.
      DEFINITION "resolved" (pair): K = 2 is valid if the smaller site weight is >= 5 % of the site localizations and >= 5 localizations;
        the pair is "bimodal" if K = 2 is valid and BIC(1) - BIC(2) >= 10; it is "resolved" if in addition the fitted separation d is within
        30 % of d0 (20 nm: 14-26 nm; 40 nm: 28-52 nm). Distance estimate = d of the K = 2 fit (reported for resolved pairs, and for all pairs).
        Pairs need >= min_n localizations in the window. A pair whose localizations come from one site only cannot be resolved by any method:
        the resolved fraction has a sampling ceiling (the simulation gives it).
      spread along = the common sd of the K = 2 fit (bimodal pairs only); spread across = sd of the across fit (all pairs); x / y by rotation.
      Consecutive-cycle pairs (STRUCTURE level, every localization used once): localizations are assigned to the nearest site within ra = 0.75 d0
      (-> its structure); two localizations of one structure in cycles f, f + 1 closer than pair_r = 0.75 d0 + 15 nm: difference / sqrt(2) =
      precision without site size / slow drift; pair z = difference / sqrt(s1^2 + s2^2) per sigma level, overall and per photon bin.
      Midpoint-band excess (resolved pairs): fraction of the window localizations with |t - m| <= d / 4 minus the fraction the fitted K = 2 model
      (two truncated Gaussians + uniform background) predicts: an estimator of merged events (two sites ON in one window give a row between
      the sites); the fit absorbs part of them in the sd / background, so it must be calibrated on a simulation.
      Across z = robust sd of (c - fitted mean) / sigma_c (sigma_c^2 = sx^2 uy^2 + sy^2 ux^2) of the window localizations (truncated at WC: only a
      relative number). Bootstrap: structures resampled (percentile 95 %); common pairs = n >= min_n in every method.
  usage: python r24_rul.py pairs --ref T.npz --d0 20 --out pairs.json [--min_locs 30 --edge 500 --fov 0 26000]
         python r24_rul.py run --pairs pairs.json --d0 20 --out DIR --methods a=A.npz b=B.npz [--tag t --levels raw --bins 5 --nboot 300 --cmp a:b]"""
import argparse, json, os, sys
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_k] = '1'      # the fits are tiny: BLAS threads only thrash
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, maximum_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nra_lib as L

WFLOOR = 0.05
JOBS = int(os.environ.get('R24_JOBS', 8))
S2PI = 2.5066282746


# ------------------------------------------------------------------ truncated ML
def _tn(t, mu, s, W):
    """Gaussian density of mean mu, sd s, normalized on [-W, W]"""
    z = np.maximum(ndtr((W - mu) / s) - ndtr((-W - mu) / s), 1e-12)
    return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * S2PI * z)


def fit1(t, W):
    """one truncated Gaussian + uniform background on [-W, W] -> lnL, m, s, wb"""
    ub = 1.0 / (2 * W)
    def nll(p): return -np.log((1 - p[2]) * _tn(t, p[0], p[1], W) + p[2] * ub + 1e-300).sum()
    best = None
    for s0 in (max(0.7 * float(np.std(t)), 1.0), 0.25 * W):
        r = minimize(nll, [float(np.median(t)), s0, 0.05], method='L-BFGS-B', bounds=[(-W, W), (0.7, 3 * W), (1e-4, 0.3)])
        if best is None or r.fun < best.fun: best = r
    return -float(best.fun), float(best.x[0]), float(best.x[1]), float(best.x[2])


def fit2(t, W, d0):
    """two truncated Gaussians of common sd at m -+ d / 2 + uniform background -> lnL, m, d, s, w1, wb"""
    ub = 1.0 / (2 * W)
    def nll(p):
        m, d, s, w1, wb = p
        return -np.log((1 - wb) * (w1 * _tn(t, m - 0.5 * d, s, W) + (1 - w1) * _tn(t, m + 0.5 * d, s, W)) + wb * ub + 1e-300).sum()
    best = None; q = np.percentile(t, [5, 50, 95]); m0s = (float(q[1]), 0.5 * float(q[0] + q[2]))
    for m0 in m0s:
        for dd in (0.6 * d0, d0, 1.4 * d0):
            for s0 in (0.15 * d0, 0.3 * d0):
                r = minimize(nll, [m0, dd, s0, 0.5, 0.05], method='L-BFGS-B', bounds=[(-W, W), (2.0, 2 * W), (0.7, W), (0.02, 0.98), (1e-4, 0.3)])
                if best is None or r.fun < best.fun: best = r
    return (-float(best.fun),) + tuple(float(v) for v in best.x)


def fit_pair(t, c, WA, WC, d0):
    n = len(t); l1, m1, s1, wb1 = fit1(t, WA); l2, m2, d, s2, w1, wb2 = fit2(t, WA, d0)
    b1 = -2 * l1 + 3 * np.log(n); b2 = -2 * l2 + 5 * np.log(n); wmin = min(w1, 1 - w1)
    valid = wmin >= WFLOOR and n * (1 - wb2) * wmin >= 5; multi = bool(valid and b1 - b2 >= 10); res = bool(multi and abs(d - d0) <= 0.3 * d0)
    lc, mc, sc, wbc = fit1(c, WC)
    # midpoint band |t - m| <= d / 4: observed fraction of the window localizations minus the fraction the K = 2 model predicts (merged-event estimator)
    h = 0.25 * d; zz = lambda mu: max(ndtr((WA - mu) / s2) - ndtr((-WA - mu) / s2), 1e-12)
    pe = sum(w_ * (ndtr((m2 + h - mu) / s2) - ndtr((m2 - h - mu) / s2)) / zz(mu) for w_, mu in ((w1, m2 - 0.5 * d), (1 - w1, m2 + 0.5 * d)))
    mid_exp = (1 - wb2) * pe + wb2 * (2 * h) / (2 * WA); mid_obs = float(np.mean(np.abs(t - m2) <= h))
    return dict(mid_obs=mid_obs, mid_exp=float(mid_exp), mid_excess=mid_obs - float(mid_exp), dbic=float(b1 - b2), multi=multi, resolved=res, d=d, s=s2 if multi else np.nan, s2=s2, s1=s1, wmin=wmin, wb=wb2 if multi else wb1, m=m2 if multi else m1,
                sd_across=sc, mc=mc, wbc=wbc)


# ------------------------------------------------------------------ sites, pairs
def find_sites(x, y, rs, blur, merge, weak_min=5, px=2.0):
    """nra_lib.find_sites with the bin size, the site radius and the duplicate-merge radius as arguments"""
    lo = min(x.min(), y.min()) - 100; hi = max(x.max(), y.max()) + 100; n = int(np.ceil((hi - lo) / px))
    h = np.histogram2d(y, x, bins=n, range=[[lo, lo + n * px], [lo, lo + n * px]])[0]
    hb = gaussian_filter(h, blur / px) * (2 * np.pi * (blur / px) ** 2)
    pk = (hb == maximum_filter(hb, int(2 * round(rs / px) + 1))) & (hb >= weak_min * 0.5)
    iy, ix = np.nonzero(pk); c = np.c_[lo + (ix + 0.5) * px, lo + (iy + 0.5) * px]
    T = cKDTree(np.c_[x, y]); P = np.c_[x, y]
    for it in range(3):
        nb = T.query_ball_point(c, rs); c = np.array([np.median(P[i], 0) if len(i) else c[j] for j, i in enumerate(nb)])
    nb = T.query_ball_point(c, rs); cnt = np.array([len(i) for i in nb]); o = np.argsort(-cnt); keep = []; Tc = cKDTree(c); dead = np.zeros(len(c), bool)
    for j in o:
        if dead[j] or cnt[j] < weak_min: continue
        keep.append(j)
        for k in Tc.query_ball_point(c[j], merge):
            if k != j: dead[k] = True
    keep = np.array(keep, int)
    return c[keep], cnt[keep]


def cmd_pairs(a):
    d0 = a.d0; WA = d0; WC = 0.5 * d0; meta = dict(d0=d0, WA=WA, WC=WC, mode='truth' if a.truth else ('picks' if a.picks else 'ref'), args={k: v for k, v in vars(a).items() if k != 'fn'})
    if a.truth:
        S = np.load(a.truth); c = np.c_[S['x_nm'], S['y_nm']]; cnt = np.asarray(S['n_full6']) if 'n_full6' in S.files else np.zeros(len(c), int); unit = np.asarray(S['ruler']).astype(int)
    else:
        if a.picks: P = np.load(a.picks); c = np.c_[P['x_nm'], P['y_nm']]; cnt = np.asarray(P['n']) if 'n' in P.files else np.zeros(len(c), int)
        else:
            T = L.load_table(a.ref); c, cnt = find_sites(T['x_nm'], T['y_nm'], a.rs if a.rs > 0 else 0.4 * d0, a.blur if a.blur > 0 else 0.15 * d0, 0.5 * d0)
            g = cnt >= a.min_locs; meta['n_candidates'] = int(len(c)); c, cnt = c[g], cnt[g]
        unit = np.arange(len(c))
        for i, j in cKDTree(c).query_pairs(1.6 * d0):
            ua, ub = unit[i], unit[j]
            if ua != ub: unit[unit == max(ua, ub)] = min(ua, ub)
        unit = np.unique(unit, return_inverse=True)[1]
    Tc = cKDTree(c); R = []; st = dict(n_sites=int(len(c)), n_structures=int(unit.max() + 1) if len(c) else 0, candidate_pairs=0, third_site=0, edge=0)
    st['sites_per_structure_hist'] = np.bincount(np.bincount(unit)).tolist()
    for i, j in sorted(Tc.query_pairs(1.35 * d0)):
        dist = float(np.hypot(*(c[i] - c[j])))
        if dist < 0.65 * d0: continue
        st['candidate_pairs'] += 1; ctr = 0.5 * (c[i] + c[j]); u = (c[j] - c[i]) / dist
        if u[0] < 0 or (u[0] == 0 and u[1] < 0): u = -u
        oth = [k for k in Tc.query_ball_point(ctr, np.hypot(WA, WC)) if k not in (i, j)]
        if any(abs((c[k] - ctr) @ u) <= WA and abs(-(c[k] - ctr)[0] * u[1] + (c[k] - ctr)[1] * u[0]) <= WC for k in oth): st['third_site'] += 1; continue
        if ctr.min() < a.fov[0] + a.edge or ctr.max() > a.fov[1] - a.edge: st['edge'] += 1; continue
        R.append(dict(id=len(R), centre=[float(ctr[0]), float(ctr[1])], axis=[float(u[0]), float(u[1])], unit=int(unit[i]), sites=[int(i), int(j)], ref_dist=dist))
    dd = np.array([r['ref_dist'] for r in R])
    st.update(n_pairs=len(R), ref_dist_mean=float(dd.mean()) if len(R) else None, ref_dist_sd=float(dd.std()) if len(R) else None, n_structures_with_pair=len({r['unit'] for r in R}))
    meta['stats'] = st; json.dump(dict(meta=meta, sites=c.tolist(), site_unit=unit.tolist(), site_n=[int(v) for v in cnt], rulers=R), open(a.out, 'w'))
    print(json.dumps({k: v for k, v in meta.items() if k != 'args'}))


# ------------------------------------------------------------------ evaluation
def load_tab(fn):
    d = np.load(fn); t = {k: np.asarray(d[k], float) for k in d.files if np.ndim(d[k]) == 1 and d[k].dtype.kind in 'fiub'}
    g = np.isfinite(t['x_nm']) & np.isfinite(t['y_nm']) & np.isfinite(t['sx_nm']) & np.isfinite(t['sy_nm'])
    return {k: v[g] for k, v in t.items() if len(v) == len(g)}


def eval_method(tab, D, d0, levels, min_n, pair_r, shift=True):
    rulers = D['rulers']; sc = np.array(D['sites']); su = np.array(D['site_unit']); WA, WC = D['meta']['WA'], D['meta']['WC']
    rc = np.array([r['centre'] for r in rulers]); ax = np.array([r['axis'] for r in rulers]); ru = np.array([r['unit'] for r in rulers]); nr = len(rc); nu = int(su.max()) + 1
    sh = L.global_shift(tab['x_nm'], tab['y_nm'], sc, rmax=0.75 * d0) if shift else np.zeros(2)
    x = tab['x_nm'] - sh[0]; y = tab['y_nm'] - sh[1]; fr = tab['frame']; ph = tab['photons']
    SGa = {'cal': (tab['sx_nm'], tab['sy_nm'])}
    for lv in levels:
        if f'sx_{lv}_nm' in tab: SGa[lv] = (tab[f'sx_{lv}_nm'], tab[f'sy_{lv}_nm'])
    # structure level: consecutive-cycle pairs (every localization once)
    g, j = L.assign(x, y, sc, 0.75 * d0); un = su[j]; o = np.lexsort((fr[g], un)); g, un = g[o], un[o]
    pp = np.nonzero((un[1:] == un[:-1]) & (fr[g[1:]] - fr[g[:-1]] == 1))[0]
    dx = x[g[pp + 1]] - x[g[pp]]; dy = y[g[pp + 1]] - y[g[pp]]; k = np.hypot(dx, dy) <= pair_r; pp, dx, dy = pp[k], dx[k], dy[k]
    M = dict(shift=sh, n_table=len(x), n_assigned=len(g), U_unit=un, U_ph=ph[g], P_unit=un[pp], P_dx=dx, P_dy=dy, P_ph=0.5 * (ph[g[pp]] + ph[g[pp + 1]]),
             P_s={lv: (np.hypot(sx[g[pp]], sx[g[pp + 1]]), np.hypot(sy[g[pp]], sy[g[pp + 1]])) for lv, (sx, sy) in SGa.items()},
             U_s={lv: (sx[g], sy[g]) for lv, (sx, sy) in SGa.items()}, ru=ru, nu=nu, T_x=x, T_y=y)
    # pair windows
    T = cKDTree(np.c_[x, y]); nb = T.query_ball_point(rc, float(np.hypot(WA, WC))); jobs = []; n = np.zeros(nr, int); Lr, Lc, Li = [], [], []
    for r in range(nr):
        i = np.array(nb[r], int)
        if not len(i): continue
        t = (x[i] - rc[r, 0]) * ax[r, 0] + (y[i] - rc[r, 1]) * ax[r, 1]; c = -(x[i] - rc[r, 0]) * ax[r, 1] + (y[i] - rc[r, 1]) * ax[r, 0]
        k = (np.abs(t) <= WA) & (np.abs(c) <= WC); n[r] = int(k.sum())
        if n[r] < min_n: continue
        jobs.append((r, t[k], c[k], WA, WC, d0)); Lr.append(np.full(n[r], r)); Lc.append(c[k]); Li.append(i[k])
    if JOBS > 1 and len(jobs) > 8:
        import multiprocessing as mp
        with mp.Pool(JOBS) as pool: fits = pool.starmap(fit_pair, [q[1:] for q in jobs], chunksize=8)
    else: fits = [fit_pair(*q[1:]) for q in jobs]
    keys = ('dbic', 'multi', 'resolved', 'd', 's', 's2', 's1', 'wmin', 'wb', 'sd_across', 'mc', 'sd_x', 'sd_y', 'mid_obs', 'mid_exp', 'mid_excess')
    Rr = {k: np.full(nr, np.nan) for k in keys}
    for (r, *_), f in zip(jobs, fits):
        for k in f:
            if k in Rr: Rr[k][r] = f[k]
        Rr['sd_x'][r] = np.sqrt((f['s'] * ax[r, 0]) ** 2 + (f['sd_across'] * ax[r, 1]) ** 2); Rr['sd_y'][r] = np.sqrt((f['s'] * ax[r, 1]) ** 2 + (f['sd_across'] * ax[r, 0]) ** 2)
    Lr = np.concatenate(Lr) if Lr else np.zeros(0, int); Li = np.concatenate(Li) if Li else np.zeros(0, int); Lc = np.concatenate(Lc) if Lc else np.zeros(0)
    M.update(n=n, ok=n >= min_n, R=Rr, W_r=Lr, W_c=Lc - Rr['mc'][Lr] if len(Lr) else Lc,
             W_sc={lv: np.sqrt((sx[Li] * ax[Lr, 1]) ** 2 + (sy[Li] * ax[Lr, 0]) ** 2) for lv, (sx, sy) in SGa.items()}, levels=list(SGa))
    return M


def summarize(M, wu, rsel, edges, d0):
    """wu: multiplicity per structure; rsel: pairs used"""
    R = M['R']; w = wu[M['ru']] * rsel; ri = np.repeat(np.arange(len(w)), w); o = dict(n_pairs=int(w.sum()))
    res = R['resolved'][ri] > 0; multi = R['multi'][ri] > 0
    o.update(frac_resolved=float(res.mean()) if len(ri) else np.nan, frac_multi=float(multi.mean()) if len(ri) else np.nan, locs_per_pair=float(np.median(M['n'][ri])) if len(ri) else np.nan)
    d = R['d'][ri]
    for nm, m in (('res', res), ('multi', multi), ('all', np.ones(len(ri), bool))):
        v = d[m]
        o[f'd_{nm}_mean'] = float(np.mean(v)) if len(v) else np.nan; o[f'd_{nm}_median'] = float(np.median(v)) if len(v) else np.nan
        o[f'd_{nm}_sd'] = float(np.std(v, ddof=1)) if len(v) > 2 else np.nan
    for k in ('s', 's1', 'sd_across', 'sd_x', 'sd_y', 'dbic'): o[k + '_median'] = float(np.nanmedian(R[k][ri])) if np.isfinite(R[k][ri]).any() else np.nan
    for k in ('mid_obs', 'mid_exp', 'mid_excess'): o[k + '_res'] = float(np.mean(R[k][ri][res])) if res.any() else np.nan
    o['s_res_median'] = float(np.nanmedian(R['s'][ri][res])) if res.any() else np.nan
    o['sd_across_multi_median'] = float(np.nanmedian(R['sd_across'][ri][multi])) if multi.any() else np.nan
    wp = wu[M['P_unit']]; pi = np.repeat(np.arange(len(wp)), wp); wl = wu[M['U_unit']]; li = np.repeat(np.arange(len(wl)), wl)
    ww = w[M['W_r']]; wi = np.repeat(np.arange(len(ww)), ww)
    o['n_locs'] = len(li); o['n_cpairs'] = len(pi); nb = len(edges) - 1
    pb = np.clip(np.searchsorted(edges, M['P_ph'][pi], side='right') - 1, 0, nb - 1); lb = np.clip(np.searchsorted(edges, M['U_ph'][li], side='right') - 1, 0, nb - 1)
    for a_, dd in (('x', M['P_dx']), ('y', M['P_dy'])): o['pair_' + a_] = float(L.rsd(dd[pi]) / np.sqrt(2)) if len(pi) > 20 else np.nan
    o['photons'] = float(np.median(M['U_ph'][li])) if len(li) else np.nan
    for lv in M['levels']:
        px_, py_ = M['P_s'][lv]; zx = M['P_dx'] / px_; zy = M['P_dy'] / py_; sx, sy = M['U_s'][lv]
        o[f'{lv}_sig_x'] = float(np.median(sx[li])) if len(li) else np.nan; o[f'{lv}_sig_y'] = float(np.median(sy[li])) if len(li) else np.nan
        o[f'{lv}_pairz_x'] = float(L.rsd(zx[pi])) if len(pi) > 20 else np.nan; o[f'{lv}_pairz_y'] = float(L.rsd(zy[pi])) if len(pi) > 20 else np.nan
        o[f'{lv}_pairz'] = float(np.sqrt(0.5 * (o[f'{lv}_pairz_x'] ** 2 + o[f'{lv}_pairz_y'] ** 2)))
        o[f'{lv}_zc'] = float(L.rsd(M['W_c'][wi] / M['W_sc'][lv][wi])) if len(wi) > 20 else np.nan
        o[f'{lv}_spread_over_sigma'] = float(np.sqrt(0.5 * (o['sd_x_median'] ** 2 + o['sd_y_median'] ** 2)) / (0.5 * (o[f'{lv}_sig_x'] + o[f'{lv}_sig_y'])))
        for i in range(nb):
            mp_ = pi[pb == i]; ml = li[lb == i]
            o[f'{lv}_bin{i}_pairz'] = float(np.sqrt(0.5 * (L.rsd(zx[mp_]) ** 2 + L.rsd(zy[mp_]) ** 2))) if len(mp_) > 20 else np.nan
            o[f'{lv}_bin{i}_sig'] = float(np.median(0.5 * (sx[ml] + sy[ml]))) if len(ml) else np.nan
        o[f'{lv}_trend_pairz'] = o[f'{lv}_bin{nb - 1}_pairz'] / o[f'{lv}_bin0_pairz']
    for i in range(nb):
        mp_ = pi[pb == i]; ml = li[lb == i]
        o[f'bin{i}_phot'] = float(np.median(M['U_ph'][ml])) if len(ml) else np.nan
        o[f'bin{i}_pair'] = float(np.sqrt(0.25 * (L.rsd(M['P_dx'][mp_]) ** 2 + L.rsd(M['P_dy'][mp_]) ** 2))) if len(mp_) > 20 else np.nan
    return o


def boot(fn, units, nu, nboot, seed=1):
    rng = np.random.default_rng(seed); est = fn(np.bincount(units, minlength=nu)); acc = {k: [] for k in est}
    for b in range(nboot):
        s = fn(np.bincount(rng.choice(units, len(units)), minlength=nu))
        for k in est: acc[k].append(s[k])
    ci = {k: ([float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(np.array(v, float)).any() else [np.nan, np.nan]) for k, v in acc.items()}
    return est, ci


def paired(MA, MB, wu, rsel):
    w = wu[MA['ru']] * rsel; ri = np.repeat(np.arange(len(w)), w); o = {}
    for k in ('sd_across', 's', 'sd_x', 'sd_y'):
        a, b = MA['R'][k][ri], MB['R'][k][ri]; g = np.isfinite(a) & np.isfinite(b)
        o['d_' + k] = float(np.median((a - b)[g])) if g.any() else np.nan
    a = np.sqrt(0.5 * (MA['R']['sd_x'][ri] ** 2 + MA['R']['sd_y'][ri] ** 2)); b = np.sqrt(0.5 * (MB['R']['sd_x'][ri] ** 2 + MB['R']['sd_y'][ri] ** 2)); g = np.isfinite(a) & np.isfinite(b)
    o['d_xy'] = float(np.median((a - b)[g])) if g.any() else np.nan; o['n_xy'] = int(g.sum())
    o['ratio_across'] = float(np.nanmedian(MA['R']['sd_across'][ri] / MB['R']['sd_across'][ri])) if len(ri) else np.nan
    o['d_resolved'] = float(np.mean(MA['R']['resolved'][ri] > 0) - np.mean(MB['R']['resolved'][ri] > 0)) if len(ri) else np.nan
    both = (MA['R']['resolved'][ri] > 0) & (MB['R']['resolved'][ri] > 0)
    o['d_dist_bothres'] = float(np.mean((MA['R']['d'] - MB['R']['d'])[ri][both])) if both.any() else np.nan
    def pr(M):
        wp = wu[M['P_unit']]; pi = np.repeat(np.arange(len(wp)), wp)
        return np.sqrt(0.25 * (L.rsd(M['P_dx'][pi]) ** 2 + L.rsd(M['P_dy'][pi]) ** 2)) if len(pi) > 20 else np.nan
    o['d_pair'] = float(pr(MA) - pr(MB))
    return o


F = lambda e, c, k, d=2: ('-' if k not in e or not np.isfinite(e[k]) else ('%.{0}f [%.{0}f, %.{0}f]'.format(d)) % (e[k], c[k][0], c[k][1]))


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True); tag = ('_' + a.tag) if a.tag else ''
    D = json.load(open(a.pairs)); rulers = D['rulers']; d0 = a.d0; pair_r = a.pair_r if a.pair_r > 0 else 0.75 * d0 + 15.0
    names = [m.split('=')[0] for m in a.methods]; files = [m.split('=', 1)[1] for m in a.methods]; Ms = {}
    for nm, fn in zip(names, files):
        Ms[nm] = eval_method(load_tab(fn), D, d0, a.levels, a.min_n, pair_r, shift=not a.no_shift)
        print(f'{nm}: {Ms[nm]["n_table"]} locs, shift {Ms[nm]["shift"][0]:.2f} {Ms[nm]["shift"][1]:.2f} nm, pairs with n >= {a.min_n}: {int(Ms[nm]["ok"].sum())} / {len(rulers)}', flush=True)
    common = np.all([M['ok'] for M in Ms.values()], 0); res = {}; ru = Ms[names[0]]['ru']; nu = Ms[names[0]]['nu']; allu = np.unique(np.array(D['site_unit']))
    for nm, M in Ms.items():
        rsel = M['ok'] if a.no_common else common
        e = np.quantile(M['U_ph'], np.linspace(0, 1, a.bins + 1)); e[0] = -np.inf; e[-1] = np.inf
        est, ci = boot(lambda wu: summarize(M, wu, rsel, e, d0), allu, nu, a.nboot, a.seed)
        res[nm] = dict(est=est, ci=ci, edges=[float(v) for v in e], shift=[float(v) for v in M['shift']], n_table=M['n_table'], n_ok=int(M['ok'].sum()),
                       frc_nm=L.frc_nm(M['T_x'], M['T_y'], a.seed, 2.5) if a.frc else float('nan'),
                       d_list=[float(v) for v in M['R']['d'][rsel]], resolved_list=[int(v) for v in M['R']['resolved'][rsel] > 0], multi_list=[int(v) for v in M['R']['multi'][rsel] > 0])
        print(f"{nm}: resolved {F(est, ci, 'frac_resolved', 3)}, d(res) {F(est, ci, 'd_res_mean')} sd {F(est, ci, 'd_res_sd')}, along {F(est, ci, 's_median')}, across {F(est, ci, 'sd_across_median')}, "
              f"pair {est['pair_x']:.2f} / {est['pair_y']:.2f}, pair z {F(est, ci, 'cal_pairz')}", flush=True)
    pr = {}
    for p in a.cmp:
        A_, B_ = p.split(':'); rs = (Ms[A_]['ok'] & Ms[B_]['ok']) if a.no_common else common
        est, ci = boot(lambda wu: paired(Ms[A_], Ms[B_], wu, rs), allu, nu, a.nboot, a.seed); pr[f'{A_} - {B_}'] = dict(est=est, ci=ci, n=int(rs.sum()))
    nb = a.bins; st = D['meta']['stats']
    md = [f'# Site-pair analysis{" " + a.tag if a.tag else ""} (d0 = {d0:g} nm)', '',
          f'pairs: {a.pairs} ({len(rulers)} pairs in {st["n_structures_with_pair"]} structures; {st["n_sites"]} sites); window +-{D["meta"]["WA"]:g} nm along, +-{D["meta"]["WC"]:g} nm across; min_n {a.min_n}; '
          f'consecutive-cycle pair radius {pair_r:g} nm; {a.nboot} bootstrap resamples over structures; ' + ('per-method valid pairs' if a.no_common else f'pairs valid in all methods: {int(common.sum())}') + '. nm; value [95 % CI].', '',
          '| method | rows | pairs | locs / pair window | resolved fraction | bimodal fraction | distance, resolved: mean | sd | distance, all pairs: median | spread along (bimodal pairs) | spread across | x | y | consecutive-cycle pair precision x / y | median sigma x / y | spread / sigma | pair z x / y | photons | FRC | midpoint-band excess, resolved pairs |',
          '|' + '---|' * 20]
    for nm, r in res.items():
        e, c = r['est'], r['ci']
        md.append('| ' + ' | '.join([nm, str(r['n_table']), str(e['n_pairs']), '%.0f' % e['locs_per_pair'], F(e, c, 'frac_resolved', 3), F(e, c, 'frac_multi', 3),
                                     F(e, c, 'd_res_mean'), F(e, c, 'd_res_sd'), F(e, c, 'd_all_median'), F(e, c, 's_median'), F(e, c, 'sd_across_median'), F(e, c, 'sd_x_median'), F(e, c, 'sd_y_median'),
                                     '%s / %s' % (F(e, c, 'pair_x'), F(e, c, 'pair_y')), '%.2f / %.2f' % (e['cal_sig_x'], e['cal_sig_y']), F(e, c, 'cal_spread_over_sigma'),
                                     '%s / %s' % (F(e, c, 'cal_pairz_x'), F(e, c, 'cal_pairz_y')), '%.0f' % e['photons'], '%.1f' % r['frc_nm'], '%s (obs %.3f, model %.3f)' % (F(e, c, 'mid_excess_res', 3), e['mid_obs_res'], e['mid_exp_res'])]) + ' |')
    for lv in ['cal'] + [l for l in a.levels if any(l in M['levels'] for M in Ms.values())]:
        md += ['', f'### Sigma level "{lv}": consecutive-cycle pair z (x, y combined) per photon bin (dimmest -> brightest); trend = brightest / dimmest', '',
               '| method | photons per bin | median sigma per bin | pair precision per bin | pair z per bin | pair z all | trend | across z (window, truncated) | spread / sigma |', '|---|---|---|---|---|---|---|---|---|']
        for nm, r in res.items():
            e, c = r['est'], r['ci']
            if f'{lv}_pairz' not in e: continue
            md.append('| ' + ' | '.join([nm, ' '.join('%.0f' % e[f'bin{i}_phot'] for i in range(nb)), ' '.join('%.2f' % e[f'{lv}_bin{i}_sig'] for i in range(nb)), ' '.join('%.2f' % e[f'bin{i}_pair'] for i in range(nb)),
                                         ' '.join('%.2f' % e[f'{lv}_bin{i}_pairz'] for i in range(nb)), F(e, c, f'{lv}_pairz'), F(e, c, f'{lv}_trend_pairz'), F(e, c, f'{lv}_zc'), F(e, c, f'{lv}_spread_over_sigma')]) + ' |')
    if pr:
        md += ['', '### Paired differences A - B over the same pairs (median over pairs; CI over structures)', '',
               '| pair | pairs | spread across | spread along (both bimodal) | xy (rms of x, y; both bimodal) | ratio across | resolved fraction | distance (both resolved) | consecutive-cycle pair precision |', '|---|---|---|---|---|---|---|---|---|']
        for k, p in pr.items():
            e, c = p['est'], p['ci']
            md.append('| ' + ' | '.join([k, str(p['n']), F(e, c, 'd_sd_across', 3), F(e, c, 'd_s', 3), F(e, c, 'd_xy', 3), F(e, c, 'ratio_across', 3), F(e, c, 'd_resolved', 3), F(e, c, 'd_dist_bothres'), F(e, c, 'd_pair', 3)]) + ' |')
    open(f'{a.out}/rul{tag}.md', 'w').write('\n'.join(md) + '\n')
    json.dump(dict(args={k: v for k, v in vars(a).items() if k != 'fn'}, n_common=int(common.sum()), methods=res, paired=pr), open(f'{a.out}/rul{tag}.json', 'w'), indent=1, default=float)
    cols = dict(pair=np.arange(len(rulers)), unit=ru, common=common.astype(int))
    for nm, M in Ms.items():
        cols[nm + '_n'] = M['n']
        for k, v in M['R'].items(): cols[f'{nm}_{k}'] = v
    np.savez(f'{a.out}/per_pair{tag}.npz', **cols)
    print('\n'.join(md))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('pairs'); s.add_argument('--ref', default=''); s.add_argument('--truth', default=''); s.add_argument('--picks', default=''); s.add_argument('--out', required=True)
    s.add_argument('--d0', type=float, required=True); s.add_argument('--rs', type=float, default=0.0); s.add_argument('--blur', type=float, default=0.0); s.add_argument('--min_locs', type=int, default=30)
    s.add_argument('--edge', type=float, default=500.0); s.add_argument('--fov', type=float, nargs=2, default=[0.0, 26000.0]); s.set_defaults(fn=cmd_pairs)
    s = sp.add_parser('run'); s.add_argument('--pairs', required=True); s.add_argument('--methods', nargs='+', required=True); s.add_argument('--out', required=True); s.add_argument('--d0', type=float, required=True)
    s.add_argument('--tag', default=''); s.add_argument('--min_n', type=int, default=40); s.add_argument('--bins', type=int, default=5)
    s.add_argument('--nboot', type=int, default=300); s.add_argument('--seed', type=int, default=1); s.add_argument('--no_common', action='store_true'); s.add_argument('--no_shift', action='store_true')
    s.add_argument('--levels', nargs='*', default=['raw']); s.add_argument('--cmp', nargs='*', default=[]); s.add_argument('--pair_r', type=float, default=0.0); s.add_argument('--frc', type=int, default=0)
    s.set_defaults(fn=cmd_run)
    a = ap.parse_args(); a.fn(a)
