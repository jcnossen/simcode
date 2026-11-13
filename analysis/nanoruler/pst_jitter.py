"""
Parts 2 and 3: noise model / excess relative intensity variance (c) of the fully-on in-situ intensities under the shared and
the per-step pattern, and cycle-to-cycle phase and amplitude jitter per step.

Leave-one-out residual of datum (sample b, frame f), amplitude group g = (b, orientation):
    A_-f = sum_{o != f} w_o x_o I_o / sum_{o != f} w_o x_o^2       (x = unit-amplitude model, w = 1 / V)
    r    = I_f - A_-f x_f,     Var r = V_f + x_f^2 / sum_{o != f} w_o x_o^2
Noise model V(mu) = a0 + a1 mu + c^2 mu^2 (mu = A x): Gaussian ML on r over all data (plain, and 4-sigma trimmed), bootstrap
over binding sites.
Jitter: per datum d = r / (A_-f y) (y = d x / d phase = rel m sin(theta)) estimates the phase error delta of its frame with
variance v = Var r / (A_-f y)^2; the product of d of two DIFFERENT emitters in the same frame is an unbiased estimate of
delta^2 whatever the noise model (weights only): U-statistic
    var_true = sum_f [(sum w d)^2 - sum w^2 d^2] / sum_f [(sum w)^2 - sum w^2].
Nulls: the cycle label of every datum shuffled (within the same step) locally (+-25 cycles: keeps slow model errors) and
globally. Same for the amplitude factor (d = r / (A_-f x)).
Common-mode phase jitter of the three steps of an orientation in one cycle: exact 3-step solve per emitter-cycle (leak-free).
usage: python analysis/nanoruler/pst_jitter.py <shared cfg> <perstep cfg>   (names in fits.pt)
"""
import sys, json, math
import numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L
from simcode2.util.proctitle import set_title

PITCH = np.array([219.9, 219.9, 219.9, 219.5, 219.5, 219.5])
RAD2NM = PITCH / (2 * math.pi)


def model_xy(D, P):
    t = lambda v, dt=torch.float64: torch.as_tensor(v, dtype=dt)
    with torch.no_grad():
        x, y, th = P.x(t(D['r']), t(D['fr'], torch.long), t(D['s'], torch.long), deriv=True)
    return x.numpy(), y.numpy(), th.numpy()


def V_of(mu, p):
    return np.maximum(p[0] + p[1] * mu + p[2] * mu * mu, 1e-3)


def loo(I, x, g, p, wt=None, iters=4):
    """leave-one-out residuals for noise parameters p. Returns dict(r, var, A_loo, A, ok, mu)."""
    ng = g.max() + 1
    A = np.bincount(g, x * I, ng) / np.maximum(np.bincount(g, x * x, ng), 1e-12)
    for _ in range(iters):
        w = 1.0 / V_of(np.maximum(A[g], 1.0) * x, p)
        A = np.bincount(g, w * x * I, ng) / np.maximum(np.bincount(g, w * x * x, ng), 1e-30)
    A = np.maximum(A, 1.0)
    mu = A[g] * x; V = V_of(mu, p); w = 1.0 / V
    Sxi = np.bincount(g, w * x * I, ng)[g] - w * x * I; Sxx = np.bincount(g, w * x * x, ng)[g] - w * x * x
    cnt = np.bincount(g, minlength=ng)[g]
    ok = (cnt >= 2) & (Sxx > 1e-12)
    Al = np.where(ok, Sxi / np.where(ok, Sxx, 1), 1.0)
    r = I - Al * x; var = V + x * x / np.where(ok, Sxx, 1)
    return dict(r=r, var=var, Al=Al, A=A[g], ok=ok, mu=mu, V=V)


def fit_noise(I, x, g, wt=None, trim=None, fix_a1=None, p0=(800.0, 1.0, 0.01), keep0=None, maxiter=600):
    """Gaussian ML of (a0, a1, c^2) on the LOO residuals. wt: bootstrap multiplicities [n]. trim: keep |r| / sd < trim under
    the current fit (iterated), with the Gaussian truncation correction of the variance."""
    wt = np.ones(len(I)) if wt is None else wt
    keep = np.ones(len(I), bool) if keep0 is None else keep0
    corr = 1.0
    if trim is not None:
        from scipy.stats import norm
        corr = 1 - 2 * trim * norm.pdf(trim) / (2 * norm.cdf(trim) - 1)       # var of a truncated standard normal

    def unpack(q):
        return (abs(q[0]), fix_a1, abs(q[1])) if fix_a1 is not None else (abs(q[0]), abs(q[1]), abs(q[2]))

    def nll(q, keep):
        o = loo(I, x, g, unpack(q))
        k = keep & o['ok']; var = o['var'][k] * corr
        return (wt[k] * (np.log(var) + o['r'][k] ** 2 / var)).sum() / wt[k].sum()

    q = np.array([p0[0], p0[2]] if fix_a1 is not None else p0)
    for rnd in range(1 if (trim is None or keep0 is not None) else 4):
        q = minimize(nll, q, args=(keep,), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-7, maxiter=maxiter)).x
        if trim is not None and keep0 is None:
            o = loo(I, x, g, unpack(q)); keep = np.abs(o['r']) < trim * np.sqrt(o['var'])
    return np.array(unpack(q)), keep


def mats(d, w, key, nkey, site):
    """sparse [nkey, nsite] sums of w d and w (data of one binding site in one key are merged: pairs are formed between
    DIFFERENT sites only, so per-emitter errors (position, amplitude) never enter a product)."""
    from scipy.sparse import csr_matrix
    ns = int(site.max()) + 1
    return (csr_matrix((w * d, (key, site)), shape=(nkey, ns)), csr_matrix((w, (key, site)), shape=(nkey, ns)))


def pairstat(Ms, Mw, lag=0, Ms2=None, Mw2=None):
    """per-key numerator / denominator of the pairwise estimator of the covariance of the common term at keys f and f + lag
    (different sites only). Ms2 / Mw2: a second data set for the partner (default the same)."""
    Ms2 = Ms if Ms2 is None else Ms2; Mw2 = Mw if Mw2 is None else Mw2
    n = Ms.shape[0] - lag
    a, b, wa, wb = Ms[:n], Ms2[lag:lag + n], Mw[:n], Mw2[lag:lag + n]
    s = lambda M: np.asarray(M.sum(1)).ravel()
    num = s(a) * s(b) - s(a.multiply(b)); den = s(wa) * s(wb) - s(wa.multiply(wb))
    return num, den


def ustat(d, w, key, nkey, site):
    Ms, Mw = mats(d, w, key, nkey, site)
    num, den = pairstat(Ms, Mw)
    return num, den, np.asarray(Ms.sum(1)).ravel(), np.asarray(Mw.sum(1)).ravel()


def boot_ratio(num, den, block=10, nb=400, seed=0):
    """ratio sum(num)/sum(den) with a block bootstrap over the index (cycles): value, se, (2.5, 97.5) percentiles."""
    n = len(num) // block * block
    N = num[:n].reshape(-1, block).sum(1); Q = den[:n].reshape(-1, block).sum(1)
    rng = np.random.default_rng(seed); ix = rng.integers(0, len(N), (nb, len(N)))
    bs = N[ix].sum(1) / Q[ix].sum(1)
    return float(num.sum() / den.sum()), float(bs.std()), [float(v) for v in np.percentile(bs, [2.5, 97.5])]


def sd_ci(v, se, ci):
    f = lambda u: float(np.sign(u) * np.sqrt(abs(u)))
    return dict(var=v, var_se=se, var_ci=ci, sd=f(v), sd_ci=[f(ci[0]), f(ci[1])], sd_upper95=f(max(v + 1.645 * se, 0)))


def jitter_analysis(I, x, y, th, g, s, fr, site, p, rng, tag='', m6=None):
    """all jitter estimates for one data realisation. Returns dict."""
    o = loo(I, x, g, p)
    cyc = fr // 6; ncyc = L.NFRAMES // 6
    out = {}
    for kind in ('phase', 'amp'):
        a = o['Al'] * (y if kind == 'phase' else x)
        ok = o['ok'] & (o['Al'] > 20) & (np.abs(a) > 1e-3)
        if kind == 'phase': ok &= np.abs(y) > 0.15                     # away from the extrema: d is ~ linear in delta there
        d = np.where(ok, o['r'] / np.where(ok, a, 1), 0); v = np.where(ok, o['var'] / np.where(ok, a, 1) ** 2, 1); w = ok / v
        res = {}
        for st in list(range(6)) + ['o0', 'o1', 'all']:
            sel = ok & ((s == st) if isinstance(st, int) else (s // 3 == int(st[1])) if st != 'all' else True)
            key = fr[sel] if not isinstance(st, int) else cyc[sel]; nkey = L.NFRAMES if not isinstance(st, int) else ncyc
            num, den, swd, sw = ustat(d[sel], w[sel], key, nkey, site[sel])
            if not isinstance(st, int):                                   # block over cycles
                num = num.reshape(-1, 6).sum(1); den = den.reshape(-1, 6).sum(1)
            e = sd_ci(*boot_ratio(num, den))
            if not isinstance(st, int):
                # pooled over steps: null (frame label shuffled within the same step) and same-step lag covariances
                Ms, Mw = mats(d[sel], w[sel], fr[sel], L.NFRAMES, site[sel])
                e['lag_cov'] = {Lg: boot_ratio(*pairstat(Ms, Mw, 6 * Lg), block=60)[:2] for Lg in (1, 2, 3, 5, 10, 20, 50, 100)}
                for nm, width in (('null_local', 25), ('null_global', None)):
                    vals = []
                    for rep in range(10):
                        kc = rng.integers(0, ncyc, sel.sum()) if width is None else np.clip(cyc[sel] + rng.integers(-width, width + 1, sel.sum()), 0, ncyc - 1)
                        n2, d2, _, _ = ustat(d[sel], w[sel], kc * 6 + s[sel], L.NFRAMES, site[sel]); vals.append(n2.sum() / d2.sum())
                    e[nm] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
            # per-frame estimates: total variance and mean se^2 (the classical decomposition, frames with >= 1 emitter)
            if isinstance(st, int):
                has = sw > 0; dh = swd[has] / sw[has]; se2 = 1 / sw[has]
                chi = dh ** 2 / se2
                e.update(n_frames=int(has.sum()), n_frames_pairs=int((den > 0).sum()), emitters_per_frame=float(sel.sum() / ncyc),
                         median_se=float(np.sqrt(np.median(se2))), var_total_w=float(np.average(dh ** 2, weights=1 / se2 ** 2)),
                         # weighted moment estimator: sum w^2 (d^2 - se^2) / sum w^2  (w = 1 / se^2)
                         var_minus_se2=float(np.sum((dh ** 2 - se2) / se2 ** 2) / np.sum(1 / se2 ** 2)),
                         var_minus_se2_se=float(np.sqrt(2 * np.sum(1 / se2 ** 2)) / np.sum(1 / se2 ** 2)),
                         chi2_red=float(chi.mean()), chi2_red_median=float(np.median(chi) / 0.4549))
                # nulls
                for nm, width in (('null_local', 25), ('null_global', None)):
                    vals = []
                    for rep in range(20):
                        if width is None: kk = rng.integers(0, ncyc, sel.sum())
                        else: kk = np.clip(cyc[sel] + rng.integers(-width, width + 1, sel.sum()), 0, ncyc - 1)
                        n2, d2, _, _ = ustat(d[sel], w[sel], kk, ncyc, site[sel])
                        vals.append(n2.sum() / d2.sum())
                    e[nm] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
                # two disjoint halves of the sites (3 splits): correlation and covariance of the per-frame estimates
                hh = []
                for sp in range(3):
                    h = np.random.default_rng(100 + sp).permutation(site.max() + 1) % 2
                    e1 = []
                    for half in (0, 1):
                        q = sel & (h[site] == half)
                        e1.append((np.bincount(cyc[q], w[q] * d[q], ncyc), np.bincount(cyc[q], w[q], ncyc)))
                    both = (e1[0][1] > 0) & (e1[1][1] > 0)
                    d1 = e1[0][0][both] / e1[0][1][both]; d2 = e1[1][0][both] / e1[1][1][both]; q = e1[0][1][both] * e1[1][1][both]
                    cov = np.sum(q * d1 * d2) / q.sum()
                    # weighted correlation
                    cor = cov / math.sqrt(np.sum(q * d1 * d1) / q.sum() * np.sum(q * d2 * d2) / q.sum())
                    hh.append(dict(cov=float(cov), corr=float(cor), n=int(both.sum())))
                e['halves'] = hh
                # lags (cycles), leak-free variant for lag 1: the amplitude of b must not use the partner frame
                lag = {}
                Ms, Mw = mats(d[sel], w[sel], cyc[sel], ncyc, site[sel])
                for Lg in (1, 2, 3, 5, 10, 20, 50, 100):
                    lag[Lg] = boot_ratio(*pairstat(Ms, Mw, Lg))[:2]
                if m6 is not None:
                    nn = sel & ~m6['next']; pp = sel & ~m6['prev']
                    M1 = mats(d[nn], w[nn], cyc[nn], ncyc, site[nn]); M2 = mats(d[pp], w[pp], cyc[pp], ncyc, site[pp])
                    lag['1_leakfree'] = boot_ratio(*pairstat(M1[0], M1[1], 1, M2[0], M2[1]))[:2]
                e['lag_cov'] = lag
            res[str(st)] = e
        out[kind] = res
    return out, o


def common_mode(I, P, D, V_p, rng, inject=None):
    """3-step solve per (sample, cycle, orientation) with all three steps fully on: common phase offset of the cycle.
    Returns per orientation the pairwise (U-statistic) variance of the common-mode phase jitter + nulls + lags."""
    x, y, th = model_xy(D, P)
    rel = np.exp(P.logrel.numpy() - P.logrel.numpy().mean()); m = P.m.numpy()
    b, s, fr = D['b'], D['s'], D['fr']; cyc = fr // 6; ncyc = L.NFRAMES // 6
    key = (b.astype(np.int64) * ncyc + cyc) * 2 + s // 3
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    full3 = cnt[inv] == 3
    order = np.lexsort((s, key)); order = order[full3[order]]
    ii = order.reshape(-1, 3)                                  # [n3, 3] data indices, steps ascending
    Iv = I[ii]; ths = th[ii]; ss = s[ii]
    M = np.stack([np.ones_like(ths), m[ss] * np.cos(ths), m[ss] * np.sin(ths)], -1) * rel[ss][..., None]     # I = M (A, C, S)
    det = np.linalg.det(M); good = np.abs(det) > 0.05
    Mi = np.linalg.inv(np.where(good[:, None, None], M, np.eye(3)))
    sol = np.einsum('nij,nj->ni', Mi, Iv); A, C, S = sol.T
    good &= (A > 20) & (np.hypot(C, S) > 0.3 * A)
    dl = np.arctan2(S, C)
    mu = np.maximum(A[:, None], 1) * (M[..., 0] + M[..., 1])       # prediction at delta = 0
    J = (C[:, None] * Mi[:, 2, :] - S[:, None] * Mi[:, 1, :]) / np.maximum(C ** 2 + S ** 2, 1e-9)[:, None]
    v = (J ** 2 * V_of(np.maximum(mu, 1), V_p)).sum(1)
    good &= (v < 0.5 ** 2)
    w = good / np.maximum(v, 1e-9); d = np.where(good, dl, 0)
    orient = ss[:, 0] // 3; cy = cyc[ii[:, 0]]; st3 = D['site'][ii[:, 0]]
    out = {}
    for a in (0, 1):
        q = good & (orient == a)
        num, den, swd, sw = ustat(d[q], w[q], cy[q], ncyc, st3[q])
        e = sd_ci(*boot_ratio(num, den)); e['n_emitter_cycles'] = int(q.sum()); e['median_se_single'] = float(np.sqrt(np.median(v[q])))
        for nm, width in (('null_local', 25), ('null_global', None)):
            vals = []
            for rep in range(20):
                kk = rng.integers(0, ncyc, q.sum()) if width is None else np.clip(cy[q] + rng.integers(-width, width + 1, q.sum()), 0, ncyc - 1)
                n2, d2, _, _ = ustat(d[q], w[q], kk, ncyc, st3[q]); vals.append(n2.sum() / d2.sum())
            e[nm] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)))
        lag = {}
        Ms, Mw = mats(d[q], w[q], cy[q], ncyc, st3[q])
        for Lg in (1, 2, 3, 5, 10, 20, 50, 100):
            lag[Lg] = boot_ratio(*pairstat(Ms, Mw, Lg))[:2]
        e['lag_cov'] = lag
        out[f'o{a}'] = e
    return out


def implied_c(m, delta, sel_frac=0.25):
    """rms relative per-frame intensity error m sin(th) delta / (1 + m cos th) for phase error sd delta, averaged over theta:
    uniform, photon-weighted, and over the fully-on selection (1 + m cos th > sel_frac)."""
    th = np.linspace(0, 2 * np.pi, 200001)[:-1]
    x = 1 + m * np.cos(th); e2 = (m * np.sin(th) / x) ** 2
    return dict(uniform=float(delta * np.sqrt(e2.mean())), photon_weighted=float(delta * np.sqrt((e2 * x).sum() / x.sum())),
                selected=float(delta * np.sqrt(e2[x > sel_frac].mean())),
                factor_uniform=float(np.sqrt(e2.mean())), factor_photon_weighted=float(np.sqrt((e2 * x).sum() / x.sum())),
                factor_selected=float(np.sqrt(e2[x > sel_frac].mean())))


if __name__ == '__main__':
    set_title('nr-pstep'); torch.set_num_threads(4)
    cs, cp = sys.argv[1], sys.argv[2]
    F = torch.load(L.OUT + 'fits.pt', weights_only=False)
    D = L.load_data(); n = len(D['I']); rng = np.random.default_rng(1)
    I, g, s, fr, site = D['I'], np.unique(D['g'], return_inverse=True)[1], D['s'], D['fr'], D['site']
    # same-step neighbours in the group (for the leak-free lag 1)
    gk = g.astype(np.int64) * 40000 + fr
    m6 = dict(next=np.isin(gk + 6, gk), prev=np.isin(gk - 6, gk))
    R = dict(shared_cfg=cs, perstep_cfg=cp, n=n)
    pats = {nm: L.Pattern.from_state(F[c]['full']['state']) for nm, c in (('shared', cs), ('perstep', cp))}
    xy = {nm: model_xy(D, P) for nm, P in pats.items()}

    # ---------- part 3: noise model / excess variance under both patterns
    noise = {}; keeps = {}
    nsite = site.max() + 1
    for nm in pats:
        x = xy[nm][0]; e = {}
        for var, kw in (('ml', {}), ('ml_trim4', dict(trim=4.0)), ('ml_trim4_a1eq1', dict(trim=4.0, fix_a1=1.0))):
            p, keep = fit_noise(I, x, g, **kw)
            if var == 'ml_trim4': keeps[nm] = keep
            e[var] = dict(a0=float(p[0]), a1=float(p[1]), c2=float(p[2]), c=float(np.sqrt(p[2])), kept=float(keep.mean()))
        noise[nm] = e
        print(nm, json.dumps(e), flush=True)
    # bootstrap over sites (paired between the two patterns), trimmed ML
    bs = {nm: [] for nm in pats}
    for rep in range(60):
        wt = np.bincount(rng.integers(0, nsite, nsite), minlength=nsite)[site].astype(float)
        for nm in pats:
            t = noise[nm]['ml_trim4']
            p, _ = fit_noise(I, xy[nm][0], g, wt=wt, trim=4.0, p0=(t['a0'], t['a1'], t['c2']), keep0=keeps[nm], maxiter=120)
            bs[nm].append(p)
    for nm in pats:
        b = np.array(bs[nm]); noise[nm]['ml_trim4_boot'] = dict(a0_se=float(b[:, 0].std()), a1_se=float(b[:, 1].std()),
                                                               c_se=float(np.sqrt(b[:, 2]).std()), c2_se=float(b[:, 2].std()))
    dc = np.sqrt(np.array(bs['shared'])[:, 2]) - np.sqrt(np.array(bs['perstep'])[:, 2])
    dc2 = np.array(bs['shared'])[:, 2] - np.array(bs['perstep'])[:, 2]
    noise['shared_minus_perstep'] = dict(c=float(noise['shared']['ml_trim4']['c'] - noise['perstep']['ml_trim4']['c']), c_se=float(dc.std()),
                                         c2=float(noise['shared']['ml_trim4']['c2'] - noise['perstep']['ml_trim4']['c2']), c2_se=float(dc2.std()))
    print(json.dumps(noise['shared_minus_perstep']), flush=True)
    # binned relative variance for the figure: (r / mu)^2 minus the amplitude term vs predicted photons
    bins = np.array([40, 80, 120, 160, 200, 250, 300, 350, 400, 500, 600, 700, 850, 1000, 1300, 2000])
    for nm in pats:
        t = noise[nm]['ml_trim4']; p = (t['a0'], t['a1'], t['c2'])
        o = loo(I, xy[nm][0], g, p); k = o['ok'] & (np.abs(o['r']) < 4 * np.sqrt(o['var']))
        mu = o['Al'] * xy[nm][0]; rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            q = k & (o['mu'] >= lo) & (o['mu'] < hi)
            if q.sum() < 200: continue
            excess = (o['r'][q] ** 2 - (o['var'][q] - o['V'][q])) / o['mu'][q] ** 2          # ~ V / mu^2 per datum
            rows.append([float(np.mean(o['mu'][q])), float(excess.mean()), float(excess.std() / math.sqrt(q.sum())), int(q.sum()),
                         float(np.mean(o['V'][q] / o['mu'][q] ** 2))])
        noise[nm]['binned'] = rows
    R['noise'] = noise

    # ---------- part 2: jitter (per-step pattern = the best smooth model; shared pattern for reference)
    jit = {}
    for nm in pats:
        t = noise[nm]['ml_trim4']; p = (t['a0'], t['a1'], t['c2'])
        x, y, th = xy[nm]
        jit[nm], o = jitter_analysis(I, x, y, th, g, s, fr, site, p, rng, m6=m6)
        jit[nm]['common_mode'] = common_mode(I, pats[nm], D, p, rng)
        # photon-noise-only standard errors (c = 0): the classical var - se^2 then contains the excess noise too
        jp, _ = jitter_analysis(I, x, y, th, g, s, fr, site, (p[0], p[1], 0.0), rng, m6=m6)
        for kind in ('phase', 'amp'):
            for st in range(6):
                jit[nm][kind][str(st)]['var_minus_se2_photon_only'] = jp[kind][str(st)]['var_minus_se2']
        print(nm, 'phase jitter var per step (U-stat):', [f"{jit[nm]['phase'][str(st)]['var']:.2e}+-{jit[nm]['phase'][str(st)]['var_se']:.1e}" for st in range(6)], flush=True)
        print(nm, 'amp jitter var per step:', [f"{jit[nm]['amp'][str(st)]['var']:.2e}+-{jit[nm]['amp'][str(st)]['var_se']:.1e}" for st in range(6)], flush=True)
        print(nm, 'common mode:', {k: (v['var'], v['var_se'], v['null_local']['mean']) for k, v in jit[nm]['common_mode'].items()}, flush=True)
    R['jitter'] = jit

    # ---------- semi-synthetic validation of the estimators (per-step pattern, fitted amplitudes, noise model)
    nm = 'perstep'; t = noise[nm]['ml_trim4']; p = (t['a0'], t['a1'], t['c2']); x, y, th = xy[nm]
    P = pats[nm]; rel = np.exp(P.logrel.numpy() - P.logrel.numpy().mean()); mm = P.m.numpy()
    o = loo(I, x, g, p); A = o['A']; cyc = fr // 6; ncyc = L.NFRAMES // 6
    val = {}
    for case, (sd_ind, sd_com, sd_amp) in dict(none=(0, 0, 0), indep_0p05=(0.05, 0, 0), common_0p05=(0, 0.05, 0), amp_0p05=(0, 0, 0.05),
                                              indep_0p03_amp_0p03=(0.03, 0, 0.03)).items():
        r2 = np.random.default_rng(7)
        dl = sd_ind * r2.standard_normal(L.NFRAMES)[fr] + sd_com * r2.standard_normal((ncyc, 2))[cyc, s // 3]
        ga = sd_amp * r2.standard_normal(L.NFRAMES)[fr]
        mu = A * rel[s] * (1 + mm[s] * np.cos(th - dl)) * (1 + ga)
        Is = mu + np.sqrt(V_of(mu, p)) * r2.standard_normal(n)
        # the same fully-on selection is NOT re-applied (truncation at 0.25 median is in the real data only)
        j, _ = jitter_analysis(Is, x, y, th, g, s, fr, site, p, r2, m6=m6)
        cm = common_mode(Is, P, D, p, r2)
        val[case] = dict(truth=dict(sd_indep=sd_ind, sd_common=sd_com, sd_amp=sd_amp),
                         phase_var=[j['phase'][str(st)]['var'] for st in range(6)], phase_var_se=[j['phase'][str(st)]['var_se'] for st in range(6)],
                         amp_var=[j['amp'][str(st)]['var'] for st in range(6)], amp_var_se=[j['amp'][str(st)]['var_se'] for st in range(6)],
                         phase_null_local=[j['phase'][str(st)]['null_local']['mean'] for st in range(6)],
                         amp_null_local=[j['amp'][str(st)]['null_local']['mean'] for st in range(6)],
                         phase_lag1=[j['phase'][str(st)]['lag_cov'][1][0] for st in range(6)],
                         phase_lag1_leakfree=[j['phase'][str(st)]['lag_cov']['1_leakfree'][0] for st in range(6)],
                         phase_var_minus_se2=[j['phase'][str(st)]['var_minus_se2'] for st in range(6)],
                         common_var=[cm['o0']['var'], cm['o1']['var']], common_var_se=[cm['o0']['var_se'], cm['o1']['var_se']])
        print('validation', case, 'phase', np.round(np.mean(val[case]['phase_var']), 6), 'amp', np.round(np.mean(val[case]['amp_var']), 6),
              'common', np.round(val[case]['common_var'], 6), 'lag1', np.round(np.mean(val[case]['phase_lag1']), 6),
              'lag1 leakfree', np.round(np.mean(val[case]['phase_lag1_leakfree']), 6), flush=True)
    R['validation'] = val
    R['implied_factor'] = {f'm_{m_}': implied_c(m_, 1.0) for m_ in (0.8, 0.85, 0.88, 0.9, 0.92)}
    json.dump(R, open(L.OUT + f'jitter{sys.argv[3] if len(sys.argv) > 3 else ""}.json', 'w'), indent=1)
    print('saved jitter.json')
