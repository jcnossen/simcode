"""Site-size study, analysis library (CPU). Per binding site: the spread of a method's localizations, the pair precision as a function of
the cycle gap Delta, a same-event vs different-event comparison at equal Delta, the saturating-exponential fit, bootstrap over sites.

Definitions (one set for every method and movie):
  assignment   one global shift (median offset to the nearest site, as nra_lib), nearest site within R (80 nm: 30 nm as the method tables;
               40 nm: d0 / 2 = 20 nm against ALL sites of the pair list), one re-centring on the method's own per-site median, re-assignment.
               Used sites: the sites of the method tables (80 nm: sites.json; 40 nm: the sites of the auto_sf pair list), n >= MIN_N in the method.
  axes         along / across = projection on the ruler axis (40 nm: the axis of the site's first pair); 'xy' = along and across pooled.
  spread       per site: robust sd (1.4826 MAD about the site median, divided by sqrt(1 - 0.43 / n)), median over sites (= the method tables).
  events       localizations of a site sorted by cycle; a new binding event starts after a gap of more than GAP cycles.
  pair prec.   all pairs i < j of localizations of one site with cycle gap dt in a Delta bin (contiguous log bins 1 | 2 | 3-7 | 8-14 | 15-31 |
               32-70 | 71-173 | 174-547 | 548-1414 | 1415-2828); sigma_pair = 1.4826 median |d| / sqrt 2 of the difference d (symmetric, so the
               median of d is 0). Each site enters a bin with weight 1 (its pairs in that bin normalized) if it has >= MIN_P pairs there in
               EVERY method of the movie (bin mask common to the methods).
               'all' = every pair with dt >= 1: the pair-weighted analogue of the spread (sqrt 2 deconvolved), the reference for the plateau.
  event test   interior localizations only (not the first / last cycle of their event; partially-on windows are worse and an inter-event pair
               at small dt would be made of event ends), same Delta bin: var(diff-event) - var(same-event) (bins 6-8 | 9-12 | 13-20 | 21-40 | 41-100; headline = mean of the first four) = a position change between
               binding events of one site beyond what time alone gives (sites with >= 3 pairs of each class in the bin, equal site weights).
  fit          sigma_pair(Delta)^2 = sigma_loc^2 + s_slow^2 (1 - exp(-Delta / tau)), weighted least squares on the variances (weights from
               the bootstrap), log tau in [log 0.3, log 1e4]; plateau = sqrt(sigma_loc^2 + s_slow^2); s_site = sign(e) sqrt(|e|),
               e = spread_all^2 - plateau^2 (also against the median per-site spread).
  bootstrap    sites resampled with replacement (percentile 95 %), the fit redone per replicate."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nra_lib as L

GAP = 3; MIN_N = 10; MIN_P = 5; DW = 0.1; NDB = 1000          # |d| histogram: 0.1 nm bins up to 100 nm (overflow in the last bin)
DELTAS = [1, 2, 5, 10, 20, 50, 100, 300, 1000, 2000]


def dbins(ncyc):
    """contiguous bins of the cycle gap around the nominal Delta values (boundaries at the geometric midpoints), up to ncyc / 2"""
    e = [1, 2, 3, 8, 15, 32, 71, 174, 548, 1415, 2829]          # bin i = [e[i], e[i+1] - 1]
    return [(D, e[i], e[i + 1] - 1) for i, D in enumerate(DELTAS) if e[i + 1] - 1 <= 0.5 * ncyc]


EBINS = [(6, 8), (9, 12), (13, 20), (21, 40), (41, 100)]
NE_HEAD = 4                                                      # s_event headline: mean of the first 4 bins (6-40 cycles)


def rsd_defl(v):
    n = len(v); return L.rsd(v) / np.sqrt(max(1 - L.KMED / n, 0.1))


def assign_sites(tab, sc, used, radius, rmax):
    sh = L.global_shift(tab['x_nm'], tab['y_nm'], sc, rmax=rmax); x = tab['x_nm'] - sh[0]; y = tab['y_nm'] - sh[1]
    g, j = L.assign(x, y, sc, radius); c1 = sc.copy()
    for s in np.unique(j): c1[s] = [np.median(x[g[j == s]]), np.median(y[g[j == s]])]
    g, j = L.assign(x, y, c1, radius)
    k = used[j]; return x, y, g[k], j[k], sh


def ar1(rng, n, sd, tau):
    r = np.exp(-1.0 / tau); e = rng.randn(n) * sd * np.sqrt(1 - r * r); o = np.zeros(n); o[0] = rng.randn() * sd
    for t in range(1, n): o[t] = r * o[t - 1] + e[t]
    return o


ZSCALE = 10.0     # mode 'z': the stored value is sqrt 2 x 10 x |d| / sqrt(s_i^2 + s_j^2), so sigma_pair / 10 = robust sd of the pair z


def per_site(tab, S, ncyc, inj=None, mode='all'):
    """tab: nra table; S: dict(sc [ns,2], axis [ns,2], used bool[ns], radius, rmax). inj: dict(ev_sd, ar_sd, ar_tau, seed) synthetic check.
    mode: 'all'; 'interior' = only localizations that are not the first / last cycle of their event (fully-on windows, removes the
    change of the photon / on-fraction composition with Delta); 'z' = pair differences divided by sqrt(s_i^2 + s_j^2) of the reported
    sigma (ZSCALE). Returns dict of per-site arrays + pair histograms."""
    tab = dict(tab)
    if inj:
        rng = np.random.RandomState(inj.get('seed', 1)); f = tab['frame']
        if inj.get('ar_sd', 0) > 0:
            tab['x_nm'] = tab['x_nm'] + ar1(rng, ncyc, inj['ar_sd'], inj['ar_tau'])[f]; tab['y_nm'] = tab['y_nm'] + ar1(rng, ncyc, inj['ar_sd'], inj['ar_tau'])[f]
    x, y, g, j, sh = assign_sites(tab, S['sc'], S['used'], S['radius'], S['rmax'])
    DB = dbins(ncyc); nb = len(DB) + 1; ne = len(EBINS); ns = len(S['sc'])
    uidx = np.nonzero(S['used'])[0]; pos = np.full(ns, -1); pos[uidx] = np.arange(len(uidx))      # histograms only for the used sites
    H = np.zeros((len(uidx), nb, 2, 2, NDB), np.int32); HE = np.zeros((len(uidx), ne, 2, 2, NDB), np.int32)
    n = np.bincount(j, minlength=ns); st = {k: np.full(ns, np.nan) for k in ('rsd_x', 'rsd_y', 'rsd_along', 'rsd_across', 'n_events', 'photons')}
    lut = np.full(max(hi for _, _, hi in DB) + 2, -1, np.int64)
    for b, (_, lo, hi) in enumerate(DB): lut[lo:hi + 1] = b
    elut = np.full(EBINS[-1][1] + 2, -1, np.int64)
    for b, (lo, hi) in enumerate(EBINS): elut[lo:hi + 1] = b
    rng = np.random.RandomState(inj.get('seed', 1) + 7) if inj else None
    for s in np.nonzero(S['used'] & (n >= MIN_N))[0]:
        i = g[j == s]; o = np.argsort(tab['frame'][i], kind='stable'); i = i[o]; fr = tab['frame'][i]
        ev = np.r_[0, np.cumsum(np.diff(fr) > GAP)]; xs = x[i].copy(); ys = y[i].copy()
        if inj and inj.get('ev_sd', 0) > 0:
            ne_ = ev[-1] + 1; xs += inj['ev_sd'] * rng.randn(ne_)[ev]; ys += inj['ev_sd'] * rng.randn(ne_)[ev]
        ax = S['axis'][s]; a = xs * ax[0] + ys * ax[1]; c = -xs * ax[1] + ys * ax[0]
        st['rsd_x'][s] = rsd_defl(xs); st['rsd_y'][s] = rsd_defl(ys); st['rsd_along'][s] = rsd_defl(a); st['rsd_across'][s] = rsd_defl(c)
        st['n_events'][s] = ev[-1] + 1; st['photons'][s] = np.median(tab['photons'][i])
        # interior flag: not the first / last cycle of its event (events of >= 3 cycles)
        first = np.r_[True, ev[1:] != ev[:-1]]; last = np.r_[ev[1:] != ev[:-1], True]; inter = ~first & ~last
        if mode == 'interior':
            k = inter
            if k.sum() < 2: continue
            fr, ev, a, c, inter, i = fr[k], ev[k], a[k], c[k], inter[k], i[k]
        m = len(i); P = np.vstack([a, c])
        if mode == 'z':
            sx, sy = tab['sx_nm'][i], tab['sy_nm'][i]; SP = np.vstack([np.hypot(sx * ax[0], sy * ax[1]), np.hypot(sx * ax[1], sy * ax[0])])
        for i0 in range(0, m, 2000):                          # rows in chunks (memory)
            ii, jj = np.triu_indices(m, 1); k = (ii >= i0) & (ii < i0 + 2000); ii, jj = ii[k], jj[k]
            dt = fr[jj] - fr[ii]; k = dt >= 1; ii, jj, dt = ii[k], jj[k], dt[k]
            cls = (ev[ii] != ev[jj]).astype(np.int64)
            dd = np.abs(P[:, jj] - P[:, ii])
            if mode == 'z': dd = dd / np.hypot(SP[:, jj], SP[:, ii]) * np.sqrt(2) * ZSCALE
            d = np.minimum((dd / DW).astype(np.int64), NDB - 1)     # [2, npairs]
            bb = np.where(dt < len(lut), lut[np.minimum(dt, len(lut) - 1)], -1)
            for ab in range(2):
                for bsel, bidx in ((bb >= 0, bb), (np.ones(len(dt), bool), np.full(len(dt), nb - 1))):
                    q = ((bidx[bsel] * 2 + ab) * 2 + cls[bsel]) * NDB + d[ab, bsel]
                    H[pos[s]] += np.bincount(q, minlength=nb * 4 * NDB).reshape(nb, 2, 2, NDB).astype(np.int32)
                ke = inter[ii] & inter[jj] & (dt < len(elut)); eb = np.where(ke, elut[np.minimum(dt, len(elut) - 1)], -1); ke = eb >= 0
                q = ((eb[ke] * 2 + ab) * 2 + cls[ke]) * NDB + d[ab, ke]
                HE[pos[s]] += np.bincount(q, minlength=ne * 4 * NDB).reshape(ne, 2, 2, NDB).astype(np.int32)
            if m <= 2000: break
    return dict(n=n, st=st, H=H, HE=HE, uidx=uidx, pos=pos, DB=DB, shift=sh, n_assigned=int(len(g)))


def hist_sd(h):
    """rows of |d| histograms [..., NDB] (weights) -> 1.4826 median |d| / sqrt 2"""
    c = np.cumsum(h, -1); tot = c[..., -1:]; half = 0.5 * tot
    k = np.argmax(c >= half, -1); ck = np.take_along_axis(c, k[..., None], -1)[..., 0]; hk = np.take_along_axis(h, k[..., None], -1)[..., 0]
    med = (k + 1 - (ck - half[..., 0]) / np.maximum(hk, 1e-300)) * DW
    return np.where(tot[..., 0] > 0, 1.4826 * med / np.sqrt(2), np.nan)


def fit_curve(D, v, w):
    """v = a + b (1 - exp(-D / tau)); w = 1 / var(v). Returns sigma_loc, s_slow, tau, plateau"""
    ok = np.isfinite(v) & np.isfinite(w)
    if ok.sum() < 4: return [np.nan] * 4
    D, v, w = D[ok], v[ok], np.sqrt(w[ok]); best = None
    for t0 in (2.0, 20.0, 200.0):
        r = least_squares(lambda p: (p[0] + p[1] * (1 - np.exp(-D / np.exp(p[2]))) - v) * w, [max(v[0], 0.1), max(v[-1] - v[0], 0.1), np.log(t0)],
                          bounds=([0, 0, np.log(0.3)], [np.inf, np.inf, np.log(1e4)]))
        if best is None or r.cost < best.cost: best = r
    a, b, lt = best.x; return [np.sqrt(a), np.sqrt(b), np.exp(lt), np.sqrt(a + b)]


def summarize(R, sites, binmask=None, nboot=500, seed=1, fitmax=1000):
    """R: per_site result; sites: index array of the sites to use (e.g. common to all methods). Returns point estimates + CIs."""
    assert (R['pos'][sites] >= 0).all(); H = R['H'][R['pos'][sites]].astype(np.float64); HE = R['HE'][R['pos'][sites]].astype(np.float64); DB = R['DB']; nb = len(DB); ns = len(sites)
    D = np.array([d for d, _, _ in DB], float); fsel = D <= fitmax
    Ha = H.sum(3)                                                         # classes pooled: [ns, nb+1, 2 axes, NDB]
    Ha = np.concatenate([Ha, Ha.sum(2, keepdims=True)], 2)                # + pooled axis ('xy'): [ns, nb+1, 3, NDB]
    cnt = Ha.sum(-1)                                                      # pairs per site / bin / axis
    bm = (cnt[..., 0] >= MIN_P) if binmask is None else binmask          # [ns, nb+1] site used in that bin
    keep = np.ones(ns, bool)
    Hn = np.where(cnt[..., None] > 0, Ha / np.maximum(cnt[..., None], 1), 0) * bm[:, :, None, None]
    # event test: per bin, both classes >= 3 pairs; each class normalized per site
    HEa = np.concatenate([HE, HE.sum(2, keepdims=True)], 2)               # [ns, ne, 3, 2 cls, NDB]
    ce = HEa.sum(-1); ke = (ce >= 3).all(-1, keepdims=True)               # [ns, ne, 3, 1]
    HEn = np.where(ce[..., None] > 0, HEa / np.maximum(ce[..., None], 1), 0) * ke[..., None]
    sp = {k: R['st'][k][sites] for k in ('rsd_x', 'rsd_y', 'rsd_along', 'rsd_across')}
    rng = np.random.RandomState(seed); W = np.vstack([np.ones(ns)] + [np.bincount(rng.randint(0, ns, ns), minlength=ns) for _ in range(nboot)])
    C = np.einsum('bs,sijk->bijk', W, Hn); sig = hist_sd(C)               # [B+1, nb+1, 3]
    CE = np.einsum('bs,sijck->bijck', W, HEn); sigE = hist_sd(CE)         # [B+1, ne, 3, 2]
    spread = np.stack([np.stack([np.nanmedian(np.repeat(sp[k][keep], W[b][keep].astype(int))) for k in ('rsd_along', 'rsd_across')]) for b in range(len(W))])
    spread = np.c_[spread, np.sqrt(0.5 * (spread ** 2).sum(1))]          # xy = rms of the two axes
    V = sig[:, :nb] ** 2; w = 1 / np.nanvar(V[1:], 0)                    # [nb, 3]
    fits = np.array([[fit_curve(D[fsel], V[b, fsel, ax], w[fsel, ax]) for ax in range(3)] for b in range(len(W))])   # [B+1, 3, 4]
    sall = sig[:, nb]                                                     # 'all' pairs
    e1 = sall ** 2 - fits[..., 3] ** 2; e2 = spread ** 2 - fits[..., 3] ** 2
    ssite = np.sign(e1) * np.sqrt(np.abs(e1)); ssite2 = np.sign(e2) * np.sqrt(np.abs(e2))
    dvE = sigE[..., 1] ** 2 - sigE[..., 0] ** 2                           # [B+1, ne, 3]
    dvm = np.nanmean(dvE[:, :NE_HEAD], 1)                                 # bins 6 .. 40 cycles
    sev = np.sign(dvm) * np.sqrt(np.abs(dvm))
    fast = V[:, list(D).index(10)] - V[:, 0] if 10 in D else np.full((len(W), 3), np.nan)
    slow = V[:, list(D).index(1000)] - V[:, list(D).index(10)] if 1000 in D else np.full((len(W), 3), np.nan)
    def ci(a): a = np.asarray(a, float); return [float(a[0]), float(np.nanpercentile(a[1:], 2.5)), float(np.nanpercentile(a[1:], 97.5))]
    AX = ('along', 'across', 'xy'); o = dict(n_sites=int(ns), n_sites_in=int(ns), deltas=D.tolist(), axes={})
    for ax, nm in enumerate(AX):
        o['axes'][nm] = dict(sigma_pair={int(D[i]): ci(sig[:, i, ax]) for i in range(nb)}, sigma_all=ci(sall[:, ax]), spread=ci(spread[:, ax]),
                              sigma_loc=ci(fits[:, ax, 0]), s_slow=ci(fits[:, ax, 1]), tau=ci(fits[:, ax, 2]), plateau=ci(fits[:, ax, 3]),
                              s_site=ci(ssite[:, ax]), s_site_vs_spread=ci(ssite2[:, ax]), fast_var_1_10=ci(fast[:, ax]), slow_var_10_1000=ci(slow[:, ax]),
                              ev_same={f'{lo}-{hi}': ci(sigE[:, e, ax, 0]) for e, (lo, hi) in enumerate(EBINS)},
                              ev_diff={f'{lo}-{hi}': ci(sigE[:, e, ax, 1]) for e, (lo, hi) in enumerate(EBINS)},
                              ev_nsites={f'{lo}-{hi}': int(ke[:, e, ax, 0].sum()) for e, (lo, hi) in enumerate(EBINS)},
                              s_event=ci(sev[:, ax]), dvar_event=ci(dvm[:, ax]))
    o['n_pairs'] = {int(D[i]): int(cnt[bm[:, i], i, 2].sum() / 2) for i in range(nb)}; o['n_sites_bin'] = {int(D[i]): int(bm[:, i].sum()) for i in range(nb)}
    o['locs_per_site_median'] = float(np.median(R['n'][sites][keep])); o['events_per_site_median'] = float(np.nanmedian(R['st']['n_events'][sites][keep]))
    return o
