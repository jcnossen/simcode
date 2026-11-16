"""Is the hollow tube visible? Cross-section profiles of a hollow-microtubule run (gen_hollow.py movie) per method.

Rows per method: the FRC set of metrics.py (disjoint windows, max(sigma) < 0.15 px, inside the FOV; SIMCODE v2 = its
sigma <= 12 nm selection v2sig, v2sel = the sel rows), so no photon is counted twice.
Profile: every localization is assigned to the nearest GT centreline point (within 150 nm); u = signed distance across the
tubule (along the local normal of the centreline, nm). Only 'clean' centreline points are used: no centreline of another tubule
within 200 nm and > 150 nm from the tubule's ends (no crossings, no neighbours, no caps). Histogram 2 nm bins, |u| <= 80 nm.
Measures:
  dip ratio   mean count in |u| < 0.3 R / mean count in 0.7 R < |u| < 1.3 R (R = the true radius; 1 = flat, < 1 = a dip between
              the two projected walls); 95 % CI by a block bootstrap over 300-nm segments of the tubules (400 resamples).
  model fit   Poisson ML on the histogram of (a) two walls: the projected hollow cylinder (arcsine density on [-R', R'], R' free)
              convolved with a two-Gaussian localization error (s1, s2, weight), offset mu, + flat bg; (b) one line: the same
              two-Gaussian error, mu, + flat bg. -> R' (and its bootstrap CI, 40 resamples), sigma (rms of the mixture), 2 dNLL, p of the likelihood-ratio test (0.5 chi2_1 mixture,
              R' = 0 on the boundary), dAIC = 2 dNLL - 2.
  'resolved'  (a dip a viewer sees) the dip ratio's upper 95 % bound < 1 AND p < 0.001.
  'detected'  (wall structure in the profile shape, no visible dip needed) p < 0.001 and the R' CI excludes 0.5 R.
usage: python analysis/papersim/hollow_analysis.py <run dir> [--movie <movie dir>] [--nframes N] [--model m1809]
outputs: <run>/hollow_<model>.json, hollow_<model>.png (profiles with fits)"""
import os, sys, json, argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import metrics as M

BIN = 2.0
UMAX = 80.0
TH = (np.arange(64) + 0.5) / 64 * np.pi          # angles for the projected-cylinder (arcsine) mixture


def centreline(gt):
    c = gt['centre'][:, :2]; tub = gt['tub']
    tg = np.zeros_like(c); s = np.zeros(len(c)); clean = np.zeros(len(c), bool)
    tree = cKDTree(c)
    for t in np.unique(tub):
        i = np.nonzero(tub == t)[0]; ci = c[i]
        d = np.zeros_like(ci); d[1:-1] = ci[2:] - ci[:-2]; d[0] = ci[1] - ci[0]; d[-1] = ci[-1] - ci[-2]
        tg[i] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        s[i] = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ci, axis=0).T))])
    L = np.array([s[tub == t].max() for t in range(tub.max() + 1)])
    ends_ok = (s * C.PX_NM > 150) & ((L[tub] - s) * C.PX_NM > 150)
    nb = tree.query_ball_point(c, 200 / C.PX_NM)
    other = np.array([np.any(tub[np.asarray(n)] != tub[k]) for k, n in enumerate(nb)])
    clean = ends_ok & ~other
    return dict(c=c, tg=tg, s=s, tub=tub, clean=clean, tree=tree)


def profile_coords(T, m, CL):
    x, y = T['x'][m], T['y'][m]
    d, j = CL['tree'].query(np.c_[x, y], distance_upper_bound=150 / C.PX_NM)
    ok = np.isfinite(d); j = np.where(ok, j, 0)
    ok &= CL['clean'][j]
    n = np.stack([-CL['tg'][j, 1], CL['tg'][j, 0]], 1)
    u = ((np.c_[x, y] - CL['c'][j]) * n).sum(1) * C.PX_NM
    blk = CL['tub'][j] * 100000 + (CL['s'][j] * C.PX_NM // 300).astype(int)
    ok &= np.abs(u) <= UMAX
    return u[ok], blk[ok]


EDGES = np.arange(-UMAX, UMAX + BIN / 2, BIN)
CEN = 0.5 * (EDGES[1:] + EDGES[:-1])


def model_counts(p, R_on):
    """p = (log A, log s1, log(s2 - s1), logit f, R, mu, log B) -> expected counts per bin. The localization error across the
    tubule is a mixture of two centred Gaussians (s1 with weight f, s2 > s1): localization precision varies per row, so a single
    Gaussian would let the two-wall model win on the shape of the tails alone."""
    A, s1 = np.exp(p[0]), np.exp(np.clip(p[1], -5, 6)); s2 = s1 + np.exp(np.clip(p[2], -8, 6)); f = 1 / (1 + np.exp(-np.clip(p[3], -30, 30)))
    R, mu, B = (abs(p[4]) if R_on else 0.0), p[5], np.exp(p[6])
    z = EDGES[:, None] - mu - R * np.cos(TH)[None]
    F = f * ndtr(z / s1).mean(1) + (1 - f) * ndtr(z / s2).mean(1)
    return A * np.diff(F) + B * BIN


def fit(h, R_on, R0):
    nll = lambda p: float((model_counts(p, R_on) - h * np.log(np.maximum(model_counts(p, R_on), 1e-12))).sum())
    tot = max(h.sum(), 1.0)
    best = None
    for s0 in (4.0, 15.0):
        for r0 in ((R0 * 0.5, R0 * 1.2) if R_on else (0.0,)):
            p0 = np.array([np.log(tot * 0.9), np.log(s0), np.log(s0), 1.0, r0, 0.0, np.log(max(tot * 0.1 / (2 * UMAX), 1e-3))])
            r = minimize(nll, p0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-3, fatol=1e-4))
            if best is None or r.fun < best.fun: best = r
    p = best.x; f = 1 / (1 + np.exp(-np.clip(p[3], -30, 30))); s1 = np.exp(np.clip(p[1], -5, 6)); s2 = s1 + np.exp(np.clip(p[2], -8, 6))
    return dict(nll=best.fun, A=float(np.exp(p[0])), sigma=float(np.sqrt(f * s1 ** 2 + (1 - f) * s2 ** 2)), sigma1=float(s1), sigma2=float(s2),
                frac1=float(f), R=float(abs(p[4])) if R_on else 0.0, mu=float(p[5]), bg=float(np.exp(p[6])), p=p.tolist())


def dip_ratio(h, R):
    c = np.abs(CEN) < 0.3 * R; w = (np.abs(CEN) > 0.7 * R) & (np.abs(CEN) < 1.3 * R)
    return float(h[c].mean() / max(h[w].mean(), 1e-12))


def analyse(u, blk, Rt, nboot=400, nboot_fit=40, rng=None):
    rng = rng or np.random.default_rng(0)
    h = np.histogram(u, EDGES)[0].astype(float)
    f2 = fit(h, True, Rt); f1 = fit(h, False, Rt)
    d2 = 2 * (f1['nll'] - f2['nll'])
    out = dict(n=int(len(u)), dip=dip_ratio(h, Rt), fit_two=f2, fit_one=f1, two_dnll=float(d2), dAIC=float(d2 - 2),
               p_lrt=float(0.5 * chi2.sf(max(d2, 0), 1)), hist=h.tolist())
    ub, inv = np.unique(blk, return_inverse=True)
    HB = np.zeros((len(ub), len(CEN)))
    np.add.at(HB, (inv, np.clip(np.searchsorted(EDGES, u) - 1, 0, len(CEN) - 1)), 1)
    dips, Rs = [], []
    for b in range(nboot):
        hb = HB[rng.integers(0, len(ub), len(ub))].sum(0)
        dips.append(dip_ratio(hb, Rt))
        if b < nboot_fit: Rs.append(fit(hb, True, Rt)['R'])
    out['dip_ci95'] = np.percentile(dips, [2.5, 97.5]).tolist()
    out['R_ci95'] = np.percentile(Rs, [2.5, 97.5]).tolist()
    out['n_blocks'] = int(len(ub))
    out['resolved'] = bool(out['dip_ci95'][1] < 1 and out['p_lrt'] < 1e-3)
    out['detected'] = bool(out['p_lrt'] < 1e-3 and out['R_ci95'][0] > 0.5 * Rt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--movie', default=None); ap.add_argument('--nframes', type=int, default=0)
    ap.add_argument('--model', default='m1809')
    a = ap.parse_args()
    from simcode.util.proctitle import set_title; set_title('psim-hollow')
    run = os.path.abspath(a.run); md = os.path.abspath(a.movie or a.run)
    g = json.load(open(os.path.join(md, 'gen.json'))); nf = a.nframes or g['nframes']
    gt = dict(np.load(os.path.join(md, 'gt.npz')))
    Rt = g['radius_nm']
    CL = centreline(gt)
    MJ = os.path.join(run, f'metrics_{a.model}.json')
    met = json.load(open(MJ))['methods'] if os.path.exists(MJ) else {}
    R = dict(run=run, movie=md, radius_nm=Rt, phot_scale=g['phot_scale'], cond=g['cond'], model=a.model, nframes=nf,
             window_photons_ge300=g['window_photons_ge300'], clean_centreline_frac=float(CL['clean'].mean()), methods={})
    # GT reference: the tube points themselves (no localization error), as a check of the profile machinery
    gu, gb = profile_coords(dict(x=gt['pts'][:, 0], y=gt['pts'][:, 1]), np.ones(len(gt['pts']), bool), CL)
    R['gt_points'] = dict(n=int(len(gu)), dip=dip_ratio(np.histogram(gu, EDGES)[0].astype(float), Rt))
    for name, path in M.method_paths(run, a.model).items():
        if name == 'v2siggt': continue
        T = M.load_method(name, path)
        _, _, filt = M.frc_set(T, g['W'], nf)
        u, blk = profile_coords(T, filt, CL)
        if len(u) < 200: R['methods'][name] = dict(n=int(len(u))); continue
        r = analyse(u, blk, Rt)
        gm = met.get(name, {}).get('gt_match', {})
        r.update(n_frc=int(filt.sum()), rms_gt_nm=gm.get('rms_nm'), rms_gt_trim2_nm=gm.get('rms_trim2_nm'), frc_nm=met.get(name, {}).get('frc_nm'))
        R['methods'][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ('hist', 'fit_two', 'fit_one')}),
              'R_fit %.1f sigma_fit %.1f' % (r['fit_two']['R'], r['fit_two']['sigma']), flush=True)
    json.dump(R, open(os.path.join(run, f'hollow_{a.model}.json'), 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = [k for k in R['methods'] if 'hist' in R['methods'][k]]
    fig, axs = plt.subplots(1, max(len(ks), 1), figsize=(3.6 * max(len(ks), 1), 3.2), squeeze=False)
    for ax, k in zip(axs[0], ks):
        r = R['methods'][k]; h = np.array(r['hist'])
        ax.bar(CEN, h, width=BIN, color='0.7')
        ax.plot(CEN, model_counts(np.array(r['fit_two']['p']), True), 'r-', label=f"two walls R'={r['fit_two']['R']:.1f}")
        ax.plot(CEN, model_counts(np.array(r['fit_one']['p']), False), 'b--', label='one line')
        for s_ in (-Rt, Rt): ax.axvline(s_, color='k', lw=0.5, ls=':')
        ax.set_title(f"{k}: dip {r['dip']:.2f} [{r['dip_ci95'][0]:.2f}, {r['dip_ci95'][1]:.2f}]", fontsize=8)
        ax.set_xlabel('u (nm)'); ax.legend(fontsize=6); ax.set_xlim(-80, 80)
    fig.suptitle(f"{g['cond']} R {Rt:g} nm, photons x{g['phot_scale']:g}", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(run, f'hollow_{a.model}.png'), dpi=90); plt.close(fig)


if __name__ == '__main__':
    main()
