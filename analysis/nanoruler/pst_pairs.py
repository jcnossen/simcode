"""
Model-free intensity variance of fully-on frames: pairs of frames of the SAME emitter sample and the SAME pattern step in
adjacent cycles (f, f + 6). Both frames have the same pattern phase (up to the drift / fluctuation within 84 ms) and the same
amplitude, so D = I_1 - I_2 has variance 2 V(mu) with no pattern model and no amplitude fit. mu = (I_1 + I_2) / 2.
Gaussian ML of V = a0 + a1 mu + c^2 mu^2 on D (4-sigma trimmed), bootstrap over binding sites; also with the pattern
prediction ratio x_1 / x_2 of the shared / per-step fit applied (D = I_1 - I_2 x_1 / x_2), which changes nothing if the
pattern is stable over one cycle. This c is the excess that is NOT due to the pattern description (emitter brightness
fluctuation, not-fully-on frames, intensity-fit errors); the leave-one-out c of pst_jitter / pst_cfree adds the pattern part.
Output: out/nanoruler/perstep/pairs.json
"""
import sys, json, math
import numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L

if __name__ == '__main__':
    D = L.load_data(); I, fr, site, s = D['I'], D['fr'], D['site'], D['s']
    g = np.unique(D['g'], return_inverse=True)[1]
    key = g.astype(np.int64) * 40000 + fr
    order = np.argsort(key); ks = key[order]
    pos = np.searchsorted(ks, key + 6); pos[pos >= len(ks)] = 0
    has = ks[pos] == key + 6
    i1 = np.nonzero(has)[0]; i2 = order[pos[has]]
    d = I[i1] - I[i2]; mu = 0.5 * (I[i1] + I[i2]); st = site[i1]; nsite = site.max() + 1
    R = dict(n_pairs=int(len(d)), median_mu=float(np.median(mu)))

    def fit(wt, keep, p0, fix_a1=None):
        def nll(q):
            a0, a1, c2 = (abs(q[0]), fix_a1, abs(q[1])) if fix_a1 is not None else np.abs(q)
            v = 2 * np.maximum(a0 + a1 * mu[keep] + c2 * mu[keep] ** 2, 1e-3) * 0.9892
            return (wt[keep] * (np.log(v) + d[keep] ** 2 / v)).sum() / wt[keep].sum()
        q = minimize(nll, p0, method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-8, maxiter=800)).x
        return np.array([abs(q[0]), fix_a1, abs(q[1])]) if fix_a1 is not None else np.abs(q)

    w1 = np.ones(len(d))
    for lab, fa in (('free', None), ('a1eq1', 1.0)):
        p = np.array([500.0, 1.0, 0.02]); keep = np.ones(len(d), bool)
        for _ in range(4):
            p = fit(w1, keep, [p[0], p[2]] if fa else p, fa)
            keep = np.abs(d) < 4 * np.sqrt(2 * (p[0] + p[1] * mu + p[2] * mu ** 2))
        rng = np.random.default_rng(0); bs = []
        for rep in range(100):
            wt = np.bincount(rng.integers(0, nsite, nsite), minlength=nsite)[st].astype(float)
            bs.append(fit(wt, keep, [p[0], p[2]] if fa else p, fa))
        bs = np.array(bs)
        R[lab] = dict(a0=float(p[0]), a1=float(p[1]), c2=float(p[2]), c=float(math.sqrt(p[2])), kept=float(keep.mean()),
                      a0_se=float(bs[:, 0].std()), a1_se=float(bs[:, 1].std()), c_se=float(np.sqrt(bs[:, 2]).std()))
        print(lab, R[lab], flush=True)
        if lab == 'a1eq1': keep1 = keep
    # binned variance (trimmed set of the a1 = 1 fit)
    bins = np.array([30, 60, 90, 120, 160, 200, 250, 300, 350, 400, 500, 600, 700, 850, 1000, 1300, 2000]); rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        q = keep1 & (mu >= lo) & (mu < hi)
        if q.sum() < 100: continue
        v = d[q] ** 2 / 2 / 0.9892
        rows.append([float(mu[q].mean()), float(v.mean()), float(v.std() / math.sqrt(q.sum())), int(q.sum())])
    R['binned_mu_var_se_n'] = rows
    # with the pattern ratio of the fitted models (the two frames differ by the drift over 6 frames only)
    F = torch.load(L.OUT + 'fits.pt', weights_only=False)
    import pst_jitter as J
    for cfg in sys.argv[1:]:
        P = L.Pattern.from_state(F[cfg]['full']['state']); x = J.model_xy(D, P)[0]
        ratio = x[i1] / x[i2]
        R[f'ratio_{cfg}'] = dict(ratio_minus_1_rms=float(np.sqrt(np.mean((ratio - 1) ** 2))),
                                 var_D_trim=float(np.mean((I[i1] - I[i2] * ratio)[keep1] ** 2)), var_D_plain=float(np.mean(d[keep1] ** 2)))
    json.dump(R, open(L.OUT + 'pairs.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k.startswith('ratio')}))
    for r_ in rows: print(f'mu {r_[0]:7.1f}  V {r_[1]:9.1f} +- {r_[2]:6.1f}  rel sd {math.sqrt(r_[1]) / r_[0]:.3f}  n {r_[3]}')
