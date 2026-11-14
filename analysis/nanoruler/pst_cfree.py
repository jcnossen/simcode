"""
Part 3 complement: excess relative intensity variance c with the per-sample pattern phase (= emitter position along k) FREE,
which is the situation of the SIMCODE solve (the position is solved, so a position / phase-field error of the in-situ
sample position is not intensity noise there), and the size of the per-site phase offsets.

Leave-one-out with two linear parameters per amplitude group (sample, orientation):
    I_o ~ A x_o + B y_o      (x = rel (1 + m cos th), y = rel m sin th = d x / d phase;  B = A * phase offset, linearised)
fitted on the other fully-on frames of the group (>= 2 others, conditioned), prediction of the left-out frame with its
variance; Gaussian ML of V = a0 + a1 mu + c^2 mu^2 as in pst_jitter.py (4-sigma trimmed), bootstrap over sites.
Per-site phase offset: covariance of the sample-mean phase error (leave-one-out d of pst_jitter) between DIFFERENT 12-frame
samples of the same binding site (independent position fits and amplitudes), for pairs closer / farther than 20 cycles.
usage: python analysis/nanoruler/pst_cfree.py <shared cfg> <perstep cfg>     -> out/nanoruler/perstep/cfree.json
"""
import sys, json, math
import numpy as np, torch
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L
import pst_jitter as J
from simcode.util.proctitle import set_title

Y = {}


def loo2(I, x, g, p, wt=None, iters=4):
    y = Y['y']
    ng = g.max() + 1
    A = np.bincount(g, x * I, ng) / np.maximum(np.bincount(g, x * x, ng), 1e-12)
    for _ in range(iters):
        w = 1.0 / J.V_of(np.maximum(A[g], 1.0) * x, p)
        A = np.bincount(g, w * x * I, ng) / np.maximum(np.bincount(g, w * x * x, ng), 1e-30)
    A = np.maximum(A, 1.0)
    mu = A[g] * x; V = J.V_of(mu, p); w = 1.0 / V
    S = lambda v: np.bincount(g, w * v, ng)[g] - w * v
    Sxx, Sxy, Syy, SxI, SyI = S(x * x), S(x * y), S(y * y), S(x * I), S(y * I)
    cnt = np.bincount(g, minlength=ng)[g]
    det = Sxx * Syy - Sxy ** 2
    ok = (cnt >= 3) & (det > 0.1 * Sxx * Syy) & (det > 1e-20)
    dt = np.where(ok, det, 1)
    Al = np.where(ok, (Syy * SxI - Sxy * SyI) / dt, 1.0); Bl = np.where(ok, (Sxx * SyI - Sxy * SxI) / dt, 0.0)
    ok &= (Al > 20) & (np.abs(Bl) < 0.6 * np.maximum(Al, 1))               # linearisation range: |phase offset| < 0.6 rad
    r = I - Al * x - Bl * y
    var = V + (x * x * Syy - 2 * x * y * Sxy + y * y * Sxx) / dt
    return dict(r=r, var=var, Al=Al, Bl=Bl, A=A[g], ok=ok, mu=mu, V=V)


if __name__ == '__main__':
    set_title('nr-pstep'); torch.set_num_threads(2)
    cs, cp = sys.argv[1], sys.argv[2]
    F = torch.load(L.OUT + 'fits.pt', weights_only=False)
    D = L.load_data(); I, s, fr, site, b = D['I'], D['s'], D['fr'], D['site'], D['b']
    g = np.unique(D['g'], return_inverse=True)[1]; nsite = site.max() + 1
    rng = np.random.default_rng(3)
    R = {}
    loo1 = J.loo
    for nm, c in (('shared', cs), ('perstep', cp)):
        P = L.Pattern.from_state(F[c]['full']['state']); x, y, th = J.model_xy(D, P); Y['y'] = y
        e = dict(cfg=c)
        J.loo = loo2
        p, keep = J.fit_noise(I, x, g, trim=4.0, p0=(1400.0, 0.01, 0.03))
        o = loo2(I, x, g, p)
        e['phase_free'] = dict(a0=float(p[0]), a1=float(p[1]), c2=float(p[2]), c=float(math.sqrt(p[2])), n_used=int((o['ok'] & keep).sum()),
                               frac_used=float((o['ok'] & keep).mean()), kept=float(keep[o['ok']].mean()),
                               phase_offset_sd_fitted_rad=float(np.std((o['Bl'] / np.maximum(o['Al'], 1))[o['ok']])))
        bs = []
        for rep in range(40):
            wt = np.bincount(rng.integers(0, nsite, nsite), minlength=nsite)[site].astype(float)
            q, _ = J.fit_noise(I, x, g, wt=wt, trim=4.0, p0=tuple(p), keep0=keep, maxiter=120); bs.append(q)
        bs = np.array(bs); e['phase_free'].update(c_se=float(np.sqrt(bs[:, 2]).std()), c2_se=float(bs[:, 2].std()), a0_se=float(bs[:, 0].std()))
        # the same subset of data with the phase fixed (1-parameter leave-one-out), for a like-for-like comparison
        J.loo = loo1
        sub = o['ok']
        gi = g.copy()
        p1, keep1 = J.fit_noise(I[sub], x[sub], np.unique(g[sub], return_inverse=True)[1], trim=4.0, p0=(1400.0, 0.01, 0.05))
        e['phase_fixed_same_data'] = dict(a0=float(p1[0]), a1=float(p1[1]), c2=float(p1[2]), c=float(math.sqrt(p1[2])))
        # binned relative variance (phase free) for the figure
        bins = np.array([40, 80, 120, 160, 200, 250, 300, 350, 400, 500, 600, 700, 850, 1000, 1300, 2000]); rows = []
        k = o['ok'] & (np.abs(o['r']) < 4 * np.sqrt(o['var']))
        for lo, hi in zip(bins[:-1], bins[1:]):
            q = k & (o['mu'] >= lo) & (o['mu'] < hi)
            if q.sum() < 200: continue
            ex = (o['r'][q] ** 2 - (o['var'][q] - o['V'][q])) / o['mu'][q] ** 2
            rows.append([float(o['mu'][q].mean()), float(ex.mean()), float(ex.std() / math.sqrt(q.sum())), int(q.sum())])
        e['phase_free']['binned'] = rows
        print(nm, json.dumps({k_: v for k_, v in e.items() if k_ != 'cfg'})[:700], flush=True)

        # ---- per-site phase offsets: sample-mean phase error per (sample, orientation), pairs of samples of the same site
        o1 = loo1(I, x, g, J_NOISE := (1400.0, 0.0, 0.058))
        a = o1['Al'] * y; ok = o1['ok'] & (o1['Al'] > 20) & (np.abs(y) > 0.15)
        d = np.where(ok, o1['r'] / np.where(ok, a, 1), 0); w = ok * np.where(ok, a, 1) ** 2 / o1['var']
        ng = g.max() + 1
        Wg = np.bincount(g, w, ng); Sg = np.bincount(g, w * d, ng); tg = np.bincount(g, w * fr, ng) / np.maximum(Wg, 1e-30) / 6
        first = np.unique(g, return_index=True)[1]; sg = site[first]; ag = D['a'][first]; has = Wg > 0
        res = {}
        for an in (0, 1):
            num = {'near': 0.0, 'far': 0.0}; den = {'near': 0.0, 'far': 0.0}; per_site = {'near': [], 'far': []}
            sel = np.nonzero(has & (ag == an))[0]; order = sel[np.argsort(sg[sel], kind='stable')]
            bounds = np.r_[0, np.nonzero(np.diff(sg[order]))[0] + 1, len(order)]
            for i0, i1 in zip(bounds[:-1], bounds[1:]):
                ii = order[i0:i1]
                if len(ii) < 2: continue
                dt_ = np.abs(tg[ii][:, None] - tg[ii][None]); pr = np.outer(Sg[ii], Sg[ii]); pw = np.outer(Wg[ii], Wg[ii])
                up = np.triu(np.ones_like(dt_, bool), 1)
                for key, m_ in (('near', up & (dt_ <= 20)), ('far', up & (dt_ > 20))):
                    num[key] += pr[m_].sum(); den[key] += pw[m_].sum(); per_site[key].append((pr[m_].sum(), pw[m_].sum()))
            for key in ('near', 'far'):
                ps = np.array(per_site[key]); bsr = [ps[q, 0].sum() / ps[q, 1].sum() for q in (rng.integers(0, len(ps), len(ps)) for _ in range(300))]
                v = num[key] / den[key]
                res[f'o{an}_{key}'] = dict(var=float(v), var_se=float(np.std(bsr)), sd_rad=float(np.sign(v) * math.sqrt(abs(v))),
                                           sd_nm=float(np.sign(v) * math.sqrt(abs(v)) * (219.9, 219.5)[an] / (2 * math.pi)))
        e['site_phase_offset'] = res
        print(nm, 'site phase offsets', json.dumps(res), flush=True)
        R[nm] = e
    json.dump(R, open(L.OUT + 'cfree.json', 'w'), indent=1)
