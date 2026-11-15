"""How strict should the SIMCODE v2 selection be? Ground-truth correctness of every pass-2 'sel' localization on simulated
movies, as a function of the selection score, and the loosest threshold at which the kept set is >= 99 % (95, 99.5 %) correct.

Datasets (GT available): papersim density movies D0.5-D4 (60 000 frames), papersim hollow x1 movies (experimental 1809/2 and
2.5 nM photons / bg, 30 000 frames), and the main pipeline's realistic simulated movies m05r3 / m25rc3 (analysis/simreal
recipes: filaments, structured / fluctuating bg, PSF and emitter variation, pattern field; 4 800 frames, 120 x 120 px).
Correctness of a localization (window start f, position r, reported sigma sx, sy), against the GT emitters ON in [f, f + 6)
with >= 30 window photons ('ON'):
  nearest ON emitter j at error e = r - g_j (after removing the dataset's global offset = median e of rows with |e| < 50 nm:
  the solve's position is defined up to the estimated pattern's gauge; the offset is reported and is ~1 nm).
  correct_20 / correct_30   |e| < 20 / 30 nm
  correct_3s                single ON emitter within 150 nm (no blend) and |e_x| < 3 sx and |e_y| < 3 sy
  failure classes (|e| >= 30 nm), in this priority:
    fringe     some ON emitter i and angle a with |r - (g_i +- pitch_a k_a / |k_a|)| < 30 nm (a one-period pattern solve error)
    blend      >= 2 ON emitters within 150 nm of r (wrong emitter / merged neighbours)
    false      no ON emitter within 150 nm
    gross      single emitter, 50 <= |e| < 150 nm
    imprecise  single emitter, 30 <= |e| < 50 nm
Base set = the rows with p >= 0.63 inside the field (simcode.table.select(T, max_precision_nm=None), the field = the run's
crop); the selection used for SIMCODE v2 is max(sx, sy) <= 12 nm on them (simcode.table.select; operating point 'sig12').
Scores ('better' first): u_all (the per-10-px-tile quantile of the mean calibrated sigma among the base rows; metrics.u_all),
sigma (mean of sx, sy), p (detector probability, high first), q_dof (low first), photons (high first).
(Removed 2026-11-14: the strict_u score, the u <= 0.45 and strict-product operating points.)
Curves: precision (fraction correct of the kept rows) vs fraction of base rows kept; per-class fractions; precision per u_all
decile (monotone?); ROC AUC per score. Operating point per target: the loosest threshold with the kept set's precision >= target
(at least 300 rows kept), its absolute score value, kept fraction, and (papersim) the FRC (fair protocol, disjoint windows) and
Jaccard (10 nm) of the kept rows.
usage: python analysis/papersim/selection.py [--model m1809] [--datasets all] -> out/papersim/selection/selection_<model>.json / .png"""
import os, sys, json, glob, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

OUTD = os.path.join(C.OUT, 'selection')
TARGETS = (0.95, 0.99, 0.995)
CLASSES = ('fringe', 'blend', 'false', 'gross', 'imprecise')


def datasets(model):
    D = []
    for d in (0.5, 1, 2, 4):
        md = os.path.join(C.OUT, f'D{d:g}')
        D.append(dict(name=f'D{d:g}', kind='papersim', locs=os.path.join(md, 'full', f'v2_{model}', 'locs_p2_sim.npz'), gt=os.path.join(md, 'gt.npz'),
                      gen=os.path.join(md, 'gen.json')))
    for h in sorted(glob.glob(os.path.join(C.OUT, 'hollow', '*_x1'))):
        D.append(dict(name='hollow_' + os.path.basename(h), kind='papersim', locs=os.path.join(h, 'full', f'v2_{model}', 'locs_p2_sim.npz'),
                      gt=os.path.join(h, 'gt.npz'), gen=os.path.join(h, 'gen.json')))
    run = {'m1809': ('psfmatch', 'run_m1809')}.get(model)
    if run:
        for s in ('m05r3', 'm25rc3'):
            D.append(dict(name=s, kind='simreal', locs=os.path.join(C.REPO, 'out', run[0], 'sims', s, run[1], 'locs_p2.npz'),
                          gt=os.path.join(C.REPO, 'out', 'psf3d_nets', 'sims', s, 'sim_gt.npz')))
    return [d for d in D if os.path.exists(d['locs'])]


def load_gt(ds):
    g = np.load(ds['gt'])
    if ds['kind'] == 'papersim':
        x, y, t0, t1 = g['ev_x'], g['ev_y'], g['ev_t0'], g['ev_t1']
        per_frame = (g['ev_I6'] if 'ev_I6' in g.files else np.full(len(x), C.I6)) / 6
        k = g['k']
    else:
        x, y, t0, t1, per_frame = g['x'], g['y'], g['t0'], g['t1'], g['bright']
        P = g['patt']; k = np.stack([P[0, :2], P[3, :2]])
    return dict(x=x.astype(np.float64), y=y.astype(np.float64), t0=t0, t1=t1, pf=per_frame, k=k)


def classify(T, G):
    """per row: err vector to the nearest ON emitter, n ON within 150 nm, fringe flag"""
    px = C.PX_NM
    o = np.argsort(G['t0']); t0, t1, gx, gy, pf = (G[k][o] for k in ('t0', 't1', 'x', 'y', 'pf'))
    maxdur = float((t1 - t0).max())
    kh = G['k'] / np.linalg.norm(G['k'], axis=1, keepdims=True); pitch = 2 * np.pi / np.linalg.norm(G['k'], axis=1)   # px
    n = len(T['x']); err = np.full((n, 2), np.nan); n150 = np.zeros(n, int); fringe = np.zeros(n, bool)
    f = T['frame']
    for w in np.unique(f):
        ii = np.nonzero(f == w)[0]
        lo = np.searchsorted(t0, w - maxdur); hi = np.searchsorted(t0, w + 6)
        j = np.arange(lo, hi); on = np.clip(np.minimum(t1[j], w + 6) - np.maximum(t0[j], w), 0, 6)
        j = j[on * pf[j] >= 30]
        if not len(j): continue
        dx = T['x'][ii, None] - gx[None, j]; dy = T['y'][ii, None] - gy[None, j]; d = np.hypot(dx, dy)
        a = d.argmin(1); err[ii] = np.c_[dx[np.arange(len(ii)), a], dy[np.arange(len(ii)), a]]
        n150[ii] = (d < 150 / px).sum(1)
        fr = np.zeros(len(ii), bool)
        for ai in range(2):
            for sgn in (-1, 1):
                sx_, sy_ = sgn * pitch[ai] * kh[ai]
                fr |= (np.hypot(dx - sx_, dy - sy_) < 30 / px).any(1)
        fringe[ii] = fr
    return err * px, n150, fringe


def evaluate(ds):
    from simcode import table as SP
    import metrics as M
    z = np.load(ds['locs'])
    crop = M.v2_crop(ds['locs']) if ds['kind'] == 'papersim' else SP.run_crop(os.path.dirname(ds['locs']))
    sel = SP.select(z, None, crop=crop)       # base = the rows with p >= 0.63 inside the field
    T = {k: z[k][sel] for k in ('x', 'y', 'frame', 'sx', 'sy', 'p', 'q_dof', 'photons')}
    T['sig12'] = SP.select(z, SP.SIGMA_MAX_NM, crop=crop)[sel]
    T['x'] = T['x'].astype(np.float64); T['y'] = T['y'].astype(np.float64)
    G = load_gt(ds)
    err, n150, fringe = classify(T, G)
    e = np.hypot(*err.T)
    off = np.nanmedian(err[e < 50], 0)
    err = err - off; e = np.hypot(*err.T)
    sig = 0.5 * (T['sx'] + T['sy']) * C.PX_NM
    single = n150 == 1
    has = np.isfinite(e)
    cls = np.full(len(e), 'correct', dtype=object)
    bad = ~has | (e >= 30)
    for name, m in (('fringe', fringe), ('blend', n150 >= 2), ('false', ~has | (n150 == 0)), ('gross', single & (e >= 50)), ('imprecise', single & (e >= 30))):
        cls[bad & m & (cls == 'correct')] = name
    cls[bad & (cls == 'correct')] = 'gross'
    corr = dict(c20=has & (e < 20), c30=has & (e < 30),
                c3s=has & single & (np.abs(err[:, 0]) < 3 * T['sx'] * C.PX_NM) & (np.abs(err[:, 1]) < 3 * T['sy'] * C.PX_NM))
    ua = M.u_all(T['x'], T['y'], 0.5 * (T['sx'] + T['sy']).astype(np.float64), np.ones(len(T['x']), bool))
    scores = dict(u_all=ua, sigma=sig, p=-T['p'].astype(np.float64), q_dof=T['q_dof'].astype(np.float64), photons=-T['photons'].astype(np.float64))
    R = dict(name=ds['name'], n_sel=int(len(e)), offset_nm=off.round(2).tolist(), frac_correct=dict((k, float(v.mean())) for k, v in corr.items()),
             class_frac={c: float((cls == c).mean()) for c in ('correct',) + CLASSES}, scores={}, deciles={})
    from sklearn.metrics import roc_auc_score
    for sn, s in scores.items():
        o = np.argsort(s, kind='stable'); keep = np.arange(1, len(o) + 1) / len(o)
        sr = {}
        for cn, c in corr.items():
            cp = np.cumsum(c[o]) / np.arange(1, len(o) + 1)
            ops = {}
            for t in TARGETS:
                okk = np.nonzero((cp >= t) & (np.arange(1, len(o) + 1) >= 300))[0]
                if len(okk):
                    i = okk[-1]; ops[str(t)] = dict(kept=float(keep[i]), n=int(i + 1), threshold=float(s[o][i]))
                else:
                    ops[str(t)] = None
            grid = np.unique(np.clip((np.linspace(0.01, 1, 100) * len(o)).astype(int) - 1, 0, len(o) - 1))
            sr[cn] = dict(auc=float(roc_auc_score(c, -s)) if 0 < c.mean() < 1 else None, ops=ops,
                          curve=dict(kept=keep[grid].round(4).tolist(), precision=cp[grid].round(5).tolist()))
        cc = {}
        grid = np.unique(np.clip((np.linspace(0.01, 1, 100) * len(o)).astype(int) - 1, 0, len(o) - 1))
        for c in CLASSES:
            cc[c] = (np.cumsum(cls[o] == c) / np.arange(1, len(o) + 1))[grid].round(5).tolist()
        sr['class_curves'] = cc
        R['scores'][sn] = sr
    # u_all deciles: precision and failure classes per decile (which classes do the looser rows add?)
    b = np.minimum((ua * 10).astype(int), 9)
    R['deciles_u_all'] = {f'{i / 10:.1f}-{(i + 1) / 10:.1f}': dict(n=int((b == i).sum()), c20=float(corr['c20'][b == i].mean()), c30=float(corr['c30'][b == i].mean()),
                                                                  c3s=float(corr['c3s'][b == i].mean()), **{c: float((cls[b == i] == c).mean()) for c in CLASSES})
                          for i in range(10) if (b == i).any()}
    R['_cache'] = dict(order={k: np.argsort(v, kind='stable') for k, v in scores.items()}, T=T, corr=corr, u_all=ua)
    return R


def frc_jaccard(ds, T, keep, nframes):
    sys.path.insert(0, os.path.join(C.REPO, 'analysis', 'paper')); import frc_fair, metrics as M
    x, y, f = T['x'][keep], T['y'][keep], T['frame'][keep]
    d = (f % 6 == 0)
    out = dict(frc_nm=float(frc_fair.frc_halves(x[d], y[d], f[d])[0]) if d.sum() > 100 else None)
    if ds['kind'] == 'papersim':
        pts = np.load(ds['gt'])['pts']
        out['jaccard'] = float(M.compute_jaccard_points(pts[:, :2], np.c_[x, y], 10 / C.PX_NM))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='m1809'); ap.add_argument('--datasets', default='all')
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('psim-select')
    os.makedirs(OUTD, exist_ok=True)
    DS = datasets(a.model)
    if a.datasets != 'all': DS = [d for d in DS if d['name'] in a.datasets.split(',')]
    res = {}
    for ds in DS:
        R = evaluate(ds); cache = R.pop('_cache'); T = cache['T']
        nframes = json.load(open(ds['gen']))['nframes'] if ds.get('gen') else int(T['frame'].max() + 6)
        # FRC / Jaccard at the sigma / u_all operating points, of all base rows and of the paper's rule (sigma <= 12 nm)
        R['at_ops'] = {}
        for sn in ('u_all', 'sigma'):
            for cn in ('c20', 'c30', 'c3s'):
                for t in TARGETS:
                    op = R['scores'][sn][cn]['ops'][str(t)]
                    if op:
                        keep = np.zeros(len(T['x']), bool); keep[cache['order'][sn][:op['n']]] = True
                        R['at_ops'][f'{sn}_{cn}_{t}'] = dict(kept=op['kept'], threshold=op['threshold'], **frc_jaccard(ds, T, keep, nframes))
        R['at_ops']['all_sel'] = dict(kept=1.0, **frc_jaccard(ds, T, np.ones(len(T['x']), bool), nframes))
        k12 = T['sig12']
        R['at_ops']['sig12'] = dict(kept=float(k12.mean()), precision_c20=float(cache['corr']['c20'][k12].mean()),
                                    precision_c30=float(cache['corr']['c30'][k12].mean()), precision_c3s=float(cache['corr']['c3s'][k12].mean()),
                                    **frc_jaccard(ds, T, k12, nframes))
        R['nframes'] = nframes
        res[ds['name']] = R
        s = R['scores']
        print(ds['name'], 'n', R['n_sel'], 'offset', R['offset_nm'], 'correct', {k: round(v, 3) for k, v in R['frac_correct'].items()},
              'classes', {k: round(v, 4) for k, v in R['class_frac'].items()},
              'AUC c30', {k: (round(v['c30']['auc'], 3) if v['c30']['auc'] else None) for k, v in s.items()},
              '99% c30 u_all', s['u_all']['c30']['ops']['0.99'], 'sigma <= 12 nm', {k: round(v, 4) for k, v in R['at_ops']['sig12'].items() if k != 'kept'}, 'sigma', s['sigma']['c30']['ops']['0.99'], flush=True)
    json.dump(res, open(os.path.join(OUTD, f'selection_{a.model}.json'), 'w'), indent=1, default=float)
    plot(res, os.path.join(OUTD, f'selection_{a.model}.png'))


def plot(res, fn):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = list(res); cm = plt.get_cmap('viridis')
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    for i, n in enumerate(names):
        c = cm(i / max(len(names) - 1, 1)); R = res[n]['scores']
        for ax, (sn, cn) in zip(axs[0], (('sigma', 'c30'), ('sigma', 'c20'), ('sigma', 'c3s'))):
            cv = R[sn][cn]['curve']; ax.plot(cv['kept'], cv['precision'], color=c, label=n)
        cc = R['sigma']['class_curves']; kept = R['sigma']['c30']['curve']['kept']
        axs[1, 0].plot(kept, np.sum([cc[k] for k in CLASSES], 0), color=c, label=n)
        for sn, ls in (('u_all', '-'), ('sigma', '--'), ('p', ':'), ('photons', '-.')):
            cv = R[sn]['c30']['curve']; axs[1, 1].plot(cv['kept'], cv['precision'], ls=ls, color=c, lw=1)
        if res[n].get('deciles_u_all'):
            dd = res[n]['deciles_u_all']; axs[1, 2].plot([float(k[:3]) + 0.05 for k in dd], [v['c30'] for v in dd.values()], 'o-', color=c, ms=3)
    for ax, t in zip(axs[0], ('correct = |e| < 30 nm', 'correct = |e| < 20 nm', 'correct = single emitter, |e| < 3 sigma')):
        for tt in TARGETS: ax.axhline(tt, color='grey', lw=0.6, ls=':')
        ax.set_xlabel('fraction of base rows kept (sigma ranking)'); ax.set_ylabel('precision of the kept set'); ax.set_title(t, fontsize=9); ax.set_ylim(0.6, 1.0)
    axs[0, 0].legend(fontsize=6)
    axs[1, 0].set_title('incorrect fraction of the kept set (all classes), sigma ranking', fontsize=9); axs[1, 0].set_xlabel('fraction kept')
    axs[1, 1].set_title('precision (|e| < 30 nm) by score: u_all -, sigma --, p :, photons -.', fontsize=9); axs[1, 1].set_xlabel('fraction kept'); axs[1, 1].set_ylim(0.6, 1)
    axs[1, 2].set_title('precision (|e| < 30 nm) per u_all decile', fontsize=9); axs[1, 2].set_xlabel('u_all'); axs[1, 2].set_ylim(0.6, 1)
    plt.tight_layout(); plt.savefig(fn, dpi=90); plt.close(fig)


if __name__ == '__main__':
    main()
