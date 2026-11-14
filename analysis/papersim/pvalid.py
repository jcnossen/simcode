"""P_valid: a per-localization probability that a SIMCODE v2 (pass 2) localization is VALID, learned on simulated movies with
ground truth and checked on held-out movies (split by movie, not by row).

Valid = not one of the failure classes of selection.py (fringe = one-pattern-period solve error; blend = >= 2 ON emitters
within 150 nm and |e| >= 30 nm; false = no ON emitter within 150 nm; gross = single emitter, |e| >= 50 nm). 'Merely imprecise'
rows (single emitter, 30-50 nm) and all |e| < 30 nm rows are valid: their error is what the calibrated sigma states.
Variant: P(|e| < 30 nm).
Rows: every row of the pass-2 table (all stored candidates, p >= 0.3).
Features (all available on real data; nothing from the ground truth):
  sig (mean calibrated sigma, nm), sig_ratio (max / min of sx, sy), p, log photons, q_dof, kin, I modulation (std / mean of the
  6 network intensities, and min / max), prior distance |r - r_prior| (nm) and its largest projection on the two pattern wave
  vectors in units of the fringe period (k from the run's pattern_est.pt), local density: distance to the nearest other row of
  the same window (nm) and number of rows of the same window within 3 px.
Model: sklearn HistGradientBoostingClassifier (monotone: sig -, p +, nearest-neighbour distance +), rows weighted so every
training movie counts equally, then isotonic calibration fitted on the out-of-fold predictions of the training movies (4 folds of
whole movies). Baselines: constant (the training rate) and sigma only (isotonic on sig).
Evaluation per held-out movie: reliability (10 bins), Brier (model / constant / sigma only), ROC AUC; selections p_valid >= t
(t 0.9 / 0.95 / 0.99) and the paper's rule sigma <= 12 nm (simcode.strict.select; it replaced u <= 0.45, removed 2026-11-14): fraction of rows kept, observed fraction valid, FRC
(frc_fair halves, disjoint windows) and Jaccard (10 nm, papersim movies).
Also: sigma check (item 5): robust z-scale of e / sigma and the fraction |z| > 3 per axis, on valid rows, single and all.
usage: python analysis/papersim/pvalid.py [--model m1809] [--test m05r3,D2,...] -> out/papersim/selection/pvalid_<model>.{json,png,joblib}"""
import os, sys, json, glob, argparse, time
import numpy as np
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import selection as S

OUTD = os.path.join(C.OUT, 'selection')
FEATS = ['sig', 'sig_ratio', 'p', 'logph', 'q_dof', 'kin', 'Imod', 'Iminmax', 'dprior', 'dprior_fringe', 'nn_dist', 'n3px']
MONO = dict(sig=-1, p=1, nn_dist=1)


def pattern_k(locs):
    import torch
    f = os.path.join(os.path.dirname(locs), 'pattern_est.pt')
    try:
        return torch.load(f, weights_only=False)['k'].numpy().astype(np.float64)
    except Exception:
        kk = 2 * np.pi / C.PITCH_NM * C.PX_NM
        return np.array([[kk, 0], [0, kk]])


def features(z, k):
    x, y, f = z['x'].astype(np.float64), z['y'].astype(np.float64), z['frame'].astype(np.float64)
    sx, sy = z['sx'].astype(np.float64), z['sy'].astype(np.float64)
    I = z['I'].astype(np.float64); Im = I.mean(1)
    dx, dy = x - z['x_prior'], y - z['y_prior']
    kh = k / np.linalg.norm(k, axis=1, keepdims=True); per = 2 * np.pi / np.linalg.norm(k, axis=1)
    fr = np.max(np.abs(np.stack([(dx * kh[a, 0] + dy * kh[a, 1]) / per[a] for a in (0, 1)])), 0)
    tree = cKDTree(np.c_[x, y, f * 1000.0])
    d, _ = tree.query(np.c_[x, y, f * 1000.0], k=2)
    nn = np.minimum(d[:, 1], 50.0)                                   # other windows are >= 1000 away; cap at 50 px
    n3 = tree.query_ball_point(np.c_[x, y, f * 1000.0], 3.0, return_length=True) - 1
    F = dict(sig=0.5 * (sx + sy) * C.PX_NM, sig_ratio=np.maximum(sx, sy) / np.maximum(np.minimum(sx, sy), 1e-6), p=z['p'].astype(np.float64),
             logph=np.log(np.maximum(z['photons'].astype(np.float64), 1.0)), q_dof=np.nan_to_num(z['q_dof'].astype(np.float64), nan=10.0),
             kin=z['kin'].astype(np.float64), Imod=I.std(1) / np.maximum(np.abs(Im), 1.0), Iminmax=I.min(1) / np.maximum(I.max(1), 1.0),
             dprior=np.hypot(dx, dy) * C.PX_NM, dprior_fringe=fr, nn_dist=nn * C.PX_NM, n3px=n3.astype(np.float64))
    return np.stack([F[c] for c in FEATS], 1)


def datasets(model):
    D = {d['name']: d for d in S.datasets(model)}
    for name in list(D):
        D[name]['nframes'] = json.load(open(D[name]['gen']))['nframes'] if D[name].get('gen') else 4800
    for e in sorted(glob.glob(os.path.join(C.OUT, 'Dexp*'))):          # experimental-photon density movies (when processed)
        L = os.path.join(e, 'full', f'v2_{model}', 'locs_p2_sim.npz')
        if os.path.exists(L):
            n = os.path.basename(e); D[n] = dict(name=n, kind='papersim', locs=L, gt=os.path.join(e, 'gt.npz'), gen=os.path.join(e, 'gen.json'),
                                                 nframes=json.load(open(os.path.join(e, 'gen.json')))['nframes'])
    for h in sorted(glob.glob(os.path.join(C.OUT, 'hollow', '1809_2_R*_x*'))):     # all hollow photon levels
        n = 'hollow_' + os.path.basename(h); L = os.path.join(h, 'full', f'v2_{model}', 'locs_p2_sim.npz')
        if n not in D and os.path.exists(L):
            D[n] = dict(name=n, kind='papersim', locs=L, gt=os.path.join(h, 'gt.npz'), gen=os.path.join(h, 'gen.json'), nframes=30000)
    return D


def labels(ds, z):
    T = {k: z[k] for k in ('x', 'y', 'frame', 'sx', 'sy')}
    T['x'] = T['x'].astype(np.float64); T['y'] = T['y'].astype(np.float64)
    G = S.load_gt(ds)
    err, n150, fringe = S.classify(T, G)
    e = np.hypot(*err.T); off = np.nanmedian(err[e < 50], 0); err = err - off; e = np.hypot(*err.T)
    has = np.isfinite(e); single = n150 == 1; bad = ~has | (e >= 30)
    fail = bad & (fringe | (n150 >= 2) | ~has | (n150 == 0) | (single & (e >= 50)))
    return dict(valid=~fail, c30=has & (e < 30), err=err, n150=n150, fringe=fringe, offset=off.tolist())


def load(ds, cache_dir):
    cf = os.path.join(cache_dir, f"{ds['name']}_{os.path.basename(os.path.dirname(ds['locs']))}.npz")
    z = np.load(ds['locs'])
    if os.path.exists(cf) and os.path.getmtime(cf) > os.path.getmtime(ds['locs']):
        c = np.load(cf); return z, c['X'], dict(valid=c['valid'], c30=c['c30'], err=c['err'], n150=c['n150'])
    X = features(z, pattern_k(ds['locs'])); L = labels(ds, z)
    np.savez(cf, X=X, valid=L['valid'], c30=L['c30'], err=L['err'], n150=L['n150'])
    return z, X, L


def fit(Xs, ys, groups, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    mono = [MONO.get(c, 0) for c in FEATS]
    def model():
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                                              monotonic_cst=mono, random_state=seed)
    w = np.concatenate([np.full(len(y), 1.0 / len(y)) for y in ys]) * len(ys)
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate([np.full(len(v), i) for i, v in enumerate(ys)])
    oof = np.zeros(len(y))
    fold = np.unique(g) % 4                                      # 4 folds of whole movies -> out-of-fold predictions
    for fi in range(4):
        te = np.isin(g, np.unique(g)[fold == fi])
        if te.all() or not te.any(): continue
        m = model().fit(X[~te], y[~te], sample_weight=w[~te]); oof[te] = m.predict_proba(X[te])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof, y, sample_weight=w)
    full = model().fit(X, y, sample_weight=w)
    sig_iso = IsotonicRegression(increasing=False, out_of_bounds='clip').fit(X[:, FEATS.index('sig')], y, sample_weight=w)
    return dict(gbt=full, iso=iso, sig_iso=sig_iso, rate=float(np.average(y, weights=w)))


def predict(M, X):
    return M['iso'].predict(M['gbt'].predict_proba(X)[:, 1])


def frc_jac(ds, z, keep):
    sys.path.insert(0, os.path.join(C.REPO, 'analysis', 'paper')); import frc_fair, metrics as MT
    x, y, f = z['x'][keep].astype(np.float64), z['y'][keep].astype(np.float64), z['frame'][keep]
    d = f % 6 == 0
    out = dict(frc_nm=float(frc_fair.frc_halves(x[d], y[d], f[d])[0]) if d.sum() > 100 else None)
    if ds['kind'] == 'papersim':
        out['jaccard'] = float(MT.compute_jaccard_points(np.load(ds['gt'])['pts'][:, :2], np.c_[x, y], 10 / C.PX_NM))
    return out


def sigma_check(z, L):
    e = L['err']; s = np.c_[z['sx'], z['sy']] * C.PX_NM; v = L['valid'] & np.isfinite(e).all(1)
    zz = e / s
    out = {}
    for nm, m in (('valid_all', v), ('valid_single', v & (L['n150'] == 1)), ('c30_single', L['c30'] & (L['n150'] == 1))):
        q = zz[m]
        out[nm] = dict(n=int(m.sum()), robust_z_scale=(1.4826 * np.median(np.abs(q), 0)).round(3).tolist(), frac_absz_gt3=(np.abs(q) > 3).mean(0).round(4).tolist(),
                       frac_both_lt3=float((np.abs(q) < 3).all(1).mean()))
    out['frac_rows_single'] = float((L['n150'] == 1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='m1809')
    ap.add_argument('--test', default='m05r3,D2,D0.5,hollow_1809_2_R12.5_x1,hollow_1809_2_R25_x8,hollow_2.5nM_2_R25_x1,Dexp2')
    ap.add_argument('--max_train_rows', type=int, default=150000)
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('psim-pvalid')
    os.makedirs(OUTD, exist_ok=True); cache = os.path.join(OUTD, 'cache'); os.makedirs(cache, exist_ok=True)
    D = datasets(a.model); test = [t for t in a.test.split(',') if t in D]; train = [n for n in D if n not in test]
    print('train', train, '\ntest', test, flush=True)
    rng = np.random.default_rng(0)
    data = {}
    for n in train + test:
        t0 = time.time(); data[n] = load(D[n], cache); print(n, len(data[n][1]), 'rows, valid %.3f' % data[n][2]['valid'].mean(), '%.0fs' % (time.time() - t0), flush=True)
    sub = lambda n: rng.permutation(len(data[n][1]))[:a.max_train_rows]
    idx = {n: sub(n) for n in train}
    MV = fit([data[n][1][idx[n]] for n in train], [data[n][2]['valid'][idx[n]] for n in train], None)
    M30 = fit([data[n][1][idx[n]] for n in train], [data[n][2]['c30'][idx[n]] for n in train], None)
    import joblib; joblib.dump(dict(valid=MV, c30=M30, feats=FEATS, model=a.model, train=train), os.path.join(OUTD, f'pvalid_{a.model}.joblib'))
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.inspection import permutation_importance
    R = dict(model=a.model, train=train, test=test, features=FEATS, movies={})
    for n in test + train:
        z, X, L = data[n]
        r = dict(role='test' if n in test else 'train', n=int(len(X)))
        for tgt, M in (('valid', MV), ('c30', M30)):
            y = L[tgt]; pv = predict(M, X); ps = M['sig_iso'].predict(X[:, FEATS.index('sig')])
            bins = np.clip((pv * 10).astype(int), 0, 9)
            rel = [dict(bin=f'{b / 10:.1f}-{(b + 1) / 10:.1f}', n=int((bins == b).sum()), pred=float(pv[bins == b].mean()), obs=float(y[bins == b].mean()))
                   for b in range(10) if (bins == b).sum() >= 50]
            r[tgt] = dict(rate=float(y.mean()), mean_pred=float(pv.mean()), brier=float(brier_score_loss(y, pv)),
                          brier_const=float(brier_score_loss(y, np.full(len(y), M['rate']))), brier_sigma=float(brier_score_loss(y, ps)),
                          auc=float(roc_auc_score(y, pv)) if 0 < y.mean() < 1 else None,
                          auc_sigma=float(roc_auc_score(y, -X[:, FEATS.index('sig')])) if 0 < y.mean() < 1 else None, reliability=rel)
            if tgt == 'valid':
                pvv = pv
        sel = {}
        for t in (0.9, 0.95, 0.99):
            k = pvv >= t
            sel[f'p_valid>={t}'] = dict(kept=float(k.mean()), n=int(k.sum()), obs_valid=float(L['valid'][k].mean()) if k.any() else None,
                                       obs_c30=float(L['c30'][k].mean()) if k.any() else None, **(frc_jac(D[n], z, k) if k.sum() > 200 else {}))
        from simcode import strict as SP
        import metrics as MT
        k = SP.select(z, SP.SIGMA_MAX_NM, crop=MT.v2_crop(D[n]['locs']) if D[n]['kind'] == 'papersim' else SP.run_crop(os.path.dirname(D[n]['locs'])))
        sel['sigma<=12nm'] = dict(kept=float(k.mean()), n=int(k.sum()), obs_valid=float(L['valid'][k].mean()), obs_c30=float(L['c30'][k].mean()),
                              mean_p_valid=float(pvv[k].mean()), **frc_jac(D[n], z, k))
        r['selections'] = sel
        r['sigma_check'] = sigma_check(z, L)
        if n in test[:3]:
            ii = rng.permutation(len(X))[:40000]
            pi = permutation_importance(MV['gbt'], X[ii], L['valid'][ii], scoring='roc_auc', n_repeats=3, random_state=0)
            r['perm_importance_auc'] = dict(zip(FEATS, pi.importances_mean.round(4).tolist()))
        R['movies'][n] = r
        print(n, r['role'], 'valid rate %.3f pred %.3f brier %.4f const %.4f sigma %.4f auc %.3f (sigma %.3f)' % (
            r['valid']['rate'], r['valid']['mean_pred'], r['valid']['brier'], r['valid']['brier_const'], r['valid']['brier_sigma'], r['valid']['auc'] or 0, r['valid']['auc_sigma'] or 0),
            {k: (round(v['kept'], 3), v['obs_valid'] and round(v['obs_valid'], 4), v.get('frc_nm') and round(v['frc_nm'], 1)) for k, v in sel.items()}, flush=True)
    json.dump(R, open(os.path.join(OUTD, f'pvalid_{a.model}.json'), 'w'), indent=1, default=float)
    plot(R, os.path.join(OUTD, f'pvalid_{a.model}.png'))


def plot(R, fn):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    tests = [n for n, r in R['movies'].items() if r['role'] == 'test']; cm = plt.get_cmap('tab10')
    for i, n in enumerate(tests):
        for ax, tgt in zip(axs[:2], ('valid', 'c30')):
            rel = R['movies'][n][tgt]['reliability']
            ax.plot([b['pred'] for b in rel], [b['obs'] for b in rel], 'o-', color=cm(i % 10), label=n, ms=3)
        s = R['movies'][n]['selections']
        axs[2].plot([s[f'p_valid>={t}']['kept'] for t in (0.9, 0.95, 0.99)], [s[f'p_valid>={t}']['obs_valid'] or np.nan for t in (0.9, 0.95, 0.99)], 'o-', color=cm(i % 10))
        axs[2].plot(s['sigma<=12nm']['kept'], s['sigma<=12nm']['obs_valid'], 'x', color=cm(i % 10), ms=9)
    for ax, t in zip(axs[:2], ('P_valid (class-based)', 'P(|e| < 30 nm)')):
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8); ax.set_xlabel('predicted'); ax.set_ylabel('observed'); ax.set_title(f'reliability, held-out movies: {t}', fontsize=9)
    axs[0].legend(fontsize=6)
    axs[2].set_xlabel('fraction of rows kept'); axs[2].set_ylabel('observed fraction valid'); axs[2].set_title('p_valid >= 0.9 / 0.95 / 0.99 (o), sigma <= 12 nm (x)', fontsize=9)
    for t in (0.9, 0.95, 0.99): axs[2].axhline(t, color='grey', lw=0.5, ls=':')
    plt.tight_layout(); plt.savefig(fn, dpi=90); plt.close(fig)


if __name__ == '__main__':
    main()
