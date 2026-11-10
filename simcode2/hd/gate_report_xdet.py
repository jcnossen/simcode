"""
Gate report v2 (critic review 27 items 5, 6, 7): as gate_report but
  - precision targets matched one-to-one (1 px) on the DETECTOR position x_det (not on the final estimate, which
    removed the estimator's own gross errors from the precision set)
  - the matchedFP operating point (a no-op for the single sc front-end) is dropped: one operating point, score >= 0.5
  - precision per nn bin (< 1.5, 1.5-2.5, 2.5-4, > 4 px) with bootstrap CIs over FOVs: RMS and median of err /
    CRLB_SF_multi (zmed = 1.4826 median |e / CRLB| per axis; medrat = median |e| / |CRLB|), trimmed RMS
  - gross = |e| / s_cal > 4 (radial, Gaussian expectation 0.03 %) and |e| / CRLB > 5 (sigma-normalized, not 50 nm)
  - 'prior off > 50 nm and solve follows' counter (prior error > 50 nm and final error > 50 nm) + 'rescued' (final <= 50)
  - modulation slope separately for nn > 4 and nn < 2.5 px (and 2.5-4), per photon bin
  - intensity efficiency as the MEDIAN per-target ratio (rs scenes are outlier-dominated), all-frame and mean-removed
    (the per-emitter common scale is absorbed by the free B in the solve); the diag free-model CRLB fr_m is the
    reference here (the full-covariance version is in simcode2/hd/decomp.py)
usage: python -m simcode2.hd.gate_report_xdet --tag _hd_sc2 --out <out>/decomp/gate_report_xdet_sc2.json
"""
import argparse, glob, json, os
from simcode2 import config
import numpy as np
from simcode2.hd.gate_report import load, match, detection

PX = 108.33
NNB = [(0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 1e9)]
NNS = {'all': (0, 1e9), 'nn<1.5': (0, 1.5), 'nn1.5-2.5': (1.5, 2.5), 'nn2.5-4': (2.5, 4.0), 'nn>4': (4.0, 1e9)}


def prec_stats(e, c, scal=None, ep=None, w=None):
    """e, c [n,2] px (error, CRLB_SF_multi); scal [n,2] chi2-aware sigma; ep [n,2] prior error. w: bootstrap counts."""
    if w is not None:
        e, c = np.repeat(e, w, 0), np.repeat(c, w, 0)
        scal = np.repeat(scal, w, 0) if scal is not None else None
        ep = np.repeat(ep, w, 0) if ep is not None else None
    zc = np.hypot(*(e / c).T)
    out = dict(rms=np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(c ** 2)), zmed=1.4826 * np.median(np.abs(e / c)),
               medrat=np.median(np.hypot(*e.T) / np.hypot(*c.T)),
               trim=np.sqrt(np.mean(e[zc <= 5] ** 2)) / np.sqrt(np.mean(c[zc <= 5] ** 2)), gross_c5=100 * np.mean(zc > 5))
    if scal is not None:
        out['gross_s4'] = 100 * np.mean(np.hypot(*(e / scal).T) > 4)
        out['zcal'] = 1.4826 * np.median(np.abs(e / scal))
    if ep is not None:
        pe, fe = np.hypot(*ep.T) * PX, np.hypot(*e.T) * PX
        out['prior_off50'] = 100 * np.mean(pe > 50)
        out['prior_off50_follows'] = 100 * np.mean((pe > 50) & (fe > 50))
        out['prior_off50_rescued'] = 100 * np.mean((pe > 50) & (fe <= 50))
        out['final_off50_prior_ok'] = 100 * np.mean((pe <= 50) & (fe > 50))
    return {k: float(v) for k, v in out.items()}


def boot(fn, fovs, nboot=200, seed=0):
    """bootstrap over FOVs of a function fn(w) -> dict (w = per-row counts); returns {key: [lo, hi]}"""
    rng = np.random.default_rng(seed)
    uf, inv = np.unique(fovs, return_inverse=True)
    rows = [[] for _ in uf]
    for i, u in enumerate(inv): rows[u].append(i)
    rows = [np.array(r) for r in rows]
    B = []
    for _ in range(nboot):
        pick = rng.integers(0, len(uf), len(uf))
        w = np.bincount(np.concatenate([rows[p] for p in pick]), minlength=len(fovs))
        B.append(fn(w))
    return {k: [float(np.percentile([b[k] for b in B], 2.5)), float(np.percentile([b[k] for b in B], 97.5))] for k in B[0]}


def slope(I, G):
    a = I - I.mean(1, keepdims=True); g = G - G.mean(1, keepdims=True)
    return float((a * g).sum() / max((g * g).sum(), 1e-9))


def int_eff_median(E, F, T=6):
    """median per-target efficiency vs a diagonal reference F (per-frame sd): all-frame mean(e^2 / F^2) and mean-removed
    |P e|^2 / E|P eps|^2 (eps ~ N(0, diag F^2)), each normalized by the Gaussian median of the same statistic."""
    r_all = np.mean(E ** 2 / F ** 2, 1)
    mr = E - E.mean(1, keepdims=True)
    vmr = (F ** 2 * (1 - 2 / T) + (F ** 2).sum(1, keepdims=True) / T ** 2).sum(1)
    r_mr = (mr ** 2).sum(1) / vmr
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(E.shape) * F
    ref_all = np.median(np.mean(eps ** 2 / F ** 2, 1))
    pm = eps - eps.mean(1, keepdims=True); ref_mr = np.median((pm ** 2).sum(1) / vmr)
    return float(np.median(r_all) / ref_all), float(np.median(r_mr) / ref_mr)


def report(d, sc, tag, thr=0.5, lo=500, hi=5000, nboot=200, phb=None):
    R, nf = load(d, sc, tag)
    rep = dict(n_seeds=nf, n_targets=int((R['gt_target'] > 0).sum()), n_fov=int(len(np.unique(R['gt_fov']))))
    rep['det_thr0.5'] = detection(R, 'sc', 'sc', thr)
    idx = np.nonzero(R['sc_score'] >= thr)[0]
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
    ti = og[og >= 0]; oi = idx[og >= 0]
    k = R['gt_target'][ti] > 0; ti, oi = ti[k], oi[k]
    ph = R['gt_phot'][ti]; c = R['gt_sf_m'][ti]; nn = R['gt_nn'][ti]
    tr = np.stack([R['gt_x'][ti], R['gt_y'][ti]], 1)
    e = np.stack([R['sc_x_sc'][oi], R['sc_y_sc'][oi]], 1) - tr
    ep = np.stack([R['sc_x_prior'][oi], R['sc_y_prior'][oi]], 1) - tr
    scal = R['sc_scal_sc'][oi]
    res = (ph >= lo) & (ph < hi) & np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * PX < 25) & np.isfinite(e).all(1)
    nt = ((R['gt_target'] > 0) & (R['gt_phot'] >= lo) & (R['gt_phot'] < hi))
    P = {}
    for nm, (a, b) in NNS.items():
        m = res & (nn >= a) & (nn < b)
        if m.sum() < 20: continue
        q = prec_stats(e[m], c[m], scal[m], ep[m])
        q['ci'] = boot(lambda w: prec_stats(e[m], c[m], scal[m], ep[m], w), R['gt_fov'][ti][m], nboot)
        q['n'] = int(m.sum())
        q['recall1px_det'] = float(((ph >= lo) & (ph < hi) & (nn >= a) & (nn < b)).sum() / max((nt & (R['gt_nn'] >= a) & (R['gt_nn'] < b)).sum(), 1))
        P[nm] = q
    if phb:     # iteration 30b: photon bins over all nn (resolvable targets)
        base = np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * PX < 25) & np.isfinite(e).all(1)
        for plo, phi in phb:
            m = base & (ph >= plo) & (ph < phi)
            if m.sum() < 20: continue
            q = prec_stats(e[m], c[m], scal[m], ep[m])
            q['ci'] = boot(lambda w: prec_stats(e[m], c[m], scal[m], ep[m], w), R['gt_fov'][ti][m], nboot)
            q['n'] = int(m.sum()); P[f'ph{plo}-{phi:.0f}'] = q
    rep['prec'] = P
    # intensities (all matched targets, GT photon bins)
    I = R['sc_I_net'][oi]; G = R['gt_I'][ti]; F = R['gt_fr_m'][ti]
    ok = np.isfinite(F).all(1) & (F.max(1) < 1e5)
    S = {}
    for nm in ('nn<2.5', 'nn2.5-4', 'nn>4'):
        a, b = {'nn<2.5': (0, 2.5), 'nn2.5-4': (2.5, 4), 'nn>4': (4, 1e9)}[nm]
        for plo, phi in ((300, 1000), (1000, 3000), (3000, 20000)):
            m = (nn >= a) & (nn < b) & (ph >= plo) & (ph < phi)
            if m.sum() < 30: continue
            q = dict(n=int(m.sum()), slope=slope(I[m], G[m]), rel_bias=float(np.median(I[m].sum(1) / G[m].sum(1)) - 1))
            mo = m & ok
            if mo.sum() >= 30:
                q['eff_med'], q['eff_mr_med'] = int_eff_median(I[mo] - G[mo], F[mo])
            S[f'{nm}|{plo}-{phi}'] = q
    rep['intensity'] = S
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(config.root('out'), 'gate'))
    ap.add_argument('--scen', default='iso,pair1,pair1.5,pair2,pair3,rs05x,rs05,rs25x,rs25,rs25s,rs05p,rs25p')
    ap.add_argument('--tag', default='_hd_sc2')
    ap.add_argument('--lo', type=float, default=500)
    ap.add_argument('--hi', type=float, default=5000)
    ap.add_argument('--phbins', type=int, default=0)       # 1: + photon bins 200-400 / 400-700 / 700-1200 / 1200-2000 / > 2000
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    REP = {}
    for sc in a.scen.split(','):
        if not glob.glob(f'{a.dir}/{sc}__s*{a.tag}.npz'): continue
        REP[sc] = r = report(a.dir, sc, a.tag, lo=a.lo, hi=a.hi,
                             phb=((200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 1e9)) if a.phbins else None)
        d = r['det_thr0.5']
        print(f'{sc:8s} P {d["precision"]:.3f} R {d["recall"]:.3f} FP/fov {d["fp_per_fov"]:.2f}', flush=True)
        for nm, q in r['prec'].items():
            ci = q['ci']
            print(f'   {nm:10s} n {q["n"]:5d} rms {q["rms"]:.2f} [{ci["rms"][0]:.2f},{ci["rms"][1]:.2f}] zmed {q["zmed"]:.2f} '
                  f'[{ci["zmed"][0]:.2f},{ci["zmed"][1]:.2f}] trim {q["trim"]:.2f} gross|e|/s>4 {q["gross_s4"]:.2f}% '
                  f'|e|/CRLB>5 {q["gross_c5"]:.2f}% prior>50&follows {q["prior_off50_follows"]:.2f}% rescued {q["prior_off50_rescued"]:.2f}% zcal {q["zcal"]:.2f}')
        for nm, q in r['intensity'].items():
            print(f'   I {nm:22s} n {q["n"]:5d} slope {q["slope"]:.3f} eff_med {q.get("eff_med", np.nan):.2f} eff_mr_med {q.get("eff_mr_med", np.nan):.2f}')
    if a.out: json.dump(REP, open(a.out, 'w'), indent=1)
