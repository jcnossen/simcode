"""Site-size study, diagnosis of the slow component: is the slow motion of the site centres (time scale 100-1000 cycles) common to the field
(residual translation drift), field-dependent (affine: rotation / scale / shear), common to the sites of one ruler (rigid motion of the
structure), or site-specific?
Per site and time block (BLK cycles): median position of its localizations in the block minus the site median (>= 3 localizations).
  var_total    mean over (site, block) of r^2 per axis, minus the noise of a block median (pi / 2 sigma_1^2 / n, sigma_1 = this method's
               sigma_pair(Delta = 1) from res_<movie>.json)
  translation  per block the robust mean over sites removed; affine: per block LS fit r = t + A (p - p0) removed (blocks with >= 12 sites)
  covariance   mean of r_i r_j over common blocks for pairs of sites: same ruler / structure, different structures < 2 um, 2-8 um, > 8 um
               (after the per-block translation is removed), as a fraction of the noise-corrected variance.
  python ss_slow.py <movie> <method> ... [--blk 250]  -> out/nanoruler/sitesize/slow_<movie>.json / .md"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ss_lib as S, ss_run as RN
OUT = RN.OUT


def blocks(tab, St, ncyc, blk):
    x, y, g, j, sh = S.assign_sites(tab, St['sc'], St['used'], St['radius'], St['rmax'])
    ns = len(St['sc']); nb = int(np.ceil(ncyc / blk)); R = np.full((ns, nb, 2), np.nan); N = np.zeros((ns, nb))
    b = tab['frame'][g] // blk
    for s in np.unique(j):
        k = j == s
        if k.sum() < S.MIN_N: continue
        xs, ys, bs = x[g[k]], y[g[k]], b[k]; mx, my = np.median(xs), np.median(ys)
        for q in np.unique(bs):
            m = bs == q
            if m.sum() >= 3: R[s, q] = [np.median(xs[m]) - mx, np.median(ys[m]) - my]; N[s, q] = m.sum()
    return R, N


def analyse(R, N, St, unit, sig1):
    ok = np.isfinite(R[..., 0]); noise = np.where(ok, np.pi / 2 * sig1 ** 2 / np.maximum(N, 1), np.nan)
    o = {}; tot = np.nanmean(R ** 2, (0, 1)) - np.nanmean(noise); o['var_total'] = tot.tolist(); o['noise_var'] = float(np.nanmean(noise))
    Rt = R.copy(); Ra = R.copy(); p = St['sc']
    for q in range(R.shape[1]):
        k = ok[:, q]
        if k.sum() < 12: Rt[:, q] = np.nan; Ra[:, q] = np.nan; continue
        Rt[k, q] = R[k, q] - np.median(R[k, q], 0)
        X = np.c_[np.ones(k.sum()), p[k] - p[k].mean(0)]
        for a in range(2):
            c, *_ = np.linalg.lstsq(X, R[k, q, a], rcond=None); Ra[k, q, a] = R[k, q, a] - X @ c
    ok2 = np.isfinite(Rt[..., 0])
    o['var_after_translation'] = (np.nanmean(Rt ** 2, (0, 1)) - np.nanmean(noise[ok2])).tolist()
    o['var_after_affine'] = (np.nanmean(Ra ** 2, (0, 1)) - np.nanmean(noise[ok2]) * 1.0).tolist()
    o['var_total_same_blocks'] = (np.nanmean(np.where(ok2[..., None], R, np.nan) ** 2, (0, 1)) - np.nanmean(noise[ok2])).tolist()
    # pair covariance after translation removal
    idx = np.nonzero(ok2.sum(1) >= 3)[0]; from scipy.spatial import cKDTree
    T = cKDTree(p[idx]); cls = {'same_structure': [], 'lt2um': [], '2_8um': [], 'gt8um': []}
    rng = np.random.RandomState(1)
    for a_, b_ in T.query_pairs(30000.0, output_type='ndarray') if len(idx) < 1500 else []:
        i, jj = idx[a_], idx[b_]; m = ok2[i] & ok2[jj]
        if m.sum() < 3: continue
        d = np.hypot(*(p[i] - p[jj])); c = (Rt[i, m] * Rt[jj, m]).sum(1).mean() / 2       # per axis mean
        k = 'same_structure' if unit[i] == unit[jj] else ('lt2um' if d < 2000 else ('2_8um' if d < 8000 else 'gt8um'))
        cls[k].append(c)
    vt = float(np.mean(o['var_after_translation']))
    o['pair_cov'] = {k: dict(n=len(v), cov=float(np.mean(v)) if v else None, frac_of_var=float(np.mean(v) / vt) if v and vt > 0 else None,
                             se=float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else None) for k, v in cls.items()}
    return o


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('movie'); ap.add_argument('methods', nargs='+'); ap.add_argument('--blk', type=int, default=250)
    a = ap.parse_args(); mv = a.movie; St = RN.sites_of(mv); nc = RN.NCYC[mv]
    if mv == 'nr80':
        d = json.load(open(f'{RN.NR}/analysis/sites.json')); unit = np.array([s['ruler'] for s in d['sites']])
    elif mv in ('nr_a', 'nr_c0'): unit = np.load(f'{RN.NR}/sim/{mv}/sites.npz')['ruler']
    else:
        fn = f'{RN.NR}/nr40/pairs_auto_sf.json' if mv == 'nr40' else f'{RN.NR}/nr40/simeval/pairs_truth_s40g.json'; unit = np.array(json.load(open(fn))['site_unit'])
    res = json.load(open(f'{OUT}/res_{mv}.json')) if os.path.exists(f'{OUT}/res_{mv}.json') else None
    o = dict(movie=mv, blk=a.blk, methods={})
    for m in a.methods:
        sig1 = res['methods'][m]['axes']['xy']['sigma_pair']['1'][0] if res and m in res['methods'] else float('nan')
        R, N = blocks(RN.load(mv, m), St, nc, a.blk); o['methods'][m] = analyse(R, N, St, unit, sig1); o['methods'][m]['sigma1'] = sig1
        r = o['methods'][m]; print(m, json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in r.items() if k != 'pair_cov'}),
                                    {k: (v['n'], None if v['frac_of_var'] is None else round(v['frac_of_var'], 2)) for k, v in r['pair_cov'].items()}, flush=True)
    json.dump(o, open(f'{OUT}/slow_{mv}_b{a.blk}.json', 'w'), indent=1)
