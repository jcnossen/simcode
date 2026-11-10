"""Check 1 (2026-11-10, field-dependent aberrations): pattern-vs-PSF position consistency per FOV tile.

d = (x_hy0 - x_prior, y_hy0 - y_prior): final pattern-solve position minus the pattern-free network prior (PSF-based).
A field-dependent PSF error that the network does not model biases the prior per tile (mean of d) and inflates its error
(width of d, normalised by the expected sqrt(s_prior^2 - s_final^2)). A non-linear pattern phase field gives the same mean
map (gauge-equivalent) but no width excess.

Per dataset, per tile (4x4 and 8x8 over the processed crop [5, 115) px) and photon bin (label units):
  n, mean dx / dy (winsorised at the bin's global median +- 4 robust sd), sd dx / dy, rms of zd = d / sqrt(sp^2 - sf^2),
  median q_dof, M4 zcal (consecutive photon-disjoint windows w, w+6), and for sims the true errors of prior and final.
Bootstrap: Poisson weights on 2x2 px spatial cells (site heterogeneity), 200 replicates.
Selection: disjoint windows (wstart % 6 == 0), p >= 0.63, finite; 'iso' subset: no other detection of the same window
within 4 px.
usage: python check1_offsets.py   (writes <out>/field/check1_<ds>.npz)
"""
import glob, os, sys, json
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import numpy as np
from scipy.spatial import cKDTree

PX = 108.33
from simcode2 import config
H = config.root('out')
SP = os.path.join(H, 'scratch')
DS = {'real05': f'{H}/full_0.5nM_sc5nb_bright_real', 'real25': f'{H}/full_2.5nM_m2_sc5nb_bright_real',
      'sim05': f'{SP}/sb2_ours/m05qh_sc5nb_b10', 'sim25': f'{SP}/sb2_ours/m25qh_sc5nb_b10'}
GT = {'sim05': f'{SP}/sb_m05qh/sim_gt.npz', 'sim25': f'{SP}/sb_m25qh/sim_gt.npz'}
PB = [(200, 700), (700, 1200), (1200, 2000), (2000, 1e9), (200, 1e9)]
LO, HI = 5.0, 115.0
OUT = config.out_dir('field')
P2K = ['x_hy0', 'y_hy0', 'x_prior', 'y_prior', 'sx_cal', 'sy_cal', 'q_dof', 'phot', 'p', 'wstart', 'dnll']


def load(run):
    Q = {k: [] for k in P2K + ['sxp', 'syp', 'nn']}
    for f2 in sorted(glob.glob(f'{run}/pass2_est_*.npz'), key=lambda s: int(s.split('_')[-1][:-4])):
        f1 = f2.replace('pass2_est_', 'pass1_')
        a, b = np.load(f1), np.load(f2)
        o = a['out'].astype(bool)
        assert o.sum() == len(b['x_prior']) and np.array_equal(a['wstart'][o], b['wstart'])
        # nearest other detection of the same window (all detections)
        P = np.c_[a['x'], a['y'], a['wstart'] * 1e4]
        dd, _ = cKDTree(P).query(P[o], k=2)
        for k in P2K: Q[k].append(b[k])
        Q['sxp'].append(a['sx'][o]); Q['syp'].append(a['sy'][o]); Q['nn'].append(dd[:, 1])
    return {k: np.concatenate(v) for k, v in Q.items()}


def gt_errors(O, gtf):
    """true error of final / prior: nearest GT event (<1 px) overlapping the window [w, w+6); global offset removed."""
    g = np.load(gtf)
    gx, gy, t0, t1 = g['x'].astype(np.float64), g['y'].astype(np.float64), g['t0'], g['t1']
    tr = cKDTree(np.c_[gx, gy])
    x, y, w = O['x_hy0'].astype(np.float64), O['y_hy0'].astype(np.float64), O['wstart']
    dd, jj = tr.query(np.c_[x, y], k=8, distance_upper_bound=1.0)
    best = np.full(len(x), -1); bd = np.full(len(x), np.inf)
    for c in range(8):
        j = jj[:, c]; ok = np.isfinite(dd[:, c])
        jc = np.where(ok, j, 0)
        ov = ok & (t1[jc] > w) & (t0[jc] < w + 6)
        upd = ov & (dd[:, c] < bd)
        best[upd] = jc[upd]; bd[upd] = dd[upd, c]
    m = best >= 0
    ef = np.full((len(x), 2), np.nan); ep = np.full((len(x), 2), np.nan)
    ef[m] = np.c_[x[m] - gx[best[m]], y[m] - gy[best[m]]] * PX
    ep[m] = np.c_[O['x_prior'][m] - gx[best[m]], O['y_prior'][m] - gy[best[m]]] * PX
    off_f = np.nanmedian(ef, 0); off_p = np.nanmedian(ep, 0)
    return ef - off_f, ep - off_p, off_f, off_p


def m4_pairs(O):
    w = O['wstart']; x, y = O['x_hy0'].astype(np.float64), O['y_hy0'].astype(np.float64)
    s = np.c_[O['sx_cal'], O['sy_cal']].astype(np.float64)
    T1 = cKDTree(np.c_[x, y, w * 1e3])
    d, j = T1.query(np.c_[x, y, (w + 6) * 1e3], distance_upper_bound=1.0)
    ok = np.isfinite(d); i = np.nonzero(ok)[0]; j = j[ok]
    d2, i2 = T1.query(np.c_[x[j], y[j], (w[j] - 6) * 1e3], distance_upper_bound=1.0)
    mut = i2 == i; i, j = i[mut], j[mut]
    z = np.c_[x[i] - x[j], y[i] - y[j]] / np.sqrt(s[i] ** 2 + s[j] ** 2)
    return i, j, z


def winsor(v, lo, hi):
    return np.clip(v, lo, hi)


def tile_stats(O, nt, sub, E=None, nboot=200, seed=0):
    """per (tile, photon bin) stats on rows `sub`; returns dict of arrays [nt, nt, nbin] (+ boot SE)."""
    rng = np.random.default_rng(seed)
    x, y = O['x_prior'][sub].astype(np.float64), O['y_prior'][sub].astype(np.float64)
    ts = (HI - LO) / nt
    tx = np.clip(((x - LO) // ts).astype(int), 0, nt - 1); ty = np.clip(((y - LO) // ts).astype(int), 0, nt - 1)
    cell = (np.clip((x - LO) // 2, 0, 54).astype(int) * 64 + np.clip((y - LO) // 2, 0, 54).astype(int))
    ucell, cinv = np.unique(cell, return_inverse=True)
    d = np.c_[O['x_hy0'][sub] - O['x_prior'][sub], O['y_hy0'][sub] - O['y_prior'][sub]].astype(np.float64) * PX
    sp = np.c_[O['sxp'][sub], O['syp'][sub]].astype(np.float64); sf = np.c_[O['sx_cal'][sub], O['sy_cal'][sub]].astype(np.float64)
    sexp = np.sqrt(np.maximum(sp ** 2 - sf ** 2, (0.2 * sp) ** 2)) * PX
    ph = O['phot'][sub]
    nb = len(PB)
    R = {k: np.full((nt, nt, nb) + sh, np.nan) for k, sh in
         [('n', ()), ('mx', ()), ('my', ()), ('sdx', ()), ('sdy', ()), ('zx', ()), ('zy', ()), ('q50', ()),
          ('mx_se', ()), ('my_se', ()), ('sdx_se', ()), ('sdy_se', ()), ('zx_se', ()), ('zy_se', ()),
          ('efx', ()), ('efy', ()), ('epx', ()), ('epy', ()), ('efx_se', ()), ('efy_se', ()), ('epx_se', ()), ('epy_se', ())]}
    R['mx_b'] = np.full((nboot, nt, nt, nb), np.nan); R['my_b'] = np.full((nboot, nt, nt, nb), np.nan)
    for bi, (lo, hi) in enumerate(PB):
        mb = (ph >= lo) & (ph < hi)
        if mb.sum() < 100: continue
        cols = {}
        for c, nm in ((0, 'x'), (1, 'y')):
            v = d[mb, c]; med = np.median(v); rs = 1.4826 * np.median(np.abs(v - med))
            cols['d' + nm] = winsor(v, med - 4 * rs, med + 4 * rs)
            zz = d[mb, c] / sexp[mb, c]
            cols['z' + nm] = winsor(zz, -4, 4)
            if E is not None:
                for en, ea in (('ef', E[0]), ('ep', E[1])):
                    vv = ea[sub][mb, c]
                    okv = np.isfinite(vv)
                    medv = np.nanmedian(vv); rsv = 1.4826 * np.nanmedian(np.abs(vv - medv))
                    cols[en + nm] = np.where(okv, winsor(vv, medv - 4 * rsv, medv + 4 * rsv), np.nan)
        tid = tx[mb] * nt + ty[mb]; ci = cinv[mb]
        q = O['q_dof'][sub][mb]
        for t in range(nt * nt):
            mt = tid == t
            if mt.sum() < 20: continue
            i_, j_ = divmod(t, nt)
            R['n'][i_, j_, bi] = mt.sum(); R['q50'][i_, j_, bi] = np.median(q[mt])
        # sums per (cell, tile) group once; Poisson cell weights then act on the groups (exactly = weighting rows by cell)
        gid = ci.astype(np.int64) * (nt * nt) + tid
        ug, ginv = np.unique(gid, return_inverse=True)
        g_cell = ug // (nt * nt); g_tile = ug % (nt * nt)
        GS = {}
        for k, v in cols.items():
            okk = np.isfinite(v); vv = np.where(okk, v, 0)
            GS[k] = [np.bincount(ginv, okk * 1.0, len(ug)), np.bincount(ginv, vv, len(ug)), np.bincount(ginv, vv * vv, len(ug))]
        def agg(wc):
            wg = wc[g_cell]
            out = {}
            for k, (s0, s1, s2) in GS.items():
                S0 = np.bincount(g_tile, wg * s0, nt * nt); S1 = np.bincount(g_tile, wg * s1, nt * nt); S2 = np.bincount(g_tile, wg * s2, nt * nt)
                mu = S1 / np.maximum(S0, 1e-9); out[k] = (mu, np.sqrt(np.maximum(S2 / np.maximum(S0, 1e-9) - mu ** 2, 0)), S0)
            return out
        A0 = agg(np.ones(len(ucell)))
        B = [agg(rng.poisson(1.0, len(ucell)).astype(np.float64)) for _ in range(nboot)]
        sh = (nt, nt)
        def put(key, k, which):
            R[key][..., bi] = A0[k][which].reshape(sh)
            R[key + '_se'][..., bi] = np.std([b[k][which] for b in B], 0).reshape(sh)
        put('mx', 'dx', 0); put('my', 'dy', 0); put('sdx', 'dx', 1); put('sdy', 'dy', 1)
        # rms of zd (sqrt(mean z^2)) -> use sd of z around tile mean ~ width; store rms
        for key, k in (('zx', 'zx'), ('zy', 'zy')):
            mu, sd, _ = A0[k]; R[key][..., bi] = np.sqrt(mu ** 2 + sd ** 2).reshape(sh)
            R[key + '_se'][..., bi] = np.std([np.sqrt(b[k][0] ** 2 + b[k][1] ** 2) for b in B], 0).reshape(sh)
        R['mx_b'][..., bi] = np.stack([b['dx'][0].reshape(sh) for b in B]); R['my_b'][..., bi] = np.stack([b['dy'][0].reshape(sh) for b in B])
        if E is not None:
            put('efx', 'efx', 0); put('efy', 'efy', 0); put('epx', 'epx', 0); put('epy', 'epy', 0)
        lowN = R['n'][..., bi] < 20
        for kk in R:
            if kk.endswith('_b'): R[kk][:, lowN, bi] = np.nan
            elif kk != 'n': R[kk][lowN, bi] = np.nan
    # M4 zcal per tile (pairs within `sub`)
    return R


def m4_tiles(O, sub, nt):
    idx = np.nonzero(sub)[0]
    Os = {k: O[k][idx] for k in ('x_hy0', 'y_hy0', 'sx_cal', 'sy_cal', 'wstart', 'phot', 'x_prior', 'y_prior')}
    i, j, z = m4_pairs(Os)
    ts = (HI - LO) / nt
    x, y = Os['x_hy0'][i].astype(np.float64), Os['y_hy0'][i].astype(np.float64)
    tx = np.clip(((x - LO) // ts).astype(int), 0, nt - 1); ty = np.clip(((y - LO) // ts).astype(int), 0, nt - 1)
    pm = 0.5 * (Os['phot'][i] + Os['phot'][j])
    Z = np.full((nt, nt, len(PB)), np.nan); N = np.zeros((nt, nt, len(PB)))
    for bi, (lo, hi) in enumerate(PB):
        mb = (pm >= lo) & (pm < hi)
        for a in range(nt):
            for b in range(nt):
                m = mb & (tx == a) & (ty == b)
                N[a, b, bi] = m.sum()
                if m.sum() >= 30: Z[a, b, bi] = 1.4826 * np.median(np.abs(z[m]))
    return Z, N


def spread(m, se):
    """rms of tile means about their n-weighted mean, noise rms, chi2/dof, excess rms (nm)."""
    ok = np.isfinite(m) & np.isfinite(se) & (se > 0)
    if ok.sum() < 3: return dict(rms=np.nan, noise=np.nan, chi2=np.nan, excess=np.nan, n=int(ok.sum()))
    w = 1 / se[ok] ** 2; mu = np.sum(w * m[ok]) / w.sum()
    r = m[ok] - mu
    rms = np.sqrt(np.mean(r ** 2)); noise = np.sqrt(np.mean(se[ok] ** 2))
    chi2 = np.sum(r ** 2 / se[ok] ** 2) / (ok.sum() - 1)
    return dict(rms=float(rms), noise=float(noise), chi2=float(chi2), excess=float(np.sqrt(max(rms ** 2 - noise ** 2, 0))), n=int(ok.sum()))


def main():
    which = sys.argv[1:] or list(DS)
    for ds in which:
        O = load(DS[ds])
        base = (O['wstart'] % 6 == 0) & (O['p'] >= 0.63) & np.isfinite(O['x_hy0']) & np.isfinite(O['sx_cal']) & \
               (O['x_prior'] >= LO) & (O['x_prior'] < HI) & (O['y_prior'] >= LO) & (O['y_prior'] < HI)
        E = None; info = {}
        if ds in GT:
            ef, ep, of, op = gt_errors(O, GT[ds]); E = (ef, ep); info['gt_off_final_nm'] = of.tolist(); info['gt_off_prior_nm'] = op.tolist()
            info['gt_match'] = float(np.isfinite(ef[base, 0]).mean())
        res = {}
        for sname, sub in (('all', base), ('iso', base & (O['nn'] >= 4.0))):
            for nt in (4, 8):
                R = tile_stats(O, nt, sub, E)
                Z, N = m4_tiles(O, sub, nt)
                R['zcal'] = Z; R['zcal_n'] = N
                for k, v in R.items(): res[f'{sname}_{nt}_{k}'] = v
        np.savez_compressed(f'{OUT}/check1_{ds}.npz', **res, pb=np.array(PB), lo=LO, hi=HI, info=json.dumps(info),
                            n_rows=int(base.sum()), n_iso=int((base & (O['nn'] >= 4.0)).sum()))
        print(ds, 'rows', int(base.sum()), 'iso', int((base & (O['nn'] >= 4.0)).sum()), info, flush=True)


if __name__ == '__main__':
    main()
