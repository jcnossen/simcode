"""Check 1b: field maps of d = final - prior on a common SENSOR grid (0.5 nM movie pixel frame = reference;
2.5 nM m2 pixel + (dx 2, dy 22) -> reference; PLAN 2026-11-08 sensor ROI shift). A sensor-fixed (optical) effect
replicates between the two datasets and photon bins; sample structure does not.
Outputs maps_<ds>.npz (cells 10 px, photon bins), maps_summary.txt (cross-dataset / cross-bin correlations,
Legendre-2 smooth-fit R^2, the width map), used by plot_field_offsets.py.
"""
import os, numpy as np, json, sys
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode2 import config
from check1_offsets import load, DS, PX, gt_errors, GT

OUT = config.out_dir('field')
SHIFT = {'real25': (2.0, 22.0), 'sim25': (0.0, 0.0)}     # (dx, dy) to the reference sensor frame
CS = 10.0; G0, G1 = 5.0, 137.0                          # common grid covering both ROIs
NC = int(np.ceil((G1 - G0) / CS))
PB = [(200, 700), (700, 1200), (1200, 2000), (2000, 1e9), (700, 1e9)]
NB = 100


def leg_basis(u, v):
    return np.stack([np.ones_like(u), u, v, 0.5 * (3 * u ** 2 - 1), u * v, 0.5 * (3 * v ** 2 - 1)], -1)


def cellmaps(O, sel, sh, E=None, seed=0):
    rng = np.random.default_rng(seed)
    x = O['x_prior'][sel] + sh[0]; y = O['y_prior'][sel] + sh[1]
    cx = np.clip(((x - G0) // CS).astype(int), 0, NC - 1); cy = np.clip(((y - G0) // CS).astype(int), 0, NC - 1)
    cid = cy * NC + cx
    d = np.c_[O['x_hy0'][sel] - O['x_prior'][sel], O['y_hy0'][sel] - O['y_prior'][sel]].astype(np.float64) * PX
    sp = np.c_[O['sxp'][sel], O['syp'][sel]].astype(np.float64); sf = np.c_[O['sx_cal'][sel], O['sy_cal'][sel]].astype(np.float64)
    z = d / (np.sqrt(np.maximum(sp ** 2 - sf ** 2, (0.2 * sp) ** 2)) * PX)
    ph = O['phot'][sel]
    sub = (np.clip((x - G0) // 2, 0, 80).astype(int) * 128 + np.clip((y - G0) // 2, 0, 80).astype(int))
    us, sinv = np.unique(sub, return_inverse=True)
    M = {}
    for bi, (lo, hi) in enumerate(PB):
        mb = (ph >= lo) & (ph < hi)
        cols = {}
        for c in (0, 1):
            v = d[mb, c]; med = np.median(v); rs = 1.4826 * np.median(np.abs(v - med))
            cols[f'd{c}'] = np.clip(v, med - 4 * rs, med + 4 * rs); cols[f'z{c}'] = np.clip(z[mb, c], -4, 4)
            if E is not None:
                for en, ea in (('ef', E[0]), ('ep', E[1])):
                    vv = ea[sel][mb, c]; ok = np.isfinite(vv); medv = np.nanmedian(vv); rsv = 1.4826 * np.nanmedian(np.abs(vv - medv))
                    cols[f'{en}{c}'] = np.where(ok, np.clip(vv, medv - 4 * rsv, medv + 4 * rsv), np.nan)
        gid = sinv[mb].astype(np.int64) * NC * NC + cid[mb]
        ug, ginv = np.unique(gid, return_inverse=True); gs, gc = ug // (NC * NC), ug % (NC * NC)
        GS = {k: [np.bincount(ginv, np.isfinite(v) * 1.0, len(ug)), np.bincount(ginv, np.nan_to_num(v), len(ug)),
                  np.bincount(ginv, np.nan_to_num(v) ** 2, len(ug))] for k, v in cols.items()}
        def agg(w):
            wg = w[gs]; out = {}
            for k, (s0, s1, s2) in GS.items():
                S0 = np.bincount(gc, wg * s0, NC * NC); S1 = np.bincount(gc, wg * s1, NC * NC); S2 = np.bincount(gc, wg * s2, NC * NC)
                mu = S1 / np.maximum(S0, 1e-9)
                out[k] = mu; out[k + '_rms'] = np.sqrt(np.maximum(S2 / np.maximum(S0, 1e-9), 0)); out[k + '_n'] = S0
            return out
        A = agg(np.ones(len(us))); B = [agg(rng.poisson(1.0, len(us)).astype(float)) for _ in range(NB)]
        for k in A:
            if k.endswith('_n'): M[f'{k}_{bi}'] = A[k].reshape(NC, NC); continue
            arr = A[k].reshape(NC, NC); se = np.std([b[k] for b in B], 0).reshape(NC, NC)
            bad = A[k.split('_')[0] + '_n'].reshape(NC, NC) < 30
            arr[bad] = np.nan; se[bad] = np.nan
            M[f'{k}_{bi}'] = arr; M[f'{k}_se_{bi}'] = se
    return M


def wcorr(a, b, sa, sb):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5: return np.nan, 0
    w = 1 / (sa[ok] ** 2 + sb[ok] ** 2)
    a_, b_ = a[ok] - np.average(a[ok], weights=w), b[ok] - np.average(b[ok], weights=w)
    return float(np.sum(w * a_ * b_) / np.sqrt(np.sum(w * a_ ** 2) * np.sum(w * b_ ** 2))), int(ok.sum())


def smooth_fit(m, se):
    """weighted Legendre-2 fit of a cell map over the reference sensor grid; returns R^2 (signal), chi2/dof residual, coefs."""
    ok = np.isfinite(m) & np.isfinite(se) & (se > 0)
    cc = G0 + CS * (np.arange(NC) + 0.5)
    X, Y = np.meshgrid(cc, cc)            # [row=y, col=x]
    u = (X - 60) / 60; v = (Y - 60) / 60
    A = leg_basis(u[ok], v[ok]); w = 1 / se[ok]
    c, *_ = np.linalg.lstsq(A * w[:, None], m[ok] * w, rcond=None)
    r = m[ok] - A @ c; mu = np.average(m[ok], weights=w ** 2)
    var_tot = np.average((m[ok] - mu) ** 2, weights=w ** 2) - np.average(se[ok] ** 2, weights=w ** 2)
    var_res = np.average(r ** 2, weights=w ** 2) - np.average(se[ok] ** 2, weights=w ** 2)
    return dict(R2=float(1 - max(var_res, 0) / max(var_tot, 1e-9)), chi2_res=float(np.sum((r / se[ok]) ** 2) / max(ok.sum() - 6, 1)),
                chi2_flat=float(np.sum(((m[ok] - mu) / se[ok]) ** 2) / max(ok.sum() - 1, 1)), coef=c.tolist(), n=int(ok.sum()))


def main():
    L = []; pr = lambda s='': (print(s, flush=True), L.append(s))
    MM = {}
    for ds in ('real05', 'real25', 'sim05', 'sim25'):
        O = load(DS[ds])
        sel = (O['wstart'] % 6 == 0) & (O['p'] >= 0.63) & np.isfinite(O['x_hy0']) & np.isfinite(O['sx_cal']) & (O['nn'] >= 4.0) & \
              (O['x_prior'] >= 5) & (O['x_prior'] < 115) & (O['y_prior'] >= 5) & (O['y_prior'] < 115)
        E = gt_errors(O, GT[ds])[:2] if ds in GT else None
        M = cellmaps(O, sel, SHIFT.get(ds, (0.0, 0.0)), E)
        np.savez_compressed(f'{OUT}/maps_{ds}.npz', **M, G0=G0, CS=CS, NC=NC, pb=np.array(PB))
        MM[ds] = M
    pr('## d = final - prior, iso (nn >= 4 px), disjoint, p >= 0.63; 10 px cells on the 0.5 nM sensor frame (2.5 nM shifted by dx 2, dy 22)')
    pr('### Legendre-2 smooth fit of the cell-mean map (u, v = (x - 60)/60, (y - 60)/60): R^2 of the noise-corrected variance, chi2/dof flat -> residual')
    for ds in MM:
        for bi, (lo, hi) in enumerate(PB):
            for c, nm in ((0, 'dx'), (1, 'dy')):
                f = smooth_fit(MM[ds][f'd{c}_{bi}'], MM[ds][f'd{c}_se_{bi}'])
                pr(f'  {ds} {lo}-{hi:.0f} {nm}: R2 {f["R2"]:.2f} chi2 flat {f["chi2_flat"]:.1f} -> resid {f["chi2_res"]:.1f} (cells {f["n"]}); '
                   f'coef [1,u,v,P2u,uv,P2v] nm {np.round(f["coef"], 2).tolist()}')
    pr('### cross-bin correlation of the d map (cell-wise, weighted), within dataset: 700-1200 vs 1200-2000 / 200-700 vs >=700')
    for ds in MM:
        for c, nm in ((0, 'dx'), (1, 'dy')):
            r1 = wcorr(MM[ds][f'd{c}_1'], MM[ds][f'd{c}_2'], MM[ds][f'd{c}_se_1'], MM[ds][f'd{c}_se_2'])
            r2 = wcorr(MM[ds][f'd{c}_0'], MM[ds][f'd{c}_4'], MM[ds][f'd{c}_se_0'], MM[ds][f'd{c}_se_4'])
            pr(f'  {ds} {nm}: r(700-1200, 1200-2000) {r1[0]:+.2f} (n {r1[1]}); r(200-700, >=700) {r2[0]:+.2f}')
    pr('### cross-DATASET correlation on the sensor frame (real05 vs real25 overlap; real vs sim as a structure control)')
    for a, b in (('real05', 'real25'), ('real05', 'sim05'), ('real25', 'sim25'), ('sim05', 'sim25')):
        for bi in (4, 0, 1, 2):
            for c, nm in ((0, 'dx'), (1, 'dy')):
                r = wcorr(MM[a][f'd{c}_{bi}'], MM[b][f'd{c}_{bi}'], MM[a][f'd{c}_se_{bi}'], MM[b][f'd{c}_se_{bi}'])
                pr(f'  {a} vs {b} {PB[bi]} {nm}: r {r[0]:+.2f} (cells {r[1]})')
    pr('### width: z-rms of d per cell (>= 700 ph) -- correlation with |mean d| and cross-dataset')
    for ds in MM:
        zr = np.sqrt(0.5 * (MM[ds]['z0_rms_4'] ** 2 + MM[ds]['z1_rms_4'] ** 2)); dm = np.hypot(MM[ds]['d0_4'] - np.nanmedian(MM[ds]['d0_4']), MM[ds]['d1_4'] - np.nanmedian(MM[ds]['d1_4']))
        ok = np.isfinite(zr) & np.isfinite(dm)
        pr(f'  {ds}: z-rms cell p10/p50/p90 {np.nanpercentile(zr, 10):.2f} / {np.nanpercentile(zr, 50):.2f} / {np.nanpercentile(zr, 90):.2f}; corr(z-rms, |d - med|) {np.corrcoef(zr[ok], dm[ok])[0, 1]:+.2f}')
    for c in (0, 1):
        r = wcorr(MM['real05'][f'z{c}_rms_4'], MM['real25'][f'z{c}_rms_4'], MM['real05'][f'z{c}_rms_se_4'] if f'z{c}_rms_se_4' in MM['real05'] else MM['real05'][f'z{c}_se_4'], MM['real25'][f'z{c}_se_4'])
        pr(f'  z-rms {"xy"[c]} real05 vs real25: r {r[0]:+.2f} (cells {r[1]})')
    pr('### sim truth: true tile-mean error maps (>= 700 ph): Legendre-2 chi2 flat (final / prior)')
    for ds in ('sim05', 'sim25'):
        for c, nm in ((0, 'x'), (1, 'y')):
            ff = smooth_fit(MM[ds][f'ef{c}_4'], MM[ds][f'ef{c}_se_4']); fp = smooth_fit(MM[ds][f'ep{c}_4'], MM[ds][f'ep{c}_se_4'])
            pr(f'  {ds} {nm}: final chi2 flat {ff["chi2_flat"]:.2f} rms {np.nanstd(MM[ds][f"ef{c}_4"]):.2f} nm | prior chi2 flat {fp["chi2_flat"]:.2f} rms {np.nanstd(MM[ds][f"ep{c}_4"]):.2f} nm')
    open(f'{OUT}/maps_summary.txt', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
