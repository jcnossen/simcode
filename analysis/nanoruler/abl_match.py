"""Nanoruler ablation, part A: matched comparison on IDENTICAL emitter windows.

Methods keep different numbers of localizations per site, so the per-site spread of two tables is not the precision of the same events.
Here the rows of a reference table (nra format, e.g. the published SIMFLUX table) are matched one-to-one to the cycle-aligned rows of a
pipeline table: same cycle, nearest position within --r nm (after removing the median offset; mutual nearest neighbours). The matched
rows of both are written as two nra tables with the same row order (-> nra_run.py, same sites): precision on the same events.
Pipeline rows: by default every cycle-aligned row (--sel all), so that a match does not depend on the pipeline's own selection; the
fraction of reference rows matched and the fraction of those inside the pipeline selections is printed.
usage: python abl_match.py <locs.npz> <drift.npy|none> <ref table.npz> <out prefix> [--sel all] [--r 40] [--also other_ref.npz]"""
import argparse, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
L = A.L


def match(ax, ay, ac, bx, by, bc, r):
    """mutual nearest neighbours within r in the same cycle -> index arrays ia, ib"""
    Z = 1e5
    ta = cKDTree(np.c_[ax, ay, ac * Z]); tb = cKDTree(np.c_[bx, by, bc * Z])
    d, j = tb.query(np.c_[ax, ay, ac * Z], distance_upper_bound=r); g = np.isfinite(d)
    d2, i2 = ta.query(np.c_[bx, by, bc * Z], distance_upper_bound=r)
    ia = np.nonzero(g)[0]; ia = ia[i2[j[ia]] == ia]
    return ia, j[ia]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('locs'); ap.add_argument('drift'); ap.add_argument('ref'); ap.add_argument('out')
    ap.add_argument('--sel', default='all'); ap.add_argument('--r', type=float, default=40.0); ap.add_argument('--also', default=''); ap.add_argument('--sig', default='cal', choices=('cal', 'u'))
    a = ap.parse_args()
    T = np.load(a.locs); T = {k: T[k] for k in T.files}; m = A.selection(T, a.sel)
    d = None if a.drift == 'none' else np.load(a.drift)
    P = A.to_nra(T, m, d, sig=('sx_u', 'sy_u') if a.sig == 'u' and 'sx_u' in T else ('sx', 'sy')); R = L.load_table(a.ref); Rf = np.load(a.ref)
    sh = np.zeros(2)
    for it in range(2):
        ia, ib = match(P['x_nm'] - sh[0], P['y_nm'] - sh[1], P['frame'], R['x_nm'], R['y_nm'], R['frame'], a.r)
        sh = np.array([np.median(P['x_nm'][ia] - R['x_nm'][ib]), np.median(P['y_nm'][ia] - R['y_nm'][ib])])
    dx = P['x_nm'][ia] - R['x_nm'][ib] - sh[0]; dy = P['y_nm'][ia] - R['y_nm'][ib] - sh[1]
    u = np.nan_to_num(np.asarray(T['strict_u'], float), nan=2.0)[P['row'][ia]]
    rep = dict(n_ref=len(R['frame']), n_pipe=len(P['frame']), n_matched=len(ia), frac_ref_matched=len(ia) / len(R['frame']), offset_nm=sh.tolist(),
               diff_rsd_nm=[L.rsd(dx), L.rsd(dy)], frac_matched_good=float(T['good'][P['row'][ia]].mean()), frac_matched_sel=float(T['sel'][P['row'][ia]].mean()),
               frac_matched_u45=float((u <= A.U_MAX).mean()), photons_ratio_median=float(np.median(P['photons'][ia] / R['photons'][ib])))
    print(f"{os.path.basename(a.ref)}: {rep['n_ref']} rows, matched {rep['n_matched']} ({100 * rep['frac_ref_matched']:.1f} %) to {rep['n_pipe']} pipeline rows ({a.sel}); "
          f"offset {sh[0]:+.2f} / {sh[1]:+.2f} nm, robust sd of the difference {rep['diff_rsd_nm'][0]:.2f} / {rep['diff_rsd_nm'][1]:.2f} nm; of the matched pipeline rows "
          f"{100 * rep['frac_matched_good']:.0f} % good, {100 * rep['frac_matched_sel']:.0f} % sel, {100 * rep['frac_matched_u45']:.0f} % u <= 0.45; photons pipeline / reference {rep['photons_ratio_median']:.2f}")
    L.save_table(a.out + '_pipe.npz', **{k: v[ia] for k, v in P.items()})
    L.save_table(a.out + '_ref.npz', **{k: np.asarray(Rf[k])[ib] for k in Rf.files})
    for nm, k in (('u45', u <= A.U_MAX),):       # the sub-set of the matched events inside the pipeline's u <= 0.45 selection
        L.save_table(a.out + f'_{nm}_pipe.npz', **{kk: v[ia][k] for kk, v in P.items()}); L.save_table(a.out + f'_{nm}_ref.npz', **{kk: np.asarray(Rf[kk])[ib][k] for kk in Rf.files})
    if a.also:                                   # a third table matched to the same reference rows (e.g. published SMLM): the 3-way common events
        Q = L.load_table(a.also); Qf = np.load(a.also)
        ja, jb = match(Q['x_nm'], Q['y_nm'], Q['frame'], R['x_nm'], R['y_nm'], R['frame'], a.r)
        c, i1, i2 = np.intersect1d(ib, jb, return_indices=True); rep['n_3way'] = len(c)
        L.save_table(a.out + '_3way_pipe.npz', **{k: v[ia][i1] for k, v in P.items()}); L.save_table(a.out + '_3way_ref.npz', **{k: np.asarray(Rf[k])[c] for k in Rf.files})
        L.save_table(a.out + '_3way_also.npz', **{k: np.asarray(Qf[k])[ja][i2] for k in Qf.files}); print('3-way common events', len(c))
    A.jdump(rep, a.out + '_match.json')


if __name__ == '__main__':
    main()
