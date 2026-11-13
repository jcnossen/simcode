"""R24: localization table -> nra / r24_rul format with OWN DME v2 drift (as abl_convert.py: DME on the table's own rows, time unit = one
6-frame cycle), without the nr80 site list abl_lib.drift_report needs. Differences to abl_convert.py: (1) the drift check is a cluster-sharpness
number (sum of squares of the 4-nm histogram blurred by 4 nm, per localization) for no / own / every reference drift, plus rms differences
and correlations to the reference traces at 1 / 50 / 250-cycle smoothing; (2) reference traces are arguments (--cmp name=file.npy [ncyc, 2] px,
"corrected = raw - drift[cycle]"; --gt <sim dir>: the simulated sample drift); (3) extra sigma columns of the variant tables are carried
(sx_raw -> sx_raw_nm ...); (4) --drift <file.npy> applies a given trace (one trace for all variants / selections).
  python r24_tab.py locs  <locs.npz> <out.npz> --ncyc N [--sel u45|sel|good|all] [--drift own|none|file.npy] [--cmp n=f.npy ...] [--gt simdir]
  python r24_tab.py table <nra table.npz> <out.npz> --ncyc N [...]
(5) common frame: the own trace (mean removed by DME) gets the MEAN of a reference trace (--ref_drift; default for real nr40 / nr20 tables: the
published total SIMFLUX correction), so that all tables of a movie share the frame of the published tables up to ~1 nm.
Writes <out>.npz, <out>_drift.npy (px, mean removed), <out>_drift.json. DME runs on the GPU (use r24_gpu.sh)."""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
L = A.L; PX = A.PX


def sharp(x_nm, y_nm, px=4.0):
    lo = min(x_nm.min(), y_nm.min()); n = int((max(x_nm.max(), y_nm.max()) - lo) / px) + 1
    h = np.histogram2d(y_nm, x_nm, bins=n, range=[[lo, lo + n * px]] * 2)[0]
    return float((gaussian_filter(h, 1.0) ** 2).sum() / len(x_nm))


def gt_drift(sim, ncyc):
    G = np.load(f'{sim}/sim_gt.npz', allow_pickle=True); d = G['drift'][:ncyc * 6].reshape(ncyc, 6, 2).mean(1).astype(np.float64)
    return d - d.mean(0)


def convert(kind, src, out, ncyc, sel='u45', drift='own', cmp=(), gt=None, log=print, mask_from=None, ref_drift=None):
    base = os.path.splitext(out)[0]; os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if kind == 'locs':
        T = np.load(src); T = {k: T[k] for k in T.files}; m = A.selection(T if not mask_from else dict(np.load(mask_from)), sel); tab = A.to_nra(T, m)
        for lv in ('raw', 'nb', 'f'):
            if 'sx_' + lv in T: tab[f'sx_{lv}_nm'] = T['sx_' + lv][m].astype(np.float64) * PX; tab[f'sy_{lv}_nm'] = T['sy_' + lv][m].astype(np.float64) * PX
        if 'strict_u' in T: tab['strict_u'] = T['strict_u'][m].astype(np.float64)
    else:
        D = np.load(src); tab = {k: np.asarray(D[k], np.float64) for k in D.files if np.ndim(D[k]) == 1}; tab['frame'] = tab['frame'].astype(np.int64)
        tab['row'] = np.arange(len(tab['frame']))
    k = tab['frame'] < ncyc; tab = {q: v[k] for q, v in tab.items()}
    cyc = tab['frame']; x = tab['x_nm'] / PX; y = tab['y_nm'] / PX; rep = dict(src=src, kind=kind, sel=sel, drift=drift, n=int(len(cyc)), ncyc=ncyc)
    refs = {n: np.load(f) for n, f in (c.split('=', 1) for c in cmp)}
    if gt: refs['truth'] = gt_drift(gt, ncyc)
    refs = {n: (d[:ncyc] - d[:ncyc].mean(0)) for n, d in refs.items()}
    if drift == 'own':
        d, info = A.dme(x, y, cyc, tab['sx_nm'] / PX, tab['sy_nm'] / PX, ncyc); rep['dme'] = info
    elif drift == 'none': d = np.zeros((ncyc, 2))
    else: d = np.load(drift)[:ncyc]
    # common frame: the own trace is mean-removed, the published traces are referenced to the start of the movie, so the corrected tables would
    # differ by a constant (nr20: -6 / +36 nm, more than a grid period of the 20 nm sample). The own trace gets the mean of the reference trace.
    if drift == 'own' and ref_drift is None and gt is None:
        import re
        mm = re.search(r'/out/nanoruler/(nr[0-9]+)/', os.path.abspath(src)); cand = f'{A.NR}/{mm.group(1)}/ref/drift_pub_total_simflux.npy' if mm else ''
        if mm and os.path.exists(cand): ref_drift = cand
    if drift == 'own' and ref_drift:
        rd = np.load(ref_drift)[:ncyc]; c = rd.mean(0) - d.mean(0); d = d + c; rep['ref_drift'] = ref_drift; rep['const_added_px'] = c.tolist()
        log(f'   own drift re-referenced to the mean of {os.path.basename(ref_drift)}: constant {c[0] * PX:+.2f} / {c[1] * PX:+.2f} nm added to the trace')
    rep.update(drift_rms_nm=(d.std(0) * PX).tolist(), drift_p2p_nm=(np.ptp(d, 0) * PX).tolist())
    sh = lambda dd: sharp((x - dd[cyc, 0]) * PX, (y - dd[cyc, 1]) * PX)
    rep['sharp'] = dict(none=sh(np.zeros((ncyc, 2))), own=sh(d), **{n: sh(r) for n, r in refs.items()})
    log(f'{os.path.basename(src)}[{sel if kind == "locs" else "table"}]: {len(cyc)} rows; drift {drift} {rep.get("dme", {}).get("chosen", "")} rms {rep["drift_rms_nm"][0]:.2f} / {rep["drift_rms_nm"][1]:.2f} nm, '
        f'p2p {rep["drift_p2p_nm"][0]:.1f} / {rep["drift_p2p_nm"][1]:.1f} nm; sharpness ' + ', '.join(f'{n} {v:.4f}' for n, v in rep['sharp'].items()))
    for n, r in refs.items():
        q = {}
        for w in (1, 50, 250):
            a, b = A.smooth(d, w), A.smooth(r, w); e = (a - b) - (a - b).mean(0)
            q[f'rms_nm_w{w}'] = (e.std(0) * PX).tolist(); q[f'corr_w{w}'] = [float(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(2)]
        q['max_abs_nm_w50'] = (np.abs(A.smooth(d - r, 50) - (d - r).mean(0)).max(0) * PX).tolist()
        q['ref_rms_nm'] = (r.std(0) * PX).tolist(); q['ref_p2p_nm'] = (np.ptp(r, 0) * PX).tolist(); rep['vs_' + n] = q
        log(f'   own - {n} (ref rms {q["ref_rms_nm"][0]:.1f} / {q["ref_rms_nm"][1]:.1f}, p2p {q["ref_p2p_nm"][0]:.1f} / {q["ref_p2p_nm"][1]:.1f} nm): rms nm raw {q["rms_nm_w1"][0]:.2f} / {q["rms_nm_w1"][1]:.2f}, '
            f'50-cycle {q["rms_nm_w50"][0]:.2f} / {q["rms_nm_w50"][1]:.2f} (corr {q["corr_w50"][0]:.3f} / {q["corr_w50"][1]:.3f}, max |diff| {q["max_abs_nm_w50"][0]:.1f} / {q["max_abs_nm_w50"][1]:.1f}), '
            f'250-cycle {q["rms_nm_w250"][0]:.2f} / {q["rms_nm_w250"][1]:.2f} (corr {q["corr_w250"][0]:.3f} / {q["corr_w250"][1]:.3f})')
    tab['x_nm'] = (x - d[cyc, 0]) * PX; tab['y_nm'] = (y - d[cyc, 1]) * PX
    L.save_table(out, **tab); np.save(base + '_drift.npy', d); A.jdump(rep, base + '_drift.json')
    return tab, d, rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=('locs', 'table')); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--ncyc', type=int, required=True)
    ap.add_argument('--sel', default='u45'); ap.add_argument('--drift', default='own'); ap.add_argument('--cmp', nargs='*', default=[]); ap.add_argument('--gt', default=None); ap.add_argument('--ref_drift', default=None, help='trace whose mean the own drift gets (default for a real nr40 / nr20 table: ref/drift_pub_total_simflux.npy)'); ap.add_argument('--mask_from', default=None, help='locs npz with identical rows: the selection is taken from it (common view of the variants)')
    a = ap.parse_args(); A.title('nr-r24')
    convert(a.kind, a.src, a.out, a.ncyc, a.sel, a.drift, a.cmp, a.gt, mask_from=a.mask_from, ref_drift=a.ref_drift)
