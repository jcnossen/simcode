"""Nanoruler ablation, part A: a localization table -> the nra input format, with DME v2 drift correction estimated on the table's own rows.

  python abl_convert.py locs <locs_p*.npz> <out.npz> [--sel m12|sel|good|all] [--drift own|none|<drift.npy>] [--ncyc 5000]
        pipeline table (px, frame = window start): cycle-aligned rows of the selection; DME v2 on these rows (time unit = cycle)
  python abl_convert.py table <nra table.npz> <out.npz> [--drift own|none|<drift.npy>]
        a table already in the nra format (nm, frame = cycle; e.g. the re-implemented baselines): same own-DME procedure
Writes <out>.npz (nra format + row = index into the input table), <out>_drift.npy ([ncyc, 2] px, mean removed; corrected = raw - drift[cycle])
and <out>_drift.json (DME settings chosen, comparison with the published Picasso trace, cluster spread without / own / Picasso)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
L = A.L


def convert(kind, src, out, sel='m12', drift='own', ncyc=5000, log=print, sig='cal'):
    base = os.path.splitext(out)[0]; os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if kind == 'locs':
        T = A.load_table(src); m = A.selection(T, sel)
        sg = ('sx_u', 'sy_u') if sig == 'u' and 'sx_u' in T else ('sx', 'sy')      # 'u': calibrated sigma x g(u) (abl_solve.py tables)
        tab = A.to_nra(T, m, sig=sg)
    else:
        D = np.load(src); tab = {k: np.asarray(D[k], np.float64) for k in D.files}; tab['frame'] = tab['frame'].astype(np.int64)
        tab['row'] = np.arange(len(tab['frame']))
    cyc = tab['frame']; x = tab['x_nm'] / A.PX; y = tab['y_nm'] / A.PX
    name = f'{os.path.basename(src)}[{sel if kind == "locs" else "table"}]'
    if drift == 'own':
        d, rep = A.drift_report(x, y, cyc, tab['sx_nm'] / A.PX, tab['sy_nm'] / A.PX, ncyc, name=name, log=log)
    elif drift == 'none':
        d, rep = np.zeros((ncyc, 2)), dict(note='no drift correction')
    else:
        d, rep = np.load(drift), dict(note=f'drift from {drift}')
    tab['x_nm'] = (x - d[cyc, 0]) * A.PX; tab['y_nm'] = (y - d[cyc, 1]) * A.PX
    L.save_table(out, **tab); np.save(base + '_drift.npy', d); A.jdump(dict(rep, src=src, kind=kind, sel=sel, drift=drift, n=len(cyc)), base + '_drift.json')
    log(f'wrote {out}: {len(cyc)} rows')
    return tab, d, rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=('locs', 'table')); ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--sel', default='m12'); ap.add_argument('--drift', default='own'); ap.add_argument('--ncyc', type=int, default=5000); ap.add_argument('--sig', default='cal', choices=('cal', 'u'))
    a = ap.parse_args(); A.title()
    convert(a.kind, a.src, a.out, a.sel, a.drift, a.ncyc, sig=a.sig)
