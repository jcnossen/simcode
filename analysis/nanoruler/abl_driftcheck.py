"""Nanoruler ablation, part A: is the own-DME drift trace real or fitted to the few active sites? Held-out test: the field is split
into two halves (x < / >= the median x of the rows); DME v2 is estimated on the rows of one half and applied to the sites of the OTHER
half. Reported: per-site spread of each half without correction, with the other half's trace (held out), with its own half's trace
(in sample), with the full trace, and with the published Picasso trace; and the agreement of the two half traces.
usage: CUDA_VISIBLE_DEVICES=g python analysis/nanoruler/abl_driftcheck.py <nra table with row/px columns from abl_convert.py (before correction: uses <table>_drift.npy to undo)>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
L = A.L

fn = sys.argv[1]; A.title(); T = np.load(fn); d_full = np.load(os.path.splitext(fn)[0] + '_drift.npy'); ncyc = len(d_full)
cyc = T['frame'].astype(np.int64); x = T['x_nm'] / A.PX + d_full[cyc, 0]; y = T['y_nm'] / A.PX + d_full[cyc, 1]      # raw positions
sx = T['sx_nm'] / A.PX; sy = T['sy_nm'] / A.PX; xm = np.median(x); h = x < xm
S, R, _ = L.load_sites(A.SITES)
D = [A.dme(x[k], y[k], cyc[k], sx[k], sy[k], ncyc)[0] for k in (h, ~h)]
sp = lambda k, d: A.site_spread((x[k] - d[cyc[k], 0]) * A.PX, (y[k] - d[cyc[k], 1]) * A.PX, cyc[k], S, R)
Z = np.zeros_like(d_full); P = A.picasso_drift(ncyc) if ncyc == 5000 else Z
out = {}
for i, k in enumerate((h, ~h)):
    for nm, d in (('none', Z), ('held-out (other half)', D[1 - i]), ('in-sample (own half)', D[i]), ('full trace', d_full), ('Picasso', P)):
        s = sp(k, d); out[f'half{i}|{nm}'] = s
        print(f'half {i} ({int(k.sum())} rows) {nm:24s} site spread {s["site_x"]:.3f} / {s["site_y"]:.3f}  pooled {s["pooled_x"]:.3f} / {s["pooled_y"]:.3f} nm ({s["n_sites"]} sites)')
for w in (1, 12, 50, 250):
    a, b = A.smooth(D[0], w), A.smooth(D[1], w); e = (a - b) - (a - b).mean(0)
    print(f'half traces, {w}-cycle mean: difference rms {e.std(0)[0] * A.PX:.2f} / {e.std(0)[1] * A.PX:.2f} nm, corr {np.corrcoef(a[:, 0], b[:, 0])[0, 1]:.2f} / {np.corrcoef(a[:, 1], b[:, 1])[0, 1]:.2f}; '
          f'rms of the traces {a.std(0)[0] * A.PX:.2f} / {a.std(0)[1] * A.PX:.2f} and {b.std(0)[0] * A.PX:.2f} / {b.std(0)[1] * A.PX:.2f} nm')
    out[f'halves_w{w}'] = dict(diff_rms_nm=(e.std(0) * A.PX).tolist(), corr=[float(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(2)])
A.jdump(out, os.path.splitext(fn)[0] + '_driftcheck.json')
