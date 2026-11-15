"""robust companion numbers for the position row of crlb_all_v2 (same detector-matched resolvable set as curves()):
per scene x photon bin: median |e| (nm, per axis) and % gross (|e| > 50 nm in x or y) for the networks and the LM reference.
usage: python lm_robust.py [txt] -> <out>/sc7/lm_robust.txt (or txt)"""
import os, sys
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import numpy as np
from simcode.benchmark import gate_report as GR
from simcode import config
OUTR = config.root('out')
PX = 108.33
D7 = os.path.join(OUTR, 'sc7'); G = os.path.join(OUTR, 'gate')
RUNS = [('sc5nb+PM', G, '_hd_sc5nbb15pm', 'sc'), ('sc6n+PM', G, '_hd_sc6nb15pm', 'sc'),
        ('LM11z pf', f'{D7}/gate_lm', '_lm11z', 'lm'), ('LM11z+PM', f'{D7}/gate_lm', '_lm11z', 'sc'), ('LM11 pf', f'{D7}/gate_lm', '_lm11', 'lm'), ('LM11+PM', f'{D7}/gate_lm', '_lm11', 'sc')]
BINS = [(150, 500), (500, 1300), (1300, 3200), (3200, 20000)]
L = []
for sc in ['iso', 'rs05q', 'rs05qm']:
    L.append(f'===== {sc}: median |e| per axis (nm) / % gross (|e| > 50 nm) / RMSD (nm) per bin (label photons)')
    for lab, d, tag, m in RUNS:
        try:
            R, nf = GR.load(d, sc, tag)
            if nf == 0: continue
        except Exception:
            continue
        sel = np.isfinite(R[f'sc_x_{m}']) & (R['sc_score'] >= 0.5); idx = np.nonzero(sel)[0]
        og = GR.match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
        ti = og[og >= 0]; oi = idx[og >= 0]
        res = np.hypot(R['gt_sf_m'][:, 0], R['gt_sf_m'][:, 1]) / np.sqrt(2) * PX < 25
        k = (R['gt_target'][ti] > 0) & res[ti]; ti, oi = ti[k], oi[k]
        e = np.stack([R[f'sc_x_{m}'][oi] - R['gt_x'][ti], R[f'sc_y_{m}'][oi] - R['gt_y'][ti]], 1) * PX
        ph = R['gt_phot'][ti]
        s = f'  {lab:10s}'
        for lo, hi in BINS:
            mm = (ph >= lo) & (ph < hi)
            if mm.sum() < 30: continue
            s += f' | {lo}-{hi}: {np.median(np.abs(e[mm])):5.1f} / {100 * np.mean((np.abs(e[mm]) > 50).any(1)):4.1f}% / {np.sqrt(np.mean(e[mm] ** 2)):5.1f}'
        L.append(s)
open(sys.argv[1] if len(sys.argv) > 1 else f'{D7}/lm_robust.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
