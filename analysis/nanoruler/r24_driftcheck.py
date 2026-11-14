"""R24: held-out check of the own DME v2 drift estimate (as abl_driftcheck.py, without a site list): DME on the rows of one half of the field
(x < 200 px / x >= 200 px) applied to the rows of the OTHER half; cluster sharpness (r24_tab.sharp) of each half with no correction, the
other half's trace (held out), its own trace (in sample), the whole-field trace and every reference trace; rms difference and correlation
of the two half traces. usage: python r24_driftcheck.py <locs.npz> <ncyc> <out.json> [--sel m12] [--cmp name=file.npy ...]   (GPU: DME)"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A, r24_tab as RT
a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
ncyc = int(a[1]); sel = g('--sel', 'm12'); cmp = a[a.index('--cmp') + 1:] if '--cmp' in a else []
A.title('nr-r24'); T = A.load_table(a[0]); m = A.selection(T, sel); cyc = (T['frame'][m] // 6).astype(np.int64); k = cyc < ncyc
x, y, sx, sy, cyc = T['x'][m][k], T['y'][m][k], T['sx'][m][k], T['sy'][m][k], cyc[k]
h = x < 200; D = {}
for nm, mk in (('left', h), ('right', ~h), ('all', np.ones(len(x), bool))): D[nm] = A.dme(x[mk], y[mk], cyc[mk], sx[mk], sy[mk], ncyc)[0]
refs = {n: np.load(f)[:ncyc] for n, f in (c.split('=', 1) for c in cmp)}; refs = {n: d - d.mean(0) for n, d in refs.items()}
sh = lambda mk, d: RT.sharp((x[mk] - d[cyc[mk], 0]) * A.PX, (y[mk] - d[cyc[mk], 1]) * A.PX)
R = dict(n=int(len(x)), n_left=int(h.sum()), sel=sel)
for nm, mk, own, oth in (('left', h, 'left', 'right'), ('right', ~h, 'right', 'left')):
    R[nm] = dict(none=sh(mk, np.zeros((ncyc, 2))), held_out=sh(mk, D[oth]), in_sample=sh(mk, D[own]), whole_field=sh(mk, D['all']), **{n: sh(mk, d) for n, d in refs.items()})
e = D['left'] - D['right']; e -= e.mean(0)
R['half_diff_rms_nm'] = (e.std(0) * A.PX).tolist(); R['half_corr'] = [float(np.corrcoef(D['left'][:, i], D['right'][:, i])[0, 1]) for i in range(2)]
R['half_diff_rms_nm_w50'] = ((A.smooth(e, 50)).std(0) * A.PX).tolist(); R['drift_p2p_nm'] = (np.ptp(D['all'], 0) * A.PX).tolist()
json.dump(R, open(a[2], 'w'), indent=1); print(json.dumps(R, indent=1))
