"""Drift of a simcode2 localization table with DME v2 (the `dme` package, github.com/jcnossen/drift-estimation, through
simcode2/util/drift.py; not part of this repository). Rows: good (p >= 0.63) and disjoint windows (frame % 6 == 0), time =
window start + 3, crlb = the calibrated sigma (px), rows with sigma >= 1 px dropped (as the evaluated runs). A SIMCODE npz
(run_simcode_real.py ndi output: frame, x, y, sx, sy) is also accepted: all its rows, time = frame + 3, crlb = its sigma.
usage: CUDA_VISIBLE_DEVICES=1 python analysis/baselines/dme_drift.py <locs_p1.npz | sc_b.npz> <drift.npy> [n_frames]
-> drift [F, 2] px (subtract at frame + 3; simcode2.run --drift / simcode2.strict --drift take it)"""
import os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode2.util import drift as DR

T = np.load(sys.argv[1]); out = sys.argv[2]
F = int(sys.argv[3]) if len(sys.argv) > 3 else int(T['frame'].max()) + 6
m = (T['good'] & T['disjoint']) if 'good' in T.files else np.ones(len(T['frame']), bool)
pos = np.c_[T['x'][m], T['y'][m]].astype(np.float32); fr = (T['frame'][m] + 3).astype(np.int64)
crlb = np.c_[T['sx'][m], T['sy'][m]].astype(np.float32)
ok = np.isfinite(pos).all(1) & np.isfinite(crlb).all(1) & (crlb.max(1) < 1.0) & (crlb.min(1) > 0) & (fr >= 0) & (fr < F)
pos, fr, crlb = pos[ok], fr[ok], crlb[ok]
t0 = time.time()
d, info = DR.dme_v2(pos, fr, crlb, F, return_info=True)
d = np.asarray(d)[:F, :2]
np.save(out, d)
print(f'DME v2 {info["version"]} on {len(pos)} rows, chosen {info["chosen"]}, {time.time() - t0:.0f} s; '
      f'drift ptp {np.ptp(d, 0) * 108.33} nm -> {out}')
