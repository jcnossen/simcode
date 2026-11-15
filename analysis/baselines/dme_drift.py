"""Drift of a simcode localization table with DME v2 (the `dme` package, github.com/jcnossen/drift-estimation, through
simcode/util/drift.py; not part of this repository). Rows: good (p >= 0.63) and disjoint windows (frame % 6 == 0), time =
window start + 3, crlb = the calibrated sigma (px), rows with sigma >= 1 px dropped (as the evaluated runs). A SIMCODE npz
(run_simcode_real.py ndi output: frame, x, y, sx, sy) is also accepted: all its rows, time = frame + 3, crlb = its sigma.
usage: CUDA_VISIBLE_DEVICES=1 python analysis/baselines/dme_drift.py <locs_p1.npz | sc_b.npz> <drift.npy> [n_frames]
-> drift [F, 2] px (subtract at frame + 3; simcode.run --drift / simcode.table --drift take it)"""
import os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
from simcode.util import drift as DR

T = np.load(sys.argv[1]); out = sys.argv[2]
F = int(sys.argv[3]) if len(sys.argv) > 3 else int(T['frame'].max()) + 6
t0 = time.time()
d, info = DR.drift_of_table(T, F, return_info=True)        # the row selection is simcode/util/drift.drift_rows
np.save(out, d)
print(f'DME v2 {info["version"]} on {info["n"]} rows, chosen {info["chosen"]}, {time.time() - t0:.0f} s; '
      f'drift ptp {np.ptp(d, 0) * 108.33} nm -> {out}')
