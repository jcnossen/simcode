"""autoc theta with the cKDTree vs spatial-hash isolation flags on an existing run dir (read-only; needs pass1_*.npz +
pattern_est.pt). usage: python analysis/speed/autoc_iso_check.py <run dir> <tag> [fast 0|1]"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, gpu_state, dump
from simcode2 import autoc
d, tag = sys.argv[1], sys.argv[2]; fast = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
REP = dict(run_dir=d, fast=fast, gpu_state=gpu_state())
for be in ('kdtree', 'hash'):
    autoc.ISO_BACKEND = be; t = time.time()
    E = autoc.run_dir(d, fast=fast, log=lambda *a: None)
    REP[be] = dict(theta=[float(v) for v in E['theta']], n_rows=E['n_rows'], s=round(time.time() - t, 1)); print(be, REP[be], flush=True)
REP['theta_max_abs_diff'] = float(np.max(np.abs(np.array(REP['kdtree']['theta']) - np.array(REP['hash']['theta']))))
print('theta max |diff|', REP['theta_max_abs_diff']); dump(REP, f'{OUT}/autoc_iso_{tag}.json')
