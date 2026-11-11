"""Pattern stage with serial (old) vs parallel (new) cKDTree queries on an existing run dir: wall times + exact equality.
usage: python analysis/speed/pattern_kd.py <run dir> <movie> [reps]"""
import sys, glob, json, time
import numpy as np, torch
from simcode2 import config
from simcode2.pipeline import pattern_est as PE
from simcode2.pipeline.common import Movie
d, mv = sys.argv[1], sys.argv[2]; reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4])); chunks = [int(f.split('_')[-1][:-4]) for f in fs]
cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
Ls = [np.load(f) for f in fs]
keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1])) for L, f0 in zip(Ls, chunks)])
R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
po = json.load(open(config.resolve('calib/calib_sc_hd_sc5nbb15_pm_bright_real.json')))['prior_off']
R['x'] = R['x'] - po[0]; R['y'] = R['y'] - po[1]
R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= 5) & (R['x'] < 115) & (R['y'] >= 5) & (R['y'] < 115)
shape = Movie(mv).shape
res = {}
for r in range(reps):
    for w in ((1, -1) if r % 2 == 0 else (-1, 1)):
        PE.NB_BACKEND = 'kdtree'; PE.KD_WORKERS = w; t = time.time()
        P, info = PE.estimate_pattern(R, shape, verbose=lambda *a: None)
        print(f'workers {w}: {time.time() - t:.1f}s  ({len(R["x"])} rows)', flush=True); res[w] = P
print('identical pattern:', all(torch.equal(torch.as_tensor(res[1][k]), torch.as_tensor(res[-1][k])) for k in res[1] if torch.is_tensor(res[1][k])))
