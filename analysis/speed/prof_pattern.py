"""cProfile of the pattern-estimation stage (= run_uncalibrated 'pattern', same row loading) on an existing run dir (read-only).
usage: python analysis/speed/prof_pattern.py <run dir with pass1_*.npz> <sc_cal json> <movie>"""
import sys, glob, json, time, cProfile, pstats, io
import numpy as np, torch
from simcode2 import config
from simcode2.pipeline.pattern_est import estimate_pattern
d = sys.argv[1]; sc_cal = config.resolve(sys.argv[2])
fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4])); chunks = [int(f.split('_')[-1][:-4]) for f in fs]
t0 = time.time()
cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc')
Ls = [np.load(f) for f in fs]
keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1])) for L, f0 in zip(Ls, chunks)])
R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}
po = json.load(open(sc_cal)).get('prior_off')
if po is not None: R['x'] = R['x'] - po[0]; R['y'] = R['y'] - po[1]
x0c, y0c, szc = 5, 5, 110
R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= x0c) & (R['x'] < x0c + szc) & (R['y'] >= y0c) & (R['y'] < y0c + szc)
print(f'load {time.time() - t0:.1f}s, {len(R["x"])} rows', flush=True)
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
from simcode2.pipeline.common import Movie
shape = Movie(sys.argv[3]).shape
P, info = estimate_pattern(R, shape, verbose=lambda *a: None)
pr.disable(); print(f'estimate_pattern {time.time() - t0:.1f}s', flush=True)
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(30); print(s.getvalue()[:6000])
torch.save(P, '/home/jc/simcode2/out/speed/pattern_prof_' + d.strip('/').replace('/', '_')[-40:] + '.pt')
