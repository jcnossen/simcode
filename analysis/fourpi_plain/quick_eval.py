"""Quick metrics of a plain-4Pi table without ground truth (variants on a part of a movie): counts, chi2 / dof, the
fraction whose solved z is off the network's astigmatic prior by more than half a 4Pi period, and the empirical
precision from emitters on in consecutive frames (pairs within 60 nm), per photon bin.

  python analysis/fourpi_plain/quick_eval.py <locs_000.npz> [...]
"""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from compare_published import consecutive_pairs, robust_sd

for f in sys.argv[1:]:
    d = dict(np.load(f)); px = 128.0
    D = dict(x=d['x'] * px, y=d['y'] * px, z=d['z'] * 1000, t=d['frame'], N=d['N'])
    sel = (d['N'] >= 500) & (np.maximum(d['sx'], d['sy']) * px < 25)
    off = np.abs(d['z'] - d['zprior']) > 0.12
    S = {k: v[sel] for k, v in D.items()}
    i, j = consecutive_pairs(S)
    Nm = np.sqrt(S['N'][i] * S['N'][j])
    print(f'{f}: n {len(d["x"])}, N>=500 & sig<25: {sel.sum()}, chi2/dof median {np.median(d["chi2"] / d["dof"]):.2f}, '
          f'|z - zprior| > 120 nm: {off[sel].mean():.3f}, pairs {len(i)}')
    for b0, b1 in ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 1e9)):
        s = (Nm >= b0) & (Nm < b1)
        if s.sum() < 30: continue
        e = [robust_sd(S[k][i][s] - S[k][j][s]) / np.sqrt(2) for k in 'xyz']
        jump = np.mean(np.abs(S['z'][i][s] - S['z'][j][s]) > 120)
        print(f'   N {b0:>5}-{b1:<7.0f} pairs {s.sum():6d}  emp x {e[0]:5.1f}  y {e[1]:5.1f}  z {e[2]:5.1f} nm   z jumps > 120 nm {jump:.3f}'
              f'   rep x {np.median(d["sx"][sel][i][s]) * px:5.1f} y {np.median(d["sy"][sel][i][s]) * px:5.1f} z {np.median(d["sz"][sel][i][s]) * 1000:5.1f}')
