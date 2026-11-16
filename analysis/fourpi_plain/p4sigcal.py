"""Sigma calibration of a plain-4Pi run on the real data: the same emitter on in consecutive frames (pairs within 60 nm,
compare_published.consecutive_pairs) gives the empirical precision robust sd(difference) / sqrt 2 per photon bin; the
factor emp / reported (median of the pair's reported sigma) per bin, interpolated in log N, calibrates the reported
sigma of every localization (sx_cal = f_x(N) sx, ...).

  python analysis/fourpi_plain/p4sigcal.py <run dir> [--out <dir>] [--files 0 1 2 3]
  -> <out>/sigcal.json (bins, factors) and <out>/locs_<file>.npz = the run's table + sx_cal, sy_cal, sz_cal (out default:
     <run dir>/sigcal)
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from compare_published import load_ours, consecutive_pairs, robust_sd, sel

EDGES = [500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 30000]


def main():
    p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--out', default=''); p.add_argument('--files', type=int, nargs='+', default=[0, 1, 2, 3])
    a = p.parse_args()
    out = a.out or os.path.join(a.run, 'sigcal'); os.makedirs(out, exist_ok=True)
    O = load_ours(a.run, a.files)
    keep = (np.maximum(O['sx'], O['sy']) < 25) & (O['N'] >= 500) & ~(np.abs(O['dzref']) > 120)
    Os = sel(O, keep)
    i, j = consecutive_pairs(Os)
    Nm = np.sqrt(Os['N'][i] * Os['N'][j])
    rows = []
    for b0, b1 in zip(EDGES[:-1], EDGES[1:]):
        s = (Nm >= b0) & (Nm < b1)
        if s.sum() < 50: continue
        r = dict(N=[b0, b1], N_med=float(np.median(Nm[s])), pairs=int(s.sum()))
        for k in 'xyz':
            emp = robust_sd(Os[k][i][s] - Os[k][j][s]) / np.sqrt(2)
            rep = float(np.median(np.sqrt(0.5 * (Os['s' + k][i][s] ** 2 + Os['s' + k][j][s] ** 2))))
            r[k] = dict(emp=float(emp), rep=rep, factor=float(emp / rep))
        rows.append(r)
    lN = np.log([r['N_med'] for r in rows])
    fac = {k: np.array([r[k]['factor'] for r in rows]) for k in 'xyz'}
    json.dump(dict(run=os.path.abspath(a.run), bins=rows, selection='max(sx, sy) < 25 nm, N >= 500, |z - zref| <= 120 nm',
                   note='factor = empirical (consecutive-frame pairs) / reported sigma; f(N) = interpolation in log N, constant outside'),
              open(os.path.join(out, 'sigcal.json'), 'w'), indent=1)
    for r in rows:
        print(f'N {r["N"]}: pairs {r["pairs"]}, emp x/y/z {r["x"]["emp"]:.1f}/{r["y"]["emp"]:.1f}/{r["z"]["emp"]:.1f} nm, rep {r["x"]["rep"]:.1f}/{r["y"]["rep"]:.1f}/{r["z"]["rep"]:.1f}, factor {r["x"]["factor"]:.2f}/{r["y"]["factor"]:.2f}/{r["z"]["factor"]:.2f}')
    for fi in a.files:
        f = os.path.join(a.run, 'process', f'locs_{fi:03d}.npz')
        if not os.path.exists(f): continue
        d = dict(np.load(f))
        ln = np.log(np.clip(d['N'], 1, None))
        for k in 'xyz':
            d[f's{k}_cal'] = d[f's{k}'] * np.interp(ln, lN, fac[k])
        np.savez(os.path.join(out, f'locs_{fi:03d}.npz'), **d)


if __name__ == '__main__':
    main()
