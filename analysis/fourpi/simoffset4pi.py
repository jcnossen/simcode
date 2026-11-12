"""Where does the lateral offset of the network output against GT in the simulated movies come from?
Per estimator, on the same GT-matched units (bright, 2000-8000 e): network prior (cx + dxy, pattern-free), pattern-solved
with the in-situ pattern ('iter'), pattern-solved with the true pattern ('oracle'), classical summed fit. Offsets
(median, nm) overall, by kinetics model (pattern rows vs free rows), by z tercile, by x / y position tercile, and by the
emitter's fractional pixel position (convention check).
Usage: python analysis/fourpi/simoffset4pi.py --tag d03
"""
import argparse, json, os, sys
import numpy as np
REPO = '/home/jc/simcode2'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simeval4pi import gt_units, match, LO, HI, PX


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', default='d03'); p.add_argument('--model', default='v2')
    a = p.parse_args()
    d = f'{REPO}/out/4pi/simmovie/{a.tag}'
    meta = json.load(open(f'{d}/meta.json'))
    U = gt_units(d, meta['T'])
    g = (U['x'] >= LO) & (U['x'] < HI) & (U['y'] >= LO) & (U['y'] < HI) & (U['N'] >= 2000) & (U['N'] < 8000)
    G = {k: v[g] for k, v in U.items()}
    F = np.load(f'{d}/front_SIM_000_{a.model}.npz')
    out = {}
    est = {}
    Li = dict(np.load(f'{d}/locs_SIM_000_{a.model}_iter.npz'))
    est['network prior (pattern-free)'] = (F['cx'] + F['dxy'][:, 0], F['cy'] + F['dxy'][:, 1], F['z'], F['w'], None)
    est['pattern-solved, in-situ pattern'] = (Li['x'], Li['y'], Li['z'], Li['w'], Li['on'])
    fg = f'{d}/locs_SIM_000_{a.model}_iter_gauge.npz'
    if os.path.exists(fg):
        Lg = dict(np.load(fg)); est['pattern-solved, in-situ + classical gauge'] = (Lg['x'], Lg['y'], Lg['z'], Lg['w'], Lg['on'])
    fo = f'{d}/locs_SIM_000_{a.model}_oracle.npz'
    if os.path.exists(fo):
        Lo = dict(np.load(fo)); est['pattern-solved, true pattern'] = (Lo['x'], Lo['y'], Lo['z'], Lo['w'], Lo['on'])
    fc = f'{d}/classical_{a.model}_q.npz'
    fc = fc if os.path.exists(fc) else f'{d}/classical_v2_q.npz'                   # same detector -> same classical rows
    if os.path.exists(fc):
        C = dict(np.load(fc))
        qok = (C['chi2px'] <= 4) & ~C['at_bound'] & C['conv']
        est['classical summed fit (filtered)'] = (np.where(qok, C['x'], np.nan), np.where(qok, C['y'], np.nan), C['z'], C['w'], None)
    zq = np.quantile(G['z'], [1 / 3, 2 / 3]); xq = np.quantile(G['x'], [1 / 3, 2 / 3]); yq = np.quantile(G['y'], [1 / 3, 2 / 3])
    for nm, (x, y, z, w, on) in est.items():
        ok = np.isfinite(x)
        idx = np.nonzero(ok)[0]
        mg, _ = match(G['w'], G['x'], G['y'], w[idx], x[idx], y[idx])
        mm = mg >= 0; j = idx[mg[mm]]
        ex = (x[j] - G['x'][mm]) * PX * 1000; ey = (y[j] - G['y'][mm]) * PX * 1000
        med = lambda s: (float(np.median(ex[s])), float(np.median(ey[s])), int(s.sum())) if s.sum() > 30 else None
        r = dict(all=med(np.ones(len(ex), bool)))
        if on is not None:
            r['pattern rows'] = med(on[j]); r['free rows (prior)'] = med(~on[j])
        zg = G['z'][mm]; xg = G['x'][mm]; yg = G['y'][mm]
        r['z terciles'] = [med(zg < zq[0]), med((zg >= zq[0]) & (zg < zq[1])), med(zg >= zq[1])]
        r['x terciles'] = [med(xg < xq[0]), med((xg >= xq[0]) & (xg < xq[1])), med(xg >= xq[1])]
        r['y terciles'] = [med(yg < yq[0]), med((yg >= yq[0]) & (yg < yq[1])), med(yg >= yq[1])]
        fx = xg - np.round(xg); fy = yg - np.round(yg)
        r['frac x <0 / >=0'] = [med(fx < 0), med(fx >= 0)]; r['frac y <0 / >=0'] = [med(fy < 0), med(fy >= 0)]
        out[nm] = r
        print(f'\n{nm}: offset x / y (nm, n) {r["all"]}')
        for k in ('pattern rows', 'free rows (prior)'):
            if k in r: print(f'   {k}: {r[k]}')
        for k in ('z terciles', 'x terciles', 'y terciles', 'frac x <0 / >=0', 'frac y <0 / >=0'):
            print(f'   {k}: ' + ' | '.join(f'{v[0]:+.1f} / {v[1]:+.1f}' if v else '-' for v in r[k]))
    json.dump(out, open(f'{d}/offset_{a.model}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
