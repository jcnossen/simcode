"""Published SIMCODE result (sfhd-ndi.hdf5) on the covpool FRC protocol (critic 40 #3), next to the sign-fixed SIMCODE (b).

Protocol (covpool.py): rows on the common window grid (window start % grid == 0; G3 at 0.5 / 2.5 nM), each set corrected
with its own DME v2 drift (window start + 3), 5-115 px crop after correction, FRC std = 1/7 threshold, alternating 300-frame
halves, 5-nm render (simcode2.evalproto.lib.frc_std). Also reported on all rows (stride-1 window starts), the protocol of the
earlier "published 19.3 / 25.0" numbers. The hdf5 is read only.

usage: python analysis/evalproto/frc_published.py --name real0.5 --published <results>/sfhd-ndi.hdf5 --pub_drift <DME npz>
           --sc <sc_b.npz> --sc_drift <DME npz> [--grid 3] [--out <base>]
"""
import argparse, json
import numpy as np
from simcode2.evalproto import lib as E


def load_drift(fn):
    d = np.load(fn, allow_pickle=True)
    if isinstance(d, np.lib.npyio.NpzFile):
        return d['R'].item()['drift_full'] if 'R' in d.files else d['drift']
    return d


def corrected(x, y, f, d):
    fi = np.clip(f + 3, 0, len(d) - 1)
    x = x - d[fi, 0]; y = y - d[fi, 1]
    k = np.isfinite(x) & (x >= 5) & (x < 115) & (y >= 5) & (y < 115)
    return x[k], y[k], f[k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--published', required=True); ap.add_argument('--pub_drift', required=True)
    ap.add_argument('--sc', required=True); ap.add_argument('--sc_drift', required=True)
    ap.add_argument('--grid', type=int, default=3); ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    import h5py
    with h5py.File(a.published, 'r') as h:
        L = h['locs'][:]
    sets = {'SIMCODE published (sfhd-ndi.hdf5)': (L['x'].astype(np.float64), L['y'].astype(np.float64), L['frame'].astype(np.int64),
                                                load_drift(a.pub_drift))}
    S = np.load(a.sc)
    sets['SIMCODE (b), sign-fixed'] = (S['x'].astype(np.float64), S['y'].astype(np.float64), S['frame'].astype(np.int64), load_drift(a.sc_drift))
    lines = [f'# {a.name}: FRC std (nm), own DME v2 drift, 5-115 px crop, alternating 300-frame halves, 1/7',
             f'| set | rows all (crop) | FRC all rows | rows G{a.grid} (crop) | FRC G{a.grid} |', '|---|---|---|---|---|']
    R = {}
    for nm, (x, y, f, d) in sets.items():
        xa, ya, fa = corrected(x, y, f, d)
        g = fa % a.grid == 0
        fr_all = float(E.frc_std(xa, ya, fa)); fr_g = float(E.frc_std(xa[g], ya[g], fa[g]))
        lines.append(f'| {nm} | {len(xa)} | {fr_all:.2f} | {int(g.sum())} | {fr_g:.2f} |')
        R[nm] = dict(n_all=len(xa), frc_all=fr_all, n_grid=int(g.sum()), frc_grid=fr_g)
        print(lines[-1], flush=True)
    print('\n'.join(lines))
    if a.out:
        open(a.out + '.txt', 'w').write('\n'.join(lines) + '\n'); json.dump(R, open(a.out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
