"""sigma study Part A (2026-11-17): two simcode.run outputs of the same movie, A (old: the removed _real calibrations, real-data
sigma multiplier) vs B (new: the simulation-fitted calibrations), per pass: identity of the rows and positions, the sigma ratio
B / A per photon bin, the paper's selection (p >= 0.63 inside the crop, max(sx, sy) < 0.15 px = 16.25 nm; also 12 nm) and
its median sigma, z.
usage: python parta_compare.py <run A> <run B> <out json>"""
import json, sys
import numpy as np
from simcode.dataset import Dataset
from simcode import table as SP

PX = 108.33


def main():
    A, B, outf = sys.argv[1:4]
    res = dict(A=A, B=B)
    for p in (1, 2):
        TA = Dataset.from_run(A, pass_=p, info=False).columns(); TB = Dataset.from_run(B, pass_=p, info=False).columns()
        r = dict(n_rows=[len(TA['x']), len(TB['x'])])
        same = len(TA['x']) == len(TB['x'])
        r['same_rows'] = bool(same)
        if same:
            for k in ('frame', 'x', 'y', 'p', 'photons', 'z'):
                if k in TA: r[f'max_abs_diff_{k}'] = float(np.nanmax(np.abs(np.asarray(TA[k], np.float64) - np.asarray(TB[k], np.float64))))
            ratio = np.asarray(TB['sx'], np.float64) / np.asarray(TA['sx'], np.float64)
            ph = np.asarray(TA['photons']); bins = [0, 300, 500, 800, 1200, 2000, 3000, 5000, 1e9]
            r['sx_ratio_by_photons'] = {f'{lo:g}-{hi:g}': float(np.nanmedian(ratio[(ph >= lo) & (ph < hi)])) for lo, hi in zip(bins[:-1], bins[1:])
                                        if ((ph >= lo) & (ph < hi)).any()}
        for nm, T in (('A', TA), ('B', TB)):
            for cut in (0.15 * PX, 12.0):
                k = SP.select(T, cut)
                r[f'{nm}_sel_{cut:.2f}'] = dict(n=int(k.sum()), sigma_med_nm=float(np.median(SP.precision_nm(T)[k])),
                                                photons_med=float(np.median(T['photons'][k])),
                                                z_med=float(np.nanmedian(T['z'][k])) if 'z' in T else None)
        if same:
            kA, kB = SP.select(TA, 0.15 * PX), SP.select(TB, 0.15 * PX)
            r['sel_16.25_both'] = int((kA & kB).sum()); r['sel_16.25_only_A'] = int((kA & ~kB).sum()); r['sel_16.25_only_B'] = int((~kA & kB).sum())
        res[f'pass{p}'] = r
        print(f'pass {p}:', json.dumps(r, indent=1), flush=True)
    json.dump(res, open(outf, 'w'), indent=1)


if __name__ == '__main__':
    main()
