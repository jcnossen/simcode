"""Strict-product q check (critic 40 #1): does the default (auto-q) strict product of simcode2.strict cover as many occupied
20-nm bins as SIMCODE's output, on the covpool protocol?

Protocol (as covpool.py): rows on the common window grid (G3 at 0.5 / 2.5 nM, G6 at 20 nM), own-DME-v2 drift-corrected
(window start + 3), 5-115 px crop after the correction, occupied 20-nm bins. Ours: the shipped table (simcode2.strict.build_table
on the whole output dir, i.e. all stride-3 rows, drift-corrected coverage), then the strict rows restricted to the grid.
Reported for the shipped default (density rule), and per sigma_ref (earlier rule) or fixed q: q, strict rows on the grid, occupied bins, ratio vs SIMCODE; plus the exactly
coverage-matched values on the grid: the global sigma threshold sigma_cov (top-N by sigma reaching SIMCODE's bins) and the
per-tile quantile q_cov.

usage: python analysis/evalproto/strictq.py --name real0.5 --ours <out dir> --tag est --drift <DME npz|npy>
           --sc <sc_b.npz> --sc_drift <DME npz|npy> --grid 3 [--sigma_ref 7.0,7.6] [--q 0.12]   (the shipped default = density rule is always run first) [--px_nm 108.33]
"""
import argparse, json, os, sys
import numpy as np
from simcode2 import strict as SP


def load_drift(fn):
    d = np.load(fn, allow_pickle=True)
    if isinstance(d, np.lib.npyio.NpzFile):
        return d['R'].item()['drift_full'] if 'R' in d.files else d['drift']
    return d


def crop_mask(x, y):
    x0, y0, sz = SP.CROP
    return (x >= x0) & (x < x0 + sz) & (y >= y0) & (y < y0 + sz)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--ours', required=True); ap.add_argument('--tag', default='est')
    ap.add_argument('--drift', required=True); ap.add_argument('--sc', required=True); ap.add_argument('--sc_drift', required=True)
    ap.add_argument('--grid', type=int, default=3); ap.add_argument('--sigma_ref', default='')
    ap.add_argument('--q', default=''); ap.add_argument('--px_nm', type=float, default=SP.PX_NM_DEFAULT)
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    px = a.px_nm
    L = []
    def log(s):
        print(s, flush=True); L.append(s)
    # SIMCODE coverage (its final rows, own drift, grid, crop)
    S = np.load(a.sc); d = load_drift(a.sc_drift)
    f = S['frame'].astype(np.int64); fi = np.clip(f + 3, 0, len(d) - 1)
    xs = S['x'] - d[fi, 0]; ys = S['y'] - d[fi, 1]
    k = (f % a.grid == 0) & crop_mask(xs, ys)
    sc_bins = len(np.unique(SP.bin20(xs[k], ys[k], px)))
    log(f'# {a.name}: ours {a.ours} ({a.tag}), grid G{a.grid}; SIMCODE {a.sc}: {int(k.sum())} rows, {sc_bins} occupied 20-nm bins')
    # drift as a plain [F, 2] array for simcode2.strict
    dnpy = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'strictq_drift_{os.getpid()}.npy')
    np.save(dnpy, load_drift(a.drift))
    runs = [('default', None)] + [('sigma_ref', float(s)) for s in a.sigma_ref.split(',') if s] + [('fixed', float(q)) for q in a.q.split(',') if q]
    log('| mode | sigma_ref (nm) | q (all rows) | strict rows on grid | occupied bins | bins / SIMCODE |')
    log('|---|---|---|---|---|---|')
    T0 = None
    R = {}
    for mode, v in runs:
        T, info = SP.build_table(a.ours, a.tag, px, dnpy, v if mode == 'fixed' else None, v if mode == 'sigma_ref' else None)
        T0 = T
        g = (T['frame'] % a.grid == 0) & T['strict']
        b = len(np.unique(SP.bin20(T['x_dc'][g], T['y_dc'][g], px)))
        lab = f'density rho {info["density_per_um2_per_start"]:.3f}' if mode == 'default' else mode
        log(f'| {lab} | {v if mode == "sigma_ref" else "-"} | {info["q"]:.3f} | {int(g.sum())} | {b} | {b / sc_bins:.3f} |')
        R[f'{mode}_{v}'] = dict(info_mode=info['mode'], q=info['q'], n=int(g.sum()), bins=b, ratio=b / sc_bins)
    os.remove(dnpy)
    # exactly coverage-matched values on the grid
    T = T0
    m = T['good'] & (T['frame'] % a.grid == 0) & crop_mask(T['x_dc'], T['y_dc'])
    x, y = T['x_dc'][m], T['y_dc'][m]; sig = 0.5 * (T['sx'][m] + T['sy'][m]).astype(np.float64)
    keys = SP.bin20(x, y, px)
    o = np.argsort(sig, kind='stable')
    N = SP.n_for_cov(keys[o], sc_bins)
    sig_cov = float(sig[o[N - 1]] * px)
    ug = SP.tile_rank(x, y, sig); ou = np.argsort(ug, kind='stable')
    Nq = SP.n_for_cov(keys[ou], sc_bins); q_cov = float(ug[ou[Nq - 1]])
    log(f'coverage-matched on G{a.grid}: global sigma_cov = {sig_cov:.2f} nm (N {N}), per-tile q_cov = {q_cov:.3f} (N {Nq}); '
        f'ours good rows on grid {int(m.sum())}, {len(np.unique(keys))} bins')
    R.update(name=a.name, sc_bins=sc_bins, sigma_cov=sig_cov, q_cov=q_cov)
    if a.out:
        open(a.out + '.txt', 'w').write('\n'.join(L) + '\n'); json.dump(R, open(a.out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
