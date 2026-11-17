"""locs_p2_net vs locs_p2_solve of a final run on the SAME rows (2026-11-17): the two tables of pass 2 have the same rows in the same
order; rows = those that pass the paper's cut in BOTH tables (p >= 0.63, crop, disjoint 6-frame windows, max(sigma) < 0.15 px on
each table's own sigma). Fair FRC (alternating 300-frame halves) of each table's positions on those rows, each with its own DME v2
drift (the caches of real_tables_frc.py: estimated on that table's own selected rows); also each table's FRC on its own selection
and on the rows only it keeps -> is the gap precision (same rows) or selection (different rows)?
usage: python same_rows_frc.py <1809_2|2.5nM_2> <tables_frc json of real_tables_frc.py> <out json>"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import frc_fair as FF
from real_sigma import SETS


def table(run, name):
    from simcode.dataset import Dataset
    C = Dataset.from_run(run, table=name, info=False).columns()
    return dict(x=C['x'].astype(np.float64), y=C['y'].astype(np.float64), f=C['frame'].astype(np.int64), sx=C['sx'].astype(np.float64),
                sy=C['sy'].astype(np.float64), p=C['p'].astype(np.float64))


if __name__ == '__main__':
    ds, tj, outj = sys.argv[1:4]
    run = SETS[ds]['v2_new']; cache = os.path.splitext(tj)[0]
    T = {k: table(run, f'locs_p2_{k}') for k in ('net', 'solve')}
    assert np.array_equal(T['net']['f'], T['solve']['f']) and np.array_equal(T['net']['p'], T['solve']['p'])
    keep = {k: FF.masks(t)[1] & (t['p'] >= 0.63) for k, t in T.items()}
    both = keep['net'] & keep['solve']
    res = dict(run=run, n_net=int(keep['net'].sum()), n_solve=int(keep['solve'].sum()), n_both=int(both.sum()),
               n_net_only=int((keep['net'] & ~keep['solve']).sum()), n_solve_only=int((keep['solve'] & ~keep['net']).sum()))
    for k, t in T.items():
        d = np.load(f'{cache}_new_locs_p2_{k}_drift.npy'); ti = np.clip(t['f'] + 3, 0, len(d) - 1)
        xd, yd = t['x'] - d[ti, 0], t['y'] - d[ti, 1]
        for lab, m in (('both', both), ('own', keep[k]), ('only', keep[k] & ~keep['solve' if k == 'net' else 'net'])):
            r = FF.frc_halves(xd[m], yd[m], t['f'][m])[0] if m.sum() > 1000 else float('nan')
            res[f'frc_{k}_{lab}_nm'] = float(r)
        res[f'median_sigma_{k}_both_nm'] = float(np.median(np.maximum(t['sx'], t['sy'])[both]) * FF.PX)
    dx = (T['net']['x'] - T['solve']['x'])[both]; dy = (T['net']['y'] - T['solve']['y'])[both]
    res['rms_net_minus_solve_nm'] = float(np.sqrt(np.mean(dx ** 2 + dy ** 2) / 2) * FF.PX)
    print(ds, json.dumps(res, indent=1), flush=True)
    json.dump(res, open(outj, 'w'), indent=1)
