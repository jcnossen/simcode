"""Fair FRC (frc_fair.evaluate: disjoint windows, crop, max(sigma) < 0.15 px, alternating 300-frame halves, own DME v2 drift) of
the four SIMCODE tables of a final run and of the previous paper run's table, same protocol as Figs 3 / 4 (2026-11-17): which
table gives which FRC / count / median sigma on real data. SIMCODE rows = p >= 0.63 at 16.25 nm (Dataset.select).
usage: CUDA_VISIBLE_DEVICES=<g> python real_tables_frc.py <1809_2|2.5nM_2> <out json>"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import frc_fair as FF
from real_sigma import SETS

NF = {'1809_2': 120000, '2.5nM_2': 180000}


def rows(run, table):
    from simcode.dataset import Dataset
    L = Dataset.from_run(run, table=table).select(16.25)
    F = L.numpy(); f8 = lambda v: v.astype(np.float64)
    return dict(x=f8(F['pos'][:, 0]), y=f8(F['pos'][:, 1]), f=F['frame'].astype(np.int64), sx=f8(F['crlb'][:, 0]), sy=f8(F['crlb'][:, 1]),
                photons=f8(F['photons']))


if __name__ == '__main__':
    ds, outj = sys.argv[1], sys.argv[2]
    P = SETS[ds]; res = {}
    cache = os.path.splitext(outj)[0]
    for lab, run, tab in (('new_locs_p2_net', P['v2_new'], 'locs_p2_net'), ('new_locs_p2_solve', P['v2_new'], 'locs_p2_solve'),
                          ('new_locs_p1_net', P['v2_new'], 'locs_p1_net'), ('new_locs_p1_solve', P['v2_new'], 'locs_p1_solve'),
                          ('prev_locs_p2', P['v2_old'], 'locs_p2')):
        T = rows(run, tab)
        R = FF.evaluate(T, NF[ds], drift_cache=f'{cache}_{lab}_drift.npy', secondary=False)
        _, m = FF.masks(T)
        r = dict(table=f'{run}/{tab}', frc_nm=R['frc_nm'], n_frc=R['n_frc'], median_sigma_nm=float(np.median(np.maximum(T['sx'], T['sy'])[m]) * FF.PX))
        res[lab] = r; print(ds, lab, json.dumps(r), flush=True)
    json.dump(res, open(outj, 'w'), indent=1)
