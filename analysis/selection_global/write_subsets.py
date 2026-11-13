"""Write the main-selection subset files (global precision rule, simcode.strict.select 'main') of the three real movies from their
existing tables: <run>/locs_p2_sig<X>.npz, and record sigma_max / n_main in <run>/locs_p2_info.json (the 0.6-5 GB tables
themselves are not rewritten; tables written from now on store sigma_max). Checks the subset row-for-row against the cached
base rows of extract.py.
usage: python analysis/selection_global/write_subsets.py [--sigma_max 12]"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from simcode import strict as SP
from simcode.util.proctitle import set_title
from extract import RUNS

ap = argparse.ArgumentParser(); ap.add_argument('--sigma_max', type=float, default=SP.SIGMA_MAX_NM); a = ap.parse_args()
set_title('sel-subsets')
for ds, run in RUNS.items():
    L = np.load(f'{REPO}/{run}/locs_p2.npz')
    T = {k: L[k] for k in ('x', 'y', 'frame', 'sx', 'sy', 'photons', 'strict_u')}; T['px_nm'] = L['px_nm']
    fn = SP.write_subset(T, f'{REPO}/{run}/locs_p2', 'main', sigma_max=a.sigma_max)
    S = np.load(fn); C = np.load(f'{REPO}/out/selection_global/cache/{ds}.npz')
    m = np.maximum(C['sx'], C['sy']) * float(L['px_nm']) <= a.sigma_max
    assert np.array_equal(S['x'], C['x'][m]) and np.array_equal(S['frame'], C['frame'][m])
    ij = f'{REPO}/{run}/locs_p2_info.json'; info = json.load(open(ij))
    info.update(main_rule='max(sx, sy) <= sigma_max (rows with a rank)', sigma_max=a.sigma_max, n_main=int(len(S['x'])),
                n_u=int(np.sum(np.nan_to_num(T['strict_u'], nan=2) <= info.get('u_max', 0.45))))
    json.dump(info, open(ij, 'w'), indent=1)
    print(ds, fn, len(S['x']), 'rows', flush=True)
