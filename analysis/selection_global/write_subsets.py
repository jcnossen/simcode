"""Write the selection subset files (global precision rule, simcode.strict.select) of the three real movies from their
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
    T = {k: L[k] for k in ('x', 'y', 'frame', 'sx', 'sy', 'photons', 'p')}; T['px_nm'] = L['px_nm']
    fn = SP.write_subset(T, f'{REPO}/{run}/locs_p2', a.sigma_max, crop=SP.run_crop(f'{REPO}/{run}'))
    S = np.load(fn); C = np.load(f'{REPO}/out/selection_global/cache/{ds}.npz')
    m = np.maximum(C['sx'], C['sy']).astype(np.float64) * float(L['px_nm']) <= a.sigma_max
    assert np.array_equal(S['x'], C['x'][m]) and np.array_equal(S['frame'], C['frame'][m])
    ij = f'{REPO}/{run}/locs_p2_info.json'; info = json.load(open(ij))
    info.update(main_rule='max(sx, sy) * px_nm <= sigma_max on the rows with p >= 0.63 inside the crop', sigma_max=a.sigma_max,
                n_main=int(len(S['x'])))
    json.dump(info, open(ij, 'w'), indent=1)
    print(ds, fn, len(S['x']), 'rows', flush=True)
