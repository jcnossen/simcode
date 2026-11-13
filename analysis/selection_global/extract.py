"""Base rows of the selection study, cached small: the pass-2 rows that have a per-tile rank (strict_u finite = p >= 0.63 and
inside the run's crop; the rows the u <= 0.45 rule ranks) of the three real movies -> out/selection_global/cache/<ds>.npz
with x, y, frame, sx, sy (calibrated, after the sig_u rescale, px), photons, strict_u.
usage: python analysis/selection_global/extract.py"""
import os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from simcode.util.proctitle import set_title

RUNS = {'1809_2': 'out/paper2/1809_2/v2_matched', '2.5nM_2': 'out/paper2/2.5nM_2/v2_matched', '20nM_2': 'out/paper2/20nM_2/v2_m25'}
OUT = f'{REPO}/out/selection_global/cache'


def main():
    set_title('sel-extract')
    os.makedirs(OUT, exist_ok=True)
    for ds, run in RUNS.items():
        fn = f'{OUT}/{ds}.npz'
        if os.path.exists(fn): continue
        t0 = time.time()
        z = np.load(f'{REPO}/{run}/locs_p2.npz')
        u = z['strict_u']; m = np.isfinite(u)
        D = {k: z[k][m] for k in ('x', 'y', 'frame', 'sx', 'sy', 'photons')}
        D['strict_u'] = u[m]; D['u_max'] = z['u_max'] if 'u_max' in z.files else np.array(0.45)
        np.savez(fn, run=np.array(run), n_rows=np.array(len(u)), **D)
        print(ds, int(m.sum()), 'of', len(u), f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
