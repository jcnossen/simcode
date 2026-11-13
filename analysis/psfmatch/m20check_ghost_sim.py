"""m20check (2026-11-12): ghost detections next to bright emitters on the 20 nM-like simulated movie m20r (per-axis 20 nM PSF, GT known),
existing runs: m25 set (out/psfmatch/sims/m20r/run_m25), mixture set (run_new), old default (run_cur). CPU only.
Per run and row class (good / u045 = strict_u <= 0.45 / strict), pass 2: rows not matched to a GT window (1 px, evalproto.match_fast) that lie
1.5 - 8 px from a GT window of the same frame with >= PB window photons, per such bright GT window ("ghosts per bright window"; the mixture
run gives the chance level in this dense movie), and their mean offset from the bright emitter.
usage: python analysis/psfmatch/m20check_ghost_sim.py > out/psfmatch/m20check/ghost_sim_m20r.txt"""
import numpy as np
from scipy.spatial import cKDTree
from simcode2.evalproto import lib as E
from simcode2.util import proctitle
proctitle.set_title('m20check')
SD = 'out/psf3d_nets/sims/m20r'
G = dict(np.load(f'{SD}/sim_gt.npz', allow_pickle=True)); G['nf'] = int(G['nf'])
GW = E.gt_windows_fast(G, grid=3)
print(f'# GT windows {len(GW)}; photons pct 50/90/99 {np.percentile(GW[:, 4], [50, 90, 99]).round(0)}; >= 4000: {int((GW[:, 4] >= 4000).sum())}; >= 8000: {int((GW[:, 4] >= 8000).sum())}')
for nm, run in (('m25', 'out/psfmatch/sims/m20r/run_m25'), ('mixture', f'{SD}/run_new'), ('old default', f'{SD}/run_cur')):
    T = dict(np.load(f'{run}/locs_p2.npz')); u = T['strict_u']
    for cls, k in (('good', T['good'].astype(bool)), ('u045', np.isfinite(u) & (u <= 0.45)), ('strict', T['strict'].astype(bool))):
        fr, x, y = T['frame'][k].astype(np.int64), T['x'][k], T['y'][k]
        gi = E.match_fast(GW, fr, x, y, r=1.0); um = gi < 0
        line = f'{nm:11s} {cls:6s} rows {k.sum():7d} unmatched {um.sum():6d} ({100 * um.mean():.2f} %)'
        for PB in (3000, 4000):
            b = GW[:, 4] >= PB
            if b.sum() == 0: continue
            tr = cKDTree(np.c_[GW[b, 2], GW[b, 3], GW[b, 0] * 1e4])
            d, j = tr.query(np.c_[x[um], y[um], fr[um] * 1e4], distance_upper_bound=8.0)
            g = np.isfinite(d) & (d > 1.5)
            jj = np.minimum(j, b.sum() - 1); dx = x[um] - GW[b, 2][jj]; dy = y[um] - GW[b, 3][jj]
            line += f' | >= {PB}: near-bright unmatched {int(g.sum()):5d} = {g.sum() / max(b.sum(), 1):.3f} per bright window, {100 * g.sum() / k.sum():.2f} % of rows, offset {dx[g].mean():+.2f} {dy[g].mean():+.2f} px'
        print(line, flush=True)
