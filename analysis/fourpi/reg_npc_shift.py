"""NPC: bead polynomial + in-situ translation per channel (robust median of per-emitter residuals), from bright
isolated emitters in frames 0-7200 of file 10 (per-channel independent Gaussian MLE, see register.insitu_positions).
usage: python reg_npc_shift.py --work <4Pi work folder> --data <raw 4Pi movie folder>"""
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register import *
from dc import movie_files, file_index
ap = argparse.ArgumentParser()
ap.add_argument('--work', required=True, help='4Pi work folder')
ap.add_argument('--data', required=True, help='folder of the raw 4Pi movies')
A = ap.parse_args(); P4 = A.work
out = f'{P4}/smlm/reg'
cell, fi = 'NPC', 10
off = np.load(f'{P4}/smlm/camera/camera_{cell}.npz')['off']
pm = PolyMap.load(f'{out}/reg_beads_{cell}.json')
fn = [f for f in movie_files(cell, P4, A.data) if file_index(f) == fi][0]
S6 = load_s6(cell, fn, 0, 7200, off)
P = insitu_positions(S6, pm, min_sum=500.0, iso=7)
ok = np.isfinite(P).all((1, 2)) & (P[:, :, 5] < 0.15).all(1) & (P[:, :, 3] < 3.8).all(1) & (P[:, :, 4] < 3.8).all(1)
P = P[ok]; np.save(f'{out}/insitu_pos_NPC_10_b.npy', P)
rep = {'n': int(len(P))}
cf = {c: v.copy() for c, v in pm.cf.items()}; ci = {c: v.copy() for c, v in pm.ci.items()}
for c in (1, 2, 3):
    x0, y0 = pm.fwd(c, P[:, c, 0], P[:, c, 1]); r = P[:, 0, :2] - np.c_[x0, y0]
    med = np.median(r, 0); mad = 1.4826 * np.median(np.abs(r - med), 0)
    # bootstrap error of the median
    rng = np.random.default_rng(0); bs = np.array([np.median(r[rng.integers(0, len(r), len(r))], 0) for _ in range(200)])
    q = np.quantile(P[:, 0, 0], [0, .33, .67, 1]); fx = [np.median(r[(P[:, 0, 0] >= q[i]) & (P[:, 0, 0] <= q[i + 1])], 0).round(3).tolist() for i in range(3)]
    q = np.quantile(P[:, 0, 1], [0, .33, .67, 1]); fy = [np.median(r[(P[:, 0, 1] >= q[i]) & (P[:, 0, 1] <= q[i + 1])], 0).round(3).tolist() for i in range(3)]
    rep[c] = dict(median_shift=med.round(4).tolist(), median_err=bs.std(0).round(4).tolist(), robust_sd=mad.round(4).tolist(),
                  by_x_tercile=fx, by_y_tercile=fy)
    cf[c][0] += med                                      # constant term of fwd (feature 0 = 1)
    xs, ys = np.mgrid[0:360:6, 0:200:6].reshape(2, -1).astype(float)
    xf, yf = pm.fwd(c, xs, ys)
    ci[c] = fit_poly(np.c_[xf + med[0], yf + med[1]], np.c_[xs, ys], pm.order)
    print(c, json.dumps(rep[c]), flush=True)
PolyMap(cf, ci, pm.order).save(f'{out}/reg_insitu_{cell}.json', source='NPC bead polynomial + in-situ translation (file 10, frames 0-7200)', report=rep)
print('n', len(P))
