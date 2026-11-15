"""Real-data statistics of the 3 paper movies for the DECODE training-sim ranges (decode_paper.py VARIANTS['sum6p']).
Source: SIMCODE v2 pass-1 chunk files (pass1_<f0>.npz: network per-frame intensities / bg in the in-situ vectorial-PSF
photon normalisation, i.e. the normalisation of the PSF the baselines are trained / fitted with), rows p >= 0.63, inside the
crop, disjoint windows (wstart % 6 == 0), every 4th chunk.  Per 6-frame sum: photons (sum of the 6 I_t), bg (sum of the 6
per-frame bg, photons / px), emitters per 40 x 40 px window (= DECODE's training frame), on-time in sums (from the linked
disjoint-window events, simcode.evaluation.metrics.link_events).
usage: python analysis/baselines/decode/real_stats_paper2.py [out.json]"""
import os, sys, glob, json
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from simcode.evaluation.metrics import link_events

RUNS = {'1809_2': f'{REPO}/out/paper/1809_2/v2/run_default', '2.5nM_2': f'{REPO}/out/full_2.5nM_m2_p2L_real',
        '20nM_2': f'{REPO}/out/full_20nM_m2_fast'}
Q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
out = {}
for nm, d in RUNS.items():
    fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4]))[::4]
    ph, bg, dens, life = [], [], [], []
    for f in fs:
        z = np.load(f)
        x0, y0, sz = [int(v) for v in z['crop']]
        X, Y = z['x'] + z['X0'], z['y'] + z['Y0']
        inc = (X >= x0) & (X < x0 + sz) & (Y >= y0) & (Y < y0 + sz)
        m = (z['p'] >= 0.63) & inc & (z['wstart'] % 6 == 0)
        ph.append(z['I'][m].sum(1)); bg.append(z['bg'][m].sum(1))
        w = z['wstart'][m]; nwin = len(np.unique(z['wstart'][z['wstart'] % 6 == 0]))
        dens.append(m.sum() / max(nwin, 1) / sz ** 2 * 1600)
        ev = link_events(X[m], Y[m], w, grid=6, r=0.75)
        _, c = np.unique(ev[ev >= 0], return_counts=True); life.append(c)
    ph, bg, life = np.concatenate(ph), np.concatenate(bg), np.concatenate(life)
    qs = lambda v: {str(q): float(np.quantile(v, q)) for q in Q}
    out[nm] = dict(chunks=len(fs), n=int(len(ph)), photons_sum6=qs(ph), log_photons_sd=float(np.std(np.log(ph[ph > 50]))),
                   bg_sum6_per_px=qs(bg), emitters_per_40x40_per_sum=dict(mean=float(np.mean(dens)), min=float(np.min(dens)),
                   max=float(np.max(dens))), on_sums_mean=float(life.mean()), on_sums_median=float(np.median(life)))
    print(nm, json.dumps(out[nm]), flush=True)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
