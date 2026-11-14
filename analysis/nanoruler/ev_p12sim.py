"""Pass 2 versus pass 1 on IDENTICAL ground-truth windows of one pipeline run on a simulated movie (locs_p1.npz, locs_p2.npz
of the same run + sim_gt.npz). Matching as ev_sigcal.py / analysis/psf3d_nets/sigcal.py (rows one-to-one within 1 px to GT event
windows, evalproto.lib.match_fast; the median error of the matched set removed per pass). Row sets on the GT windows matched by
BOTH passes: 'all' (any stored row), 'good' (p >= 0.63 in both), 'strict' (strict in both), 'good, fully on' (on in all 6 frames).
Per set: RMS per axis (nm, errors > 50 nm are gross and excluded from both), pass 2 - pass 1 with a 95 % CI (bootstrap over GT
events), ratio, the pattern-free prior RMS of both passes, robust zcal. Also recall of the GT target windows by good rows.
  nanoruler setup:  python ev_p12sim.py <run> <sim> nr      (65 nm px, cycle-aligned windows only, exact GT windows, drift added)
  SIMCODE setup:    python ev_p12sim.py <run> <sim> sc      (108.33 nm px, all window starts, evalproto gt_windows_fast)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode.evalproto import lib as E
run, sim, mode = sys.argv[1:4]
PX = 65.0 if mode == 'nr' else 108.33
G = dict(np.load(f'{sim}/sim_gt.npz', allow_pickle=True)); G['nf'] = int(G['nf'])
if mode == 'nr':
    import ev_sigcal as SG
    GW = SG.gt_windows(sim); full = GW[:, 6] >= 1
else:
    GW = E.gt_windows_fast(G, grid=3)
    w = GW[:, 0].astype(np.int64); ev_ = GW[:, 1].astype(np.int64)
    full = (G['t0'][ev_] <= w) & (G['t1'][ev_] >= w + 6)
    if 'drift' in G and np.ndim(G['drift']) == 2:
        d = np.stack([G['drift'][np.clip(w + j, 0, G['nf'] - 1)] for j in range(6)]).mean(0); GW[:, 2] += d[:, 0]; GW[:, 3] += d[:, 1]
tg = GW[:, 5] > 0
rms = lambda e: float(np.sqrt(np.mean(e ** 2)) * PX)
rob = lambda v: float(1.4826 * np.median(np.abs(v)))
D = {}
print(f'# {run} on {sim} ({mode}, px {PX} nm): GT windows {len(GW)}, targets {int(tg.sum())}')
for p in (1, 2):
    T = dict(np.load(f'{run}/locs_p{p}.npz'))
    k = np.ones(len(T['frame']), bool) if mode != 'nr' else (T['frame'] % 6 == 0)
    T = {c: v[k] for c, v in T.items() if np.ndim(v) and len(v) == len(k)}
    # good rows first (as sigcal), then the remaining rows on the still unmatched GT windows
    gi = -np.ones(len(T['x']), np.int64); g = T['good'].astype(bool)
    gi[g] = E.match_fast(GW, T['frame'][g].astype(np.int64), T['x'][g], T['y'][g], r=1.0)
    free = np.ones(len(GW), bool); free[gi[gi >= 0]] = False; idx = np.nonzero(free)[0]
    r2 = E.match_fast(GW[free], T['frame'][~g].astype(np.int64), T['x'][~g], T['y'][~g], r=1.0)
    gi[~g] = np.where(r2 >= 0, idx[np.maximum(r2, 0)], -1)
    m = gi >= 0
    row = -np.ones(len(GW), np.int64); row[gi[m]] = np.nonzero(m)[0]
    e = np.c_[T['x'] - GW[np.maximum(gi, 0), 2], T['y'] - GW[np.maximum(gi, 0), 3]]; ep = np.c_[T['x_prior'] - GW[np.maximum(gi, 0), 2], T['y_prior'] - GW[np.maximum(gi, 0), 3]]
    gm = m & g
    e -= np.median(e[gm], 0); ep -= np.median(ep[gm], 0)
    hit = np.zeros(len(GW), bool); hit[gi[gm]] = True; anyhit = row >= 0
    print(f'  pass {p}: rows {len(g)}, good {int(g.sum())}; GT targets with a row {anyhit[tg].mean():.4f}, with a good row {hit[tg].mean():.4f}; unmatched good rows {100 * np.mean(gi[g] < 0):.2f} %')
    D[p] = dict(row=row, e=e, ep=ep, good=g, strict=T['strict'].astype(bool), s=np.c_[T['sx'], T['sy']])
both = (D[1]['row'] >= 0) & (D[2]['row'] >= 0)
r1, r2 = D[1]['row'], D[2]['row']
sets = {'all': both, 'good': both & D[1]['good'][np.maximum(r1, 0)] & D[2]['good'][np.maximum(r2, 0)],
        'strict': both & D[1]['strict'][np.maximum(r1, 0)] & D[2]['strict'][np.maximum(r2, 0)]}
sets['good, fully on'] = sets['good'] & full
sets['good, 200-2000 ph'] = sets['good'] & (GW[:, 4] >= 200) & (GW[:, 4] < 2000)
rng = np.random.default_rng(0)
print('| row set (GT windows found by both passes) | n | GT ph median | gross % p1 / p2 | RMS p1 | RMS p2 | p2 - p1 [95 % CI] | ratio | prior RMS p1 | prior RMS p2 | zcal p1 / p2 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for name, m in sets.items():
    i = np.nonzero(m)[0]; a, b = D[1]['e'][r1[i]], D[2]['e'][r2[i]]; pa, pb = D[1]['ep'][r1[i]], D[2]['ep'][r2[i]]
    ga, gb = np.hypot(*a.T) * PX >= 50, np.hypot(*b.T) * PX >= 50; ok = ~ga & ~gb
    ev = GW[i, 1].astype(np.int64)[ok]; a, b, pa, pb = a[ok], b[ok], pa[ok], pb[ok]
    ue, inv = np.unique(ev, return_inverse=True)
    sa = np.bincount(inv, (a ** 2).sum(1)); sb = np.bincount(inv, (b ** 2).sum(1)); n = np.bincount(inv)
    dd = []
    for _ in range(400):
        j = rng.integers(0, len(ue), len(ue)); dd.append((np.sqrt(sb[j].sum() / (2 * n[j].sum())) - np.sqrt(sa[j].sum() / (2 * n[j].sum()))) * PX)
    okp = (np.hypot(*pa.T) * PX < 50) & (np.hypot(*pb.T) * PX < 50)
    za = rob((a / D[1]['s'][r1[i]][ok]).ravel()); zb = rob((b / D[2]['s'][r2[i]][ok]).ravel())
    print(f'| {name} | {len(i)} | {np.median(GW[i, 4]):.0f} | {100 * ga.mean():.2f} / {100 * gb.mean():.2f} | {rms(a):.2f} | {rms(b):.2f} | {rms(b) - rms(a):+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}] '
          f'| {rms(b) / rms(a):.3f} | {rms(pa[okp]):.2f} | {rms(pb[okp]):.2f} | {za:.2f} / {zb:.2f} |')
