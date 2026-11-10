"""Critic 34 #2 for the 20 nM-like sim: gross errors (> 50 nm) and the d <= 1 px merged fraction, ours vs SIMCODE, on a sim movie.
Protocol of analysis/baselines/compare.py: W3 windows (start % 3 == 0), GT targets (inside the crop, central half, >= 1 on-frame),
1 px greedy one-to-one matching per window on the detection / network point (xd, yd), global offset (400-2000 label ph median)
removed before the error.
  gross: per label bin, % of matched targets with |e| > 50 nm (own matches) and on the PAIRED set (targets matched by every method)
  merged (targets >= 200 label ph whose nearest GT neighbour in the same window (>= 50 window ph) is at d <= 1 px, split 0-0.5 /
         0.5-1 px, plus 1-1.5 px): matched / merged (unmatched, but the neighbour is matched by that method) / undetected
(development n20/merged_gross.py)
usage: python analysis/simreal/merged_gross.py <sim_dir> <out.txt> name=kind:path ...   (kinds as compare.py: ours_est / ours_p2L ... :<run_dir>, sc_net / sc_mrg:<npz>)"""
import sys, json
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
import compare as C       # analysis/baselines/compare.py

PX = 108.33
sd, outp = sys.argv[1], sys.argv[2]
specs = [a.split('=', 1) for a in sys.argv[3:]]
G = dict(np.load(f'{sd}/sim_gt.npz')); NF = int(G['nf'])
GW = C.gt_windows(G, np.arange(0, NF - 5, 3))
tg = GW[:, 5] > 0; P = GW[:, 4]
# nearest visible GT neighbour per window
nnd = np.full(len(GW), 99.0); nnj = np.full(len(GW), -1)
o = np.argsort(GW[:, 0], kind='stable'); ws = GW[o, 0]
cut = np.flatnonzero(np.diff(ws)) + 1
for a, b in zip(np.r_[0, cut], np.r_[cut, len(ws)]):
    ii = o[a:b]
    if len(ii) < 2: continue
    X, Y = GW[ii, 2], GW[ii, 3]
    D = np.hypot(X[:, None] - X[None], Y[:, None] - Y[None]); np.fill_diagonal(D, 1e9)
    D[:, P[ii] < 50] = 1e9
    j = D.argmin(1); nnd[ii] = D[np.arange(len(ii)), j]; nnj[ii] = ii[j]
res = {}; ERR = {}
for name, ks in specs:
    kind, path = ks.split(':', 1)
    M = C.load_method(kind, path, NF)
    w3 = (M['w'] % 3 == 0) & (M['w'] <= NF - 6)
    Mw = {k: v[w3] for k, v in M.items()}
    gi = C.match(GW, Mw['w'], Mw['xd'], Mw['yd'])
    mt = gi >= 0
    rec = np.zeros(len(GW), bool); rec[gi[mt]] = True
    g = gi[mt]; e = np.c_[Mw['x'][mt] - GW[g, 2], Mw['y'][mt] - GW[g, 3]]
    sel = tg[g] & (P[g] >= 400) & (P[g] < 2000)
    e = e - np.median(e[sel], 0)
    err = np.full(len(GW), np.nan); err[g] = np.hypot(*e.T) * PX
    ERR[name] = err
    r = {}
    for lo, hi in ((200, 700), (700, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (200, 1e9)):
        m = tg & (P >= lo) & (P < hi) & rec
        r[f'{lo:.0f}-{hi:.0f}'] = dict(n=int(m.sum()), gross50=float(100 * np.mean(err[m] > 50)) if m.sum() else np.nan)
    mg = {}
    for lo, hi in ((0, 0.5), (0.5, 1.0), (0, 1.0), (1.0, 1.5)):
        t = tg & (P >= 200) & (nnd >= lo) & (nnd < hi)
        nb = nnj[t]
        matched = rec[t]; merged = ~rec[t] & rec[nb]; und = ~rec[t] & ~rec[nb]
        mg[f'{lo}-{hi}'] = dict(n=int(t.sum()), matched=float(matched.mean()), merged=float(merged.mean()), undet=float(und.mean()))
    res[name] = dict(gross=r, merged=mg)
names = [n for n, _ in specs]
paired = {}
for lo, hi in ((200, 700), (700, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (200, 1e9)):
    m = tg & (P >= lo) & (P < hi) & np.all([np.isfinite(ERR[n]) for n in names], 0)
    paired[f'{lo:.0f}-{hi:.0f}'] = dict(n=int(m.sum()), **{n: float(100 * np.mean(ERR[n][m] > 50)) if m.sum() else np.nan for n in names})
L = [f'# {sd}: gross (> 50 nm, % of matched) and d <= 1 px merged fractions; W3 windows, 1 px matching on detection points']
L.append('\n== gross > 50 nm, own matched targets: ' + ' | '.join(names))
for b in res[names[0]]['gross']:
    L.append(f'  {b:>10s} ' + ' | '.join(f'{n} {res[n]["gross"][b]["gross50"]:5.2f}% (n {res[n]["gross"][b]["n"]})' for n in names))
L.append('\n== gross > 50 nm, PAIRED (targets matched by all methods)')
for b, v in paired.items():
    L.append(f'  {b:>10s} n {v["n"]:6d} ' + ' | '.join(f'{n} {v[n]:5.2f}%' for n in names))
L.append('\n== targets >= 200 label ph by nearest-GT-neighbour distance d (px): matched / merged / undetected fractions')
for b in res[names[0]]['merged']:
    L.append(f'  d {b:>8s} n {res[names[0]]["merged"][b]["n"]:6d} ' + ' | '.join(
        f'{n} {res[n]["merged"][b]["matched"]:.3f}/{res[n]["merged"][b]["merged"]:.3f}/{res[n]["merged"][b]["undet"]:.3f}' for n in names))
open(outp, 'w').write('\n'.join(L) + '\n'); json.dump(dict(res=res, paired=paired), open(outp.replace('.txt', '.json'), 'w'), indent=1)
print('\n'.join(L))
