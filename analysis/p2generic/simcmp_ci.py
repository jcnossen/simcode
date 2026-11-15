"""Generic pass 2 (2026-11-13): run A vs run B on one simulated movie with GT, IDENTICAL GT windows (= analysis/psf3d_nets/sigcal.py
cmp: good rows p >= 0.63 matched one-to-one within 1 px to GT event windows, median error removed per run, errors > 50 nm gross and
excluded from both) + 95 % CI of RMS_A - RMS_B and of the ratio by bootstrap over GT events (400 resamples); recall of the GT target
windows by good rows (A - B with CI); also the 'fully on' windows (all 6 frames on) separately.
usage: python simcmp_ci.py <sim_dir> <est|est2> <run A> <run B> [px_nm=108.33]"""
import sys
import numpy as np
from simcode.evaluation import metrics as E
from simcode.util.proctitle import set_title
set_title('p2gen-simcmp')
sim, tag, A, B = sys.argv[1:5]; PX = float(sys.argv[5]) if len(sys.argv) > 5 else 108.33
G = dict(np.load(f'{sim}/sim_gt.npz', allow_pickle=True)); G['nf'] = int(G['nf'])
GW = E.gt_windows_fast(G, grid=3); tg = GW[:, 5] > 0
R = {}
for nm, run in (('A', A), ('B', B)):
    T = dict(np.load(f'{run}/locs_p{1 if tag == "est" else 2}.npz')); k = T['good'].astype(bool)
    gi = E.match_fast(GW, T['frame'][k].astype(np.int64), T['x'][k], T['y'][k], r=1.0)
    e = np.c_[T['x'][k] - GW[np.maximum(gi, 0), 2], T['y'][k] - GW[np.maximum(gi, 0), 3]]; e[gi < 0] = np.nan
    e -= np.nanmedian(e, 0)
    hit = np.zeros(len(GW), bool); hit[gi[gi >= 0]] = True
    R[nm] = (gi, e, hit, int((gi < 0).sum()), int(k.sum()))
print(f'# {sim} {tag}: A = {A}, B = {B}; GT target windows {int(tg.sum())}')
ev = GW[:, 1].astype(np.int64); rng = np.random.default_rng(0)
ue, inv = np.unique(ev, return_inverse=True); W = [np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)) for _ in range(400)]
for nm in 'AB':
    gi, e, hit, um, ng = R[nm]
    print(f'  {nm}: good rows {ng}, recall(target windows) {hit[tg].mean():.4f}, unmatched good rows {um} ({100 * um / ng:.2f} %)')
ha, hb = R['A'][2][tg], R['B'][2][tg]; wv = [w[inv[tg]] for w in W]
dr = [(w * ha).sum() / w.sum() - (w * hb).sum() / w.sum() for w in wv]
print(f'  recall A - B {ha.mean() - hb.mean():+.4f} [{np.percentile(dr, 2.5):+.4f}, {np.percentile(dr, 97.5):+.4f}]')
ga, ea = R['A'][0], R['A'][1]; gb, eb = R['B'][0], R['B'][1]
com, ia, ib = np.intersect1d(ga[ga >= 0], gb[gb >= 0], return_indices=True)
ia = np.nonzero(ga >= 0)[0][ia]; ib = np.nonzero(gb >= 0)[0][ib]
P = GW[com, 4]; a_all, b_all = ea[ia] * PX, eb[ib] * PX; evc = inv[com]
on_all = None
if 'on' in G or True:
    # fully on: GT event covers all 6 frames of the window (t0 <= w, t1 >= w + 6)
    t0 = np.asarray(G.get('t0', G.get('ton'))) if ('t0' in G or 'ton' in G) else None
    t1 = np.asarray(G.get('t1', G.get('toff'))) if ('t1' in G or 'toff' in G) else None
    if t0 is not None and t1 is not None:
        w = GW[com, 0]; e_ = GW[com, 1].astype(np.int64); on_all = (t0[e_] <= w) & (t1[e_] >= w + 6)
print(f'  identical GT windows: {len(com)}')
sets = [(f'GT ph {lo:g}-{hi:g}', (P >= lo) & (P < hi)) for lo, hi in ((200, 700), (700, 2000), (2000, 4000), (200, 2000), (0, 1e9))]
if on_all is not None: sets += [('fully on, 200-2000', on_all & (P >= 200) & (P < 2000)), ('switching, 200-2000', ~on_all & (P >= 200) & (P < 2000))]
for name, m in sets:
    a, b = a_all[m], b_all[m]; ok = (np.hypot(*a.T) < 50) & (np.hypot(*b.T) < 50); a, b, ec = a[ok], b[ok], evc[m][ok]
    rms = lambda x, w=None: np.sqrt(np.average((x ** 2).sum(1) / 2, weights=w))
    ra, rb = rms(a), rms(b); bw = [w[ec] for w in W]
    d = [rms(a, x) - rms(b, x) for x in bw]; q = [rms(a, x) / rms(b, x) for x in bw]
    print(f'  {name:22s} n {ok.sum():6d}  RMS nm A {ra:6.2f} B {rb:6.2f}  A - B {ra - rb:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]'
          f'  ratio {ra / rb:.4f} [{np.percentile(q, 2.5):.4f}, {np.percentile(q, 97.5):.4f}]  gross % A {100 * np.mean(np.hypot(*a_all[m].T) >= 50):.2f} B {100 * np.mean(np.hypot(*b_all[m].T) >= 50):.2f}')
