"""nr80 evaluation, step-order flip: pass 1 / pass 2 on identical cycle-aligned GT windows, SEPARATELY for
  (a) fully-on windows (emitter on in all six stored frames) and (b) partially-on windows (a switch-on or switch-off inside
  the six frames: minimum stored on-fraction < 1, exact on-fractions in the stored order of each movie; P >= PMIN photons),
for several cases side by side (name=run_dir:sim_dir ...). Per case and class: number of GT windows, recall by good rows
(p >= 0.63) of each pass, and on the windows with a good row in BOTH passes: RMS per axis (nm; errors > 50 nm gross, excluded
from both), pass 2 - pass 1 with 95 % CI (bootstrap over GT events), gross %, prior RMS, fraction of the good rows with
kin = True (the solve's kinetic flag). With --cross A,B (two cases on the SAME simulation: identical GT windows row by row) the
same pass is compared between the two movies on windows good in both.
usage: python ev_flipclass.py [--cross rev,flip] name=run:sim [...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_sigcal as SG
from simcode.evaluation import metrics as E
PX = 65.0; PMIN = float(os.environ.get('EV_PMIN', 300))
rng = np.random.default_rng(0)
rms = lambda e: float(np.sqrt(np.mean(e ** 2)) * PX) if len(e) else float('nan')


def load(run, sim):
    GW = SG.gt_windows(sim); D = {}
    for p in (1, 2):
        T = dict(np.load(f'{run}/locs_p{p}.npz')); k = T['frame'] % 6 == 0
        T = {c: v[k] for c, v in T.items() if np.ndim(v) and len(v) == len(k)}
        g = T['good'].astype(bool); gi = -np.ones(len(g), np.int64)
        gi[g] = E.match_fast(GW, T['frame'][g].astype(np.int64), T['x'][g], T['y'][g], r=1.0)
        m = gi >= 0; row = -np.ones(len(GW), np.int64); row[gi[m]] = np.nonzero(m)[0]
        e = np.c_[T['x'] - GW[np.maximum(gi, 0), 2], T['y'] - GW[np.maximum(gi, 0), 3]]; ep = np.c_[T['x_prior'] - GW[np.maximum(gi, 0), 2], T['y_prior'] - GW[np.maximum(gi, 0), 3]]
        e -= np.median(e[m], 0); ep -= np.median(ep[m], 0)
        D[p] = dict(row=row, e=e, ep=ep, kin=T['kin'].astype(bool), unm=float(np.mean(gi[g] < 0)), n=int(g.sum()))
    cls = {'fully on': GW[:, 6] >= 1, 'partially on': (GW[:, 6] < 1) & (GW[:, 4] >= PMIN)}
    return GW, D, cls


def boot(ev, a, b):
    ue, inv = np.unique(ev, return_inverse=True); sa = np.bincount(inv, (a ** 2).sum(1)); sb = np.bincount(inv, (b ** 2).sum(1)); n = np.bincount(inv); d = []
    for _ in range(400):
        j = rng.integers(0, len(ue), len(ue)); d.append((np.sqrt(sb[j].sum() / (2 * n[j].sum())) - np.sqrt(sa[j].sum() / (2 * n[j].sum()))) * PX)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def pair(GW, i, ea, eb):
    ga, gb = np.hypot(*ea.T) * PX >= 50, np.hypot(*eb.T) * PX >= 50; ok = ~ga & ~gb
    lo, hi = boot(GW[i, 1].astype(np.int64)[ok], ea[ok], eb[ok]) if ok.sum() > 10 else (np.nan, np.nan)
    return rms(ea[ok]), rms(eb[ok]), lo, hi, 100 * ga.mean(), 100 * gb.mean(), ok


args = sys.argv[1:]; cross = None
if args[0] == '--cross': cross = args[1].split(','); args = args[2:]
C = {}
print(f'# cycle-aligned windows, px {PX} nm; partially on = min stored on-fraction < 1 and GT window photons >= {PMIN:g}; good = p >= 0.63')
print('| case | class | GT windows | good recall p1 | good recall p2 | n good in both | RMS p1 | RMS p2 | p2 - p1 [95 % CI] | ratio | gross % p1 / p2 | prior RMS p1 / p2 | kin = True % p1 / p2 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for a in args:
    name, rs = a.split('='); run, sim = rs.split(':'); GW, D, cls = load(run, sim); C[name] = (GW, D, cls)
    for cn, cm in cls.items():
        r1, r2 = D[1]['row'], D[2]['row']; i = np.nonzero(cm & (r1 >= 0) & (r2 >= 0))[0]
        a_, b_ = D[1]['e'][r1[i]], D[2]['e'][r2[i]]; ra, rb, lo, hi, ga, gb, ok = pair(GW, i, a_, b_)
        k1 = 100 * D[1]['kin'][r1[cm & (r1 >= 0)]].mean(); k2 = 100 * D[2]['kin'][r2[cm & (r2 >= 0)]].mean()
        pa, pb = D[1]['ep'][r1[i]][ok], D[2]['ep'][r2[i]][ok]
        print(f'| {name} | {cn} | {int(cm.sum())} | {(r1 >= 0)[cm].mean():.3f} | {(r2 >= 0)[cm].mean():.3f} | {len(i)} | {ra:.2f} | {rb:.2f} | {rb - ra:+.2f} [{lo:+.2f}, {hi:+.2f}] | {rb / ra:.3f} '
              f'| {ga:.2f} / {gb:.2f} | {rms(pa):.2f} / {rms(pb):.2f} | {k1:.1f} / {k2:.1f} |')
    print(f'| {name} | (rows) | | good rows p1 {D[1]["n"]}, unmatched {100 * D[1]["unm"]:.2f} % | p2 {D[2]["n"]}, unmatched {100 * D[2]["unm"]:.2f} % | | | | | | | | |')
if cross:
    A, B = cross; GWa, Da, cls = C[A]; GWb, Db, _ = C[B]
    assert len(GWa) == len(GWb) and np.array_equal(GWa[:, :2], GWb[:, :2]), 'the two cases do not share their GT windows'
    print(f'\n# same pass, {B} versus {A} movie, identical GT windows with a good row in both movies (classes as in {A})')
    print(f'| pass | class | n | RMS {A} | RMS {B} | {B} - {A} [95 % CI] | ratio | gross % {A} / {B} | good recall {A} / {B} |')
    print('|---|---|---|---|---|---|---|---|---|')
    for p in (1, 2):
        for cn, cm in cls.items():
            ra_, rb_ = Da[p]['row'], Db[p]['row']; i = np.nonzero(cm & (ra_ >= 0) & (rb_ >= 0))[0]
            ra, rb, lo, hi, ga, gb, ok = pair(GWa, i, Da[p]['e'][ra_[i]], Db[p]['e'][rb_[i]])
            print(f'| {p} | {cn} | {len(i)} | {ra:.2f} | {rb:.2f} | {rb - ra:+.2f} [{lo:+.2f}, {hi:+.2f}] | {rb / ra:.3f} | {ga:.2f} / {gb:.2f} | {(ra_ >= 0)[cm].mean():.3f} / {(rb_ >= 0)[cm].mean():.3f} |')
