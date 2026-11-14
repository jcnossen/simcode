"""psf3d retrain (2026-11-11), item 2: selection-aware sigma rescale (the model's sig_u) against the winner's curse of a
sigma-ranked selection. sigma' = sigma * g(u), u = the per-10-px-tile sigma rank quantile among the rows with p >= 0.63 inside the
run's crop (simcode.strict.tile_rank, as simcode.strict.build_table computes it for the rescale), g = the robust zcal of the
matched rows per u bin, fitted on sim movies with GT. u is computed here from the table's sx, sy: the fit / test tables must be
built WITHOUT a sig_u rescale (then u equals the u of the table build); only the reported sx, sy change. zcal = 1.4826 median |e / sigma| per axis on rows with |e| < 50 nm (robust),
e = position - GT (window rows matched one-to-one within 1 px to GT event windows, evalproto/lib.match_fast; the median error of
the matched set, a global gauge offset, is removed per movie).
  fit:  python sigcal.py fit <out.json> <run_dir>:<sim_dir>:<tag est|est2>:<f_lo>:<f_hi> [...]
  test: python sigcal.py test <fit.json | none> <run_dir>:<sim_dir>:<tag>:<f_lo>:<f_hi> [...]   (zcal of the sigma <= 12 nm set in
        [0.9, 1.15]; it was the strict product's set, removed 2026-11-14)
  cmp:  python sigcal.py cmp <sim_dir> <tag> <run A> <run B>   new (A) vs reference (B) on one sim: good rows (p >= 0.63) matched to GT
        windows; recall of the GT target windows, unmatched good rows (FP), GT RMS on identical GT windows per GT-photon bin, gross > 50 nm"""
import json, os, sys
import numpy as np
from simcode.evalproto import lib as E
from simcode import strict as SP

PX = 108.33
UE = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
rob = lambda v: float(1.4826 * np.median(np.abs(v))) if len(v) else np.nan


def load(spec):
    run, sim, tag, flo, fhi = spec.split(':'); flo, fhi = int(flo), int(fhi)
    T = dict(np.load(f'{run}/locs_p{1 if tag == "est" else 2}.npz'))
    G = dict(np.load(f'{sim}/sim_gt.npz', allow_pickle=True))
    G['nf'] = int(G['nf'])
    GW = E.gt_windows_fast(G, grid=3)
    crop = SP.run_crop(run); n = len(T['x'])
    base = SP.select(T, None, crop=crop)                             # p >= 0.63 inside the crop: the rows the rescale ranks
    u = np.full(n, np.nan); ib = np.nonzero(base)[0]
    u[ib] = SP.tile_rank(T['x'][ib], T['y'][ib], 0.5 * (T['sx'][ib] + T['sy'][ib]).astype(np.float64), crop)
    sel12 = SP.select(T, SP.SIGMA_MAX_NM, crop=crop)
    k = base & (T['frame'] >= flo) & (T['frame'] < fhi)
    T = {c: v[k] for c, v in T.items() if np.ndim(v) and len(v) == n}; u = u[k]; sel12 = sel12[k]
    gi = E.match_fast(GW, T['frame'].astype(np.int64), T['x'], T['y'], r=1.0)
    m = gi >= 0
    e = np.full((len(gi), 2), np.nan); e[m] = np.c_[T['x'][m] - GW[gi[m], 2], T['y'][m] - GW[gi[m], 3]]
    e -= np.nanmedian(e[m], 0)
    s = np.c_[T['sx'], T['sy']]
    ok = m & (np.hypot(*e.T) * PX < 50)
    return dict(e=e[ok], s=s[ok], u=u[ok], sel12=sel12[ok], n_good=int(k.sum()), n_match=int(m.sum()),
                name=os.path.basename(run.rstrip('/')) + f'[{flo},{fhi})')


def g_of(fit, u):
    if fit is None: return np.ones_like(u)
    return np.interp(u, fit['u_mid'], fit['g'])


def zc(e, s):
    return rob((e / s).ravel())


def cmp(sim, tag, A, B):
    G = dict(np.load(f'{sim}/sim_gt.npz', allow_pickle=True)); G['nf'] = int(G['nf'])
    GW = E.gt_windows_fast(G, grid=3); gid = {}
    print(f'# {sim} {tag}: A = {A}, B = {B}; GT target windows {int(GW[:, 5].sum())}')
    for nm, run in (('A', A), ('B', B)):
        T = dict(np.load(f'{run}/locs_p{1 if tag == "est" else 2}.npz')); k = T['good'].astype(bool)
        gi = E.match_fast(GW, T['frame'][k].astype(np.int64), T['x'][k], T['y'][k], r=1.0)
        e = np.c_[T['x'][k] - GW[np.maximum(gi, 0), 2], T['y'][k] - GW[np.maximum(gi, 0), 3]]; e[gi < 0] = np.nan
        e -= np.nanmedian(e, 0); s_ = np.c_[T['sx'][k], T['sy'][k]]
        gid[nm] = (gi, e, s_)
        tg = GW[:, 5] > 0; hit = np.zeros(len(GW), bool); hit[gi[gi >= 0]] = True
        print(f'  {nm}: good {k.sum()}, recall(target windows) {hit[tg].mean():.4f}, unmatched good rows {int((gi < 0).sum())} '
              f'({(gi < 0).mean() * 100:.2f} %)')
    ga, ea, sa = gid['A']; gb, eb, sb = gid['B']
    com, ia, ib = np.intersect1d(ga[ga >= 0], gb[gb >= 0], return_indices=True)
    ia = np.nonzero(ga >= 0)[0][ia]; ib = np.nonzero(gb >= 0)[0][ib]
    P = GW[com, 4]
    print(f'  identical GT windows: {len(com)}')
    for lo, hi in ((200, 700), (700, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (0, 1e9)):
        m = (P >= lo) & (P < hi); a, b = ea[ia][m] * PX, eb[ib][m] * PX
        ok = (np.hypot(*a.T) < 50) & (np.hypot(*b.T) < 50)
        ra, rb = np.sqrt(np.mean(a[ok] ** 2)), np.sqrt(np.mean(b[ok] ** 2))
        print(f'  GT ph {lo:>5g}-{hi:<6g} n {m.sum():6d}  RMS nm A {ra:6.2f} B {rb:6.2f} ratio {ra / rb:.3f}  gross>50nm % A {100 * np.mean(np.hypot(*a.T) >= 50):5.2f} '
              f'B {100 * np.mean(np.hypot(*b.T) >= 50):5.2f}  zcal A {zc(ea[ia][m][ok], sa[ia][m][ok]):.3f} B {zc(eb[ib][m][ok], sb[ib][m][ok]):.3f}')


def main():
    if sys.argv[1] == 'cmp':
        return cmp(*sys.argv[2:6])
    mode, arg = sys.argv[1], sys.argv[2]; specs = sys.argv[3:]
    D = [load(s) for s in specs]
    if mode == 'fit':
        e = np.concatenate([d['e'] for d in D]); s = np.concatenate([d['s'] for d in D]); u = np.concatenate([d['u'] for d in D])
        mid, g, n = [], [], []
        for lo, hi in zip(UE[:-1], UE[1:]):
            m = (u >= lo) & (u < hi)
            mid.append(float(np.median(u[m]))); g.append(zc(e[m], s[m])); n.append(int(m.sum()))
        fit = dict(u_edges=UE, u_mid=mid, g=g, n=n, specs=specs,
                   note='2026-11-11 psf3d retrain item 2: sigma x g(u), u = per-tile sigma rank, g = robust zcal per u bin on the fit sims (analysis/psf3d_nets/sigcal.py)')
        json.dump(fit, open(arg, 'w'), indent=1)
        print('fit', arg, 'g(u):', ' '.join(f'{a:.3f}:{b:.3f}' for a, b in zip(mid, g)))
        return
    fit = None if arg == 'none' else json.load(open(arg))
    print(f'# test, rescale {arg}; zcal = robust (1.4826 median |e/sigma|, both axes, |e| < 50 nm)')
    print(f'{"set":34s}{"n_good":>8s}{"n_match":>8s}{"sig12 n":>9s}{"zc sig12":>10s}{"zc sig12*":>11s}{"zc all":>8s}{"zc all*":>8s}{"zc u<.02":>9s}{"*":>6s}  PASS(sig12* in [0.9,1.15])')
    for d in D:
        gs = g_of(fit, d['u'])[:, None]; st = d['sel12']; t1 = d['u'] < 0.02
        a, b = zc(d['e'][st], d['s'][st]), zc(d['e'][st], d['s'][st] * gs[st])
        print(f'{d["name"]:34s}{d["n_good"]:8d}{d["n_match"]:8d}{int(st.sum()):9d}{a:10.3f}{b:11.3f}{zc(d["e"], d["s"]):8.3f}'
              f'{zc(d["e"], d["s"] * gs):8.3f}{zc(d["e"][t1], d["s"][t1]):9.3f}{zc(d["e"][t1], d["s"][t1] * gs[t1]):6.3f}  {0.9 <= b <= 1.15}')


if __name__ == '__main__':
    main()
