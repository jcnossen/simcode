"""nr80 evaluation: selection-aware sigma rescale and GT comparison on the simulated nanoruler movies.
Variant of analysis/psf3d_nets/sigcal.py with the SAME statistics (sigma' = sigma * g(u), u = per-10-px-tile rank of the calibrated sigma among the
good rows inside the run's crop (simcode.strict.tile_rank, as the table build's sigma rescale), g = robust zcal
1.4826 median |e / sigma| of the matched good rows per u bin, same u bins, |e| < 50 nm, median error of the matched set removed
per movie, test criterion zcal of the paper's selection m12 (simcode.strict.select: max(sx, sy) <= 12 nm, 65 nm px) after
the rescale in [0.9, 1.15] (the strict product of earlier versions was removed 2026-11-14); cmp: recall of the GT target windows, unmatched good
rows, RMS on identical GT windows per GT-photon bin, gross > 50 nm). Differences:
  1. pixel size 65 nm (sigcal.py: PX = 108.33 hard-coded; it sets the 50 nm limits and the nm columns).
  2. cycle-aligned windows only: table rows with frame % 6 == 0 (frame = window start, simcode/strict.py) and GT windows on
     the grid 6. The movies are stored reordered within each 6-frame cycle, windows starting at 6 c + 3 mix two cycles.
  3. GT windows from analysis/nanoruler/sim_nr_lib.gt_windows_exact (exact stored on-fractions; evalproto.lib.gt_windows_fast
     ignores the within-cycle permutation and flags targets inside the hard-coded 5-115 px crop of the other setup). Target =
     photon time centroid in [1, 4) and inside the FOV minus MARGIN px (default 5 = the --crop 5,5,246 of the runs).
  4. the GT positions are drift-free, the table positions are not drift corrected: the simulated sample drift (sim_gt 'drift',
     mean over the 6 frames of the window) is added to the GT position before the error is taken (set EV_NODRIFT=1 to skip).
  6. a ground truth with the key stored_perm (step-flipped movie, ev_flipsteps.py gt) is read with that within-cycle order.
  5. cmp has the extra GT-photon bands 800-2500 and 1200-3500 and prints the per-bin zcal as sigcal.py; fit prints n per u bin.
  fit:  python ev_sigcal.py fit <out.json> <run_dir>:<sim_dir>:<tag est|est2>:<f_lo>:<f_hi> [...]
  test: python ev_sigcal.py test <fit.json | none> <run_dir>:<sim_dir>:<tag>:<f_lo>:<f_hi> [...]
  cmp:  python ev_sigcal.py cmp <sim_dir> <tag> <run A> <run B>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_nr_lib as NL
from simcode.evalproto import lib as E

PX = 65.0
MARGIN = float(os.environ.get('EV_MARGIN', 5))
UE = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
rob = lambda v: float(1.4826 * np.median(np.abs(v))) if len(v) else np.nan


def gt_windows(sim):
    """GT rows on the cycle-aligned grid: columns w, event, x, y (at the window, incl. sample drift), P, target, min onf, sum onf"""
    G = dict(np.load(f'{sim}/sim_gt.npz', allow_pickle=True)); G['nf'] = int(G['nf'])
    assert abs(float(G['pixel_nm']) - PX) < 1e-6, 'pixel size of the sim differs from 65 nm'
    perm0 = NL.PERM
    if 'stored_perm' in G: NL.PERM = np.asarray(G['stored_perm'])     # flipped movie (ev_flipsteps.py gt): stored position -> acquisition index
    try: GW = NL.gt_windows_exact(G, grid=6, margin=MARGIN)
    finally: NL.PERM = perm0
    assert (GW[:, 0] % 6 == 0).all()
    if 'drift' in G and not os.environ.get('EV_NODRIFT'):
        w = GW[:, 0].astype(np.int64)
        d = np.stack([G['drift'][np.clip(w + j, 0, G['nf'] - 1)] for j in range(6)]).mean(0)
        GW[:, 2] += d[:, 0]; GW[:, 3] += d[:, 1]
    return GW


def table(run, tag, flo=0, fhi=10 ** 9, need_u=False):
    """good, cycle-aligned rows of the run's table, with u (per-tile sigma rank, nan outside the crop) and m12 (the paper's selection)"""
    from simcode import strict as SP
    import abl_lib as A
    T = A.load_table(f'{run}/locs_p{1 if tag == "est" else 2}.npz'); crop = tuple(float(v) for v in T['crop'])
    inc = T['good'].astype(bool) & SP.in_crop(T['x'], T['y'], crop); ii = np.nonzero(inc)[0]
    T['u'] = np.full(len(inc), np.nan); T['u'][ii] = SP.tile_rank(T['x'][ii], T['y'][ii], 0.5 * (T['sx'] + T['sy']).astype(np.float64)[ii], crop)
    T['m12'] = SP.select(T, A.SIGMA_MAX_NM, px_nm=PX)
    k = T['good'].astype(bool) & (T['frame'] % 6 == 0) & (T['frame'] >= flo) & (T['frame'] < fhi)
    if need_u: k &= np.isfinite(T['u'])
    return {c: v[k] for c, v in T.items() if np.ndim(v) and len(v) == len(k)}


def load(spec):
    run, sim, tag, flo, fhi = spec.split(':'); flo, fhi = int(flo), int(fhi)
    GW = gt_windows(sim)
    T = table(run, tag, flo, fhi, need_u=True)
    gi = E.match_fast(GW, T['frame'].astype(np.int64), T['x'], T['y'], r=1.0)
    m = gi >= 0
    e = np.full((len(gi), 2), np.nan); e[m] = np.c_[T['x'][m] - GW[gi[m], 2], T['y'][m] - GW[gi[m], 3]]
    e -= np.nanmedian(e[m], 0)
    s = np.c_[T['sx'], T['sy']]
    ok = m & (np.hypot(*e.T) * PX < 50)
    return dict(e=e[ok], s=s[ok], u=T['u'][ok], m12=T['m12'][ok].astype(bool), n_good=int(len(m)), n_match=int(m.sum()),
                name=os.path.basename(run.rstrip('/')) + f'[{flo},{fhi})')


def g_of(fit, u):
    if fit is None: return np.ones_like(u)
    return np.interp(u, fit['u_mid'], fit['g'])


def zc(e, s):
    return rob((e / s).ravel())


def cmp(sim, tag, A, B):
    GW = gt_windows(sim); gid = {}
    print(f'# {sim} {tag}: A = {A}, B = {B}; cycle-aligned windows only; GT target windows {int(GW[:, 5].sum())}; px {PX} nm')
    for nm, run in (('A', A), ('B', B)):
        T = table(run, tag)
        gi = E.match_fast(GW, T['frame'].astype(np.int64), T['x'], T['y'], r=1.0)
        e = np.c_[T['x'] - GW[np.maximum(gi, 0), 2], T['y'] - GW[np.maximum(gi, 0), 3]]; e[gi < 0] = np.nan
        e -= np.nanmedian(e, 0); s_ = np.c_[T['sx'], T['sy']]
        gid[nm] = (gi, e, s_)
        tg = GW[:, 5] > 0; hit = np.zeros(len(GW), bool); hit[gi[gi >= 0]] = True
        print(f'  {nm}: good {len(gi)}, recall(target windows) {hit[tg].mean():.4f}, unmatched good rows {int((gi < 0).sum())} '
              f'({(gi < 0).mean() * 100:.2f} %), m12 {int(T["m12"].sum())}')
    ga, ea, sa = gid['A']; gb, eb, sb = gid['B']
    com, ia, ib = np.intersect1d(ga[ga >= 0], gb[gb >= 0], return_indices=True)
    ia = np.nonzero(ga >= 0)[0][ia]; ib = np.nonzero(gb >= 0)[0][ib]
    P = GW[com, 4]
    print(f'  identical GT windows: {len(com)}')
    for lo, hi in ((200, 700), (700, 2000), (2000, 4000), (4000, 1e9), (800, 2500), (1200, 3500), (200, 2000), (0, 1e9)):
        m = (P >= lo) & (P < hi); a, b = ea[ia][m] * PX, eb[ib][m] * PX
        if m.sum() < 5:
            print(f'  GT ph {lo:>5g}-{hi:<6g} n {m.sum():6d}'); continue
        ok = (np.hypot(*a.T) < 50) & (np.hypot(*b.T) < 50)
        ra, rb = np.sqrt(np.mean(a[ok] ** 2)), np.sqrt(np.mean(b[ok] ** 2))
        print(f'  GT ph {lo:>5g}-{hi:<6g} n {m.sum():6d}  RMS nm A {ra:6.2f} B {rb:6.2f} ratio {ra / rb:.3f}  gross>50nm % A {100 * np.mean(np.hypot(*a.T) >= 50):5.2f} '
              f'B {100 * np.mean(np.hypot(*b.T) >= 50):5.2f}  zcal A {zc(ea[ia][m][ok], sa[ia][m][ok]):.3f} B {zc(eb[ib][m][ok], sb[ib][m][ok]):.3f}')


def main():
    from simcode.util.proctitle import set_title; set_title('nr-eval')
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
        assert min(n) >= 20 and np.isfinite(g).all(), f'too few matched rows per u bin: {n}'
        fit = dict(u_edges=UE, u_mid=mid, g=g, n=n, specs=specs,
                   note='nr80 evaluation: sigma x g(u), g = robust zcal per u bin on the fit sims, cycle-aligned windows only, 65 nm px (analysis/nanoruler/ev_sigcal.py)')
        json.dump(fit, open(arg + '.tmp', 'w'), indent=1); os.replace(arg + '.tmp', arg)
        print('fit', arg, 'g(u):', ' '.join(f'{a:.3f}:{b:.3f}' for a, b in zip(mid, g)), '| n per bin', n)
        return
    fit = None if arg == 'none' else json.load(open(arg))
    print(f'# test, rescale {arg}; zcal = robust (1.4826 median |e/sigma|, both axes, |e| < 50 nm); cycle-aligned windows only, px {PX} nm')
    print(f'{"set":34s}{"n_good":>8s}{"n_match":>8s}{"m12 n":>9s}{"zc m12":>10s}{"zc m12*":>11s}{"zc all":>8s}{"zc all*":>8s}{"zc u<.02":>9s}{"*":>6s}  PASS(m12* in [0.9,1.15])')
    for d in D:
        gs = g_of(fit, d['u'])[:, None]; st = d['m12']; t1 = d['u'] < 0.02
        a, b = zc(d['e'][st], d['s'][st]), zc(d['e'][st], d['s'][st] * gs[st])
        print(f'{d["name"]:34s}{d["n_good"]:8d}{d["n_match"]:8d}{int(st.sum()):9d}{a:10.3f}{b:11.3f}{zc(d["e"], d["s"]):8.3f}'
              f'{zc(d["e"], d["s"] * gs):8.3f}{zc(d["e"][t1], d["s"][t1]):9.3f}{zc(d["e"][t1], d["s"][t1] * gs[t1]):6.3f}  {0.9 <= b <= 1.15}')


if __name__ == '__main__':
    main()
