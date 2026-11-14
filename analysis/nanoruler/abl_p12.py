"""Nanoruler ablation, part E: pass 1 versus pass 2 on IDENTICAL events (variant A tables of abl_solve.py, each pass with its own calibration).

The pass tables differ in their rows (pass 2 has its own detector and its own sigma), so their max sigma <= 12 nm tables
are not the same events. Here both passes are compared on the same windows:
  sim   <movie nr_a|nr_c0>: every cycle-aligned row of each pass is matched to a GT window (same start, 1 px; ev_sigcal.gt_windows); for the GT
        windows found by BOTH passes: RMS error per axis of the final position and of the pattern-free prior position, gross fraction
        (> 50 nm), paired RMS difference pass 2 - pass 1 with a bootstrap over rulers; on the row sets
          all (both passes have a row), good (p >= 0.63 in both), m12_p1 (max sigma <= 12 nm in pass 1), m12_p2 (max sigma <= 12 nm in pass 2), m12_both,
          good_fully_on / good_partly_on (emitter on during all six stored frames or not);
        and the recall of each pass on the GT target windows, and truth-based z of the calibrated sigma.
  real  pass-1 and pass-2 rows matched one to one (same cycle, 40 nm, mutual nearest); on the matched events per-site spread, pooled
        spread, consecutive-cycle pair precision and their paired differences (nra_lib, fixed sites, bootstrap over rulers), same drift trace
        (DME v2 of pass 1, variant A) for both; row sets all / good / m12_p1 / m12_p2 / m12_both.
usage: python analysis/nanoruler/abl_p12.py sim nr_a | sim nr_c0 | real   [--nboot 300]  -> out/nanoruler/ablation/p12_<name>.json, tab_p12_<name>.md"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
L = A.L
SETS5 = ('all', 'good', 'm12_p1', 'm12_p2', 'm12_both')
ci = lambda v: [float(v[0]), float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))]
f3 = lambda c, d=2: ('%.{0}f [%.{0}f, %.{0}f]'.format(d)) % tuple(c)
fd = lambda c: '%+.3f [%+.3f, %+.3f]' % tuple(c)
def m12_of(T):
    """the paper's selection (simcode.strict.select: p >= 0.63, inside the crop, max(sx, sy) * 65 nm <= 12 nm) of every row"""
    from simcode import strict as SP
    return SP.select(T, A.SIGMA_MAX_NM, px_nm=A.PX)


def sim(movie, nboot):
    import ev_sigcal as ES
    k = 'sim' + movie[3:]; sd = A.SETS[k + '_p1']['sim']
    T = [A.load_table(f'{A.OUT}/runs/{k}_p{p}/A/locs.npz') for p in (1, 2)]
    GW = ES.gt_windows(sd); G = np.load(f'{sd}/sim_gt.npz', allow_pickle=True); SI = np.load(f'{sd}/sites.npz')
    ros = np.zeros(int(SI['site'].max()) + 1, int); ros[SI['site']] = SI['ruler']; gw_ruler = ros[G['site']][GW[:, 1].astype(int)]; nr = int(SI['ruler'].max()) + 1
    row = []                                  # per pass: table row index per GT window (-1 = none)
    for t in T:
        rows = np.nonzero((t['frame'] % 6 == 0) & np.isfinite(t['x']))[0]
        gi = ES.E.match_fast(GW, t['frame'][rows].astype(np.int64), t['x'][rows], t['y'][rows], r=1.0)
        r = np.full(len(GW), -1); ok = gi >= 0
        o = np.argsort(-t['p'][rows][ok]); r[gi[ok][o][::-1]] = rows[ok][o][::-1]      # highest p wins a doubly matched window
        row.append(r)
    tgt = GW[:, 5] > 0; both = (row[0] >= 0) & (row[1] >= 0)
    E_, EP, S_ = [], [], []
    for t, r in zip(T, row):
        rr = np.maximum(r, 0)
        e = np.c_[t['x'][rr] - GW[:, 2], t['y'][rr] - GW[:, 3]]; ep = np.c_[t['x_prior'][rr] - GW[:, 2], t['y_prior'][rr] - GW[:, 3]]
        e = (e - np.nanmedian(e[both], 0)) * A.PX; ep = (ep - np.nanmedian(ep[both], 0)) * A.PX
        E_.append(e); EP.append(ep); S_.append(np.c_[t['sx'][rr], t['sy'][rr]] * A.PX)
    good = [t['good'][np.maximum(r, 0)] & (r >= 0) for t, r in zip(T, row)]; m12 = [m12_of(t)[np.maximum(r, 0)] & (r >= 0) for t, r in zip(T, row)]
    full = GW[:, 6] >= 1; iso = np.ones(len(GW), bool)
    masks = dict(all=both, good=both & good[0] & good[1], m12_p1=both & m12[0], m12_p2=both & m12[1], m12_both=both & m12[0] & m12[1],
                 good_fully_on=both & good[0] & good[1] & full, good_partly_on=both & good[0] & good[1] & ~full)
    rng = np.random.default_rng(1); W = [np.ones(nr, int)] + [np.bincount(rng.integers(0, nr, nr), minlength=nr) for _ in range(nboot)]
    rms = lambda e: float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
    R = dict(movie=movie, n_gt=int(len(GW)), n_gt_target=int(tgt.sum()), recall_target={f'p{p + 1}': dict(any=float((row[p] >= 0)[tgt].mean()), good=float(good[p][tgt].mean()), m12=float(m12[p][tgt].mean())) for p in (0, 1)}, sets={})
    for nm, m in masks.items():
        ok = [np.hypot(*E_[p].T) < 50 for p in (0, 1)]; use = m & ok[0] & ok[1]; acc = {}
        for w in W:
            li = np.repeat(np.nonzero(use)[0], w[gw_ruler[use]])
            a, b = rms(E_[0][li]), rms(E_[1][li]); pa, pb = rms(EP[0][li][np.hypot(*EP[0][li].T) < 100]), rms(EP[1][li][np.hypot(*EP[1][li].T) < 100])
            for kk, v in (('rms_p1', a), ('rms_p2', b), ('d_rms', b - a), ('ratio', b / a), ('prior_p1', pa), ('prior_p2', pb), ('d_prior', pb - pa),
                          ('z_p1', 1.4826 * np.median(np.abs(E_[0][li] / S_[0][li])) if len(li) else np.nan), ('z_p2', 1.4826 * np.median(np.abs(E_[1][li] / S_[1][li])) if len(li) else np.nan)):
                acc.setdefault(kk, []).append(v)
        R['sets'][nm] = dict(n=int(use.sum()), n_before_gross=int(m.sum()), gross_p1_pct=float(100 * (m & ~ok[0]).sum() / max(m.sum(), 1)), gross_p2_pct=float(100 * (m & ~ok[1]).sum() / max(m.sum(), 1)),
                             photons_median=float(np.median(GW[use, 4])) if use.any() else np.nan, **{kk: ci(np.array(v)) for kk, v in acc.items()})
    A.jdump(R, f'{A.OUT}/p12_{movie}.json')
    r_ = R['recall_target']
    Ls = [f'# Pass 1 versus pass 2 on identical GT windows: simulated movie {movie}', '', f'{R["n_gt_target"]} GT target windows (cycle-aligned). Fraction with a row / a good row / a max sigma <= 12 nm row: '
          f'pass 1 {r_["p1"]["any"]:.3f} / {r_["p1"]["good"]:.3f} / {r_["p1"]["m12"]:.3f}, pass 2 {r_["p2"]["any"]:.3f} / {r_["p2"]["good"]:.3f} / {r_["p2"]["m12"]:.3f}. '
          'Variant A, each pass with its own calibration. RMS per axis, nm; [95 % CI, bootstrap over rulers]; errors > 50 nm are gross.', '',
          '| GT windows found by both passes, row set | n | GT photons (median) | gross % p1 / p2 | RMS pass 1 | RMS pass 2 | pass 2 - pass 1 | ratio | prior RMS pass 1 | prior RMS pass 2 | prior: pass 2 - pass 1 | z cal p1 | z cal p2 |', '|' + '---|' * 13]
    for nm in SETS5 + ('good_fully_on', 'good_partly_on'):
        s = R['sets'][nm]
        Ls.append('| ' + ' | '.join([nm, str(s['n']), '%.0f' % s['photons_median'], '%.2f / %.2f' % (s['gross_p1_pct'], s['gross_p2_pct']), f3(s['rms_p1']), f3(s['rms_p2']), fd(s['d_rms']), f3(s['ratio'], 3),
                                     f3(s['prior_p1']), f3(s['prior_p2']), fd(s['d_prior']), '%.2f' % s['z_p1'][0], '%.2f' % s['z_p2'][0]]) + ' |')
    md = '\n'.join(Ls) + '\n'; open(f'{A.OUT}/tab_p12_{movie}.md', 'w').write(md); print(md)


def real(nboot):
    import abl_match as MM
    T = [A.load_table(f'{A.OUT}/runs/p{p}/A/locs.npz') for p in (1, 2)]
    d = np.load(f'{A.OUT}/runs/p1/drift_A_m12.npy'); sites, rulers, _ = L.load_sites(A.SITES)
    tabs = [A.to_nra(t, A.selection(t, 'all'), d) for t in T]
    ia, ib = MM.match(tabs[0]['x_nm'], tabs[0]['y_nm'], tabs[0]['frame'], tabs[1]['x_nm'], tabs[1]['y_nm'], tabs[1]['frame'], 40.0)
    r1, r2 = tabs[0]['row'][ia], tabs[1]['row'][ib]
    g = [T[0]['good'][r1], T[1]['good'][r2]]; u = [m12_of(T[0])[r1], m12_of(T[1])[r2]]
    masks = dict(all=np.ones(len(ia), bool), good=g[0] & g[1], m12_p1=u[0], m12_p2=u[1], m12_both=u[0] & u[1])
    dd = np.c_[tabs[0]['x_nm'][ia] - tabs[1]['x_nm'][ib], tabs[0]['y_nm'][ia] - tabs[1]['y_nm'][ib]]
    R = dict(n_p1=len(tabs[0]['frame']), n_p2=len(tabs[1]['frame']), n_matched=len(ia), n_good=[int(T[0]['good'][tabs[0]['row']].sum()), int(T[1]['good'][tabs[1]['row']].sum())],
             n_m12=[int(m12_of(T[0])[tabs[0]['row']].sum()), int(m12_of(T[1])[tabs[1]['row']].sum())],
             offset_nm=np.median(dd, 0).tolist(), diff_rsd_nm=[L.rsd(dd[:, 0]), L.rsd(dd[:, 1])], sets={})
    for nm, m in masks.items():
        Ms = [L.eval_method({k: v[i][m] for k, v in t.items()}, sites, rulers, 30.0, 10) for t, i in zip(tabs, (ia, ib))]
        rsel = Ms[0]['ruler_ok'] & Ms[1]['ruler_ok']; edges = np.array([-np.inf, np.inf]); o = dict(n=int(m.sum()), n_rulers=int(rsel.sum()))
        for p, M in enumerate(Ms):
            est, c = L.bootstrap(M, rsel, edges, nboot, 1)
            o[f'p{p + 1}'] = {k: [est[k], *c[k]] for k in ('site_rsd_x_median', 'site_rsd_y_median', 'pooled_rsd_x', 'pooled_rsd_y', 'pair_rsd_x', 'pair_rsd_y', 'pair_zsd_x', 'pair_zsd_y', 'all_ratio_x', 'all_ratio_y',
                                                           'all_sig_med_x', 'all_sig_med_y', 'dist_mean', 'dist_sd', 'locs_per_site_median')}
        P = L.paired(Ms[1], Ms[0], rsel, nboot, 1)['xy']; o['paired_xy'] = [P['median_diff_nm']['est'], *P['median_diff_nm']['ci']]; o['paired_ratio'] = [P['median_ratio']['est'], *P['median_ratio']['ci']]
        R['sets'][nm] = o
    A.jdump(R, f'{A.OUT}/p12_real.json')
    Ls = ['# Pass 1 versus pass 2 on identical events: real movie', '', f'Cycle-aligned rows: pass 1 {R["n_p1"]} (good {R["n_good"][0]}, max sigma <= 12 nm {R["n_m12"][0]}), pass 2 {R["n_p2"]} (good {R["n_good"][1]}, max sigma <= 12 nm {R["n_m12"][1]}); '
          f'matched one to one (same cycle, 40 nm) {R["n_matched"]}; position difference pass 1 - pass 2: median {R["offset_nm"][0]:+.2f} / {R["offset_nm"][1]:+.2f} nm, robust sd {R["diff_rsd_nm"][0]:.2f} / {R["diff_rsd_nm"][1]:.2f} nm. '
          'Variant A, own calibration per pass, the same drift trace. nm; [95 % CI over rulers].', '',
          '| matched events, row set | n | rulers | rows / site | site spread x / y pass 1 | pass 2 | paired per-site xy: pass 2 - pass 1 | ratio | pair precision x / y pass 1 | pass 2 | pair z x / y pass 1 | pass 2 | spread / sigma pass 1 | pass 2 |', '|' + '---|' * 14]
    for nm in SETS5:
        s = R['sets'][nm]; a, b = s['p1'], s['p2']; q = lambda z, k: '%.2f / %.2f' % (z[k + '_x'][0], z[k + '_y'][0])
        Ls.append('| ' + ' | '.join([nm, str(s['n']), str(s['n_rulers']), '%.0f' % a['locs_per_site_median'][0], q(a, 'site_rsd') if False else '%.2f / %.2f' % (a['site_rsd_x_median'][0], a['site_rsd_y_median'][0]),
                                     '%.2f / %.2f' % (b['site_rsd_x_median'][0], b['site_rsd_y_median'][0]), fd(s['paired_xy']), f3(s['paired_ratio'], 3), q(a, 'pair_rsd'), q(b, 'pair_rsd'), q(a, 'pair_zsd'), q(b, 'pair_zsd'),
                                     q(a, 'all_ratio'), q(b, 'all_ratio')]) + ' |')
    md = '\n'.join(Ls) + '\n'; open(f'{A.OUT}/tab_p12_real.md', 'w').write(md); print(md)


if __name__ == '__main__':
    a = sys.argv[1:]; nb = int(a[a.index('--nboot') + 1]) if '--nboot' in a else 300; A.title()
    if a[0] == 'sim': sim(a[1], nb)
    else: real(nb)
