"""Nanoruler precision analysis CLI (see nra_lib.py for the definitions).

  python nra_run.py sites --ref a.npz [b.npz ...] --out sites.json [--picks picks.npz] [--ruler_picks picks3.npz]
  python nra_run.py run --sites sites.json --methods name=table.npz [name2=table2.npz ...] --out DIR [--tag t]
                        [--radius 30] [--min_n 10] [--bins 6] [--nboot 1000] [--seed 1] [--no_common] [--pairs A:B ...] [--pair_dt 1] [--frc_px 2.5]

Method table npz: frame, x_nm, y_nm, sx_nm, sy_nm, photons (+ crlb_x_nm, crlb_y_nm), common coordinate frame.
By default the metrics of every method are computed on the rulers that are valid (all sites with >= min_n assigned
localizations) in ALL methods of the run (--no_common: per-method valid rulers).
"""
import argparse, json, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nra_lib as L


def cmd_sites(a):
    tabs = [L.load_table(f) for f in a.ref]
    sh = [[0.0, 0.0]]
    if len(tabs) > 1:              # align the further reference tables to the first one before merging
        c0, n0, s0 = L.find_sites(tabs[0]['x_nm'], tabs[0]['y_nm'], a.radius, fov=tuple(a.fov)); g = n0 >= a.min_locs
        for t in tabs[1:]:
            s = L.global_shift(t['x_nm'], t['y_nm'], c0[g]); t['x_nm'] = t['x_nm'] - s[0]; t['y_nm'] = t['y_nm'] - s[1]; sh.append([float(s[0]), float(s[1])])
    ref = L.merge_tables(tabs)
    c, n, sd = L.find_sites(ref['x_nm'], ref['y_nm'], a.radius, blur=a.blur, weak_min=a.weak_min, fov=tuple(a.fov))
    sites, rulers, st = L.build_rulers(c, n, sd, a.min_locs, a.dist, a.tol, a.crowd, a.edge, tuple(a.fov), a.sd_factor, a.radius)
    meta = dict(ref=a.ref, ref_shifts_nm=sh, params={k: v for k, v in vars(a).items() if k not in ('fn', 'cmd')}, stats=st)
    d = np.array([np.hypot(sites[r['sites'][0]]['x'] - sites[r['sites'][1]]['x'], sites[r['sites'][0]]['y'] - sites[r['sites'][1]]['y']) for r in rulers])
    st['ref_dist_median'] = float(np.median(d)) if len(d) else None
    if a.picks:
        p = np.load(a.picks); meta['paper_site_picks'] = L.pick_overlap(p['x_nm'], p['y_nm'], sites, rulers, cand=c[n >= a.min_locs])
    if a.ruler_picks:
        p = np.load(a.ruler_picks); meta['paper_ruler_picks'] = L.ruler_pick_overlap(p['x_nm'], p['y_nm'], rulers)
    L.save_sites(a.out, sites, rulers, meta)
    np.savez(os.path.splitext(a.out)[0] + '_candidates.npz', x_nm=c[:, 0], y_nm=c[:, 1], n=n, sd=sd)
    print(json.dumps({k: meta[k] for k in meta if k != 'params'}, indent=1))


def cmd_run(a):
    t0 = time.time(); os.makedirs(a.out, exist_ok=True); tag = ('_' + a.tag) if a.tag else ''
    sites, rulers, meta = L.load_sites(a.sites)
    names = [m.split('=')[0] for m in a.methods]; files = [m.split('=', 1)[1] for m in a.methods]
    Ms = {}
    for nm, fn in zip(names, files):
        tab = L.load_table(fn); M = L.eval_method(tab, sites, rulers, a.radius, a.min_n, shift=not a.no_shift, pair_dt=a.pair_dt)
        M['T_x'] = tab['x_nm'] - M['shift'][0]; M['T_y'] = tab['y_nm'] - M['shift'][1]; Ms[nm] = M
        print('%s: %d locs, global shift %.2f %.2f nm, assigned %d, rulers valid %d / %d' % (
            nm, M['n_table'], M['shift'][0], M['shift'][1], len(M['L_site']), M['ruler_ok'].sum(), len(rulers)), flush=True)
    common = np.all([M['ruler_ok'] for M in Ms.values()], 0)
    res = {}
    for nm, M in Ms.items():
        rsel = M['ruler_ok'] if a.no_common else common
        edges = L.photon_edges(M, rsel, a.bins) if not a.photon_edges else np.r_[-np.inf, a.photon_edges, np.inf]
        est, ci = L.bootstrap(M, rsel, edges, a.nboot, a.seed)
        g = rsel[M['site_ruler'][M['L_site']]]
        res[nm] = dict(file=fn, est=est, ci=ci, edges=[float(e) for e in edges], nbins=len(edges) - 1, shift_nm=[float(M['shift'][0]), float(M['shift'][1])],
                       n_table=M['n_table'], n_rulers_valid_own=int(M['ruler_ok'].sum()),
                       frc_rulers_nm=L.frc_nm(M['L_x'][g], M['L_y'][g], a.seed, a.frc_px) if a.frc_px > 0 else float('nan'),
                       frc_all_nm=L.frc_nm(M['T_x'], M['T_y'], a.seed, a.frc_px) if a.frc_px > 0 else float('nan'), drift_trace=M['trace'])
        M['rsel'] = rsel
        print('%s done (%.0f s): dist %s, pooled x %s y %s, ratio x %s y %s, FRC %.1f / %.1f' % (
            nm, time.time() - t0, L.fmt(est, ci, 'dist_mean'), L.fmt(est, ci, 'pooled_rsd_x'), L.fmt(est, ci, 'pooled_rsd_y'),
            L.fmt(est, ci, 'all_ratio_x'), L.fmt(est, ci, 'all_ratio_y'), res[nm]['frc_rulers_nm'], res[nm]['frc_all_nm']), flush=True)
    pairs = [p.split(':') for p in a.pairs] if a.pairs else [[x, y] for i, x in enumerate(names) for y in names[i + 1:]]
    pr = {}
    for A, B in pairs:
        rs = Ms[A]['ruler_ok'] & Ms[B]['ruler_ok'] if a.no_common else common
        pr['%s - %s' % (A, B)] = L.paired(Ms[A], Ms[B], rs, a.nboot, a.seed)
    md = L.markdown(res)
    if pr:
        md += '\n### Paired per-site spread differences A - B (same sites; CI over rulers)\n\n| pair | axis | n sites | median diff (nm) | mean diff (nm) | median ratio A / B | fraction of sites with A < B |\n|---|---|---|---|---|---|---|\n'
        f = lambda d: '%.2f [%.2f, %.2f]' % (d['est'], d['ci'][0], d['ci'][1])
        for k, P in pr.items():
            for ax in ('x', 'y', 'along', 'across', 'xy'):
                md += '| %s | %s | %d | %s | %s | %s | %s |\n' % (k, ax, P[ax]['n_sites'], f(P[ax]['median_diff_nm']), f(P[ax]['mean_diff_nm']), f(P[ax]['median_ratio']), f(P[ax]['frac_sites_A_smaller']))
    hdr = '# Nanoruler analysis%s\n\nsites: %s (%d rulers: %d 2-site, %d 3-site); radius %.0f nm, min_n %d, %d bootstrap resamples (seed %d), %s.\n\n' % (
        ' ' + a.tag if a.tag else '', a.sites, len(rulers), sum(r['n_sites'] == 2 for r in rulers), sum(r['n_sites'] == 3 for r in rulers), a.radius, a.min_n,
        a.nboot, a.seed, 'per-method valid rulers' if a.no_common else 'rulers valid in all methods (%d)' % common.sum())
    open(a.out + '/metrics%s.md' % tag, 'w').write(hdr + md)
    json.dump(dict(args={k: v for k, v in vars(a).items() if k != 'fn'}, sites_meta=meta, n_common_rulers=int(common.sum()), methods=res, paired=pr),
              open(a.out + '/metrics%s.json' % tag, 'w'), indent=1)
    # per-site table
    ns = len(sites); cols = dict(site=np.arange(ns), ruler=np.array([s['ruler'] for s in sites]), x_ref=np.array([s['x'] for s in sites]),
                                 y_ref=np.array([s['y'] for s in sites]), n_sites_ruler=np.array([rulers[s['ruler']]['n_sites'] for s in sites]),
                                 common=common[[s['ruler'] for s in sites]].astype(int))
    for nm, M in Ms.items():
        cols[nm + '_n'] = M['n']; cols[nm + '_cx'] = M['centre'][:, 0]; cols[nm + '_cy'] = M['centre'][:, 1]
        for k, v in M['S'].items(): cols[nm + '_' + k] = v
    np.savez(a.out + '/per_site%s.npz' % tag, **cols)
    K = list(cols); np.savetxt(a.out + '/per_site%s.csv' % tag, np.c_[tuple(cols[k] for k in K)], delimiter=',', header=','.join(K), comments='', fmt='%.4f')
    rs0 = common if not a.no_common else np.any([M['ruler_ok'] for M in Ms.values()], 0)
    pick = L.fig_gallery(a.out + '/gallery%s.png' % tag, Ms, rulers, rs0, a.gallery_n, a.seed)
    L.fig_spread(a.out + '/spread_vs_photons%s.png' % tag, res)
    L.fig_dist(a.out + '/distance_hist%s.png' % tag, Ms, rs0 if not a.no_common else np.ones(len(rulers), bool), res)
    print(hdr + md); print('gallery rulers', pick, '\nwrote', a.out, '(%.0f s)' % (time.time() - t0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('sites'); s.add_argument('--ref', nargs='+', required=True); s.add_argument('--out', required=True)
    s.add_argument('--picks', default=''); s.add_argument('--ruler_picks', default='')
    s.add_argument('--radius', type=float, default=30.0); s.add_argument('--blur', type=float, default=8.0); s.add_argument('--weak_min', type=int, default=5)
    s.add_argument('--min_locs', type=int, default=30); s.add_argument('--dist', type=float, default=80.0); s.add_argument('--tol', type=float, default=15.0)
    s.add_argument('--crowd', type=float, default=250.0); s.add_argument('--edge', type=float, default=500.0)
    s.add_argument('--fov', type=float, nargs=2, default=[0.0, 26000.0]); s.add_argument('--sd_factor', type=float, default=2.0); s.set_defaults(fn=cmd_sites)
    s = sp.add_parser('run'); s.add_argument('--sites', required=True); s.add_argument('--methods', nargs='+', required=True); s.add_argument('--out', required=True)
    s.add_argument('--tag', default=''); s.add_argument('--radius', type=float, default=30.0); s.add_argument('--min_n', type=int, default=10)
    s.add_argument('--bins', type=int, default=6); s.add_argument('--photon_edges', type=float, nargs='*', default=[])
    s.add_argument('--nboot', type=int, default=1000); s.add_argument('--seed', type=int, default=1); s.add_argument('--no_common', action='store_true')
    s.add_argument('--no_shift', action='store_true'); s.add_argument('--pair_dt', type=int, default=1); s.add_argument('--frc_px', type=float, default=2.5); s.add_argument('--pairs', nargs='*', default=[]); s.add_argument('--gallery_n', type=int, default=24)
    s.set_defaults(fn=cmd_run)
    a = ap.parse_args(); a.fn(a)
