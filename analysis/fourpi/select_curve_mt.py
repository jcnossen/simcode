"""Selection curves on the real 4Pi MT data: quality versus number of localizations kept, network (head v10) against the
classical published-style and pixel-MLE rows of the SAME files, each ranked by its own quality scores.

Region, tube segments, registration and z drift: exactly those of showcase_real_mt.py (prepare()): the drift-corrected
cores of MT boxes 0 and 1, the 14 isolated straight segments picked on the classical table, the classical DME curve and the
structure-based sample z drift of the classical table applied to every row.
Per selection (lowest score first, kept fractions 100 .. 10 %, plus the operating points):
  tube cross-section over the segments: lateral FWHM, axial FWHM, hollow-tube dip (valley / lower peak of the lateral
    profile; 1 = no dip) and peak separation;
  FRC (x-y, box-0 core, two halves = alternating 500-window blocks; fig_cl.frc, 1/7);
  DAFL x / y / z: same emitter in adjacent windows of the same file (nearest within 0.35 px, no second within 0.7 px),
    robust sd of the difference / sqrt 2, all pairs and in the 2-4k photon bin; fringe-hop rate = fraction |dz| > 80 nm.
Scores (lower = better). Network: lateral sigma sqrt(sx^2 + sy^2), axial sigma sz, combined sigma
sqrt(sx^2 + sy^2 + sz^2) (nm), chi2/dof, -N, -nn, -pdet, and combined sigma x sqrt(max(chi2/dof, 1)). Classical rows:
their own sigmas (published sx, sy, sz; pixel MLE sxp, syp, szp), chi2px, -N, -nn, and the same combination.
There is no fringe-posterior margin column in the network tables (not ranked); the classical table has no llr / MD columns.
Operating points (network): all kept rows; count matched to the classical published-style rows of the same files in the
region; count matched in the two zoom boxes. Brightness control: the matched selection resampled to the classical photon
histogram, and DAFL inside the fixed 2-4k photon bin. Sampling control: classical rows of all 100 files subsampled at
random to several counts.
Usage: python analysis/fourpi/select_curve_mt.py --work <4Pi work folder> --net_locs '<work>/net/locs_MT_[0-9][0-9][0-9]_v10_box.npz' --tag v10_first20
Writes <work>/fig/MT_selection_curves_<tag>.png / .json and the showcase figure MT_showcase_real_<tag>_matched.png.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
sys.path.insert(0, f'{REPO}/analysis/fourpi')
import showcase_real_mt as SR
from showcase_real_mt import FC, PX, FPF, inrect, lg
import matplotlib.pyplot as plt

FRACS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1]


def sub(T, idx): return {k: v[idx] for k, v in T.items()}


def scores(T, kind):
    if kind == 'clp': sx, sy, sz = T['sxp'] * PX, T['syp'] * PX, T['szp'] * 1000
    else: sx, sy, sz = T['sx'] * PX, T['sy'] * PX, T['sz'] * 1000
    lat = np.hypot(sx, sy); comb = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    c2 = T['chi2'] / T['dof'] if kind == 'net' else T['chi2px']
    S = {'lateral sigma': lat, 'axial sigma': sz, 'combined sigma': comb, 'chi2/dof' if kind == 'net' else 'chi2 (pixel)': c2,
         'photons N': -T['N'], 'neighbour distance nn': -T['nn']}
    if kind == 'net': S['detector p'] = -T['pdet']
    S['combined sigma x chi2'] = comb * np.sqrt(np.maximum(c2 / (np.nanmedian(c2) if kind != 'net' else 1.0), 1.0))
    return {k: np.where(np.isfinite(v), v, np.inf) for k, v in S.items()}


def pairs(T, r=0.35 * PX):
    """same emitter in adjacent windows of the same file (vectorised: window index as a far third coordinate)"""
    gw = T['f'].astype(np.int64) * 3001 + (T['frame'] % FPF) // 6
    K = 1e6
    tr = cKDTree(np.c_[T['x'], T['y'], gw * K])
    d, k = tr.query(np.c_[T['x'], T['y'], (gw + 1) * K], k=2, distance_upper_bound=2 * r)
    ok = (d[:, 0] < r) & ~(d[:, 1] < 2 * r)
    return np.nonzero(ok)[0], k[ok, 0]


def rsd(v): return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 20 else float('nan')


def metrics(T, segs, core0, frc=True):
    out = dict(n=int(len(T['x'])))
    u, v, nseg = SR.cross_section(T, segs)
    if len(u) > 100:
        lat = u[np.abs(v) < 60]; ax = v[np.abs(u) < 40]
        pk = SR.peaks(lat)
        out.update(n_cross=int(len(u)), fwhm_lat=FC.fwhm(lat, -150, 150, 2.0), fwhm_ax=FC.fwhm(ax, -150, 150, 2.0),
                   dip=pk['dip'], peak_sep=pk['sep_nm'], dip_resolved=pk['resolved'])
    i, j = pairs(T)
    Np = 0.5 * (T['N'][i] + T['N'][j]); b = (Np >= 2000) & (Np < 4000)
    d = {q: T[q][i] - T[q][j] for q in ('x', 'y', 'z')}
    out.update(n_pairs=int(len(i)), dafl=[rsd(d[q]) / np.sqrt(2) for q in 'xyz'], n_pairs_2_4k=int(b.sum()),
               dafl_2_4k=[rsd(d[q][b]) / np.sqrt(2) for q in 'xyz'],
               fringe_hop=float(np.mean(np.abs(d['z'] - np.median(d['z'])) > 80)) if len(i) > 20 else float('nan'))
    if frc:
        m = inrect(T['x'], T['y'], core0); h = ((T['frame'][m] // 6) // 500) % 2 == 0
        x, y = T['x'][m], T['y'][m]
        out['frc'] = FC.frc(x[h], y[h], x[~h], y[~h]) if h.sum() > 200 and (~h).sum() > 200 else float('nan')
    return out


def curve(T, S, segs, core0, counts):
    o = np.argsort(S, kind='stable'); res = []
    for n in counts:
        n = int(min(n, len(o)))
        r = metrics(sub(T, np.sort(o[:n])), segs, core0); r['n_keep'] = n; res.append(r)
    return res


def photon_match(Tsel, Nref, rng, nb=20):
    """subsample Tsel so that its photon histogram (quantile bins of Nref inside the common N range) has the shape of
    Nref; as many rows as possible. Returns indices and the fraction of Nref inside the common range."""
    lo, hi = max(Nref.min(), Tsel['N'].min()), min(Nref.max(), Tsel['N'].max())
    r = Nref[(Nref >= lo) & (Nref <= hi)]; cover = float(len(r) / len(Nref))
    e = np.quantile(np.log(r), np.linspace(0, 1, nb + 1)); e[0] = -np.inf; e[-1] = np.inf
    br = np.clip(np.searchsorted(e, np.log(r)) - 1, 0, nb - 1)
    ok = (Tsel['N'] >= lo) & (Tsel['N'] <= hi)
    bs = np.where(ok, np.clip(np.searchsorted(e, np.log(np.maximum(Tsel['N'], 1e-9))) - 1, 0, nb - 1), -1)
    pr = np.bincount(br, minlength=nb) / len(r); cs = np.bincount(bs[bs >= 0], minlength=nb)
    tot = np.min(np.where(pr > 0, cs / np.maximum(pr, 1e-12), np.inf))
    keep = []
    for k in range(nb):
        idx = np.nonzero(bs == k)[0]; m = int(round(pr[k] * tot))
        if m > 0: keep.append(rng.choice(idx, min(m, len(idx)), replace=False))
    return np.sort(np.concatenate(keep)), cover


def _curve_job(args):
    k, sn = args
    return k, sn, curve(_G['M'][k], _G['SC'][k][sn], _G['segs'], _G['core0'], [int(round(f * _G['n'][k])) for f in FRACS])


_G = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True, help='4Pi work folder')
    ap.add_argument('--cl', default=None, help='default <work>/classical/real/cl_MT_wv_all.npz')
    ap.add_argument('--net_locs', nargs='*', default=None); ap.add_argument('--net_all', default=None)
    ap.add_argument('--head', default='v10'); ap.add_argument('--tag', required=True)
    ap.add_argument('--regions', default=None, help='default <work>/patches/regions_MT.json')
    ap.add_argument('--rule', default='combined sigma', help='network score used for the operating points and the matched showcase row (the ranking of all scores is reported regardless)')
    a = ap.parse_args()
    P4 = a.work; a.cl = a.cl or f'{P4}/classical/real/cl_MT_wv_all.npz'; a.regions = a.regions or f'{P4}/patches/regions_MT.json'
    FC.OUT = f'{P4}/fig/tmp_showcase'       # never write into the classical worker's directory
    OUTD = f'{P4}/fig'
    rng = np.random.default_rng(0)
    R, G, segs, info, files, src, ncl = SR.prepare(a, SR.CL_EXTRA, SR.NET_EXTRA)
    cores = info['cores_nm']; core0 = cores[G['box']]
    inreg = lambda T: inrect(T['x'], T['y'], cores[0]) | inrect(T['x'], T['y'], cores[1])
    same = lambda T: np.isin(T['f'], files)
    M = {'net': sub(R['net'], np.nonzero(inreg(R['net']))[0]),
         'cl': sub(R['cl'], np.nonzero(inreg(R['cl']) & same(R['cl']))[0]),
         'clp': sub(R['clp'], np.nonzero(inreg(R['clp']) & same(R['clp']))[0])}
    n = {k: len(T['x']) for k, T in M.items()}; lg(f'rows in the region (same files {files}): {n}')
    zb = [(c[0] - s / 2, c[0] + s / 2, c[1] - s / 2, c[1] + s / 2) for c, s in ((G['zoom1']['c'], G['zoom1']['size']), (G['zoom2']['c'], G['zoom2']['size']))]
    inzoom = lambda T: inrect(T['x'], T['y'], zb[0]) | inrect(T['x'], T['y'], zb[1])
    n_cl_zoom = int(inzoom(M['cl']).sum())
    out = dict(tag=a.tag, head=a.head, files=files, n_files=len(files), region='drift-corrected cores of MT boxes 0 and 1',
               n_rows=n, n_cl_zoom=n_cl_zoom, n_segments=len(segs), fracs=FRACS, curves={}, info=info,
               notes=['no fringe-posterior margin column in the network tables: not ranked', 'classical table has no llr / MD columns',
                      'FRC on the box-0 core only, halves = alternating 500-window blocks',
                      'tube FWHM, dip and FRC depend on the number of distinct molecules (files), not only on the count: see classical_file_subsets vs classical_subsampled_100files; with few files a selection narrows the per-segment centre-line fit onto few molecules (FWHM biased low)'])
    SC = {k: scores(M[k], k) for k in M}
    _G.update(M=M, SC=SC, segs=segs, core0=core0, n=n)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    jobs = [(k, sn) for k in M for sn in SC[k]]
    with ProcessPoolExecutor(min(len(jobs), 16), mp_context=mp.get_context('fork')) as ex:
        res = list(ex.map(_curve_job, jobs))
    for k, sn, c in res:
        out['curves'].setdefault(k, {})[sn] = c
        lg(f"{k} {sn}: " + ' | '.join(f"{r['n_keep']}: lat {r.get('fwhm_lat', np.nan):.0f} dip {r.get('dip', np.nan):.2f} dafl {np.nanmean(r['dafl'][:2]):.2f}/{r['dafl'][2]:.2f} hop {r['fringe_hop']:.3f} frc {r['frc']:.1f}" for r in c))
    # best network score: lowest mean lateral DAFL at the matched count, among scores
    n_m = n['cl']
    pts = {}
    for sn, S in SC['net'].items():
        o = np.argsort(S, kind='stable'); pts[sn] = metrics(sub(M['net'], np.sort(o[:n_m])), segs, core0); pts[sn]['n_keep'] = n_m
    rank = sorted(pts, key=lambda s: np.nanmean(pts[s]['dafl'][:2]) + 0.5 * pts[s]['dafl'][2])
    out['matched_count_by_score'] = pts; out['best_score'] = rank[0]; out['score_rank'] = rank
    best = a.rule if a.rule else rank[0]; out['rule'] = best
    lg(f'best network score at matched count {n_m}: {rank[0]}; ranking {rank}; rule used: {best}')
    # operating points (best score)
    o = np.argsort(SC['net'][best], kind='stable')
    cz = np.cumsum(inzoom(M['net'])[o]); n_z = int(np.searchsorted(cz, n_cl_zoom) + 1)
    OP = {'all': metrics(M['net'], segs, core0), 'matched count': pts[best], 'matched count in zooms': metrics(sub(M['net'], np.sort(o[:n_z])), segs, core0)}
    OP['all']['n_keep'] = n['net']; OP['matched count in zooms']['n_keep'] = n_z
    OP['classical published (same files)'] = metrics(M['cl'], segs, core0); OP['classical published (same files)']['n_keep'] = n['cl']
    OP['classical pixel MLE (same files)'] = metrics(M['clp'], segs, core0); OP['classical pixel MLE (same files)']['n_keep'] = n['clp']
    # brightness control
    sel = sub(M['net'], np.sort(o[:n_m]))
    for nm_, A_, ref in (('matched count, photon-matched to classical', sel, M['cl']['N']), ('all, photon-matched to classical', M['net'], M['cl']['N']),
                         ('classical, photon-matched to the selected network', M['cl'], sel['N']),
                         ('pixel MLE, photon-matched to the selected network', M['clp'], sel['N'])):
        pm, cover = photon_match(A_, ref, rng)
        OP[nm_] = metrics(sub(A_, pm), segs, core0); OP[nm_]['n_keep'] = int(len(pm)); OP[nm_]['ref_photon_mass_covered'] = cover
        OP[nm_]['median_N'] = float(np.median(A_['N'][pm])); OP[nm_]['median_N_ref'] = float(np.median(ref))
    out['photon_median'] = dict(net_all=float(np.median(M['net']['N'])), net_matched=float(np.median(sel['N'])), cl=float(np.median(M['cl']['N'])),
                                clp=float(np.median(M['clp']['N'])))
    out['operating_points'] = OP; out['n_zoom_matched'] = n_z
    for k_, r in OP.items():
        lg(f"OP {k_}: n {r['n_keep']} lat {r.get('fwhm_lat', np.nan):.1f} ax {r.get('fwhm_ax', np.nan):.1f} dip {r.get('dip', np.nan):.2f} sep {r.get('peak_sep')} "
           f"dafl {np.round(r['dafl'], 2).tolist()} 2-4k {np.round(r['dafl_2_4k'], 2).tolist()} hop {r['fringe_hop']:.3f} frc {r['frc']:.1f}")
    # sampling control: classical published, all 100 files, random subsets
    CA = sub(R['cl'], np.nonzero(inreg(R['cl']))[0]); nA = len(CA['x'])
    sc_counts = sorted(set([n['cl'], 2 * n['cl'], n['net'], 4 * n['cl'], 8 * n['cl'], nA]))
    sc_counts = [c for c in sc_counts if c <= nA]
    out['classical_subsampled_100files'] = []
    for c in sc_counts:
        r = metrics(sub(CA, np.sort(rng.choice(nA, c, replace=False))), segs, core0); r['n_keep'] = c
        out['classical_subsampled_100files'].append(r)
        lg(f"cl subsample {c}: lat {r.get('fwhm_lat', np.nan):.1f} dip {r.get('dip', np.nan):.2f} frc {r['frc']:.1f}")
    # sampling control 2: classical published rows of random FILE subsets (distinct molecules, not only the count, set the
    # cross-section and the FRC): k = number of network files, 2k, 4k, all 100; 3 draws each
    allf = np.unique(CA['f']); out['classical_file_subsets'] = []
    for kf in sorted(set([len(files), 2 * len(files), 4 * len(files)])):
        if kf > len(allf): continue
        for d in range(3):
            fs = rng.choice(allf, kf, replace=False)
            r = metrics(sub(CA, np.nonzero(np.isin(CA['f'], fs))[0]), segs, core0); r['n_keep'] = int(np.isin(CA['f'], fs).sum()); r['n_files'] = int(kf)
            out['classical_file_subsets'].append(r)
            lg(f"cl {kf} random files: n {r['n_keep']} lat {r.get('fwhm_lat', np.nan):.1f} dip {r.get('dip', np.nan):.2f} frc {r['frc']:.1f}")
    r = metrics(CA, segs, core0); r['n_keep'] = nA; r['n_files'] = int(len(allf)); out['classical_file_subsets'].append(r)
    fn = f'{OUTD}/MT_selection_curves_{a.tag}'
    json.dump(out, open(fn + '.json', 'w'), indent=1, default=float)
    plot(out, fn + '.png', best)
    # showcase with the matched-count network row (classical rows on the same files)
    Rv = {k: (v if k == 'net' else {q: w[np.isin(v['f'], files)] for q, w in v.items()}) for k, v in R.items()}
    so = np.argsort(scores(R['net'], 'net')[best], kind='stable')
    # same rule as in the region: keep the rows better than the score threshold reached at the matched count
    thr = np.sort(SC['net'][best])[n_m - 1]
    Rv['net_m'] = sub(R['net'], np.nonzero(scores(R['net'], 'net')[best] <= thr)[0])
    meth = [('net', '4Pi-SIMCODE (network),\nall kept'), ('net_m', f'4Pi-SIMCODE, {best}\n≤ {thr:.1f}' + (' nm' if 'sigma' in best else ''))] + SR.METH[1:]
    nfiles = {k: len(files) for k in Rv}
    note = (f" Row 2: the network rows with {best} ≤ {thr:.2f}, the threshold at which the network keeps as many localizations in boxes 0 and 1 "
            f"as the classical published-style row of the same files ({n_m:,}).")
    J = SR.figure(Rv, G, segs, a.tag + '_matched', a, info, nfiles, files, meth=meth, extra_note=note)
    J.update(best_score=best, threshold=float(thr), n_matched=n_m)
    json.dump(J, open(f'{OUTD}/MT_showcase_real_{a.tag}_matched.json', 'w'), indent=1, default=float)
    lg(f'-> {fn}.png, {OUTD}/MT_showcase_real_{a.tag}_matched.png')


def plot(out, fn, best):
    C = out['curves']
    cols = dict(net='#2a78d6', cl='#eb6834', clp='#1baf7a')
    names = dict(net='network', cl='classical published-style', clp='classical pixel MLE')
    clbest = {k: min(C[k], key=lambda s: np.nanmean(C[k][s][len(FRACS) // 2]['dafl'][:2])) for k in ('cl', 'clp')}
    panels = [('fwhm_lat', 'tube FWHM, lateral (nm)'), ('fwhm_ax', 'tube FWHM, axial (nm)'), ('dip', 'hollow-tube dip\n(valley / peak; 1 = none)'),
              ('frc', 'FRC resolution, box-0 core (nm)\n(limited by distinct molecules)'), ('dafl_lat', 'DAFL lateral, mean x/y (nm)'), ('dafl_z', 'DAFL axial (nm)'),
              ('fringe_hop', 'fringe hops, |Δz| > 80 nm\n(adjacent windows)'), ('dafl_lat_24', 'DAFL lateral, 2–4k photons (nm)')]
    def val(r, key):
        if key == 'dafl_lat': return np.nanmean(r['dafl'][:2])
        if key == 'dafl_z': return r['dafl'][2]
        if key == 'dafl_lat_24': return np.nanmean(r['dafl_2_4k'][:2])
        return r.get(key, np.nan)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
    for ax, (key, lab) in zip(axs.ravel(), panels):
        for sn, c in C['net'].items():
            if sn in (best, 'photons N'): continue
            ax.plot([r['n_keep'] for r in c], [val(r, key) for r in c], color='#b9b8b1', lw=0.8)
        for k, sn, ls, lw in (('net', 'photons N', '--', 1.4), ('net', best, '-', 2.0), ('cl', clbest['cl'], '-', 2.0), ('clp', clbest['clp'], '-', 1.6)):
            c = C[k][sn]
            ax.plot([r['n_keep'] for r in c], [val(r, key) for r in c], color=cols[k], ls=ls, lw=lw, marker='o', ms=3,
                    label=f'{names[k]}, ranked by {sn}')
        cs = out['classical_subsampled_100files']
        ax.plot([r['n_keep'] for r in cs], [val(r, key) for r in cs], color=cols['cl'], ls=':', lw=1.4, label='classical published, 100 files, random rows')
        cs = out['classical_file_subsets']
        ax.plot([r['n_keep'] for r in cs], [val(r, key) for r in cs], 's', color=cols['cl'], mfc='none', ms=5, label='classical published, random file subsets (k, 2k, 4k, 100 files)')
        for nm_, mk in (('matched count', '*'), ('matched count in zooms', 'D'), ('matched count, photon-matched to classical', 'v'),
                        ('classical, photon-matched to the selected network', 's'), ('pixel MLE, photon-matched to the selected network', '^')):
            if nm_ not in out['operating_points'] or out['operating_points'][nm_]['n_keep'] < 5000: continue   # failed photon match: not drawn
            r = out['operating_points'][nm_]; ax.plot(r['n_keep'], val(r, key), mk, color='#0b0b0b', ms=8 if mk == '*' else 5, mfc='none' if mk != '*' else '#0b0b0b',
                                                      label=f'network operating point: {nm_}')
        ax.set_xscale('log'); ax.set_title(lab, fontsize=8); ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlabel('localizations in boxes 0 + 1')
    axs[0, 0].plot([], [], color='#b9b8b1', lw=0.8, label='network, other scores')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Real MT, quality vs number kept (each method ranked by its own score; network files {out['files']}, classical rows = the same files "
                 f"unless noted; {out['n_segments']} tube segments)", fontsize=9)
    plt.tight_layout(rect=(0, 0.1, 1, 0.97)); fig.savefig(fn, dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
