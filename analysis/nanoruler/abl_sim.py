"""Nanoruler ablation, part E: the solve variants on the SIMULATED nanoruler movies, against ground truth.

Input: the variant tables of abl_solve.py for a simulated set (sima_p1 / sima_p2 = nr_a with brightness variation, simc0_* = nr_c0 without).
GT rows: ev_sigcal.gt_windows (exact cycle-aligned windows, incl. the simulated sample drift); a table row is matched to the nearest GT
window of the same start within 1 px; errors > 50 nm are counted as gross and excluded from RMS / z (as ev_sigcal.load).
Views as abl_nra.py: common = cycle-aligned rows with max sigma <= 12 nm (m12) in variant A (same windows for all), own = each variant's own.
Per variant, per GT-photon bin (edges from A) and overall, bootstrap over rulers (the events of a ruler are resampled together):
  RMS error per axis (nm), gross fraction, median bias, robust z sd = robust sd of error / sigma for the four sigma levels
  (cal, nb, raw, f; see abl_solve.py), sigma_rms / RMS, and the paired RMS difference to A on the same rows (common view).
No CRLB exists for these movies (the simulator does not provide one); RMS / CRLB per variant is in tab_gate_<front>.md (gate scenes).
The site-spread metrics on the simulated sites come from abl_nra.py (sites of sim/<movie>/sites.npz).

usage: python analysis/nanoruler/abl_sim.py <set> [--nboot 300] [--bins 6] -> out/nanoruler/ablation/sim_<set>.json, tab_sim_<set>.md"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
LEVELS = (('cal', 'sx', 'sy'), ('u', 'sx_u', 'sy_u'), ('nb', 'sx_nb', 'sy_nb'), ('raw', 'sx_raw', 'sy_raw'), ('f', 'sx_f', 'sy_f'))
rob = lambda v: float(1.4826 * np.median(np.abs(v))) if len(v) else np.nan


def sim_sites(sim):
    """sites / rulers of a simulated movie in the nra_lib format (nm, common frame)"""
    D = np.load(f'{sim}/sites.npz'); o = np.argsort(D['site'])
    sites = [dict(id=int(D['site'][i]), ruler=int(D['ruler'][i]), x=float(D['x'][i] * A.PX), y=float(D['y'][i] * A.PX)) for i in o]
    rulers = []
    for r in range(len(D['ruler_cx'])):
        ids = [s['id'] for s in sites if s['ruler'] == r]; ids = sorted(ids, key=lambda i: int(D['idx'][o][i]))
        i0 = ids[0]; rulers.append(dict(id=r, sites=ids, n_sites=len(ids), axis=[float(D['ax'][o][i0]), float(D['ay'][o][i0])]))
    return sites, rulers


def evaluate(setn, nboot=300, nbins=6):
    import ev_sigcal as ES
    from abl_nra import LABEL, ci, f3
    S = A.SETS[setn]; sim = S['sim']; base = f'{A.OUT}/runs/{setn}'
    order = [n for n in ['A', 'auto', 'autoS', 'B', 'C', 'D', 'E', 'E1', 'F', 'G', 'H', 'An', 'Dn'] if os.path.exists(f'{base}/{n}/locs.npz')]
    GW = ES.gt_windows(sim); G = np.load(f'{sim}/sim_gt.npz', allow_pickle=True); SI = np.load(f'{sim}/sites.npz')
    ruler_of_site = np.zeros(int(SI['site'].max()) + 1, int); ruler_of_site[SI['site']] = SI['ruler']
    ev_ruler = ruler_of_site[G['site']]; nr = int(SI['ruler'].max()) + 1
    T = {n: A.load_table(f'{base}/{n}/locs.npz') for n in order}; mA = A.selection(T['A'], 'm12')
    rng = np.random.default_rng(1); W = [np.ones(nr, int)] + [np.bincount(rng.integers(0, nr, nr), minlength=nr) for _ in range(nboot)]
    R = dict(set=setn, order=order, views={}, n_gt_targets=int((GW[:, 5] > 0).sum()))
    edges = None
    for view in ('common', 'own'):
        V_ = {}; EA = None
        for n in order:
            t = T[n]; m = mA if view == 'common' else A.selection(t, 'm12'); rows = np.nonzero(m)[0]
            gi = ES.E.match_fast(GW, t['frame'][rows].astype(np.int64), t['x'][rows], t['y'][rows], r=1.0); ok = gi >= 0
            e = np.full((len(rows), 2), np.nan); e[ok] = np.c_[t['x'][rows][ok] - GW[gi[ok], 2], t['y'][rows][ok] - GW[gi[ok], 3]]
            e = (e - np.nanmedian(e, 0)) * A.PX; gross = ok & ~(np.hypot(*e.T) < 50); use = ok & ~gross
            P = np.full(len(rows), np.nan); P[ok] = GW[gi[ok], 4]; rl = np.zeros(len(rows), int); rl[ok] = ev_ruler[GW[gi[ok], 1].astype(int)]
            full = np.zeros(len(rows), bool); full[ok] = GW[gi[ok], 6] >= 1
            if edges is None:
                edges = np.quantile(P[use], np.linspace(0, 1, nbins + 1)); edges[0] = -np.inf; edges[-1] = np.inf
            b = np.clip(np.searchsorted(edges, P, side='right') - 1, 0, nbins - 1)
            sig = {lv: np.c_[t[kx][rows], t[ky][rows]].astype(np.float64) * A.PX for lv, kx, ky in LEVELS}
            if n == 'A': EA = dict(e=e, use=use)
            acc = {}
            for w in W:
                wl = np.where(use, w[rl], 0); li = np.repeat(np.arange(len(rows)), wl)
                for bi in list(range(nbins)) + ['all', 'full']:
                    k = li if bi == 'all' else li[full[li]] if bi == 'full' else li[b[li] == bi]
                    put = lambda key, v: acc.setdefault(f'{bi}|{key}', []).append(v)
                    if len(k) < 20:
                        for key in ['n', 'phot', 'rms', 'rms_x', 'rms_y', 'bias_x', 'bias_y'] + [f'{lv}_{q}' for lv in sig for q in ('z', 'sig_over_rms', 'sig')] + ['d_rms']: put(key, np.nan)
                        continue
                    ek = e[k]; rms = float(np.sqrt(np.mean(ek ** 2)))
                    put('n', len(k)); put('phot', float(np.median(P[k]))); put('rms', rms); put('rms_x', float(np.sqrt(np.mean(ek[:, 0] ** 2)))); put('rms_y', float(np.sqrt(np.mean(ek[:, 1] ** 2))))
                    put('bias_x', float(np.median(ek[:, 0]))); put('bias_y', float(np.median(ek[:, 1])))
                    for lv, s in sig.items():
                        put(f'{lv}_z', rob((ek / s[k]).ravel())); put(f'{lv}_sig_over_rms', float(np.sqrt(np.mean(s[k] ** 2))) / rms); put(f'{lv}_sig', float(np.median(s[k])))
                    if view == 'common' and EA is not None:
                        k2 = k[EA['use'][k]]; put('d_rms', float(np.sqrt(np.mean(e[k2] ** 2)) - np.sqrt(np.mean(EA['e'][k2] ** 2))) if len(k2) > 20 else np.nan)
                    else: put('d_rms', np.nan)
            V_[n] = dict(stats={k: ci(v) for k, v in acc.items()}, n_rows=int(len(rows)), n_matched=int(ok.sum()), gross_pct=float(100 * gross.sum() / max(ok.sum(), 1)),
                         unmatched_pct=float(100 * (~ok).mean()))
        R['views'][view] = dict(variants=V_, edges=[float(x) for x in edges])
    A.jdump(R, f'{A.OUT}/sim_{setn}.json')
    Ls = [f'# Solve variants on the simulated nanoruler movie, against ground truth: {setn}', '', f'{sim}; {R["n_gt_targets"]} GT target windows (cycle-aligned). nm; [95 % CI, bootstrap over rulers]. '
          'z = robust sd of error / sigma (1 = calibrated). Errors > 50 nm are gross (excluded from RMS / z).', '']
    for view in ('common', 'own'):
        V_ = R['views'][view]['variants']; sA = V_['A']['stats']
        Ls += [f'## {view} rows', '', '| variant | rows | unmatched % | gross % | RMS (per axis, all) | d RMS vs A (same rows) | RMS fully-on windows | bias x / y | z cal | z cal x g(u) | z without bin factor | z uncalibrated | z Fisher | '
               'uncalibrated z, dimmest bin | uncalibrated z, brightest bin |', '|' + '---|' * 15]
        for n in order:
            v = V_[n]; s = v['stats']
            Ls.append('| ' + ' | '.join([LABEL.get(n, n), str(v['n_rows']), '%.1f' % v['unmatched_pct'], '%.2f' % v['gross_pct'], f3(s['all|rms']),
                                         ('%+.3f [%+.3f, %+.3f]' % tuple(s['all|d_rms'])) if n != 'A' and view == 'common' else '-', f3(s['full|rms']), '%+.2f / %+.2f' % (s['all|bias_x'][0], s['all|bias_y'][0]),
                                         f3(s['all|cal_z']), f3(s['all|u_z']), f3(s['all|nb_z']), f3(s['all|raw_z']), f3(s['all|f_z']), f3(s['0|raw_z']), f3(s[f'{nbins - 1}|raw_z'])]) + ' |')
        for key, txt in (('rms', 'RMS error (nm, per axis)'), ('cal_z', 'z sd, calibrated sigma'), ('u_z', 'z sd, calibrated sigma x g(u)'), ('raw_z', 'z sd, uncalibrated model sigma')):
            Ls += ['', f'{txt} per GT-photon bin ({view} rows):', '', '| variant | ' + ' | '.join('%.0f ph' % sA[f'{i}|phot'][0] for i in range(nbins)) + ' |', '|' + '---|' * (nbins + 1)]
            for n in order: Ls.append('| ' + ' | '.join([LABEL.get(n, n)] + [f3(V_[n]['stats'][f'{i}|{key}']) for i in range(nbins)]) + ' |')
        Ls.append('')
    md = '\n'.join(Ls) + '\n'; open(f'{A.OUT}/tab_sim_{setn}.md', 'w').write(md); print(md)


if __name__ == '__main__':
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    A.title(); evaluate(a[0], int(g('--nboot', 300)), int(g('--bins', 6)))
