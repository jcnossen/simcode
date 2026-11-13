"""R24: a pipeline table of a simulated nanoruler movie (r24_sim.py) against ground truth, with closely spaced sites in mind.
GT windows: ev_sigcal.gt_windows (exact cycle-aligned windows; the simulated sample drift is added to the GT position). Only cycle-aligned
table rows (frame % 6 == 0). A row is matched to the NEAREST GT window of the same window start within 1.5 px. Its 'companions' are the
other GT windows of the same start within 1.5 px of the matched one (the other sites of the ruler): f2 = companion photons / all photons.
  single = f2 < 0.05 (one emitter of the ruler ON in the window), mixed = f2 >= 0.05 (simultaneous binding on the same ruler).
Per selection (good, sel, u <= 0.45, strict): rows, unmatched fraction, mixed fraction (and the mixed fraction of the GT windows themselves: what
the selection removes), RMS error per axis of the single rows (|e| < 50 nm; median error of the movie removed), gross fraction, robust
z = 1.4826 median |e / sigma| overall and per GT-photon bin, error of the mixed rows against the nearest site and against the photon-weighted
mean position. Recall = fraction of the fully-on GT target windows with a matched good row. Bootstrap over rulers (95 %).
usage: python analysis/nanoruler/r24_simeval.py <run dir or locs npz> <sim dir> [--out file.json] [--name label] [--sig sx_raw,sy_raw] [--ref A/locs.npz]"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_sigcal as ES
PX = 65.0; rob = lambda v: float(1.4826 * np.median(np.abs(v))) if len(v) else float('nan')
U_MAX = 0.45


def match(run, sim, sig=('sx', 'sy')):
    fn = run if run.endswith('.npz') else f'{run}/locs_p1.npz'
    T = dict(np.load(fn)); GW = ES.gt_windows(sim); G = np.load(f'{sim}/sim_gt.npz', allow_pickle=True); S = np.load(f'{sim}/sites.npz')
    rul = S['ruler'][G['site'][GW[:, 1].astype(int)]]
    ca = (T['frame'] % 6 == 0) & np.isfinite(T['x']) & np.isfinite(T['y'])
    Tg = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e3])
    d, j = Tg.query(np.c_[T['x'][ca], T['y'][ca], T['frame'][ca] * 1e3], distance_upper_bound=1.5); m = np.isfinite(d); j = np.where(m, j, 0)
    nb = Tg.query_ball_point(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e3], 1.5)
    Pt = np.array([GW[i, 4].sum() for i in nb]); f2 = 1 - GW[:, 4] / np.maximum(Pt, 1e-9)
    cen = np.array([[np.sum(GW[i, 2] * GW[i, 4]), np.sum(GW[i, 3] * GW[i, 4])] for i in nb]) / np.maximum(Pt, 1e-9)[:, None]
    R = dict(ca=np.nonzero(ca)[0], m=m, gi=j, f2=f2[j], ex=(T['x'][ca] - GW[j, 2]) * PX, ey=(T['y'][ca] - GW[j, 3]) * PX,
             cx=(T['x'][ca] - cen[j, 0]) * PX, cy=(T['y'][ca] - cen[j, 1]) * PX, P=GW[j, 4], Ptot=Pt[j], full=GW[j, 6] >= 1, rul=rul[j],
             sx=T[sig[0]][ca] * PX, sy=T[sig[1]][ca] * PX, phot=T['photons'][ca])
    for k in ('good', 'sel', 'strict'): R[k] = T[k][ca].astype(bool)
    R['u45'] = np.nan_to_num(T['strict_u'][ca].astype(float), nan=2.0) <= U_MAX
    g0 = m & R['good'] & (R['f2'] < 0.05) & (np.hypot(R['ex'], R['ey']) < 50)
    med = (np.median(R['ex'][g0]), np.median(R['ey'][g0]))
    for a, b in (('ex', 0), ('ey', 1), ('cx', 0), ('cy', 1)): R[a] = R[a] - med[b]
    R['median_error_nm'] = [float(v) for v in med]
    tg = (GW[:, 5] > 0) & (GW[:, 6] >= 1); hit = np.zeros(len(GW), bool); hit[j[m & R['good']]] = True
    R['gt'] = dict(n_windows=int(len(GW)), n_target_full=int(tg.sum()), recall_good=float(hit[tg].mean()), mixed_frac_gt_full=float((f2[tg] >= 0.05).mean()),
                   recall_good_single=float(hit[tg & (f2 < 0.05)].mean()), recall_good_mixed=float(hit[tg & (f2 >= 0.05)].mean()) if (tg & (f2 >= 0.05)).any() else float('nan'),
                   P_full_p5_p50_p95=[float(v) for v in np.percentile(GW[tg, 4], [5, 50, 95])])
    hs = np.zeros(len(GW), bool)
    for k in ('sel', 'u45'):
        hs[:] = False; hs[j[m & R[k]]] = True; R['gt']['recall_' + k] = float(hs[tg].mean())
    R['nrul'] = int(S['ruler'].max()) + 1
    return R


def stats(R, sel, edges, w=None):
    k = R[sel]; rw = np.ones(len(k), int) if w is None else w[R['rul']]
    idx = np.repeat(np.nonzero(k)[0], rw[k]); m = R['m'][idx]; o = dict(n=int(len(idx)), unmatched=float(1 - m.mean()))
    i = idx[m]; mixed = R['f2'][i] >= 0.05; o['mixed'] = float(mixed.mean()); s = i[~mixed]
    e = np.c_[R['ex'][s], R['ey'][s]]; ok = np.hypot(*e.T) < 50; o['gross'] = float(1 - ok.mean()); s = s[ok]; e = e[ok]; sg = np.c_[R['sx'][s], R['sy'][s]]
    o.update(n_single=int(len(s)), rms_x=float(np.sqrt(np.mean(e[:, 0] ** 2))), rms_y=float(np.sqrt(np.mean(e[:, 1] ** 2))), rms=float(np.sqrt(np.mean(e ** 2))),
             z=rob((e / sg).ravel()), zx=rob(e[:, 0] / sg[:, 0]), zy=rob(e[:, 1] / sg[:, 1]), sig_med=float(np.median(sg)), sig_rms=float(np.sqrt(np.mean(sg ** 2))),
             rsd=rob((e - np.median(e, 0)).ravel()))
    f = R['full'][s]; o['rms_full'] = float(np.sqrt(np.mean(e[f] ** 2))) if f.any() else np.nan; o['z_full'] = rob((e[f] / sg[f]).ravel()); o['frac_full'] = float(f.mean())
    b = np.clip(np.searchsorted(edges, R['P'][s], side='right') - 1, 0, len(edges) - 2)
    for q in range(len(edges) - 1):
        mm = b == q
        o[f'bin{q}_P'] = float(np.median(R['P'][s][mm])) if mm.any() else np.nan; o[f'bin{q}_n'] = int(mm.sum())
        o[f'bin{q}_rms'] = float(np.sqrt(np.mean(e[mm] ** 2))) if mm.sum() > 5 else np.nan; o[f'bin{q}_z'] = rob((e[mm] / sg[mm]).ravel()) if mm.sum() > 5 else np.nan
        o[f'bin{q}_sig'] = float(np.median(sg[mm])) if mm.any() else np.nan
    nb = len(edges) - 1; o['z_trend'] = o[f'bin{nb - 1}_z'] / o['bin0_z']
    x = i[mixed]
    if len(x) > 5:
        o.update(mixed_rms_nearest=float(np.sqrt(np.mean(np.r_[R['ex'][x], R['ey'][x]] ** 2))), mixed_rms_centroid=float(np.sqrt(np.mean(np.r_[R['cx'][x], R['cy'][x]] ** 2))),
                 mixed_z_nearest=rob(np.r_[R['ex'][x] / R['sx'][x], R['ey'][x] / R['sy'][x]]), mixed_sig_med=float(np.median(np.r_[R['sx'][x], R['sy'][x]])))
    return o


def run(run, sim, sig=('sx', 'sy'), nboot=200, bins=5, seed=1, ref=None):
    """ref: another table with IDENTICAL rows (variant A): the selections are taken from it and the paired RMS difference (this - ref) on the
    same rows is added (d_rms, d_rms_full; bootstrap over rulers)."""
    R = match(run, sim, sig); out = dict(gt=R['gt'], median_error_nm=R['median_error_nm'], sel={}); Q = None
    if ref:
        Q = match(ref, sim); assert len(Q['ca']) == len(R['ca'])
        for k in ('good', 'sel', 'u45', 'strict'): R[k] = Q[k]
        for k in ('m', 'f2', 'gi', 'P', 'full'): R[k] = Q[k]          # same GT assignment and classes as the reference
    g = R['m'] & R['sel'] & (R['f2'] < 0.05); edges = np.quantile(R['P'][g], np.linspace(0, 1, bins + 1)); edges[0] = -np.inf; edges[-1] = np.inf
    rng = np.random.default_rng(seed)
    def both(sel, w=None):
        s = stats(R, sel, edges, w)
        if Q is not None:
            q = stats(Q, sel, edges, w); s['d_rms'] = s['rms'] - q['rms']; s['d_rms_full'] = s['rms_full'] - q['rms_full']
        return s
    for sel in ('good', 'sel', 'u45', 'strict'):
        est = both(sel); acc = {k: [] for k in est}
        for b in range(nboot):
            s = both(sel, np.bincount(rng.integers(0, R['nrul'], R['nrul']), minlength=R['nrul']))
            for k in est: acc[k].append(s.get(k, np.nan))
        out['sel'][sel] = dict(est=est, ci={k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in acc.items()})
    return out, bins


def text(out, bins, name=''):
    g = out['gt']; L = [f'## {name}', f"GT: {g['n_target_full']} fully-on target windows (photons p5 / p50 / p95 {g['P_full_p5_p50_p95'][0]:.0f} / {g['P_full_p5_p50_p95'][1]:.0f} / {g['P_full_p5_p50_p95'][2]:.0f}), "
         f"of which mixed (another site of the ruler ON, >= 5 % of the photons) {100 * g['mixed_frac_gt_full']:.1f} %; recall good {g['recall_good']:.3f} (single {g['recall_good_single']:.3f}, "
         f"mixed {g['recall_good_mixed']:.3f}), sel {g['recall_sel']:.3f}, u <= 0.45 {g['recall_u45']:.3f}; median error removed {out['median_error_nm'][0]:+.2f} / {out['median_error_nm'][1]:+.2f} nm", '',
         '| selection | rows | unmatched % | mixed % | gross % | RMS x | RMS y | RMS fully-on | sigma median | z | z per GT-photon bin (dim -> bright) | z trend | mixed rows: RMS to nearest site / to weighted mean / z |', '|' + '---|' * 13]
    for sel, r in out['sel'].items():
        e, c = r['est'], r['ci']; f = lambda k, d=2: ('%.{0}f [%.{0}f, %.{0}f]'.format(d)) % (e[k], c[k][0], c[k][1]) if k in e else '-'
        L.append('| ' + ' | '.join([sel, str(e['n']), '%.2f' % (100 * e['unmatched']), '%.2f' % (100 * e['mixed']), '%.2f' % (100 * e['gross']), f('rms_x'), f('rms_y'), f('rms_full'), '%.2f' % e['sig_med'], f('z'),
                                    ' '.join('%.2f' % e[f'bin{q}_z'] for q in range(bins)) + ' (P ' + ' '.join('%.0f' % e[f'bin{q}_P'] for q in range(bins)) + ')', f('z_trend'),
                                    ('%.1f / %.1f / %.2f' % (e['mixed_rms_nearest'], e['mixed_rms_centroid'], e['mixed_z_nearest'])) if 'mixed_rms_nearest' in e else '-']) + ' |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    sig = tuple(g('--sig', 'sx,sy').split(','))
    out, bins = run(a[0], a[1], sig, ref=g('--ref')); t = text(out, bins, g('--name', a[0]))
    if g('--out'): json.dump(out, open(g('--out'), 'w'), indent=1, default=float); open(os.path.splitext(g('--out'))[0] + '.md', 'w').write(t)
    print(t)
