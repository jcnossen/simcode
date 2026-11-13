"""Merged events on the real 20 nm movie (2026-11-13 evening, part (3) of the merged-event study): the midpoint-band estimator of r24_midc.py,
calibrated on simulations with ground truth, applied to every method's table of the real nr20 movie. CPU only.

Estimator (per resolved site pair of an r24_rul.py per_pair npz): excess_c = observed fraction of the pair-window rows with |t - m| <= d / 4
minus the fraction two Gaussians at -+d / 2 (fitted weights and background, sd = the ACROSS sd of the pair, truncated on the window) predict;
also dv = along sd^2 - across sd^2. Merged rows (two sites of a structure ON in one window -> one row at the photon-weighted mean) fill the
midpoint band and widen the along spread only. Mean over the resolved pairs valid in all methods of the file; bootstrap over structures.
Truth (simulations): mixed fraction of the table rows (companion photons >= 5 % of the window, f2 >= 0.05, as r24_simeval.py):
  grid simulations s20g_a / s20g_c0 (3 x 4 grids at 20 nm, occupancy 0.35; m12 / u45 / sel rows; r24_simeval json),
  linear simulation s20_a (sel, u45 [r24_simeval json], the merged-score cuts combo90 / combo80 / photons80 / oracle [rows x f2 of
  r24_merged.py scores]) and its re-implemented SMLM / SIMFLUX tables (f2 computed here as r24_blmerged.py).
Calibration = least-squares line mixed = a + b excess_c over the GRID points (the main calibration; the linear-sim points are shown as a
check of transfer between geometries and methods). Applied to the real tables with the calibration's residual sd as a systematic band.
usage: python analysis/nanoruler/nr_merged.py [out prefix, default out/nanoruler/nr20/merged/real_merged]"""
import json, os, sys
import numpy as np
from scipy.special import ndtr
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode.util import proctitle
proctitle.set_title('nr-merged')
NR = '/home/jc/simcode2/out/nanoruler'; O = f'{NR}/nr20'; D0 = 20.0
out = sys.argv[1] if len(sys.argv) > 1 else f'{O}/merged/real_merged'
rng = np.random.default_rng(1)


def est(P, m, nboot=300):
    g = (P['common'] > 0) & (P[m + '_resolved'] > 0) & np.isfinite(P[m + '_sd_across'])
    d = P[m + '_d'][g]; s = P[m + '_sd_across'][g]; w1 = P[m + '_wmin'][g]; wb = np.nan_to_num(P[m + '_wb'][g]); h = d / 4; pe = 0
    for w_, mu in ((w1, -d / 2), (1 - w1, d / 2)):
        z = np.maximum(ndtr((D0 - mu) / s) - ndtr((-D0 - mu) / s), 1e-12); pe = pe + w_ * (ndtr((h - mu) / s) - ndtr((-h - mu) / s)) / z
    ex = P[m + '_mid_obs'][g] - ((1 - wb) * pe + wb * (2 * h) / (2 * D0)); dv = P[m + '_s2'][g] ** 2 - s ** 2; u = P['unit'][g]; uu = np.unique(u)
    bs = []
    for b in range(nboot):
        c = np.bincount(rng.choice(uu, len(uu)), minlength=u.max() + 1)[u]; bs.append((np.average(ex, weights=c), np.average(dv, weights=c)))
    bs = np.array(bs)
    return dict(n=int(g.sum()), excess=float(ex.mean()), excess_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), dv=float(dv.mean()),
                dv_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), resolved=float(np.mean(P[m + '_resolved'][P['common'] > 0] > 0)))


def simeval_mixed(fn, sel):
    return float(json.load(open(fn))['sel'][sel]['est']['mixed'])


def bl_mixed(tab, sim):
    """f2 >= 0.05 fraction of a baseline table of a simulated movie (rows matched as r24_blmerged.py)"""
    import ev_sigcal as ES
    T = np.load(tab); GW = ES.gt_windows(sim); x = T['x_nm'] / 65; y = T['y_nm'] / 65; fr = T['frame'] * 6
    Tg = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e3]); d, j = Tg.query(np.c_[x, y, fr * 1e3], distance_upper_bound=1.5); m = np.isfinite(d); j = np.where(m, j, 0)
    nb = Tg.query_ball_point(np.c_[GW[j, 2], GW[j, 3], GW[j, 0] * 1e3], 1.5); f2 = np.zeros(len(j))
    for q, (jj, ii) in enumerate(zip(j, nb)):
        o = [i for i in ii if i != jj]
        if o and m[q]: P = GW[o, 4]; f2[q] = P.sum() / (P.sum() + GW[jj, 4])
    return float(np.mean(f2[m] >= 0.05))


def cut_mixed(name):
    S = np.load(f'{O}/merged/s20_a_sel_scores.npz'); f2 = dict(zip(S['row'].tolist(), S['f2'].tolist()))
    T = np.load(f'{O}/merged/tab_s20_a_sel_{name}.npz'); v = np.array([f2.get(r, np.nan) for r in T['row'].tolist()])
    return float(np.nanmean(v >= 0.05))


cal = []     # (label, group, per_pair file, method, true mixed fraction)
for s in ('s20g_a', 's20g_c0'):
    for m in ('m12', 'u45', 'sel'):
        cal.append((f'{s} SIMCODE {m}', 'grid', f'{O}/simeval/per_pair_m12_{s}_truth.npz', m, simeval_mixed(f'{O}/simeval/gt_{s}_p1m12.json', m)))
for m in ('sel', 'u45'):
    cal.append((f's20_a SIMCODE {m}', 'linear', f'{O}/simeval/per_pair_s20_a_truth.npz', m, simeval_mixed(f'{O}/simeval/gt_s20_a_p1.json', m)))
for c in ('combo90', 'combo80', 'photons80', 'oracle'):
    cal.append((f's20_a SIMCODE sel, cut {c}', 'linear', f'{O}/merged/per_pair_s20_a_cut.npz', c, cut_mixed(c)))
for b in ('smlm', 'simflux'):
    cal.append((f's20_a re-implemented {b.upper()}', 'linear-bl', f'{O}/merged/per_pair_bl_s20_a_truth.npz', f'bl_{b}', bl_mixed(f'{O}/merged/tabs/bl_s20_a_{b}.npz', f'{O}/sim/s20_a')))
rows = []
for lab, grp, fn, m, mix in cal:
    e = est(np.load(fn), m); rows.append(dict(label=lab, group=grp, method=m, mixed=mix, **e))
G = [r for r in rows if r['group'] == 'grid']; X = np.array([r['excess'] for r in G]); Y = np.array([r['mixed'] for r in G])
b, a = np.polyfit(X, Y, 1); res = Y - (a + b * X); rsd = float(np.sqrt(np.sum(res ** 2) / max(len(Y) - 2, 1)))
for r in rows: r['pred'] = float(a + b * r['excess'])
real = []
for lst in ('picks_g2d', 'auto_sf'):
    for tag, mlist in ((f'm12_{lst}', ('pub_smlm', 'pub_simflux', 'bl_smlm', 'bl_simflux', 'simcode_m12', 'simcode_u45')), (f'{lst}_matched_3way', ('pub_smlm', 'pub_simflux', 'simcode'))):
        fn = f'{O}/methods_p1/per_pair_{tag}.npz'
        if not os.path.exists(fn): continue
        P = np.load(fn)
        for m in mlist:
            e = est(P, m); real.append(dict(pairs=lst, file=tag, method=m, pred=float(a + b * e['excess']), pred_ci=[float(a + b * v) for v in e['excess_ci']], **e))
L = ['# Merged events on the real 20 nm movie: midpoint-band estimator calibrated on simulations', '',
     f'Calibration on the grid simulations (s20g_a, s20g_c0; SIMCODE pass 1 rows m12 / u45 / sel): mixed fraction = {a:+.4f} + {b:.3f} x excess_c, '
     f'residual sd {rsd:.4f} ({len(G)} points). mixed = rows whose window has another site of the structure ON with >= 5 % of the photons.', '',
     '| simulation table | group | resolved pairs | resolved fraction | excess_c [95 % CI] | along^2 - across^2, nm^2 | true mixed fraction | predicted |', '|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f"| {r['label']} | {r['group']} | {r['n']} | {r['resolved']:.3f} | {r['excess']:+.4f} [{r['excess_ci'][0]:+.4f}, {r['excess_ci'][1]:+.4f}] | {r['dv']:+.2f} | {r['mixed']:.4f} | {r['pred']:.4f} |")
L += ['', '## Real nr20 movie (pass 1)', '', '| pair list | table set | method | resolved pairs | excess_c [95 % CI] | along^2 - across^2, nm^2 [95 % CI] | estimated mixed fraction [CI of excess only] |', '|---|---|---|---|---|---|---|']
for r in real:
    L.append(f"| {r['pairs']} | {r['file']} | {r['method']} | {r['n']} | {r['excess']:+.4f} [{r['excess_ci'][0]:+.4f}, {r['excess_ci'][1]:+.4f}] | {r['dv']:+.2f} [{r['dv_ci'][0]:+.2f}, {r['dv_ci'][1]:+.2f}] | "
             f"{r['pred']:.3f} [{r['pred_ci'][0]:.3f}, {r['pred_ci'][1]:.3f}] |")
L += ['', f'Systematic band of the calibration: +- {rsd:.3f} (residual sd over the grid points). Real-data caveat: site size / static error add to the across '
      'and along sd alike and cancel in excess_c to first order; label density and kinetics of the real sample are not those of the simulation.']
open(out + '.md', 'w').write('\n'.join(L) + '\n'); json.dump(dict(a=a, b=b, rsd=rsd, cal=rows, real=real), open(out + '.json', 'w'), indent=1, default=float)
print('\n'.join(L))
