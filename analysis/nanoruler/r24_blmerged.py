"""R24: merged versus single rows in the re-implemented SMLM / SIMFLUX baseline of a simulated movie (tables of r24_blsim.py), as a function of the
modulation-error filter (published processing: 0.02 for the 20 nm movie, 0.012 for 80 nm; re-implementation: 0.1). GT classes as r24_merged.py:
row matched to the nearest GT window of its cycle within 1.5 px; f2 = companion photons / all photons; single = f2 < 0.05 and fully on,
merged = f2 >= 0.2. Reports per threshold: rows kept, merged fraction among the kept rows, fraction of the merged / single rows kept, and the
ROC AUC of the modulation error (and photons, chi2) for merged versus single, overall and per pair axis (ky / kx / diagonal).
usage: python analysis/nanoruler/r24_blmerged.py <mv> <sim name> [out prefix]"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_sigcal as ES
mv, name = sys.argv[1], sys.argv[2]; B = f'/home/jc/simcode2/out/nanoruler/{mv}'; sim = f'{B}/sim/{name}'; out = sys.argv[3] if len(sys.argv) > 3 else f'{B}/merged/bl_{name}'
T = np.load(f'{B}/big/blsim/{name}_smlm.npz'); GW = ES.gt_windows(sim); x = T['x_nm'] / 65; y = T['y_nm'] / 65; fr = T['frame'] * 6
Tg = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e3]); d, j = Tg.query(np.c_[x, y, fr * 1e3], distance_upper_bound=1.5); m = np.isfinite(d); j = np.where(m, j, 0)
nb = Tg.query_ball_point(np.c_[GW[j, 2], GW[j, 3], GW[j, 0] * 1e3], 1.5); f2 = np.zeros(len(j)); ang = np.full(len(j), np.nan)
for q, (jj, ii) in enumerate(zip(j, nb)):
    o = [i for i in ii if i != jj]
    if not o or not m[q]: continue
    P = GW[o, 4]; f2[q] = P.sum() / (P.sum() + GW[jj, 4]); ang[q] = np.degrees(np.arctan2(abs((GW[o, 3] * P).sum() / P.sum() - GW[jj, 3]), abs((GW[o, 2] * P).sum() / P.sum() - GW[jj, 2])))
single = m & (f2 < 0.05) & (GW[j, 6] >= 1); merged = m & (f2 >= 0.2); axis = np.where(ang >= 67.5, 'ky', np.where(ang <= 22.5, 'kx', 'diag')); me = T['moderr']
def auc(s, pos, neg):
    v = np.r_[s[pos], s[neg]]; r = rankdata(v); n1, n0 = pos.sum(), neg.sum(); return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')
R = dict(n_rows=int(len(x)), unmatched=float(1 - m.mean()), n_single=int(single.sum()), n_merged=int(merged.sum()), merged_fraction_smlm=float(merged.mean()), thr={},
         auc={k: dict(all=auc(np.asarray(T[k], float), merged, single), **{ax: auc(np.asarray(T[k], float), merged & (axis == ax), single) for ax in ('ky', 'kx', 'diag')}) for k in ('moderr', 'photons', 'chisq')})
L = [f'## Baseline SMLM / SIMFLUX on {name}: merged rows versus the modulation-error filter', '',
     f'SMLM rows (cycle-aligned) {len(x)}, unmatched to GT {100 * (1 - m.mean()):.2f} %; single {single.sum()}, merged (f2 >= 0.2) {merged.sum()} = {100 * merged.mean():.2f} % of the SMLM table.', '',
     '| modulation-error filter | rows kept | of all | merged % among kept | merged rows kept | single rows kept | merged kept: pair along ky / kx / diagonal |', '|---|---|---|---|---|---|---|']
for thr in (0.012, 0.02, 0.05, 0.1, np.inf):
    k = me < thr; r = dict(n=int(k.sum()), frac=float(k.mean()), merged_among_kept=float(merged[k].mean()) if k.any() else float('nan'), merged_kept=float(k[merged].mean()), single_kept=float(k[single].mean()),
                           by_axis={ax: float(k[merged & (axis == ax)].mean()) for ax in ('ky', 'kx', 'diag')}); R['thr'][str(thr)] = r
    L.append(f"| {'none (SMLM table)' if not np.isfinite(thr) else '< %g' % thr} | {r['n']} | {r['frac']:.3f} | {100 * r['merged_among_kept']:.2f} | {r['merged_kept']:.3f} | {r['single_kept']:.3f} | " + ' / '.join(f"{r['by_axis'][ax]:.3f}" for ax in ('ky', 'kx', 'diag')) + ' |')
k = T['in_sf'].astype(bool); L += ['', f"SIMFLUX table of the run (filter 0.1 + SIMFLUX fit limits): {k.sum()} rows, merged {100 * merged[k].mean():.2f} %, merged kept {k[merged].mean():.3f}, single kept {k[single].mean():.3f}.", '',
     'ROC AUC merged versus single (all / ky / kx / diagonal): ' + '; '.join(f"{n} {a['all']:.3f} / {a['ky']:.3f} / {a['kx']:.3f} / {a['diag']:.3f}" for n, a in R['auc'].items())]
os.makedirs(os.path.dirname(out), exist_ok=True); open(out + '.md', 'w').write('\n'.join(L) + '\n'); json.dump(R, open(out + '.json', 'w'), indent=1); print('\n'.join(L))
