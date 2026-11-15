"""R24: can merged rows (two sites of one structure ON in the same 6-frame window -> ONE row at the photon-weighted mean) be told from
single-emitter rows? Simulated movie with ground truth, pass-1 pipeline table (cycle-aligned rows).
GT: ev_sigcal.gt_windows; a row is matched to the nearest GT window of its start within 1.5 px; companions = other GT windows of that start
within 1.5 px; f2 = companion photons / all photons. Classes: single = f2 < 0.05 and the emitter ON in all 6 frames; merged = f2 >= 0.2;
(0.05 <= f2 < 0.2 and partially-on singles are 'other': kept in the yield, not in the AUC). Pair axis = direction from the matched emitter to the
photon-weighted companion position: 'ky' within 22.5 deg of the y axis (the wave vector of orientation 0, steps 0-2), 'kx' within 22.5 deg of x
(orientation 1), else 'diag'.
Per-row statistics (larger = more merged-like after the stated sign):
  q_dof      solve residual per degree of freedom            dnll     the solve's dnll column           kin   kinetic flag (0 / 1)
  1-p        1 - network detection probability               photons  window photons
  dprior     |solve - prior position| (nm)                   dprior_z the same / reported sigma
  moderr     max_k |I_k / sum(I) - e_k|, e_k = normalized pattern expectation at the solved position (the SIMFLUX modulation error)
  1-rho_o    1 - apparent modulation relative to the pattern, orientation o: least squares of the 3 intensities of that orientation on
             A rel_k (1 + rho m_k cos theta_k) at the solved position; 1-rho_min = the larger of the two
  chi2_I     sum_k (I_k - N e_k)^2 / (I_k + 4)  (Poisson-like residual of the six intensities against the pattern)
  combo      logistic regression on the standardized statistics, 2-fold cross-validated over rulers (odd / even ruler index)
Outputs: ROC AUC per statistic (all merged rows and per pair axis), merged fraction remaining versus the yield kept (cut on one statistic),
and <out>_scores.npz (row index into the table, class, f2, axis, every statistic, combo) for cut tables.
usage: python analysis/nanoruler/r24_merged.py <run dir> <sim dir> <out prefix> [--sel sel|m12|good]"""
import json, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ev_sigcal as ES, abl_lib as A
from simcode.processing.common import pattern_frames
PX = 65.0
a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
run, sim, out = a[0], a[1], a[2]; sel = g('--sel', 'sel')
T = A.load_table(f'{run}/locs_p1.npz'); GW = ES.gt_windows(sim); G = np.load(f'{sim}/sim_gt.npz', allow_pickle=True); S = np.load(f'{sim}/sites.npz')
ca = (T['frame'] % 6 == 0) & np.isfinite(T['x']) & np.isfinite(T['y'])
k = ca & (T[sel].astype(bool) if sel != 'm12' else A.selection(T, 'm12')); idx = np.nonzero(k)[0]
Tg = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e3]); d, j = Tg.query(np.c_[T['x'][idx], T['y'][idx], T['frame'][idx] * 1e3], distance_upper_bound=1.5); m = np.isfinite(d)
idx, j = idx[m], j[m]
nb = Tg.query_ball_point(np.c_[GW[j, 2], GW[j, 3], GW[j, 0] * 1e3], 1.5)
f2 = np.zeros(len(j)); ang = np.full(len(j), np.nan)
for q, (jj, ii) in enumerate(zip(j, nb)):
    o = [i for i in ii if i != jj]
    if not o: continue
    P = GW[o, 4]; f2[q] = P.sum() / (P.sum() + GW[jj, 4]); cx = (GW[o, 2] * P).sum() / P.sum() - GW[jj, 2]; cy = (GW[o, 3] * P).sum() / P.sum() - GW[jj, 3]
    ang[q] = np.degrees(np.arctan2(abs(cy), abs(cx)))
full = GW[j, 6] >= 1; cls = np.where((f2 < 0.05) & full, 0, np.where(f2 >= 0.2, 1, 2))          # 0 single, 1 merged, 2 other
axis = np.where(ang >= 67.5, 'ky', np.where(ang <= 22.5, 'kx', 'diag')); rul = S['ruler'][G['site'][GW[j, 1].astype(int)]]
# statistics
x, y, fr = T['x'][idx], T['y'][idx], T['frame'][idx]; I = T['I'][idx].astype(np.float64)
Pt = torch.load(f'{run}/pattern_est.pt', weights_only=False)
uw, inv = np.unique(fr, return_inverse=True); PW = pattern_frames(Pt, (uw[:, None] + np.arange(6)[None]).reshape(-1), 0, 0, 'cpu').reshape(len(uw), 6, 5).numpy()[inv]
th = PW[..., 0] * x[:, None] + PW[..., 1] * y[:, None] - PW[..., 2]; mk = PW[..., 3]; rel = PW[..., 4]
e = rel * (1 + mk * np.cos(th)); e = e / e.sum(1, keepdims=True); N = I.sum(1, keepdims=True)
F = {}
F['q_dof'] = T['q_dof'][idx].astype(float); F['dnll'] = np.nan_to_num(T['dnll'][idx].astype(float)); F['kin'] = T['kin'][idx].astype(float); F['1-p'] = 1 - T['p'][idx].astype(float)
F['photons'] = T['photons'][idx].astype(float)
dp = np.hypot(x - T['x_prior'][idx], y - T['y_prior'][idx]) * PX; F['dprior'] = dp; F['dprior_z'] = dp / (0.5 * (T['sx'][idx] + T['sy'][idx]) * PX)
F['moderr'] = np.abs(I / N - e).max(1); F['chi2_I'] = ((I - N * e) ** 2 / (np.maximum(I, 0) + 4)).sum(1)
for o in (0, 1):
    s = slice(3 * o, 3 * o + 3); b1 = rel[:, s]; b2 = rel[:, s] * mk[:, s] * np.cos(th[:, s])
    a11 = (b1 * b1).sum(1); a12 = (b1 * b2).sum(1); a22 = (b2 * b2).sum(1); r1 = (b1 * I[:, s]).sum(1); r2 = (b2 * I[:, s]).sum(1); det = a11 * a22 - a12 ** 2
    A = (r1 * a22 - r2 * a12) / det; Ar = (a11 * r2 - a12 * r1) / det; F[f'1-rho_{o}'] = 1 - np.clip(Ar / np.maximum(A, 1e-6), -1, 3)
F['1-rho_min'] = np.maximum(F['1-rho_0'], F['1-rho_1'])
names = list(F); X = np.c_[tuple(F[n] for n in names)]; X = np.nan_to_num(X)


def auc(s, pos, neg):
    from scipy.stats import rankdata
    v = np.r_[s[pos], s[neg]]; r = rankdata(v); n1, n0 = pos.sum(), neg.sum()
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


combo = np.full(len(idx), np.nan)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xl = np.c_[X, np.log(np.maximum(F['photons'], 1)), np.log(np.maximum(F['q_dof'], 1e-3))]
    for h in (0, 1):
        tr = (rul % 2 != h) & (cls < 2); te = rul % 2 == h; sc = StandardScaler().fit(Xl[tr])
        lr = LogisticRegression(max_iter=500, class_weight='balanced').fit(sc.transform(Xl[tr]), cls[tr]); combo[te] = lr.decision_function(sc.transform(Xl[te]))
    F['combo'] = combo; names.append('combo')
except Exception as ex: print('no combination (sklearn):', ex)
single = cls == 0; merged = cls == 1; R = dict(run=run, sel=sel, n_rows=int(len(idx)), n_single=int(single.sum()), n_merged=int(merged.sum()), n_other=int((cls == 2).sum()),
                                               merged_fraction=float(merged.mean()), mixed_fraction_f2_ge_0p05=float((f2 >= 0.05).mean()),
                                               merged_by_axis={ax: int((merged & (axis == ax)).sum()) for ax in ('ky', 'kx', 'diag')}, auc={}, yield_curve={})
L = [f'## Merged versus single rows: {run} ({sel} rows, cycle-aligned, matched to GT)', '',
     f'rows {len(idx)}: single (f2 < 0.05, fully on) {single.sum()}, merged (f2 >= 0.2) {merged.sum()} = {100 * merged.mean():.2f} % '
     f'(along ky {R["merged_by_axis"]["ky"]}, along kx {R["merged_by_axis"]["kx"]}, diagonal {R["merged_by_axis"]["diag"]}), other {(cls == 2).sum()}; rows with f2 >= 0.05: {100 * (f2 >= 0.05).mean():.2f} %', '',
     '| statistic | median single | median merged | AUC all merged | AUC pair along ky | along kx | diagonal | merged % left at yield 90 % | 80 % | 70 % | single rows kept at 90 / 80 / 70 % |', '|' + '---|' * 11]
for n in names:
    s = np.nan_to_num(F[n]); A = dict(all=auc(s, merged, single), **{ax: auc(s, merged & (axis == ax), single) for ax in ('ky', 'kx', 'diag')}); R['auc'][n] = A
    yc = {}
    for yv in (0.9, 0.8, 0.7):
        thr = np.quantile(s, yv); keep = s <= thr
        if keep.mean() > yv + 0.03 or keep.mean() < yv - 0.03: keep = np.argsort(np.argsort(s, kind='stable')) < yv * len(s)       # ties (flags)
        yc[str(yv)] = dict(merged_left=float(merged[keep].mean()), mixed_left=float((f2[keep] >= 0.05).mean()), single_kept=float(keep[single].mean()), merged_removed=float(1 - keep[merged].mean()),
                           by_axis={ax: float(1 - keep[merged & (axis == ax)].mean()) for ax in ('ky', 'kx', 'diag')})
    R['yield_curve'][n] = yc
    L.append('| ' + ' | '.join([n, '%.3g' % np.median(s[single]), '%.3g' % np.median(s[merged]), '%.3f' % A['all'], '%.3f' % A['ky'], '%.3f' % A['kx'], '%.3f' % A['diag']] +
                               ['%.2f' % (100 * yc[v]['merged_left']) for v in ('0.9', '0.8', '0.7')] + [' / '.join('%.3f' % yc[v]['single_kept'] for v in ('0.9', '0.8', '0.7'))]) + ' |')
L += ['', 'AUC = probability that a merged row has the larger statistic than a single row (0.5 = no separation). Yield = fraction of ALL rows kept when the rows with the largest ' +
      'statistic are cut; "merged %% left" = merged rows among the kept rows (uncut: %.2f %%).' % (100 * merged.mean())]
if 'combo' in F:
    yc = R['yield_curve']['combo']
    L += ['', 'combination: fraction of the merged rows REMOVED at yield 90 / 80 / 70 %: ' + ' / '.join('%.2f' % yc[v]['merged_removed'] for v in ('0.9', '0.8', '0.7')) +
          '; by pair axis at 80 %: ' + ', '.join(f'{ax} {yc["0.8"]["by_axis"][ax]:.2f}' for ax in ('ky', 'kx', 'diag'))]
open(out + '.md', 'w').write('\n'.join(L) + '\n'); json.dump(R, open(out + '.json', 'w'), indent=1)
np.savez_compressed(out + '_scores.npz', row=idx, cls=cls, f2=f2, axis=axis, ruler=rul, **{n.replace('-', 'm'): F[n] for n in names})
print('\n'.join(L))
