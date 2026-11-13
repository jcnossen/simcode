"""R24: merged-row estimator from the per-pair results of r24_rul.py (per_pair_<tag>.npz), calibrated on simulations.
The K = 2 fit along the pair axis absorbs merged rows (rows between the two sites) into its sd, so 'observed - fitted' in the midpoint band is ~0
whatever the merged content. Merged rows are displaced ALONG the pair axis only; the across spread is not affected. Estimator per resolved pair:
  excess_c = observed fraction of the window rows with |t - m| <= d / 4  -  the fraction predicted by two Gaussians at -+d / 2 with the fitted
             weights and background but with sd = the ACROSS sd of that pair (truncated on the window +-d0);
  along2 - across2 = fitted along sd^2 - across sd^2 (nm^2), the same information as a variance.
Mean over the resolved common pairs, bootstrap over structures. On a simulation the true merged fraction of the table is known (r24_simeval:
'mixed %'), which calibrates excess_c -> merged fraction.
usage: python r24_midc.py <per_pair npz> <d0> [method ...]"""
import sys, numpy as np
from scipy.special import ndtr
P = np.load(sys.argv[1]); d0 = float(sys.argv[2]); WA = d0
ms = sys.argv[3:] or sorted({k[:-len('_mid_obs')] for k in P.files if k.endswith('_mid_obs')})
unit = P['unit']; rng = np.random.default_rng(1)
print('| method | resolved common pairs | observed mid fraction | predicted with the across sd | excess_c [95 % CI] | along sd^2 - across sd^2, nm^2 [95 % CI] | fit-based excess |\n|---|---|---|---|---|---|---|')
for m in ms:
    g = (P['common'] > 0) & (P[m + '_resolved'] > 0) & np.isfinite(P[m + '_sd_across'])
    d = P[m + '_d'][g]; s = P[m + '_sd_across'][g]; w1 = P[m + '_wmin'][g]; wb = np.nan_to_num(P[m + '_wb'][g]); h = d / 4; pe = 0
    for w_, mu in ((w1, -d / 2), (1 - w1, d / 2)):
        z = np.maximum(ndtr((WA - mu) / s) - ndtr((-WA - mu) / s), 1e-12); pe = pe + w_ * (ndtr((h - mu) / s) - ndtr((-h - mu) / s)) / z
    ex = P[m + '_mid_obs'][g] - ((1 - wb) * pe + wb * (2 * h) / (2 * WA)); dv = P[m + '_s2'][g] ** 2 - s ** 2; u = unit[g]; uu = np.unique(u)
    bs = []
    for b in range(300):
        c = np.bincount(rng.choice(uu, len(uu)), minlength=u.max() + 1)[u]; bs.append((np.average(ex, weights=c), np.average(dv, weights=c)))
    bs = np.array(bs); ci = np.percentile(bs, [2.5, 97.5], 0)
    print(f"| {m} | {g.sum()} | {P[m + '_mid_obs'][g].mean():.3f} | {(P[m + '_mid_obs'][g] - ex).mean():.3f} | {ex.mean():+.3f} [{ci[0, 0]:+.3f}, {ci[1, 0]:+.3f}] | {dv.mean():+.2f} [{ci[0, 1]:+.2f}, {ci[1, 1]:+.2f}] | {P[m + '_mid_excess'][g].mean():+.3f} |")
