"""S1 k_z without errors-in-variables attenuation: link astigmatic localizations of consecutive cycles (< 0.75 px apart,
same emitter ON), replace each localization's (x, y, z) by its track mean (tracks >= MINLEN cycles; z noise / sqrt(n)),
then the pattern regression of s1_pattern on a fine k_z grid (both signs, parabola at the minimum), q and psi(T) re-solved.
Also the same with the per-cycle z, and with track means over tracks split in halves (attenuation check).
usage: python s1_kz.py <gpu> -> out/zimflux/s1/kz_tracks.json, pattern_final.json"""
import sys, json, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/jc/simcode2/analysis/zimflux')
from s1_common import *
import s1_pattern as SP
dev = torch.device(f'cuda:{sys.argv[1]}')
A = dict(np.load(WORK + 'astig.npz')); ncyc = int(np.load(WORK + 'rois.npz')['ncycles'])
g = SP.good_mask(A) & (A['Nk'].sum(1) > 300)
gi = np.nonzero(g)[0]
xa = A['corner'][gi, 1] + 7.5 + A['th'][gi, 0]; ya = A['corner'][gi, 0] + 7.5 + A['th'][gi, 1]; cy = A['cycle'][gi]
order = np.argsort(cy, kind='stable'); gi, xa, ya, cy = gi[order], xa[order], ya[order], cy[order]
tid = np.arange(len(gi)); starts = np.searchsorted(cy, np.arange(ncyc + 1))
for c in range(ncyc - 1):
    a0, a1, b1 = starts[c], starts[c + 1], starts[c + 2]
    if a1 == a0 or b1 == a1: continue
    d, j = cKDTree(np.c_[xa[a0:a1], ya[a0:a1]]).query(np.c_[xa[a1:b1], ya[a1:b1]], distance_upper_bound=0.75)
    ok = np.isfinite(d)
    tid[a1:b1][ok] = tid[a0:a1][j[ok]]
_, inv, cnt = np.unique(tid, return_inverse=True, return_counts=True)
L = cnt[inv]
print('localizations', len(gi), 'in tracks >= 5:', int((L >= 5).sum()), 'median track len', float(np.median(cnt[cnt >= 2])), flush=True)


def track_mean(sel_mask, split=False):
    th = A['th'].copy()
    key = inv.copy()
    if split:   # two halves per track (alternate cycles): independent z noise for the regression check
        key = inv * 2 + (np.arange(len(inv)) % 2)
    for col in range(3):
        s = np.bincount(key, weights=A['th'][gi, col]); n = np.bincount(key)
        th[gi, col] = (s / np.maximum(n, 1))[key]
    return th


res = {}
grid = np.arange(-24, 24.01, 0.25)
for lab, minlen in (('per_cycle', None), ('track_mean_5', 5), ('track_mean_10', 10)):
    B = dict(A)
    sel = gi if minlen is None else gi[L >= minlen]
    if minlen is not None: B['th'] = track_mean(None)
    q0 = json.load(open(WORK + 'pattern.json'))['q']
    xa_, ya_, z_, r_, Nv_, t_ = SP.load(dev, B, sel)
    tbin = (t_ / ncyc * SP.NB).long().clamp(0, SP.NB - 1)
    q = torch.tensor(q0, device=dev)
    rs = np.array([float(SP.inner(q, torch.tensor(k, device=dev), xa_, ya_, z_, r_, Nv_, tbin)[1]) for k in grid])
    i = int(np.argmin(rs)); c = np.polyfit(grid[i - 2:i + 3], rs[i - 2:i + 3], 2); kb = float(-c[1] / (2 * c[0]))
    zs = z_.cpu().numpy()
    res[lab] = {'n': int(len(sel)), 'kz': kb, 'pax_nm': 2 * np.pi / abs(kb) * 1000, 'z_std_nm': float(zs.std() * 1000),
                'res_min': float(rs[i]), 'res_at_minus_kz': float(rs[np.argmin(np.abs(grid + kb))])}
    print(lab, res[lab], flush=True)
json.dump(res, open(WORK + 'kz_tracks.json', 'w'), indent=1)
kb = res['track_mean_10']['kz']
B = dict(A); B['th'] = track_mean(None)
pat = SP.fit_pattern(dev, B, gi[L >= 5], ncyc, json.load(open(WORK + 'pattern.json'))['q'], kz_fixed=kb, verbose=True)
pat['kz_source'] = 'track_mean_10'
json.dump(pat, open(WORK + 'pattern_final.json', 'w'), indent=1)
print('final', {k: pat[k] for k in ('q', 'kz', 'eta', 'm')})
