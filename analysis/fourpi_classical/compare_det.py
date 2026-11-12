"""Classical wavelet detector (full_cl.py, as the published code) vs the network's DoG candidates (out/4pi/patches/NPC/
meta_<file>.npz, v3 pass 1 before the abort) on the NPC files that have both: counts per window, overlap at 1.5 / 3 px,
how many kept classical rows had a DoG candidate, photons of the unmatched candidates.
Usage: python analysis/fourpi_classical/compare_det.py
"""
import glob, json, os, re
import numpy as np
from scipy.spatial import cKDTree
REPO = '/home/jc/simcode2'


def hits(w1, x1, y1, w2, x2, y2, r):
    h = np.zeros(len(w1), bool)
    o2 = np.argsort(w2, kind='stable'); ws = w2[o2]
    for w in np.unique(w1):
        i = np.nonzero(w1 == w)[0]; j = o2[np.searchsorted(ws, w):np.searchsorted(ws, w, 'right')]
        if len(j): h[i] = cKDTree(np.c_[x2[j], y2[j]]).query(np.c_[x1[i], y1[i]])[0] <= r
    return h


def main():
    res = []
    for fm in sorted(glob.glob(f'{REPO}/out/4pi/patches/NPC/meta_*.npz')):
        fi = int(re.findall(r'meta_(\d+)', fm)[0])
        fc = f'{REPO}/out/4pi/classical/real/cl_NPC_{fi:03d}_wv.npz'
        if not os.path.exists(fc): continue
        C = np.load(fc); D = np.load(fm)
        W = int(min(C['w'].max(), D['w'].max())) + 1
        a = C['w'] < W; b = D['w'] < W
        r = dict(file=fi, windows=W, per_window_wavelet=float(a.sum() / W), per_window_dog=float(b.sum() / W))
        for rr in (1.5, 3.0):
            h1 = hits(C['w'][a], C['cx'][a], C['cy'][a], D['w'][b], D['cx'][b], D['cy'][b], rr)
            h2 = hits(D['w'][b], D['cx'][b], D['cy'][b], C['w'][a], C['cx'][a], C['cy'][a], rr)
            r[f'wavelet_with_dog_{rr}'] = float(h1.mean()); r[f'dog_with_wavelet_{rr}'] = float(h2.mean())
            if rr == 1.5:
                k = C['keep'][a]
                r['kept_with_dog'] = float(h1[k].mean()); r['unmatched_kept_frac'] = float(k[~h1].mean())
                r['N_median_matched'] = float(np.median(C['N'][a][h1])); r['N_median_unmatched'] = float(np.median(C['N'][a][~h1]))
        res.append(r)
    json.dump(res, open(f'{REPO}/out/4pi/classical/compare_detection.json', 'w'), indent=1)
    L = ['| NPC file | windows | cand./window wavelet / DoG | wavelet cand. with a DoG cand. (1.5 px) | DoG cand. with a wavelet cand. (1.5 px) | kept classical rows with a DoG cand. | unmatched wavelet cand. kept | median N matched / unmatched |',
         '|---|---|---|---|---|---|---|---|']
    for r in res:
        L.append(f"| {r['file']:03d} | {r['windows']} | {r['per_window_wavelet']:.1f} / {r['per_window_dog']:.1f} | {r['wavelet_with_dog_1.5']:.3f} | "
                 f"{r['dog_with_wavelet_1.5']:.3f} | {r['kept_with_dog']:.3f} | {r['unmatched_kept_frac']:.3f} | {r['N_median_matched']:.0f} / {r['N_median_unmatched']:.0f} |")
    open(f'{REPO}/out/4pi/classical/compare_detection.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
