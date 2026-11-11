"""Real-data precision check (GT-free): adjacent-window localization pairs of the same molecule (DAFL, as in the
4Pi-SIMFLUX paper: sigma = robust sd of the difference / sqrt(2)), for the network table and the classical tables of
the same file, and the ratio of that empirical precision to the reported sigma (calibration).
Pairs: window w and w + 1, nearest neighbour within r_link px, both with sigma_xy < 0.15 px; only unambiguous pairs
(no second candidate within 2 r_link). Bins by the network photon count.
Usage: python analysis/fourpi/dafl4pi.py --cell MT --file 5 --model v2
"""
import argparse, json, os
import numpy as np
from scipy.spatial import cKDTree

REPO = '/home/jc/simcode2'
PX = 128.0


def pairs(w, x, y, r=0.35):
    out_i, out_j = [], []
    order = np.argsort(w, kind='stable'); ws = w[order]
    starts = np.searchsorted(ws, np.arange(ws.max() + 2))
    for wv in range(ws.max()):
        a = order[starts[wv]:starts[wv + 1]]; b = order[starts[wv + 1]:starts[wv + 2]]
        if len(a) == 0 or len(b) == 0: continue
        t = cKDTree(np.c_[x[b], y[b]])
        d, k = t.query(np.c_[x[a], y[a]], k=2, distance_upper_bound=2 * r)
        ok = (d[:, 0] < r) & ~(d[:, 1] < 2 * r)
        out_i.append(a[ok]); out_j.append(b[k[ok, 0]])
    return np.concatenate(out_i), np.concatenate(out_j)


def rsd(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def analyse(name, w, x, y, sx, sy, N, z=None, sz=None, edges=(1000, 2000, 4000, 8000, 20000)):
    i, j = pairs(w, x, y)
    res = dict(name=name, n_pairs=int(len(i)), bins=[])
    Nm = 0.5 * (N[i] + N[j])
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (Nm >= lo) & (Nm < hi)
        if s.sum() < 100: continue
        b = dict(lo=lo, hi=hi, n=int(s.sum()))
        for k, (v, sv) in dict(x=(x, sx), y=(y, sy), **({'z': (z, sz)} if z is not None else {})).items():
            sc = PX if k != 'z' else 1000.0
            d = (v[i] - v[j])[s] * sc
            emp = rsd(d) / np.sqrt(2)
            rep = float(np.sqrt(np.median(0.5 * (sv[i] ** 2 + sv[j] ** 2)[s]))) * sc
            b[k] = dict(dafl_nm=emp, reported_nm=rep, ratio=emp / rep if rep > 0 else None)
        res['bins'].append(b)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell', default='MT'); p.add_argument('--file', type=int, default=5); p.add_argument('--model', default='v2')
    a = p.parse_args()
    R = []
    D = np.load(f'{REPO}/out/4pi/net/locs_{a.cell}_{a.file:03d}_{a.model}.npz')
    for mode, sel in (('net kin', D['on']), ('net free (lateral = prior)', ~D['on'])):
        s = sel & (D['sx'] < 0.15) & (D['sy'] < 0.15) & (D['N'] > 300)
        R.append(analyse(mode, D['w'][s], D['x'][s], D['y'][s], D['sx'][s], D['sy'][s], D['N'][s], D['z'][s], D['sz'][s]))
    s = (D['sx'] < 0.15) & (D['sy'] < 0.15) & (D['N'] > 300)
    R.append(analyse('net all, z', D['w'][s], D['x'][s], D['y'][s], D['sx'][s], D['sy'][s], D['N'][s], D['z'][s], D['sz'][s]))
    s = s & (D['szprior'] < 0.2)
    R.append(analyse('net prior z (astig, Phi-free)', D['w'][s], D['x'][s], D['y'][s], D['sx'][s], D['sy'][s], D['N'][s], D['zprior'][s], D['szprior'][s]))
    fp1 = f'{REPO}/out/4pi/smlm/p1/p1_{a.cell}_{a.file:03d}.npz'
    if os.path.exists(fp1):
        C = np.load(fp1, allow_pickle=True, mmap_mode='r')
        g, cr = np.asarray(C['gfit']), np.asarray(C['gcrlb'])
        s = (cr[:, 0] < 0.15) & (cr[:, 1] < 0.15) & (g[:, 2] > 300)
        R.append(analyse('classical pass 1 (Gaussian, summed), unfiltered', np.asarray(C['w'])[s], np.asarray(C['x'])[s], np.asarray(C['y'])[s],
                         cr[s, 0], cr[s, 1], g[s, 2] * 1.35))
        # standard rejections: Pearson chi2 <= 4 x ROI^2 (ROI 15), no parameter at a bound (sigma at 0.6 / 4.5 px, centre
        # within 1 px of the ROI edge), N > 300, CRLB < 0.15 px
        c2 = np.asarray(C['gchi2'])
        q = s & (c2 <= 4 * 15 * 15) & (g[:, 4] > 0.61) & (g[:, 4] < 4.49) & (g[:, 5] > 0.61) & (g[:, 5] < 4.49) & (np.abs(g[:, :2]) < 6).all(1)
        R.append(analyse('classical pass 1 (Gaussian, summed), filtered', np.asarray(C['w'])[q], np.asarray(C['x'])[q], np.asarray(C['y'])[q],
                         cr[q, 0], cr[q, 1], g[q, 2] * 1.35))
        R[-1]['kept_frac'] = float(q.sum() / max(s.sum(), 1))
    fp2 = f'{REPO}/out/4pi/smlm/p1/p2_{a.cell}_{a.file:03d}.npz'
    if os.path.exists(fp2):
        C = np.load(fp2, allow_pickle=True)
        pf, pc = C['pfit'], C['pcrlb']
        s = (pc[:, 0] < 0.15) & (pc[:, 1] < 0.15) & (pf[:, 3] > 300)
        R.append(analyse('classical PSF refit (summed 3D PSF), unfiltered', C['w'][s], C['x'][s], C['y'][s], pc[s, 0], pc[s, 1], pf[s, 3], pf[s, 2], pc[s, 2]))
        q = s & (C['pchi2'] <= 4 * 15 * 15) & (np.abs(pf[:, 2]) < 0.88) & (pc[:, 2] < 0.06)
        R.append(analyse('classical PSF refit (summed 3D PSF), filtered', C['w'][q], C['x'][q], C['y'][q], pc[q, 0], pc[q, 1], pf[q, 3], pf[q, 2], pc[q, 2]))
        R[-1]['kept_frac'] = float(q.sum() / max(s.sum(), 1))
    os.makedirs(f'{REPO}/out/4pi/net/check', exist_ok=True)
    json.dump(R, open(f'{REPO}/out/4pi/net/check/dafl_{a.cell}_{a.file:03d}_{a.model}.json', 'w'), indent=1)
    for r in R:
        print(f"\n{r['name']} ({r['n_pairs']} pairs" + (f", kept {r['kept_frac']:.2f} of the unfiltered rows" if 'kept_frac' in r else '') + ')')
        for b in r['bins']:
            s = '  '.join(f"{k}: {b[k]['dafl_nm']:.2f} (rep {b[k]['reported_nm']:.2f}, x{b[k]['ratio']:.2f})" for k in ('x', 'y', 'z') if k in b)
            print(f"  N {b['lo']}-{b['hi']} n {b['n']}: {s}")


if __name__ == '__main__':
    main()
