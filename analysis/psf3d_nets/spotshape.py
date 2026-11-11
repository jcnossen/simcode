"""PSF-shape realism check (2026-11-11 psf3d retrain): mean shape statistics of isolated, bright, always-on spots selected by the
SAME pass-1 networks on a real movie chunk and on sim movies. Rows: p >= 0.6, network window photons 1000-4000, (no on-time cut)
no other p >= 0.3 row within 4.5 px in the same window, inside 12..108 px. Per row: 6-frame sum of
(ADU - offset map) x gain over an 11 x 11 patch at the rounded position, minus the head bg (sum over frames); statistics:
E3 = sum(3x3) / sum(11x11), E5 = sum(5x5) / sum(11x11), Mxx / Myy = second moments (px^2, r <= 4 px about the patch centroid).
usage: python spotshape.py name=<movie.tif>:<offset map npz>:<pass1 npz>:<f0> [...]"""
import sys, numpy as np, tifffile
G = 0.56


def run(tif, cam, p1, f0):
    d = np.load(p1); off = np.load(cam)['offset_adu'].astype(np.float32)
    w = d['wstart']; x, y, p, ph, I, bg = d['x'], d['y'], d['p'], d['phot'], d['I'], d['bg']
    keep = (p >= 0.3)
    sel = (p >= 0.6) & (ph >= 1000) & (ph <= 4000) & (x > 12) & (x < 108) & (y > 12) & (y < 108)
    idx = np.nonzero(sel)[0]
    o = np.argsort(w[keep], kind='stable'); wk, xk, yk = w[keep][o], x[keep][o], y[keep][o]
    good = []
    for i in idx:
        a, b = np.searchsorted(wk, w[i]), np.searchsorted(wk, w[i], 'right')
        dd = np.hypot(xk[a:b] - x[i], yk[a:b] - y[i]); dd = dd[dd > 1e-3]
        if len(dd) == 0 or dd.min() > 4.5: good.append(i)
    good = np.array(good[:6000], dtype=int); t = tifffile.TiffFile(tif)
    cache = {}
    def fr(k):
        if k not in cache:
            if len(cache) > 400: cache.clear()
            cache[k] = (t.pages[k].asarray().astype(np.float32) - off) * G
        return cache[k]
    S = []
    for i in good[np.argsort(w[good])]:
        cx, cy = int(round(x[i])), int(round(y[i])); f = int(w[i])
        P = sum(fr(f + j - f0 if False else f + j)[cy - 5:cy + 6, cx - 5:cx + 6] for j in range(6)) - bg[i].sum()
        o_ = np.arange(11) - 5; s = P.sum()
        m = (np.abs(o_) <= 4); Q = P[np.ix_(m, m)]; oo = o_[m]; s4 = Q.sum()
        mx = (Q.sum(0) * oo).sum() / s4; my = (Q.sum(1) * oo).sum() / s4
        S.append([P[4:7, 4:7].sum() / s, P[3:8, 3:8].sum() / s, (Q.sum(0) * (oo - mx) ** 2).sum() / s4, (Q.sum(1) * (oo - my) ** 2).sum() / s4, ph[i]])
    S = np.array(S)
    return len(S), np.median(S, 0), S


if __name__ == '__main__':
    print(f'{"set":14s}{"n":>6s}{"E3":>8s}{"E5":>8s}{"Mxx":>8s}{"Myy":>8s}{"ph":>7s}')
    for a in sys.argv[1:]:
        nm, spec = a.split('=', 1); tif, cam, p1, f0 = spec.split(':')
        n, med, S = run(tif, cam, p1, int(f0))
        se = 1.2533 * S.std(0) / np.sqrt(max(n, 1))
        print(f'{nm:14s}{n:6d}' + ''.join(f'{v:8.3f}' for v in med[:4]) + f'{med[4]:7.0f}   (SE E3 {se[0]:.3f}, Mxx {se[2]:.3f})', flush=True)
