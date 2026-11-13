"""Camera gain / read noise from the movie: simcode2/insitu/camera_gain.py with the level bins as an argument (this movie's
levels differ from the Zyla movies). Robust variance-vs-level fit of same-step frame differences (frames 6 apart)."""
import argparse, sys, numpy as np, tifffile
ap = argparse.ArgumentParser()
ap.add_argument('--movie', default='/home/jc/simcode2/out/nanoruler/raw/sim4_1_reord.tif')
ap.add_argument('--f0', type=int, default=0); ap.add_argument('--nframes', type=int, default=6000)
ap.add_argument('--bin', type=float, default=1.0)
ap.add_argument('--offsets', default='100.0')
a = ap.parse_args()
t = tifffile.TiffFile(a.movie)
mov = np.stack([t.pages[i].asarray() for i in range(a.f0, a.f0 + a.nframes)]).astype(np.float32)
d = (mov[6:] - mov[:-6]).reshape(len(mov) - 6, -1)
lvl = mov.mean(0).ravel(); lm = np.median(mov, 0).ravel()
print('per-pixel temporal mean level: p1 %.1f p50 %.1f p99 %.1f; median level p50 %.1f; global min ADU %d' % (*np.percentile(lvl, [1, 50, 99]), np.median(lm), mov.min()))
lvl = lm
corr = 1 - 2 * 1.6449 * 0.10314 / 0.9
X = []; Y = []; Ym = []
lo0, hi0 = np.percentile(lvl, [0.5, 99.5])
for lo in np.arange(np.floor(lo0), hi0, a.bin):
    s = (lvl >= lo) & (lvl < lo + a.bin)
    if s.sum() < 200: continue
    x = d[:, s].ravel(); q = np.percentile(x, [5, 95]); xt = x[(x > q[0]) & (x < q[1])]
    v = xt.var() / corr / 2
    mad = (np.percentile(np.abs(x - np.median(x)), 50) * 1.4826) ** 2 / 2
    X.append(lvl[s].mean()); Y.append(v); Ym.append(mad)
    print(f'level {lvl[s].mean():6.1f} ADU  npx {s.sum():6d}  var(trim) {v:6.2f}  var(MAD) {mad:6.2f}')
X = np.array(X); Y = np.array(Y); Ym = np.array(Ym)
for nm, y in (('trimmed', Y), ('MAD', Ym)):
    p = np.polyfit(X, y, 1); print('%s: slope %.3f -> gain %.3f e/ADU; var=0 at level %.1f' % (nm, p[0], 1 / p[0], -p[1] / p[0]))
    for off in [float(v) for v in a.offsets.split(',')]:
        rn2 = np.polyval(p, off); print(f'  offset {off}: rn^2 {rn2:.2f} ADU^2 -> rn {np.sqrt(max(rn2, 0)) * (1 / p[0]):.2f} e')
