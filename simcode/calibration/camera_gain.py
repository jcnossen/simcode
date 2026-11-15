"""
Camera gain and read noise from the movie (no dark / flat frames): robust variance-vs-level fit of same-pattern-step frame
differences (frames 6 apart), per 2-ADU bin of the per-pixel median level (101-139 ADU). The variance is a 5 %-each-side
trimmed variance, corrected to sigma^2 of a normal, which rejects blinking emitters; the MAD variance is printed alongside.
The slope gives the gain (0.5 nM: 0.552 e/ADU, which confirms the 0.56 of the SIMCODE paper); the intercept at the offset
gives the read noise (~0.9-1.1 e). Ported unchanged (the development robust-gain script) apart from the CLI.
usage: python -m simcode.calibration.camera_gain [--movie <tif>] [--nframes 12000]
"""
import argparse
import numpy as np, tifffile
from simcode.calibration import MOVIE_05, data

ap = argparse.ArgumentParser()
ap.add_argument('--movie', default=MOVIE_05, help='movie (absolute or relative to the data root)')
ap.add_argument('--nframes', type=int, default=12000)
ap.add_argument('--offsets', default='99.3,100.0', help='offsets [ADU] at which the read noise is reported')
a = ap.parse_args()
t = tifffile.TiffFile(data(a.movie))
mov = np.stack([t.pages[i].asarray() for i in range(a.nframes)]).astype(np.float32)
d = (mov[6:] - mov[:-6]).reshape(len(mov) - 6, -1)
lvl = np.median(mov, 0).ravel()
corr = 1 - 2 * 1.6449 * 0.10314 / 0.9   # variance of a 5%-each-side trimmed normal / sigma^2
X = []; Y = []; Ym = []
for lo in np.arange(101, 140, 2):
    s = (lvl >= lo) & (lvl < lo + 2)
    if s.sum() < 30: continue
    x = d[:, s].ravel(); q = np.percentile(x, [5, 95]); xt = x[(x > q[0]) & (x < q[1])]
    v = xt.var() / corr / 2
    mad = (np.percentile(np.abs(x - np.median(x)), 50) * 1.4826) ** 2 / 2
    X.append(lvl[s].mean()); Y.append(v); Ym.append(mad)
    print(f'level {lvl[s].mean():6.1f} ADU  npx {s.sum():5d}  var(trim) {v:6.2f}  var(MAD) {mad:6.2f}')
X = np.array(X); Y = np.array(Y)
p = np.polyfit(X, Y, 1); print('trimmed: slope %.3f -> gain %.3f e/ADU; var=0 at level %.1f' % (p[0], 1 / p[0], -p[1] / p[0]))
for off in [float(v) for v in a.offsets.split(',')]:
    rn2 = np.polyval(p, off); print(f'  offset {off}: rn^2 {rn2:.2f} ADU^2 -> rn {np.sqrt(max(rn2, 0)) * (1 / p[0]):.2f} e')
