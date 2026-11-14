"""20 nM render with zooms at the highest-density locations (not crossings): densest non-overlapping 8 px boxes of ours pass-1 rows,
plus a raw-data column (single frame + 6-frame sum of the busiest window in that box). Development scratch/hdzoom/hdzoom.py.
-> <out>/plots/recon_full_20nM_dense.png (or argv[1]) and <out>/real20/zooms_dense.json (or argv[2]).
usage: python hdzoom.py [out.png] [zooms.json]"""
import os, sys, json, numpy as np
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib20 as L
from simcode import config
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter
from simcode.psf import TiffMovie
PX = L.PX
cols = [('smlm', None), ('sf', None), ('decode_summed', None), ('sc_b', None), ('p1P', 6), ('p2P', 6)]
Dc = {}
for n, g in cols:
    d = L.corrected(n)
    if g: d = L.sub(d, d['f'] % g == 0)
    Dc[n] = d
# density of ours pass-1 rows per window on a 0.25 px grid, box-averaged over 8 px
ref = Dc['p1P']; z = 4; S = 8.0
h = np.histogram2d(ref['y'], ref['x'], bins=[110 * z, 110 * z], range=[[5, 115], [5, 115]])[0]
dens = uniform_filter(h, int(S * z))
Z = []; dd = dens.copy(); m = int(S * z / 2)
# also require not at the edge
dd[:m, :] = dd[-m:, :] = dd[:, :m] = dd[:, -m:] = 0
for _ in range(3):
    iy, ix = np.unravel_index(np.argmax(dd), dd.shape); cy, cx = 5 + (iy + .5) / z, 5 + (ix + .5) / z
    Z.append((cx - S / 2, cx + S / 2, cy - S / 2, cy + S / 2))
    dd[max(0, iy - 2 * m):iy + 2 * m, max(0, ix - 2 * m):ix + 2 * m] = 0
nwin = 120000 / 6
dens_list = [((ref['x'] >= a) & (ref['x'] < b) & (ref['y'] >= c) & (ref['y'] < d)).sum() / nwin / ((b - a) * (d - c) * (PX / 1000) ** 2) for a, b, c, d in Z]
print('zooms', Z, 'ours p1 locs / window / um2:', np.round(dens_list, 2))
# busiest window per zoom (raw-frame column), using raw (not drift-corrected) ours positions
mov = TiffMovie(config.resolve('dnapaint/20nM/2/cover1_500mW_5.5ms_cf660.tif', 'data'))
R = L.load('p1P')
raws = []
for a, b, c, d in Z:
    k = (R['xr'] >= a) & (R['xr'] < b) & (R['yr'] >= c) & (R['yr'] < d) & (R['f'] % 6 == 0)
    ws, cnt = np.unique(R['f'][k], return_counts=True); w = int(ws[np.argmax(cnt)])
    x0, y0 = int(np.floor(a)) - 2, int(np.floor(c)) - 2; Sp = int(S) + 4
    fr = mov.read(w, w + 6, y0, x0, Sp).astype(float)
    sel = k & (R['f'] == w)
    raws.append((fr, w, x0, y0, Sp, R['xr'][sel], R['yr'][sel], cnt.max()))
fig, axs = plt.subplots(4, len(cols) + 1, figsize=(4.0 * (len(cols) + 1), 4.2 * 4))
for j, (n, g) in enumerate(cols):
    x, y = Dc[n]['x'], Dc[n]['y']
    name = L.LBL[n[:2]] + (' (G6 rows)' if g else '') if n[:2] in L.PASS else L.LBL[n]
    for r, (lim, zz, bl) in enumerate([((5, 115, 5, 115), 10, 1.0)] + [(Z[k], 80, 1.4) for k in range(3)]):
        a, b, c, d = lim
        im = gaussian_filter(np.histogram2d(y, x, bins=[int((d - c) * zz), int((b - a) * zz)], range=[[c, d], [a, b]])[0], bl)
        ax = axs[r, j]; ax.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.6), extent=(a, b, d, c))
        nin = ((x >= a) & (x < b) & (y >= c) & (y < d)).sum()
        ax.set_title((f'{name}\n' if r == 0 else '') + (f'{len(x)} locs' if r == 0 else f'zoom {r}: {nin} locs'), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            for q in Z: ax.add_patch(plt.Rectangle((q[0], q[2]), q[1] - q[0], q[3] - q[2], fill=False, ec='c'))
# raw column
ax = axs[0, -1]; ax.axis('off')
ax.text(0.02, 0.5, 'raw data at each zoom:\nsum of the busiest 6-frame window\n(most ours pass-1 locs in the box);\ncyan = ours pass-1 locs in that window\n\n'
        'ours pass-1 density (locs / window / um^2):\n' + '\n'.join(f'  zoom {k+1}: {v:.2f}' for k, v in enumerate(dens_list)), fontsize=10, va='center')
for r, (fr, w, x0, y0, Sp, lx, ly, cmax) in enumerate(raws, start=1):
    ax = axs[r, -1]; s6 = fr.sum(0)
    ax.imshow(s6, cmap='gray', extent=(x0, x0 + Sp, y0 + Sp, y0)); ax.plot(lx, ly, 'c+', ms=8)
    a, b, c, d = Z[r - 1]; ax.add_patch(plt.Rectangle((a, c), b - a, d - c, fill=False, ec='y', ls='--'))
    ax.set_title(f'raw 6-frame sum, frames {w}-{w+5}\n{cmax} ours locs in box', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('real 20 nM movie 2 (120000 frames), each method its own DME v2 drift; zooms = the 3 highest-density 0.87 um boxes (ours pass-1 rows)', fontsize=12, y=0.997)
plt.tight_layout(rect=(0, 0, 1, 0.985))
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.out_dir('plots'), 'recon_full_20nM_dense.png'); plt.savefig(fn, dpi=70); print('->', fn)
json.dump(dict(zooms=Z, dens=dens_list), open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(config.out_dir('real20'), 'zooms_dense.json'), 'w'))
