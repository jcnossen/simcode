"""20 nM render with the strict output filter (sigma_cal <= 6 nm per axis), ours vs SIMCODE; dense + crossing zooms.
(development scratch/hdzoom/table.py; zooms: <out>/real20/zooms_dense.json from hdzoom.py, <out>/real20/zooms.json = the crossing zooms of the
development real20/analyze20.py render stage.)  -> <out>/plots/recon_full_20nM_strict.png (or argv[1])"""
import os, sys, json, numpy as np
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib20 as L
from simcode import config
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
PX = L.PX
sc = L.corrected('sc_b')
D = {}
for p in ('p1P', 'p2P'):
    d = L.corrected(p); D[p] = L.sub(d, d['f'] % 6 == 0)
def strict(d, s=6.0):
    return L.sub(d, (d['sx'] * PX <= s) & (d['sy'] * PX <= s))
def topn(d, n):
    o = np.argsort(d['sig'])[:n]; return L.sub(d, np.sort(o))
cols = [('SIMCODE (sign-fixed), full output', sc),
        ('ours pass 1, full (G6)', D['p1P']),
        ('ours pass 1, strict: sigma <= 6 nm/axis', strict(D['p1P'])),
        ('ours pass 2, strict: sigma <= 6 nm/axis', strict(D['p2P'])),
        (f'ours pass 2, best N = SIMCODE count by sigma', topn(D['p2P'], sc['n']))]
Zd = json.load(open(os.path.join(config.root('out'), 'real20', 'zooms_dense.json')))['zooms']
Zc = json.load(open(os.path.join(config.root('out'), 'real20', 'zooms.json')))['zooms']
Z = [('dense', z) for z in Zd] + [('crossing', z) for z in Zc]
fig, axs = plt.subplots(1 + len(Z), len(cols), figsize=(4.0 * len(cols), 4.1 * (1 + len(Z))))
for j, (name, d) in enumerate(cols):
    x, y = d['x'], d['y']
    for r, (lim, zz, bl, ttl) in enumerate([((5, 115, 5, 115), 10, 1.0, 'full FOV')] + [(z, 80, 1.4, f'{k} zoom ({(z[1]-z[0])*PX/1000:.2f} um)') for k, z in Z]):
        a, b, c, e = lim
        im = gaussian_filter(np.histogram2d(y, x, bins=[int((e - c) * zz), int((b - a) * zz)], range=[[c, e], [a, b]])[0], bl)
        ax = axs[r, j]; ax.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.6), extent=(a, b, e, c))
        nin = ((x >= a) & (x < b) & (y >= c) & (y < e)).sum()
        ax.set_title((f'{name}\n{len(x):,} locs, median sigma {np.median(d["sig"])*PX:.1f} nm\n' if r == 0 else '') + f'{ttl}: {nin:,} locs', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            for k, q in Z: ax.add_patch(plt.Rectangle((q[0], q[2]), q[1] - q[0], q[3] - q[2], fill=False, ec='c' if k == 'dense' else 'w'))
plt.suptitle('real 20 nM movie 2 (120000 frames), own DME v2 drift per method; strict filter = calibrated sigma <= 6 nm per axis '
             '(cyan boxes: densest regions, white: crossings)', fontsize=12, y=0.997)
plt.tight_layout(rect=(0, 0, 1, 0.99))
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.out_dir('plots'), 'recon_full_20nM_strict.png'); plt.savefig(fn, dpi=65); print('->', fn)
for name, d in cols: print(f'{name:50s} N {d["n"]:9,d} median sigma {np.median(d["sig"])*PX:.2f} nm')
