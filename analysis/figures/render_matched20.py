"""critic 37 #5: matched 20 nM render. All columns G6 rows (window starts % 6 == 0), one sigma definition (mean of the two axes),
each method its own DME v2 drift:
  SIMCODE (b) G6 rows (full output on G6) | ours pass 2 top-N by sigma_cal, N = SIMCODE's G6 count | ours pass 2 top-N, N
  coverage-matched (occupied 20-nm bins = SIMCODE G6's) | ours pass 1 full (phase-calibrated selection, G6)
Rows: full FOV, the 3 dense zooms (zooms_dense.json), the 3 crossing zooms (zooms.json). SIMCODE's sigma is not printed (it is an
uncalibrated CRLB). -> plots/recon_full_20nM_matched.png
(development real20/render_matched.py; zooms: <out>/real20/zooms_dense.json from hdzoom.py, <out>/real20/zooms.json = the crossing zooms of the
development real20/analyze20.py render stage.)  -> <out>/plots/recon_full_20nM_matched.png (or argv[1])"""
import os, sys, json, numpy as np
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib20 as L
from simcode2 import config
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
PX = L.PX
NB = int(110 * PX / 20)
g6 = lambda d: L.sub(d, d['f'] % 6 == 0)
sc = g6(L.corrected('sc_b')); p2 = g6(L.corrected('p2P')); p1 = g6(L.corrected('p1P'))


def occ(d):
    ix = np.clip(((d['x'] - 5) / 110 * NB).astype(int), 0, NB - 1); iy = np.clip(((d['y'] - 5) / 110 * NB).astype(int), 0, NB - 1)
    return iy * NB + ix


def topn(d, n):
    o = np.argsort(d['sig'], kind='stable')[:n]; return L.sub(d, np.sort(o))


N = sc['n']; target = len(np.unique(occ(sc)))
o = np.argsort(p2['sig'], kind='stable'); b = occ(p2)[o]
u, first = np.unique(b, return_index=True); fi = np.sort(first); Ncov = int(fi[target - 1] + 1)
cols = [('SIMCODE (b), G6 rows', sc, False),
        (f'ours pass 2, top-N by sigma_cal\nN = SIMCODE G6 count', topn(p2, N), True),
        (f'ours pass 2, top-N by sigma_cal\nN coverage-matched (20-nm bins = SIMCODE)', topn(p2, Ncov), True),
        ('ours pass 1, full output (phase-cal. sel., G6)', p1, True)]
Zd = json.load(open(os.path.join(config.root('out'), 'real20', 'zooms_dense.json')))['zooms']
Zc = json.load(open(os.path.join(config.root('out'), 'real20', 'zooms.json')))['zooms']
Z = [('dense', z) for z in Zd] + [('crossing', z) for z in Zc]
fig, axs = plt.subplots(1 + len(Z), len(cols), figsize=(4.0 * len(cols), 4.15 * (1 + len(Z))))
for j, (name, d, ours) in enumerate(cols):
    x, y = d['x'], d['y']
    no = len(np.unique(occ(d)))
    for r, (lim, zz, bl, ttl) in enumerate([((5, 115, 5, 115), 10, 1.0, 'full FOV')] + [(z, 80, 1.4, f'{k} zoom ({(z[1] - z[0]) * PX / 1000:.2f} um)') for k, z in Z]):
        a, bb, c, e = lim
        im = gaussian_filter(np.histogram2d(y, x, bins=[int((e - c) * zz), int((bb - a) * zz)], range=[[c, e], [a, bb]])[0], bl)
        ax = axs[r, j]; ax.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.6), extent=(a, bb, e, c))
        nin = ((x >= a) & (x < bb) & (y >= c) & (y < e)).sum()
        head = ''
        if r == 0:
            head = f'{name}\n{len(x):,} locs, {no:,} occupied 20-nm bins' + (f'\nmedian sigma_cal {np.median(d["sig"]) * PX:.1f} nm (mean of axes)' if ours else '\n(sigma not shown: uncalibrated CRLB)') + '\n'
        ax.set_title(head + f'{ttl}: {nin:,} locs', fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            for k, q in Z: ax.add_patch(plt.Rectangle((q[0], q[2]), q[1] - q[0], q[3] - q[2], fill=False, ec='c' if k == 'dense' else 'w'))
plt.suptitle(f'real 20 nM movie 2 (120000 frames), all columns G6 rows, own DME v2 drift; N_SC(G6) = {N:,}, coverage-matched N = {Ncov:,} '
             f'({Ncov / N:.2f}x)\ncyan boxes: densest regions, white: crossings', fontsize=11, y=0.998)
plt.tight_layout(rect=(0, 0, 1, 0.985))
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.out_dir('plots'), 'recon_full_20nM_matched.png'); plt.savefig(fn, dpi=65); print('->', fn)
for name, d, ours in cols:
    print(f'{name.replace(chr(10), " "):70s} N {d["n"]:9,d} occupied 20-nm bins {len(np.unique(occ(d))):,}' + (f' median sigma {np.median(d["sig"]) * PX:.2f} nm' if ours else ''))
