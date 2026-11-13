"""Nanoruler galleries for the report (2026-11-13 evening): the SAME structures for every method, picked by a rule that does not look at
any method's precision. CPU only, plain matplotlib (nra_lib style).

Structures: 80 nm = the fixed ruler list (out/nanoruler/analysis/sites.json, nra_lib.load_sites); 40 / 20 nm = the structures (connected
sites) of an r24_rul.py pair list. Picking rule ("proper ruler picking"):
  1. complete enough: 80 nm 3-site rulers; 40 nm >= 3 corners; 20 nm >= --min_sites grid sites (default 4);
  2. sampled in EVERY method: each site of the structure has >= --min_locs localizations within 0.4 d0 in every method's table;
  3. isolated: in every method >= --iso (default 0.85) of the localizations in the box lie within 0.6 d0 of a site of the structure
     (no neighbouring structure or unassigned cluster in the box);
  4. among those, the --n structures with the largest minimum (over methods and sites) localization count, then shown sorted by id.
Rendering: 1 nm bins (80 nm: 2 nm) blurred with a fixed Gaussian (--blur nm), each panel scaled to its own maximum (same rule for every
method), inferno; scale bar. One global offset per method (median offset to the nearest site, nra_lib.global_shift).
usage: python analysis/nanoruler/nr_gallery.py <80|40|20> <out.png> name=table.npz ... [--pairs pairs.json] [--n 10] [--box 120] [--blur 1.5]
       [--min_locs 10] [--min_sites 4] [--iso 0.85] [--title t]"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nra_lib as L
from simcode.util import proctitle

a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
proctitle.set_title('nr-gallery')
kind = a[0]; out = a[1]; tabs = [s.rsplit('=', 1) for s in a[2:] if '=' in s and not s.startswith('--')]
d0 = float(kind); n = int(g('--n', 10)); blur = float(g('--blur', 2.5 if kind == '80' else 1.5)); px = 2.0 if kind == '80' else 1.0
box = float(g('--box', 260 if kind == '80' else 120)); min_locs = int(g('--min_locs', 10)); iso_min = float(g('--iso', 0.85))
if kind == '80':
    sites, rulers, _ = L.load_sites(g('--sites', '/home/jc/simcode2/out/nanoruler/analysis/sites.json'))
    sc = np.array([[s['x'], s['y']] for s in sites]); su = np.array([s['ruler'] for s in sites])
    units = [r['id'] for r in rulers if r['n_sites'] == 3]
else:
    D = json.load(open(g('--pairs'))); sc = np.array(D['sites']); su = np.array(D['site_unit'])
    ms = int(g('--min_sites', 3 if kind == '40' else 4)); units = [u for u in np.unique(su) if (su == u).sum() >= ms]
plt = L._style()
M = {}
for nm, fn in tabs:
    T = L.load_table(fn); sh = L.global_shift(T['x_nm'], T['y_nm'], sc, rmax=min(15.0, 0.4 * d0)); M[nm] = (T['x_nm'] - sh[0], T['y_nm'] - sh[1])
score = {}
for u in units:
    S = sc[su == u]; c = 0.5 * (S.min(0) + S.max(0))
    if np.any(np.abs(S - c) > box / 2 - 0.5 * d0): continue          # structure must fit in the box with a margin
    mins = []; ok = True
    for nm, (x, y) in M.items():
        m = (np.abs(x - c[0]) < box / 2) & (np.abs(y - c[1]) < box / 2)
        if m.sum() == 0: ok = False; break
        dd, j = cKDTree(S).query(np.c_[x[m], y[m]])
        cnt = np.bincount(j[dd < 0.4 * d0], minlength=len(S)); iso = np.mean(dd < 0.6 * d0)
        if cnt.min() < min_locs or iso < iso_min: ok = False; break
        mins.append(cnt.min())
    if ok: score[u] = (min(mins), c)
pick = sorted(sorted(score, key=lambda u: -score[u][0])[:n])
print(f'{len(units)} candidate structures, {len(score)} pass the rule, picked {len(pick)}: {pick}')
if not pick: sys.exit('nothing picked')
nb = int(box / px); fig, axs = plt.subplots(len(M), len(pick), figsize=(1.2 * len(pick), 1.2 * len(M) + 0.45), squeeze=False)
for i, (nm, (x, y)) in enumerate(M.items()):
    for j, u in enumerate(pick):
        ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); c = score[u][1]
        for s_ in ax.spines.values(): s_.set_visible(False)
        m = (np.abs(x - c[0]) < box / 2 + 3 * blur) & (np.abs(y - c[1]) < box / 2 + 3 * blur)
        h = gaussian_filter(np.histogram2d(y[m], x[m], bins=nb, range=[[c[1] - box / 2, c[1] + box / 2], [c[0] - box / 2, c[0] + box / 2]])[0], blur / px)
        ax.imshow(h, cmap='inferno', vmin=0, vmax=max(h.max(), 1e-9), origin='lower', interpolation='nearest')
        if j == 0:
            ax.set_title(nm, loc='left', fontsize=8.5); sb = 20.0 if kind != '80' else 80.0
            ax.plot([0.06 * nb, 0.06 * nb + sb / px], [0.06 * nb] * 2, color='w', lw=2)
        if i == len(M) - 1: ax.set_xlabel(str(u), fontsize=7)
t = g('--title', f'{kind} nm')
fig.suptitle(f'{t}: same structures for every method (ids below); picked without looking at precision (sites sampled in every method, '
             f'isolated, most localizations); {box:g} nm boxes, {px:g} nm bins blurred {blur:g} nm, each panel scaled to its maximum; '
             f'scale bar {20 if kind != "80" else 80} nm', x=0.01, ha='left', fontsize=7.5, wrap=True)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86 if len(M) <= 3 else 0.9, bottom=0.04, wspace=0.05, hspace=0.25); fig.savefig(out, dpi=180)
json.dump(dict(pick=[int(u) for u in pick], score={int(u): int(score[u][0]) for u in pick}, tables=dict(tabs), rule=dict(min_locs=min_locs, iso=iso_min, box=box)),
          open(os.path.splitext(out)[0] + '.json', 'w'), indent=1)
print('saved', out)
