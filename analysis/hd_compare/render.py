"""Renders for the high-density sim comparison: row 0 raw frames (density), row 1 full FOV, rows 2-4 zooms at filament
crossings picked on the GT (analysis/figures/zoompick.py pick_crossings; same boxes for all methods).
Columns: GT (target positions, W3 windows, >= 200 label ph), SMLM (paper pipeline), SMLM tuned, SF (paper pipeline), DECODE
(6-frame sums, trained on the sim's vectorial PSF), SIMCODE (sfhd-ndi+smlm, sign fixed), ours p1, ours p2.
Every method: its own full output on the W3 windows inside the crop, global offset of score.py removed (read from
<score out>/score_<name>.json, so run score.py on the same spec first).
Same json spec as analysis/hd_compare/score.py, plus the optional key
  "box_sim": "<sim dir>"   GT used to pick the zoom boxes (default: the spec's "sim"). The paper figures used the m20qh sim dir
                           for both m20qh and m25qh (same filament geometry, seed 7), so that both movies get the same boxes.
Methods 'SIMCODE-net' and 'DECODE per-frame' are scored but not rendered; methods whose path does not exist are skipped.
usage: python analysis/hd_compare/render.py <spec.json> [out.png]   (default <simcode2 out>/plots/hd_compare_<name[:3]>.png)"""
import os, sys, json
import numpy as np, tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

_H = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
    print(__doc__); sys.exit(0)
SPEC_FILE = sys.argv[1]
OUT_PNG = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = [sys.argv[0], SPEC_FILE]                       # score.py reads the spec from argv[1] at import
sys.path.insert(0, _H)
sys.path.insert(0, os.path.join(_H, '..', 'figures'))
import score as S          # analysis/hd_compare/score.py (imports analysis/baselines/compare.py)
C = S.C
import zoompick as ZP      # analysis/figures/zoompick.py
from simcode2 import config

nm = S.nm
PX = S.PX; X0, X1 = S.X0, S.X1
REP = json.load(open(f'{S.HC}/score_{nm}.json'))
G = dict(np.load(f'{S.sd}/sim_gt.npz')); NF = int(G['nf'])
GW = C.gt_windows(G, np.arange(0, NF - 5, 3))
gm = (GW[:, 5] > 0) & (GW[:, 4] >= 200)
pts = {'GT': (GW[gm, 2], GW[gm, 3])}
for name, kind, path in S.METHODS:
    if name in ('SIMCODE-net', 'DECODE per-frame'): continue
    if not os.path.exists(path) or name not in REP:
        print('skip', name, path); continue
    M = S.load(kind, path, NF)
    w3 = (M['w'] % 3 == 0) & (M['w'] <= NF - 6)
    off = REP[name]['offset_px']
    pts[name] = (M['x'][w3] - off[0], M['y'][w3] - off[1])
cols = list(pts)
# boxes: pick_crossings(n=10, size 8 px) on the m20qh GT targets gives candidates; #3 (61.5, 67.0), #0 (103.7, 87.3) and #2
# (108.5, 59.4) are the real filament crossings (checked by eye); same filament geometry (seed 7) in
# m20qh and m25qh, so the same boxes are used for both movies and all methods.
_bs = S.SPEC.get('box_sim', S.sd)
_c = ZP.pick_crossings(*[v for v in (lambda G_: (lambda W_: (W_[(W_[:, 5] > 0) & (W_[:, 4] >= 200), 2], W_[(W_[:, 5] > 0) & (W_[:, 4] >= 200), 3]))(
    C.gt_windows(G_, np.arange(0, int(G_['nf']) - 5, 3))))(dict(np.load(f'{_bs}/sim_gt.npz')))], lim=(X0, X1, X0, X1), n=10, size=8.0)
Z = [_c[3], _c[0], _c[2]]


def render(x, y, lim, pix_nm, sig_px):
    a, b, c, d = lim; z = PX / pix_nm
    H, _, _ = np.histogram2d(y, x, bins=[int((d - c) * z), int((b - a) * z)], range=[[c, d], [a, b]])
    return gaussian_filter(H, sig_px)


def show(ax, im, title=None, q=99.7):
    v = np.percentile(im[im > 0], q) if (im > 0).any() else 1
    ax.imshow(im, cmap='inferno', vmin=0, vmax=v, origin='upper', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=9)


NC = len(cols)
fig, AX = plt.subplots(2 + len(Z), NC, figsize=(NC * 3.0, (2 + len(Z)) * 3.1))
for ax in AX[0, 6:]: ax.axis('off')
# row 0: raw frames
mov = tifffile.TiffFile(f'{S.sd}/sim_movie.tif')
cam = np.load(config.model('insitu/camera_maps_0.5nM.npz'))
f0 = 2400
fr = np.stack([(mov.pages[f0 + i].asarray().astype(np.float32) - cam['offset_adu']) * float(cam['gain']) for i in range(6)])
sl = slice(int(X0), int(X1))
on = (G['t0'] < f0 + 1) & (G['t1'] > f0) & (G['x'] >= X0) & (G['x'] < X1) & (G['y'] >= X0) & (G['y'] < X1)
onw = (G['t0'] < f0 + 6) & (G['t1'] > f0) & (G['x'] >= X0) & (G['x'] < X1) & (G['y'] >= X0) & (G['y'] < X1)
for i in range(3):
    ax = AX[0, i]; im = fr[i][sl, sl]
    ax.imshow(im, cmap='gray', vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.8)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'raw frame {f0 + i} (photons)', fontsize=9)
ax = AX[0, 3]; im = fr.sum(0)[sl, sl]
ax.imshow(im, cmap='gray', vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.8)); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'6-frame sum {f0}-{f0 + 5}', fontsize=9)
ax = AX[0, 4]; ax.imshow(im, cmap='gray', vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.8)); ax.set_xticks([]); ax.set_yticks([])
ax.plot(G['x'][onw] - X0, G['y'][onw] - X0, 'o', mfc='none', mec='c', ms=3, mew=0.5)
ax.set_xlim(-0.5, X1 - X0 - 0.5); ax.set_ylim(X1 - X0 - 0.5, -0.5)
ax.set_title(f'+ GT emitters on in window ({onw.sum()})', fontsize=9)
# zoomed raw frame around the first crossing
a, b, c, d = Z[0]; m = 10
ys, xs = slice(int(c) - m, int(d) + m), slice(int(a) - m, int(b) + m)
ax = AX[0, 5]; im = fr[0][ys, xs]
ax.imshow(im, cmap='gray', vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.8), extent=(xs.start, xs.stop, ys.stop, ys.start))
ax.plot(G['x'][on], G['y'][on], 'o', mfc='none', mec='c', ms=5, mew=0.7)
ax.set_xlim(xs.start, xs.stop); ax.set_ylim(ys.stop, ys.start); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'frame {f0} zoom (GT on: cyan)', fontsize=9)
# row 1: full FOV
lim = (X0, X1, X0, X1)
lab = {'GT': 'GT targets', 'SMLM (paper pipeline)': 'SMLM (paper pipeline)', 'SMLM tuned': 'SMLM tuned (ROI 6, spacing 4)',
       'SF (paper pipeline)': 'SF (paper pipeline)', 'DECODE': 'DECODE (6-frame sums, vect. PSF)',
       'SIMCODE': 'SIMCODE paper (sign fixed)', 'ours p1': 'ours pass 1', 'ours p2': 'ours pass 2 (p2L)'}
for j, n in enumerate(cols):
    x, y = pts[n]
    show(AX[1, j], render(x, y, lim, 20.0, 1.0), f'{lab.get(n, n)}\nN = {len(x):,}')
    for k, (a, b, c, d) in enumerate(Z):
        AX[1, j].add_patch(Rectangle(((a - X0) * PX / 20, (c - X0) * PX / 20), (b - a) * PX / 20, (d - c) * PX / 20, fill=False, ec='c', lw=0.8))
        if j == 0: AX[1, j].text((a - X0) * PX / 20, (c - X0) * PX / 20 - 8, str(k + 1), color='c', fontsize=8)
for i, (a, b, c, d) in enumerate(Z):
    for j, n in enumerate(cols):
        x, y = pts[n]; mm = (x >= a) & (x < b) & (y >= c) & (y < d)
        show(AX[2 + i, j], render(x[mm], y[mm], (a, b, c, d), 5.0, 1.5), f'zoom {i + 1}: N = {mm.sum():,}' if j else f'zoom {i + 1} ({(b - a) * PX / 1000:.2f} um)')
d = REP['gt']
fig.suptitle(f'{nm}: {d["on_per_frame"]:.0f} emitters on per frame in the {(X1 - X0) * PX / 1000:.1f} um crop '
             f'({d["on_per_frame_per_um2"]:.2f}/um2), {d["on_per_window"]:.0f} per 6-frame window; target window photons median '
             f'{d["target_phot_q"][2]:.0f} (label units). Renders: W3 windows, 20 nm px (full) / 5 nm px (zooms)', fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.98))
out = OUT_PNG or os.path.join(config.out_dir('plots'), f'hd_compare_{nm[:3]}.png')
plt.savefig(out, dpi=110); print(out, Z)
