"""0.5 nM, all 90000 frames, each method with its own DME drift (dme_full05.py, fpb 50): SIMCODE sfhd-ndi vs ours (disjoint
windows, p >= 0.63); zooms at automatically detected microtubule crossings (zoompick.py). -> <out>/plots/recon_full_0.5nM_dme_cross.png"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode2 import config
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import smoke_metrics as SM, zoompick as ZP
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run', default='full_0.5nM_sc5nb_bright_real', help='full-movie run dir (pass2_est_*.npz); relative: under the out root')
ap.add_argument('--simcode', default=SM.SIMCODE_05, help="SIMCODE's published sfhd-ndi.hdf5; relative: under the data root")
ap.add_argument('--out', default='', help='png (default <out>/plots/recon_full_0.5nM_dme_cross.png)')
ap.add_argument('--drift_dir', default='drift', help='drift_ours_fpb50.npy / drift_simcode_fpb50.npy (dme_full05.py); relative: under the out root')
a = ap.parse_args()
RUN = os.path.join(config.root('out'), a.run); SIMH5 = config.resolve(a.simcode, 'data')
OUTPNG = a.out or os.path.join(config.out_dir('plots'), 'recon_full_0.5nM_dme_cross.png')
D = os.path.join(config.root('out'), a.drift_dir) + '/'
A = SM.load_ours(RUN, 'est'); S = SM.load_simcode(SIMH5, 0, 90000)
ins = (S['x'] >= 5) & (S['x'] < 115) & (S['y'] >= 5) & (S['y'] < 115); S = {k: v[ins] for k, v in S.items()}
S['x'] = S['x'] - 2.0 / 108.33; S['y'] = S['y'] + 0.5 / 108.33
dO = np.load(D + 'drift_ours_fpb50.npy'); dS = np.load(D + 'drift_simcode_fpb50.npy')
fa = (A['wstart'] + 3).astype(int).clip(0, 89999); fs = (S['frame'] + 3).astype(int).clip(0, 89999)
dj = (A['wstart'] % 6 == 0) & (A['p'] >= 0.63)
ox, oy = A['x_hy0'][dj] - dO[fa[dj], 0], A['y_hy0'][dj] - dO[fa[dj], 1]
sets = [(f'SIMCODE sfhd-ndi (paper), own DME drift, n={len(S["x"])}', S['x'] - dS[fs, 0], S['y'] - dS[fs, 1]),
        (f'ours, disjoint windows p>=0.63, our DME drift, n={dj.sum()}', ox, oy)]
Z = ZP.pick_crossings(ox, oy, n=3, size=8.0)
print('zooms', [tuple(round(v, 1) for v in z) for z in Z])
def render(x, y, lim, z, blur):
    a, b, c, d = lim
    H, _, _ = np.histogram2d(y, x, bins=[int((d - c) * z), int((b - a) * z)], range=[[c, d], [a, b]]); return gaussian_filter(H, blur)
fig, axs = plt.subplots(2, 4, figsize=(24, 12.6))
cols = ['c', 'lime', 'w']
for j, (t, x, y) in enumerate(sets):
    im = render(x, y, (5, 115, 5, 115), 10, 1.0); ax = axs[j, 0]
    ax.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.6), extent=(5, 115, 115, 5)); ax.set_title(t, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for k, (a, b, c, d) in enumerate(Z):
        ax.add_patch(plt.Rectangle((a, c), b - a, d - c, fill=False, ec=cols[k], lw=1.5))
        im = render(x, y, (a, b, c, d), 80, 1.3); ax2 = axs[j, 1 + k]
        ax2.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.7), extent=(a, b, d, c))
        ax2.set_title(f'crossing {k + 1}: {(b - a) * 0.10833:.2f} µm, 1.35 nm px', fontsize=10, color='k'); ax2.set_xticks([]); ax2.set_yticks([])
        for s in ax2.spines.values(): s.set_edgecolor(cols[k]); s.set_linewidth(3)
plt.suptitle('0.5 nM, all 90000 frames, DME drift correction; zooms at automatically detected microtubule crossings', fontsize=13)
plt.tight_layout(); plt.savefig(OUTPNG, dpi=65); print('ok')
