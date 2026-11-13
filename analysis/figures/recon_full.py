"""0.5 nM real-data smoke test (all 90000 frames): reconstructions SIMCODE sfhd-ndi vs ours (sc5nb, PM), same frames.
-> <out>/plots/recon_full_0.5nM.png"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode import config
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import smoke_metrics as SM
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run', default='full_0.5nM_sc5nb_bright_real', help='full-movie run dir (pass2_est_*.npz); relative: under the out root')
ap.add_argument('--simcode', default=SM.SIMCODE_05, help="SIMCODE's published sfhd-ndi.hdf5; relative: under the data root")
ap.add_argument('--out', default='', help='png (default <out>/plots/recon_full_0.5nM.png)')
a = ap.parse_args()
RUN = os.path.join(config.root('out'), a.run); SIMH5 = config.resolve(a.simcode, 'data')
OUTPNG = a.out or os.path.join(config.out_dir('plots'), 'recon_full_0.5nM.png')
A = SM.load_ours(RUN, 'est'); A['x'] = A['x_hy0']; A['y'] = A['y_hy0']
S = SM.load_simcode(SIMH5, 0, 90000)
ins = (S['x'] >= 5) & (S['x'] < 115) & (S['y'] >= 5) & (S['y'] < 115)
S = {k: v[ins] for k, v in S.items()}
off = (-2.0 / 108.33, 0.5 / 108.33)            # M0 global offset ours - SIMCODE (px)
S['x'] = S['x'] + off[0]; S['y'] = S['y'] + off[1]
PX = 108.33
def render(x, y, lim, z, blur):
    a, b, c, d = lim
    H, _, _ = np.histogram2d(y, x, bins=[int((d - c) * z), int((b - a) * z)], range=[[c, d], [a, b]])
    return gaussian_filter(H, blur)
q = A.get('q', None)
print('ours keys', list(A.keys()))
sel_all = np.ones(len(A['x']), bool)
sel_400 = (A['wstart'] % 6 == 0) & (A['p'] >= 0.63)
full = (5, 115, 5, 115); zl = (40, 70, 40, 70)
rows = [('SIMCODE sfhd-ndi (paper)', S['x'], S['y']),
        (f'ours, all ({sel_all.sum()})', A['x'], A['y']),
        (f'ours, disjoint windows, p>=0.63 ({sel_400.sum()})', A['x'][sel_400], A['y'][sel_400])]
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for j, (t, x, y) in enumerate(rows):
    im = render(x, y, full, 10, 1.0); ax = axs[0, j]
    ax.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.5), extent=(5, 115, 115, 5)); ax.set_title(t + f'\n{len(x)} locs, all 90000 frames', fontsize=10)
    ax.add_patch(plt.Rectangle((40, 40), 30, 30, fill=False, ec='c')); ax.set_xticks([]); ax.set_yticks([])
    im = render(x, y, zl, 40, 1.2); ax = axs[1, j]
    ax.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.7), extent=zl[:2] + zl[2:][::-1]); ax.set_title('zoom 3.25 µm', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(OUTPNG, dpi=75); print('ok')
