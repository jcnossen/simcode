"""0.5 nM, all 90000 frames: SIMCODE sfhd-ndi vs ours (sc5nb), the same RCC drift (estimated from ours, rcc_drift.py) applied
to both. -> <out>/plots/recon_full_0.5nM_dc.png, drift_full_0.5nM.png"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simcode import config
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import smoke_metrics as SM, rcc_drift as R
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run', default='full_0.5nM_sc5nb_bright_real', help='full-movie run dir (pass2_est_*.npz); relative: under the out root')
ap.add_argument('--simcode', default=SM.SIMCODE_05, help="SIMCODE's published sfhd-ndi.hdf5; relative: under the data root")
ap.add_argument('--out', default='', help='png (default <out>/plots/recon_full_0.5nM_dc.png)')
ap.add_argument('--out_drift', default='', help='drift png (default <out>/plots/drift_full_0.5nM.png)')
a = ap.parse_args()
RUN = os.path.join(config.root('out'), a.run); SIMH5 = config.resolve(a.simcode, 'data')
OUTPNG = a.out or os.path.join(config.out_dir('plots'), 'recon_full_0.5nM_dc.png')
OUTDRIFT = a.out_drift or os.path.join(config.out_dir('plots'), 'drift_full_0.5nM.png')
A = SM.load_ours(RUN, 'est')
S = SM.load_simcode(SIMH5, 0, 90000)
ins = (S['x'] >= 5) & (S['x'] < 115) & (S['y'] >= 5) & (S['y'] < 115); S = {k: v[ins] for k, v in S.items()}
S['x'] = S['x'] - 2.0 / 108.33; S['y'] = S['y'] + 0.5 / 108.33
fa = (A['frame'] + 3).astype(int).clip(0, 89999); fs = (S['frame'] + 3).astype(int).clip(0, 89999)   # window centre frame
dj = (A['wstart'] % 6 == 0) & (A['p'] >= 0.63)
dx, dy, cen, sh = R.rcc(A['x_hy0'][dj], A['y_hy0'][dj], fa[dj], 90000)
dxs, dys, _, shs = R.rcc(S['x'], S['y'], fs, 90000)
print('drift (ours) range x %.1f nm y %.1f nm; SIMCODE-based estimate differs rms %.1f / %.1f nm' % (
    np.ptp(dx) * 108.33, np.ptp(dy) * 108.33, np.std(dx - dxs) * 108.33, np.std(dy - dys) * 108.33))
ax_, ay_ = A['x_hy0'] - dx[fa], A['y_hy0'] - dy[fa]; sx_, sy_ = S['x'] - dx[fs], S['y'] - dy[fs]   # same drift for both
def render(x, y, lim, z, blur):
    a, b, c, d = lim
    H, _, _ = np.histogram2d(y, x, bins=[int((d - c) * z), int((b - a) * z)], range=[[c, d], [a, b]]); return gaussian_filter(H, blur)
rows = [(f'SIMCODE sfhd-ndi (paper), {len(sx_)} locs', sx_, sy_), (f'ours, all windows, {len(ax_)} locs', ax_, ay_),
        (f'ours, disjoint windows p>=0.63, {dj.sum()} locs', ax_[dj], ay_[dj])]
fig, axs = plt.subplots(3, 3, figsize=(18, 18.5))
for j, (t, x, y) in enumerate(rows):
    for r, (lim, z, bl, nm) in enumerate((((5, 115, 5, 115), 10, 1.0, 'full FOV 11.9 µm'), ((40, 70, 40, 70), 40, 1.2, 'zoom 3.25 µm'), ((50, 60, 50, 60), 80, 1.5, 'zoom 1.08 µm (1.35 nm px)'))):
        im = render(x, y, lim, z, bl); ax = axs[r, j]
        ax.imshow(im, cmap='inferno', vmax=np.percentile(im[im > 0], 99.6), extent=(lim[0], lim[1], lim[3], lim[2]))
        ax.set_title((t + '\n' if r == 0 else '') + nm, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.add_patch(plt.Rectangle((40, 40), 30, 30, fill=False, ec='c'))
        if r == 1: ax.add_patch(plt.Rectangle((50, 50), 10, 10, fill=False, ec='c'))
plt.suptitle('0.5 nM DNA-PAINT, all 90000 frames, same RCC drift correction applied to both (estimated from ours)', fontsize=13)
plt.tight_layout(); plt.savefig(OUTPNG, dpi=65)
fig, ax = plt.subplots(figsize=(7, 3.5)); ax.plot(cen, sh[:, 0] * 108.33, 'o-', label='x (ours)'); ax.plot(cen, sh[:, 1] * 108.33, 'o-', label='y (ours)')
ax.plot(cen, shs[:, 0] * 108.33, 'x--', label='x (SIMCODE)'); ax.plot(cen, shs[:, 1] * 108.33, 'x--', label='y (SIMCODE)'); ax.set_xlabel('frame'); ax.set_ylabel('drift (nm)'); ax.legend()
plt.tight_layout(); plt.savefig(OUTDRIFT, dpi=80); print('ok')
