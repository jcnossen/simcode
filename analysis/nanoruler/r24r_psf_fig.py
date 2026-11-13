"""R24 reuse check: figure of the PSF comparison (from psf_images.npz of r24r_psf.py). Per movie one row: the movie's averaged PSF,
its difference to the nr80 PSF (averaged, orientation 0, orientation 1) and, for scale, the movie's own orientation-1 minus
orientation-0 difference (photoselection). Differences in % of the PSF peak, one common diverging scale.
usage: python analysis/nanoruler/r24r_psf_fig.py -> out/nanoruler/nr40/reuse/psf_diff.png"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/home/jc/simcode2/out/nanoruler'
fig, ax = plt.subplots(2, 5, figsize=(13.5, 5.6), constrained_layout=True)
V = 6.0
for r, n in enumerate(('nr40', 'nr20')):
    D = np.load(f'{ROOT}/{n}/reuse/psf_images.npz'); pk = D[f'avg_{n}'].max()
    im0 = ax[r, 0].imshow(D[f'avg_{n}'] / pk, cmap='Blues', vmin=0, vmax=1); ax[r, 0].set_title(f'{n} PSF (averaged), peak = 1', fontsize=10)
    panels = [(f'{n} - nr80, averaged', D[f'avg_{n}'] - D['avg_nr80']), (f'{n} - nr80, orientation 0', D[f'orient0_{n}'] - D['orient0_nr80']),
              (f'{n} - nr80, orientation 1', D[f'orient1_{n}'] - D['orient1_nr80']), (f'{n}: orientation 1 - 0', D[f'orient1_{n}'] - D[f'orient0_{n}'])]
    for c, (t, d) in enumerate(panels, 1):
        im = ax[r, c].imshow(100 * d / pk, cmap='RdBu_r', vmin=-V, vmax=V)
        ax[r, c].set_title(f'{t}\nrms {100 * np.sqrt((d ** 2).mean()) / pk:.2f} %, max {100 * np.abs(d).max() / pk:.1f} % of peak', fontsize=9)
    for a in ax[r]: a.set_xticks([0, 7, 14]); a.set_yticks([0, 7, 14]); a.tick_params(labelsize=8, color='0.6'); [s.set_color('0.7') for s in a.spines.values()]
    fig.colorbar(im0, ax=ax[r, 0], shrink=0.8).ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax[r, 1:], shrink=0.8, pad=0.01); cb.set_label('difference, % of the PSF peak', fontsize=9); cb.ax.tick_params(labelsize=8)
fig.suptitle('In-situ PSF of nr40 / nr20 against nr80 (15 x 15 px of 65 nm, unit-sum images, emitter centred, z = 0.01 um)', fontsize=11)
fig.savefig(f'{ROOT}/nr40/reuse/psf_diff.png', dpi=130); print('saved')
