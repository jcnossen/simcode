"""m20check figures (2026-11-12) from out/psfmatch/m20check/psf_images.npz + psf_compare.json (analysis/psfmatch/m20check_psf.py):
  psf_images.png   in-situ PSFs (z = 0.055 um, unit sum) of 20nM_2, 2.5nM_2, 1809_2 and their pairwise differences (% of the peak)
  psf_crlb.png     isolated-emitter CRLB vs photons (SMLM | SIMFLUX panels) for the three PSFs
usage: python analysis/psfmatch/m20check_fig.py"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from simcode import config

OUT = os.path.join(config.root('out'), 'psfmatch', 'm20check')
MOV = ['20nM_m2', '2p5nM_m2', '1809_2']; LAB = {'20nM_m2': '20nM_2', '2p5nM_m2': '2.5nM_2 (m25 training PSF)', '1809_2': '1809_2 (m1809 training PSF)'}
COL = {'20nM_m2': '#2a78d6', '2p5nM_m2': '#eb6834', '1809_2': '#1baf7a'}
INK, MUTED = '#1a1a19', '#6b6a63'
SEQ = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#cde2fb', '#6da7ec', '#256abf', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#104281', '#6da7ec', '#eeeeec', '#f0a07c', '#9c3a12'])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})
I = np.load(f'{OUT}/psf_images.npz'); J = json.load(open(f'{OUT}/psf_compare.json'))

fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)
vm = max(I[f'z055_{m}'].max() for m in MOV)
for k, m in enumerate(MOV):
    a = I[f'z055_{m}']; s = J['shape_z055'][m]
    im = ax[0, k].imshow(a, cmap=SEQ, vmin=0, vmax=vm, extent=[-7.5, 7.5, 7.5, -7.5])
    ax[0, k].set_title(f'{LAB[m]}\nGaussian-fit sd x {s["sx"]:.2f} px, y {s["sy"]:.2f} px; peak {100 * s["peak"]:.1f} %', fontsize=9)
fig.colorbar(im, ax=ax[0, :], shrink=0.8, label='fraction of photons per pixel')
dm = 0
D = []
for a, b in [(MOV[0], MOV[1]), (MOV[0], MOV[2]), (MOV[1], MOV[2])]:
    d = 100 * (I[f'z055_{a}'] - I[f'z055_{b}']) / max(I[f'z055_{a}'].max(), I[f'z055_{b}'].max()); D.append((a, b, d)); dm = max(dm, np.abs(d).max())
for k, (a, b, d) in enumerate(D):
    r = J['image_diff'][f'{a}|{b}|z055']
    im = ax[1, k].imshow(d, cmap=DIV, vmin=-dm, vmax=dm, extent=[-7.5, 7.5, 7.5, -7.5])
    ax[1, k].set_title(f'{LAB[a].split(" ")[0]} minus {LAB[b].split(" ")[0]}\nrms {r["rms_pct_peak"]:.1f} %, max {r["max_pct_peak"]:.1f} % of the peak; L1 {r["L1"]:.2f}', fontsize=9)
fig.colorbar(im, ax=ax[1, :], shrink=0.8, label='difference, % of the larger peak')
for q in ax.ravel(): q.set_xlabel('x (px, 108.33 nm)'); q.set_ylabel('y (px)')
fig.suptitle('In-situ PSFs (angle-averaged model, emitter at z = 0.055 µm, 15 × 15 px, unit sum) and their differences', fontsize=10)
fig.savefig(f'{OUT}/psf_images.png', dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True, sharey=True)
ph = np.array(J['photons'])
for k, (mode, ttl) in enumerate((('smlm', 'SMLM (6 frames summed)'), ('sf', 'SIMFLUX (6 frames, known pattern)'))):
    for m in MOV:
        v = [r['xy'] for r in J['crlb_nm'][f'{m}|{mode}']]
        ax[k].plot(ph, v, '-o', color=COL[m], lw=2, ms=5, label=LAB[m])
    ax[k].set_xscale('log'); ax[k].set_yscale('log'); ax[k].set_title(ttl, fontsize=10); ax[k].set_xlabel('photons per 6-frame window')
    ax[k].grid(True, which='major', color='#e4e3df', lw=0.6); ax[k].set_xticks(ph); ax[k].set_xticklabels([f'{int(p)}' for p in ph]); ax[k].minorticks_off()
    ax[k].set_xlim(ph[0] * 0.85, ph[-1] * 1.35)
ax[0].set_ylabel('lateral CRLB (nm, rms of x and y)'); ax[0].set_yticks([1, 2, 5, 10, 20, 50]); ax[0].set_yticklabels(['1', '2', '5', '10', '20', '50'])
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle(f'Isolated-emitter CRLB, bg {J["bg_per_px_frame"]:.0f} photons / px / frame, z = 0.055 µm', fontsize=10)
fig.savefig(f'{OUT}/psf_crlb.png', dpi=150); plt.close(fig)
print('wrote', f'{OUT}/psf_images.png', f'{OUT}/psf_crlb.png')
