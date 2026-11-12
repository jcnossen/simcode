"""Rendered overview of the density simulations: rows = densities, columns = ground truth and the methods, same crop
(the preprint's Fig 2 zoom) and the same rendering for every panel (fig_density.render: FRC-set rows, 5.4 nm pixels,
1-pixel blur, 99.5 % clip). Each panel is labelled with its FRC and Jaccard (10 nm, full movie).
usage: python analysis/papersim/fig_overview.py [--prefix Dexp|D] [--netset m1809]
output: out/papersim/overview_<prefix>_<netset>.png"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M
from fig_density import render, CROP
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='Dexp'); ap.add_argument('--netset', default='m1809')
ap.add_argument('--suffix', default='', help="movie dir suffix, e.g. '_gauss' (D<d>_gauss: the preprint's Gaussian 1.3 px PSF reproduction)")
a = ap.parse_args()
DENS = [0.5, 1, 2, 4]
COLS = [('smlm', 'SMLM'), ('sf_gtp', 'SIMFLUX\n(true pattern)'), ('decode*', 'DECODE'), ('v1*', 'SIMCODE (preprint)'), ('v2q45', 'SIMCODE v2\n(u ≤ 0.45)')]
(cx, cy), w = CROP
fig, ax = plt.subplots(len(DENS), len(COLS) + 1, figsize=(2.75 * (len(COLS) + 1), 2.95 * len(DENS)), squeeze=False)
notes = set()
for i, d in enumerate(DENS):
    md = os.path.join(C.OUT, f'{a.prefix}{d:g}{a.suffix}'); R = json.load(open(os.path.join(md, 'full', f'metrics_{a.netset}.json')))
    gt = np.load(os.path.join(md, 'gt.npz'))
    ax[i, 0].imshow(render(gt['pts'][:, 0], gt['pts'][:, 1], (cx, cy), w), cmap='inferno')
    ax[i, 0].set_ylabel(f'density {d:g}', fontsize=12); ax[i, 0].plot([8, 8 + 500 / (C.PX_NM / 20)], [w * 20 - 10] * 2, 'w-', lw=3)
    if i == 0: ax[i, 0].set_title('ground truth', fontsize=11); ax[i, 0].text(8, w * 20 - 18, '500 nm', color='w', fontsize=8)
    for j, (k, lab) in enumerate(COLS):
        tag = ''
        if k == 'decode*':
            k = 'decode_pre' if 'decode_pre' in R['methods'] else ('decode_sim' if 'decode_sim' in R['methods'] else 'decode')
            if k == 'decode_pre': notes.add('the preprint\'s own DECODE model (decode-3f-g1.3, Gaussian PSF), 6-frame sums from frame 0, crlb 0.2 px'); tag = ''
            if k != 'decode_pre': tag = ' †' if k == 'decode' else ' ‡'; notes.add(('† DECODE model trained for the real 1809/2 movie: photons and background are in its range here, the emitter density of rows 2 and 4 is above it' if a.prefix == 'Dexp' else '† DECODE model trained for the real 1809/2 movie (photons, background, on-time and density out of its range here)') if k == 'decode' else '‡ DECODE retrained for the simulation conditions (training plateaued; weak)')
        if k == 'v1*':
            k = 'v1' if ('v1' in R['methods'] and R['methods']['v1']['n_frc'] > 2000) else 'v1gt'
            if k == 'v1gt': tag = ' §'; notes.add('§ preprint SIMCODE given the true pattern (as published it fails / gives no output on this movie)')
        A = ax[i, j + 1]
        if k not in R['methods']:
            A.text(0.5, 0.5, 'no result', ha='center', transform=A.transAxes); A.set_xticks([]); A.set_yticks([]); continue
        r = R['methods'][k]; T = M.load_method(k, r['path']); _, _, f = M.frc_set(T, R['W'], R['nframes'])
        A.imshow(render(T['x'][f], T['y'][f], (cx, cy), w), cmap='inferno')
        A.text(0.02, 0.02, f"FRC {r.get('frc_nm', float('nan')):.1f} nm   Jaccard {r['jaccard_end']:.2f}\n{r['n_frc'] / 1e3:.0f}k localizations{tag}", transform=A.transAxes,
               color='w', fontsize=8, va='bottom', bbox=dict(facecolor='k', alpha=0.55, pad=1.5, lw=0))
        if i == 0: A.set_title(lab, fontsize=11)
    for A in ax[i]: A.set_xticks([]); A.set_yticks([])
cond = 'experimental photons and background of 1809/2 (median 817 photons per window, 5.1 photons/px/frame)' if a.prefix == 'Dexp' else "preprint condition (500 photons per window, background 2.5 photons/px/frame)"
fig.suptitle(f'Simulated microtubules, 60 000 frames, measured 1809/2 PSF; {cond}.\nDensity in emitters/µm²/frame; same crop ({w * C.PX_NM / 1000:.2f} µm) and rendering in every panel; SIMCODE v2 with the PSF-matched networks.', fontsize=10)
fig.text(0.01, 0.005, '   '.join(sorted(notes)), fontsize=8)
plt.tight_layout(rect=[0, 0.02, 1, 0.95]); fn = os.path.join(C.OUT, f'overview_{a.prefix}{a.suffix}_{a.netset}.png'); plt.savefig(fn, dpi=130); print(fn)
