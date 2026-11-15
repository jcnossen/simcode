"""Rendered overview of the density simulations: rows = densities, columns = ground truth and the methods, same crop
(the preprint's Fig 2 zoom) and the same rendering for every panel (fig_density.render: FRC-set rows, 5.4 nm pixels,
1-pixel blur, 99.5 % clip). Each panel is labelled with its FRC and Jaccard (10 nm, full movie).
usage: python analysis/papersim/fig_overview.py [--prefix Dexp|D] [--model m1809]
output: out/papersim/overview_<prefix>_<model>.png
With <run>/gtfrc_<model>.json (gtfrc_run.py) the panels also give the GT-FRC (reconstruction vs ground truth); a PSF panel
(psfpanel.py: the PSF the movies were simulated with, xy + xz, log) with its CRLBs at the condition's photons / background
sits on the right."""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M
from fig_density import render, CROP
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='Dexp'); ap.add_argument('--model', default='m1809')
ap.add_argument('--suffix', default='', help="movie dir suffix, e.g. '_gauss' (D<d>_gauss: the preprint's Gaussian 1.3 px PSF reproduction)")
a = ap.parse_args()
DENS = [0.5, 1, 2, 4]
COLS = [('smlm', 'SMLM'), ('sf_gtp', 'SIMFLUX\n(true pattern)'), ('decode*', 'DECODE'), ('v1*', 'SIMCODE (preprint)'), ('v2sig', 'SIMCODE v2\n(σ ≤ 12 nm)')]
(cx, cy), w = CROP
fig, ax = plt.subplots(len(DENS), len(COLS) + 1, figsize=(2.75 * (len(COLS) + 1) + 2.6, 2.95 * len(DENS)), squeeze=False)
notes = set()
for i, d in enumerate(DENS):
    md = os.path.join(C.OUT, f'{a.prefix}{d:g}{a.suffix}'); R = json.load(open(os.path.join(md, 'full', f'metrics_{a.model}.json')))
    gf = os.path.join(md, 'full', f'gtfrc_{a.model}.json'); GF = json.load(open(gf))['methods'] if os.path.exists(gf) else {}
    gt = np.load(os.path.join(md, 'gt.npz'))
    ax[i, 0].imshow(render(gt['pts'][:, 0], gt['pts'][:, 1], (cx, cy), w), cmap='inferno')
    ax[i, 0].set_ylabel(f'density {d:g}', fontsize=12); ax[i, 0].plot([8, 8 + 500 / (C.PX_NM / 20)], [w * 20 - 10] * 2, 'w-', lw=3)
    if i == 0: ax[i, 0].set_title('ground truth', fontsize=11); ax[i, 0].text(8, w * 20 - 18, '500 nm', color='w', fontsize=8)
    for j, (k, lab) in enumerate(COLS):
        tag = ''
        if k == 'decode*':
            k = 'decode_pre' if 'decode_pre' in R['methods'] else ('decode_sim' if 'decode_sim' in R['methods'] else 'decode')
            if k == 'decode_pre': notes.add('the preprint\'s own DECODE model (decode-3f-g1.3, Gaussian PSF), 6-frame sums from frame 0, crlb 0.2 px'); tag = ''
            if k != 'decode_pre' and a.suffix == '_v2psf':
                tag = ' †'; notes.add('† DECODE retrained for the v2 PSF, heights 0-0.40 um and these photon / background / density ranges (dec_sim1809v2)')
            elif k != 'decode_pre': tag = ' †' if k == 'decode' else ' ‡'; notes.add(('† DECODE model trained for the real 1809/2 movie: photons and background are in its range here, the emitter density of rows 2 and 4 is above it' if a.prefix == 'Dexp' else '† DECODE model trained for the real 1809/2 movie (photons, background, on-time and density out of its range here)') if k == 'decode' else '‡ DECODE retrained for the simulation conditions (training plateaued; weak)')
        if k == 'v1*':
            k = 'v1' if ('v1' in R['methods'] and R['methods']['v1']['n_frc'] > 2000) else 'v1gt'
            if k == 'v1gt': tag = ' §'; notes.add('§ preprint SIMCODE given the true pattern (as published it fails / gives no output on this movie)')
        A = ax[i, j + 1]
        if k not in R['methods']:
            A.text(0.5, 0.5, 'no result', ha='center', transform=A.transAxes); A.set_xticks([]); A.set_yticks([]); continue
        r = R['methods'][k]; T = M.load_method(k, r['path']); _, _, f = M.frc_set(T, R['W'], R['nframes'])
        A.imshow(render(T['x'][f], T['y'][f], (cx, cy), w), cmap='inferno')
        gtl = f"   GT-FRC {GF[k]['gt_frc']['res_nm']:.1f} nm" if k in GF else ''
        A.text(0.02, 0.02, f"FRC {r.get('frc_nm', float('nan')):.1f} nm{gtl}\nJaccard {r['jaccard_end']:.2f}   {r['n_frc'] / 1e3:.0f}k localizations{tag}", transform=A.transAxes,
               color='w', fontsize=8, va='bottom', bbox=dict(facecolor='k', alpha=0.55, pad=1.5, lw=0))
        if i == 0: A.set_title(lab, fontsize=11)
    for A in ax[i]: A.set_xticks([]); A.set_yticks([])
cond = 'experimental photons and background of 1809/2 (median 817 photons per window, 5.1 photons/px/frame)' if a.prefix == 'Dexp' else "preprint condition (500 photons per window, background 2.5 photons/px/frame)"
psfname = 'Gaussian 1.3 px PSF (the preprint simulation)' if a.suffix == '_gauss' else 'measured 1809/2 PSF'
if a.suffix == '_v2psf':        # v2-PSF sets (2026-11-15): conditions from the movies' gen.json, in v2 camera photon units
    g0 = json.load(open(os.path.join(C.OUT, f'{a.prefix}{DENS[0]:g}{a.suffix}', 'gen.json')))
    cond = (f"experimental photons and background of 1809/2 in v2 camera units (median {g0['brightness_fit']['fit_median']:.0f} photons per window, "
            f"{g0['bg_per_frame']['mean']:.2f} photons/px/frame)" if a.prefix == 'Dexp' else
            "preprint condition, nominal (500 photons per window, background 2.5 photons/px/frame)")
    zd = g0['z_dist_um']
    psfname = f"in-situ PSF v2 of 1809/2 + v2 camera, emitter heights 0-400 nm (median {zd['p50'] * 1e3:.0f} nm)"

fig.suptitle(f'Simulated microtubules, 60 000 frames, {psfname}; {cond}.\nDensity in emitters/µm²/frame; same crop ({w * C.PX_NM / 1000:.2f} µm) and rendering in every panel; SIMCODE v2 with the PSF-matched networks.', fontsize=10)
if any(os.path.exists(os.path.join(C.OUT, f'{a.prefix}{d:g}{a.suffix}', 'full', f'gtfrc_{a.model}.json')) for d in DENS):
    notes.add('GT-FRC = FRC between the rendered reconstruction and the rendered ground truth (all labelling sites; 5 nm pixels, 1/7); FRC = the fair half-split FRC')
fig.text(0.01, 0.005, '\n'.join(sorted(notes)), fontsize=8)
plt.tight_layout(rect=[0, 0.065, 2.75 * (len(COLS) + 1) / (2.75 * (len(COLS) + 1) + 2.6), 0.95])
# PSF panel (the PSF of the simulation) with its CRLBs at the condition's photons / background
import psfpanel
g0 = json.load(open(os.path.join(C.OUT, f'{a.prefix}{DENS[0]:g}{a.suffix}', 'gen.json')))
spec = 'gauss:' + g0['psf'][5:] if g0['psf'].startswith('gauss') else g0['psf']
if a.prefix == 'Dexp': ph, bg = g0['brightness_fit']['fit_median'], g0['bg_per_frame']['mean']
else: ph, bg = g0['intensity_per_6frames'], g0['bg']
pat = g0.get('pattern') or {}                          # (the Dexp movies: the density movies' pattern, 0 / 90 deg, 224 nm, 0.9)
info = psfpanel.psf_info(spec, ph, bg, pattern={k: pat[k] for k in ('angles_deg', 'pitch_nm', 'depth') if k in pat})
fw = 2.75 * (len(COLS) + 1) + 2.6
psfpanel.add_panel(fig, [1 - 2.35 / fw, 0.42, 1.9 / fw, 0.5], info, fontsize=7)
fn = os.path.join(C.OUT, f'overview_{a.prefix}{a.suffix}_{a.model}.png'); plt.savefig(fn, dpi=130); print(fn)
