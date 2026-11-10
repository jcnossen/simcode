"""<out>/plots/field_dep_offsets.png: sensor-frame cell maps (10 px) of d = final - prior (dx, dy; >= 700 ph, iso), its normalised
width, and the pattern depth (check 3), for real 0.5 / 2.5 nM and the single-PSF sim nulls; bottom: tile-mean excess rms
(4x4 tiles, check 1) per photon bin. Inputs in <out>/field: maps_<ds>.npz, maps_summary.txt (field_maps.py), check3_<ds>.npz
(check3_pattern.py), check1_summary.json (check1_summary.py); order: check1_offsets.py, check1_summary.py, field_maps.py,
check3_pattern.py, plot_field_offsets.py (all in this directory).
usage: python plot_field_offsets.py [out.png]"""
import os, sys
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simcode2 import config
F = os.path.join(config.root('out'), 'field')
OUTP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.out_dir('plots'), 'field_dep_offsets.png')
DSL = [('real05', 'real 0.5 nM'), ('real25', 'real 2.5 nM (shifted dx 2, dy 22)'), ('sim05', 'sim m05qh (1 global PSF)'), ('sim25', 'sim m25qh (1 global PSF)')]
J = json.load(open(f'{F}/check1_summary.json'))

fig = plt.figure(figsize=(17, 19))
gs = fig.add_gridspec(5, 5, height_ratios=[1, 1, 1, 1, 0.9], hspace=0.32, wspace=0.28)
ext = None
for r, (ds, lab) in enumerate(DSL):
    M = np.load(f'{F}/maps_{ds}.npz'); C = np.load(f'{F}/check3_{ds}.npz')
    G0, CS, NC = float(M['G0']), float(M['CS']), int(M['NC'])
    ext = [G0, G0 + CS * NC, G0 + CS * NC, G0]
    dx = M['d0_4'] - np.nanmedian(M['d0_4']); dy = M['d1_4'] - np.nanmedian(M['d1_4'])
    zw = np.sqrt(0.5 * (M['z0_rms_4'] ** 2 + M['z1_rms_4'] ** 2))
    panels = [(dx, 'RdBu_r', (-8, 8), 'mean dx (nm), final - prior'), (dy, 'RdBu_r', (-8, 8), 'mean dy (nm)'),
              (zw, 'magma', (0.9, 1.6), 'width: rms of d / sqrt(s_p^2 - s_f^2)'),
              (C['m0'], 'viridis', (0.7, 1.0), 'depth angle 0 (cross-cycle)'), (C['m1'], 'viridis', (0.7, 1.0), 'depth angle 1')]
    for c, (img, cm, lim, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1], extent=ext, interpolation='nearest')
        ax.set_title(f'{ttl}' if r == 0 else '', fontsize=10)
        if c == 0: ax.set_ylabel(f'{lab}\nsensor y (px, 0.5 nM frame)', fontsize=10)
        ax.tick_params(labelsize=8)
        if r == 3: ax.set_xlabel('sensor x (px)', fontsize=9)
        if c < 2:
            ok = np.isfinite(img)
            ax.text(0.02, 0.02, f'cell rms {np.nanstd(img):.1f} nm', transform=ax.transAxes, fontsize=8, color='k',
                    bbox=dict(fc='w', ec='none', alpha=0.7, pad=1))
        if c >= 3:
            ax.text(0.02, 0.02, f'p5-p95 {np.nanpercentile(img, 5):.2f}-{np.nanpercentile(img, 95):.2f}', transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc='w', ec='none', alpha=0.7, pad=1))
        if r == 3: plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.18)
# bottom: excess rms per photon bin (4x4 tiles, iso)
labs = ['200-700', '700-1200', '1200-2000', '>2000']
cols = {'real05': '#1f5fbf', 'real25': '#c2410c', 'sim05': '#7aa6e8', 'sim25': '#f0a070'}
for c, comp in enumerate(('mx', 'my')):
    ax = fig.add_subplot(gs[4, c * 2:(c * 2 + 2)])
    for ds, lab in DSL:
        v = [J.get(f'{ds}_iso_4_{l}', {}).get(comp, {}).get('excess', np.nan) for l in labs]
        ch = [J.get(f'{ds}_iso_4_{l}', {}).get(comp, {}).get('chi2', np.nan) for l in labs]
        ax.plot(range(4), v, 'o-' if ds.startswith('real') else 's--', color=cols[ds], lw=2, ms=8, label=lab.split(' (')[0])
        for i, (vv, cc) in enumerate(zip(v, ch)):
            if np.isfinite(vv): ax.annotate(f'χ²/dof {cc:.1f}', (i, vv), textcoords='offset points', xytext=(4, 4), fontsize=7, color='#444')
    ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_xlabel('photons (label units)')
    ax.set_ylabel(f'tile-to-tile excess rms of mean d{"xy"[c]} (nm)'); ax.set_ylim(0, 5)
    ax.set_title(f'4x4 tiles, isolated: spread of mean d{"xy"[c]} beyond bootstrap noise', fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper left')
ax = fig.add_subplot(gs[4, 4]); ax.axis('off')
txt = open(f'{F}/maps_summary.txt').read()
ax.text(0, 1, 'sensor-frame replication (r, cells):\n real05 vs real25 dx +0.45, dy +0.28\n (different samples)\n'
        ' phase angle0 +0.43, angle1 +0.30\n depth angle0 +0.69, angle1 +0.43\n real vs sim: |r| < 0.2\n\n'
        'cross-photon-bin r (dx):\n real 0.79-0.88, sim 0.07-0.24\n\nLegendre-2 share of d variance:\n real 0.1-0.3 (mostly small-scale)\n'
        'depth: L2 R² 0.7-0.9 (smooth)', va='top', fontsize=9, family='monospace')
fig.suptitle('Field dependence of the pattern-vs-PSF position difference d = x_hy0 - x_prior (sc5nb + pattern solve), 10 px cells on the sensor frame, isolated (nn ≥ 4 px), ≥ 700 ph', fontsize=12, y=0.995)
fig.savefig(OUTP, dpi=90, bbox_inches='tight')
print(OUTP)
