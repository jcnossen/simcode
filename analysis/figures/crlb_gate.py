"""CRLB plots (RMSD and CRLB vs total photons, log-log) from the gate-suite records (3 seeds pooled).
Per GT target: nearest prediction of the method in the same FOV within 1 px (greedy nearest; the gate report itself uses
Hungarian at 0.5 px). RMSD over matched targets incl. outliers; recall shown in the lower panel.
Fronts pf / det / hd (the development-era hybrid front-ends; gate npz with pf_* / det_* / hd_* columns). -> <out>/plots/crlb_gate.png"""
import argparse, os
from simcode2 import config
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--gate_dir', default=os.path.join(config.root('out'), 'gate'), help='gate npz directory (simcode2.hd.gate)')
ap.add_argument('--out', default='', help='png (default <out>/plots/crlb_gate.png)')
A_ = ap.parse_args()
OUTPNG = A_.out or os.path.join(config.out_dir('plots'), 'crlb_gate.png')
import numpy as np, glob
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
PX = 108.33
scenes = ['iso', 'rs05x', 'rs25x', 'rs25', 'pair2', 'pair1.5']
meths = [('pf', 'hy', 'pf front-end + hybrid', 'tab:orange'), ('det', 'hy', 'det+merge + hybrid', 'tab:blue'),
         ('hd', 'hdf', 'new HD network (hd_b) + hybrid', 'tab:red')]
bins = np.array([300, 500, 800, 1300, 2000, 3200, 5000, 8000, 13000, 20000])
fig, axs = plt.subplots(3, len(scenes), figsize=(4.3 * len(scenes), 10), gridspec_kw=dict(height_ratios=[3, 1.4, 1.1]))
for j, sc in enumerate(scenes):
    files = sorted(glob.glob(f'{A_.gate_dir}/{sc}__s30?.npz'))     # untagged records (the development glob s30*.npz predates the tagged runs)
    if not files: continue
    recs = {m[0]: [] for m in meths}; G = []
    for f in files:
        Z = np.load(f, allow_pickle=True)
        tgt = Z['gt_target'].astype(bool) & ~Z['gt_dontcare'].astype(bool)
        g = dict(fov=Z['gt_fov'][tgt], x=Z['gt_x'][tgt], y=Z['gt_y'][tgt], ph=Z['gt_phot'][tgt],
                 cm=np.sqrt((Z['gt_sf_m'][tgt] ** 2).mean(1)) if Z['gt_sf_m'].ndim > 1 else Z['gt_sf_m'][tgt],
                 ci=np.sqrt((Z['gt_sf_i'][tgt] ** 2).mean(1)) if Z['gt_sf_i'].ndim > 1 else Z['gt_sf_i'][tgt],
                 cs=np.sqrt((Z['gt_sm_m'][tgt] ** 2).mean(1)) if Z['gt_sm_m'].ndim > 1 else Z['gt_sm_m'][tgt])
        G.append(g)
        for pre, est, _, _ in meths:
            b = Z[f'{pre}_b']; x = Z[f'{pre}_x_{est}']; y = Z[f'{pre}_y_{est}']
            err = np.full(len(g['x']), np.nan)
            for fv in np.unique(g['fov']):
                mg = g['fov'] == fv; mp = (b == fv) & np.isfinite(x)
                if mp.sum() == 0: continue
                d, i = cKDTree(np.stack([x[mp], y[mp]], 1)).query(np.stack([g['x'][mg], g['y'][mg]], 1))
                e = np.where(d < 1.0, d, np.nan); err[np.nonzero(mg)[0]] = e
            recs[pre].append(err)
    g = {k: np.concatenate([gg[k] for gg in G]) for k in G[0]}
    cen = np.sqrt(bins[:-1] * bins[1:]); ax, ax3, ax2 = axs[0, j], axs[1, j], axs[2, j]
    def cr(c):
        return np.array([np.sqrt(np.nanmean(c[(g['ph'] >= lo) & (g['ph'] < hi)] ** 2)) * PX if ((g['ph'] >= lo) & (g['ph'] < hi)).sum() > 30 else np.nan for lo, hi in zip(bins[:-1], bins[1:])])
    ax.loglog(cen, cr(g['cs']) / np.sqrt(1), 'k--', lw=1, label='CRLB SMLM (multi-emitter)')
    ax.loglog(cen, cr(g['ci']), 'k:', lw=1.5, label='CRLB SIMFLUX isolated')
    ax.loglog(cen, cr(g['cm']), 'k-', lw=2, label='CRLB SIMFLUX multi-emitter')
    for pre, est, lab, col in meths:
        e = np.concatenate(recs[pre]); rm, rc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (g['ph'] >= lo) & (g['ph'] < hi)
            rc.append(np.isfinite(e[s]).mean() if s.sum() > 30 else np.nan)
            m = s & np.isfinite(e)
            rm.append(np.sqrt(np.mean(e[m] ** 2)) / np.sqrt(2) * PX if m.sum() > 30 else np.nan)
            rr = (np.sqrt(np.mean(e[m] ** 2)) / np.sqrt(2)) / np.sqrt(np.mean(g['cm'][m] ** 2)) if m.sum() > 30 else np.nan
            rc[-1] = (rc[-1], rr)
        ax.loglog(cen, rm, 'o-', color=col, label=lab, ms=4)
        ax2.semilogx(cen, [r[0] if isinstance(r, tuple) else r for r in rc], 'o-', color=col, ms=4)
        ax3.semilogx(cen, [r[1] if isinstance(r, tuple) else np.nan for r in rc], 'o-', color=col, ms=4)
    ax.set_title(sc + (' (exact model)' if sc.endswith('x') else ''), fontsize=10); ax.grid(True, which='both', alpha=0.3)
    ax.set_xlabel('true photons per window'); ax.set_ylabel('x/y RMSD (nm)')
    ax3.axhline(1, color='k', lw=0.8); ax3.set_ylim(0.5, 5); ax3.set_yscale('log'); ax3.grid(True, which='both', alpha=0.3); ax3.set_ylabel('RMSD / multi-CRLB\n(same detected set)')
    ax2.set_ylim(0, 1.02); ax2.grid(True, alpha=0.3); ax2.set_ylabel('recall (1 px)'); ax2.set_xlabel('true photons per window')
    if j == 0: ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUTPNG, dpi=85); print('saved')
