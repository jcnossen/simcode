"""RMSD vs GT window photons on the gate scenes: our sc front-ends vs SIMCODE (paper network L48 + eq.-2 pattern fit + MVU
merge, simcode/benchmark/preprint_gate.py), with the CRLB lines. Matching as gate_report_xdet: one-to-one within 1 px on the
detection position (x_det), targets with CRLB_SF_multi < 25 nm only (as gate_report_xdet), score >= 0.5; RMSD per axis of the final estimate (incl. outliers). -> <out>/plots/crlb_vs_simcode.png"""
import argparse, os
from simcode import config
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--gate_dir', default=os.path.join(config.root('out'), 'gate'), help='gate npz directory (simcode.benchmark.gate)')
ap.add_argument('--out', default='', help='png (default <out>/plots/crlb_vs_simcode.png)')
A_ = ap.parse_args()
OUTPNG = A_.out or os.path.join(config.out_dir('plots'), 'crlb_vs_simcode.png')
import glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from simcode.benchmark.gate_report import load, match
PX = 108.33
D = A_.gate_dir
scenes = ['iso', 'pair2', 'rs05p', 'rs25p', 'rs05pm', 'rs25pm']
meths = [('_hd_sc5nb15pm', 'ours sc5n + PM', 'tab:red', '-'), ('_hd_sc5nbb15', 'ours sc5nb (MAP)', 'tab:orange', '-'),
         ('_scodenf', 'SIMCODE net+eq.2+MVU, no filters', 'tab:blue', '-'), ('_scode', 'SIMCODE paper filters', 'tab:cyan', '--')]
bins = np.array([150, 250, 400, 700, 1200, 2000, 3500, 6000, 10000, 20000])
cen = np.sqrt(bins[:-1] * bins[1:])
fig, axs = plt.subplots(3, len(scenes), figsize=(4.2 * len(scenes), 10), gridspec_kw=dict(height_ratios=[3, 1.4, 1.1]))
for j, sc in enumerate(scenes):
    ax, ax3, ax2 = axs[:, j]
    first = True
    for tag, lab, col, ls in meths:
        if not glob.glob(f'{D}/{sc}__s30*{tag}.npz'): continue
        R, nf = load(D, sc, tag)
        tgt = (R['gt_target'] > 0) & np.isfinite(R['gt_sf_m']).all(1) & (np.hypot(*np.nan_to_num(R['gt_sf_m'], nan=1e9).T) / np.sqrt(2) * PX < 25)   # as gate_report_xdet
        if first:
            ph = R['gt_phot'][tgt]
            for key, st, l_ in (('gt_sm_m', 'k--', 'CRLB SMLM multi'), ('gt_sf_i', 'k:', 'CRLB SIMFLUX isolated'), ('gt_sf_m', 'k-', 'CRLB SIMFLUX multi')):
                c = np.sqrt((R[key][tgt] ** 2).mean(1))
                ax.loglog(cen, [np.sqrt(np.nanmean(c[(ph >= a) & (ph < b)] ** 2)) * PX if ((ph >= a) & (ph < b)).sum() > 30 else np.nan
                                for a, b in zip(bins[:-1], bins[1:])], st, lw=1.5, label=l_)
            first = False
        idx = np.nonzero(R['sc_score'] >= 0.5)[0]
        og = match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
        ti, oi = og[og >= 0], idx[og >= 0]; k = tgt[ti]; ti, oi = ti[k], oi[k]
        e = np.c_[R['sc_x_sc'][oi] - R['gt_x'][ti], R['sc_y_sc'][oi] - R['gt_y'][ti]]
        p = R['gt_phot'][ti]; c = R['gt_sf_m'][ti]
        rm, rr, rc = [], [], []
        pall = R['gt_phot'][tgt]
        for a, b in zip(bins[:-1], bins[1:]):
            m = (p >= a) & (p < b) & np.isfinite(e).all(1) & np.isfinite(c).all(1)
            nt = ((pall >= a) & (pall < b)).sum()
            rc.append(((p >= a) & (p < b)).sum() / nt if nt > 30 else np.nan)
            rm.append(np.sqrt(np.mean(e[m] ** 2)) * PX if m.sum() > 30 else np.nan)
            rr.append(np.sqrt(np.mean(e[m] ** 2)) / np.sqrt(np.mean(c[m] ** 2)) if m.sum() > 30 else np.nan)
        ax.loglog(cen, rm, 'o' + ls, color=col, ms=4, label=lab)
        ax3.semilogx(cen, rr, 'o' + ls, color=col, ms=4); ax2.semilogx(cen, rc, 'o' + ls, color=col, ms=4)
    ax.set_title(sc, fontsize=10); ax.grid(True, which='both', alpha=0.3); ax.set_ylabel('x/y RMSD (nm)')
    ax3.axhline(1, color='k', lw=0.8); ax3.set_yscale('log'); ax3.set_ylim(0.7, 8); ax3.grid(True, which='both', alpha=0.3)
    ax3.set_ylabel('RMSD / CRLB_SF multi')
    ax2.set_ylim(0, 1.02); ax2.grid(True, alpha=0.3); ax2.set_ylabel('recall (1 px, x_det)'); ax2.set_xlabel('GT window photons (label units)')
    if j == 0: ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUTPNG, dpi=85); print('saved')
