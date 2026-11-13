"""
CRLB plots for the sc (network-intensity) method on the gate suite (<out>/gate, 3 seeds pooled):
RMSD (x/y, nm) vs true window photons, log-log, with the CRLB lines computed on the SAME detected (matched) set of each
method: SIMFLUX multi-emitter (solid), SIMFLUX isolated (dotted), SMLM multi-emitter (dashed); the exact-PSF single-
emitter MLE-SF is a sanity line in 'iso' only. Matching (iteration 29, critic 27 #6a): Hungarian within 1 px on the
DETECTOR position x_det (score >= 0.5), so the estimator's own gross errors stay in the RMSD (the MLE line: its true-pixel start); resolvable targets only (multi-emitter CRLB < 25 nm, as the gate report). Lower panels: RMSD / CRLB_SF multi (same set) and recall (1 px).
usage: python analysis/figures/plot_crlb_sc3.py --runs '':sc,_sc2:sc2 [--out <out>/plots/crlb_sc3.png]
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from simcode import config
from simcode.hd import gate_report as GR

PX = 108.33
BINS = np.array([300, 500, 800, 1300, 2000, 3200, 5000, 8000, 13000, 20000])
COL = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


def curves(R, f, m):
    sel = np.isfinite(R[f'{f}_x_{m}']) & (R[f'{f}_score'] >= 0.5)
    idx = np.nonzero(sel)[0]
    og = GR.match(R['gt_fov'], R['gt_x'], R['gt_y'], R[f'{f}_b'][idx], R[f'{f}_x_det'][idx], R[f'{f}_y_det'][idx], 1.0)
    ti = og[og >= 0]; oi = idx[og >= 0]
    tg = R['gt_target'] > 0
    res = np.hypot(R['gt_sf_m'][:, 0], R['gt_sf_m'][:, 1]) / np.sqrt(2) * PX < 25     # resolvable (as the gate report)
    keep = tg[ti] & res[ti]; ti, oi = ti[keep], oi[keep]
    e = np.stack([R[f'{f}_x_{m}'][oi] - R['gt_x'][ti], R[f'{f}_y_{m}'][oi] - R['gt_y'][ti]], 1)
    ph = R['gt_phot'][ti]
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (ph >= lo) & (ph < hi); nt = (tg & (R['gt_phot'] >= lo) & (R['gt_phot'] < hi)).sum()
        if s.sum() < 30:
            rows.append([np.nan] * 6); continue
        rms = lambda c: np.sqrt(np.nanmean(c[ti[s]] ** 2)) * PX
        rm = np.sqrt(np.mean(e[s] ** 2)) * PX
        rows.append([rm, rms(R['gt_sf_m']), rms(R['gt_sf_i']), rms(R['gt_sm_m']), s.sum() / max(nt, 1), rm / rms(R['gt_sf_m'])])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(config.root('out'), 'gate'))
    ap.add_argument('--scen', default='iso,pair1.5,pair2,rs05x,rs25x,rs25')
    ap.add_argument('--runs', default=':sc')            # tag:label pairs of sc runs
    ap.add_argument('--out', default='')
    ap.add_argument('--bins', default='')               # comma list of photon bin edges (30b: from 150)
    a = ap.parse_args()
    global BINS
    if a.bins: BINS = np.array([float(v) for v in a.bins.split(',')])
    runs = [tuple(r.split(':')) for r in a.runs.split(',')]
    scs = a.scen.split(',')
    fig, axs = plt.subplots(3, len(scs), figsize=(4.0 * len(scs), 9.5), gridspec_kw=dict(height_ratios=[3, 1.3, 1.1]), squeeze=False)
    cen = np.sqrt(BINS[:-1] * BINS[1:])
    for j, sc in enumerate(scs):
        ax, ax3, ax2 = axs[0, j], axs[1, j], axs[2, j]
        first = True
        for k, (tag, lab) in enumerate(runs):
            try:
                R, _ = GR.load(a.dir, sc, tag)
            except Exception:
                continue
            if 'sc_b' not in R: continue
            C = curves(R, 'sc', 'sc')
            col = COL[k % len(COL)]
            if first:   # CRLB lines on this method's matched set
                ax.loglog(cen, C[:, 1], '-', color='k', lw=2, label='CRLB SIMFLUX multi-emitter (same set)')
                ax.loglog(cen, C[:, 2], ':', color='k', lw=1.5, label='CRLB SIMFLUX isolated')
                ax.loglog(cen, C[:, 3], '--', color='0.4', lw=1, label='CRLB SMLM multi-emitter')
                first = False
            ax.loglog(cen, C[:, 0], 'o-', color=col, ms=4, label=lab)
            ax3.semilogx(cen, C[:, 5], 'o-', color=col, ms=4)
            ax2.semilogx(cen, C[:, 4], 'o-', color=col, ms=4)
            if sc == 'iso' and k == 0 and 'mle_b' in R:
                M = curves(R, 'mle', 'mle')
                ax.loglog(cen, M[:, 0], 's--', color='0.55', ms=3, lw=1, label='exact-PSF MLE-SF (sanity)')
                ax3.semilogx(cen, M[:, 5], 's--', color='0.55', ms=3, lw=1)
        ax.set_title(sc + (' (exact model)' if sc.endswith('x') else (' (exact model, paper photons)' if sc.endswith('p') else '')), fontsize=10)
        ax.grid(True, which='both', alpha=0.3); ax.set_ylabel('x/y RMSD (nm)')
        ax3.axhline(1, color='k', lw=0.8); ax3.axhline(1.3, color='0.6', lw=0.8, ls='--')
        ax3.set_ylim(0.7, 5); ax3.set_yscale('log'); ax3.grid(True, which='both', alpha=0.3); ax3.set_ylabel('RMSD / CRLB_SF multi')
        ax2.set_ylim(0, 1.02); ax2.grid(True, alpha=0.3); ax2.set_ylabel('recall (1 px)'); ax2.set_xlabel('true photons per window')
        if j == 0: ax.legend(fontsize=7, loc='lower left')
    if not a.out: a.out = os.path.join(config.out_dir('plots'), 'crlb_sc3.png')
    plt.tight_layout(); plt.savefig(a.out, dpi=90); print('saved', a.out)


if __name__ == '__main__':
    main()
