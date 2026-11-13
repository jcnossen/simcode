"""Side-by-side renders of the paper zoom boxes, u <= 0.45 vs the global precision rule (sigma_max), for 1809_2 and 2.5nM_2;
plus where the two selections differ: maps of the rows kept only by u <= 0.45 / only by the global rule, and the per-2-px-tile
count ratio (global / u045) against the local u045 row density and the local median photons.
Rows shown = the fair-FRC set of each selection (as fig_exp_new.py: disjoint windows, max sigma < 0.15 px, crop, own DME v2
drift = the cached drifts of the figure runs), rendered with the figure's settings.
usage: python analysis/selection_global/render_compare.py [--sigma_max 12] -> out/selection_global/render_compare_<ds>.png, zoomstats.json"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'analysis', 'paper'), os.path.join(REPO, 'analysis', 'figures')]
from paper_common import plt, ZOOMS
import frc_fair as FF
from fig_exp_new import to_dataset, SETS
from simcode.util.proctitle import set_title

P2 = f'{REPO}/out/paper2'; OUT = f'{REPO}/out/selection_global'
RUN = {'1809_2': ('1809_2/v2_matched', 'fig3'), '2.5nM_2': ('2.5nM_2/v2_matched', 'fig4')}


def frcset(path, F, drift):
    T = FF.load('v2', path); keep = T['f'] + 6 <= F; T = {k: v[keep] for k, v in T.items()}
    R = FF.evaluate(T, F, drift_cache=drift, secondary=False); m = R['mask']
    return dict(x=R['x_dc'][m], y=R['y_dc'][m], f=T['f'][m], sx=T['sx'][m], sy=T['sy'][m], ph=T['photons'][m], xr=T['x'][m], yr=T['y'][m]), R['frc_nm']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sigma_max', type=float, default=12.0); a = ap.parse_args()
    set_title('sel-render')
    plt.rcParams.update({'font.size': 10})
    tag = f'sig{a.sigma_max:g}'.replace('.', 'p')
    stats = {}
    for ds, (run, fig) in RUN.items():
        F = SETS[ds]['nframes']
        A, fa = frcset(f'{P2}/{run}/locs_p2_u045.npz', F, f'{P2}/fig/{ds}_{fig}_matched_u045_v2_drift.npy')
        B, fb = frcset(f'{P2}/{run}/locs_p2_{tag}.npz', F, f'{P2}/fig/{ds}_{fig}_matched_{tag}_v2_drift.npy')
        dsA = to_dataset(A['x'], A['y'], A['f'], A['sx'], A['sy'], A['ph']); dsB = to_dataset(B['x'], B['y'], B['f'], B['sx'], B['sy'], B['ph'])
        # rows only in one set: key = (window start, raw x) (the same pipeline rows)
        ka = A['f'].astype(np.int64) * 10 ** 9 + np.round(A['xr'] * 1e5).astype(np.int64); kb = B['f'].astype(np.int64) * 10 ** 9 + np.round(B['xr'] * 1e5).astype(np.int64)
        onlyA = ~np.isin(ka, kb); onlyB = ~np.isin(kb, ka)
        boxes = ZOOMS[ds]['renders']
        shared = dict(zoom=40, clip_percentile=96, scalebar_nm=500, scalebar_fontsize=11, intensities=False, transpose=False, sigma=2)
        fig_ = plt.figure(figsize=(17, 4.6 * len(boxes) + 4.6))
        gs = fig_.add_gridspec(len(boxes) + 1, 4, hspace=0.3, wspace=0.35)
        zs = []
        for i, (c, w, ra) in enumerate(boxes):
            r = dict(shared); r.update(ra)
            for j, (d, lab) in enumerate(((dsA, f'u ≤ 0.45 (FRC {fa:.1f} nm, {len(A["x"]) / 1e6:.2f} M)'), (dsB, f'max σ ≤ {a.sigma_max:g} nm (FRC {fb:.1f} nm, {len(B["x"]) / 1e6:.2f} M)'))):
                ax = fig_.add_subplot(gs[i, j]); d.crop(c, (c[0] + w, c[1] + w), silent=True).renderFigure(axes=ax, **r)
                ax.set_title(f'zoom {i + 1}: {lab}', fontsize=10)
            inb = lambda S: (S['x'] >= c[0]) & (S['x'] < c[0] + w) & (S['y'] >= c[1]) & (S['y'] < c[1] + w)
            ax = fig_.add_subplot(gs[i, 2])
            for S, k, col, lab in ((A, onlyA, '#eb6834', 'only u ≤ 0.45'), (B, onlyB, '#2a78d6', f'only σ ≤ {a.sigma_max:g} nm')):
                m = k & inb(S); ax.plot(S['x'][m], S['y'][m], '.', ms=0.6, color=col, alpha=0.5, label=f'{lab}: {m.sum()}')
            for g_ in np.arange(5, 116, 10):                # the 10-px tiles of the per-tile rank (u)
                ax.axvline(g_, color='#888', lw=0.6, ls=':'); ax.axhline(g_, color='#888', lw=0.6, ls=':')
            ax.set_xlim(c[0], c[0] + w); ax.set_ylim(c[1] + w, c[1]); ax.set_aspect('equal'); ax.legend(fontsize=8, markerscale=12, loc='upper right')
            ax.set_title(f'zoom {i + 1}: rows in one selection only (dotted: 10-px u tiles)', fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            nA, nB = int(inb(A).sum()), int(inb(B).sum())
            zs.append(dict(box=[list(c), w], n_u045=nA, n_global=nB, ratio=nB / max(nA, 1), only_u045=int((onlyA & inb(A)).sum()), only_global=int((onlyB & inb(B)).sum()),
                           median_sigma_u045_nm=float(np.median(np.maximum(A['sx'], A['sy'])[inb(A)]) * FF.PX), median_sigma_global_nm=float(np.median(np.maximum(B['sx'], B['sy'])[inb(B)]) * FF.PX)))
            ax = fig_.add_subplot(gs[i, 3]); ax.axis('off')
            ax.text(0, 0.9, '\n'.join(f'{k}: {v:.3g}' if isinstance(v, float) else f'{k}: {v}' for k, v in zs[-1].items()), fontsize=9, va='top', family='monospace')
        # whole field: per-2-px-tile ratio global / u045 vs local density and photons
        t = 2.0; nb = int(110 / t)
        def grid(S, v=None):
            ix = np.clip(((S['x'] - 5) / t).astype(int), 0, nb - 1); iy = np.clip(((S['y'] - 5) / t).astype(int), 0, nb - 1)
            return np.bincount(iy * nb + ix, weights=v, minlength=nb * nb).reshape(nb, nb)
        gA, gB = grid(A), grid(B); phA = grid(A, A['ph']) / np.maximum(gA, 1)
        ratio = np.where(gA > 20, gB / np.maximum(gA, 1), np.nan)
        ax = fig_.add_subplot(gs[-1, 0]); im = ax.imshow(np.log2(ratio), cmap='RdBu_r', vmin=-1.5, vmax=1.5, extent=[5, 115, 115, 5])
        plt.colorbar(im, ax=ax, fraction=0.046).set_label('log2 count ratio', fontsize=9)
        for c, w, _ in boxes: ax.add_patch(plt.Rectangle(c, w, w, fill=False, ec='k', lw=1.5))
        ax.set_title('log2 count(σ rule) / count(u ≤ 0.45)\nper 2-px tile (> 20 u045 rows)', fontsize=10)
        ok = np.isfinite(ratio)
        ax = fig_.add_subplot(gs[-1, 1]); ax.scatter(gA[ok], ratio[ok], s=3, alpha=0.4, color='#2a78d6'); ax.set_xscale('log'); ax.set_yscale('log')
        ax.axhline(len(B['x']) / len(A['x']), color='k', lw=1, ls='--'); ax.set_xlabel('u ≤ 0.45 rows per tile (local density)'); ax.set_ylabel('count ratio'); ax.set_title('ratio vs local density', fontsize=10)
        ax = fig_.add_subplot(gs[-1, 2]); ax.scatter(phA[ok], ratio[ok], s=3, alpha=0.4, color='#1baf7a'); ax.set_yscale('log')
        ax.axhline(len(B['x']) / len(A['x']), color='k', lw=1, ls='--'); ax.set_xlabel('median photons per row in the tile (u ≤ 0.45 rows)'); ax.set_title('ratio vs local brightness', fontsize=10)
        q = np.nanpercentile(ratio, [10, 50, 90])
        lo_d = ok & (gA <= np.nanpercentile(gA[ok], 25)); hi_d = ok & (gA >= np.nanpercentile(gA[ok], 75))
        lo_p = ok & (phA <= np.nanpercentile(phA[ok], 25)); hi_p = ok & (phA >= np.nanpercentile(phA[ok], 75))
        st = dict(frc_u045=fa, frc_global=fb, n_u045=len(A['x']), n_global=len(B['x']), overall_ratio=len(B['x']) / len(A['x']),
                  tile_ratio_p10_p50_p90=q.tolist(), ratio_median_low_density_q1=float(np.nanmedian(ratio[lo_d])), ratio_median_high_density_q4=float(np.nanmedian(ratio[hi_d])),
                  ratio_median_dim_q1=float(np.nanmedian(ratio[lo_p])), ratio_median_bright_q4=float(np.nanmedian(ratio[hi_p])),
                  spearman_ratio_vs_density=float(np.corrcoef(np.argsort(np.argsort(gA[ok])), np.argsort(np.argsort(ratio[ok])))[0, 1]),
                  spearman_ratio_vs_photons=float(np.corrcoef(np.argsort(np.argsort(phA[ok])), np.argsort(np.argsort(ratio[ok])))[0, 1]), zooms=zs)
        ax = fig_.add_subplot(gs[-1, 3]); ax.axis('off')
        ax.text(0, 0.95, '\n'.join(f'{k}: {np.round(v, 3).tolist() if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v)}' for k, v in st.items() if k != 'zooms'), fontsize=9, va='top', family='monospace')
        fig_.suptitle(f'{ds}: u ≤ 0.45 vs global max(σx, σy) ≤ {a.sigma_max:g} nm (fair-FRC sets, own DME v2 drift)', fontsize=13)
        fn = f'{OUT}/render_compare_{ds}.png'; plt.savefig(fn, dpi=110, bbox_inches='tight'); plt.close('all')
        stats[ds] = st; print(ds, json.dumps({k: v for k, v in st.items() if k != 'zooms'}), flush=True)
        for z in zs: print('  ', json.dumps(z), flush=True)
    json.dump(stats, open(f'{OUT}/zoomstats_{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
