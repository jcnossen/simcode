"""Paper Fig 3 / Fig 4 (+ the 20 nM movie) regenerated for the 4 reprocessed methods (SMLM, SIMFLUX, DECODE, SIMCODE v2) with
the fair FRC (frc_fair.py). Layout as the paper's plot_results (fig_exp.py): overview (SIMCODE v2) with the 2 zoom boxes, the 2
zooms x 4 methods, FRC curves, localization counts, Pearson correlation (RSP) with the widefield mean image (fig_exp.compute_pc
= the paper's pearson_corr.compute_correlation, read-only from ~/smlmtorch/scripts).

What every panel shows (all methods identical): the frc_fair set = disjoint windows (start % 6 == 0), max(sigma) < 0.15 px,
crop [5, 115) px, each method's own DME v2 drift. Counts = that set's size (the rows in the FRC). RSP = compute_pc on that set.
Zoom AOIs: the paper's (fig_exp.FIGS) for 1809/2 and 2.5 nM/2; 20 nM/2: analysis/figures/zoompick.py crossings, picked once
on the union of the 4 methods' sets (the same boxes for every method; written to <out>_values.json).

usage: CUDA_VISIBLE_DEVICES=1 python analysis/paper/fig_exp_new.py <1809_2|2.5nM_2|20nM_2> [--nframes N] [--tag t]
          [--smlm p] [--sf p] [--decode p] [--v2 p] [--aoi auto]
  defaults: out/paper2/<ds>/{smlm/smlm.npy, sf/sf.hdf5, decode/locs.npz, v2/locs_p2.npz}; output out/paper2/fig/<ds>[_<tag>].*"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(REPO, 'analysis', 'figures')]
import fig_exp as FE                          # paper helpers (draw_rect, compute_pc) and rc params
from fig_exp import plt, Dataset, draw_rect, compute_pc
import frc_fair as FF

P2 = f'{REPO}/out/paper2'
SETS = {
    '1809_2': dict(renders=FE.FIGS[3]['renders'], render_args=FE.FIGS[3]['render_args'], nframes=120000,
                   tif=f'{P2}/1809_2/movie/simflux_5.5ms_500mW.tif', title='Low density (1809/cover1/2)'),
    '2.5nM_2': dict(renders=FE.FIGS[4]['renders'], render_args=FE.FIGS[4]['render_args'], nframes=180000,
                    tif=f'{P2}/2.5nM_2/movie/cover1_600mW_5.5ms_cf660.tif', title='Medium density (2.5 nM/2)'),
    '20nM_2': dict(renders=None, render_args={'clip_percentile': 98}, nframes=120000,
                   tif=f'{P2}/20nM_2/movie/cover1_500mW_5.5ms_cf660.tif', title='High density (20 nM/2)'),
}
LABELS = ['SMLM', 'SIMFLUX', 'DECODE', 'SIMCODE']
KINDS = ['smlm', 'sf', 'decode', 'v2']
COLORS = ['blue', 'green', 'orange', 'red']        # the paper's method_colors


def to_dataset(x, y, f, sx, sy, ph, imgshape=(120, 120), px=FF.PX):
    n = len(x)
    ds = Dataset(n, 2, imgshape)
    ds.data.frame[:] = f; ds.data.estim.pos[:, 0] = x; ds.data.estim.pos[:, 1] = y
    ds.data.estim.photons[:] = ph; ds.data.estim.bg[:] = 0
    ds.data.crlb.pos[:, 0] = sx; ds.data.crlb.pos[:, 1] = sy; ds.data.crlb.photons[:] = 0; ds.data.crlb.bg[:] = 0
    ds.data.roi_id[:] = np.arange(n); ds.data.chisq[:] = 0; ds.data.group[:] = 0; ds.data.roipos[:] = 0
    ds['pixelsize'] = px
    return ds


def pick_aois(sets, sizes=(10.0, 15.0)):
    from zoompick import pick_crossings
    x = np.concatenate([d.pos[:, 0] for d in sets]); y = np.concatenate([d.pos[:, 1] for d in sets])
    out = []
    for sz in sizes:
        b = pick_crossings(x, y, n=4, size=sz)
        for (x0, x1, y0, y1) in b:     # first box not overlapping the ones already chosen
            if all(max(abs(x0 - c[0][0]), abs(y0 - c[0][1])) > max(sz, c[1]) for c in out):
                out.append(((int(round(x0)), int(round(y0))), int(sz), {'scalebar_nm': 500, 'clip_percentile': 98})); break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(SETS))
    ap.add_argument('--nframes', type=int, default=None)
    ap.add_argument('--tag', default='')
    for k in KINDS: ap.add_argument('--' + k, default=None)
    ap.add_argument('--aoi', default=None, help="'auto' = zoompick crossings (default for 20nM_2)")
    a = ap.parse_args()
    S = SETS[a.dataset]; F = a.nframes or S['nframes']
    dflt = dict(smlm=f'{P2}/{a.dataset}/smlm/smlm.npy', sf=f'{P2}/{a.dataset}/sf/sf.hdf5',
                decode=f'{P2}/{a.dataset}/decode/locs.npz', v2=f'{P2}/{a.dataset}/v2/locs_p2.npz')
    paths = {k: getattr(a, k) or dflt[k] for k in KINDS}
    name = a.dataset + (f'_{a.tag}' if a.tag else '')
    out = f'{P2}/fig/{name}'; os.makedirs(f'{P2}/fig', exist_ok=True)
    res, dsets = [], []
    for lab, k in zip(LABELS, KINDS):
        T = FF.load(k, paths[k])
        keep = T['f'] + 6 <= F
        T = {kk: v[keep] for kk, v in T.items()}
        R = FF.evaluate(T, F, drift_cache=f'{out}_{k}_drift.npy')
        m = R['mask']
        dsets.append(to_dataset(R['x_dc'][m], R['y_dc'][m], T['f'][m], T['sx'][m], T['sy'][m], T['photons'][m]))
        res.append(R)
        print(lab, json.dumps(FF.summary(R)), flush=True)

    renders = S['renders'] if (S['renders'] is not None and a.aoi != 'auto') else pick_aois(dsets)
    shared = dict(zoom=40, clip_percentile=96, scalebar_nm=2000, scalebar_fontsize=20, intensities=False, transpose=False, sigma=2)
    shared.update(S['render_args'] or {})
    full = dict(shared); zoom = 10; full.update(zoom=zoom, scalebar_nm=2000)
    pc, rmsd, _ = compute_pc(S['tif'], dsets)

    fig = plt.figure(figsize=(14, 20))
    root = fig.add_gridspec(3, 1, hspace=0.1, wspace=0.1, height_ratios=[1, 1, 0.4])
    top = root[0].subgridspec(2, 4, hspace=0.2, wspace=0.1); mid = root[1].subgridspec(2, 4, hspace=0.2, wspace=0.1)
    bot = root[2].subgridspec(1, 3, hspace=0.1, wspace=0.2, width_ratios=[1, 0.3, 0.3])
    styles = ['-', '--', '-.']
    ax = fig.add_subplot(top[:2, :2])
    dsets[3].renderFigure(axes=ax, title='Overview', **full)
    for i, (c, w, _) in enumerate(renders):
        draw_rect((c, (c[0] + w, c[1] + w)), zoom, ax).set_linestyle(styles[i])

    def draw_zoom(gs, zi, mi):
        ax = fig.add_subplot(gs)
        c, w, ra = renders[zi]
        r = dict(shared); r.update(ra)
        dsets[mi].crop(c, (c[0] + w, c[1] + w), silent=True).renderFigure(axes=ax, title=LABELS[mi], **r)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS[mi], ls=styles[zi], lw=6, clip_on=False,
                                   transform=ax.transAxes))
    for zi, gs in enumerate([top, mid]):
        for mi, (r_, c_) in enumerate([(0, 2), (0, 3), (1, 2), (1, 3)]):
            draw_zoom(gs[r_, c_], zi, mi)

    ax = fig.add_subplot(bot[0])
    ax.axhline(1 / 7, color='grey', linestyle='--')
    for mi, R in enumerate(res):
        sel = (R['freq'] > 0.002) & (R['freq'] < 0.1)
        ax.plot(R['freq'][sel], R['curve'][sel], color=COLORS[mi], label=f'{LABELS[mi]} (FRC={R["frc_nm"]:.1f} nm)')
    ax.legend(fontsize=13); ax.set_yticklabels([]); ax.set_xlabel('Spatial freq. [nm$^{-1}$]'); ax.set_title('FRC')
    ax.get_yaxis().set_visible(False)
    ax = fig.add_subplot(bot[1])
    cnt = np.array([R['n_frc'] for R in res]) * 1e-6
    ax.bar(LABELS, cnt, color=COLORS); ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=45)
    ax.set_ylabel('Spot count (${10}^6$)'); ax.set_title('# Localizations')
    ax = fig.add_subplot(bot[2])
    ax.bar(LABELS, pc, color=COLORS); ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=45); ax.set_title('Pearson corr.')
    fig.text(0.5, 0.005, 'All methods: disjoint 6-frame windows, sigma < 0.15 px, own DME v2 drift; FRC with alternating '
             '300-frame halves.' + (f' Frames 0-{F}.' if a.nframes else ''), ha='center', fontsize=12)
    plt.savefig(out + '.svg', dpi=300); plt.savefig(out + '.png', dpi=150); plt.close('all')
    vals = dict(dataset=a.dataset, nframes=F, labels=LABELS, paths=paths, frc_nm=[R['frc_nm'] for R in res],
                n_frc=[R['n_frc'] for R in res], n_disjoint=[R['n_disjoint'] for R in res], n_all=[R['n_all'] for R in res],
                frc_nm_nofilter=[R['frc_nm_nofilter'] for R in res], frc_nm_30frame_halves=[R['frc_nm_30frame_halves'] for R in res],
                pearson=np.asarray(pc).tolist(), rmsd=np.asarray(rmsd).tolist(), drift_ptp_nm=[R['drift_ptp_nm'] for R in res],
                drift_info=[R['drift_info'] for R in res], aois=[[list(c), w] for c, w, _ in renders],
                protocol=dict(sigma_max_px=FF.SIGMA_MAX, grid=FF.GRID, halves_frames=FF.BLK, crop=FF.CROP))
    json.dump(vals, open(out + '_values.json', 'w'), indent=1)
    print(json.dumps({k: vals[k] for k in ('frc_nm', 'n_frc', 'pearson', 'aois')}), flush=True)


if __name__ == '__main__':
    main()
