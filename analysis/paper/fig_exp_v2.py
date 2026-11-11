"""
SIMCODE paper Fig 3 / Fig 4 with SIMCODE v2 added as a 5th method (separate figure; fig_exp.py stays the exact
reproduction).

The four paper methods are produced by fig_exp.plot_results() unchanged (same files, filters, DL drift, FRC, RSP); its
drift caches are reused. SIMCODE v2 is added through the same smlmtorch Dataset code paths (crop / crlb_filter / render /
get_frc / compute_pc), in one of two variants:

  paper  (fig_exp_<name>_v2):  paper-consistent. v2 pass-2 table rows `sel` (phase-calibrated selection, the full output),
         uncertainty filter sigma > 0.15 px rejected using v2's calibrated sx, sy (crlb_filter(0.15) as for every paper method),
         drift = the paper's DME drift estimated on the SIMCODE intensities (sf_intensities, get_drift 0.15) applied at the
         row's frame (= window start, the same convention as the SIMCODE result files).
  own    (fig_exp_<name>_v2own): v2's own strict product (simcode2.strict, per-tile sigma quantile) with its own DME v2 drift
         (analysis/baselines/dme_drift.py, applied at window start + 3), as in docs/RESULTS.md section 1. The other four
         methods are still the paper's (paper drift).

usage: python analysis/paper/fig_exp_v2.py 3|4 paper|own
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_exp as FE
from fig_exp import plt, Dataset, plot_frcs, get_frc, draw_rect, compute_pc, FIGS, PAPER

V2 = {
    3: dict(table=f'{PAPER}/1809_2/v2/run_default/locs_p2.npz', table_p1=f'{PAPER}/1809_2/v2/run_default/locs_p1.npz',
            own_table=f'{PAPER}/1809_2/v2/run_default/locs_p2_owndrift.npz',
            own_drift=f'{PAPER}/1809_2/v2/drift_dmev2_p2.npy', name='lowdensity', figdir=f'{PAPER}/fig3'),
    4: dict(table=f'{PAPER}/2.5nM_2/v2/p2L_real_view/locs_p2.npz', table_p1=f'{PAPER}/2.5nM_2/v2/p1_real_view/locs_p1.npz',
            own_table=f'{PAPER}/2.5nM_2/v2/p2L_real_view/locs_p2_owndrift.npz',
            own_drift=f'{PAPER}/2.5nM_2/v2/drift_dmev2_p2.npy', name='meddensity', figdir=f'{PAPER}/fig4'),
}


def v2_dataset(T, m, imgshape, pixelsize, xcol='x', ycol='y'):
    n = int(m.sum())
    ds = Dataset(n, 2, imgshape)
    ds.data.frame[:] = T['frame'][m]
    ds.data.estim.pos[:, 0] = T[xcol][m]
    ds.data.estim.pos[:, 1] = T[ycol][m]
    ds.data.estim.photons[:] = T['photons'][m]
    ds.data.estim.bg[:] = 0
    ds.data.crlb.pos[:, 0] = T['sx'][m]
    ds.data.crlb.pos[:, 1] = T['sy'][m]
    ds.data.crlb.photons[:] = 0
    ds.data.crlb.bg[:] = 0
    ds.data.roi_id[:] = np.arange(n)
    ds.data.chisq[:] = 0
    ds.data.group[:] = 0
    ds.data.roipos[:] = 0
    ds['pixelsize'] = pixelsize
    return ds


def offset_vs(ds_a, ds_b, zoom=20):
    """integer-pixel (1/zoom px) translation between the renders of two datasets (registration check)"""
    from pearson_corr import find_translation
    a = ds_a.render(zoom=zoom, sigma=2); b = ds_b.render(zoom=zoom, sigma=2)
    dy, dx = find_translation(a, b)
    return dx / zoom, dy / zoom


def main(k, variant):
    cfg = dict(FIGS[k]); vc = V2[k]
    fig_out_paper = cfg.pop('fig_out')
    pixelsize = cfg['pixelsize']
    r = FE.plot_results(**cfg)          # the 4 paper methods, drift-corrected in place (cached DL drift)
    plt.close('all')
    method_ds = [r['smlm'], r['sf'], r['decode'], r['sf_combined']]
    frcs = [r['frcs'][0], r['frcs'][1], r['frcs'][2], r['frcs'][5]]
    imgshape = r['sf_combined'].imgshape
    dl_drift = r['dl_drift']['drift']
    crlb_threshold = 0.15
    frc_kwargs = dict(crlb_filter=crlb_threshold, split_ds=False, use_cache=False, zoom=cfg.get('frc_zoom', 20))
    v2dir = os.path.dirname(vc['table'])

    extra = {}
    if variant == 'paper':
        T = np.load(vc['table'])
        v2 = v2_dataset(T, T['sel'], imgshape, pixelsize)
        v2.applyDrift(dl_drift)
        v2_label = 'SIMCODE v2'
        v2_fn = f'{v2dir}/v2_p2_sel_paperdrift.hdf5'
        # pass 1 numbers (not plotted)
        T1 = np.load(vc['table_p1'])
        v21 = v2_dataset(T1, T1['sel'], imgshape, pixelsize); v21.applyDrift(dl_drift)
        f1 = get_frc(v21, f'{v2dir}/v2_p1_sel_paperdrift.hdf5', **frc_kwargs)
        extra['p1_sel_paperdrift'] = dict(frc_nm=float(f1.frc), count=int(len(v21.crlb_filter(crlb_threshold, inplace=False))))
        out = os.path.join(vc['figdir'], f'fig_exp_{vc["name"]}_v2')
    else:
        T = np.load(vc['own_table'])
        v2 = v2_dataset(T, T['strict'], imgshape, pixelsize, 'x_dc', 'y_dc')
        v2_label = 'SIMCODE v2*'
        v2_fn = f'{v2dir}/v2_p2_strict_owndrift.hdf5'
        out = os.path.join(vc['figdir'], f'fig_exp_{vc["name"]}_v2own')
    v2_frc = get_frc(v2, v2_fn, **frc_kwargs)
    extra['registration_px_vs_SIMCODE'] = offset_vs(r['sf_combined'].crlb_filter(crlb_threshold, inplace=False),
                                                    v2.crlb_filter(crlb_threshold, inplace=False))

    method_ds = method_ds + [v2]
    frcs = frcs + [v2_frc]
    method_labels = ['SMLM', 'SIMFLUX', 'DECODE', 'SIMCODE', v2_label]
    method_colors = ['blue', 'green', 'orange', 'red', 'purple']
    renders = cfg['renders']

    shared_render_args = dict(zoom=40, clip_percentile=96, scalebar_nm=2000, scalebar_fontsize=20, intensities=False,
                              transpose=False, sigma=2)
    shared_render_args.update(cfg.get('render_args') or {})
    full_fov_args = shared_render_args.copy(); zoom = 10
    full_fov_args.update(zoom=zoom, scalebar_nm=2000)

    pc, rmsd, _ = compute_pc(cfg['tif_path'], method_ds)

    # layout: as the paper (overview + zoom 1 on top, zoom 2 in the middle, FRC / counts / RSP below), 5 method columns
    fig = plt.figure(figsize=(20, 19))
    root_gs = fig.add_gridspec(3, 1, hspace=0.12, height_ratios=[1, 0.5, 0.4])
    top_gs = root_gs[0].subgridspec(2, 5, hspace=0.2, wspace=0.1)
    mid_gs = root_gs[1].subgridspec(1, 5, wspace=0.1)
    bottom_gs = root_gs[2].subgridspec(1, 3, wspace=0.25, width_ratios=[1, 0.35, 0.35])
    zoom_linestyles = ['-', '--', '-.']

    ax = fig.add_subplot(top_gs[:2, :2])
    ds = r['sf_combined'][:]
    ds.crlb_filter(crlb_threshold)
    ds.renderFigure(axes=ax, title='Overview (SIMCODE)', **full_fov_args)
    for i, (corner, zoom_width, _) in enumerate(renders):
        rr = draw_rect((corner, (corner[0] + zoom_width, corner[1] + zoom_width)), zoom, ax)
        rr.set_linestyle(zoom_linestyles[i])

    def draw_zoom(gs, zoom_ix, mi):
        ax = fig.add_subplot(gs)
        corner, zoom_width, plot_render_args = renders[zoom_ix]
        ra = shared_render_args.copy(); ra.update(plot_render_args)
        d = method_ds[mi].crop(corner, (corner[0] + zoom_width, corner[1] + zoom_width), silent=True)
        d.crlb_filter(crlb_threshold)
        d.renderFigure(axes=ax, title=method_labels[mi], **ra)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=method_colors[mi], ls=zoom_linestyles[zoom_ix],
                                   lw=6, clip_on=False, transform=ax.transAxes))

    for mi, (rr, cc) in enumerate([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3)]):
        draw_zoom(top_gs[rr, cc], 0, mi)
    for mi in range(5):
        draw_zoom(mid_gs[0, mi], 1, mi)

    frc_ax = fig.add_subplot(bottom_gs[0])
    plot_frcs(frc_ax, frcs, labels=method_labels, colors=method_colors)
    frc_ax.set_title('FRC'); frc_ax.get_yaxis().set_visible(False)

    ax = fig.add_subplot(bottom_gs[1])
    spot_counts = np.array([len(d.crlb_filter(crlb_threshold, inplace=False)) for d in method_ds]) * 1e-6
    ax.bar(method_labels, spot_counts, color=method_colors)
    ax.set_xticks(range(5)); ax.set_xticklabels(method_labels, rotation=45, ha='right')
    ax.set_ylabel('Spot count (${10}^6$)'); ax.set_title('# Localizations')
    ax.set_ylim([0, max(4, 1.1 * spot_counts.max())])

    ax = fig.add_subplot(bottom_gs[2])
    ax.bar(method_labels, pc, color=method_colors)
    ax.set_xticks(range(5)); ax.set_xticklabels(method_labels, rotation=45, ha='right')
    ax.set_title('Pearson corr.')

    if variant == 'own':
        fig.text(0.5, 0.005, 'SIMCODE v2* = v2 strict product (per-tile sigma quantile) with its own DME v2 drift; '
                 'the other methods use the paper processing (DME drift from the SIMCODE intensities).',
                 ha='center', fontsize=13)
    else:
        fig.text(0.5, 0.005, 'SIMCODE v2 = pass-2 phase-calibrated selection, sigma > 0.15 px rejected, paper DME drift '
                 '(from the SIMCODE intensities), as for all methods.', ha='center', fontsize=13)
    plt.savefig(out + '.svg', dpi=300)
    plt.savefig(out + '.png', dpi=150)
    plt.close('all')

    vals = dict(labels=method_labels, frc_nm=[float(f.frc) for f in frcs], spot_counts=(spot_counts * 1e6).astype(int).tolist(),
                pearson=np.asarray(pc).tolist(), variant=variant, v2_table=vc['table'] if variant == 'paper' else vc['own_table'],
                **extra)
    print(json.dumps(vals, indent=1))
    json.dump(vals, open(out + '_values.json', 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
