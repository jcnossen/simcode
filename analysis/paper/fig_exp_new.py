"""Paper Fig 3 / Fig 4 (+ the 20 nM movie) regenerated for the 4 reprocessed methods (SMLM, SIMFLUX, DECODE, SIMCODE v2) with
the fair FRC (frc_fair.py). Layout as the preprint's experimental figures: overview (SIMCODE v2) with the 2 zoom boxes, the 2
zooms x 4 methods, FRC curves, localization counts, Pearson correlation (RSP) with the widefield mean image (paper_common.compute_pc
= the paper's pearson_corr.compute_correlation, read-only from ~/smlmtorch/scripts).

What every panel shows (all methods identical): the frc_fair set = disjoint windows (start % 6 == 0), max(sigma) < 0.15 px,
crop [5, 115) px, each method's own DME v2 drift. Counts = that set's size (the rows in the FRC). RSP = compute_pc on that set.
Zoom AOIs: the preprint's (paper_common.ZOOMS) for 1809/2 and 2.5 nM/2; 20 nM/2: analysis/figures/zoompick.py crossings, picked once
on the union of the 4 methods' sets (the same boxes for every method; written to <out>_values.json).

usage: CUDA_VISIBLE_DEVICES=1 python analysis/paper/fig_exp_new.py <1809_2|2.5nM_2|20nM_2> [--nframes N] [--tag t]
          [--smlm p] [--sf p] [--decode p] [--v2 <run dir or table>] [--v2_sigma_max nm] [--aoi auto]
  defaults: out/paper2/<ds>/{smlm/smlm.npy, sf/sf.hdf5, decode/locs.npz, v2 (run directory: locs_p2)}; v2 rows: sel, or with
  --v2_sigma_max the paper's selection at that threshold (Dataset.select); output out/paper2/fig/<ds>[_<tag>].*
  (2026-11-15) --outdir d: output directory instead of out/paper2/fig; --reuse_drift <prefix>: the methods other than v2 take
  their drift from an earlier figure's caches <prefix>_<kind>_drift.npy (copied to this figure's cache names; the same rows give
  the same DME v2 drift); --psf <calib json>: PSF panel (psfpanel.py: xy at the plotted v2 set's median height + xz, rms
  aberration, SMLM / SIMFLUX CRLB at that set's median photons, the run's pass-1 background and the camera read noise;
  psfdata.py), the v2 argument must then be a run directory"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(REPO, 'analysis', 'figures')]
from paper_common import plt, Dataset, draw_rect, compute_pc, ZOOMS     # also sets the matplotlib rc params
import frc_fair as FF
import shutil

P2 = f'{REPO}/out/paper2'
SETS = {
    '1809_2': dict(**ZOOMS['1809_2'], nframes=120000,
                   tif=f'{P2}/1809_2/movie/simflux_5.5ms_500mW.tif', title='Low density (1809/cover1/2)'),
    '2.5nM_2': dict(**ZOOMS['2.5nM_2'], nframes=180000,
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


def torch_mask(m):
    import torch
    return torch.from_numpy(np.asarray(m, bool))


def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o


def panel_text(info, cond):
    """the PSF panel's own label (short; everything else goes into the caption: caption_text)"""
    return f"In-situ vectorial PSF fit\nrms aberration {info['rms_mwaves']:.0f} mλ"


def caption_text(info, cond, foot):
    """caption material for the PSF panel and the processing protocol (written next to the figure as <out>_caption.txt)"""
    c = info['crlb']; p = cond['pattern']
    rn_sd = cond['rn2_e'] ** 0.5
    L = [f"PSF panel: in-situ vectorial PSF fit (Zernike Noll 5-21), xy section at the emitters' median height "
         f"({cond['z_median_um'] * 1e3:.0f} nm) and xz section, log scale; rms aberration over Noll >= 5 {info['rms_mwaves']:.0f} mλ.",
         f"Cramér-Rao lower bounds for a single isolated emitter at the data's median signal ({info['detected_photons']:.0f} detected photons "
         f"per 6-frame window), background {cond['bg_px_frame']:.2f} photons/px/frame and camera read noise {rn_sd:.1f} e- rms: "
         f"SMLM x/y/z {c['smlm']['x']:.1f}/{c['smlm']['y']:.1f}/{c['smlm']['z']:.0f} nm, "
         f"SIMFLUX (pattern pitch {p['pitch_nm']:.0f} nm, depth {p['depth']:.2f}, 2 x 3 phase steps) "
         f"x/y/z {c['sf']['x']:.1f}/{c['sf']['y']:.1f}/{c['sf']['z']:.0f} nm.",
         foot]
    return '\n'.join(L)


def main():
    LABELS, KINDS, COLORS = list(globals()['LABELS']), list(globals()['KINDS']), list(globals()['COLORS'])
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=list(SETS))
    ap.add_argument('--nframes', type=int, default=None)
    ap.add_argument('--tag', default='')
    for k in KINDS: ap.add_argument('--' + k, default=None)
    ap.add_argument('--v1', default=None, help="opt-in 5th method: the preprint's SIMCODE localizations (merged table, e.g. "
                    "out/paper/<ds>/published/sfhd-ndi+smlm.hdf5), through the same fair protocol; v2 is then labelled 'SIMCODE v2'")
    ap.add_argument('--note', default='', help='extra footnote line (e.g. a provisional marker)')
    ap.add_argument('--v2_sigma_max', type=float, default=None, help="nm: v2 rows = the paper's selection at this threshold (default: the rows sel)")
    ap.add_argument('--aoi', default=None, help="'auto' = zoompick crossings (default for 20nM_2)")
    ap.add_argument('--outdir', default=None, help='output directory (default out/paper2/fig)')
    ap.add_argument('--reuse_drift', default=None, help='prefix of earlier drift caches for the methods other than v2')
    ap.add_argument('--psf', default=None, help='calib json of the PSF panel (needs --v2 = a run directory)')
    a = ap.parse_args()
    S = SETS[a.dataset]; F = a.nframes or S['nframes']
    dflt = dict(smlm=f'{P2}/{a.dataset}/smlm/smlm.npy', sf=f'{P2}/{a.dataset}/sf/sf.hdf5',
                decode=f'{P2}/{a.dataset}/decode/locs.npz', v2=f'{P2}/{a.dataset}/v2')
    paths = {k: getattr(a, k) or dflt[k] for k in KINDS}       # the 4 reprocessed methods
    name = a.dataset + (f'_{a.tag}' if a.tag else '')
    od = a.outdir or f'{P2}/fig'
    out = f'{od}/{name}'; os.makedirs(od, exist_ok=True)
    if a.v1:
        LABELS[3:] = ['SIMCODE (preprint)', 'SIMCODE v2']; KINDS[3:] = ['v1', 'v2']; COLORS[3:] = ['red', 'purple']
        paths['v1'] = a.v1
    NM = len(LABELS)
    res, dsets = [], []
    for lab, k in zip(LABELS, KINDS):
        T = FF.load('sf' if k == 'v1' else k, paths[k], **(dict(v2_sigma_max=a.v2_sigma_max) if k == 'v2' else {}))
        keep = T['f'] + 6 <= F
        T = {kk: v[keep] for kk, v in T.items()}
        if a.reuse_drift and k != 'v2' and not os.path.exists(f'{out}_{k}_drift.npy'):
            shutil.copyfile(f'{a.reuse_drift}_{k}_drift.npy', f'{out}_{k}_drift.npy')
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
    psf_cond = psf_info = None
    if a.psf:
        import psfdata
        from simcode.dataset import Dataset as SD
        V = SD.load(paths['v2']); V = V.select(a.v2_sigma_max) if a.v2_sigma_max is not None else V[V.sel]
        T = FF.load('v2', paths['v2'], v2_sigma_max=a.v2_sigma_max); keep = T['f'] + 6 <= F
        V = V[torch_mask(keep)][torch_mask(res[KINDS.index('v2')]['mask'])]       # the plotted set (= the FRC rows)
        psf_cond = psfdata.data_conditions(paths['v2'], V, a.psf)
        psf_info = psfdata.panel_info(a.psf, psf_cond)

    # layout: A overview | zoom 1 (one panel per method) ; zoom 2 (one row) ; FRC, counts, Pearson. Panel letters run in the
    # preprint's order (overview, zoom 1 methods, zoom 2 methods, FRC, counts, Pearson).
    letters = iter('ABCDEFGHIJKLMNOPQ')
    def letter(ax, dx=-0.02, dy=1.03):
        ax.text(dx, dy, next(letters), transform=ax.transAxes, fontsize=20, fontweight='bold', ha='right', va='bottom')
    zshared = dict(shared, scalebar_fontsize=12)
    nc = 2 if NM <= 4 else 3
    fig = plt.figure(figsize=(14, 16.5 if NM <= 4 else 15.5))
    root = fig.add_gridspec(3, 1, hspace=0.16, height_ratios=[1, 0.5 if NM <= 4 else 0.42, 0.42])
    top = root[0].subgridspec(2, 2 + nc, hspace=0.18, wspace=0.1)
    mid = root[1].subgridspec(1, NM, wspace=0.1)
    if psf_info is None:
        bot = root[2].subgridspec(1, 3, wspace=0.42, width_ratios=[1, 0.42, 0.42])
    else:
        bot = root[2].subgridspec(1, 4, wspace=0.42, width_ratios=[1, 0.42, 0.42, 1.0])
    styles = ['-', '--', '-.']
    ax = fig.add_subplot(top[:2, :2])
    dsets[-1].renderFigure(axes=ax, title='Overview' + (' (SIMCODE v2)' if a.v1 else ''), **full)
    letter(ax)
    for i, (c, w, _) in enumerate(renders):
        draw_rect((c, (c[0] + w, c[1] + w)), zoom, ax).set_linestyle(styles[i])

    def draw_zoom(gs, zi, mi):
        ax = fig.add_subplot(gs)
        c, w, ra = renders[zi]
        r = dict(zshared); r.update(ra)
        dsets[mi].crop(c, (c[0] + w, c[1] + w), silent=True).renderFigure(axes=ax, title=LABELS[mi], **r)
        ax.set_title(''); ax.set_title('      ' + LABELS[mi], loc='left', fontsize=14)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS[mi], ls=styles[zi], lw=5, clip_on=False,
                                   transform=ax.transAxes))
        ax.text(0.0, 1.03, next(letters), transform=ax.transAxes, fontsize=20, fontweight='bold', ha='left', va='bottom')
    for mi in range(NM): draw_zoom(top[mi // nc, 2 + mi % nc], 0, mi)
    for mi in range(NM): draw_zoom(mid[0, mi], 1, mi)

    short = [l.replace(' (preprint)', '\n(preprint)').replace('SIMCODE v2', 'SIMCODE\nv2') for l in LABELS]
    ax = fig.add_subplot(bot[0])
    ax.axhline(1 / 7, color='grey', linestyle='--')
    for mi, R in enumerate(res):
        sel = (R['freq'] > 0.002) & (R['freq'] < 0.1)
        ax.plot(R['freq'][sel], R['curve'][sel], color=COLORS[mi], label=f'{LABELS[mi]}: {R["frc_nm"]:.1f} nm')
    ax.legend(fontsize=11, loc='upper right', title='FRC resolution', title_fontsize=11, framealpha=0.9)
    ax.set_xlabel('Spatial freq. [nm$^{-1}$]'); ax.set_title('FRC'); ax.get_yaxis().set_visible(False); letter(ax)
    ax = fig.add_subplot(bot[1])
    cnt = np.array([R['n_frc'] for R in res]) * 1e-6
    ax.bar(range(NM), cnt, color=COLORS); ax.set_xticks(range(NM)); ax.set_xticklabels(short, rotation=90, fontsize=10, linespacing=0.85)
    ax.set_title('Localizations\n($10^6$)', fontsize=13); letter(ax, dx=-0.3)
    ax = fig.add_subplot(bot[2])
    ax.bar(range(NM), pc, color=COLORS); ax.set_xticks(range(NM)); ax.set_xticklabels(short, rotation=90, fontsize=10, linespacing=0.85)
    ax.set_title('RSP', fontsize=13); letter(ax, dx=-0.3)
    if psf_info is not None:
        import psfpanel
        g = bot[3].subgridspec(2, 2, height_ratios=[1.0, 0.3], hspace=0.15, wspace=0.55)  # xy | xz, same size; label below
        axy, axz, axt = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1]), fig.add_subplot(g[1, :])
        psfpanel.draw(axy, axz, psf_info, fontsize=8)
        letter(axy, dx=-0.12, dy=1.12)
        axt.axis('off')
        axt.text(-0.05, 0.98, panel_text(psf_info, psf_cond), transform=axt.transAxes, fontsize=10, va='top', ha='left', linespacing=1.2)
    foot = ('All methods: disjoint 6-frame windows, sigma < 0.15 px, own DME v2 drift; FRC with alternating 300-frame halves.'
            + (f' Frames 0-{F}.' if a.nframes else '') + (f'\n{a.note}' if a.note else ''))
    if psf_info is not None:
        open(f'{out}_caption.txt', 'w').write(caption_text(psf_info, psf_cond, foot) + '\n')
    else:
        open(f'{out}_caption.txt', 'w').write(foot + '\n')
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', bbox_inches='tight', pad_inches=0.15, **kw)
    plt.close('all')
    vals = dict(dataset=a.dataset, nframes=F, labels=LABELS, paths=paths, frc_nm=[R['frc_nm'] for R in res],
                n_frc=[R['n_frc'] for R in res], n_disjoint=[R['n_disjoint'] for R in res], n_all=[R['n_all'] for R in res],
                frc_nm_nofilter=[R['frc_nm_nofilter'] for R in res], frc_nm_30frame_halves=[R['frc_nm_30frame_halves'] for R in res],
                pearson=np.asarray(pc).tolist(), rmsd=np.asarray(rmsd).tolist(), drift_ptp_nm=[R['drift_ptp_nm'] for R in res],
                drift_info=[R['drift_info'] for R in res], note=a.note, aois=[[list(c), w] for c, w, _ in renders],
                protocol=dict(sigma_max_px=FF.SIGMA_MAX, grid=FF.GRID, halves_frames=FF.BLK, crop=FF.CROP))
    if psf_info is not None:
        vals['psf_panel'] = dict(conditions=psf_cond, rms_mwaves=psf_info['rms_mwaves'], z_um=psf_info['z_um'],
                                 collected=psf_info['collected'], intensity_param=psf_info['intensity_param'],
                                 detected_photons=psf_info['detected_photons'],
                                 crlb_nm=jsonable(psf_info['crlb']), text=panel_text(psf_info, psf_cond))
    if a.reuse_drift: vals['drift_reused_from'] = a.reuse_drift
    json.dump(vals, open(out + '_values.json', 'w'), indent=1)
    print(json.dumps({k: vals[k] for k in ('frc_nm', 'n_frc', 'pearson', 'aois')}), flush=True)


if __name__ == '__main__':
    main()
