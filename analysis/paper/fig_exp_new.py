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
  defaults: out/paper2/<ds>/{smlm/smlm.npy, sf/sf.hdf5, decode/locs.npz (locs_fixed.npz when it exists), v2 (run directory: locs_p2)}; v2 rows: sel, or with
  --v2_sigma_max the paper's selection at that threshold (Dataset.select); output out/paper2/fig/<ds>[_<tag>].*
  (2026-11-15) paper version (no --v1): labels 'SIMCODE', caption / PSF table without version labels (paper_text); a redraw with
  the same --outdir / --tag reuses every <out>_<k>_drift.npy (no DME) and the cached Pearson maps (no movie read).
  (2026-11-15) --outdir d: output directory instead of out/paper2/fig; --reuse_drift <prefix>: the methods other than v2 take
  their drift from an earlier figure's caches <prefix>_<kind>_drift.npy (copied to this figure's cache names; the same rows give
  the same DME v2 drift); --psf <calib json>: PSF panel (psfpanel.py: xy at the plotted v2 set's median height + xz, rms
  aberration, SMLM / SIMFLUX CRLB at that set's median photons, the run's pass-1 background and the camera read noise;
  psfdata.py), the v2 argument must then be a run directory; --aoi <values.json>: the zoom boxes of an earlier figure. PSF panel layout (psf_axes): a narrow column, xy image on top
  (top on the J / K / L axes tops), xz section below it at the same width (equal nm scale in x and z), label below, letter on the J / K / L baseline; <out>_psf_table.{md,csv} = the SI
  table of the panel's conditions and CRLBs (psf_table)"""
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


def psf_axes(fig, cell, axJ, info, xz_title_in=0.3, label_in=0.5, right_in=0.45):
    """(2026-11-15) axes of the PSF panel inside the (narrow) gridspec cell: one column, xy image on top (square, top on the top
    of the FRC axes = the J / K / L axes tops), xz section directly below it at the same width and equal nm scale in x and z
    (height = width x the z / x extent ratio: wide and short); xz_title_in inches between them for the xz title, label_in inches
    below the xz image for the panel label; right_in inches right of the images for the xz z ticks / label (drawn on the right).
    The width is the largest that fits both the cell width and the height from the J top to the cell bottom."""
    fw, fh = fig.get_size_inches()
    cb = cell.get_position(fig); top = axJ.get_position().y1
    x_ext = info['roi_px'] * info['pixel_nm']
    z_ext = (info['z_grid_um'][-1] - info['z_grid_um'][0]) * 1e3
    q = z_ext / x_ext
    W, Hc = cb.width * fw, (top - cb.y0) * fh
    w = min(W - right_in, (Hc - xz_title_in - label_in) / (1 + q))
    x0 = cb.x0 + max(0.0, (W - right_in - w) / 2) / fw
    axy = fig.add_axes([x0, top - w / fh, w / fw, w / fh])
    axz = fig.add_axes([x0, top - (w + xz_title_in + q * w) / fh, w / fw, q * w / fh])
    return axy, axz


def raw_grid_count(kind, path, F, smlm_path=None, cache=None):
    """(2026-11-15) a method's localizations BEFORE its own selection, counted like the FRC rows (fair protocol steps 2 and 4:
    window starts f % 6 == 0, the evalproto crop, window end within the F frames; finite x, y) -> int, or None (unknown).
    SMLM / SIMFLUX: every detected ROI with an SMLM fit (the localizer's summed-fit cache next to the smlm table, before the
    chi^2 filter; the same detections feed SIMFLUX); DECODE: every row of its table (the network's own detection output);
    SIMCODE: its reported detections = the run table's rows at its detection threshold p >= 0.63 (simcode.table.P_GOOD; the
    table's lower-p candidate pool is internal, like DECODE's sub-threshold probability map), before the sigma <= 12 nm cut.
    cache: json file of earlier counts."""
    import glob, pickle
    from simcode.dataset import Dataset
    src = os.path.abspath(smlm_path if kind in ('smlm', 'sf') else path)
    key = f'{kind}|{src}|{F}' + ('|p>=0.63' if kind == 'v2' else '')
    C = json.load(open(cache)) if cache and os.path.exists(cache) else {}
    if key in C: return C[key]
    if kind in ('smlm', 'sf'):
        fs = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(src)), 'work', '*', 'cache', '*', 'summed_fits')))
        if len(fs) != 1: return None
        L = Dataset.from_smlmtorch(pickle.load(open(fs[0], 'rb'))[0], window=6).numpy()
    elif kind == 'decode':
        from simcode.dataset import decode_table
        L = Dataset.from_decode(decode_table(src), window=6).numpy()
    elif kind == 'v2':
        L = Dataset.load(src)
        L = L[L.mask(max_precision_nm=None, in_region=False)].numpy()        # p >= P_GOOD only
    else:
        return None
    x, y, f = L['pos'][:, 0].astype(np.float64), L['pos'][:, 1].astype(np.float64), np.asarray(L['frame']).astype(np.int64)
    n = int((np.isfinite(x) & np.isfinite(y) & (f % FF.GRID == 0) & (f + 6 <= F) & FF.in_crop(x, y)).sum())
    if cache:
        C = json.load(open(cache)) if os.path.exists(cache) else {}
        C[key] = n; json.dump(C, open(cache, 'w'), indent=1)
    return n


REMOVED = dict(smlm='SMLM: chi^2 > 4 x ROI^2 or max(sigma_x, sigma_y) >= 0.15 px',
               sf='SIMFLUX: the SMLM chi^2 filter, modulation error >= 0.1, SIMFLUX fit at a parameter limit or > 0.5 px from its '
                  'SMLM fit, or max(sigma_x, sigma_y) >= 0.15 px',
               decode='DECODE: its predicted max(sigma_x, sigma_y) >= 0.15 px',
               v2='SIMCODE: calibrated max(sigma_x, sigma_y) > 12 nm')


def paper_text(s):
    """(2026-11-15) paper version of a caption / table text: the method is 'SIMCODE' and no calibration carries a version label
    (no 'SIMCODE v2', 'PSF v2', 'v2 camera'); 'DME v2' (the drift method's published name) and file names stay"""
    import re
    s = s.replace('SIMCODE = SIMCODE v2, networks', 'SIMCODE: networks').replace('SIMCODE v2', 'SIMCODE')
    s = re.sub(r'(PSF|calibration) v2\b', r'\1', s)
    s = s.replace('+ v2 camera', '+ camera calibration').replace('v2 camera', 'camera calibration')
    return s


def psf_table(info, cond, values=None, sc='SIMCODE v2'):
    """(2026-11-15) the SI table of the PSF panel: conditions + single-emitter CRLBs (rows: quantity, value, unit, note); returns
    (markdown, csv) strings. sc = the method's label (paper version: 'SIMCODE')"""
    c = info['crlb']; p = cond['pattern']; q = cond['z_quantiles_um']
    rows = [
        ('PSF model', os.path.basename(cond['calib']), '', 'in-situ vectorial PSF fit (Zernike Noll 5-21)'),
        ('rms aberration (Noll >= 5)', f"{info['rms_mwaves']:.0f}", 'mλ', ''),
        ('focal plane', f"{cond['focus_um'] * 1e3:.0f}", 'nm', 'above the coverslip'),
        ('emitter height, median', f"{cond['z_median_um'] * 1e3:.0f}", 'nm', f'pass-1 z of the plotted {sc} set'),
        ('emitter height, q25-q75', f"{q['q25'] * 1e3:.0f}-{q['q75'] * 1e3:.0f}", 'nm', f"q05-q95 {q['q05'] * 1e3:.0f}-{q['q95'] * 1e3:.0f} nm"),
        ('rows without z', f"{100 * cond['z_nan_frac']:.2f}", '%', 'no pass-1 partner'),
        ('signal, median (PSF intensity parameter)', f"{cond['photons_median']:.0f}", 'photons / 6-frame window', 'network photons of the plotted set'),
        ('collected fraction at the median height', f"{info['collected']:.2f}", '', '41 x 41 px PSF image'),
        ('signal, median (detected)', f"{info['detected_photons']:.0f}", 'photons / 6-frame window', 'intensity parameter x collected fraction'),
        ('background, median', f"{cond['bg_px_frame']:.2f}", 'photons / px / frame', 'pass-1 network bg, p >= 0.63, disjoint windows, crop'),
        ('camera read noise', f"{cond['rn2_e'] ** 0.5:.2f}", 'e- rms', f"mean variance {cond['rn2_e']:.2f} e^2 over the crop"),
        ('camera gain', f"{cond['gain_e_per_adu']:.4f}", 'e- / ADU', ''),
        ('pattern pitch', f"{p['pitch_nm']:.0f}", 'nm', 'mean of the 2 orientations'),
        ('modulation depth', f"{p['depth']:.2f}", '', 'mean of the 2 orientations'),
        ('pattern', '2 x 3', 'orientations x phase steps', ''),
    ]
    for k, lab in (('smlm', 'SMLM'), ('sf', 'SIMFLUX')):
        r = c[k]
        rows.append((f'CRLB {lab} x / y / z', f"{r['x']:.1f} / {r['y']:.1f} / {r['z']:.0f}", 'nm', '3D fit (x, y, z, I, bg)'))
        rows.append((f'CRLB {lab} x / y, z known', f"{r['xy_zfixed'][0]:.1f} / {r['xy_zfixed'][1]:.1f}", 'nm', '2D fit at the emitter height'))
    if values is not None:
        rows.append((f'{sc} rows in the figure (n_frc)', f"{values:,}".replace(',', ' '), '', 'fair-FRC set'))
    md = ['| quantity | value | unit | note |', '|---|---|---|---|'] + [f'| {a} | {b} | {u} | {n} |' for a, b, u, n in rows]
    import csv, io
    s = io.StringIO(); w = csv.writer(s); w.writerow(['quantity', 'value', 'unit', 'note']); w.writerows(rows)
    return '\n'.join(md) + '\n', s.getvalue()


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
    ap.add_argument('--aoi', default=None, help="'auto' = zoompick crossings (default for 20nM_2); or the path of an earlier "
                    "figure's <out>_values.json: its zoom boxes ('aois'), e.g. to keep the 20 nM boxes when a method's table changes")
    ap.add_argument('--outdir', default=None, help='output directory (default out/paper2/fig)')
    ap.add_argument('--reuse_drift', default=None, help='prefix of earlier drift caches for the methods other than v2')
    ap.add_argument('--psf', default=None, help='calib json of the PSF panel (needs --v2 = a run directory)')
    ap.add_argument('--select', default=None, help="(2026-11-15) <prefix>_frcopt.json of frcopt_real.py: every method's rows = its "
                    "fair-FRC set further cut at max(sx, sy) <= select[k]['sigma_max_px'] (the half-split-FRC-optimal threshold); drift "
                    "from the caches of the unfiltered sets; panel K = rows kept (solid) and removed by the cut (light) per method")
    a = ap.parse_args()
    SELJ = json.load(open(a.select)) if a.select else {}
    SEL = SELJ.get('select'); SEL_RULE = SELJ.get('rule', 'the threshold minimizing its half-split FRC')
    if SEL and a.v2_sigma_max is not None:       # v2 rows loaded up to the larger of the two thresholds (then cut below)
        a.v2_sigma_max = max(a.v2_sigma_max, SEL['v2']['sigma_max_px'] * FF.PX + 1e-6)
    S = SETS[a.dataset]; F = a.nframes or S['nframes']
    dflt = dict(smlm=f'{P2}/{a.dataset}/smlm/smlm.npy', sf=f'{P2}/{a.dataset}/sf/sf.hdf5',
                decode=f'{P2}/{a.dataset}/decode/locs.npz', v2=f'{P2}/{a.dataset}/v2')
    paths = {k: getattr(a, k) or dflt[k] for k in KINDS}       # the 4 reprocessed methods
    from simcode.dataset import decode_table
    paths['decode'] = decode_table(paths['decode'])      # locs_fixed.npz: the DECODE frame-label fix (2026-11-15)
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
        if SEL and k in SEL:
            m = R['mask'] & (np.maximum(T['sx'], T['sy']) <= SEL[k]['sigma_max_px'])
            R['n_frc_unfiltered'] = SEL[k]['n_current']; R['frc_nm_unfiltered'] = SEL[k]['frc_current']
            R['frc_nm'], R['freq'], R['curve'] = FF.frc_halves(R['x_dc'][m], R['y_dc'][m], T['f'][m])
            R['n_frc'] = int(m.sum()); R['mask'] = m
            assert SEL[k].get('n') is None or R['n_frc'] == SEL[k]['n'] and abs(R['frc_nm'] - SEL[k]['frc_nm']) < 1e-6, (k, R['n_frc'], R['frc_nm'], SEL[k])
        R['n_raw'] = raw_grid_count(k, paths[k], F, smlm_path=paths['smlm'], cache=f'{od}/raw_grid_counts.json')
        m = R['mask']
        dsets.append(to_dataset(R['x_dc'][m], R['y_dc'][m], T['f'][m], T['sx'][m], T['sy'][m], T['photons'][m]))
        res.append(R)
        print(lab, json.dumps(FF.summary(R)), flush=True)

    if a.aoi and a.aoi.endswith('.json'):
        renders = [(tuple(c), w, {'scalebar_nm': 500, 'clip_percentile': 98}) for c, w in json.load(open(a.aoi))['aois']]
    else:
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
        bot = root[2].subgridspec(1, 4, wspace=0.42, width_ratios=[1.6, 0.52, 0.52, 0.8])
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
    ax = axJ = fig.add_subplot(bot[0])
    ax.axhline(1 / 7, color='grey', linestyle='--')
    for mi, R in enumerate(res):
        sel = (R['freq'] > 0.002) & (R['freq'] < 0.1)
        ax.plot(R['freq'][sel], R['curve'][sel], color=COLORS[mi], label=f'{LABELS[mi]}: {R["frc_nm"]:.1f} nm')
    ax.legend(fontsize=11, loc='upper right', title='FRC resolution', title_fontsize=11, framealpha=0.9)
    ax.set_xlabel('Spatial freq. [nm$^{-1}$]'); ax.set_title('FRC'); ax.get_yaxis().set_visible(False); letter(ax)
    ax = fig.add_subplot(bot[1])
    # K: per method one stacked bar = accepted (solid: the rows of the FRC / renders) + removed by the method's selection (light,
    # hatched; down to its raw localizations on the same windows / region, raw_grid_count); --select: the rows removed by the
    # FRC-optimal cut (light, no hatch) between them; % accepted of the raw count above each bar
    cnt = np.array([R['n_frc'] for R in res]) * 1e-6
    c0 = np.array([R.get('n_frc_unfiltered', R['n_frc']) for R in res]) * 1e-6 if SEL else cnt
    raw = np.array([np.nan if R.get('n_raw') is None else R['n_raw'] for R in res], float) * 1e-6
    tot = np.where(np.isfinite(raw), np.maximum(raw, c0), c0)
    ax.bar(range(NM), cnt, color=COLORS, label='accepted')
    if SEL:
        ax.bar(range(NM), np.maximum(c0 - cnt, 0), bottom=cnt, color=COLORS, alpha=0.3, label='removed by FRC-optimal cut')
    ax.bar(range(NM), tot - c0, bottom=c0, color=COLORS, alpha=0.25, hatch='///', edgecolor='0.4', lw=0, label='removed by selection')
    for i in range(NM):
        if np.isfinite(raw[i]):
            ax.text(i, tot[i] + 0.015 * tot.max(), f'{100 * cnt[i] / raw[i]:.0f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(NM)); ax.set_xticklabels(short, rotation=90, fontsize=10, linespacing=0.85)
    ax.set_ylim(0, 1.12 * tot.max())
    from matplotlib.patches import Patch
    hl = [Patch(facecolor='0.35', label='accepted')] + ([Patch(facecolor='0.35', alpha=0.3, label='removed by FRC-optimal cut')] if SEL else []) \
        + [Patch(facecolor='0.35', alpha=0.25, hatch='///', edgecolor='0.4', lw=0, label='removed by selection')]
    axK = ax       # legend below the K / L tick labels
    ax.set_title('Localizations\n($10^6$)', fontsize=13); letter(ax, dx=-0.3)
    ax = fig.add_subplot(bot[2])
    ax.bar(range(NM), pc, color=COLORS); ax.set_xticks(range(NM)); ax.set_xticklabels(short, rotation=90, fontsize=10, linespacing=0.85)
    ax.set_title('RSP', fontsize=13); letter(ax, dx=-0.3)
    fig.canvas.draw()
    yb = min(t.get_window_extent().y0 for t in axK.get_xticklabels() + ax.get_xticklabels())
    yb = fig.transFigure.inverted().transform((0, yb))[1]
    fig.legend(handles=hl, fontsize=10, loc='upper left', bbox_to_anchor=(axK.get_position().x0 - 0.01, yb - 0.004), frameon=False,
               handlelength=1.4, labelspacing=0.3, borderaxespad=0)
    if psf_info is not None:
        import psfpanel
        from matplotlib.transforms import blended_transform_factory
        axy, axz = psf_axes(fig, bot[3], axJ, psf_info)
        psfpanel.draw(axy, axz, psf_info, fontsize=8)
        axz.yaxis.tick_right(); axz.yaxis.set_label_position('right')
        axy.set_title(f"PSF xy (log)\nz = {psf_info['z_um'] * 1e3:.0f} nm", fontsize=9, pad=2); axz.set_title('PSF xz (log)', fontsize=9, pad=2)
        # letter M on the baseline of J, K, L (y from the J axes, x left of the xy image)
        axy.text(-0.25, 1.03, next(letters), transform=blended_transform_factory(axy.transAxes, axJ.transAxes), fontsize=20,
                 fontweight='bold', ha='right', va='bottom')
        axz.annotate(panel_text(psf_info, psf_cond), (0.0, 0.0), xycoords='axes fraction', xytext=(0, -5), textcoords='offset points',
                     fontsize=10, va='top', ha='left', linespacing=1.2)
    foot = ('All methods: disjoint 6-frame windows, sigma < 0.15 px, own DME v2 drift; FRC with alternating 300-frame halves.'
            + (' Precision cut: each method additionally cut at its own max(sigma_x, sigma_y) <= '
               + ', '.join(f"{LABELS[KINDS.index(k)]} " + ('no cut' if SEL[k]['sigma_max_px'] >= FF.SIGMA_MAX else f"{SEL[k]['sigma_max_px'] * FF.PX:.1f} nm") for k in KINDS if k in SEL)
               + f' ({SEL_RULE}; drift of the uncut sets).' if SEL else '')
            + ' K: per method accepted localizations (solid, the rows of the FRC and the renders; % of the method\'s raw localizations above '
               'the bar) and those removed by its selection (light, hatched), both counted on the disjoint 6-frame windows and the common '
               'region of the FRC (window starts f % 6 == 0; no window subsampling counted as removed). Raw = each method\'s reported detections: SMLM / SIMFLUX '
               'every detected ROI with an SMLM fit (the same detections), DECODE every row of its output, SIMCODE every detection at its '
               'detection threshold p >= 0.63. Removed: '
               + '; '.join(REMOVED[k] for k in KINDS if k in REMOVED) + '.'
            + (' K, light without hatching (between the two): rows removed by the precision cut.' if SEL else '')
            + (f' Frames 0-{F}.' if a.nframes else '') + (f'\n{a.note}' if a.note else ''))
    txt = (lambda t: t) if a.v1 else paper_text     # paper version (no --v1): no version labels in the caption / table text
    if psf_info is not None:
        open(f'{out}_caption.txt', 'w').write(txt(caption_text(psf_info, psf_cond, foot)) + '\n')
        md, csv_s = psf_table(psf_info, psf_cond, int(res[KINDS.index('v2')]['n_frc']), sc=LABELS[KINDS.index('v2')])
        open(f'{out}_psf_table.md', 'w').write(txt(md)); open(f'{out}_psf_table.csv', 'w').write(txt(csv_s))
    else:
        open(f'{out}_caption.txt', 'w').write(txt(foot) + '\n')
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', bbox_inches='tight', pad_inches=0.15, **kw)
    plt.close('all')
    vals = dict(dataset=a.dataset, nframes=F, labels=LABELS, paths=paths, frc_nm=[R['frc_nm'] for R in res], n_raw=[R.get('n_raw') for R in res],
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
    if SEL: vals['select'] = SEL; vals['n_frc_unfiltered'] = [R.get('n_frc_unfiltered') for R in res]
    json.dump(vals, open(out + '_values.json', 'w'), indent=1)
    print(json.dumps({k: vals[k] for k in ('frc_nm', 'n_frc', 'pearson', 'aois')}), flush=True)


if __name__ == '__main__':
    main()
