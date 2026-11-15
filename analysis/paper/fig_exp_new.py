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
  table of the panel's conditions and CRLBs (psf_table)
  (2026-11-15 late) one paper style (figstyle.py): drawn at 180 mm print width, lowercase panel letters, method palette,
  one render clip percentile, FRC axes 0-0.1 nm^-1 with y axis; --drawcache <pkl>: a pure redraw (no table read); caption
  rewritten per panel in plain words (<out>_caption.txt), the protocol in code terms in <out>_notes_internal.txt; PSF-panel
  CRLB with the run's own estimated pattern (psfdata.run_pattern), not the calibration file's constant pattern block"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(REPO, 'analysis', 'figures')]
sys.path.append(os.path.join(REPO, 'analysis', 'papersim'))        # psfpanel (also on a draw-cache redraw)
from paper_common import plt, Dataset, draw_rect, compute_pc, ZOOMS     # also sets the matplotlib rc params
import frc_fair as FF
import figstyle as FS
import shutil, pickle
FS.apply()        # the shared paper style (overrides paper_common's rc params)

P2 = f'{REPO}/out/paper2'
SETS = {
    '1809_2': dict(**ZOOMS['1809_2'], nframes=120000,
                   tif=f'{P2}/1809_2/movie/simflux_5.5ms_500mW.tif', title='Low density (1809/cover1/2)',
                   imager='0.5 nM', density='low', name='1809/cover1/2'),
    '2.5nM_2': dict(**ZOOMS['2.5nM_2'], nframes=180000,
                    tif=f'{P2}/2.5nM_2/movie/cover1_600mW_5.5ms_cf660.tif', title='Medium density (2.5 nM/2)',
                    imager='2.5 nM', density='medium', name='2.5nM/2',
                    fit_note='SMLM and SIMFLUX: 2D fits at the fit plane that gave each baseline its best FRC (0.318 um for SMLM, '
                             '0.086 um for SIMFLUX; tested the median network height 0.086 um and the in-situ calibration\'s '
                             'spot-fit median 0.318 um).'),
    '20nM_2': dict(renders=None, render_args={'clip_percentile': 98}, nframes=120000,
                   tif=f'{P2}/20nM_2/movie/cover1_500mW_5.5ms_cf660.tif', title='High density (20 nM/2)',
                   imager='20 nM', density='high', name='20nM/2'),
}
# (2026-11-15 late) one style for all paper figures (figstyle.py): palette, Arial-metric font, 180 mm print width, one render
# clip percentile (FS.CLIP_PCT; the per-set 'clip_percentile' of ZOOMS / render_args is no longer used)
LABELS = ['SMLM', 'SIMFLUX', 'DECODE', 'SIMCODE']
KINDS = ['smlm', 'sf', 'decode', 'v2']
COLORS = [FS.COL[k] for k in KINDS]
OV_ZOOM, ZM_ZOOM, SIGMA_RPX = 10, 40, 2        # render magnification of the overview / the zooms; Gaussian spot sd in render px


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


def psf_axes(fig, x0_in, top_in, col_w_in, avail_h_in, info, xz_title_in=0.17, label_in=0.36, right_in=0.30):
    """(2026-11-15) the PSF panel as a narrow column in a figure of known size (inches, y from the top): xy image on top (square,
    its top at top_in = the J / K / L axes tops), xz section directly below it at the same width and equal nm scale in x and z;
    xz_title_in between them, label_in below for the panel label, right_in right of the images for the xz z ticks (right side).
    The width is the largest that fits the column width and the available height."""
    fw, fh = fig.get_size_inches()
    q = (info['z_grid_um'][-1] - info['z_grid_um'][0]) * 1e3 / (info['roi_px'] * info['pixel_nm'])
    w = min(col_w_in - right_in, (avail_h_in - xz_title_in - label_in) / (1 + q))
    x = x0_in + max(0.0, (col_w_in - right_in - w) / 2)
    axy = fig.add_axes([x / fw, (fh - top_in - w) / fh, w / fw, w / fh])
    axz = fig.add_axes([x / fw, (fh - top_in - w - xz_title_in - q * w) / fh, w / fw, q * w / fh])
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
    # (2026-11-15 late) SMLM: its own table's run (realpath: baselines_psfv2/smlm may link to another fit-plane run); SIMFLUX:
    # the run of its own table (its SMLM detections), which can differ from the SMLM table's run (2.5 nM: 0.318 vs 0.086 um plane)
    src = os.path.realpath(smlm_path if kind == 'smlm' else path)
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




# (2026-11-15 late) the same filters in plain words for the figure caption (REMOVED keeps the code terms for the internal notes)
REMOVED_WORDS = dict(
    smlm='SMLM: fits that fail the goodness-of-fit test (chi-square above 4 times the number of fitted pixels) or have a '
         'precision of 0.15 px or worse',
    sf='SIMFLUX: the same goodness-of-fit test on the SMLM fit, a pattern-modulation error of 0.1 or more, a SIMFLUX fit that ends '
       'at a parameter bound or more than 0.5 px from its SMLM position, or a precision of 0.15 px or worse',
    decode='DECODE: a network uncertainty of 0.15 px or worse',
    v1='SIMCODE (preprint): a precision of 0.15 px or worse',
    v2='SIMCODE: a calibrated uncertainty above 12 nm')


def pattern_words(p):
    """'pitch 224 nm, modulation depth 0.90 / 0.92 (orientations 1 / 2)' from a conditions pattern dict"""
    d = p.get('depth_per_angle')
    dw = ' / '.join(f'{v:.2f}' for v in d) + ' (orientations 1 / 2)' if d else f"{p['depth']:.2f}"
    return f"pitch {p['pitch_nm']:.0f} nm, modulation depth {dw}"


def psf_table(info, cond, values=None, sc='SIMCODE v2'):
    """(2026-11-15) the SI table of the PSF panel: conditions + single-emitter CRLBs (rows: quantity, value, unit, note); returns
    (markdown, csv) strings. sc = the method's label (paper version: 'SIMCODE'). Pattern = the run's own estimated pattern
    (psfdata.run_pattern; older values files: the calibration file's pattern)"""
    c = info['crlb']; p = cond['pattern']; q = cond['z_quantiles_um']
    src = 'estimated from this run (pattern_est.pt)' if p.get('source') else "calibration file's pattern"
    pa, da = p.get('pitch_nm_per_angle'), p.get('depth_per_angle')
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
        ('pattern pitch', f"{p['pitch_nm']:.1f}", 'nm', 'mean of the 2 orientations, ' + src
         + (f"; per orientation {pa[0]:.1f} / {pa[1]:.1f} nm" if pa else '')),
        ('modulation depth', f"{p['depth']:.2f}", '', 'mean of the 2 orientations, ' + src
         + (f"; per orientation {da[0]:.2f} / {da[1]:.2f}" if da else '')),
        ('pattern', '2 x 3', 'orientations x phase steps', 'relative step intensities of the run included' if p.get('relint') else ''),
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


def panel_text(info, cond=None):
    """the PSF panel's own label (short; everything else goes into the caption)"""
    return f"In-situ vectorial PSF fit\n{info['rms_mwaves']:.0f} mλ rms\n(Zernike modes 5-21)"


def crlb_sentence(info, cond):
    c = info['crlb']
    return (f"Cramér-Rao lower bounds for one isolated emitter at the median signal of the SIMCODE localizations "
            f"({info['detected_photons']:.0f} detected photons per 6-frame window), median background {cond['bg_px_frame']:.2f} "
            f"photons per pixel and frame, camera read noise {cond['rn2_e'] ** 0.5:.1f} e- rms and the median emitter height: "
            f"SMLM (sum of the 6 frames) x / y / z {c['smlm']['x']:.1f} / {c['smlm']['y']:.1f} / {c['smlm']['z']:.0f} nm; "
            f"SIMFLUX (the 6 frames with this recording's estimated pattern: {pattern_words(cond['pattern'])}, 2 orientations x 3 phase "
            f"steps) x / y / z {c['sf']['x']:.1f} / {c['sf']['y']:.1f} / {c['sf']['z']:.0f} nm.")


def caption_text(info, cond, foot):
    """(old style, kept for the internal notes) PSF panel + protocol"""
    L = [f"PSF panel: in-situ vectorial PSF fit (Zernike Noll 5-21), xy section at the emitters' median height "
         f"({cond['z_median_um'] * 1e3:.0f} nm) and xz section, log scale; {info['rms_mwaves']:.0f} mλ rms over Noll >= 5.",
         crlb_sentence(info, cond), foot]
    return '\n'.join(L)


def zoom_bar_nm(w_px):
    """scale bar of a zoom w_px camera pixels wide: 1 µm from 2.5 µm field width on, else 500 nm"""
    return 1000 if w_px * FF.PX >= 2500 else 500


def exp_caption(ds, S, F, labels, kinds, res, renders, pc, psf_info, psf_cond, SEL, SEL_RULE, fig_no):
    """(2026-11-15 late) the per-panel caption in plain words (no code syntax; notation of out/paper_figs_v2/NOTATION.md)"""
    lab = dict(zip(kinds, labels)); nm = len(kinds)
    L = lambda ls: ', '.join(ls[:-1]) + ' and ' + ls[-1]
    ab = 'abcdefghijklmnopq'
    z1 = ab[1:1 + nm]; z2 = ab[1 + nm:1 + 2 * nm]; j, k, l, m = ab[1 + 2 * nm:5 + 2 * nm]
    w1, w2 = renders[0][1] * FF.PX / 1000, renders[1][1] * FF.PX / 1000
    cyc = F // 6
    t = [
         f"Fig. {fig_no}. DNA-PAINT imaging of tubulin in COS-7 cells at {S['density']} emitter density ({S['imager']} CF660 imager; "
         f"recording {S['name']}): {F // 1000} 000 frames = {cyc // 1000} 000 cycles of 2 pattern orientations x 3 phase steps, "
         f"5 ms exposure per frame, 120 x 120 camera pixels of {FF.PX:g} nm. Every method localizes 6-frame windows (one pattern "
         "cycle); only windows that start at a cycle boundary are used, so no photon is counted twice, and all methods are compared "
         f"on the same windows and the same region (the central {FF.CROP[2] * FF.PX / 1000:.1f} x {FF.CROP[2] * FF.PX / 1000:.1f} µm "
         "of the field). Localizations shown and counted per method: precision < 0.15 px (CRLB for SMLM / SIMFLUX, network "
         "uncertainty for DECODE), calibrated uncertainty <= 12 nm for SIMCODE (in both cases the larger of the x and y values; "
         "SIMCODE also keeps only detections with a detection probability of at least 0.63). Each method's sample drift is "
         "estimated from its own localizations (DME v2) and removed." + (' ' + S['fit_note'].replace(' um', ' µm') if S.get('fit_note') else '')]
    if SEL:
        cut = [f"{lab[q]} " + ('none' if SEL[q]['sigma_max_px'] >= FF.SIGMA_MAX else f"{SEL[q]['sigma_max_px'] * FF.PX:.1f} nm")
               for q in kinds if q in SEL]
        t.append("In this version every method's localizations are further cut at a precision threshold transferred from the "
                 "matched simulation (Fig. 2: the threshold with the best ground-truth FRC there; drift from the uncut sets): "
                 + ', '.join(cut) + '.')
    t.append(f"Renders: every localization drawn as a Gaussian spot of {SIGMA_RPX * FF.PX / OV_ZOOM:.0f} nm sd (a) or "
             f"{SIGMA_RPX * FF.PX / ZM_ZOOM:.1f} nm sd (zooms) on {FF.PX / OV_ZOOM:.1f} nm ({FF.PX / ZM_ZOOM:.1f} nm) pixels, colour "
             f"scale saturated at the {FS.CLIP_PCT:g}th percentile of each image's pixels. Titles: the method and its FRC resolution (j).")
    t.append(f"({ab[0]}) SIMCODE, full field. Solid and dashed boxes: the two zoom regions; they are the regions of the first version "
             "of this figure, kept unchanged (not re-picked for the new results).")
    t.append(f"({z1[0]}-{z1[-1]}) Zoom 1 ({w1:.2f} x {w1:.2f} µm, solid box) by {L(labels)}.")
    t.append(f"({z2[0]}-{z2[-1]}) Zoom 2 ({w2:.2f} x {w2:.2f} µm, dashed box), same order.")
    t.append(f"({j}) Fourier ring correlation (FRC) between two halves of each method's localizations (alternating blocks of 300 "
             "frames), rendered on 5 nm pixels; curves smoothed over 5 frequency bins. Resolution (legend) = the frequency where "
             "the curve first falls below 1/7 (dashed line): " + ', '.join(f"{lab[q]} {R['frc_nm']:.1f} nm" for q, R in zip(kinds, res)) + '.')
    acc = ', '.join(f"{lab[q]} {R['n_frc'] / 1e6:.2f} million" + (f" ({100 * R['n_frc'] / R['n_raw']:.0f} %)" if R.get('n_raw') else '')
                    for q, R in zip(kinds, res))
    t.append(f"({k}) Localizations per method: accepted (solid; the localizations of the renders and the FRC) and removed by the "
             "method's quality filters (light, hatched)" + (', with those removed by the precision cut in between (light)' if SEL else '')
             + ", counted on the same windows and region; the percentage above each bar is accepted / all localizations the method "
             "reported (" + ("SMLM and SIMFLUX: every detected spot with an SMLM fit at the method's own fit plane, so the two counts "
                            "differ; " if S.get('fit_note') else "SMLM and SIMFLUX: every detected spot with an SMLM fit, the same "
                            "detections for both; ") + "DECODE: every "
             "localization it reported; SIMCODE: every detection with a detection probability of at least 0.63). Accepted: "
             + acc + '. Quality filters: ' + '; '.join(REMOVED_WORDS[q] for q in kinds) + '.')
    t.append(f"({l}) Resolution-scaled Pearson correlation (RSP) of each method's render with the widefield image (the mean of all "
             "frames): " + ', '.join(f"{lab[q]} {v:.2f}" for q, v in zip(kinds, pc)) + '.')
    if psf_info is not None:
        t.append(f"({m}) The in-situ PSF of this recording (vectorial model fitted to the data, Zernike modes 5-21; "
                 f"{psf_info['rms_mwaves']:.0f} mλ rms aberration): xy section at the median emitter height "
                 f"({psf_cond['z_median_um'] * 1e3:.0f} nm above the coverslip) and xz section, log scale over 3 decades; in the "
                 "xz section the white dashed line marks the focal plane and the cyan dotted line the median emitter height. "
                 "Scale bars 500 nm. " + crlb_sentence(psf_info, psf_cond))
    return '\n'.join(t) + '\n'


def compute(a, S, F, LABELS, KINDS, SEL, paths, out, od):
    """everything the drawing needs (tables, drift, FRC, RSP, PSF panel) -> ctx dict (picklable; no per-row arrays except the
    plotted localizations)"""
    res, arr = [], []
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
        if k == 'v2': v2_rows = (keep, m)
        arr.append(dict(x=R['x_dc'][m], y=R['y_dc'][m], f=T['f'][m], sx=T['sx'][m], sy=T['sy'][m], ph=T['photons'][m]))
        res.append({kk: v for kk, v in R.items() if kk not in ('drift', 'mask', 'x_dc', 'y_dc')})
        print(lab, json.dumps(FF.summary(R)), flush=True)
    dsets = [to_dataset(**d) for d in arr]
    if a.aoi and a.aoi.endswith('.json'):
        renders = [(tuple(c), w, {}) for c, w in json.load(open(a.aoi))['aois']]
    else:
        renders = S['renders'] if (S['renders'] is not None and a.aoi != 'auto') else pick_aois(dsets)
    pc, rmsd, _ = compute_pc(S['tif'], dsets)
    psf_cond = psf_info = None
    if a.psf:
        import psfdata
        from simcode.dataset import Dataset as SD
        V = SD.load(paths['v2']); V = V.select(a.v2_sigma_max) if a.v2_sigma_max is not None else V[V.sel]
        keep, m2 = v2_rows
        V = V[torch_mask(keep)][torch_mask(m2)]       # the plotted set (= the FRC rows, after the --select cut)
        assert len(V) == res[KINDS.index('v2')]['n_frc'], (len(V), res[KINDS.index('v2')]['n_frc'])
        psf_cond = psfdata.data_conditions(paths['v2'], V, a.psf)
        psf_info = psfdata.panel_info(a.psf, psf_cond)
    return dict(res=res, arr=arr, renders=renders, pc=np.asarray(pc), rmsd=np.asarray(rmsd), psf_cond=psf_cond, psf_info=psf_info)


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
    ap.add_argument('--drawcache', default=None, help="(2026-11-15 late) pickle of everything the drawing needs: read when it "
                    "exists (no table read, no FRC / RSP / PSF computation: a pure redraw), else computed and written")
    ap.add_argument('--fig_no', default=None, help='figure number in the caption (default 3 for 1809_2, 4 for 2.5nM_2)')
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
        LABELS[3:] = ['SIMCODE (preprint)', 'SIMCODE v2']; KINDS[3:] = ['v1', 'v2']; COLORS[3:] = [FS.COL['v1'], FS.COL['v2']]
        paths['v1'] = a.v1
    NM = len(LABELS)
    if a.drawcache and os.path.exists(a.drawcache):
        ctx = pickle.load(open(a.drawcache, 'rb')); print('draw cache', a.drawcache, flush=True)
    else:
        ctx = compute(a, S, F, LABELS, KINDS, SEL, paths, out, od)
        if a.drawcache: pickle.dump(ctx, open(a.drawcache, 'wb'))
    res, renders, pc, rmsd = ctx['res'], ctx['renders'], ctx['pc'], ctx['rmsd']
    psf_cond, psf_info = ctx['psf_cond'], ctx['psf_info']
    dsets = [to_dataset(**d) for d in ctx['arr']]
    draw(a, S, F, NM, LABELS, KINDS, COLORS, SEL, res, dsets, renders, pc, psf_info)
    FS.save(plt.gcf(), out)
    plt.close('all')

    # caption (plain words) + internal notes (the processing protocol in code terms)
    foot = ('All methods: disjoint 6-frame windows, sigma < 0.15 px, own DME v2 drift; FRC with alternating 300-frame halves.'
            + (' Precision cut: each method additionally cut at its own max(sigma_x, sigma_y) <= '
               + ', '.join(f"{LABELS[KINDS.index(k)]} " + ('no cut' if SEL[k]['sigma_max_px'] >= FF.SIGMA_MAX else f"{SEL[k]['sigma_max_px'] * FF.PX:.1f} nm") for k in KINDS if k in SEL)
               + f' ({SEL_RULE}; drift of the uncut sets).' if SEL else '')
            + ' K: accepted / removed counted on the windows with start f % 6 == 0 and the FRC crop. Raw: SMLM / SIMFLUX every detected '
              'ROI with an SMLM fit (summed-fit cache, before the chi^2 filter), DECODE every row, SIMCODE every row with p >= 0.63. Removed: '
            + '; '.join(REMOVED[k] for k in KINDS if k in REMOVED) + '.'
            + (f" {S['fit_note']} SMLM raw = its own run's detected ROIs, SIMFLUX raw = those of the SIMFLUX table's run." if S.get('fit_note') else '')
            + (f' Frames 0-{F}.' if a.nframes else '') + (f'\n{a.note}' if a.note else ''))
    txt = (lambda t: t) if a.v1 else paper_text     # paper version (no --v1): no version labels in the caption / table text
    fig_no = a.fig_no or {'1809_2': '3', '2.5nM_2': '4'}.get(a.dataset, 'S?')
    open(f'{out}_caption.txt', 'w').write(txt(exp_caption(a.dataset, S, F, LABELS, KINDS, res, renders, pc, psf_info, psf_cond,
                                                          SEL, SEL_RULE, fig_no)))
    open(f'{out}_notes_internal.txt', 'w').write(txt(caption_text(psf_info, psf_cond, foot) if psf_info is not None else foot) + '\n')
    if psf_info is not None:
        md, csv_s = psf_table(psf_info, psf_cond, int(res[KINDS.index('v2')]['n_frc']), sc=LABELS[KINDS.index('v2')])
        open(f'{out}_psf_table.md', 'w').write(txt(md)); open(f'{out}_psf_table.csv', 'w').write(txt(csv_s))
    vals = dict(dataset=a.dataset, nframes=F, labels=LABELS, paths=paths, frc_nm=[R['frc_nm'] for R in res], n_raw=[R.get('n_raw') for R in res],
                n_frc=[R['n_frc'] for R in res], n_disjoint=[R['n_disjoint'] for R in res], n_all=[R['n_all'] for R in res],
                frc_nm_nofilter=[R['frc_nm_nofilter'] for R in res], frc_nm_30frame_halves=[R['frc_nm_30frame_halves'] for R in res],
                pearson=np.asarray(pc).tolist(), rmsd=np.asarray(rmsd).tolist(), drift_ptp_nm=[R['drift_ptp_nm'] for R in res],
                drift_info=[R['drift_info'] for R in res], note=a.note, aois=[[list(c), w] for c, w, _ in renders],
                protocol=dict(sigma_max_px=FF.SIGMA_MAX, grid=FF.GRID, halves_frames=FF.BLK, crop=FF.CROP),
                style=dict(clip_percentile=FS.CLIP_PCT, font=FS.FONT, width_mm=180))
    if psf_info is not None:
        vals['psf_panel'] = dict(conditions=psf_cond, rms_mwaves=psf_info['rms_mwaves'], z_um=psf_info['z_um'],
                                 collected=psf_info['collected'], intensity_param=psf_info['intensity_param'],
                                 detected_photons=psf_info['detected_photons'],
                                 crlb_nm=jsonable(psf_info['crlb']), text=panel_text(psf_info, psf_cond))
    if a.reuse_drift: vals['drift_reused_from'] = a.reuse_drift
    if SEL: vals['select'] = SEL; vals['n_frc_unfiltered'] = [R.get('n_frc_unfiltered') for R in res]
    json.dump(jsonable(vals), open(out + '_values.json', 'w'), indent=1)
    print(json.dumps({k: vals[k] for k in ('frc_nm', 'n_frc', 'pearson', 'aois')}), flush=True)


def draw(a, S, F, NM, LABELS, KINDS, COLORS, SEL, res, dsets, renders, pc, psf_info):
    """(2026-11-15 late) the figure at its print size (180 mm wide), all positions in inches (y from the top)"""
    from matplotlib.patches import Patch, Rectangle
    W = FS.FIG_W; lm, rm, g = 0.16, 0.03, 0.17
    nc = 2 if NM <= 4 else 3; ncol = 2 + nc
    cw = (W - lm - rm - (ncol - 1) * g) / ncol          # zoom image size
    top0 = 0.17                                         # title row above the first images
    Sov = 2 * cw + g                                    # overview size (2 zoom rows / columns)
    y_mid = top0 + Sov + g + 0.02
    y_bot = y_mid + cw + 0.30
    hb = 1.18 if NM <= 4 else 1.1                       # bottom-row axes height
    H = y_bot + hb + 0.45
    fig = plt.figure(figsize=(W, H))
    add = lambda x, yt, w, h: fig.add_axes([x / W, (H - yt - h) / H, w / W, h / H])
    letters = iter('abcdefghijklmnopq')
    def let_img(ax, x, yt):
        fig.text((x - 0.03) / W, (H - yt + 0.025) / H, next(letters), fontsize=FS.FS_LETTER, fontweight='bold', ha='right', va='bottom')
    def let_plot(x, yt):
        fig.text(x / W, (H - yt + 0.06) / H, next(letters), fontsize=FS.FS_LETTER, fontweight='bold', ha='left', va='bottom')
    rk = dict(intensities=False, transpose=False, sigma=SIGMA_RPX, clip_percentile=FS.CLIP_PCT)
    styles = ['-', (0, (3.5, 2))]

    ax = add(lm, top0, Sov, Sov); ax.set_xticks([]); ax.set_yticks([])
    img = dsets[-1].render(zoom=OV_ZOOM, **rk); ax.imshow(img, cmap='inferno', interpolation='none')
    ax.set_title(f"{LABELS[-1]}, full field", fontsize=FS.FS_LABEL)
    let_img(ax, lm, top0)
    for i, (c, w, _) in enumerate(renders):
        ax.add_patch(Rectangle((c[0] * OV_ZOOM, c[1] * OV_ZOOM), w * OV_ZOOM, w * OV_ZOOM, fill=False, ec='w', lw=1.0, ls=styles[i]))
    n = img.shape[1]; L = 2000 / FF.PX * OV_ZOOM
    FS.scalebar(ax, 0.05 * n, 0.95 * n, L, 2000)
    for sp in ax.spines.values(): sp.set_visible(False)

    def draw_zoom(x, yt, zi, mi):
        ax = add(x, yt, cw, cw); ax.set_xticks([]); ax.set_yticks([])
        c, w, _ = renders[zi]
        img = dsets[mi].crop(c, (c[0] + w, c[1] + w), silent=True).render(zoom=ZM_ZOOM, **rk)
        ax.imshow(img, cmap='inferno', interpolation='none')
        ax.set_title(f"{LABELS[mi]}, FRC {res[mi]['frc_nm']:.1f} nm", fontsize=FS.FS_LABEL)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS[mi], ls=styles[zi], lw=1.5, clip_on=False,
                               transform=ax.transAxes))
        n = img.shape[1]; bn = zoom_bar_nm(w)
        FS.scalebar(ax, 0.06 * n, 0.93 * n, bn / FF.PX * ZM_ZOOM, bn)
        let_img(ax, x, yt)
    for mi in range(NM):
        r, cc = mi // nc, 2 + mi % nc
        draw_zoom(lm + cc * (cw + g), top0 + r * (cw + g), 0, mi)
    for mi in range(NM):
        draw_zoom(lm + mi * (cw + g), y_mid, 1, mi)

    # bottom row: j FRC | k counts | l RSP | m PSF
    xJ, wJ = 0.42, (2.05 if NM <= 4 else 1.9)
    xK, wK = xJ + wJ + 0.50, 0.92
    xL, wL = xK + wK + 0.55, 0.92
    ax = axJ = add(xJ, y_bot, wJ, hb)
    for mi, R in enumerate(res):
        sel = (R['freq'] > 0.002) & (R['freq'] <= FS.FRC_XMAX)
        ax.plot(R['freq'][sel], R['curve'][sel], color=COLORS[mi], lw=1.4 if KINDS[mi] == 'v2' else FS.LW,
                label=f'{LABELS[mi]}: {R["frc_nm"]:.1f} nm')
    FS.frc_axes(ax)
    ax.legend(loc='upper right', title='Resolution (1/7)', handlelength=1.2)
    ax.set_title('FRC (two halves of the data)')
    let_plot(xJ - 0.40, y_bot)

    ax = axK = add(xK, y_bot, wK, hb)
    cnt = np.array([R['n_frc'] for R in res]) * 1e-6
    c0 = np.array([R.get('n_frc_unfiltered', R['n_frc']) for R in res]) * 1e-6 if SEL else cnt
    raw = np.array([np.nan if R.get('n_raw') is None else R['n_raw'] for R in res], float) * 1e-6
    tot = np.where(np.isfinite(raw), np.maximum(raw, c0), c0)
    ax.bar(range(NM), cnt, 0.72, color=COLORS)
    if SEL:
        ax.bar(range(NM), np.maximum(c0 - cnt, 0), 0.72, bottom=cnt, color=COLORS, alpha=0.3)
    ax.bar(range(NM), tot - c0, 0.72, bottom=c0, color=COLORS, alpha=0.25, hatch='//////', edgecolor='0.4', lw=0)
    for i in range(NM):
        if np.isfinite(raw[i]):
            ax.text(i, tot[i] + 0.012 * tot.max(), f'{100 * cnt[i] / raw[i]:.0f}%', ha='center', va='bottom', fontsize=FS.FS_SMALL)
    ax.set_xticks(range(NM)); ax.set_xticklabels(LABELS, rotation=40, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.6, NM - 0.4)
    ax.set_ylim(0, (1.62 if not SEL else 2.25) * tot.max()); ax.set_ylabel('Localizations (10$^6$)')
    hl = [Patch(facecolor='0.35', label='accepted')] + ([Patch(facecolor='0.35', alpha=0.3, label='removed by\nprecision cut')] if SEL else []) \
        + [Patch(facecolor='0.35', alpha=0.25, hatch='//////', edgecolor='0.4', lw=0, label='removed by\nquality filters')]
    ax.legend(handles=hl, loc='upper left', handlelength=1.0, borderaxespad=0.25, fontsize=FS.FS_SMALL)
    let_plot(xK - 0.42, y_bot)

    ax = add(xL, y_bot, wL, hb)
    ax.bar(range(NM), pc, 0.72, color=COLORS); ax.set_xticks(range(NM))
    ax.set_xticklabels(LABELS, rotation=40, ha='right', rotation_mode='anchor'); ax.set_xlim(-0.6, NM - 0.4)
    ax.set_ylim(0, max(0.8, 1.12 * np.max(pc))); ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1e-9, 0.2))
    ax.set_ylabel('RSP (Pearson correlation\nwith the widefield image)')
    let_plot(xL - 0.47, y_bot)

    if psf_info is not None:
        import psfpanel
        x0 = xL + wL + 0.22
        axy, axz = psf_axes(fig, x0, y_bot, W - rm - x0, H - y_bot - 0.03, psf_info)
        psfpanel.draw(axy, axz, psf_info, fontsize=FS.FS_SMALL)
        axz.yaxis.tick_right(); axz.yaxis.set_label_position('right')
        axy.set_title(f"PSF xy (log)\nz = {psf_info['z_um'] * 1e3:.0f} nm", fontsize=FS.FS_TICK, pad=2)
        axz.set_title('PSF xz (log)', fontsize=FS.FS_TICK, pad=2)
        fig.text((x0 - 0.02) / W, (H - y_bot + 0.06) / H, next(letters), fontsize=FS.FS_LETTER, fontweight='bold', ha='left', va='bottom')
        axz.annotate(panel_text(psf_info), (0.0, 0.0), xycoords='axes fraction', xytext=(0, -3), textcoords='offset points',
                     fontsize=FS.FS_TICK, va='top', ha='left', linespacing=1.15)
    return fig


if __name__ == '__main__':
    main()
