"""Main-text Fig 2 (\\label{fig:simulation}) with ONE predefined selection rule for every method (common_rule.py; user decision
2026-11-15: no threshold tuned on the ground truth): each method's own quality filters + max(σx, σy) < 0.15 px (16.25 nm) on its
own reported precision (SIMCODE: p >= 0.63 and its posterior σ), disjoint 6-frame windows. GT-FRC = sustained 1/7 crossing.
Ground truth of the *_wsm3d sets: webSMLM labels at their own 3D positions, panel a (and the zoom b) colour-coded by height.
Layout:
  a GT (z) | b GT zoom (z) | c SMLM | d SIMFLUX
  e DECODE | f SIMCODE | g localization error classes (2 columns)
  h GT-FRC curves | i GT-FRC vs frames (time to fidelity) | j PSF (xy | xz)
usage: python analysis/papersim/fig_sim_common.py [--set Dexp1_v2psf_wsm3d] [--zr 0,250] [--zoom_z 1]
output: out/paper_figs_v2/fig_sim_<set, density 1 dropped>.{pdf,svg,png}, _caption.txt, _values.json, _notes_internal.txt"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, gtfrc as GR
from fig_density import render, crop_of
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'paper'))
import figstyle as FS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
FS.apply()

METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE')]
COLM = {k: FS.COL[k] for k, _ in METH}
ZOOM = 20                                         # render magnification of the zooms (5.4 nm pixels)


def draw(a, md, gt, zpt, W, cx, cy, w, B, R, psfpanel):
    Wf = FS.FIG_W; lm, rm, g = 0.16, 0.03, 0.17
    cw = (Wf - lm - rm - 3 * g) / 4
    top0 = 0.27; gv = 0.33
    y1 = top0 + cw + gv
    yb = y1 + cw + 0.42
    hb = 1.30
    H = yb + hb + 0.36
    fig = plt.figure(figsize=(Wf, H))
    add = lambda x, yt, ww, hh: fig.add_axes([x / Wf, (H - yt - hh) / H, ww / Wf, hh / H])
    colx = lambda c: lm + c * (cw + g)
    def let(x, yt, s, img=True):
        fig.text((x - (0.03 if img else 0)) / Wf, (H - yt + (0.025 if img else 0.07)) / H, s, fontsize=FS.FS_LETTER,
                 fontweight='bold', ha='right' if img else 'left', va='bottom')
    def noframe(A):
        A.set_xticks([]); A.set_yticks([]); [q.set_visible(False) for q in A.spines.values()]
    n = w * ZOOM; zbar = 500 / (C.PX_NM / ZOOM); p = gt['pts']; zr = a.zr; znm = zpt * 1e3
    # a: ground truth, colour = height
    A = add(colx(0), top0, cw, cw); noframe(A)
    sr = 10.0 / C.PX_NM          # 10 nm render pixels
    A.imshow(FS.zrgb(p[:, 0], p[:, 1], znm, -0.5, -0.5, W, W, sr, zr, blur=1.0, clip_pct=99.0, gamma=0.8),
             extent=[-0.5, W - 0.5, W - 0.5, -0.5], interpolation='none')
    A.add_patch(Rectangle((cx, cy), w, w, fill=False, ec='w', lw=1.0))
    A.set_xlim(-0.5, W - 0.5); A.set_ylim(W - 0.5, -0.5)
    FS.scalebar(A, 0.05 * W, 0.94 * W, 2000 / C.PX_NM, 2000)
    cax = A.inset_axes([0.56, 0.905, 0.38, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=FS.zcmap(), norm=plt.Normalize(*zr)), cax=cax, orientation='horizontal')
    cb.set_ticks([zr[0], (zr[0] + zr[1]) / 2, zr[1]]); cb.ax.tick_params(labelsize=FS.FS_SMALL, length=1.5, pad=1, colors='w')
    cb.outline.set_edgecolor('w'); cb.outline.set_linewidth(0.4)
    cax.text(-0.04, 0.5, 'z (nm)', transform=cax.transAxes, fontsize=FS.FS_SMALL, color='w', ha='right', va='center', bbox=FS.BACKING)
    A.set_title('Ground truth\n(all labelling sites, colour = height)'); let(colx(0), top0, 'a')
    # b: ground-truth zoom (z-coded with the brightness of the method renders, or intensity)
    A = add(colx(1), top0, cw, cw); noframe(A)
    if a.zoom_z:
        A.imshow(FS.zrgb(p[:, 0], p[:, 1], znm, cx, cy, w, w, 1 / ZOOM, zr, blur=1.0, clip_pct=FS.CLIP_PCT, nonzero=False), interpolation='none')
        A.set_title('Ground truth (zoom)\ncolour = height')
    else:
        A.imshow(render(p[:, 0], p[:, 1], (cx, cy), w, zoom=ZOOM, clip_pct=FS.CLIP_PCT), cmap='inferno', interpolation='none')
        A.set_title('Ground truth (zoom)')
    let(colx(1), top0, 'b'); FS.scalebar(A, 0.06 * n, 0.93 * n, zbar, 500)
    slots = {'smlm': (top0, 2, 'c'), 'sf': (top0, 3, 'd'), 'decode': (y1, 0, 'e'), 'v2': (y1, 1, 'f')}
    for k, lk in METH:
        r = R['methods'][k]; yt, c, Lt = slots[k]; A = add(colx(c), yt, cw, cw); noframe(A)
        A.imshow(render(B[k]['x'], B[k]['y'], (cx, cy), w, zoom=ZOOM, clip_pct=FS.CLIP_PCT), cmap='inferno', interpolation='none')
        A.set_title(f"{lk}\nGT-FRC {r[GR.GT_KEY]:.1f} nm, {r['n'] / 1e3:.0f}k loc.")
        let(colx(c), yt, Lt); FS.scalebar(A, 0.06 * n, 0.93 * n, zbar, 500)
    # g: localization error classes (+ true-positive rms above the bars), columns 2-3 of row 1
    xe = colx(2) + 0.30; we = 2 * cw + g - 0.30 - 1.25; E = add(xe, y1, we, cw)
    for i, (k, lk) in enumerate(METH):
        pc, ic, fc = R['methods'][k]['classes']
        E.bar(i, pc / 1e3, 0.72, color=COLM[k]); E.bar(i, ic / 1e3, 0.72, bottom=pc / 1e3, color=COLM[k], alpha=0.4)
        E.bar(i, fc / 1e3, 0.72, bottom=(pc + ic) / 1e3, color='w', ec=COLM[k], hatch='//////', lw=0.6)
        E.text(i, (pc + ic + fc) / 1e3 * 1.02 + 1, f"{R['methods'][k]['rms_tp_nm']:.1f} nm", ha='center', va='bottom', fontsize=FS.FS_SMALL)
    E.bar([0], [0], color='0.35', label=f'error ≤ {a.r_prec:g} nm'); E.bar([0], [0], color='0.35', alpha=0.4, label=f'error {a.r_prec:g}-{a.r_fp:g} nm')
    E.bar([0], [0], color='w', ec='0.35', hatch='//////', lw=0.6, label=f'false positive\n(> {a.r_fp:g} nm)')
    E.set_xticks(range(len(METH))); E.set_xticklabels([lk for _, lk in METH])
    E.set_xlim(-0.6, len(METH) - 0.4)
    top = max(sum(R['methods'][k]['classes']) for k, _ in METH) / 1e3
    E.set_ylim(0, top * 1.15); E.set_ylabel('Localizations (×10$^3$)')
    E.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), title='number above bar:\nrms error of the\ntrue positives', fontsize=FS.FS_SMALL,
             title_fontsize=FS.FS_SMALL, handlelength=1.2, alignment='left')
    E.set_title('Localization error'); let(xe - 0.30, y1, 'g', img=False)
    # bottom row: h GT-FRC curves | i GT-FRC vs frames | j PSF
    xI, wI = 0.42, 2.30; xJ, wJ = xI + wI + 0.52, 2.05
    I = add(xI, yb, wI, hb); J = add(xJ, yb, wJ, hb)
    for k, lk in METH:
        r = R['methods'][k]; gc = r['gt_curve']; lw = 1.4 if k == 'v2' else FS.LW
        f_ = np.asarray(gc['freq'][2:]); c_ = np.asarray(gc['curve'][2:]); sel = f_ <= FS.FRC_XMAX
        I.plot(f_[sel], c_[sel], color=COLM[k], lw=lw, label=f"{lk}: {r[GR.GT_KEY]:.1f} nm")
        t = r['ttf']; J.plot(np.array(t['frames']) / 1e3, t['res_nm'], '-o', color=COLM[k], lw=lw, ms=2.6 if k == 'v2' else 2.2, label=lk)
    FS.frc_axes(I, ylabel='GT-FRC')
    I.set_title('FRC against the ground truth'); I.legend(loc='upper right', title='GT-FRC resolution', handlelength=1.2)
    let(xI - 0.40, yb, 'h', img=False)
    J.set_xscale('log'); J.set_yscale('log'); J.axhline(30, color='0.5', lw=0.7, ls=':')
    J.text(0.27, 31, '30 nm', color='0.4', fontsize=FS.FS_SMALL, ha='left', va='bottom')
    J.set_xticks([0.25, 1, 4, 15, 60]); J.set_xticklabels(['0.25', '1', '4', '15', '60']); J.set_ylim(12, 250)
    J.set_yticks([15, 20, 30, 50, 100, 200]); J.set_yticklabels(['15', '20', '30', '50', '100', '200']); J.minorticks_off()
    J.set_xlabel('Frames (×1000)'); J.set_ylabel('GT-FRC resolution (nm)')
    J.legend(loc='upper right', handlelength=1.6, fontsize=FS.FS_SMALL)
    J.set_title('GT-FRC vs frames (time to fidelity)'); let(xJ - 0.40, yb, 'i', img=False)
    info = psfpanel.sim_psf_info(md, a.set.startswith('Dexp'))
    import fig_exp_new as FE
    x0 = xJ + wJ + 0.30
    axy, axz = FE.psf_axes(fig, x0, yb, Wf - rm - x0, H - yb - 0.03, info)
    psfpanel.draw(axy, axz, info, fontsize=FS.FS_SMALL)
    axz.yaxis.tick_right(); axz.yaxis.set_label_position('right')
    axy.set_title(f"PSF xy (log)\nz = {info['z_um'] * 1e3:.0f} nm", fontsize=FS.FS_TICK, pad=2); axz.set_title('PSF xz (log)', fontsize=FS.FS_TICK, pad=2)
    fig.text((x0 - 0.02) / Wf, (H - yb + 0.07) / H, 'j', fontsize=FS.FS_LETTER, fontweight='bold', ha='left', va='bottom')
    axz.annotate(psfpanel.panel_label(info), (0.0, 0.0), xycoords='axes fraction', xytext=(0, -3), textcoords='offset points',
                 fontsize=FS.FS_TICK, va='top', ha='left', linespacing=1.15)
    fig._psf_info = info
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='Dexp1_v2psf_wsm3d'); ap.add_argument('--zr', default='0,250', help='z colour range (nm)')
    ap.add_argument('--zoom_z', type=int, default=1, help='1 = the GT zoom (b) colour-coded by height, 0 = intensity as the method zooms')
    ap.add_argument('--r_prec', type=float, default=20.0); ap.add_argument('--r_fp', type=float, default=50.0)
    ap.add_argument('--name', default=None, help='output name (default fig_sim_<set without the density 1>)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2')); a = ap.parse_args()
    a.zr = tuple(float(v) for v in a.zr.split(','))
    import common_rule as CR, psfpanel
    assert (a.r_prec, a.r_fp) == (CR.R_PREC, CR.R_FP)
    md = os.path.join(C.OUT, a.set); R, B = CR.load(a.set)
    g0 = json.load(open(os.path.join(md, 'gen.json')))
    gt = np.load(os.path.join(md, 'gt.npz')); W = R['W']; nf = R['nframes']; (cx, cy), w = crop_of(md)
    zpt = gt['pts_z_um']
    lab = dict(METH)
    fig = draw(a, md, gt, zpt, W, cx, cy, w, B, R, psfpanel); info = fig._psf_info
    name = a.name or f"fig_sim_{a.set.replace('Dexp1_', 'Dexp_')}"
    os.makedirs(a.out, exist_ok=True); out = os.path.join(a.out, name)
    Mt = R['methods']; zd = g0['z_dist_um']; ws = g0.get('wsm') or {}; wz = g0.get('wsm_z_raw_nm') or {}
    ft = lambda k, t: (lambda v: 'not reached' if v is None else f"{v / 1e3:.1f}k")(Mt[k]['ttf']['frames_to'].get(str(t)))
    is3d = str(g0.get('gt', '')).startswith('wsm3d')
    hcap = (f"Every label is simulated at its own 3D position on the filament (heights {zd['p05'] * 1e3:.0f}-{zd['p95'] * 1e3:.0f} nm above the "
            f"coverslip, 5th-95th percentile, median {zd['p50'] * 1e3:.0f} nm" + (f"; the {100 * wz['frac_below0']:.1f} % of label positions below "
            "the coverslip surface are set to 0 nm" if wz.get('frac_below0') else '') + ")." if is3d else
            "Their heights are drawn from the experimental height distribution.")
    cap = [f"Fig. 2. Simulation at {g0['density']:g} emitter per µm² per frame, {nf // 1000} 000 frames, {W} x {W} camera pixels of "
           f"{C.PX_NM} nm ({W * C.PX_NM / 1000:.2f} µm field). Simulated with the in-situ vectorial PSF of the recording of Fig. 3 and "
           f"that recording's sCMOS camera maps (gain, offset, per-pixel read noise); photons and background as measured in that "
           f"recording (median {g0['brightness_fit']['fit_median']:.0f} detected photons per 6-frame window, background "
           f"{g0['bg_per_frame']['mean']:.2f} photons per pixel and frame); mean on-time {g0['avg_on_time']:g} frames; pattern of 2 "
           f"orientations x 3 phase steps, pitch {C.PITCH_NM:g} nm, modulation depth {C.DEPTH:g}.",
           f"(a) Ground truth: all labelling sites of a simulated microtubule network (webSMLM, microtubules-in-cells sample; "
           f"{ws.get('labels_per_nm', 0.5):.1f} labels per nm of filament, {ws.get('mt_length_um', float('nan')):.0f} µm of filament, "
           f"{ws.get('n_labels', 0) // 1000} {ws.get('n_labels', 0) % 1000:03d} sites in the field), colour = height above the coverslip "
           f"({a.zr[0]:.0f}-{a.zr[1]:.0f} nm colour scale; brightness = label density). The labels sit ~25-30 nm from the tube axis "
           "(13-protofilament lattice, 12 nm binder, 2-5 nm linker). " + hcap + " White box: the zoom of b-f "
           f"({w * C.PX_NM / 1000:.2f} x {w * C.PX_NM / 1000:.2f} µm, picked at filament crossings). (b) The ground truth of the zoom, "
           "rendered as the reconstructions" + (" (colour = height, brightness as in c-f)." if a.zoom_z else "."),
           f"(c-f) Renders of the zoom by (c) SMLM, (d) SIMFLUX, (e) DECODE and (f) SIMCODE ({C.PX_NM / ZOOM:.1f} nm pixels, 1-pixel "
           f"Gaussian blur, colour scale saturated at the {FS.CLIP_PCT:g}th percentile of the image's pixels). One selection rule for "
           "every method, the rule used for the experimental data, set before the analysis and without the ground truth: each method's "
           "localizations on disjoint 6-frame windows (one localization per window and emitter, no photon used twice) that pass the "
           "method's own quality filters and have a reported precision below 0.15 px (16.25 nm; the larger of the x and y values; "
           "SMLM / SIMFLUX: CRLB; DECODE: its network uncertainty; SIMCODE: the posterior uncertainty of its pattern solve, with its detection probability of "
           "at least 0.63). Titles: GT-FRC resolution and the number of localizations. SIMFLUX uses the pattern estimated from the data, "
           "as SIMCODE does. SMLM and DECODE get the sum of the 6 frames; SMLM and SIMFLUX fit a 2D model of the PSF at the median "
           f"emitter height ({g0['z_um'] * 1e3:.0f} nm); DECODE was retrained for this PSF following its published training settings (Supplementary "
           "Note: DECODE training), with emitter heights of 0-400 nm and these photon, background and density ranges. GT-FRC = the FRC between the rendered reconstruction and the rendered ground truth (x-y "
           f"projection, 5 nm pixels, full field, all {nf // 1000} 000 frames); resolution = the frequency after which the curve stays "
           "below 1/7 for 0.01 nm^-1 (the hollow tube's spectrum has zeros near 65, 28 and 18 nm, where a curve can dip below 1/7 and "
           "recover).",
           f"(g) The rendered localizations by their 2D error to the nearest emitter that is on in the window: <= {a.r_prec:g} nm "
           f"(solid), {a.r_prec:g}-{a.r_fp:g} nm (light), no emitter on within {a.r_fp:g} nm = false positive (hatched): "
           + '; '.join(f"{lab[k]} {Mt[k]['n']:,} ({Mt[k]['classes'][0]:,} / {Mt[k]['classes'][1]:,} / {Mt[k]['classes'][2]:,})".replace(',', ' ') for k, _ in METH)
           + ". Number above each bar: rms error per axis of the true positives (localizations with an emitter on within 3 times "
           "their own reported precision): " + ', '.join(f"{lab[k]} {Mt[k]['rms_tp_nm']:.1f} nm" for k, _ in METH) + ".",
           f"(h) GT-FRC curves (all {nf // 1000} 000 frames); dashed line = 1/7; resolution in the legend. The curves are drawn exactly as "
           "the resolution is read from them: smoothed over 5 frequency bins (a 5-bin moving average, no extra smoothing for the "
           "plot); the first two frequency bins are not drawn.",
           "(i) Time to fidelity: GT-FRC of the same selection on the first N frames (N = 250 ... "
           f"{nf // 1000} 000); dotted line = 30 nm. Frames to reach 40 / 30 / 20 nm: "
           + '; '.join(f"{lab[k]} " + ('none reached' if all(Mt[k]['ttf']['frames_to'].get(str(t)) is None for t in (40, 30, 20))
                                          else f"{ft(k, 40)} / {ft(k, 30)} / {ft(k, 20)}") for k, _ in METH) + ".",
           f"(j) The PSF of the simulation (the in-situ vectorial PSF fit of the recording of Fig. 3, Zernike modes 5-21; "
           f"{info['rms_mwaves']:.0f} mλ rms aberration): xy section at the median emitter height ({info['z_um'] * 1e3:.0f} nm) and xz "
           "section, log scale over 3 decades; in the xz section the white dashed line marks the focal plane and the cyan dotted line "
           "the median emitter height. Scale bars 500 nm. " + psfpanel.caption_text(info).split('\n', 1)[1]]
    assert GR.RULE == 'sustained', 'the caption text states the sustained crossing (gtfrc.RULE)'
    open(out + '_caption.txt', 'w').write('\n'.join(cap) + '\n')
    v12 = Mt.get('v2_12nm', {})
    notes = [f"Internal notes to {name} (not for the caption).",
             f"Set {a.set}: GT {g0.get('gt')}, heights {g0.get('z_rule')}; raw webSMLM z {wz}; height quantiles {zd}.",
             f"Selection: common_rule.py (out/papersim/common_rule/{a.set}.json): disjoint windows, own filters, max(sx, sy) < 0.15 px; "
             "SIMCODE p >= 0.63 + its posterior sigma (no GT-tuned threshold, frcopt not used).",
             f"SIMCODE at the previous 12 nm cut: n {v12.get('n')} GT-FRC {v12.get(GR.GT_KEY, float('nan')):.2f} nm (common rule: n "
             f"{Mt['v2']['n']} GT-FRC {Mt['v2'][GR.GT_KEY]:.2f} nm).",
             f"SMLM / SIMFLUX fit plane {g0['z_um']:.4f} um (median GT height); zoom box (x0, y0, w) = ({cx}, {cy}, {w}) px; z colour range {a.zr} nm.",
             f"Style: figstyle.py (font {FS.FONT}, 180 mm wide, render clip {FS.CLIP_PCT:g} %)."]
    open(out + '_notes_internal.txt', 'w').write('\n'.join(notes) + '\n')
    vals = dict(set=a.set, rule=R['rule'], sig_max_nm=R['sig_max_nm'], r_prec_nm=a.r_prec, r_fp_nm=a.r_fp, z_dist_um=zd, fit_plane_um=g0['z_um'],
                methods={k: dict(label=lab.get(k, k), n=Mt[k]['n'], gt_frc_nm=Mt[k][GR.GT_KEY], gt_frc_first_nm=Mt[k]['gt_first'],
                                 half_split_frc_nm=Mt[k][GR.HS_KEY], half_split_frc_first_nm=Mt[k]['hs_first'], half_split_frc_sust_nm=Mt[k]['hs_sust'], crossing_rule=GR.RULE, fp_3sigma=Mt[k]['fp'],
                                 rms_tp_nm=Mt[k]['rms_tp_nm'], classes=dict(zip(('precise', 'imprecise', 'false_positive'), Mt[k]['classes'])),
                                 ttf=Mt[k]['ttf']) for k in list(lab) + [k for k in ('v2_12nm', 'sf_gtp', 'v1') if k in Mt]})
    json.dump(vals, open(out + '_values.json', 'w'), indent=1, default=float)
    FS.save(fig, out)
    print(json.dumps({k: (v['n'], round(v['gt_frc_nm'], 1), v['classes'], v['ttf']['frames_to']) for k, v in vals['methods'].items()}))


if __name__ == '__main__':
    main()
