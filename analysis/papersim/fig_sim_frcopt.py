"""Main-text Fig 2 (\\label{fig:simulation}) with each method's FRC-optimal selection (frcopt.py sweep), webSMLM hollow-tube sets.

One selection per method: the method's Fig 2 base set (disjoint 6-frame windows, own max(σx, σy) < 16.25 nm; SIMCODE: p >= 0.63)
cut at the threshold on its OWN reported precision that minimizes its GT-FRC (--crit, default gt_sust = the sustained 1/7
crossing, metrics.frc_sustained; ties -> more rows). Every GT-FRC number in the figure is the sustained crossing.
Layout:
  A GT | B GT zoom | C SMLM | D GT-FRC vs fraction kept (marker = the chosen threshold; axes box aligned with H's image)
  E localizations (precise <= --r_prec nm / imprecise / false positive = no emitter ON within --r_fp nm) | F SIMFLUX | G DECODE | H SIMCODE
  I GT-FRC curves | J GT-FRC vs frames (time to fidelity) | K PSF (xy | xz)
usage: python analysis/papersim/fig_sim_frcopt.py [--set Dexp1_v2psf_wsm] [--crit gt_sust] [--v2fam sig_p] [--name <out name>]
output: out/paper_figs_v2/fig_sim_<set, density 1 dropped>.{pdf,svg,png}, _caption.txt, _values.json"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M, gtfrc as G
from fig_density import render, crop_of, COL, gt_desc
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'paper'))
import figstyle as FS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
FS.apply()

METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE')]
COLM = {k: FS.COL[k] for k, _ in METH}           # = fig_density.COL (smlm, sf, decode, v2sig)
assert all(COLM[k] == COL[{'v2': 'v2sig'}.get(k, k)] for k in COLM)
ZOOM = 20                                         # render magnification of the zooms (5.4 nm pixels)
FRCD = os.path.join(C.OUT, 'frcopt')
GRID = (250, 500, 1000, 2000, 4000, 8000, 15000, 30000, 45000, 60000)


def classes(B, m, r_prec, r_fp):
    """(precise, imprecise, false positive) counts of the rows m (2D error to the nearest ON emitter of the row's window)"""
    e = np.hypot(B['ex'][m], B['ey'][m]) * C.PX_NM; e = np.where(np.isfinite(e), e, np.inf)
    return int((e <= r_prec).sum()), int(((e > r_prec) & (e <= r_fp)).sum()), int((e > r_fp).sum())


def load(name, v2fam):
    R = json.load(open(os.path.join(FRCD, f'{name}.json')))
    Z = np.load(os.path.join(FRCD, f'{name}_rows.npz'))
    B = {}
    for k, _ in METH:
        ck = 'v2_' + v2fam if k == 'v2' else k
        B[k] = {c.split('__', 1)[1]: Z[c] for c in Z.files if c.split('__', 1)[0] == ck}
    return R, B


def ttf(S, B, m, nf, targets=(40, 30, 20)):
    """GT-FRC (sustained crossing) on the rows with frame < N for N in GRID; frames to each target (log-interpolated)"""
    grid = [n for n in GRID if n < nf] + [nf]; res = []
    for N in grid:
        mm = m & (B['f'] < N)
        g = G.gt_frc(np.c_[B['x'][mm], B['y'][mm]], None, S.lo, S.hi, S.pix, C.PX_NM, gt_img=S.gimg)
        res.append(M.frc_sustained(g['freq'], g['curve']) if len(g['freq']) else float('nan'))
    reach = {}
    for t in targets:
        r = np.array(res); ok = np.isfinite(r) & (r <= t)
        if not ok.any(): reach[str(t)] = None; continue
        i = int(np.argmax(ok))
        if i == 0 or not np.isfinite(r[i - 1]): reach[str(t)] = float(grid[i]); continue
        u = (r[i - 1] - t) / (r[i - 1] - r[i])
        reach[str(t)] = float(np.exp(np.log(grid[i - 1]) + u * (np.log(grid[i]) - np.log(grid[i - 1]))))
    return dict(frames=grid, res_nm=res, frames_to=reach)


def draw(a, md, gt, W, cx, cy, w, B, SEL, lab, psfpanel):
    """(2026-11-15 late) the figure at its print size (180 mm wide; figstyle), positions in inches (y from the top).
    a GT | b GT zoom | c SMLM | d GT-FRC vs fraction kept / e localization error | f SIMFLUX | g DECODE | h SIMCODE /
    i GT-FRC curves | j GT-FRC vs frames | k PSF"""
    Wf = FS.FIG_W; lm, rm, g = 0.16, 0.03, 0.17
    cw = (Wf - lm - rm - 3 * g) / 4
    top0 = 0.27; gv = 0.33                     # two-line titles above the images
    y1 = top0 + cw + gv
    yb = y1 + cw + 0.50                        # bottom row (e's tick labels + i / j titles in between)
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
    n = w * ZOOM; zbar = 500 / (C.PX_NM / ZOOM)
    # a: ground truth
    A = add(colx(0), top0, cw, cw); A.set_facecolor('k'); noframe(A)
    p = gt['pts']; A.plot(p[:, 0], p[:, 1], ',', color='#ffb070', alpha=0.9, ms=0.5, rasterized=True)
    A.add_patch(Rectangle((cx, cy), w, w, fill=False, ec='w', lw=1.0))
    A.set_xlim(-0.5, W - 0.5); A.set_ylim(W - 0.5, -0.5)
    FS.scalebar(A, 0.05 * W, 0.94 * W, 2000 / C.PX_NM, 2000)
    A.set_title('Ground truth\n(all labelling sites)'); let(colx(0), top0, 'a')
    # b: ground-truth zoom
    A = add(colx(1), top0, cw, cw); noframe(A)
    A.imshow(render(p[:, 0], p[:, 1], (cx, cy), w, zoom=ZOOM, clip_pct=FS.CLIP_PCT), cmap='inferno', interpolation='none')
    A.set_title('Ground truth (zoom)'); let(colx(1), top0, 'b'); FS.scalebar(A, 0.06 * n, 0.93 * n, zbar, 500)
    slots = {'smlm': (top0, 2, 'c'), 'sf': (y1, 1, 'f'), 'decode': (y1, 2, 'g'), 'v2': (y1, 3, 'h')}
    for k, lk in METH:
        s = SEL[k]; yt, c, Lt = slots[k]; A = add(colx(c), yt, cw, cw); noframe(A)
        A.imshow(render(B[k]['x'][s['m']], B[k]['y'][s['m']], (cx, cy), w, zoom=ZOOM, clip_pct=FS.CLIP_PCT), cmap='inferno',
                 interpolation='none')
        A.set_title(f"{lk}\nGT-FRC {s['r'][a.crit]:.1f} nm, {100 * s['r']['n'] / s['n_base']:.0f} % kept")
        let(colx(c), yt, Lt); FS.scalebar(A, 0.06 * n, 0.93 * n, zbar, 500)
    # d: GT-FRC vs fraction kept (plot axes inside column 3, row 0; room for its y label on the left)
    xd = colx(3) + 0.30; D = add(xd, top0 + 0.02, cw - 0.30, cw - 0.30)
    for k, lk in METH:
        s = SEL[k]; fr = np.array([q['n'] for q in s['rows']]) / s['n_base'] * 100; v = np.array([q.get(a.crit, np.nan) for q in s['rows']])
        D.plot(fr, v, '-', color=COLM[k], lw=1.4 if k == 'v2' else FS.LW, label=lk)
        D.plot(100 * s['r']['n'] / s['n_base'], s['r'][a.crit], 'o', color=COLM[k], mec='k', mew=0.5, ms=4)
    D.set_yscale('log'); D.set_yticks([15, 20, 30, 50, 100]); D.set_yticklabels(['15', '20', '30', '50', '100']); D.minorticks_off()
    D.set_ylim(14, 120); D.set_xlim(0, 103); D.set_xticks([0, 50, 100])
    D.set_xlabel('Localizations kept (%)'); D.set_ylabel('GT-FRC resolution (nm)')
    D.legend(loc='upper right', ncol=2, columnspacing=0.8, handlelength=1.0, fontsize=FS.FS_SMALL)
    D.set_title('GT-FRC vs fraction kept\n(precision cut)'); let(xd - 0.30, top0 + 0.02, 'd', img=False)
    # e: localization error classes (+ true-positive rms above the bars)
    xe = colx(0) + 0.30; E = add(xe, y1, cw - 0.30, cw)
    for i, (k, lk) in enumerate(METH):
        pc, ic, fc = SEL[k]['cls']
        E.bar(i, pc / 1e3, 0.72, color=COLM[k]); E.bar(i, ic / 1e3, 0.72, bottom=pc / 1e3, color=COLM[k], alpha=0.4)
        E.bar(i, fc / 1e3, 0.72, bottom=(pc + ic) / 1e3, color='w', ec=COLM[k], hatch='//////', lw=0.6)
        E.text(i, (pc + ic + fc) / 1e3 * 1.02 + 1, f"{SEL[k]['r']['rms_tp_nm']:.1f} nm", ha='center', va='bottom', fontsize=FS.FS_SMALL)
    E.bar([0], [0], color='0.35', label=f'error ≤ {a.r_prec:g} nm'); E.bar([0], [0], color='0.35', alpha=0.4, label=f'error {a.r_prec:g}-{a.r_fp:g} nm')
    E.bar([0], [0], color='w', ec='0.35', hatch='//////', lw=0.6, label=f'false positive (> {a.r_fp:g} nm)')
    E.set_xticks(range(len(METH))); E.set_xticklabels([lk for _, lk in METH], rotation=40, ha='right', rotation_mode='anchor')
    E.set_xlim(-0.6, len(METH) - 0.4)
    top = max(sum(SEL[k]['cls']) for k, _ in METH) / 1e3
    E.set_ylim(0, top * 1.85); E.set_yticks(np.arange(0, top * 1.85, 40)); E.set_ylabel('Localizations (×10$^3$)')
    E.legend(loc='upper left', title='number: rms error of\nthe true positives', fontsize=FS.FS_SMALL, title_fontsize=FS.FS_SMALL,
             handlelength=1.2, alignment='left')
    E.set_title('Localization error'); let(xe - 0.30, y1, 'e', img=False)
    # bottom row: i GT-FRC curves | j GT-FRC vs frames | k PSF
    xI, wI = 0.42, 2.30; xJ, wJ = xI + wI + 0.52, 2.05
    I = add(xI, yb, wI, hb); J = add(xJ, yb, wJ, hb)
    for k, lk in METH:
        s = SEL[k]; gc = s['sc']['gt_curve']; lw = 1.4 if k == 'v2' else FS.LW
        f_ = np.asarray(gc['freq'][2:]); c_ = np.asarray(gc['curve'][2:]); sel = f_ <= FS.FRC_XMAX
        I.plot(f_[sel], c_[sel], color=COLM[k], lw=lw, label=f"{lk}: {s['r'][a.crit]:.1f} nm")
        t = s['ttf']; J.plot(np.array(t['frames']) / 1e3, t['res_nm'], '-o', color=COLM[k], lw=lw, ms=2.6 if k == 'v2' else 2.2, label=lk)
    FS.frc_axes(I, ylabel='GT-FRC')
    I.set_title('FRC against the ground truth'); I.legend(loc='upper right', title='GT-FRC resolution', handlelength=1.2)
    let(xI - 0.40, yb, 'i', img=False)
    J.set_xscale('log'); J.set_yscale('log'); J.axhline(30, color='0.5', lw=0.7, ls=':')
    J.text(0.27, 31, '30 nm', color='0.4', fontsize=FS.FS_SMALL, ha='left', va='bottom')
    J.set_xticks([0.25, 1, 4, 15, 60]); J.set_xticklabels(['0.25', '1', '4', '15', '60']); J.set_ylim(12, 250)
    J.set_yticks([15, 20, 30, 50, 100, 200]); J.set_yticklabels(['15', '20', '30', '50', '100', '200']); J.minorticks_off()
    J.set_xlabel('Frames (×1000)'); J.set_ylabel('GT-FRC resolution (nm)')
    J.legend(loc='upper right', handlelength=1.6, fontsize=FS.FS_SMALL)
    J.set_title('GT-FRC vs frames (time to fidelity)'); let(xJ - 0.40, yb, 'j', img=False)
    info = psfpanel.sim_psf_info(md, a.set.startswith('Dexp'))
    import fig_exp_new as FE
    x0 = xJ + wJ + 0.30
    axy, axz = FE.psf_axes(fig, x0, yb, Wf - rm - x0, H - yb - 0.03, info)
    psfpanel.draw(axy, axz, info, fontsize=FS.FS_SMALL)
    axz.yaxis.tick_right(); axz.yaxis.set_label_position('right')
    axy.set_title(f"PSF xy (log)\nz = {info['z_um'] * 1e3:.0f} nm", fontsize=FS.FS_TICK, pad=2); axz.set_title('PSF xz (log)', fontsize=FS.FS_TICK, pad=2)
    fig.text((x0 - 0.02) / Wf, (H - yb + 0.07) / H, 'k', fontsize=FS.FS_LETTER, fontweight='bold', ha='left', va='bottom')
    axz.annotate(psfpanel.panel_label(info), (0.0, 0.0), xycoords='axes fraction', xytext=(0, -3), textcoords='offset points',
                 fontsize=FS.FS_TICK, va='top', ha='left', linespacing=1.15)
    fig._psf_info = info
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='Dexp1_v2psf_wsm'); ap.add_argument('--crit', default='gt_sust')
    ap.add_argument('--v2fam', default='sig_p'); ap.add_argument('--r_prec', type=float, default=20.0); ap.add_argument('--r_fp', type=float, default=50.0)
    ap.add_argument('--name', default=None, help='output name (default fig_sim_<set without the density 1>)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2')); a = ap.parse_args()
    import frcopt as FO, psfpanel
    md = os.path.join(C.OUT, a.set); R, B = load(a.set, a.v2fam)
    g0 = json.load(open(os.path.join(md, 'gen.json')))
    gt = np.load(os.path.join(md, 'gt.npz')); W = R['W']; nf = R['nframes']; (cx, cy), w = crop_of(md)
    Sc = FO.Scorer(W, gt['pts'][:, :2])
    lab = dict(METH)
    # the chosen selection per method
    SEL = {}
    for k, _ in METH:
        Mk = R['methods'][k]; fam = a.v2fam if k == 'v2' else 'sig'; rows = Mk['fam'][fam]
        j = Mk['opt'][fam][a.crit]['idx']; ro = rows[j]; m = B[k]['key'] <= ro['thr']
        sc = Sc(B[k], m, curves=True)
        assert sc['n'] == ro['n'] and abs(sc[a.crit] - ro[a.crit]) < 1e-9, (k, sc['n'], ro['n'])
        SEL[k] = dict(fam=fam, rows=rows, j=j, r=ro, m=m, sc=sc, n_base=Mk['n_base'], cls=classes(B[k], m, a.r_prec, a.r_fp),
                      ttf=ttf(Sc, B[k], m, nf))

    fig = draw(a, md, gt, W, cx, cy, w, B, SEL, lab, psfpanel); info = fig._psf_info

    name = a.name or f"fig_sim_{a.set.replace('Dexp1_', 'Dexp_')}"
    os.makedirs(a.out, exist_ok=True); out = os.path.join(a.out, name)
    H = R['methods']
    hs_gap = {k: (SEL[k]['rows'][H[k]['opt'][SEL[k]['fam']]['hs_first']['idx']][a.crit] - SEL[k]['r'][a.crit]) for k, _ in METH}
    thr = ', '.join(f"{lab[k]} {SEL[k]['r']['thr'] * C.PX_NM:.1f} nm" for k, _ in METH)
    f30 = lambda k: (lambda v: 'not reached' if v is None else '{:.1f}k'.format(v / 1e3))(SEL[k]['ttf']['frames_to'].get('30'))
    zd = g0['z_dist_um']; ws = json.load(open(os.path.join(md, 'gen.json'))).get('wsm') or {}
    ft = lambda k, t: (lambda v: 'not reached' if v is None else f"{v / 1e3:.1f}k")(SEL[k]['ttf']['frames_to'].get(str(t)))
    cap = [f"Fig. 2. Simulation at {g0['density']:g} emitter per µm² per frame, {nf // 1000} 000 frames, {W} x {W} camera pixels of "
           f"{C.PX_NM} nm ({W * C.PX_NM / 1000:.2f} µm field). Simulated with the in-situ vectorial PSF of the recording of Fig. 3 and "
           f"that recording's sCMOS camera maps (gain, offset, per-pixel read noise); emitter heights 0-400 nm above the coverslip "
           f"(median {zd['p50'] * 1e3:.0f} nm); photons and background as measured in that recording (median "
           f"{g0['brightness_fit']['fit_median']:.0f} detected photons per 6-frame window, background {g0['bg_per_frame']['mean']:.2f} "
           f"photons per pixel and frame); mean on-time {g0['avg_on_time']:g} frames; pattern of 2 orientations x 3 phase steps, pitch "
           f"{C.PITCH_NM:g} nm, modulation depth {C.DEPTH:g}.",
           f"(a) Ground truth: all labelling sites (orange) of a simulated microtubule network (webSMLM, microtubules-in-cells sample; "
           f"{ws.get('labels_per_nm', 0.5):.1f} labels per nm of filament, {ws.get('mt_length_um', float('nan')):.0f} µm of filament, "
           f"{ws.get('n_labels', 0) // 1000} {ws.get('n_labels', 0) % 1000:03d} sites in the field). The labels sit ~25-30 nm from the tube axis (13-protofilament lattice, 12 nm "
           "binder, 2-5 nm linker); their heights are drawn from the experimental height distribution. White box: the zoom of b, c and "
           f"f-h ({w * C.PX_NM / 1000:.2f} x {w * C.PX_NM / 1000:.2f} µm, picked at filament crossings). (b) The ground truth of the zoom, "
           "rendered as the reconstructions.",
           f"(c, f-h) Renders of the zoom by (c) SMLM, (f) SIMFLUX, (g) DECODE and (h) SIMCODE ({C.PX_NM / ZOOM:.1f} nm pixels, 1-pixel "
           f"Gaussian blur, colour scale saturated at the {FS.CLIP_PCT:g}th percentile of the image's pixels). Each method's "
           "localizations on disjoint 6-frame windows (one localization per window and emitter, no photon used twice) with a "
           "precision < 0.15 px (16.2 nm; the larger of the x and y values; SIMCODE also needs a detection probability of at least "
           "0.63), then cut at the threshold on the method's own reported precision (SMLM / SIMFLUX: CRLB; DECODE: its network "
           f"uncertainty; SIMCODE: its calibrated uncertainty) that gives the best GT-FRC: {thr}. Titles: GT-FRC resolution and the "
           "localizations kept (% of the localizations before the cut). SIMFLUX uses the pattern estimated from the data, as SIMCODE "
           "does. SMLM and DECODE get the sum of the 6 frames; DECODE was retrained for this PSF, the emitter heights and these photon, "
           "background and density ranges. GT-FRC = the FRC between the rendered reconstruction and the rendered ground truth "
           f"(5 nm pixels, full field, all {nf // 1000} 000 frames); resolution = the frequency after which the curve stays below 1/7 "
           "for 0.01 nm^-1 (the hollow tube's spectrum has zeros near 65, 28 and 18 nm, where a curve can dip below 1/7 and recover).",
           "(d) GT-FRC vs the fraction of localizations kept when the precision threshold is lowered from all localizations to 10 %; "
           "dot = the chosen threshold. Choosing the threshold by the FRC between two halves of the data instead (no ground truth, as "
           "for the experimental data) changes the GT-FRC by "
           + ', '.join(f"{lab[k]} {hs_gap[k]:+.1f} nm" for k, _ in METH) + ".",
           f"(e) The rendered localizations by their 2D error to the nearest emitter that is on in the window: <= {a.r_prec:g} nm "
           f"(solid), {a.r_prec:g}-{a.r_fp:g} nm (light), no emitter on within {a.r_fp:g} nm = false positive (hatched): "
           + '; '.join(f"{lab[k]} {SEL[k]['r']['n']:,} ({SEL[k]['cls'][0]:,} / {SEL[k]['cls'][1]:,} / {SEL[k]['cls'][2]:,})".replace(',', ' ') for k, _ in METH)
           + ". Number above each bar: rms error per axis of the true positives (localizations with an emitter on within 3 times "
           "their own reported precision): " + ', '.join(f"{lab[k]} {SEL[k]['r']['rms_tp_nm']:.1f} nm" for k, _ in METH) + ".",
           f"(i) GT-FRC curves of the rendered sets (all {nf // 1000} 000 frames); dashed line = 1/7; resolution in the legend. The "
           "curves are drawn exactly as the resolution is read from them: smoothed over 5 frequency bins (a 5-bin moving average, "
           "no extra smoothing for the plot); the first two frequency bins are not drawn.",
           "(j) Time to fidelity: GT-FRC of the same selection on the first N frames (N = 250 ... "
           f"{nf // 1000} 000); dotted line = 30 nm. Frames to reach 40 / 30 / 20 nm: "
           + '; '.join(f"{lab[k]} " + ('none reached' if all(SEL[k]['ttf']['frames_to'].get(str(t)) is None for t in (40, 30, 20))
                                          else f"{ft(k, 40)} / {ft(k, 30)} / {ft(k, 20)}") for k, _ in METH) + ".",
           f"(k) The PSF of the simulation (the in-situ vectorial PSF fit of the recording of Fig. 3, Zernike modes 5-21; "
           f"{info['rms_mwaves']:.0f} mλ rms aberration): xy section at the median emitter height ({info['z_um'] * 1e3:.0f} nm) and xz "
           "section, log scale over 3 decades; in the xz section the white dashed line marks the focal plane and the cyan dotted line "
           "the median emitter height. Scale bars 500 nm. " + psfpanel.caption_text(info).split('\n', 1)[1]]
    cap = [c.replace('µm²', 'µm²') for c in cap]
    open(out + '_caption.txt', 'w').write('\n'.join(cap) + '\n')
    notes = [f"Internal notes to {name} (not for the caption).",
             f"Ground truth: {gt_desc(md, paper=True)}; webSMLM 'microtubules (cells)' sample, which embeds the inSiliScope CellField "
             f"world model; webSMLM defaults except the label efficiency (label_eff {ws.get('label_eff', float('nan')):.3f} -> "
             f"{ws.get('labels_per_nm', float('nan')):.2f} labels per nm); spec {ws.get('spec')}, seed {ws.get('seed')}, offset "
             f"{ws.get('offset_um')} um, {ws.get('n_mt_in_fov')} microtubules in the field; zoom box (x0, y0, w) = ({cx}, {cy}, {w}) px.",
             f"Selection: set {a.set}, criterion {a.crit} (sustained 1/7 crossing), SIMCODE family {a.v2fam}; frcopt sweep {FRCD}/{a.set}.json.",
             f"Style: figstyle.py (font {FS.FONT}, 180 mm wide, render clip {FS.CLIP_PCT:g} %)."]
    open(out + '_notes_internal.txt', 'w').write('\n'.join(notes) + '\n')
    vals = dict(set=a.set, criterion=a.crit, v2_family=a.v2fam, r_prec_nm=a.r_prec, r_fp_nm=a.r_fp, methods={
        k: dict(label=lab[k], threshold_nm=SEL[k]['r']['thr'] * C.PX_NM, quantile=SEL[k]['r']['q'], n=SEL[k]['r']['n'], n_base=SEL[k]['n_base'],
                kept=SEL[k]['r']['n'] / SEL[k]['n_base'], gt_frc_nm=SEL[k]['r'][a.crit], gt_frc_first_nm=SEL[k]['r']['gt_first'],
                half_split_frc_nm=SEL[k]['r']['hs_first'], fp_3sigma=SEL[k]['r']['fp'], rms_tp_nm=SEL[k]['r']['rms_tp_nm'],
                classes=dict(zip(('precise', 'imprecise', 'false_positive'), SEL[k]['cls'])), ttf=SEL[k]['ttf'],
                gt_frc_at_half_split_optimum_delta_nm=hs_gap[k]) for k, _ in METH})
    json.dump(vals, open(out + '_values.json', 'w'), indent=1, default=float)
    FS.save(fig, out)
    print(json.dumps({k: (round(v['threshold_nm'], 2), v['n'], round(100 * v['kept']), round(v['gt_frc_nm'], 1), v['classes'], v['ttf']['frames_to']) for k, v in vals['methods'].items()}))


if __name__ == '__main__':
    main()
