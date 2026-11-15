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
from fig_sim_fig2 import letter, scalebar, FS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE')]
COLM = dict(smlm=COL['smlm'], sf=COL['sf'], decode=COL['decode'], v2=COL['v2sig'])
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

    fig = plt.figure(figsize=(13.0, 10.9))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2, 1.0], hspace=0.28, left=0.055, right=0.985, top=0.935, bottom=0.06)
    gs = outer[0].subgridspec(2, 4, hspace=0.30, wspace=0.16)
    A = fig.add_subplot(gs[0, 0]); A.set_facecolor('k')
    p = gt['pts']; A.plot(p[:, 0], p[:, 1], ',', color='#ffb070', alpha=0.9, ms=0.5, rasterized=True)
    A.add_patch(Rectangle((cx, cy), w, w, fill=False, ec='w', lw=2.5))
    A.set_xlim(-0.5, W - 0.5); A.set_ylim(W - 0.5, -0.5); A.set_xticks([]); A.set_yticks([])
    L = 2000 / C.PX_NM; A.plot([0.05 * W, 0.05 * W + L], [0.93 * W] * 2, 'w-', lw=4, solid_capstyle='butt')
    A.text(0.05 * W, 0.90 * W, '2000 nm', color='w', fontsize=10, va='bottom')
    A.set_title('Ground truth\n(all labelling sites)', fontsize=FS); letter(A, 'A')
    A = fig.add_subplot(gs[0, 1]); A.set_xticks([]); A.set_yticks([])
    A.imshow(render(p[:, 0], p[:, 1], (cx, cy), w), cmap='inferno'); A.set_title('Ground truth (zoom)', fontsize=FS); letter(A, 'B'); scalebar(A, w, 500)
    slots = {'smlm': (0, 2, 'C'), 'sf': (1, 1, 'F'), 'decode': (1, 2, 'G'), 'v2': (1, 3, 'H')}
    AX = {}
    for k, lk in METH:
        s = SEL[k]; r, c, Lt = slots[k]; A = AX[k] = fig.add_subplot(gs[r, c]); A.set_xticks([]); A.set_yticks([])
        A.imshow(render(B[k]['x'][s['m']], B[k]['y'][s['m']], (cx, cy), w), cmap='inferno')
        A.set_title(f"{lk}\nGT-FRC = {s['r'][a.crit]:.1f} nm, {100 * s['r']['n'] / s['n_base']:.0f} % kept", fontsize=FS - 1)
        letter(A, Lt); scalebar(A, w, 500)
    # D: GT-FRC vs fraction kept, one marker = the chosen threshold
    D = fig.add_subplot(gs[0, 3])
    for k, lk in METH:
        s = SEL[k]; fr = np.array([q['n'] for q in s['rows']]) / s['n_base'] * 100; v = np.array([q.get(a.crit, np.nan) for q in s['rows']])
        D.plot(fr, v, '-', color=COLM[k], lw=1.6, label=lk)
        D.plot(100 * s['r']['n'] / s['n_base'], s['r'][a.crit], 'o', color=COLM[k], mec='k', mew=0.8, ms=8)
    D.set_yscale('log'); D.set_yticks([15, 20, 30, 50, 100]); D.set_yticklabels(['15', '20', '30', '50', '100']); D.minorticks_off()
    D.set_ylim(14, 120); D.set_xlim(0, 103); D.set_xlabel('localizations kept (%)', fontsize=10); D.set_ylabel('GT-FRC (nm)', fontsize=10)
    D.legend(fontsize=7.5, frameon=False, loc='upper right', ncol=2); D.tick_params(labelsize=9)
    D.set_title('GT-FRC vs selection threshold', fontsize=FS)
    # E: localization classes at the chosen selection
    E = fig.add_subplot(gs[1, 0]); xs = np.arange(len(METH))
    for i, (k, lk) in enumerate(METH):
        pc, ic, fc = SEL[k]['cls']
        E.bar(i, pc / 1e5, 0.75, color=COLM[k]); E.bar(i, ic / 1e5, 0.75, bottom=pc / 1e5, color=COLM[k], alpha=0.4)
        E.bar(i, fc / 1e5, 0.75, bottom=(pc + ic) / 1e5, color='w', ec=COLM[k], hatch='////', lw=0.8)
    E.bar([0], [0], color='0.35', label=f'error ≤ {a.r_prec:g} nm'); E.bar([0], [0], color='0.35', alpha=0.4, label=f'error {a.r_prec:g}-{a.r_fp:g} nm')
    E.bar([0], [0], color='w', ec='0.35', hatch='////', label=f'false positive (> {a.r_fp:g} nm)')
    E.set_xticks(xs); E.set_xticklabels([lk for _, lk in METH], rotation=35, ha='right', fontsize=9, rotation_mode='anchor')
    top = max(sum(SEL[k]['cls']) for k, _ in METH) / 1e5
    E.set_ylim(0, top * 1.45); E.set_ylabel('Counts (x 10$^5$)', fontsize=11); E.tick_params(axis='y', labelsize=10)
    E.legend(fontsize=7.5, frameon=False, loc='upper left'); E.set_title('Localizations', fontsize=FS); letter(E, 'E', dx=-0.17)
    # bottom row: I GT-FRC curves | J GT-FRC vs frames | K PSF
    gb = outer[1].subgridspec(1, 3, wspace=0.26, width_ratios=[1.25, 1.25, 0.85])
    I = fig.add_subplot(gb[0]); J = fig.add_subplot(gb[1])
    for k, lk in METH:
        s = SEL[k]; gc = s['sc']['gt_curve']; lw = 2.4 if k == 'v2' else 1.5
        I.plot(gc['freq'][2:], gc['curve'][2:], color=COLM[k], lw=lw, label=f"{lk}  {s['r'][a.crit]:.1f} nm")
        t = s['ttf']; J.plot(np.array(t['frames']) / 1e3, t['res_nm'], '-o', color=COLM[k], lw=lw, ms=5 if k == 'v2' else 4, label=lk)
    I.axhline(1 / 7, color='grey', ls='--', lw=1); I.set_xlim(0, 0.09); I.set_ylim(-0.03, 1.0)
    I.set_xlabel('Spatial freq. [nm$^{-1}$]', fontsize=11); I.set_ylabel('GT-FRC', fontsize=11); I.tick_params(labelsize=10)
    I.set_title('FRC against the ground truth', fontsize=FS); I.legend(fontsize=9, loc='upper right'); letter(I, 'I', dx=-0.12)
    J.set_xscale('log'); J.set_yscale('log'); J.axhline(30, color='0.5', lw=1, ls=':')
    J.text(0.27, 30.8, '30 nm', color='0.4', fontsize=9, ha='left', va='bottom')
    J.set_xticks([0.25, 1, 4, 15, 60]); J.set_xticklabels(['0.25', '1', '4', '15', '60']); J.set_ylim(12, 200)
    J.set_yticks([15, 20, 30, 50, 100]); J.set_yticklabels(['15', '20', '30', '50', '100']); J.minorticks_off(); J.tick_params(labelsize=10)
    J.set_xlabel('frames (x 1000)', fontsize=11); J.set_ylabel('GT-FRC resolution (nm)', fontsize=11)
    J.set_title('GT-FRC vs frames (time to fidelity)', fontsize=FS); letter(J, 'J', dx=-0.12)
    info = psfpanel.sim_psf_info(md, a.set.startswith('Dexp'))
    gp = gb[2].subgridspec(2, 2, height_ratios=[1.0, 0.30], wspace=0.75, hspace=0.05)
    axy, axz, axt = fig.add_subplot(gp[0, 0]), fig.add_subplot(gp[0, 1]), fig.add_subplot(gp[1, :])
    psfpanel.draw(axy, axz, info, fontsize=8); axz.set_box_aspect(1); letter(axy, 'K', dx=-0.12, dy=1.12)
    axt.axis('off'); axt.text(0.0, 0.9, psfpanel.panel_label(info), transform=axt.transAxes, fontsize=FS - 1, va='top', ha='left', linespacing=1.2)
    # D: axes box exactly above H's image (x-extent) and level with C's image (y-extent)
    fig.canvas.draw()
    pH = AX['v2'].get_position(); pC = AX['smlm'].get_position()
    dy = 0.045; D.set_position([pH.x0, pC.y0 + dy, pH.width, pC.height - dy]); letter(D, 'D', dx=-0.22, dy=1.1)

    name = a.name or f"fig_sim_{a.set.replace('Dexp1_', 'Dexp_')}"
    os.makedirs(a.out, exist_ok=True); out = os.path.join(a.out, name)
    H = R['methods']
    hs_gap = {k: (SEL[k]['rows'][H[k]['opt'][SEL[k]['fam']]['hs_first']['idx']][a.crit] - SEL[k]['r'][a.crit]) for k, _ in METH}
    thr = ', '.join(f"{lab[k]} {SEL[k]['r']['thr'] * C.PX_NM:.1f} nm" for k, _ in METH)
    f30 = lambda k: (lambda v: 'not reached' if v is None else '{:.1f}k'.format(v / 1e3))(SEL[k]['ttf']['frames_to'].get('30'))
    zd = g0['z_dist_um']
    cap = [f"Fig 2 (\\label{{fig:simulation}}), simulation at {g0['density']:g} emitter/µm²/frame, {nf} frames, {W} x {W} px of {C.PX_NM} nm "
           f"({W * C.PX_NM / 1000:.2f} µm field). Simulated with the in-situ vectorial PSF of the 1809/2 movie and that movie's sCMOS camera maps "
           f"(gain, offset, per-pixel read noise); emitter heights 0-400 nm above the coverslip (median {zd['p50'] * 1e3:.0f} nm); experimental "
           f"photons and background of 1809/2 (median {g0['brightness_fit']['fit_median']:.0f} detected photons per 6-frame window, background "
           f"{g0['bg_per_frame']['mean']:.2f} photons/px/frame); mean on-time {g0['avg_on_time']:g} frames; pattern 2 angles x 3 phase steps, pitch "
           f"{C.PITCH_NM:g} nm, modulation depth {C.DEPTH:g}.",
           f"(A) Ground truth: all labelling sites of the {gt_desc(md, paper=True)} (orange); white box = the zoom of B, C, F-H "
           f"({w * C.PX_NM / 1000:.2f} µm, centred on filament crossings). Ground truth from webSMLM (microtubules (cells) sample, which embeds the "
           "inSiliScope world model), webSMLM defaults except the label efficiency (0.5 labels per nm of filament); the labels sit on the "
           "13-protofilament lattice with a 12 nm binder and 2-5 nm linker (~25-30 nm from the tube axis), their heights are drawn from the "
           "experimental height distribution as for every emitter. (B) The ground truth of the zoom, rendered exactly as the reconstructions.",
           f"(C, F-H) Renders of the zoom by (C) SMLM, (F) SIMFLUX, (G) DECODE and (H) SIMCODE (5.4 nm pixels, 1-pixel Gaussian blur, 99.5 % clip). "
           "Each method's localizations: one per emitter and disjoint 6-frame window with its own estimated precision max(σx, σy) < 16.25 nm "
           "(SIMCODE: and detection score p >= 0.63), then cut at the threshold on the method's own reported precision (SMLM / SIMFLUX: CRLB; DECODE: "
           f"its predicted σ; SIMCODE: calibrated σ) that minimizes its GT-FRC: {thr}. Titles: GT-FRC and the localizations kept (% of the set "
           "before the cut). SIMFLUX uses the pattern estimated from the data, as SIMCODE does. SMLM and DECODE get 6-frame sums; DECODE was "
           "retrained for this PSF, the emitter heights and these photon / background / density ranges. GT-FRC = resolution from the FRC between "
           "the rendered reconstruction and the rendered ground truth (5 nm pixels, full field, 60 000 frames, 5-shell smoothing): the frequency "
           "after which the curve stays below 1/7 for 0.01 nm^-1 (the hollow tube's spectrum has nulls, near 65, 28 and 18 nm for its 25-30 nm "
           "radius, where a curve can dip below 1/7 and recover).",
           "(D) GT-FRC vs the fraction of localizations kept when the precision threshold is lowered from all rows to 10 %; dot = the chosen "
           "threshold. Choosing the threshold by the half-split FRC instead (no ground truth, as for the experimental data) changes the GT-FRC by "
           + ', '.join(f"{lab[k]} {hs_gap[k]:+.1f} nm" for k, _ in METH) + " (all densities: SI table).",
           f"(E) Localizations of the rendered sets by their 2D error to the nearest emitter ON in the window: ≤ {a.r_prec:g} nm (solid), "
           f"{a.r_prec:g}-{a.r_fp:g} nm (light), no emitter ON within {a.r_fp:g} nm = false positive (hatched): "
           + '; '.join(f"{lab[k]} {SEL[k]['r']['n']} ({SEL[k]['cls'][0]} / {SEL[k]['cls'][1]} / {SEL[k]['cls'][2]})" for k, _ in METH) + ".",
           "(I) GT-FRC curves of the rendered sets (60 000 frames); dashed line = 1/7; resolution in the legend; curves smoothed with a 5-sample "
           "moving average, the first two frequency samples not drawn.",
           f"(J) Time to fidelity: GT-FRC on the first N frames (N = 250 ... 60 000), same selection; dotted line = 30 nm. Frames to 30 nm: "
           + ', '.join(f"{lab[k]} {f30(k)}" for k, _ in METH) + ".",
           psfpanel.caption_text(info, what='(K)').replace('(K):', '(K)', 1)]
    open(out + '_caption.txt', 'w').write('\n'.join(cap) + '\n')
    vals = dict(set=a.set, criterion=a.crit, v2_family=a.v2fam, r_prec_nm=a.r_prec, r_fp_nm=a.r_fp, methods={
        k: dict(label=lab[k], threshold_nm=SEL[k]['r']['thr'] * C.PX_NM, quantile=SEL[k]['r']['q'], n=SEL[k]['r']['n'], n_base=SEL[k]['n_base'],
                kept=SEL[k]['r']['n'] / SEL[k]['n_base'], gt_frc_nm=SEL[k]['r'][a.crit], gt_frc_first_nm=SEL[k]['r']['gt_first'],
                half_split_frc_nm=SEL[k]['r']['hs_first'], fp_3sigma=SEL[k]['r']['fp'], rms_tp_nm=SEL[k]['r']['rms_tp_nm'],
                classes=dict(zip(('precise', 'imprecise', 'false_positive'), SEL[k]['cls'])), ttf=SEL[k]['ttf'],
                gt_frc_at_half_split_optimum_delta_nm=hs_gap[k]) for k, _ in METH})
    json.dump(vals, open(out + '_values.json', 'w'), indent=1, default=float)
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')
    print(json.dumps({k: (round(v['threshold_nm'], 2), v['n'], round(100 * v['kept']), round(v['gt_frc_nm'], 1), v['classes'], v['ttf']['frames_to']) for k, v in vals['methods'].items()}))


if __name__ == '__main__':
    main()
