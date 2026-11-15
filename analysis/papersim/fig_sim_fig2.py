"""Main-text Fig 2 (\\label{fig:simulation}) of SIMCODE v2, reworked FROM the preprint's Fig 2 (fig_sim2D_v4_d09.pdf).

Preprint layout (3 columns): A ground truth (field + zoom box) | B SMLM | C SIMFLUX; D localization counts | E DECODE | F SIMCODE;
G FRC curves | H Jaccard vs frames. Kept: the same grid (ground truth top-left with the zoom box, counts below it, zoom renders to
the right with the FRC in the tile title, two curve panels across the bottom), the preprint's density (1 emitter / µm² / frame)
and zoom (the same 16 px crop, fig_density.CROP). Changed for the v2 data / method:
  - one column more: the preprint SIMCODE (estimated pattern) and SIMCODE v2 are both shown; the spare top-right slot holds the
    PSF panel (xy | xz of the in-situ vectorial PSF the movies were simulated with, psfpanel.py)
  - A: ground truth over the illumination pattern of the first frame (angle 1, step 1)
  - render titles: GT-FRC (FRC against the ground truth) instead of the half-split FRC
  - counts: split into localizations with an emitter ON within 3 σ in their window and false positives
  - G FRC curves -> GT-FRC curves; H Jaccard vs frames -> GT-FRC resolution vs frames (time to fidelity)
  - the per-density summary is the SI table of fig_sim_main.py (out/papersim/figures/fig_sim_Dexp_v2psf_table.{md,csv})
usage: python analysis/papersim/fig_sim_fig2.py [--density 1] [--prefix Dexp] [--suffix _v2psf] [--model m1809v2]
output: out/paper_figs_v2/fig_sim_<prefix><suffix>.{pdf,svg,png}, _caption.txt"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M
from fig_density import render, CROP, COL
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# render order as in the preprint (B SMLM, C SIMFLUX, E DECODE, F SIMCODE) + SIMCODE v2
METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v1', 'SIMCODE (preprint)'), ('v2sig', 'SIMCODE v2')]
FS = 12          # title font size (preprint: large titles at 0.8 linewidth)


def letter(ax, s, dx=-0.06, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom', ha='right')


def scalebar(A, w, nm, zoom=20, fs=10):
    L = nm / (C.PX_NM / zoom); n = w * zoom
    A.plot([0.05 * n, 0.05 * n + L], [0.93 * n] * 2, 'w-', lw=4, solid_capstyle='butt')
    A.text(0.05 * n, 0.90 * n, f'{nm:g} nm', color='w', fontsize=fs, va='bottom')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--density', type=float, default=1.0); ap.add_argument('--prefix', default='Dexp')
    ap.add_argument('--suffix', default='_v2psf'); ap.add_argument('--model', default='m1809v2')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2')); a = ap.parse_args()
    run = os.path.join(C.OUT, f'{a.prefix}{a.density:g}{a.suffix}')
    R = json.load(open(os.path.join(run, 'full', f'metrics_{a.model}.json')))
    G = json.load(open(os.path.join(run, 'full', f'gtfrc_{a.model}.json')))['methods']
    gt = np.load(os.path.join(run, 'gt.npz')); W = int(R['W']); nfr = int(R['nframes'])
    (cx, cy), w = CROP
    lab = dict(METH)

    fig = plt.figure(figsize=(13.0, 10.6))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2, 1.0], hspace=0.30, left=0.055, right=0.985, top=0.935, bottom=0.065)
    gs = outer[0].subgridspec(2, 4, hspace=0.30, wspace=0.12)

    # A: ground truth (all labelling sites) over the illumination of frame 0 (angle 1, step 1), zoom box
    A = fig.add_subplot(gs[0, 0]); n = 400
    u = (np.arange(n) + 0.5) / n * W; X, Y = np.meshgrid(u, u)
    ill = C.illum(dict(k=gt['k'], phi=gt['phi'], m=gt['m'], rel=gt['rel']), X.ravel(), Y.ravel(), np.zeros(X.size, int)).reshape(n, n)
    A.imshow(ill / ill.max(), cmap='gray', vmin=0, vmax=3.2, extent=[0, W, W, 0], interpolation='bilinear')
    p = gt['pts']; A.plot(p[:, 0], p[:, 1], ',', color='#ffb070', alpha=0.9, ms=0.5, rasterized=True)
    A.add_patch(Rectangle((cx, cy), w, w, fill=False, ec='w', lw=2.5))
    A.set_xlim(0, W); A.set_ylim(W, 0); A.set_xticks([]); A.set_yticks([])
    L = 2000 / C.PX_NM; A.plot([0.05 * W, 0.05 * W + L], [0.93 * W] * 2, 'w-', lw=4, solid_capstyle='butt')
    A.text(0.05 * W, 0.90 * W, '2000 nm', color='w', fontsize=10, va='bottom')
    A.set_title('Ground truth\n+ illumination pattern', fontsize=FS); letter(A, 'A')

    # renders (B, C top row; F, G, H second row) of the preprint's zoom
    slots = {'smlm': (0, 1, 'B'), 'sf': (0, 2, 'C'), 'decode': (1, 1, 'F'), 'v1': (1, 2, 'G'), 'v2sig': (1, 3, 'H')}
    cnt = {}
    for k, lab_k in METH:
        r, c, L = slots[k]; A = fig.add_subplot(gs[r, c]); A.set_xticks([]); A.set_yticks([])
        T = M.load_method(k, R['methods'][k]['path']); _, _, f = M.frc_set(T, W, nfr)
        A.imshow(render(T['x'][f], T['y'][f], (cx, cy), w), cmap='inferno')
        A.set_title(f"{lab_k}\nGT-FRC = {G[k]['gt_frc']['res_nm']:.1f} nm", fontsize=FS); letter(A, L)
        scalebar(A, w, 500)
        assert G[k]['n'] == R['methods'][k]['n_frc'], (k, G[k]['n'], R['methods'][k]['n_frc'])
        cnt[k] = (G[k]['n'], G[k]['ghost']['ghost_fraction'])

    # D: PSF panel (top-right slot): xy | xz, two-line label
    import psfpanel
    info = psfpanel.sim_psf_info(run, a.prefix == 'Dexp')
    gp = gs[0, 3].subgridspec(2, 2, height_ratios=[1.0, 0.30], wspace=0.75, hspace=0.05)
    axy, axz, axt = fig.add_subplot(gp[0, 0]), fig.add_subplot(gp[0, 1]), fig.add_subplot(gp[1, :])
    psfpanel.draw(axy, axz, info, fontsize=8); axz.set_box_aspect(1); letter(axy, 'D', dx=-0.12, dy=1.12)
    axt.axis('off'); axt.text(0.0, 0.9, psfpanel.panel_label(info), transform=axt.transAxes, fontsize=FS - 1, va='top', ha='left', linespacing=1.2)

    # E: localization counts (FRC set), split into correct and false positives
    A = fig.add_subplot(gs[1, 0]); keys = [k for k, _ in METH]; xs = np.arange(len(keys))
    nn = np.array([cnt[k][0] for k in keys]) / 1e5; gf = np.array([cnt[k][1] for k in keys])
    A.bar(xs, nn * (1 - gf), color=[COL[k] for k in keys], width=0.75)
    A.bar(xs, nn * gf, bottom=nn * (1 - gf), color=[COL[k] for k in keys], alpha=0.35, hatch='///', ec='w', lw=0, width=0.75)
    A.bar([0], [0], color='0.4', label='emitter ON within 3σ'); A.bar([0], [0], color='0.4', alpha=0.35, hatch='///', ec='w', lw=0, label='false positive')
    A.set_xticks(xs); A.set_xticklabels([lab[k] for k in keys], rotation=35, ha='right', fontsize=9, rotation_mode='anchor')
    A.set_ylabel('Counts (x 10$^5$)', fontsize=11); A.tick_params(axis='y', labelsize=10); A.legend(fontsize=8, frameon=False, loc='upper left')
    A.set_ylim(0, nn.max() * 1.3); A.set_title('Localizations', fontsize=FS); letter(A, 'E', dx=-0.17)

    # I: GT-FRC curves; J: GT-FRC resolution vs frames (time to fidelity)
    gb = outer[1].subgridspec(1, 2, wspace=0.22)
    A = fig.add_subplot(gb[0]); B = fig.add_subplot(gb[1])
    for k in keys:
        g = G[k]; lw = 2.4 if k == 'v2sig' else 1.5
        A.plot(g['gt_frc']['freq'][2:], g['gt_frc']['curve'][2:], color=COL[k], lw=lw, label=f"{lab[k]}  {g['gt_frc']['res_nm']:.1f} nm")
        t = g['ttf']; B.plot(np.array(t['frames']) / 1e3, t['res_nm'], '-o', color=COL[k], lw=lw, ms=4 if k != 'v2sig' else 5, label=lab[k])
    A.axhline(1 / 7, color='grey', ls='--', lw=1); A.set_xlim(0, 0.09); A.set_ylim(-0.03, 1.0)
    A.set_xlabel('Spatial freq. [nm$^{-1}$]', fontsize=11); A.set_ylabel('GT-FRC', fontsize=11); A.tick_params(labelsize=10)
    A.set_title('FRC against the ground truth', fontsize=FS); A.legend(fontsize=9, frameon=True, loc='upper right'); letter(A, 'I', dx=-0.09)
    B.set_xscale('log'); B.set_yscale('log'); B.axhline(30, color='0.5', lw=1, ls=':')
    B.text(60, 30.8, '30 nm', color='0.4', fontsize=9, ha='right', va='bottom')
    B.set_xticks([0.25, 1, 4, 15, 60]); B.set_xticklabels(['0.25', '1', '4', '15', '60']); B.set_ylim(12, 150)
    B.set_yticks([15, 20, 30, 50, 100]); B.set_yticklabels(['15', '20', '30', '50', '100']); B.minorticks_off(); B.tick_params(labelsize=10)
    B.set_xlabel('frames (x 1000)', fontsize=11); B.set_ylabel('GT-FRC resolution (nm)', fontsize=11)
    B.set_title('GT-FRC vs frames (time to fidelity)', fontsize=FS); letter(B, 'J', dx=-0.09)

    os.makedirs(a.out, exist_ok=True); out = os.path.join(a.out, f'fig_sim_{a.prefix}{a.suffix}')
    # caption material
    g0 = json.load(open(os.path.join(run, 'gen.json'))); zd = g0['z_dist_um']
    ft = {k: G[k]['ttf']['frames_to'].get('30') for k in keys}
    f30 = lambda k, v: 'not reached' if v is None else '{:.1f}k'.format(max(v, G[k]['ttf']['frames'][0]) / 1e3)
    ftt = ', '.join(f"{lab[k]} {f30(k, v)}" for k, v in ft.items())
    cap = [f"Fig 2 (\\label{{fig:simulation}}), simulation at {a.density:g} emitter/µm²/frame (the preprint's density), 60 000 frames, "
           f"{W} x {W} px of {C.PX_NM} nm ({W * C.PX_NM / 1000:.2f} µm field). Simulated with the in-situ vectorial PSF of the 1809/2 movie and that "
           f"movie's sCMOS camera maps (gain, offset, per-pixel read noise); emitter heights 0-400 nm above the coverslip (median {zd['p50'] * 1e3:.0f} nm); "
           f"experimental photons and background of 1809/2 (median {g0['brightness_fit']['fit_median']:.0f} detected photons per 6-frame window, "
           f"background {g0['bg_per_frame']['mean']:.2f} photons/px/frame); mean on-time {g0['avg_on_time']:g} frames; pattern 2 angles x 3 phase steps, "
           f"pitch {C.PITCH_NM:g} nm, modulation depth {C.DEPTH:g}.",
           "(A) Ground truth: all labelling sites of the randomly generated microtubules (orange) over the illumination pattern of the first frame "
           "(angle 1, phase step 1; grey); white box = the zoom of B, C, F-H (the preprint's zoom, "
           f"{w * C.PX_NM / 1000:.2f} µm).",
           "(B, C, F, G, H) Renders of the zoom by (B) SMLM, (C) SIMFLUX, (F) DECODE, (G) SIMCODE as in the preprint and (H) SIMCODE v2; same rendering "
           "for all (5.4 nm pixels, 1-pixel Gaussian blur, 99.5 % clip). Rendered = each method's comparison set: one localization per emitter and "
           "disjoint 6-frame window, estimated precision < 0.15 px; SIMCODE v2 additionally p >= 0.63 and calibrated max(σx, σy) <= 12 nm. "
           "SIMFLUX and the preprint SIMCODE use the pattern estimated from the data, as SIMCODE v2 does (true-pattern results: SI table). "
           "SMLM and DECODE get 6-frame sums; DECODE was retrained for this PSF, the emitter heights and these photon / background / density ranges. "
           "GT-FRC in the titles = resolution from the FRC between the rendered reconstruction and the rendered ground truth (5 nm pixels, 1/7 threshold, "
           "full field, 60 000 frames).",
           psfpanel.caption_text(info, what='(D)').replace('(D):', '(D)', 1),
           f"(E) Number of localizations in the comparison set: {', '.join(f'{lab[k]} {cnt[k][0]}' for k in keys)}; hatched = false "
           f"positives (no emitter ON in the localization's window within 3 σ, σ = the localization's own precision): "
           f"{', '.join(f'{lab[k]} {100 * cnt[k][1]:.1f} %' for k in keys)}.",
           "(I) FRC curves of each method's reconstruction against the ground truth (60 000 frames); dashed line = 1/7; resolution in the legend; curves smoothed with a 5-sample moving average (as for the resolution values), the first two frequency samples (edge of the smoothing) not drawn. "
           "Replaces the preprint's half-split FRC curves (half-split FRC of the same sets: SI table).",
           f"(J) Time to fidelity: GT-FRC resolution on the first N frames (N = 250 ... 60 000); dotted line = 30 nm. Frames to 30 nm: {ftt}. "
           "Replaces the preprint's Jaccard index vs frames; the per-density summary (0.5 / 1 / 2 / 4 emitters/µm²/frame: localizations, GT-FRC, "
           "frames to 30 nm, half-split FRC, 10 nm Jaccard, false-positive fraction, with and without the true pattern) is the SI table "
           "out/papersim/figures/fig_sim_Dexp_v2psf_table.{md,csv} (fig_sim_main.py)."]
    open(out + '_caption.txt', 'w').write('\n'.join(cap) + '\n')
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')
    print(out + '_caption.txt')


if __name__ == '__main__':
    main()
