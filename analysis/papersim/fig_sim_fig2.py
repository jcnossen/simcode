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
  - the per-density summary is the SI table of fig_sim_main.py (out/papersim/figures/fig_sim_<prefix><suffix>_table.{md,csv})
2026-11-15 (the *_wsm sets, reviewer request): A = the ground truth on a plain black background (no illumination pattern); a
ground-truth zoom (B: all labelling sites, rendered exactly as the method zooms) next to the method zooms; the PSF panel moved
to the bottom row. The zoom box = the movie's gen.json 'zoom_crop' (webSMLM GT: picked at a filament crossing) or the preprint's
crop. Layout (--internal 1, *_internal): A GT | B GT zoom | C SMLM | D SIMFLUX / E counts | F DECODE | G SIMCODE (preprint) | H SIMCODE v2 /
I PSF | J GT-FRC curves | K GT-FRC vs frames.
2026-11-15 (paper version, default): no preprint column and no version labels (the method is 'SIMCODE'); the freed tile puts the
PSF panel back in the top-right slot and the bottom row is the preprint's two curve panels: A GT | B GT zoom | C SMLM | D PSF /
E counts | F SIMFLUX | G DECODE | H SIMCODE / I GT-FRC curves | J GT-FRC vs frames.
2026-11-15 (b): every GT-FRC value (titles, curve legend, time to fidelity, frames to 30 nm) under the shared crossing rule
gtfrc.RULE (sustained 1/7 crossing; the hollow-tube spectral null), stated in the caption; all numbers of the figure in
_values.json; file paths / run names only in _notes_internal.txt (not in the caption).
usage: python analysis/papersim/fig_sim_fig2.py [--density 1] [--prefix Dexp] [--suffix _v2psf] [--model m1809v2] [--internal 0|1]
output: out/paper_figs_v2/fig_sim_<prefix><suffix>.{pdf,svg,png}, _caption.txt, _values.json, _notes_internal.txt"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M, gtfrc as GR
from fig_density import render, crop_of, gt_desc, COL
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# render order as in the preprint (SMLM, SIMFLUX, DECODE, SIMCODE). Paper version: SIMCODE = the current method only (no preprint
# column, no version labels); --internal 1: + the preprint SIMCODE column, labels 'SIMCODE (preprint)' / 'SIMCODE v2' (*_internal)
METH_PAPER = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2sig', 'SIMCODE')]
METH_INTERNAL = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v1', 'SIMCODE (preprint)'), ('v2sig', 'SIMCODE v2')]
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
    ap.add_argument('--internal', type=int, default=0, help='1: internal comparison version with the preprint SIMCODE column '
                    '(labels SIMCODE (preprint) / SIMCODE v2) -> fig_sim_<prefix><suffix>_internal.*')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2')); a = ap.parse_args()
    run = os.path.join(C.OUT, f'{a.prefix}{a.density:g}{a.suffix}')
    R = json.load(open(os.path.join(run, 'full', f'metrics_{a.model}.json')))
    G = json.load(open(os.path.join(run, 'full', f'gtfrc_{a.model}.json')))['methods']
    gt = np.load(os.path.join(run, 'gt.npz')); W = int(R['W']); nfr = int(R['nframes'])
    (cx, cy), w = crop_of(run)
    METH = METH_INTERNAL if a.internal else METH_PAPER
    lab = dict(METH); SC = lab['v2sig']
    # panel letters. Paper: A GT | B GT zoom | C SMLM | D PSF / E counts | F SIMFLUX | G DECODE | H SIMCODE / I GT-FRC | J vs frames
    # internal:            A GT | B GT zoom | C SMLM | D SIMFLUX / E counts | F DECODE | G preprint | H v2 / I PSF | J | K
    if a.internal:
        slots = {'smlm': (0, 2, 'C'), 'sf': (0, 3, 'D'), 'decode': (1, 1, 'F'), 'v1': (1, 2, 'G'), 'v2sig': (1, 3, 'H')}
        LP, LC, LT = 'I', 'J', 'K'
    else:
        slots = {'smlm': (0, 2, 'C'), 'sf': (1, 1, 'F'), 'decode': (1, 2, 'G'), 'v2sig': (1, 3, 'H')}
        LP, LC, LT = 'D', 'I', 'J'

    fig = plt.figure(figsize=(13.0, 10.6))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2, 1.0], hspace=0.30, left=0.055, right=0.985, top=0.935, bottom=0.065)
    gs = outer[0].subgridspec(2, 4, hspace=0.30, wspace=0.12)

    # A: ground truth (all labelling sites) on black, zoom box. Field = pixel edges -0.5 .. W - 0.5 (pixel centre = integer)
    A = fig.add_subplot(gs[0, 0]); A.set_facecolor('k')
    p = gt['pts']; A.plot(p[:, 0], p[:, 1], ',', color='#ffb070', alpha=0.9, ms=0.5, rasterized=True)
    A.add_patch(Rectangle((cx, cy), w, w, fill=False, ec='w', lw=2.5))
    A.set_xlim(-0.5, W - 0.5); A.set_ylim(W - 0.5, -0.5); A.set_xticks([]); A.set_yticks([])
    L = 2000 / C.PX_NM; A.plot([0.05 * W, 0.05 * W + L], [0.93 * W] * 2, 'w-', lw=4, solid_capstyle='butt')
    A.text(0.05 * W, 0.90 * W, '2000 nm', color='w', fontsize=10, va='bottom')
    A.set_title('Ground truth\n(all labelling sites)', fontsize=FS); letter(A, 'A')

    # B: the ground-truth zoom, rendered as the method zooms (same box, renderer, colormap)
    A = fig.add_subplot(gs[0, 1]); A.set_xticks([]); A.set_yticks([])
    A.imshow(render(p[:, 0], p[:, 1], (cx, cy), w), cmap='inferno')
    A.set_title('Ground truth (zoom)', fontsize=FS); letter(A, 'B'); scalebar(A, w, 500)

    # method renders of the zoom
    cnt = {}
    for k, lab_k in METH:
        r, c, L = slots[k]; A = fig.add_subplot(gs[r, c]); A.set_xticks([]); A.set_yticks([])
        T = M.load_method(k, R['methods'][k]['path']); _, _, f = M.frc_set(T, W, nfr)
        A.imshow(render(T['x'][f], T['y'][f], (cx, cy), w), cmap='inferno')
        A.set_title(f"{lab_k}\nGT-FRC = {GR.res(G[k]['gt_frc']):.1f} nm", fontsize=FS); letter(A, L)
        scalebar(A, w, 500)
        assert G[k]['n'] == R['methods'][k]['n_frc'], (k, G[k]['n'], R['methods'][k]['n_frc'])
        cnt[k] = (G[k]['n'], G[k]['ghost']['ghost_fraction'])

    # PSF panel: xy | xz, two-line label. Paper: the top-right slot (as the first Fig 2 rework); internal: bottom row, left
    import psfpanel
    info = psfpanel.sim_psf_info(run, a.prefix == 'Dexp')
    if a.internal:
        gb = outer[1].subgridspec(1, 3, wspace=0.26, width_ratios=[0.85, 1.25, 1.25])
        gp = gb[0].subgridspec(2, 2, height_ratios=[1.0, 0.30], wspace=0.75, hspace=0.05)
        gc, gt_ = gb[1], gb[2]; ldx = -0.12
    else:
        gp = gs[0, 3].subgridspec(2, 2, height_ratios=[1.0, 0.30], wspace=0.75, hspace=0.05)
        gb = outer[1].subgridspec(1, 2, wspace=0.22); gc, gt_ = gb[0], gb[1]; ldx = -0.09
    axy, axz, axt = fig.add_subplot(gp[0, 0]), fig.add_subplot(gp[0, 1]), fig.add_subplot(gp[1, :])
    psfpanel.draw(axy, axz, info, fontsize=8); axz.set_box_aspect(1); letter(axy, LP, dx=-0.12, dy=1.12)
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

    # GT-FRC curves; GT-FRC resolution vs frames (time to fidelity)
    A = fig.add_subplot(gc); B = fig.add_subplot(gt_)
    for k in keys:
        g = G[k]; lw = 2.4 if k == 'v2sig' else 1.5
        A.plot(g['gt_frc']['freq'][2:], g['gt_frc']['curve'][2:], color=COL[k], lw=lw, label=f"{lab[k]}  {GR.res(g['gt_frc']):.1f} nm")
        t = GR.ttf(g['ttf']); B.plot(np.array(t['frames']) / 1e3, t['res_nm'], '-o', color=COL[k], lw=lw, ms=4 if k != 'v2sig' else 5, label=lab[k])
    A.axhline(1 / 7, color='grey', ls='--', lw=1); A.set_xlim(0, 0.09); A.set_ylim(-0.03, 1.0)
    A.set_xlabel('Spatial freq. [nm$^{-1}$]', fontsize=11); A.set_ylabel('GT-FRC', fontsize=11); A.tick_params(labelsize=10)
    A.set_title('FRC against the ground truth', fontsize=FS); A.legend(fontsize=9, frameon=True, loc='upper right'); letter(A, LC, dx=ldx)
    B.set_xscale('log'); B.set_yscale('log'); B.axhline(30, color='0.5', lw=1, ls=':')
    B.text(60, 30.8, '30 nm', color='0.4', fontsize=9, ha='right', va='bottom')
    B.set_xticks([0.25, 1, 4, 15, 60]); B.set_xticklabels(['0.25', '1', '4', '15', '60']); B.set_ylim(12, 150)
    B.set_yticks([15, 20, 30, 50, 100]); B.set_yticklabels(['15', '20', '30', '50', '100']); B.minorticks_off(); B.tick_params(labelsize=10)
    B.set_xlabel('frames (x 1000)', fontsize=11); B.set_ylabel('GT-FRC resolution (nm)', fontsize=11)
    B.set_title('GT-FRC vs frames (time to fidelity)', fontsize=FS); letter(B, LT, dx=ldx)

    os.makedirs(a.out, exist_ok=True); out = os.path.join(a.out, f'fig_sim_{a.prefix}{a.suffix}' + ('_internal' if a.internal else ''))
    # caption material (paper version: no reference to the preprint / earlier versions)
    g0 = json.load(open(os.path.join(run, 'gen.json'))); zd = g0['z_dist_um']
    ft = {k: GR.ttf(G[k]['ttf'])['frames_to'].get('30') for k in keys}
    f30 = lambda k, v: 'not reached' if v is None else '{:.1f}k'.format(max(v, G[k]['ttf']['frames'][0]) / 1e3)
    ftt = ', '.join(f"{lab[k]} {f30(k, v)}" for k, v in ft.items())
    I = a.internal; zl = 'B-D, F-H' if I else 'B, C, F-H'; rl = 'C, D, F, G, H' if I else 'C, F, G, H'
    who = ("(C) SMLM, (D) SIMFLUX, (F) DECODE, (G) SIMCODE as in the preprint and (H) SIMCODE v2" if I else
           "(C) SMLM, (F) SIMFLUX, (G) DECODE and (H) SIMCODE")
    cap = [f"Fig 2 (\\label{{fig:simulation}}), simulation at {a.density:g} emitter/µm²/frame" + (" (the preprint's density)" if I else '') + ", 60 000 frames, "
           f"{W} x {W} px of {C.PX_NM} nm ({W * C.PX_NM / 1000:.2f} µm field). Simulated with the in-situ vectorial PSF of the 1809/2 movie and that "
           f"movie's sCMOS camera maps (gain, offset, per-pixel read noise); emitter heights 0-400 nm above the coverslip (median {zd['p50'] * 1e3:.0f} nm); "
           f"experimental photons and background of 1809/2 (median {g0['brightness_fit']['fit_median']:.0f} detected photons per 6-frame window, "
           f"background {g0['bg_per_frame']['mean']:.2f} photons/px/frame); mean on-time {g0['avg_on_time']:g} frames; pattern 2 angles x 3 phase steps, "
           f"pitch {C.PITCH_NM:g} nm, modulation depth {C.DEPTH:g}.",
           f"(A) Ground truth: all labelling sites of the {gt_desc(run, paper=not I)} (orange); white box = the zoom of {zl} ("
           + (f"{w * C.PX_NM / 1000:.2f} µm, centred on filament crossings" if g0.get('zoom_crop') else
              (f"the preprint's zoom, {w * C.PX_NM / 1000:.2f} µm" if I else f"{w * C.PX_NM / 1000:.2f} µm")) + ")"
           + (". Ground truth from webSMLM (microtubules (cells) sample, which embeds the inSiliScope world model), webSMLM defaults except the label "
              "efficiency (0.5 labels per nm of filament" + (" as in the preprint ground truth" if I else '') + "); the labels sit on the 13-protofilament "
              "lattice with a 12 nm binder and 2-5 nm linker (~25-30 nm from the tube axis), their heights are drawn from the experimental height "
              "distribution as for every emitter (see above)." if g0.get('wsm') else "."),
           "(B) The ground truth of the zoom (all labelling sites), rendered exactly as the reconstructions. "
           f"({rl}) Renders of the zoom by {who}; same rendering "
           f"for {zl} (5.4 nm pixels, 1-pixel Gaussian blur, 99.5 % clip). Rendered = each method's comparison set: one localization per emitter and "
           f"disjoint 6-frame window, estimated precision < 0.15 px; {SC} additionally p >= 0.63 and calibrated max(σx, σy) <= 12 nm. "
           + ("SIMFLUX and the preprint SIMCODE use the pattern estimated from the data, as SIMCODE v2 does (true-pattern results: SI table). " if I else
              "SIMFLUX uses the pattern estimated from the data, as SIMCODE does (true-pattern results: SI table). ") +
           "SMLM and DECODE get 6-frame sums; DECODE was retrained for this PSF, the emitter heights and these photon / background / density ranges. "
           "GT-FRC in the titles = resolution from the FRC between the rendered reconstruction and the rendered ground truth (5 nm pixels, 1/7 threshold, "
           "full field, 60 000 frames). " + GR.RULE_CAPTION,
           psfpanel.caption_text(info, what=f'({LP})').replace(f'({LP}):', f'({LP})', 1),
           f"(E) Number of localizations in the comparison set: {', '.join(f'{lab[k]} {cnt[k][0]}' for k in keys)}; hatched = false "
           f"positives (no emitter ON in the localization's window within 3 σ, σ = the localization's own precision): "
           f"{', '.join(f'{lab[k]} {100 * cnt[k][1]:.1f} %' for k in keys)}.",
           f"({LC}) FRC curves of each method's reconstruction against the ground truth (60 000 frames); dashed line = 1/7; resolution in the legend; curves smoothed with a 5-sample moving average (as for the resolution values), the first two frequency samples (edge of the smoothing) not drawn. "
           + ("Replaces the preprint's half-split FRC curves (half-split FRC of the same sets: SI table)." if I else "Half-split FRC of the same sets: SI table."),
           f"({LT}) Time to fidelity: GT-FRC resolution (same crossing rule) on the first N frames (N = 250 ... 60 000); dotted line = 30 nm. "
           f"Frames to 30 nm (log-interpolated between the N): {ftt}. "
           + ("Replaces the preprint's Jaccard index vs frames; the" if I else "The") +
           " per-density summary (0.5 / 1 / 2 / 4 emitters/µm²/frame: localizations, GT-FRC, "
           "frames to 30 nm, half-split FRC, 10 nm Jaccard, false-positive fraction, with and without the true pattern) is in the "
           "Supplementary Information."]
    open(out + '_caption.txt', 'w').write('\n'.join(cap) + '\n')
    # internal notes (paths, run / model names: never in the caption)
    open(out + '_notes_internal.txt', 'w').write('\n'.join([
        f"generator: python analysis/papersim/fig_sim_fig2.py --density {a.density:g} --prefix {a.prefix} --suffix {a.suffix} --model {a.model} --internal {a.internal}",
        f"inputs: {os.path.join(run, 'full', f'metrics_{a.model}.json')}, {os.path.join(run, 'full', f'gtfrc_{a.model}.json')}",
        f"SIMCODE rows: {R['methods']['v2sig']['path']} (method key v2sig, networks {a.model})",
        f"crossing rule: gtfrc.RULE = {GR.RULE!r} (gtfrc.frc_sustained, width {GR.SUSTAIN_WIDTH} / nm); first-crossing values in _values.json",
        f"per-density SI table: out/papersim/figures/fig_sim_{a.prefix}{a.suffix}_table.{{md,csv}} (fig_sim_main.py); SI overview: "
        f"out/papersim/overview_{a.prefix}{a.suffix}_{a.model}.* (fig_overview.py)",
        f"values: {out}_values.json"]) + '\n')
    # every number of the figure (and of its caption)
    gj = lambda k: G[k]['gt_frc']; best = lambda ks: min(ks, key=lambda k: GR.res(gj(k)))
    others = [k for k in keys if k != 'v2sig']; nb = best(others)
    V = dict(figure=os.path.basename(out), density=a.density, nframes=nfr, field_px=W, px_nm=C.PX_NM, field_um=W * C.PX_NM / 1000,
             crossing_rule=GR.RULE, crossing_rule_text=GR.RULE_CAPTION, threshold=GR.THRESHOLD, sustain_width_per_nm=GR.SUSTAIN_WIDTH,
             zoom_box_px=dict(x0=cx, y0=cy, w=w), zoom_um=w * C.PX_NM / 1000, scalebars_nm=dict(A=2000, zooms=500),
             conditions=dict(median_photons_per_window=g0['brightness_fit']['fit_median'] if 'brightness_fit' in g0 else g0.get('intensity_per_6frames'),
                             bg_photons_px_frame=g0['bg_per_frame']['mean'] if 'bg_per_frame' in g0 else g0.get('bg'),
                             z_median_nm=zd['p50'] * 1e3, avg_on_time_frames=g0['avg_on_time'], pitch_nm=C.PITCH_NM, depth=C.DEPTH),
             psf_panel_label=psfpanel.panel_label(info), psf_caption=psfpanel.caption_text(info, what=f'({LP})'),
             panels=dict(zooms={lab[k]: slots[k][2] for k in keys}, psf=LP, counts='E', gtfrc_curves=LC, time_to_fidelity=LT),
             methods={}, next_best=lab[nb],
             simcode_gain_over_next_best=(GR.res(gj(nb)) - GR.res(gj('v2sig'))) / GR.res(gj(nb)),
             simcode_gain_over={lab[k]: (GR.res(gj(k)) - GR.res(gj('v2sig'))) / GR.res(gj(k)) for k in others})
    for k in keys:
        g = G[k]; t = GR.ttf(g['ttf']); n, gf = cnt[k]; f3 = t['frames_to'].get('30')
        V['methods'][lab[k]] = dict(
            key=k, n_comparison_set=n, false_positive_fraction=gf, n_false_positive=int(round(n * gf)), n_on_within_3sigma=n - int(round(n * gf)),
            gt_frc_nm=GR.res(g['gt_frc']), gt_frc_first_nm=GR.resolution(g['gt_frc']['freq'], g['gt_frc']['curve'], 'first'),
            gt_frc_sustained_nm=GR.resolution(g['gt_frc']['freq'], g['gt_frc']['curve'], 'sustained'),
            frames_to_30nm=f3, frames_to_30nm_plotted=None if f3 is None else max(f3, t['frames'][0]),
            frames_to=t['frames_to'], frames_to_first=t.get('frames_to_first'), frames_to_sustained=t.get('frames_to_sustained'),
            ttf_frames=t['frames'], ttf_res_nm=t['res_nm'],
            jaccard_10nm=R['methods'][k].get('jaccard_end'), sigma_jaccard=g['sigma_jaccard']['jaccard'],
            fair_frc_nm=GR.fair_frc_nm(R['methods'][k]), fair_frc_first_nm=R['methods'][k].get('frc_nm'),
            gt_frc_curve=dict(freq_per_nm=g['gt_frc']['freq'][2:], frc=g['gt_frc']['curve'][2:]))
    s30 = V['methods'][SC]['frames_to_30nm']
    V['frames_to_30nm_ratio_vs_simcode'] = {lab[k]: (None if (V['methods'][lab[k]]['frames_to_30nm'] is None or not s30) else
                                                     V['methods'][lab[k]]['frames_to_30nm'] / s30) for k in others}
    json.dump(V, open(out + '_values.json', 'w'), indent=1)
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')
    print(out + '_caption.txt'); print(out + '_values.json'); print(out + '_notes_internal.txt')


if __name__ == '__main__':
    main()
