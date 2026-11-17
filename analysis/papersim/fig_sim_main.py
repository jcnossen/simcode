"""Main-text simulation figure (agreed with the paper session, 2026-11-15).

A  renders: rows = densities 0.5 / 1 / 2 / 4, columns = ground truth, SMLM, SIMFLUX, DECODE, SIMCODE (paper version; --internal 1:
   + SIMCODE (preprint), labels SIMCODE v2, *_internal);
   SIMFLUX and the preprint SIMCODE with the ESTIMATED pattern (as SIMCODE v2, which estimates it in situ); same crop and
   rendering as fig_overview.py; each tile gives only the GT-FRC and the number of localizations (the FRC set).
B  PSF panel, xy | xz (psfpanel.py), short label; conditions / CRLBs go to the caption.
C  GT-FRC (60k frames) vs density, one line per method.
D  frames to 30 nm GT-FRC vs density (the time-to-fidelity data of gtfrc_run.py), one line per method.
SI table (next to the figure, .md + .csv): per density and method, incl. the true-pattern rows: n, GT-FRC, frames to 30 nm,
fair FRC, 10 nm Jaccard, false-positive fraction.
All FRC values (GT-FRC, frames to 30 nm, fair FRC) under the shared crossing rule gtfrc.RULE (sustained 1/7 crossing),
stated in the caption and the table legend; run / model names and paths only in _notes_internal.txt.
usage: python analysis/papersim/fig_sim_main.py [--prefix Dexp] [--suffix _v2psf] [--model m1809v2]
output: out/papersim/figures/fig_sim_<prefix><suffix>.{pdf,svg,png}, _caption.txt, _table.md, _table.csv"""
import os, sys, json, csv, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M, gtfrc as GR
from fig_density import render, crop_of, gt_desc, COL
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DENS = [0.5, 1, 2, 4]
# paper version (default): no preprint SIMCODE columns / rows, method label 'SIMCODE'; --internal 1 (*_internal): with them
METH_INTERNAL = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v1', 'SIMCODE\n(preprint)'), ('v2sig', 'SIMCODE v2')]
TABLE_INTERNAL = [('smlm', 'SMLM'), ('sf', 'SIMFLUX (estimated pattern)'), ('sf_gtp', 'SIMFLUX (true pattern)'), ('decode', 'DECODE (retrained)'),
                  ('v1', 'SIMCODE preprint (estimated pattern)'), ('v1gt', 'SIMCODE preprint (true pattern)'), ('v2sig', 'SIMCODE v2')]
METH_PAPER = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2sig', 'SIMCODE')]
TABLE_PAPER = [('smlm', 'SMLM'), ('sf', 'SIMFLUX (estimated pattern)'), ('sf_gtp', 'SIMFLUX (true pattern)'), ('decode', 'DECODE (retrained)'),
               ('v2sig', 'SIMCODE')]


def letter(ax, s, dx=-0.08, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='Dexp'); ap.add_argument('--suffix', default='_v2psf')
    ap.add_argument('--model', default='m1809v2')
    ap.add_argument('--internal', type=int, default=0, help='1: with the preprint SIMCODE (columns, table rows) -> fig_sim_<..>_internal.*')
    ap.add_argument('--v2key', default='v2sig', help="SIMCODE row: v2sig (calibrated sigma <= 12 nm) or v2c (2026-11-15: the common rule of every "
                    "method, sigma < 0.15 px; outputs tagged _common)")
    a = ap.parse_args()
    METH, TABLE = (METH_INTERNAL, TABLE_INTERNAL) if a.internal else (METH_PAPER, TABLE_PAPER)
    METH = [(a.v2key if k == 'v2sig' else k, l) for k, l in METH]; TABLE = [(a.v2key if k == 'v2sig' else k, l) for k, l in TABLE]
    SC = dict(METH)[a.v2key]
    runs = [os.path.join(C.OUT, f'{a.prefix}{d:g}{a.suffix}') for d in DENS]
    R = [json.load(open(os.path.join(r, 'full', f'metrics_{a.model}.json'))) for r in runs]
    G = [json.load(open(os.path.join(r, 'full', f'gtfrc_{a.model}.json')))['methods'] for r in runs]
    if a.v2key == 'v2c':      # common rule (2026-11-15): GT-FRC and time to fidelity of common_rule.py (sustained 1/7 crossing, as Fig 2)
        import common_rule as CR
        for i, r in enumerate(runs):
            CRm = json.load(open(os.path.join(CR.OUTD, os.path.basename(r) + '.json')))['methods']
            for k, ck in (('smlm', 'smlm'), ('sf', 'sf'), ('sf_gtp', 'sf_gtp'), ('decode', 'decode'), ('v1', 'v1'), ('v2c', 'v2')):
                if k in G[i] and ck in CRm:
                    assert G[i][k]['n'] == CRm[ck]['n'], (r, k, G[i][k]['n'], CRm[ck]['n'])
                    rr = dict(first=CRm[ck]['gt_first'], sustained=CRm[ck]['gt_sust'])
                    G[i][k]['gt_frc'] = dict(G[i][k]['gt_frc'], res_nm=rr[GR.RULE], res_nm_first=rr['first'], res_nm_sustained=rr['sustained'], rule=GR.RULE)
                    G[i][k]['ttf'] = dict(CRm[ck]['ttf'], rule=CRm[ck]['ttf'].get('rule', 'sustained'))       # common_rule.ttf: sustained
                    G[i][k]['ghost'] = dict(ghost_fraction=CRm[ck]['fp'])
                    hs = dict(first=CRm[ck].get('hs_first'), sustained=CRm[ck].get('hs_sust'))
                    if hs[GR.RULE] is not None: G[i][k]['fair_frc_nm'] = hs[GR.RULE]      # half-split FRC of the common-rule rows
    (cx, cy), w = crop_of(runs[0])
    nc = len(METH) + 1; tw = 2.3
    fig = plt.figure(figsize=(nc * tw + 5.6, len(DENS) * tw + 1.0))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[nc * tw, 4.3], wspace=0.13, left=0.03, right=0.985, top=0.95, bottom=0.07)
    gimg = outer[0].subgridspec(len(DENS), nc, wspace=0.03, hspace=0.03)
    for i, d in enumerate(DENS):
        gt = np.load(os.path.join(runs[i], 'gt.npz'))
        A = fig.add_subplot(gimg[i, 0]); A.imshow(render(gt['pts'][:, 0], gt['pts'][:, 1], (cx, cy), w), cmap='inferno')
        A.set_ylabel(f'density {d:g}', fontsize=12); A.set_xticks([]); A.set_yticks([])
        if i == 0:
            A.set_title('Ground truth', fontsize=12); letter(A, 'A', dx=-0.12)
        if i == len(DENS) - 1:
            A.plot([8, 8 + 500 / (C.PX_NM / 20)], [w * 20 - 10] * 2, 'w-', lw=3); A.text(8, w * 20 - 18, '500 nm', color='w', fontsize=9)
        for j, (k, lab) in enumerate(METH):
            A = fig.add_subplot(gimg[i, j + 1]); A.set_xticks([]); A.set_yticks([])
            if i == 0: A.set_title(lab, fontsize=12)
            if k not in R[i]['methods']:
                A.set_facecolor('k'); A.text(0.5, 0.5, 'no result', color='w', ha='center', transform=A.transAxes); continue
            r = R[i]['methods'][k]; T = M.load_method(k, r['path']); _, _, f = M.frc_set(T, R[i]['W'], R[i]['nframes'])
            A.imshow(render(T['x'][f], T['y'][f], (cx, cy), w), cmap='inferno')
            gl = f"GT-FRC {GR.res(G[i][k]['gt_frc']):.1f} nm\n" if k in G[i] else ''
            A.text(0.03, 0.03, f"{gl}{r['n_frc'] / 1e3:.0f}k localizations", transform=A.transAxes, color='w', fontsize=9, va='bottom',
                   bbox=dict(facecolor='k', alpha=0.55, pad=1.5, lw=0))
    # right column: B PSF, C GT-FRC vs density, D frames to 30 nm vs density
    gr = outer[1].subgridspec(3, 1, height_ratios=[1.0, 1.2, 1.2], hspace=0.62)
    import psfpanel
    info = psfpanel.sim_psf_info(runs[0], a.prefix == 'Dexp')
    gp = gr[0].subgridspec(2, 2, height_ratios=[1.0, 0.42], wspace=0.6, hspace=0.12)
    axy, axz, axt = fig.add_subplot(gp[0, 0]), fig.add_subplot(gp[0, 1]), fig.add_subplot(gp[1, :])
    psfpanel.draw(axy, axz, info, fontsize=8); axz.set_box_aspect(1); letter(axy, 'B', dx=-0.1, dy=1.12)
    axt.axis('off'); axt.text(0.0, 0.95, psfpanel.panel_label(info), transform=axt.transAxes, fontsize=10, va='top', ha='left', linespacing=1.2)
    keys = [k for k, _ in METH]; labs = dict(METH)
    A = fig.add_subplot(gr[1]); B = fig.add_subplot(gr[2])
    fmax = R[0]['nframes']
    for k in keys:
        lw, ms = (2.4, 6) if k == a.v2key else (1.4, 4.5)
        y = [GR.res(G[i][k]['gt_frc']) if k in G[i] else np.nan for i in range(len(DENS))]
        A.plot(DENS, y, '-o', color=COL['v2sig' if k == a.v2key else k], lw=lw, ms=ms, label=labs[k].replace('\n', ' '))
        t = [GR.ttf(G[i][k]['ttf'])['frames_to'].get('30') if k in G[i] else None for i in range(len(DENS))]
        yt = np.array([np.nan if v is None else max(v, G[i][k]['ttf']['frames'][0]) for i, v in enumerate(t)], float)
        B.plot(DENS, yt / 1e3, '-o', color=COL['v2sig' if k == a.v2key else k], lw=lw, ms=ms)
        nr = [i for i, v in enumerate(t) if v is None and k in G[i]]       # not reached in the movie: open marker above the end
        if nr: B.plot([DENS[i] for i in nr], [fmax / 1e3 * 1.35] * len(nr), 'o', mfc='none', color=COL['v2sig' if k == a.v2key else k], ms=ms + 1, mew=1.5)
    for X in (A, B):
        X.set_xscale('log', base=2); X.set_xticks(DENS); X.set_xticklabels([f'{d:g}' for d in DENS]); X.minorticks_off()
        X.set_xlabel('emitter density (µm$^{-2}$ frame$^{-1}$)', fontsize=10); X.tick_params(labelsize=9)
    A.set_ylabel('GT-FRC (nm)', fontsize=10); A.set_title('GT-FRC vs density (60 000 frames)', fontsize=11)
    A.set_yscale('log'); A.set_ylim(10, 140); A.set_yticks([10, 15, 20, 30, 50, 100]); A.set_yticklabels(['10', '15', '20', '30', '50', '100']); A.minorticks_off()
    A.legend(fontsize=8, frameon=False, ncol=2, loc='upper left'); letter(A, 'C', dx=-0.16)
    B.set_yscale('log'); B.axhline(fmax / 1e3, color='0.5', lw=0.8, ls=':'); B.set_ylim(0.15, fmax / 1e3 * 1.8)
    B.set_yticks([0.25, 1, 4, 15, 60]); B.set_yticklabels(['0.25', '1', '4', '15', '60']); B.minorticks_off()
    B.set_ylabel('frames (x1000)', fontsize=10); B.set_title('Frames to 30 nm GT-FRC vs density', fontsize=11); letter(B, 'D', dx=-0.16)
    out = os.path.join(C.OUT, 'figures', f'fig_sim_{a.prefix}{a.suffix}' + ('_common' if a.v2key == 'v2c' else '') + ('_internal' if a.internal else '')); os.makedirs(os.path.dirname(out), exist_ok=True)
    # SI table
    rows = []
    for i, d in enumerate(DENS):
        for k, lab in TABLE:
            if k not in R[i]['methods']: continue
            m = R[i]['methods'][k]; g = G[i].get(k, {})
            ft = GR.ttf(g['ttf'])['frames_to'].get('30') if g else None
            rows.append(dict(density=d, method=lab, n_frc=m['n_frc'], gt_frc_nm=GR.res(g['gt_frc']) if g else None,
                             frames_to_30nm=None if ft is None else round(max(ft, g['ttf']['frames'][0])),
                             fair_frc_nm=g['fair_frc_nm'] if 'fair_frc_nm' in g else GR.fair_frc_nm(m),
                             jaccard_10nm=m.get('jaccard_end'), false_positive_fraction=g['ghost']['ghost_fraction'] if g else None))
    with open(out + '_table.csv', 'w', newline='') as fh:
        W = csv.DictWriter(fh, fieldnames=list(rows[0])); W.writeheader(); W.writerows(rows)
    f1 = lambda v, s='{:.1f}': '-' if v is None else s.format(v)
    g0 = json.load(open(os.path.join(runs[0], 'gen.json')))
    cond = (f"experimental photons and background of 1809/2 (median {g0['brightness_fit']['fit_median']:.0f} detected photons per 6-frame window, "
            f"{g0['bg_per_frame']['mean']:.2f} photons/px/frame)" if a.prefix == 'Dexp' else
            (f"preprint condition ({g0['intensity_per_6frames']:.0f} photons per 6-frame window, background {g0['bg']:g} photons/px/frame)" if a.internal else
             f"{g0['intensity_per_6frames']:.0f} detected photons per 6-frame window, background {g0['bg']:g} photons/px/frame"))
    md = [f"# Simulation table ({cond}; 60 000 frames)" if not a.internal else f'# Simulation table ({a.prefix}{a.suffix}, {SC} = {a.model}; 60 000 frames)', '',
          'n = localizations in the FRC set (disjoint 6-frame windows, σ < 0.15 px' + ('; SIMCODE p >= 0.63, network σ' if a.v2key == 'v2c' else '') + '); GT-FRC = FRC(reconstruction, ground truth), 1/7; '
          'frames to 30 nm GT-FRC ("-" = not reached in 60 000 frames, values <= 250 = reached in the first 250 frames); fair FRC = half-split FRC; '
          'Jaccard at 10 nm; false positive = a localization with no emitter ON in its window within 3 own σ. '
          'GT-FRC, frames to 30 nm and fair FRC: ' + GR.RULE_CAPTION, '',
          '| density | method | n | GT-FRC (nm) | frames to 30 nm | fair FRC (nm) | Jaccard 10 nm | false positive fraction |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['density']:g} | {r['method']} | {r['n_frc']} | {f1(r['gt_frc_nm'])} | {f1(r['frames_to_30nm'], '{:.0f}')} | "
                  f"{f1(r['fair_frc_nm'])} | {f1(r['jaccard_10nm'], '{:.3f}')} | {f1(r['false_positive_fraction'], '{:.3f}')} |")
    open(out + '_table.md', 'w').write('\n'.join(md) + '\n')
    # caption
    zd = g0.get('z_dist_um')
    hts = (f"emitter heights = the labels' own 3D positions on the filaments ({zd['p05'] * 1e3:.0f}-{zd['p95'] * 1e3:.0f} nm, 5th-95th percentile, "
           f"median {zd['p50'] * 1e3:.0f} nm)" if zd and str(g0.get('gt', '')).startswith('wsm3d') else
           f"emitter heights 0-400 nm (median {zd['p50'] * 1e3:.0f} nm)" if zd else '')
    psfs = (f"the in-situ PSF of 1809/2 and the sCMOS camera maps of that movie (gain, offset, per-pixel read noise), {hts}" if zd else 'a Gaussian PSF')
    cap = [f"Simulated {gt_desc(runs[0], paper=not a.internal)}, 60 000 frames, {psfs}; {cond}; the same 2 x 3-step pattern for all densities.",
           f"(A) Renders of the same {w * C.PX_NM / 1000:.2f} µm crop for every method and emitter density (emitters/µm²/frame), same rendering in every "
           "tile; each tile gives the GT-FRC (FRC between the rendered reconstruction and the rendered ground truth of all labelling sites; 5 nm pixels, 1/7) "
           "and the number of localizations (disjoint 6-frame windows, σ < 0.15 px" + ("; SIMCODE: detection probability >= 0.63 and its network σ < 0.15 px, "
           "the same precision cut as every method" if a.v2key == 'v2c' else "") + "). "
           + ("SIMFLUX and the preprint SIMCODE use the pattern estimated from the data, as SIMCODE v2 does; " if a.internal else
              "SIMFLUX uses the pattern estimated from the data, as SIMCODE does; ") + "DECODE was retrained for this PSF following its published training settings (Supplementary Note: DECODE training), with the emitter heights and these photon / background / density ranges. "
           "Results with the true pattern, the fair half-split FRC, the Jaccard index and the false-positive fraction are in the SI table. "
           + GR.RULE_CAPTION,
           psfpanel.caption_text(info, what='(B)'),
           "(C) GT-FRC on the full movie vs emitter density. (D) Frames needed to reach 30 nm GT-FRC (GT-FRC on the first N frames, log-interpolated); "
           "open markers above the dotted line: not reached within the 60 000 frames."]
    open(out + '_caption.txt', 'w').write('\n'.join(cap) + '\n')
    open(out + '_notes_internal.txt', 'w').write('\n'.join([
        f"generator: python analysis/papersim/fig_sim_main.py --prefix {a.prefix} --suffix {a.suffix} --model {a.model} --v2key {a.v2key} --internal {a.internal}",
        f"SIMCODE = method key {a.v2key}, networks {a.model}; runs: " + ', '.join(runs),
        f"crossing rule: gtfrc.RULE = {GR.RULE!r} (GT-FRC, frames to 30 nm, fair FRC = gtfrc.fair_frc_nm on the stored half-split curve; "
        "v2c: the common_rule.py values of that rule)",
        "methods missing per density (no result): " + '; '.join(f"{d:g}: " + ','.join(k for k, _ in TABLE if k not in R[i]['methods']) for i, d in enumerate(DENS))]) + '\n')
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')
    print(out + '_caption.txt'); print(out + '_table.md'); print(out + '_table.csv')


if __name__ == '__main__':
    main()
