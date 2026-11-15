"""SI figure 'si_simulated_live_smlm' (moving microtubules, SI S8) in the 2026-11-15 paper-figure style (fig_sim_main.py).

A  renders of two 1000-frame blocks (rows): one input frame of the block, the ground truth (all trajectory points of the block),
   SMLM, SIMFLUX, DECODE, SIMCODE (preprint), SIMCODE v2; every method rendered from its FRC set of the block (disjoint 6-frame
   windows, max(sigma) < 0.15 px, as fig_sim_main.py); SIMFLUX and the preprint SIMCODE with the ESTIMATED pattern. Each tile
   gives the block's FRC and its number of localizations.
B  PSF panel (psfpanel.py), two-line label; conditions / CRLBs in the caption.
C  per-block FRC over the movie (metrics.py moving: within each 1000-frame block, halves alternating every 60 frames).
D  GT-matched localization error (this script, moving_gterr): each FRC-set localization is matched to the nearest emitter that is
   ON in its 6-frame window (GT position = the photon-weighted mean of the emitter's per-frame positions in that window; the
   structure moves during the window), within 100 nm; bars = RMS error of the matched localizations (per axis, mean of x and
   y), markers = fraction of all FRC-set localizations that are unmatched or off by > 30 nm.
E  mean block Jaccard at 10 / 20 / 40 nm (the preprint's metric: GT = all trajectory points of the block; it measures how much of the
   moving structure is covered and rewards the number of localizations, see the caption).
SI table (.md + .csv): per method incl. SIMFLUX with the true pattern: rows, FRC-set rows per block, median per-block FRC,
Jaccard 10 / 20 / 40, GT-matched RMS error, fraction > 30 nm / unmatched, recall of GT emitter windows >= the window threshold.
usage: python analysis/papersim/fig_moving_main.py [--movie out/papersim/moving_exp_v2psf] [--model m1809v2] [--name si_simulated_live_smlm_extended]
       [--outdir out/paper_figs_v2] [--blocks 10,40]
output: <outdir>/<name>.{pdf,svg,png}, <name>_caption.txt, <name>_table.md / .csv; <movie>/full/moving_gterr_<model>.json (cache)"""
import os, sys, json, csv, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, metrics as M
from fig_density import render, COL
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

METH = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v1', 'SIMCODE\n(preprint)'), ('v2sig', 'SIMCODE v2')]
TABLE = [('smlm', 'SMLM'), ('sf', 'SIMFLUX (estimated pattern)'), ('sf_gtp', 'SIMFLUX (true pattern)'), ('decode', 'DECODE (retrained)'),
         ('v1', 'SIMCODE preprint (estimated pattern)'), ('v2sig', 'SIMCODE v2')]
RMAX_NM, BAD_NM = 100.0, 30.0
TICK = dict(smlm='SMLM', sf='SIMFLUX', decode='DECODE', v1='SIMCODE\n(preprint)', v2sig='SIMCODE\nv2')


def letter(ax, s, dx=-0.08, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')


def gt_windows(md, nframes):
    """GT emitter windows (disjoint 6-frame windows): window index, photon-weighted mean x / y (px), detected photons"""
    g = np.load(os.path.join(md, 'gt_rows.npz'))
    fr, sp, x, y, ph = g['frame'].astype(np.int64), g['spot'].astype(np.int64), g['x'].astype(np.float64), g['y'].astype(np.float64), g['photons'].astype(np.float64)
    ok = fr < nframes - nframes % 6
    fr, sp, x, y, ph = fr[ok], sp[ok], x[ok], y[ok], ph[ok]
    key = (fr // 6) * (sp.max() + 1) + sp
    u, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, ph); w0 = np.maximum(w, 1e-12)
    return dict(win=u // (sp.max() + 1), x=np.bincount(inv, ph * x) / w0, y=np.bincount(inv, ph * y) / w0, ph=w)


def moving_gterr(T, G, W, nframes, thr_ph):
    from scipy.spatial import cKDTree
    _, _, f = M.frc_set(T, W, nframes)
    lx, ly, lw = T['x'][f], T['y'][f], T['f'][f] // 6
    S = 1000.0
    tree = cKDTree(np.c_[G['x'], G['y'], G['win'] * S])
    d, j = tree.query(np.c_[lx, ly, lw * S], distance_upper_bound=RMAX_NM / C.PX_NM)
    m = np.isfinite(d)
    ex = (lx[m] - G['x'][j[m]]) * C.PX_NM; ey = (ly[m] - G['y'][j[m]]) * C.PX_NM
    r = np.hypot(ex, ey)
    # recall: GT windows with >= thr_ph detected photons (inside the field) that have a localization within 100 nm
    gi = (G['ph'] >= thr_ph) & (G['x'] >= 0) & (G['x'] < W) & (G['y'] >= 0) & (G['y'] < W) & (G['win'] * 6 + 6 <= nframes)
    if len(lx):
        dl, _ = cKDTree(np.c_[lx, ly, lw * S]).query(np.c_[G['x'][gi], G['y'][gi], G['win'][gi] * S], distance_upper_bound=RMAX_NM / C.PX_NM)
        rec = float(np.isfinite(dl).mean())
    else:
        rec = 0.0
    return dict(n_frc=int(f.sum()), matched_frac=float(m.mean()) if len(m) else float('nan'),
                rms_nm=float(np.sqrt(0.5 * (ex ** 2 + ey ** 2).mean())) if m.any() else float('nan'),
                median_err_nm=float(np.median(r)) if m.any() else float('nan'),
                bad_frac=float(1 - (r <= BAD_NM).sum() / len(m)) if len(m) else float('nan'),
                recall=rec, recall_photon_threshold=float(thr_ph), n_gt_windows=int(gi.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movie', default=os.path.join(C.OUT, 'moving_exp_v2psf')); ap.add_argument('--model', default='m1809v2')
    ap.add_argument('--name', default='si_simulated_live_smlm_extended'); ap.add_argument('--outdir', default=os.path.join(os.path.dirname(C.OUT), 'paper_figs_v2'))
    ap.add_argument('--blocks', default='10,40'); ap.add_argument('--redo_err', type=int, default=0)
    a = ap.parse_args()
    md = os.path.abspath(a.movie); run = os.path.join(md, 'full')
    R = json.load(open(os.path.join(run, f'metrics_{a.model}.json'))); g0 = json.load(open(os.path.join(md, 'gen.json')))
    W, nf, bs, nb = g0['W'], R['nframes'], R['block'], R['blocks']
    traj = np.load(os.path.join(md, 'traj.npy'), mmap_mode='r'); ts = int(g0['traj_step'])
    T = {k: M.load_method(k, R['methods'][k]['path']) for k, _ in TABLE if k in R['methods']}
    # GT-matched errors (cached)
    ef = os.path.join(run, f'moving_gterr_{a.model}.json')
    E = json.load(open(ef)) if os.path.exists(ef) and not a.redo_err else {}
    if set(T) - set(E):
        G = gt_windows(md, nf); thr = float(g0.get('window_threshold') or 0.0)
        for k in T:
            if k not in E: E[k] = moving_gterr(T[k], G, W, nf, thr); print(k, E[k], flush=True)
        json.dump(E, open(ef, 'w'), indent=1)
    blocks = [int(v) for v in a.blocks.split(',')]
    nc = len(METH) + 2; tw = 2.2
    fig = plt.figure(figsize=(nc * tw + 4.4, len(blocks) * tw + 4.4))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[len(blocks) * tw + 0.4, 3.4], hspace=0.2, left=0.05, right=0.985, top=0.95, bottom=0.1)
    top = outer[0].subgridspec(1, 2, width_ratios=[nc * tw, 3.6], wspace=0.06)
    gimg = top[0].subgridspec(len(blocks), nc, wspace=0.03, hspace=0.03)
    import tifffile
    zoom = 10
    with tifffile.TiffFile(os.path.join(md, 'movie.tif')) as tf:
        for i, b in enumerate(blocks):
            f0, f1 = b * bs, (b + 1) * bs
            fr = tf.pages[f0 + bs // 2].asarray().astype(np.float32)
            A = fig.add_subplot(gimg[i, 0]); A.imshow(fr, cmap='gray', vmin=np.percentile(fr, 1), vmax=np.percentile(fr, 99.8)); A.set_xticks([]); A.set_yticks([])
            A.set_ylabel(f'frames {f0 // 1000}k-{f1 // 1000}k', fontsize=12)
            if i == 0: A.set_title('Input frame', fontsize=12); letter(A, 'A', dx=-0.14)
            P = np.asarray(traj[f0 // ts:f1 // ts]).reshape(-1, 2)
            A = fig.add_subplot(gimg[i, 1]); A.imshow(render(P[:, 0], P[:, 1], (0, 0), W, zoom), cmap='inferno'); A.set_xticks([]); A.set_yticks([])
            if i == 0: A.set_title('Ground truth', fontsize=12)
            if i == len(blocks) - 1:
                A.plot([8, 8 + 1000 / (C.PX_NM / zoom)], [W * zoom - 12] * 2, 'w-', lw=3); A.text(8, W * zoom - 22, '1 µm', color='w', fontsize=9)
            for j, (k, lab) in enumerate(METH):
                A = fig.add_subplot(gimg[i, j + 2]); A.set_xticks([]); A.set_yticks([])
                if i == 0: A.set_title(lab, fontsize=12)
                if k not in T:
                    A.set_facecolor('k'); A.text(0.5, 0.5, 'no result', color='w', ha='center', transform=A.transAxes); continue
                t = T[k]; _, _, f = M.frc_set(t, W, nf); s = f & (t['f'] >= f0) & (t['f'] < f1)
                A.imshow(render(t['x'][s], t['y'][s], (0, 0), W, zoom), cmap='inferno')
                fb = R['methods'][k]['frc_block_nm'][b]
                fl = f'FRC {fb:.0f} nm\n' if fb == fb else 'FRC n/a\n'
                A.text(0.03, 0.03, f"{fl}{s.sum() / 1e3:.1f}k localizations", transform=A.transAxes, color='w', fontsize=9, va='bottom',
                       bbox=dict(facecolor='k', alpha=0.55, pad=1.5, lw=0))
    # B PSF panel
    import psfpanel
    rn = np.load(os.path.join(md, 'camera_sim.npz'))['readnoise_e']
    v2 = bool(g0.get('v2psf'))
    pat = g0.get('pattern') or {}
    info = psfpanel.psf_info(g0['psf'], g0['brightness_fit']['fit_median'] if g0.get('brightness_fit') else g0['intensity_per_6frames'], g0['bg_mean'],
                             pattern={k: pat[k] for k in ('angles_deg', 'pitch_nm', 'depth') if k in pat},
                             **(dict(detected=True, rn2_e=float((rn.astype(float) ** 2).mean()), zmax=0.4, z_um=g0['z_um']) if v2 else {}))
    gp = top[1].subgridspec(3, 2, height_ratios=[0.25, 1.0, 0.9], wspace=0.6, hspace=0.12)
    axy, axz, axt = fig.add_subplot(gp[1, 0]), fig.add_subplot(gp[1, 1]), fig.add_subplot(gp[2, :])
    psfpanel.draw(axy, axz, info, fontsize=8); axz.set_box_aspect(1); letter(axy, 'B', dx=-0.1, dy=1.12)
    axt.axis('off'); axt.text(0.0, 0.95, psfpanel.panel_label(info), transform=axt.transAxes, fontsize=10, va='top', ha='left', linespacing=1.2)
    # C / D / E
    gb = outer[1].subgridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.28)
    Cx, Dx, Ex = fig.add_subplot(gb[0]), fig.add_subplot(gb[1]), fig.add_subplot(gb[2])
    keys = [k for k, _ in METH if k in T]; labs = {k: l.replace('\n', ' ') for k, l in METH}
    xb = np.arange(nb) * bs / 1000 + bs / 2000
    for k in keys:
        lw = 2.2 if k == 'v2sig' else 1.2
        Cx.plot(xb, R['methods'][k]['frc_block_nm'], color=COL[k], lw=lw, label=labs[k])
    Cx.set_xlabel('time (x1000 frames)', fontsize=10); Cx.set_ylabel('FRC per 1000-frame block (nm)', fontsize=10)
    Cx.set_title('FRC per block', fontsize=11); Cx.set_ylim(0, None); Cx.legend(fontsize=8, frameon=False, ncol=3, loc='upper left'); letter(Cx, 'C', dx=-0.07)
    cy = Cx.get_ylim(); Cx.set_ylim(0, cy[1] * 1.25)
    xs = np.arange(len(keys))
    Dx.bar(xs, [E[k]['rms_nm'] for k in keys], color=[COL[k] for k in keys])
    Dx.set_ylabel('RMS error, matched (nm)', fontsize=10); Dx.set_title('Error vs ground truth', fontsize=11)
    D2 = Dx.twinx(); D2.plot(xs, [100 * E[k]['bad_frac'] for k in keys], 'kD', ms=6, mfc='white', mew=1.5)
    D2.set_ylabel(f'% unmatched or > {BAD_NM:.0f} nm (◇)', fontsize=10); D2.set_ylim(0, 100)
    Dx.set_xticks(xs); Dx.set_xticklabels([TICK[k] for k in keys], fontsize=8.5); letter(Dx, 'D', dx=-0.18)
    w = 0.27
    for j, (t, al) in enumerate(zip(R['thresholds_nm'], (1.0, 0.65, 0.35))):
        Ex.bar(xs + (j - 1) * w, [R['methods'][k]['jaccard_mean'][j] for k in keys], w, color=[COL[k] for k in keys], alpha=al, label=f'{t} nm', edgecolor='k', lw=0.3)
    Ex.set_ylim(0, 1.15); Ex.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); Ex.set_ylabel('mean Jaccard index', fontsize=10); Ex.set_title('Jaccard (10 / 20 / 40 nm)', fontsize=11)
    Ex.set_xticks(xs); Ex.set_xticklabels([TICK[k] for k in keys], fontsize=8.5); letter(Ex, 'E', dx=-0.15)
    from matplotlib.patches import Patch
    Ex.legend(handles=[Patch(facecolor='0.4', alpha=al, edgecolor='k', label=f'{t} nm') for t, al in zip(R['thresholds_nm'], (1.0, 0.65, 0.35))], fontsize=8, frameon=False, loc='upper left', ncol=3)
    for X in (Cx, Dx, D2, Ex): X.tick_params(labelsize=9)
    os.makedirs(a.outdir, exist_ok=True); out = os.path.join(a.outdir, a.name)
    # table
    rows = []
    for k, lab in TABLE:
        if k not in T: continue
        r = R['methods'][k]; e = E[k]
        rows.append(dict(method=lab, rows=r['n_all'], frc_rows_per_block=round(e['n_frc'] / nb), frc_block_median_nm=r['frc_block_median_nm'],
                         jaccard_10nm=r['jaccard_mean'][0], jaccard_20nm=r['jaccard_mean'][1], jaccard_40nm=r['jaccard_mean'][2],
                         rms_error_matched_nm=e['rms_nm'], median_error_nm=e['median_err_nm'], frac_unmatched_or_gt30nm=e['bad_frac'], recall=e['recall']))
    with open(out + '_table.csv', 'w', newline='') as fh:
        Wr = csv.DictWriter(fh, fieldnames=list(rows[0])); Wr.writeheader(); Wr.writerows(rows)
    f1 = lambda v, s='{:.1f}': '-' if v is None or v != v else s.format(v)
    tmd = [f'# Moving-microtubule simulation table ({os.path.basename(md)}, SIMCODE v2 = {a.model}; {nf} frames, {nb} blocks of {bs})', '',
           'rows = all output rows; FRC set = disjoint 6-frame windows, σ < 0.15 px (SIMCODE v2: its σ ≤ 12 nm selection first); FRC per block = median over the '
           'blocks of the half-split FRC within each block; Jaccard = mean over blocks (GT = all trajectory points of the block); error = GT-matched '
           f'(nearest ON emitter of the window within {RMAX_NM:.0f} nm); recall = GT emitter windows with >= {E[keys[0]]["recall_photon_threshold"]:.0f} detected photons '
           f'that have an FRC-set localization within {RMAX_NM:.0f} nm.', '',
           '| method | rows | FRC-set rows per block | FRC per block, median (nm) | Jaccard 10 / 20 / 40 nm | RMS error (nm) | median error (nm) | unmatched or > 30 nm | recall |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        tmd.append(f"| {r['method']} | {r['rows']} | {r['frc_rows_per_block']} | {f1(r['frc_block_median_nm'])} | {r['jaccard_10nm']:.3f} / {r['jaccard_20nm']:.3f} / {r['jaccard_40nm']:.3f} | "
                   f"{f1(r['rms_error_matched_nm'])} | {f1(r['median_error_nm'])} | {f1(r['frac_unmatched_or_gt30nm'], '{:.3f}')} | {f1(r['recall'], '{:.3f}')} |")
    open(out + '_table.md', 'w').write('\n'.join(tmd) + '\n')
    # caption
    zd = g0.get('z_dist_um')
    bf = g0.get('brightness_fit')
    cond = (f"experimental photons and background of 1809/2 (median {bf['fit_median']:.0f} detected photons per 6-frame window, lognormal, "
            f"{g0['bg_mean']:.2f} photons/px/frame, mean on-time {g0['avg_on_time_used']:.0f} frames)" if bf else
            f"preprint condition ({g0['intensity_per_6frames']:.0f} photons per 6-frame window, background {g0['bg']:g} photons/px/frame)")
    psfs = (f"the in-situ PSF of 1809/2 and the sCMOS camera maps of that movie (gain, offset, per-pixel read noise), emitter heights 0-400 nm "
            f"(median {zd['p50'] * 1e3:.0f} nm, one height per labelling site)" if zd else 'the earlier averaged in-situ PSF at a fixed height of 55 nm and a pure-Poisson camera')
    v2r = rows[-1] if rows[-1]['method'] == 'SIMCODE v2' else None
    cap = [f"Simulated moving microtubules: the preprint's simulation ({g0['n_tubules']} microtubules with Brownian motion and spring-like mechanics, "
           f"{W} x {W} px of {C.PX_NM} nm, {nf} frames after {g0['skip']} skipped), rendered with {psfs}; {cond}; 2 x 3-step pattern, pitch "
           f"{pat.get('pitch_nm', 224):.0f} nm, depth {pat.get('depth', 0.9):g}. The reconstruction is divided into {nb} blocks of {bs} frames; each block gives one "
           "reconstructed frame (movie in the supplemental media).",
           f"(A) One input frame and the reconstructions of two blocks (frames {blocks[0] * bs}-{(blocks[0] + 1) * bs} and {blocks[1] * bs}-{(blocks[1] + 1) * bs}); ground truth = all "
           "positions of the labelling sites during the block. Every method is shown with the localizations that enter its FRC (disjoint 6-frame windows, σ < 0.15 px; "
           "SIMCODE v2: localizations with calibrated precision ≤ 12 nm); each tile gives the block's FRC and number of localizations. SIMFLUX and the preprint "
           "SIMCODE use the pattern estimated from the data, as SIMCODE v2 does; DECODE was retrained for this PSF, the emitter heights and the photon / background / "
           "density ranges.",
           psfpanel.caption_text(info, what='(B)'),
           "(C) FRC of each 1000-frame block (half-split within the block, halves alternating every 60 frames). (D) Localization error against the ground truth: "
           f"each localization is matched to the nearest emitter that is on in its 6-frame window (photon-weighted mean position over the window) within {RMAX_NM:.0f} nm; "
           f"bars = RMS error of the matched localizations, diamonds = % of localizations that are unmatched or more than {BAD_NM:.0f} nm off. "
           "(E) Mean Jaccard index over the blocks at 10 / 20 / 40 nm (the preprint's measure: ground truth = all trajectory points of the block). With "
           "~10^6 ground-truth points per block it mostly measures how much of the moving structure is covered, so it grows with the number of localizations "
           "rather than with their accuracy; read it together with C and D."]
    if v2r:
        cap.append("Numbers (" + os.path.basename(md) + "): " + "; ".join(
            f"{r['method']}: FRC per block {f1(r['frc_block_median_nm'], '{:.0f}')} nm, RMS error {f1(r['rms_error_matched_nm'])} nm, "
            f"{100 * r['frac_unmatched_or_gt30nm']:.0f} % unmatched / > 30 nm, Jaccard {r['jaccard_10nm']:.2f} / {r['jaccard_20nm']:.2f} / {r['jaccard_40nm']:.2f}" for r in rows) + ".")
    open(out + '_caption.txt', 'w').write('\n\n'.join(cap) + '\n')
    for ext, kw in (('svg', dict(dpi=300)), ('pdf', dict(dpi=300)), ('png', dict(dpi=150))):
        plt.savefig(f'{out}.{ext}', **kw); print(f'{out}.{ext}')
    print(out + '_caption.txt'); print(out + '_table.md')


if __name__ == '__main__':
    main()
