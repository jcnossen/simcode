"""Per-density figure in the preprint's Fig 2 / SI S1 style + the summary table over densities + old-vs-new SIMCODE.

Per density (<movie>/full, net set --model): row 1 renders of the same crop (the preprint's Fig 2 zoom: corner (9, 10) px,
16 px wide) for the ground truth and every method; row 2 FRC curves, localization counts (FRC set), Jaccard (10 nm, end of the
movie), Jaccard vs frames. Rows rendered = each method's FRC set (disjoint windows, max sigma < 0.15 px; SIMCODE v1 = its merged
'sfhd-ndi+smlm' rows in that set; v2 = its main selection sigma <= 12 nm (v2sig)), 5.4 nm pixels (zoom 20), 1-pixel Gaussian blur, 99.5 % clip.
DECODE = dec_1809_2, OUT of its training range for these movies (flagged in the labels).
usage: python analysis/papersim/fig_density.py [--model m1809] [--densities 0.5,1,2,4]
outputs: out/papersim/fig_density_D<d>_<model>.png, out/papersim/summary_density_<model>.json / .md
With <run>/gtfrc_<model>.json (gtfrc_run.py): GT-FRC in the panel titles and the columns GT-FRC, frames to 40 / 30 nm, sigma-gated
Jaccard, ghost fraction, coverage CV excess in the summary (fig_gtfrc.py has the full table and the time-to-fidelity figure)."""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import metrics as M

ORDER = ['smlm', 'sf', 'sf_gtp', 'decode', 'decode_pre', 'decode_sim', 'v1', 'v1gt', 'v2sig']      # table order; figures: FIG_ORDER
FIG_ORDER = ['smlm', 'sf', 'sf_gtp', 'decode', 'decode_pre', 'decode_sim', 'v1', 'v2sig']
LAB = dict(smlm='SMLM', sf='SIMFLUX (est. pattern)', sf_gtp='SIMFLUX (GT pattern)', decode='DECODE (dec_1809_2, out of range)', decode_sim='DECODE (retrained dec_sim1809)', decode_pre='DECODE (preprint model)',
           v1='SIMCODE v1 (preprint)', v1gt='SIMCODE v1, GT pattern (diag.)', v2sig='SIMCODE v2, σ <= 12 nm', v2sel='SIMCODE v2 all')
COL = dict(smlm='tab:blue', sf='tab:green', sf_gtp='olive', decode='tab:orange', decode_sim='darkorange', decode_pre='orangered', v1='tab:brown', v1gt='peru', v2sig='crimson', v2sel='salmon')
CROP = ((9, 10), 16)


def render(x, y, corner, w, zoom=20):
    from scipy.ndimage import gaussian_filter
    n = w * zoom
    img = np.histogram2d(y, x, bins=n, range=[[corner[1], corner[1] + w], [corner[0], corner[0] + w]])[0]
    img = gaussian_filter(img, 1.0)
    return np.clip(img / (np.percentile(img, 99.5) + 1e-9), 0, 1)


def fig(d, model, sub='full', prefix='D', suffix=''):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    md = os.path.join(C.OUT, f'{prefix}{d:g}{suffix}'); run = os.path.join(md, sub)
    R = json.load(open(os.path.join(run, f'metrics_{model}.json')))
    gf = os.path.join(run, f'gtfrc_{model}.json'); GF = json.load(open(gf))['methods'] if os.path.exists(gf) else {}
    R['_gtfrc'] = GF
    gt = np.load(os.path.join(md, 'gt.npz'))
    ks = [k for k in FIG_ORDER if k in R['methods']]
    fig = plt.figure(figsize=(2.3 * (len(ks) + 1), 7.5))
    gs = fig.add_gridspec(2, len(ks) + 1, height_ratios=[1, 0.9])
    (cx, cy), w = CROP
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(render(gt['pts'][:, 0], gt['pts'][:, 1], (cx, cy), w), cmap='inferno'); ax.set_title('ground truth', fontsize=8); ax.axis('off')
    ax.plot([5, 5 + 200 / (C.PX_NM / 20)], [w * 20 - 8] * 2, 'w-', lw=3); ax.text(5, w * 20 - 14, '200 nm', color='w', fontsize=7)
    for i, k in enumerate(ks):
        T = M.load_method(k, R['methods'][k]['path']); _, _, f = M.frc_set(T, R['W'], R['nframes'])
        ax = fig.add_subplot(gs[0, i + 1]); ax.imshow(render(T['x'][f], T['y'][f], (cx, cy), w), cmap='inferno'); ax.axis('off')
        r = R['methods'][k]; gl = f", GT-FRC {GF[k]['gt_frc']['res_nm']:.1f}" if k in GF else ''
        ax.set_title(f"{LAB[k]}\nFRC {r.get('frc_nm', float('nan')):.1f} nm{gl}, J {r['jaccard_end']:.2f}", fontsize=7)
    n4 = len(ks) + 1; q = n4 // 4
    a1 = fig.add_subplot(gs[1, 0:q]); a2 = fig.add_subplot(gs[1, q:2 * q]); a3 = fig.add_subplot(gs[1, 2 * q:3 * q]); a4 = fig.add_subplot(gs[1, 3 * q:])
    for k in ks:
        r = R['methods'][k]
        if 'frc_curve' in r: a1.plot(r['frc_freq'], r['frc_curve'], color=COL[k], label=f"{LAB[k].split(' (')[0] if k != 'sf_gtp' else 'SIMFLUX GTp'} {r['frc_nm']:.1f}")
        a4.plot(np.array(r['jaccard_frames']) / 1000, r['jaccard'], color=COL[k], lw=1.8)
    a1.axhline(1 / 7, color='grey', ls='--'); a1.set_xlim(0, 0.08); a1.set_xlabel('1/nm'); a1.set_title('FRC (fair protocol)', fontsize=8); a1.legend(fontsize=5.5)
    a2.bar(range(len(ks)), [R['methods'][k]['n_frc'] / 1e3 for k in ks], color=[COL[k] for k in ks]); a2.set_title('localizations (FRC set, x1000)', fontsize=8)
    a3.bar(range(len(ks)), [R['methods'][k]['jaccard_end'] for k in ks], color=[COL[k] for k in ks]); a3.set_title(f'Jaccard 10 nm ({R["nframes"] // 1000}k frames)', fontsize=8); a3.set_ylim(0, 1)
    for a in (a2, a3): a.set_xticks(range(len(ks))); a.set_xticklabels([LAB[k].split(' (')[0].replace('SIMCODE ', 'SC ') + (' GTp' if k == 'sf_gtp' else '') for k in ks], rotation=60, fontsize=6)
    a4.set_ylim(0, 1); a4.set_xlabel('frames (x1000)'); a4.set_title('Jaccard vs frames (10 nm, 120 bins)', fontsize=8)
    psfn = 'in-situ PSF v2 of 1809/2 (heights 0-0.40 um), v2 camera' if suffix == '_v2psf' else ('Gaussian 1.3 px PSF' if suffix == '_gauss' else 'averaged 1809/2 PSF')
    fig.suptitle(f'density {d:g} emitters / um^2 / frame, {R["nframes"]} frames, {psfn}; v2 net set {model}; crop corner {CROP[0]} px, {w} px', fontsize=9)
    plt.tight_layout(); fn = os.path.join(C.OUT, f'fig_density_{prefix}{d:g}{suffix}_{model}' + ('' if sub == 'full' else f'_{sub}') + '.png'); plt.savefig(fn, dpi=110); plt.close(fig)
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='m1809'); ap.add_argument('--densities', default='0.5,1,2,4'); ap.add_argument('--sub', default='full'); ap.add_argument('--suffix', default=''); ap.add_argument('--prefix', default='D', help="'D' (preprint 500 ph / bg 2.5) or 'Dexp' (1809/2 photons / bg, on-time 20)")
    a = ap.parse_args()
    if a.suffix == '_v2psf':      # v2-PSF sets (2026-11-15): 'decode' = the DECODE retrained for the v2 PSF and these conditions
        LAB['decode'] = 'DECODE (retrained dec_sim1809v2)'
    rows = []; md_lines = [f'# Density sweep {a.prefix}{a.suffix}, v2 net set {a.model} (60 000 frames; FRC fair protocol; Jaccard 10 nm at 60k frames; GT RMS = FRC set vs its ON emitter)', '',
                           '| density | method | n_all | n_frc | FRC nm | FRC ideal: same rows at robust σ / shuffled own errors (nm) | robust σ nm | err > 30 nm | Jaccard | GT RMS x,y nm | recall (>=150 ph) | med sigma nm | GT-FRC nm | frames to 40 / 30 / 20 nm GT-FRC | σ-Jaccard | false posst | coverage CV excess |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for d in [float(v) for v in a.densities.split(',')]:
        try:
            R = fig(d, a.model, a.sub, a.prefix, a.suffix)
        except FileNotFoundError as e:
            print('skip', d, e); continue
        fl = R.get('frc_floor', {})
        md_lines.append(f"| {d:g} | (floor: all {fl.get('n_events')} GT events at sigma 1 / 5 / 10 nm) | | | {fl.get('1', float('nan')):.1f} / {fl.get('5', float('nan')):.1f} / {fl.get('10', float('nan')):.1f} | | | | | | | | | | | | |")
        GF = R.get('_gtfrc', {}); fk = lambda v: '-' if v is None else (f'<= {v / 1e3:g}k' if v <= 250 else f'{v / 1e3:.1f}k')
        for k in [k for k in ORDER if k in R['methods']]:
            r = R['methods'][k]; g = r['gt_match']
            rows.append(dict(density=d, method=k, n_all=r['n_all'], n_frc=r['n_frc'], frc_nm=r.get('frc_nm'), frc_ideal_nm=r.get('frc_ideal_nm'), frc_ideal_shuffled_nm=r.get('frc_ideal_shuffled_nm'), sigma_robust_nm=r.get('sigma_robust_nm'), jaccard=r['jaccard_end'],
                             rms_nm=g.get('rms_nm'), recall=g.get('recall_gt_ph150'), med_sigma_nm=g.get('median_sigma_nm')))
            md_lines.append(f"| {d:g} | {LAB[k]} | {r['n_all']} | {r['n_frc']} | {r.get('frc_nm', float('nan')):.1f} | {r.get('frc_ideal_nm', float('nan')):.1f} / {r.get('frc_ideal_shuffled_nm', float('nan')):.1f} | {r.get('sigma_robust_nm', float('nan')):.1f} | {g.get('frc_err_gt30', g.get('frac_err_gt30', float('nan'))):.3f} | {r['jaccard_end']:.3f} | "
                            f"{','.join(f'{v:.1f}' for v in g.get('rms_nm', []))} | {g.get('recall_gt_ph150', float('nan')):.3f} | {','.join(f'{v:.1f}' for v in g.get('median_sigma_nm', []))} | "
                            + (f"{GF[k]['gt_frc']['res_nm']:.1f} | {fk(GF[k]['ttf']['frames_to']['40'])} / {fk(GF[k]['ttf']['frames_to']['30'])} / {fk(GF[k]['ttf']['frames_to'].get('20'))} | {GF[k]['sigma_jaccard']['jaccard']:.3f} | "
                               f"{GF[k]['ghost']['ghost_fraction']:.3f} | {GF[k]['coverage']['excess_cv']:.2f} |" if k in GF else '| | | | |'))
            if k in GF: rows[-1].update(gt_frc_nm=GF[k]['gt_frc']['res_nm'], frames_to_40=GF[k]['ttf']['frames_to']['40'], frames_to_30=GF[k]['ttf']['frames_to']['30'],
                                        sigma_jaccard=GF[k]['sigma_jaccard']['jaccard'], ghost=GF[k]['ghost']['ghost_fraction'], coverage_excess_cv=GF[k]['coverage']['excess_cv'])
    json.dump(rows, open(os.path.join(C.OUT, f'summary_density_{a.prefix}{a.suffix}_{a.model}.json'), 'w'), indent=1)
    open(os.path.join(C.OUT, f'summary_density_{a.prefix}{a.suffix}_{a.model}.md'), 'w').write('\n'.join(md_lines) + '\n')
    print('\n'.join(md_lines))


if __name__ == '__main__':
    main()
