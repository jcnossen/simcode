"""GT-FRC summary over densities + the time-to-fidelity figure (gtfrc_run.py outputs <movie>/full/gtfrc_<model>.json).

Table per density and method: GT-FRC (nm, 60k frames), frames to reach 40 / 30 nm GT-FRC, sigma-gated Jaccard (3 sigma) and
its share of correct localizations, ghost fraction (no ON emitter within 3 sigma), coverage CV per 250 nm segment (with Poisson
floor and excess), n (FRC set), next to the earlier metrics (fair FRC, 10 nm Jaccard) from metrics_<model>.json (the
reference v2 'v2sig_ref' from metrics_<ref>.json).
Figure: one panel per density, GT-FRC resolution vs frames (log), one line per method; second row the GT-FRC curves (60k).
usage: python analysis/papersim/fig_gtfrc.py --prefix D [--suffix _gauss] --model m1809_3d [--ref m1809]
outputs: out/papersim/summary_gtfrc_<prefix><suffix>_<model>.md / .json, ttf_<prefix><suffix>_<model>.png"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
from fig_density import LAB, COL, ORDER

LAB = dict(LAB, v2sig='SIMCODE v2 (3D, 15 Nov), σ <= 12 nm', v2sig_ref='SIMCODE v2 (2D, 13 Nov), σ <= 12 nm')
COL = dict(COL, v2sig_ref='grey')
TORDER = [k for k in ORDER if k != 'v2sig'] + ['v2sig_ref', 'v2sig']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='D'); ap.add_argument('--suffix', default='')
    ap.add_argument('--model', default='m1809_3d'); ap.add_argument('--ref', default='m1809'); ap.add_argument('--densities', default='0.5,1,2,4')
    a = ap.parse_args()
    if a.suffix.startswith('_v2psf'):      # v2-PSF sets (2026-11-15)
        LAB.update(decode='DECODE (retrained dec_sim1809v2)', v2sig='SIMCODE v2 (PSF v2 networks), σ <= 12 nm', v2sig_ref=f'SIMCODE v2 ({a.ref}), σ <= 12 nm')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    dens = [float(v) for v in a.densities.split(',')]
    fig, ax = plt.subplots(2, len(dens), figsize=(4.2 * len(dens), 7.6), squeeze=False)
    md = [f'# GT-FRC and fidelity diagnostics, {a.prefix}{a.suffix}, v2 {a.model} (ref {a.ref}); 60 000 frames', '',
          'GT-FRC = FRC(rendered reconstruction, rendered ground truth = all labelling sites), 5 nm pixels, whole field, 1/7; rows = the FRC set '
          '(disjoint windows, max σ < 0.15 px). Frames to 40 / 30 / 20 nm: GT-FRC on the first N frames, log-interpolated ("-" = not reached in 60k). '
          'σ-Jaccard: correct = a structure point within 3 own σ (Mahalanobis); in brackets the share of correct localizations. '
          'False positive = no emitter ON in the window within 3 own σ. Coverage CV over 250 nm filament segments (recovered / expected from the GT '
          'emitter-windows): CV (Poisson floor, excess). Fair FRC and 10 nm Jaccard (end of the movie, all rows) as before.', '',
          '| density | method | n FRC set | GT-FRC nm | frames to 40 nm | frames to 30 nm | frames to 20 nm | σ-Jaccard (correct) | false-pos. frac | coverage CV (Poisson, excess) | fair FRC nm | Jaccard 10 nm |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    rows = []
    fk = lambda v, f0=250: '-' if v is None else (f'<= {v / 1e3:g}k' if v <= f0 else f'{v / 1e3:.1f}k')
    for j, d in enumerate(dens):
        run = os.path.join(C.OUT, f'{a.prefix}{d:g}{a.suffix}', 'full')
        fn = os.path.join(run, f'gtfrc_{a.model}.json')
        if not os.path.exists(fn): print('skip', fn); continue
        G = json.load(open(fn))
        Mn = json.load(open(os.path.join(run, f'metrics_{a.model}.json')))['methods']
        Mr = json.load(open(os.path.join(run, f'metrics_{a.ref}.json')))['methods'] if a.ref and os.path.exists(os.path.join(run, f'metrics_{a.ref}.json')) else {}
        for k in [k for k in TORDER if k in G['methods']]:
            r = G['methods'][k]; m = Mr.get('v2sig', {}) if k == 'v2sig_ref' else Mn.get(k, {})
            t = r['ttf']; sj = r['sigma_jaccard']; cv = r['coverage']
            rows.append(dict(density=d, method=k, n=r['n'], gt_frc_nm=r['gt_frc']['res_nm'], frames_to_40=t['frames_to']['40'], frames_to_30=t['frames_to']['30'], frames_to_20=t['frames_to'].get('20'),
                             ttf_frames=t['frames'], ttf_res_nm=t['res_nm'], sigma_jaccard=sj['jaccard'], frac_correct=sj['frac_correct'],
                             ghost=r['ghost']['ghost_fraction'], coverage_cv=cv['cv'], coverage_poisson_cv=cv['poisson_cv'], coverage_excess_cv=cv['excess_cv'],
                             frc_nm=m.get('frc_nm'), jaccard10=m.get('jaccard_end')))
            md.append(f"| {d:g} | {LAB[k]} | {r['n']} | {r['gt_frc']['res_nm']:.1f} | {fk(t['frames_to']['40'], t['frames'][0])} | {fk(t['frames_to']['30'], t['frames'][0])} | {fk(t['frames_to'].get('20'), t['frames'][0])} | "
                      f"{sj['jaccard']:.3f} ({sj['frac_correct']:.3f}) | {r['ghost']['ghost_fraction']:.3f} | {cv['cv']:.2f} ({cv['poisson_cv']:.2f}, {cv['excess_cv']:.2f}) | "
                      f"{m.get('frc_nm', float('nan')):.1f} | {m.get('jaccard_end', float('nan')):.3f} |")
            ax[0, j].plot(np.array(t['frames']) / 1e3, t['res_nm'], marker='o', ms=3, color=COL[k], lw=2.2 if k == 'v2sig' else 1.3,
                          ls='--' if k in ('v2sig_ref', 'v1gt') else '-', label=LAB[k])
            ax[1, j].plot(r['gt_frc']['freq'], r['gt_frc']['curve'], color=COL[k], lw=2.2 if k == 'v2sig' else 1.2, ls='--' if k in ('v2sig_ref', 'v1gt') else '-',
                          label=f"{LAB[k].replace(', σ <= 12 nm', '')} {r['gt_frc']['res_nm']:.1f}")
        A = ax[0, j]; A.set_xscale('log'); A.set_yscale('log'); A.set_title(f'density {d:g}: GT-FRC vs frames', fontsize=9)
        [A.axhline(v, color='0.6', lw=0.7, ls=':') for v in (40, 30, 20)]
        A.set_xticks([0.25, 1, 4, 15, 60]); A.set_xticklabels(['0.25', '1', '4', '15', '60']); A.minorticks_off()
        A.set_xlabel('frames (x1000)'); A.set_ylabel('GT-FRC resolution (nm)'); A.set_ylim(10, 150)
        A.set_yticks([10, 15, 20, 30, 40, 60, 100, 150]); A.set_yticklabels(['10', '15', '20', '30', '40', '60', '100', '150'])
        B = ax[1, j]; B.axhline(1 / 7, color='grey', ls='--', lw=0.8); B.set_xlim(0, 0.08); B.set_ylim(-0.05, 1)
        B.set_xlabel('1/nm'); B.set_title('FRC(reconstruction, ground truth), 60k frames', fontsize=8); B.legend(fontsize=5.5)
    ax[0, 0].legend(fontsize=6, loc='upper right')
    psf = 'Gaussian 1.3 px PSF' if a.suffix == '_gauss' else ('in-situ PSF v2 of 1809/2, v2 camera' if a.suffix.startswith('_v2psf') else 'measured 1809/2 PSF')
    cond = '1809/2 photons and background, on-time 20' if a.prefix == 'Dexp' else 'preprint condition (500 ph / window, bg 2.5)'
    if a.suffix.startswith('_v2psf'): cond += ' (v2 photon units)'
    refl = f', dashed grey = v2 {a.ref}' + ('' if a.suffix.startswith('_v2psf') else ' (13 Nov)') if a.ref else ''
    fig.suptitle(f'Time to fidelity: GT-FRC (reconstruction vs ground truth, 1/7) on the first N frames; {psf}, {cond}; v2 {a.model}{refl}', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    base = os.path.join(C.OUT, f'{{}}_{a.prefix}{a.suffix}_{a.model}')
    plt.savefig(base.format('ttf') + '.png', dpi=110); plt.close(fig)
    json.dump(rows, open(base.format('summary_gtfrc') + '.json', 'w'), indent=1)
    open(base.format('summary_gtfrc') + '.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md)); print('written', base.format('ttf') + '.png')


if __name__ == '__main__':
    main()
