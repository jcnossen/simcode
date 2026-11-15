"""Supplementary Table S1: imaging conditions and single-emitter CRLBs per real data set, from the PSF-panel values that
fig_exp_new.py --psf writes (<out>_values.json, key psf_panel; the same numbers as its caption_text / <out>_psf_table).

One column per data set; a data set whose values file does not exist yet is shown as 'pending'. The CRLB is computed for one
isolated emitter at the median emitter height and the median signal of the plotted SIMCODE v2 set (analysis/paper/psfdata.py,
analysis/papersim/psfpanel.py): SMLM = 3D fit (x, y, z, I, bg) of the 6-frame sum; SIMFLUX = the 6 frames with the calibration's
pattern (2 orientations x 3 phase steps), uniformly random pattern phases; read noise as an added per-pixel variance.

usage: python analysis/paper/table_s1.py [--sets label=values.json ...] [--out out/paper_figs_v2/table_S1_conditions]
  writes <out>.md and <out>.tex (a LaTeX tabular, booktabs)"""
import os, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
V = os.path.realpath(os.path.join(REPO, 'out', 'paper_v2psf'))
DEFAULT = [('Low density (1809/2, Fig. 3)', f'{V}/1809_2_fig3_psfv2_values.json'),
           ('Medium density (2.5 nM/2, Fig. 4)', f'{V}/2.5nM_2_fig4_psfv2_values.json'),
           ('High density (20 nM/2)', f'{V}/20nM_2_fig_psfv2_values.json')]

ROWS = [  # (label md, label tex, unit, function of psf_panel -> str)
    ('PSF model', 'PSF model', '', lambda p: 'in-situ vectorial fit (Zernike Noll 5-21)'),
    ('rms aberration (Noll ≥ 5)', r'rms aberration (Noll $\geq$ 5)', 'mλ', lambda p: f"{p['rms_mwaves']:.0f}"),
    ('focal plane above the coverslip', 'focal plane above the coverslip', 'nm', lambda p: f"{p['conditions']['focus_um'] * 1e3:.0f}"),
    ('median emitter height (q25-q75)', 'median emitter height (q25--q75)', 'nm',
     lambda p: f"{p['conditions']['z_median_um'] * 1e3:.0f} ({p['conditions']['z_quantiles_um']['q25'] * 1e3:.0f}-"
               f"{p['conditions']['z_quantiles_um']['q75'] * 1e3:.0f})"),
    ('collected fraction at that height', 'collected fraction at that height', '', lambda p: f"{p['collected']:.2f}"),
    ('median signal, detected', 'median signal, detected', 'photons / 6-frame window', lambda p: f"{p['detected_photons']:.0f}"),
    ('median background', 'median background', 'photons / px / frame', lambda p: f"{p['conditions']['bg_px_frame']:.2f}"),
    ('camera read noise', 'camera read noise', 'e- rms', lambda p: f"{p['conditions']['rn2_e'] ** 0.5:.2f}"),
    ('pattern pitch / modulation depth', 'pattern pitch / modulation depth', 'nm / -',
     lambda p: f"{p['conditions']['pattern']['pitch_nm']:.0f} / {p['conditions']['pattern']['depth']:.2f}"),
    ('CRLB SMLM x / y / z', 'CRLB SMLM $x$ / $y$ / $z$', 'nm',
     lambda p: f"{p['crlb_nm']['smlm']['x']:.1f} / {p['crlb_nm']['smlm']['y']:.1f} / {p['crlb_nm']['smlm']['z']:.0f}"),
    ('CRLB SIMFLUX x / y / z', 'CRLB SIMFLUX $x$ / $y$ / $z$', 'nm',
     lambda p: f"{p['crlb_nm']['sf']['x']:.1f} / {p['crlb_nm']['sf']['y']:.1f} / {p['crlb_nm']['sf']['z']:.0f}"),
    ('CRLB SMLM x / y, z known', 'CRLB SMLM $x$ / $y$, $z$ known', 'nm',
     lambda p: f"{p['crlb_nm']['smlm']['xy_zfixed'][0]:.1f} / {p['crlb_nm']['smlm']['xy_zfixed'][1]:.1f}"),
    ('CRLB SIMFLUX x / y, z known', 'CRLB SIMFLUX $x$ / $y$, $z$ known', 'nm',
     lambda p: f"{p['crlb_nm']['sf']['xy_zfixed'][0]:.1f} / {p['crlb_nm']['sf']['xy_zfixed'][1]:.1f}"),
]


def tex_escape(s):
    return s.replace('μ', r'$\mu$').replace('λ', r'$\lambda$').replace('e- rms', r'e$^-$ rms').replace('%', r'\%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='*', default=None, help='label=values.json')
    ap.add_argument('--out', default=os.path.join(REPO, 'out', 'paper_figs_v2', 'table_S1_conditions'))
    a = ap.parse_args()
    sets = [tuple(s.split('=', 1)) for s in a.sets] if a.sets else DEFAULT
    cols = []
    for lab, f in sets:
        p = json.load(open(f)).get('psf_panel') if os.path.exists(f) else None
        cols.append((lab, p, f))
    md = ['| | unit | ' + ' | '.join(l for l, _, _ in cols) + ' |', '|---|---|' + '---|' * len(cols)]
    tex = [r'\begin{tabular}{ll' + 'c' * len(cols) + '}', r'\toprule',
           ' & unit & ' + ' & '.join(l for l, _, _ in cols) + r' \\', r'\midrule']
    for lmd, ltex, unit, fn in ROWS:
        vals = [fn(p) if p is not None else 'pending' for _, p, _ in cols]
        md.append(f'| {lmd} | {unit} | ' + ' | '.join(vals) + ' |')
        tex.append(f'{ltex} & {tex_escape(unit)} & ' + ' & '.join(tex_escape(v) for v in vals) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    notes = ('Signal, background and height: medians of the SIMCODE v2 localizations shown in the figure (the fair-FRC set); '
             'background = pass-1 network background of the p >= 0.63 rows in disjoint windows inside the crop; read noise = '
             'mean read-noise variance of the camera map over the crop. CRLB: one isolated emitter at the median signal and '
             'height, SMLM on the 6-frame sum, SIMFLUX on the 6 frames with the pattern of the calibration file (pitch, depth; 2 x 3 steps, random phases).\n\nSources: '
             + '; '.join(f'{l}: {f}' + ('' if p is not None else ' (not yet written)') for l, p, f in cols))
    open(a.out + '.md', 'w').write('\n'.join(md) + '\n\n' + notes + '\n')
    open(a.out + '.tex', 'w').write('% ' + notes.replace('\n\n', '\n% ') + '\n' + '\n'.join(tex) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
