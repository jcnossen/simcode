"""Solve-gap diagnostic (2026-11-14): out/solvegap/solvegap_p<1|2>.json + inputs_p<1|2>.json -> out/solvegap/tables.md
usage: python analysis/solvegap/report.py"""
import os, json

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'out', 'solvegap')
BN = ['500-1000', '1000-2000', '2000-4000', '4000-10000', '10000-inf']
HB = ['500-1k', '1-2k', '2-4k', '4-10k', '>10k']
ROWS = [('A', 'A  classical SIMFLUX pixel MLE (from SMLM fit, true pattern)'),
        ('A_true', 'A\'  same, started at the truth'),
        ('B', 'B  production solve (c = cmult = cmult_ang = 0.1)'),
        ('B0', 'B0 production solve, c = 0'),
        ('B0_nokin', 'B0 without the kinetic model'),
        ('B0_map', 'B0 MAP instead of posterior mean'),
        ('C', 'C  oracle inputs (free-model CRLB noise, exact cov), c = 0'),
        ('C_nokin', 'C  without the kinetic model'),
        ('C_lin', 'C  linear GLS at the truth (oracle inputs)'),
        ('D', 'D  network inputs, c = 0, start set centred on the truth'),
        ('D_nf0', 'D  single start at the truth (no fringe alternatives, no kinetics)'),
        ('E', 'E  network inputs, c = 0, empirical per-bin covariance'),
        ('E_nokin', 'E  without the kinetic model'),
        ('L', 'L  linear GLS at the truth on the realised network inputs (c = 0 weights)'),
        ('B0_ds_nokin', 'B0, intensities de-shrunk by the per-bin modulation slope, no kinetics'),
        ('L_ds', 'L  on the de-shrunk intensities'),
        ('prior', 'network prior position')]


def main():
    L = ['# Solve-gap diagnostic, iso m1809a, seeds 301-303 pooled', '',
         'h3-matched targets (Hungarian 3 px on the detector point, score >= 0.5), GT-photon bins. Cells: RMSD / CRLB_SF (gt_sf_i) x / y, '
         'then % of axis values beyond 3 CRLB (Gaussian: 0.27 %). Scripts analysis/solvegap/ (sffit.py = A, solvegap.py = B-E, L, '
         'inputs.py = input diagnostics).', '']
    for ps in (2, 1):
        J = json.load(open(f'{OUT}/solvegap_p{ps}.json')); T = J['tab']
        L += [f'## Pass {ps}', '', '| estimate | ' + ' | '.join(HB) + ' |', '|---|' + '---|' * len(HB)]
        n = [str(T['B'][b]['n']) for b in BN]
        L.append('| n | ' + ' | '.join(n) + ' |')
        for k, nm in ROWS:
            if k not in T: continue
            L.append(f'| {nm} | ' + ' | '.join(f"{T[k][b]['rx']:.3f} / {T[k][b]['ry']:.3f} ({100 * T[k][b]['tail3']:.2f} %)" for b in BN) + ' |')
        if ps == 1: L += ['', '(A rows of pass 1 = the pass-2 h3 rows that are also h3-matched in pass 1.)']
        L += ['', f'Rows beyond 3 CRLB_SF on either axis, pass {ps} (categories in order: wrong fringe = |k_a . dr| >= half a period '
              'on an angle; start = the same solve started on the truth is within 3 sigma; c-term = within 3 sigma at c = 0; kinetic = '
              'kinetic model selected; PM = MAP within 3 sigma; inputs = the linear GLS at the truth on the same inputs is also beyond 3 '
              'sigma; other = nonlinear solve on otherwise good inputs). Prior distance: median (max) nm; z = in units of the '
              'calibrated network sigma; fr = in units of the free-model CRLB; all = median over every row of the bin.', '']
        for key, nm in (('cls_B', 'B (c calibrated)'), ('cls_B0', 'B0 (c = 0)')):
            L += [f'{nm}:', '', '| bin | bad / n | wrong fringe | start | c-term | kinetic | PM | inputs | other | prior err nm | prior z | prior / fr | prior nm (all) | solve err nm | max fringe units |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
            for b, h in zip(BN, HB):
                r = J[key][b]
                if not r.get('n_bad'): L.append(f'| {h} | 0 | | | | | | | | | | | | | |'); continue
                L.append(f"| {h} | {r['n_bad']} / {r['n_bin']} | {r['wrong_fringe']} | {r['start']} | {r['c_term']} | {r['kinetic']} | {r['pm_mix']} | "
                         f"{r['inputs']} | {r['other']} | {r['prior_err_nm_median']:.1f} ({r['prior_err_nm_max']:.0f}) | {r['prior_z_median']:.2f} | "
                         f"{r['prior_err_over_frxy_median']:.2f} | {r['prior_err_nm_median_all']:.1f} | {r['solve_err_nm_median']:.1f} | {r['solve_err_fringe_units_max']:.2f} |")
            L.append('')
        I = json.load(open(f'{OUT}/inputs_p{ps}.json'))
        L += [f'Input diagnostics, pass {ps} (linear at the truth, x / y over CRLB_SF; rho = correlation of the prior error with the '
              'intensity-only position error; comb = best linear combination of prior and intensity-only position given rho; indep = '
              'the combination that assumes independence, as the solve does; s = modulation slope of the network intensities; '
              'resid = their mean-removed residual after the slope over the free-model CRLB):', '',
              '| bin | prior | I-only network | I-only oracle | rho | comb (knows rho) | indep | s | resid |', '|---|---|---|---|---|---|---|---|---|']
        f2 = lambda v: f'{v[0]:.3f} / {v[1]:.3f}'
        for b, h in zip(BN, HB):
            r = I[b]
            L.append(f"| {h} | {f2(r['prior'])} | {f2(r['Ionly_net'])} | {f2(r['Ionly_oracle'])} | {f2(r['rho_prior_Ionly'])} | "
                     f"{f2(r['comb_opt'])} | {f2(r['comb_indep'])} | {r['mod_slope']:.3f} | {r['resid_rms_over_crlb_fr']:.2f} |")
        L.append('')
    J2 = json.load(open(f'{OUT}/solvegap_p2.json'))
    L += [f"Checks: offline B / B0 reproduce the cached production solve to {J2['chk']['B']:.1e} / {J2['chk']['B0']:.1e} px; the Fisher "
          f"bound of the fit model of A at the truth over the stored gt_sf_i: median {J2['chk']['crlb_model_over_sf_i_median']:.5f}.", '']
    open(f'{OUT}/tables.md', 'w').write('\n'.join(L))


if __name__ == '__main__':
    main()
