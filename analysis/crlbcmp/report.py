"""CRLB comparison, stage 3: tables (out/crlbcmp/tables.md) and figures (out/crlbcmp/fig_*.png) from out/crlbcmp/crlbcmp.json.
usage: python analysis/crlbcmp/report.py"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.minor.size'] = 0
from matplotlib.ticker import NullFormatter
_nf = NullFormatter()

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'out', 'crlbcmp')
R = json.load(open(f'{OUT}/crlbcmp.json'))
SCEN = ['iso', 'pair2', 'rs05qm', 'rs05pm']
REF = {'iso': ('sm_i', 'sf_i'), 'pair2': ('sm_m', 'sf_m'), 'rs05qm': ('sm_m', 'sf_m'), 'rs05pm': ('sm_m', 'sf_m')}
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
L = []
w = L.append


def ax2(s, k, f=np.mean):
    return f([s[a][k] for a in 'xy' if a in s])


def rms2(s, k):
    return float(np.sqrt(np.mean([s[a][k] ** 2 for a in 'xy' if a in s])))


def prec_rows(T, Th1=None, Tn3=None, extra=None):
    w('| GT photons | recall | n | bias x / y nm | sd x / y nm | RMSD nm | CRLB nm | RMSD/CRLB x / y | h1 | n3 | miss-pen | z_rms | >3 CRLB % | A1 x / y nm | Bayes-cell / CRLB |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for b, s in T.items():
        if 'x' not in s or 'y' not in s: continue
        h1 = rms2(Th1[b], 'ratio') if Th1 and b in Th1 and 'x' in Th1[b] else np.nan
        n3 = rms2(Tn3[b], 'ratio') if Tn3 and b in Tn3 and 'x' in Tn3[b] else np.nan
        x, y = s['x'], s['y']
        w(f"| {b} | {x['recall']:.3f} | {x['n']} | {x['bias']:+.2f} / {y['bias']:+.2f} | {x['sd']:.2f} / {y['sd']:.2f} | {rms2(s, 'rmsd'):.2f} | "
          f"{rms2(s, 'crlb'):.2f} | {x['ratio']:.3f} / {y['ratio']:.3f} | {h1:.3f} | {n3:.3f} | {rms2(s, 'ratio_pen'):.2f} | {rms2(s, 'z_rms'):.3f} | "
          f"{100 * ax2(s, 'tail3'):.2f} | {x['A1']:+.2f} / {y['A1']:+.2f} (±{x['A1_se']:.2f}) | {ax2(s, 'bayes_u_over_crlb'):.3f} |")
    w('')


def curve(T, k='ratio'):
    b = [kk for kk in T if 'x' in T[kk] and 'y' in T[kk]]
    lo = np.array([float(kk.split('-')[0]) for kk in b]); hi = np.array([float(kk.split('-')[1]) for kk in b])
    c = np.sqrt(lo * np.where(np.isfinite(hi), hi, lo * 2.5))
    return c, np.array([rms2(T[kk], k) for kk in b])


w('# CRLB comparison of the m1809 position estimates (gate scenes, seeds 301-303 pooled)\n')
w('Ratios are RMSD / CRLB_rms per GT-photon bin; RMSD includes the bias. Matching on the detector point, score >= 0.5: '
  'h3 = Hungarian 3 px (main), h1 = Hungarian 1 px (gate rule), n3 = nearest within 3 px. miss-pen = RMSD over all targets with '
  'each miss counted as 1 px per axis, over CRLB_rms of all targets. A1 = first-harmonic sub-pixel bias (pull to the pixel centre > 0). '
  'Bayes-cell = Gaussian-likelihood MMSE with a uniform 1-px cell prior (genie), over the CRLB.\n')
for p in (1, 2):
    for sc in SCEN:
        k = f'{sc}_p{p}'
        if k not in R: continue
        sm, sf = REF[sc]
        for est, ck in (('prior', sm), ('prior', 'fr_xy'), ('prior', sf), ('solve', sf), ('solve_c0', sf), ('solve_tp', sf)):
            key = f'{est}|{ck}|h3'
            if key not in R[k] or not R[k][key]: continue
            if p == 1 and est == 'prior' and ck == sf: continue
            w(f'## {sc}, pass {p}: {est} vs CRLB {ck}\n')
            prec_rows(R[k][key], R[k].get(f'{est}|{ck}|h1'), R[k].get(f'{est}|{ck}|n3'))
    for sc in SCEN:
        k = f'{sc}_p{p}'
        if k not in R or not R[k]['gap']: continue
        w(f'## Solve-gap decomposition, {sc}, pass {p} (h3 rows, emitters on in all 6 frames; nm = rms over x, y)\n')
        w('| GT photons | n | SF CRLB | two-stage ideal | solve | solve c=0 | solve true patt | solve trim5 / SF trim5 | GLS net inputs, block-diag | GLS joint cov | '
          'GLS solve weights | GLS solve weights c=0 | I_t RMSE / CRLB_fr | I_t z_rms | prior-I rho rms / max | R2(eI ~ eprior) |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for b, g in R[k]['gap'].items():
            w(f"| {b} | {g['n']} | {g['sf']:.2f} | {g['ts_full']:.2f} | {g['solve_rmsd']:.2f} | {g['solve_c0_rmsd']:.2f} | {g['solve_tp_rmsd']:.2f} | "
              f"{g['solve_trim5'] / g['sf_trim5']:.3f} | {g['gls_blockdiag']:.2f} | {g['gls_joint']:.2f} | {g['gls_solveweights']:.2f} | "
              f"{g['gls_solveweights_c0']:.2f} | {g['I_eff']:.3f} | {g['zI_rms']:.3f} | {g['rho_rms']:.3f} / {g['rho_max']:.3f} | {g['R2_I_on_prior']:.3f} |")
        w('')
open(f'{OUT}/tables.md', 'w').write('\n'.join(L))

# ---- figures
fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for a, sc in zip(axs, SCEN):
    sm, sf = REF[sc]
    for i, (k, key, lab) in enumerate(((f'{sc}_p1', f'prior|{sm}|h3', 'P1 prior / CRLB_SM (flat illum.)'),
                                       (f'{sc}_p1', 'prior|fr_xy|h3', 'P1 prior / CRLB pattern-free (modulated data)'),
                                       (f'{sc}_p1', f'prior|{sm}|h1', 'P1 prior / CRLB_SM, 1-px match'),
                                       (f'{sc}_p1', f'prior|{sm}|h3', 'Bayes cell genie / CRLB_SM'))):
        if k not in R or not R[k].get(key): continue
        c, r = curve(R[k][key], 'bayes_u_over_crlb' if lab.startswith('Bayes') else 'ratio')
        if not np.isfinite(r).any(): continue
        a.plot(c, r, '-o' if not lab.startswith('Bayes') else ':', color=COL[i], lw=2, ms=4, label=lab)
    a.axhline(1, color='#888', lw=1); a.set_xscale('log'); a.set_title(f'{sc} ({sm})'); a.set_xlabel('GT window photons')
    a.grid(alpha=0.3); a.xaxis.set_minor_formatter(_nf)
axs[0].set_ylabel('RMSD / CRLB'); axs[0].legend(fontsize=7); axs[0].set_ylim(0.3, 1.8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_prior.png', dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
for a, sc in zip(axs, SCEN):
    sm, sf = REF[sc]
    for i, (k, key, lab) in enumerate(((f'{sc}_p1', f'solve|{sf}|h3', 'P1 solve'), (f'{sc}_p1', f'solve_c0|{sf}|h3', 'P1 solve, c = 0'),
                                       (f'{sc}_p2', f'solve|{sf}|h3', 'P2 solve'), (f'{sc}_p2', f'solve_c0|{sf}|h3', 'P2 solve, c = 0'),
                                       (f'{sc}_p2', f'prior|{sf}|h3', 'P2 prior'))):
        if k not in R or not R[k].get(key): continue
        c, r = curve(R[k][key])
        a.plot(c, r, '-o', color=COL[i], lw=2, ms=4, label=lab, ls='--' if 'c = 0' in lab else '-')
    a.axhline(1, color='#888', lw=1); a.set_xscale('log'); a.set_title(f'{sc} vs CRLB {sf}'); a.set_xlabel('GT window photons')
    a.grid(alpha=0.3); a.xaxis.set_minor_formatter(_nf)
axs[0].set_ylabel('RMSD / CRLB_SF'); axs[0].legend(fontsize=7); axs[0].set_ylim(0.8, 3.0)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_solve.png', dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, k in zip(axs, ('iso_p1', 'iso_p2')):
    if k not in R: continue
    G = R[k]['gap']; b = list(G)
    lo = np.array([float(x.split('-')[0]) for x in b]); hi = np.array([float(x.split('-')[1]) for x in b])
    c = np.sqrt(lo * np.where(np.isfinite(hi), hi, lo * 2.5))
    for i, (kk, lab) in enumerate((('solve_rmsd', 'solve (actual)'), ('solve_c0_rmsd', 'solve, c = 0 (actual)'),
                                   ('gls_solveweights', 'linear GLS, solve weights (predicted)'),
                                   ('gls_blockdiag', 'linear GLS, ideal weights of the net inputs'),
                                   ('gls_joint', '+ prior-intensity cross-covariance'))):
        a.plot(c, [G[x][kk] / G[x]['sf'] for x in b], '-o', color=COL[i], lw=2, ms=4, label=lab)
    a.axhline(1, color='#888', lw=1); a.set_xscale('log'); a.set_title(f'{k}: RMSD / CRLB_SF (iso)'); a.set_xlabel('GT window photons')
    a.grid(alpha=0.3); a.xaxis.set_minor_formatter(_nf)
axs[0].legend(fontsize=7); axs[0].set_ylim(0.6, 2.6)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_gap_iso.png', dpi=130); plt.close(fig)
print('\n'.join(L[:5]))
