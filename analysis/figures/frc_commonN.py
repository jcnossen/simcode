"""Critic 34 #2: FRC (300-frame alternating halves, 1/7, 5 nm px, crop 5-115 px) after each set's own DME v2, with every set
randomly subsampled to a common N (several seeds). Plus FRC vs N for ours disjoint and SIMCODE (b). Output: <out>/critic34/frc_commonN.json and a first
version of <out>/plots/frc_vs_N.png (or [out.png]; frc_plot.py makes the final one).  CPU only."""
import sys, numpy as np, json
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frc_common as CM
from simcode import config
OUTPNG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.out_dir('plots'), 'frc_vs_N.png')
JSON = os.path.join(config.out_dir('critic34'), 'frc_commonN.json')
SM = CM.SM
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SETS = ['sc_paper_h5', 'sc_a', 'sc_b', 'ours_disj', 'ours_ovl']
LBL = {'sc_paper_h5': 'SIMCODE published', 'sc_a': 'SIMCODE (a) paper sign', 'sc_b': 'SIMCODE (b) sign fixed',
       'ours_disj': 'ours disjoint', 'ours_ovl': 'ours stride-3'}
SEEDS = 5
NLOW = {'0.5': 250000, '2.5': 500000}
CURVE = [125000, 250000, 500000, 1000000, 2000000]
out = {}
for c in ('0.5', '2.5'):
    D = {n: CM.corrected(c, n)[:3] for n in SETS}
    Ns = {n: len(D[n][0]) for n in SETS}
    Nc = min(Ns.values())
    print(f'\n### {c} nM  N per set (crop): ' + ', '.join(f'{n} {Ns[n]}' for n in SETS) + f'  -> common N {Nc}, low N {NLOW[c]}', flush=True)
    res = {}
    def fr_sub(n, N, s):
        x, y, f = D[n]
        if N >= len(x): return SM.frc_split(x, y, f)
        i = np.random.default_rng(1000 + s).choice(len(x), N, replace=False)
        return SM.frc_split(x[i], y[i], f[i])
    print(f'| set | N | FRC full N | FRC @ N={Nc} (mean ± sd, min–max, {SEEDS} seeds) | FRC @ N={NLOW[c]} |')
    for n in SETS:
        full = fr_sub(n, 10 ** 12, 0)
        a = np.array([fr_sub(n, Nc, s) for s in range(SEEDS)]) if Ns[n] > Nc else np.array([full])
        b = np.array([fr_sub(n, NLOW[c], s) for s in range(SEEDS)])
        res[n] = dict(N=Ns[n], full=full, common=a.tolist(), low=b.tolist())
        print(f'| {LBL[n]} | {Ns[n]} | {full:.1f} | {a.mean():.2f} ± {a.std():.2f} ({a.min():.1f}–{a.max():.1f}) | {b.mean():.2f} ± {b.std():.2f} ({b.min():.1f}–{b.max():.1f}) |', flush=True)
    cur = {}
    for n in ('ours_disj', 'sc_b', 'sc_paper_h5'):
        pts = [N for N in CURVE if N < Ns[n]]
        v = [np.array([fr_sub(n, N, s) for s in range(3)]) for N in pts] + [np.array([res[n]['full']])]
        cur[n] = dict(N=pts + [Ns[n]], m=[a.mean() for a in v], s=[a.std() for a in v])
        print(f'  FRC vs N {LBL[n]}: ' + ', '.join(f'{N}: {m:.2f}±{s:.2f}' for N, m, s in zip(cur[n]['N'], cur[n]['m'], cur[n]['s'])), flush=True)
    out[c] = dict(Nc=Nc, res=res, curve=cur)
json.dump(out, open(JSON, 'w'), indent=1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
col = {'ours_disj': 'k', 'sc_b': 'tab:blue', 'sc_paper_h5': 'tab:red'}
for ax, c in zip(axs, ('0.5', '2.5')):
    for n, cu in out[c]['curve'].items():
        ax.errorbar(cu['N'], cu['m'], yerr=cu['s'], color=col[n], marker='o', ms=4, capsize=2, label=LBL[n])
    ax.axvline(out[c]['Nc'], color='0.6', ls=':', lw=1); ax.text(out[c]['Nc'], ax.get_ylim()[1], ' common N', color='0.4', fontsize=8, va='top')
    ax.set_xscale('log'); ax.set_xlabel('localizations N (random subsample; last point = full set)'); ax.set_ylabel('FRC resolution (nm, 1/7)')
    ax.set_title(f'{c} nM: FRC (300-frame halves) vs N, own DME v2'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPNG, dpi=90)
