"""<out>/plots/frc_vs_N.png (final version; usage: python frc_plot.py [out.png]): FRC (300-frame halves, own DME v2) vs N for ours disjoint and SIMCODE (b) (+ published): random subsample
(<out>/critic34/frc_commonN.json from frc_commonN.py, mean ± sd over seeds) and best-N by each set's own precision estimate (solid vs dashed)."""
import sys, json, numpy as np
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frc_common as CM
from simcode import config
OUTPNG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.out_dir('plots'), 'frc_vs_N.png')
JSON = os.path.join(config.out_dir('critic34'), 'frc_commonN.json')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
J = json.load(open(JSON))
LBL = {'ours_disj': 'ours disjoint', 'sc_b': 'SIMCODE (b) sign fixed', 'sc_paper_h5': 'SIMCODE published'}
col = {'ours_disj': 'k', 'sc_b': 'tab:blue', 'sc_paper_h5': 'tab:red'}
best = {}
for c in ('0.5', '2.5'):
    for n in ('ours_disj', 'sc_b'):
        x, y, f, idx = CM.corrected(c, n); pos, fr, crlb, ex = CM.load_set(c, n)
        o = np.argsort(np.hypot(crlb[idx, 0], crlb[idx, 1]))
        Ns = [N for N in (125000, 250000, 500000, 1000000, 2000000) if N < len(x)]
        best[(c, n)] = (Ns, [CM.SM.frc_split(x[o[:N]], y[o[:N]], f[o[:N]]) for N in Ns])
        print(c, n, 'best-N', dict(zip(Ns, np.round(best[(c, n)][1], 2))), flush=True)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, c in zip(axs, ('0.5', '2.5')):
    for n, cu in J[c]['curve'].items():
        ax.errorbar(cu['N'], cu['m'], yerr=cu['s'], color=col[n], marker='o', ms=4, capsize=2, label=f'{LBL[n]}, random N')
    for n in ('ours_disj', 'sc_b'):
        Ns, v = best[(c, n)]
        ax.plot(Ns, v, color=col[n], ls='--', marker='^', ms=4, label=f'{LBL[n]}, best N (own precision)')
    ax.axvline(J[c]['Nc'], color='0.6', ls=':', lw=1)
    ax.set_xscale('log'); ticks = [1.25e5, 2.5e5, 5e5, 1e6, 2e6, 4e6]
    ax.set_xticks(ticks); ax.set_xticklabels(['125k', '250k', '500k', '1M', '2M', '4M']); ax.minorticks_off()
    ax.set_xlim(1e5, 4.5e6 if c == '2.5' else 1.3e6)
    ax.set_xlabel('localizations N (last point of each random curve = full set; dotted = common N)'); ax.set_ylabel('FRC resolution (nm, 1/7)')
    ax.set_title(f'{c} nM: FRC (300-frame halves) vs N, own DME v2'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPNG, dpi=90)
