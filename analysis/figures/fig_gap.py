"""Summary figure of the real-data per-localization gap analysis (values copied from the development realgap/*.txt analyses
of 2026-11-10, 'real-data per-localization gap': simproxy_m05qh / m25qh, decomp_*, paired_*, frc_matched_*, selection_*,
simreal.txt; the numbers are hard-coded below).  -> <out>/plots/realgap_summary.png
usage: python fig_gap.py [out.png]"""
import os, sys
if any(a in ('-h', '--help') for a in sys.argv[1:]): print(__doc__); sys.exit()
import numpy as np
from simcode import config
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'legend.frameon': False})
fig, ax = plt.subplots(2, 3, figsize=(14, 8.2))
# (a, b) sim: TRUE sigma vs NN proxy (simproxy_m05qh / m25qh.txt; label-photon bins)
SIM = {'m05qh': dict(bins=['200-700', '700-2k', '2k-4k', '>4k'], ot=[12.55, 8.00, 5.45, 4.30], op=[9.80, 6.35, 4.20, 3.15],
                     st=[13.90, 9.25, 7.90, 7.55], sp=[8.30, 5.95, 4.55, 3.50]),
       'm25qh': dict(bins=['700-2k', '2k-4k', '>4k'], ot=[10.50, 7.55, 5.75], op=[8.05, 5.70, 4.25], st=[11.65, 9.80, 8.85], sp=[7.15, 5.55, 4.70])}
for a, (m, D) in zip(ax[0, :2], SIM.items()):
    x = np.arange(len(D['bins']))
    a.plot(x, D['ot'], '-o', color=BLUE, lw=2, ms=7, label='ours, true error')
    a.plot(x, D['op'], '--o', color=BLUE, lw=2, ms=7, mfc='white', label='ours, NN proxy')
    a.plot(x, D['st'], '-s', color=ORANGE, lw=2, ms=7, label='SIMCODE, true error')
    a.plot(x, D['sp'], '--s', color=ORANGE, lw=2, ms=7, mfc='white', label='SIMCODE, NN proxy')
    a.set_xticks(x, D['bins']); a.set_xlabel('label photons per window'); a.set_ylabel('sigma per axis (nm)')
    a.set_title(f'({"ab"[m == "m25qh"]}) sim {m}: proxy hides SIMCODE\'s correlated error', loc='left', fontsize=10)
    a.set_ylim(0, 15)
ax[0, 0].legend(fontsize=8, loc='upper right')
ax[0, 0].text(0.02, 0.04, 'corr(e_f, e_f+6) same event: ours 0.24-0.39, SIMCODE 0.50-0.65', transform=ax[0, 0].transAxes, fontsize=7.5, color=MUTED)
ax[0, 1].text(0.02, 0.04, 'corr: ours 0.25-0.45, SIMCODE 0.53-0.66', transform=ax[0, 1].transAxes, fontsize=7.5, color=MUTED)
# (c) ratio ours / SIMCODE on identical emitter-window pairs: pattern-free vs final (decomp_*.txt)
lab = ['m05qh\ntrue', 'm05qh\nproxy', 'm25qh\ntrue', 'm25qh\nproxy', 'real 0.5\nproxy', 'real 2.5\nproxy']
prior = [8.20 / 9.10, 7.60 / 7.45, 8.05 / 9.25, 7.25 / 7.10, 9.75 / 8.75, 10.80 / 9.10]
final = [5.85 / 7.40, 4.55 / 4.95, 6.35 / 8.45, 4.85 / 5.45, 5.05 / 5.35, 6.35 / 6.45]
x = np.arange(len(lab)); w = 0.38
a = ax[0, 2]
a.bar(x - w / 2 - 0.01, prior, w, color='#86b6ef', label='pattern-free (ours prior / SIMCODE net)')
a.bar(x + w / 2 + 0.01, final, w, color=BLUE, label='final (ours / SIMCODE NDI)')
a.axhline(1, color=INK, lw=1)
a.set_xticks(x, lab, fontsize=7.5); a.set_ylabel('sigma ratio ours / SIMCODE (same rows)'); a.set_ylim(0.6, 1.3)
a.set_title('(c) identical emitter-window pairs: < 1 = ours better', loc='left', fontsize=10); a.legend(fontsize=7.5, loc='upper left')
# (d) real doubly-shared per SC-photon bin (paired_*.txt section C, 1-D mean of x/y)
R = {'0.5 nM': dict(b=['300-500', '500-800', '800-1.3k', '1.3k-2.5k'], o=[6.90, 5.375, 4.35, 3.70], s=[7.375, 5.75, 4.575, 3.85]),
     '2.5 nM': dict(b=['300-500', '500-800', '800-1.3k', '1.3k-2.5k', '>2.5k'], o=[9.375, 7.675, 6.55, 5.50, 4.525], s=[9.475, 8.00, 6.725, 5.60, 4.70])}
a = ax[1, 0]
for k, D in R.items():
    x = np.arange(len(D['b'])); ls = '-' if k == '0.5 nM' else ':'
    a.plot(x, D['o'], ls, marker='o', color=BLUE, lw=2, ms=7, label=f'ours {k}')
    a.plot(x, D['s'], ls, marker='s', color=ORANGE, lw=2, ms=7, label=f'SIMCODE {k}')
a.set_xticks(np.arange(5), R['2.5 nM']['b']); a.set_xlabel('SIMCODE photons per window'); a.set_ylabel('NN-proxy sigma per axis (nm)')
a.set_title('(d) real, same emitter-windows (f, f+6 both shared)', loc='left', fontsize=10); a.legend(fontsize=7.5)
# (e) FRC at equal N (paired_*.txt D, frc_matched_*.txt, selection_*.txt)
cats = ['random rows\n(critic 34)', 'photon-matched\nto SIMCODE', 'ours dnll>3,\nphoton-matched', 'shared rows\n(same rows)', 'unique rows\n(ours-only vs SC-only)']
F = {'0.5 nM': dict(o=[20.34, 17.63, 17.43, 18.20, 25.97], s=[18.25, 19.66, 19.58, 19.10, 22.84]),
     '2.5 nM': dict(o=[28.93, 24.46, 23.82, 22.84, 33.12], s=[24.94, 24.94, 24.94, 24.78, 29.73])}
a = ax[1, 1]
y = np.arange(len(cats))[::-1]
for k, D in F.items():
    off = 0.17 if k == '0.5 nM' else -0.17; mk = 'o' if k == '0.5 nM' else 'D'
    for yi, o, s in zip(y, D['o'], D['s']):
        a.plot([o, s], [yi + off] * 2, color=GRID, lw=3, zorder=1)
    a.scatter(D['o'], y + off, color=BLUE, marker=mk, s=45, zorder=3, label=f'ours {k}')
    a.scatter(D['s'], y + off, color=ORANGE, marker=mk, s=45, zorder=3, label=f'SIMCODE {k}')
a.set_yticks(y, cats, fontsize=8); a.set_xlabel('FRC resolution at equal N (nm, lower = better)', labelpad=2)
a.set_title('(e) FRC at equal N: the gap is the photon population', loc='left', fontsize=10); a.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4, handletextpad=0.2, columnspacing=0.8)
# (f) fringe ambiguity sim vs real (simreal.txt (1))
b = ['300-500', '500-800', '800-1.3k', '1.3k-2.5k']; x = np.arange(4)
a = ax[1, 2]
a.plot(x, [0.042, 0.010, 0.003, 0.001], '--o', color=BLUE, mfc='white', lw=2, ms=7, label='sim m05qh')
a.plot(x, [0.067, 0.023, 0.009, 0.005], '-o', color=BLUE, lw=2, ms=7, label='real 0.5 nM')
a.plot(x, [0.218, 0.078, 0.029, 0.012], '--s', color=MUTED, mfc='white', lw=2, ms=7, label='sim m25qh')
a.plot(x, [0.316, 0.201, 0.119, 0.058], '-s', color=MUTED, lw=2, ms=7, label='real 2.5 nM')
a.set_yscale('log'); a.set_xticks(x, b); a.set_xlabel('photons per window (SIMCODE units)'); a.set_ylabel('fraction of ours rows with dnll < 2')
a.set_title('(f) ours: fringe-ambiguous rows, real vs sim', loc='left', fontsize=10); a.legend(fontsize=7.5)
a.text(0.02, 0.05, 'median q_dof rises with photons on real\n(0.75->1.14, 0.99->1.32); sim flat 0.6-0.74', transform=a.transAxes, fontsize=7.5, color=MUTED)
plt.tight_layout()
plt.savefig(sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.out_dir('plots'), 'realgap_summary.png'), dpi=110)
