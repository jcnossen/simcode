"""Covariance ablation, part 1 figure (from cov_desc.json) -> out/ablation_cov/fig_covariance.png"""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'out', 'ablation_cov')
D = json.load(open(f'{OUT}/cov_desc.json'))
BLUE, ORANGE, AQUA, INK, MUTED, GRID, SURF = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
seq = LinearSegmentedColormap.from_list('b', ['#fcfcfb', '#cde2fb', '#86b6ef', '#2a78d6', '#0d366b'])
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.titlesize': 9, 'axes.titlecolor': INK})
fig, ax = plt.subplots(2, 4, figsize=(16.5, 8.2), gridspec_kw=dict(wspace=0.32, hspace=0.38)); HB = np.linspace(-1, 1, 81); hc = 0.5 * (HB[1:] + HB[:-1])
LAB = ['a1 s1', 'a1 s2', 'a1 s3', 'a2 s1', 'a2 s2', 'a2 s3']


def mat(a, M, title):
    M = np.array(M, float).copy(); np.fill_diagonal(M, np.nan)
    im = a.imshow(M, cmap=seq, vmin=0, vmax=0.45)
    for i in range(6):
        for j in range(6):
            if i != j: a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7, color='#ffffff' if M[i, j] > 0.27 else INK)
    a.set_xticks(range(6)); a.set_yticks(range(6)); a.set_xticklabels(LAB, fontsize=7); a.set_yticklabels(LAB, fontsize=7)
    a.axhline(2.5, color=SURF, lw=2); a.axvline(2.5, color=SURF, lw=2); a.set_title(title, loc='left')
    for s in a.spines.values(): s.set_visible(False)
    a.tick_params(length=0)
    return im


g = D['gate:rs05qm:s2']['all|200-1e+09']
mat(ax[0, 0], g['Rpred_ok'], f'Sim, realistic sparse: predicted\ncorrelation (n {g["n"]}, pass 2)')
mat(ax[0, 1], g['Remp'], 'Same emitters: correlation of the\nactual intensity errors')
g = D['gate:rs25qm:s2']['all|200-1e+09']
mat(ax[0, 2], g['Remp'], f'Sim, realistic dense: actual errors\n(n {g["n"]}; predicted mean {(6*g["within"]+9*g["cross"])/15:.2f})')
r = D['real:1809_2:s2']['all|200-1e+09']
im = mat(ax[0, 3], r['Rpred'], f'Real 1809_2: predicted correlation\n(n {r["n"]}, pass 2)')
cb = fig.colorbar(im, ax=ax[0, :], fraction=0.012, pad=0.01); cb.set_label('correlation coefficient'); cb.outline.set_visible(False)

for a, key, name in ((ax[1, 0], 'real:1809_2:s2', 'Real 1809_2'), (ax[1, 1], 'gate:rs25qm:s2', 'Sim, realistic dense')):
    for sp, col, lab in (('iso', BLUE, 'isolated'), ('nbr', ORANGE, 'neighbour within 3 px')):
        q = D[key][f'{sp}|200-1e+09']; h = np.array(q['hist_within'], float) + np.array(q['hist_cross'], float); h /= h.sum() * (HB[1] - HB[0])
        a.plot(hc, h, color=col, lw=2, label=f'{lab} (mean {(6 * q["within"] + 9 * q["cross"]) / 15:+.2f})')
    a.axvline(0, color=GRID, lw=1, zorder=0); a.set_xlim(-0.5, 0.9); a.set_ylim(0, a.get_ylim()[1] * 1.3); a.set_xlabel('predicted off-diagonal correlation coefficient'); a.set_ylabel('density')
    a.set_title(f'{name}:\nall 15 off-diagonal coefficients per emitter', loc='left'); a.legend(frameon=False, fontsize=8, loc='upper left'); a.grid(axis='y', color=GRID, lw=0.6)

a = ax[1, 2]; bins = ['200-700', '700-2000', '2000-1e+09']; xb = np.arange(3)
m15 = lambda q, p='': (6 * q[p + 'within'] + 9 * q[p + 'cross']) / 15
ser = [('Sim sparse, predicted', 'gate:rs05qm:s2', '', ORANGE, 's', '-'),
       ('Sim sparse, actual errors', 'gate:rs05qm:s2', 'emp_', ORANGE, 's', '--'), ('Sim isolated (exact model), predicted', 'gate:iso:s2', '', AQUA, '^', '-'),
       ('Sim isolated, actual errors', 'gate:iso:s2', 'emp_', AQUA, '^', '--'), ('Real, predicted', 'real:1809_2:s2', '', BLUE, 'o', '-')]
for lab, key, p, col, mk, ls in ser:
    y = [m15(D[key][f'all|{b}'], p) if f'all|{b}' in D[key] else np.nan for b in bins]
    a.plot(xb, y, color=col, marker=mk, ms=7, lw=2, ls=ls, label=lab, markeredgecolor=SURF, markeredgewidth=1.5)
a.set_xticks(xb); a.set_xticklabels(['200-700', '700-2000', '> 2000']); a.set_xlabel('window photons (sim: true; real: network)'); a.set_ylabel('mean off-diagonal correlation')
a.set_ylim(0, 0.62); a.grid(axis='y', color=GRID, lw=0.6); a.legend(frameon=False, fontsize=7.5, loc='upper left'); a.set_title('Correlation grows with brightness\n(solid: predicted, dashed: actual errors)', loc='left')

a = ax[1, 3]
for fr, col, lab in (('s2', BLUE, 'pass 2'), ('s1', ORANGE, 'pass 1')):
    X, Y = [], []
    for k, G in D.items():
        if not k.startswith('gate:') or not k.endswith(fr): continue
        for kk, q in G.items():
            if kk.startswith('all|') or '200-1e+09' in kk: continue
            X += [q['pred_within_ok'], q['pred_cross_ok']]; Y += [q['emp_within'], q['emp_cross']]
    a.scatter(X, Y, s=34, color=col, edgecolor=SURF, linewidth=1.2, label=f'{lab} (r = {np.corrcoef(X, Y)[0, 1]:.2f})', zorder=3)
a.plot([-0.1, 0.6], [-0.1, 0.6], color=MUTED, lw=1, zorder=1); a.set_xlim(-0.1, 0.6); a.set_ylim(-0.1, 0.6); a.set_aspect('equal')
a.set_xlabel('predicted mean correlation'); a.set_ylabel('correlation of the actual errors'); a.grid(color=GRID, lw=0.6); a.legend(frameon=False, fontsize=8, loc='upper left')
a.set_title('Predicted vs actual, per scene / isolation /\nphoton bin (within- and cross-angle means)', loc='left')
fig.savefig(f'{OUT}/fig_covariance.png', dpi=140, bbox_inches='tight'); print('saved')
