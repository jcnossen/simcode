"""Edge deficit: fraction of rows with |f| > 0.45 (uniform: 0.10) for the estimates, the matched GT and all GT events, and a
fine (20-bin) bias curve, to separate 'missing rows near pixel edges' (detection / dedup) from 'rows pushed off the edges'
(position bias).  usage: python analysis/pixbias/edge.py D1 Dexp1  -> out/pixbias/edge_<D>.json, fig_edge_<D>.png"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pblib as B, sim as S
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ef = lambda f: float(np.mean(np.abs(f) > 0.45) / 0.10)
for D in sys.argv[1:]:
    DS = S.papersim(D); gt = np.load(os.path.join(S.PS, D, 'gt.npz'))
    R = dict(gt_all_events=dict(x=ef(B.sub(gt['ev_x'])), y=ef(B.sub(gt['ev_y']))))
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    for i, ps in enumerate(('P1', 'P2')):
        G = DS[ps]; m0 = G['sel']['sig12']; m = m0 & np.isfinite(G['tx'])
        r = dict(gt_matched=dict(x=ef(B.sub(G['tx'][m])), y=ef(B.sub(G['ty'][m]))), match_frac=float(m.sum() / m0.sum()))
        for j, (en, (x, y)) in enumerate(G['est'].items()):
            r[en] = dict(est_all=dict(x=ef(B.sub(x[m0 & np.isfinite(x)]))), est_matched=dict(x=ef(B.sub(x[m]))))
            e = x[m] - G['tx'][m]; fg = B.sub(G['tx'][m]); k = np.abs(e) < 0.5
            ib = np.clip(((fg + 0.5) * 20).astype(int), 0, 19)
            mu = np.array([e[k & (ib == b)].mean() for b in range(20)]) * B.PX
            se = np.array([e[k & (ib == b)].std() / np.sqrt((k & (ib == b)).sum()) for b in range(20)]) * B.PX
            r[en]['bias20_nm'] = mu.round(2).tolist()
            fc = (np.arange(20) + 0.5) / 20 - 0.5
            axs[i, 0].plot(fc, mu, label=en); axs[i, 0].fill_between(fc, mu - 2 * se, mu + 2 * se, alpha=0.2)
            B.hist1d(axs[i, 1 + j], B.sub(x[m0 & np.isfinite(x)]), label='est (all sig12 rows)', color='C3')
            B.hist1d(axs[i, 1 + j], fg, label='GT of matched rows', color='k')
            axs[i, 1 + j].set_ylim(0.4, 1.6); axs[i, 1 + j].set_title(f'{D} {ps} {en} sig12', fontsize=8); axs[i, 1 + j].legend(fontsize=6)
        axs[i, 0].axhline(0, color='k', lw=0.5); axs[i, 0].set_title(f'{D} {ps} sig12: <est-GT> x vs GT f (20 bins)', fontsize=8); axs[i, 0].legend(fontsize=7)
        R[ps] = r
    fig.tight_layout(); fig.savefig(os.path.join(B.OUT, f'fig_edge_{D}.png'), dpi=90); plt.close(fig)
    json.dump(R, open(os.path.join(B.OUT, f'edge_{D}.json'), 'w'), indent=1)
    print(D, json.dumps(R), flush=True)
