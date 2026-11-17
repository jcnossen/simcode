"""sigma study (2026-11-17): figures of evalsim.py results (held-out simulated movies), one PNG per model: per pass, rms(error /
sigma) (|z| < 4) per network-photon bin and per sigma-rank bin u, and the precision (robust error sd) relative to the production
until now, for the production sigma (with / without its g(u)) and the joint-covariance networks + solve (posterior sd).
usage: python plots.py <out dir> <evalsim json> [...]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COL = dict(id='#eb6834', id_g='#1baf7a', k1='#6250d6', joint='#2a78d6')
LAB = dict(id='production until now (kap + sigma factors)', id_g='production + g(u) (tables until now)', k1='old networks, no kap, no factors',
           joint='joint covariance (posterior sd)')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def xph(b):
    return np.sqrt(max(b['lo'], 100) * min(b['hi'], 8000))


def main():
    out, files = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    for f in files:
        R = json.load(open(f)); m = R['model']
        fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
        for i, ps in enumerate(('est', 'est2')):
            P = R['passes'][ps]
            for lab, C in P['calib'].items():
                ls = '--' if lab == 'id_g' else '-'
                ax[i, 0].plot([xph(b) for b in C['photons']], [b['z_core'] for b in C['photons']], marker='o', ls=ls, color=COL[lab], lw=2, ms=5, label=LAB[lab])
                ax[i, 1].plot([0.5 * (b['lo'] + b['hi']) for b in C['u']], [b['z_core'] for b in C['u']], marker='o', ls=ls, color=COL[lab], lw=2, ms=5)
            ref = {b['lo']: b['sd_rob'] for b in P['precision']['id']}
            for lab, Q in P['precision'].items():
                if lab == 'id': continue
                ax[i, 2].plot([xph(b) for b in Q], [b['sd_rob'] / ref[b['lo']] for b in Q], marker='o', color=COL[lab], lw=2, ms=5)
            ax[i, 2].axhline(1, color=COL['id'], lw=2, zorder=0)
            for j in (0, 2):
                a = ax[i, j]; a.set_xscale('log'); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                a.set_xticks([200, 500, 1000, 2000, 5000]); a.set_xticklabels(['200', '500', '1000', '2000', '5000']); a.set_xlabel('network window photons')
            ax[i, 1].set_xlabel('sigma rank u (per 10 px tile)')
            for a in ax[i, :2]: a.axhline(1, color=MUTED, lw=1, zorder=0)
            for a in ax[i]:
                a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
                for s in ('top', 'right'): a.spines[s].set_visible(False)
            ax[i, 0].set_ylabel(f'pass {1 if ps == "est" else 2}: rms(error / sigma), |z| < 4')
            ax[i, 2].set_ylabel('robust error sd / production')
        h, l = ax[0, 0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=8, loc='lower center', ncol=4)
        fig.suptitle(f'{m}: held-out simulated movies ({", ".join(R["sims"])})', color=INK, fontsize=10)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        p = os.path.join(out, f'zcal_{m}{"_" + "_".join(R["labels"]) if len(R["labels"]) > 2 else ""}.png'); fig.savefig(p, dpi=150); plt.close(fig); print('wrote', p)


if __name__ == '__main__':
    main()
