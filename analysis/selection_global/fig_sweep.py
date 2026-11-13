"""SI figure: global precision threshold sweep (sweep_real.json, sweep_sim.json, sweep_sim_moving_<run>.json).
Rule on the x axis: rows with a rank (p >= 0.63, in the crop / field) and calibrated max(sigma_x, sigma_y) <= sigma_max.
Everything is shown relative to the earlier per-tile rule u <= 0.45 on the same base rows; the chosen value is the vertical line.
  a  real movies: fair FRC (one drift per movie, see sweep_real.py); dashed = u <= 0.45
  b  real movies: FRC-set count / u <= 0.45 count
  c  real movies: occupied 20-nm bins / u <= 0.45 (coverage)
  d  moving-microtubule sims: mean per-block Jaccard 20 nm; markers = u <= 0.45
  e-h simulations with GT (3 conditions x 4 densities): FRC difference, Jaccard difference (10 nm, preprint definition),
      GT RMS ratio, count ratio, each vs u <= 0.45
usage: python analysis/selection_global/fig_sweep.py [--chosen 12] -> out/selection_global/fig_sweep.{png,pdf,svg}"""
import os, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); OUT = f'{REPO}/out/selection_global'
C3 = ['#2a78d6', '#eb6834', '#1baf7a']            # sims: categorical slots 1-3 (validated default palette)
CR = ['#e87ba4', '#008300', '#6250d6']            # real movies: slots 5-7 (other entities, other hues)
LS = ['-', '--', '-.', ':']
INK = '#52514e'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--chosen', type=float, default=12.0); a = ap.parse_args()
    R = json.load(open(f'{OUT}/sweep_real.json')); S = json.load(open(f'{OUT}/sweep_sim.json'))
    import glob
    Mv = {}
    for f in sorted(glob.glob(f'{OUT}/sweep_sim_moving_*.json')): Mv.update(json.load(open(f)))
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': INK, 'ytick.color': INK, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.6))
    tk = lambda r: sorted(float(k.split('<=')[1]) for k in r if k.startswith('max<='))
    # ---- real
    names = {'1809_2': '1809/2 (low density)', '2.5nM_2': '2.5 nM/2', '20nM_2': '20 nM/2'}
    for i, (ds, lab) in enumerate(names.items()):
        r = R[ds]; t = tk(r); u = r['u045']; g = lambda k: np.array([r[f'max<={v:g}'][k] for v in t])
        ax[0, 0].plot(t, g('frc_nm'), color=CR[i], lw=2, label=lab); ax[0, 0].axhline(u['frc_nm'], color=CR[i], ls='--', lw=1)
        ax[0, 1].plot(t, g('n_frc') / u['n_frc'], color=CR[i], lw=2, label=lab)
        ax[0, 2].plot(t, g('bins20') / u['bins20'], color=CR[i], lw=2, label=lab)
        m = r['match_max']; ax[0, 0].plot(m['thr_nm'], m['frc_nm'], 'o', ms=6, color=CR[i], mec='white', mew=1.5)
    ax[0, 0].set_ylabel('FRC resolution (nm)'); ax[0, 0].set_title('a  real movies: fair FRC\n(dashed: u ≤ 0.45; dot: same count as u ≤ 0.45)', loc='left', fontsize=9)
    ax[0, 1].set_ylabel('count / count(u ≤ 0.45)'); ax[0, 1].set_title('b  real movies: localizations in the FRC set', loc='left', fontsize=9)
    ax[0, 2].set_ylabel('occupied 20-nm bins / u ≤ 0.45'); ax[0, 2].set_title('c  real movies: coverage', loc='left', fontsize=9)
    for k in (1, 2): ax[0, k].axhline(1, color=INK, lw=0.8, ls=':')
    ax[0, 0].legend(fontsize=8, frameon=False)
    # ---- moving
    mlab = {'moving': 'preprint photons', 'moving_exp': 'experimental photons', 'moving_gauss': 'Gaussian PSF (g13)'}
    for i, (run, lab) in enumerate(mlab.items()):
        if run not in Mv: continue
        r = Mv[run]; t = tk(r)
        ax[0, 3].plot(t, [r[f'max<={v:g}']['jaccard_mean'][1] for v in t], color=C3[i], lw=2, marker='s', ms=4, label=lab)
        ax[0, 3].plot([16.6], [r['u045']['jaccard_mean'][1]], marker='<', ms=8, color=C3[i], clip_on=False)
    ax[0, 3].set_ylabel('mean per-block Jaccard (20 nm)'); ax[0, 3].set_ylim(0, 1)
    ax[0, 3].set_title('d  moving-microtubule sims\n(triangles at the right: u ≤ 0.45)', loc='left', fontsize=9); ax[0, 3].legend(fontsize=8, frameon=False)
    # ---- density sims
    conds = [('D', '', 'measured PSF, preprint photons'), ('Dexp', '', 'measured PSF, experimental photons'), ('D', '_gauss', 'Gaussian 1.3 px PSF (g13)')]
    for i, (p, s, lab) in enumerate(conds):
        for j, d in enumerate(('0.5', '1', '2', '4')):
            r = S.get(f'{p}{d}{s}')
            if r is None: continue
            t = tk(r); u = r['u045']; g = lambda k: np.array([r[f'max<={v:g}'][k] for v in t], float)
            kw = dict(color=C3[i], ls=LS[j], lw=1.6, label=f'{lab}' if j == 0 else None)
            frc = g('frc_nm'); few = g('n_frc') < 2000; frc[few] = np.nan; rms = g('rms_nm'); rms[few] = np.nan
            ax[1, 0].plot(t, frc - u['frc_nm'], **kw)
            ax[1, 1].plot(t, g('jaccard') - u['jaccard'], **kw)
            ax[1, 2].plot(t, rms / u['rms_nm'], **kw)
            ax[1, 3].plot(t, g('n_all') / u['n_all'], **kw)
    ax[1, 0].set_ylim(-3, 5); ax[1, 0].set_ylabel('FRC − FRC(u ≤ 0.45) (nm)'); ax[1, 0].set_title('e  sims: FRC (lower is better; sets < 2000 rows hidden)', loc='left', fontsize=9)
    ax[1, 1].set_ylim(-0.25, 0.2); ax[1, 1].set_ylabel('Jaccard − Jaccard(u ≤ 0.45)'); ax[1, 1].set_title('f  sims: Jaccard 10 nm (higher is better)', loc='left', fontsize=9)
    ax[1, 2].set_ylim(0.7, 1.6); ax[1, 2].set_ylabel('RMS error vs GT / u ≤ 0.45'); ax[1, 2].set_title('g  sims: RMS error vs GT (sets < 2000 rows hidden)', loc='left', fontsize=9)
    ax[1, 3].set_yscale('log'); ax[1, 3].set_ylim(0.003, 3); ax[1, 3].set_ylabel('count / count(u ≤ 0.45)'); ax[1, 3].set_title('h  sims: localizations', loc='left', fontsize=9)
    for k in range(4): ax[1, k].axhline(0 if k < 2 else 1, color=INK, lw=0.8, ls=':')
    h = ax[1, 3].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    dh = [Line2D([], [], color=INK, ls=LS[j], lw=1.6, label=f'density {d}') for j, d in enumerate(('0.5', '1', '2', '4'))]
    ax[1, 3].legend(h[0] + dh, h[1] + [x.get_label() for x in dh], fontsize=7, frameon=False, loc='lower right')
    for A in ax.flat:
        A.axvline(a.chosen, color='#0b0b0b', lw=1, alpha=0.5); A.set_xlim(5, 16.8); A.grid(axis='y', color='#e5e4e0', lw=0.6)
    for A in ax[1]: A.set_xlabel(r'$\sigma_{max}$: max($\sigma_x$, $\sigma_y$) ≤ $\sigma_{max}$ (nm)')
    for A in ax[0]: A.set_xlabel(r'$\sigma_{max}$ (nm)')
    fig.suptitle(f'Global precision threshold (calibrated σ, rows with p ≥ 0.63) vs the per-tile rule u ≤ 0.45; vertical line: $\\sigma_{{max}}$ = {a.chosen:g} nm',
                 fontsize=10, x=0.01, ha='left')
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'): plt.savefig(f'{OUT}/fig_sweep.{ext}', dpi=160)
    print('written', f'{OUT}/fig_sweep.png')


if __name__ == '__main__':
    main()
