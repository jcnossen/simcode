"""Site-size study, figure: sigma_pair(Delta) (axis xy = along and across pooled) per method with bootstrap CIs and the fit, per movie.
  python ss_fig.py  -> out/nanoruler/sitesize/fig_sigma_pair.png"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/home/jc/simcode2/out/nanoruler/sitesize'
MAIN = [('pub_smlm_dme', 'published SMLM', '#2a78d6', '-'), ('bl_smlm_dme', 're-impl. SMLM', '#2a78d6', '--'),
        ('pub_simflux_dme', 'published SIMFLUX', '#eb6834', '-'), ('bl_simflux_dme', 're-impl. SIMFLUX', '#eb6834', '--'),
        ('pub_simflux_paper', 'published SIMFLUX, paper drift', '#eda100', ':'), ('pub_smlm_paper', 'published SMLM, paper drift', '#6250d6', ':'),
        ('simcode_m12_dme', 'SIMCODE v2 pass 1 (m12)', '#1baf7a', '-'), ('simcode_sel_dme', 'SIMCODE v2 pass 1 (sel)', '#1baf7a', '-'), ('simcode_m12_dme+inj', 'SIMCODE + injected 3 nm / event, 2 nm AR(5)', '#e34948', '--')]
PANELS = [('nr80', '80 nm rulers (real)'), ('nr40', '40 nm squares (real)'), ('nr_a', 'sim 80 nm nr_a (point sites)'), ('nr_c0', 'sim 80 nm nr_c0'),
          ('s40g_a', 'sim 40 nm grid s40g_a'), ('s40g_c0', 'sim 40 nm grid s40g_c0'), ('s20_a', 'sim 20 nm linear s20_a (10 nm radius)')]


def model(D, r):
    a = r['sigma_loc'][0] ** 2; b = r['s_slow'][0] ** 2; return np.sqrt(a + b * (1 - np.exp(-D / r['tau'][0])))


fig, axs = plt.subplots(2, 4, figsize=(21, 9.5), sharex=True); axs = axs.ravel()
for ax, (mv, title) in zip(axs, PANELS):
    fn = f'{OUT}/res_{mv}.json'
    if not os.path.exists(fn): ax.set_visible(False); continue
    R = json.load(open(fn)); Dd = np.logspace(0, np.log10(2000), 200)
    for k, (m, lab, c, ls) in enumerate(MAIN):
        if m not in R['methods']: continue
        a = R['methods'][m]['axes']['xy']; D = np.array(R['methods'][m]['deltas']); v = np.array([a['sigma_pair'][str(int(d))] for d in D])
        off = 1 + 0.02 * (k - 4)
        ax.errorbar(D * off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='o', ms=4, color=c, mfc='white' if ls != '-' else c, capsize=2, lw=1)
        ax.plot(Dd, model(Dd, a), ls, color=c, lw=1.6, label=f'{lab}: loc {a["sigma_loc"][0]:.1f}, slow {a["s_slow"][0]:.1f} (tau {a["tau"][0]:.0f}), s_ev {a["s_event"][0]:.1f}')
        ax.axhline(a['spread'][0], color=c, ls=ls, lw=0.7, alpha=0.5)
    ax.set_xscale('log'); ax.set_title(f'{title}, {R["n_common_sites"]} sites', fontsize=11); ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylabel('pair precision sigma_pair (nm)'); ax.legend(fontsize=7, loc='lower right', frameon=True, framealpha=0.85)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
for ax in axs[4:]: ax.set_xlabel('cycle gap Delta (6-frame cycles, 84 ms)')
axs[-1].set_visible(False)
fig.suptitle('Pair precision of localizations of one binding site vs cycle gap (DME v2 for every method; lines = fit sigma_loc^2 + s_slow^2 (1 - exp(-Delta/tau)); '
             'thin horizontal = median per-site spread)', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f'{OUT}/fig_sigma_pair.png', dpi=130); print('wrote', f'{OUT}/fig_sigma_pair.png')
