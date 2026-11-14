"""R24 figures from r24_rul.py results (plain matplotlib PNG; style and colours of nra_lib).
  python r24_fig.py dist <rul_*.json> <out.png> [--d0 20] [--title t]      distance histogram of ALL common pairs per method (resolved pairs darker)
  python r24_fig.py resolved <out.png> <label=rul.json> [...] [--methods a,b,..]   resolved fraction per method (95 % CI), one panel per pair list
  python r24_fig.py calib <rul_variants.json> <out.png> [--levels raw,cal] [--methods A,C,D,E]   pair z versus photons per variant
  python r24_fig.py gallery <pairs.json> <out.png> name=table.npz ... [--n 12 --box 120]    same structures for every method"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nra_lib as L
a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
plt = L._style(); C = L.COLORS; INK = '#0b0b0b'; MUTED = '#52514e'
LAB = dict(pub_smlm='published SMLM', pub_simflux='published SIMFLUX', bl_smlm='baseline SMLM', bl_simflux='baseline SIMFLUX', simcode_m12='SIMCODE v2, max sigma <= 12 nm', simcode_sel='SIMCODE v2, sel',
           simcode='SIMCODE v2 (matched events)')
if a[0] == 'dist':
    R = json.load(open(a[1])); d0 = float(g('--d0', R['args']['d0'])); Ms = R['methods']; n = len(Ms)
    fig, axs = plt.subplots(n, 1, figsize=(6.2, 1.55 * n + 0.8), sharex=True, squeeze=False); bins = np.arange(0, 2 * d0 + 0.01, d0 / 20)
    for i, (nm, r) in enumerate(Ms.items()):
        ax = axs[i, 0]; d = np.array(r['d_list']); res = np.array(r['resolved_list'], bool); e, c = r['est'], r['ci']
        ax.hist(np.clip(d, 0, 2 * d0 - 1e-6), bins=bins, color=C[i % len(C)], alpha=0.35, edgecolor='#fcfcfb', linewidth=0.5)
        ax.hist(d[res], bins=bins, color=C[i % len(C)], edgecolor='#fcfcfb', linewidth=0.5)
        ax.axvline(d0, color=MUTED, lw=1, ls='--')
        ax.set_title(f"{LAB.get(nm, nm)}: resolved {e['frac_resolved']:.2f} [{c['frac_resolved'][0]:.2f}, {c['frac_resolved'][1]:.2f}] of {e['n_pairs']} pairs; "
                     f"distance of the resolved {e['d_res_mean']:.1f} +- {e['d_res_sd']:.1f} nm", loc='left', fontsize=8.5, color=INK)
        ax.set_ylabel('site pairs')
    axs[-1, 0].set_xlabel(f'fitted site separation (nm); dark: resolved pairs, light: all pairs; dashed: {d0:g} nm')
    if g('--title'): fig.suptitle(g('--title'), x=0.01, ha='left', fontsize=10)
    fig.tight_layout(); fig.savefig(a[2], dpi=160)
elif a[0] == 'resolved':
    sets = [s.split('=', 1) for s in a[2:] if '=' in s and not s.startswith('--')]; fig, axs = plt.subplots(1, len(sets), figsize=(4.3 * len(sets), 3.4), sharey=True, squeeze=False)
    for j, (lab, fn) in enumerate(sets):
        R = json.load(open(fn))['methods']; ms = g('--methods').split(',') if g('--methods') else list(R); ax = axs[0, j]
        for i, nm in enumerate(ms):
            e, c = R[nm]['est'], R[nm]['ci']; v = e['frac_resolved']
            ax.barh(i, v, height=0.55, color=C[i % len(C)]); ax.plot(c['frac_resolved'], [i, i], color=INK, lw=1.5)
            ax.text(min(c['frac_resolved'][1] + 0.02, 0.9), i, f'{v:.2f}', va='center', fontsize=8.5, color=INK)
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([LAB.get(m, m) for m in ms] if j == 0 else []); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis='y', visible=False)
        ax.set_title(f"{lab} ({R[ms[0]]['est']['n_pairs']} pairs)", loc='left', fontsize=9); ax.set_xlabel('fraction of site pairs resolved (95 % CI over structures)')
    fig.tight_layout(); fig.savefig(a[1], dpi=160)
elif a[0] == 'calib':
    R = json.load(open(a[1]))['methods']; lv = g('--levels', 'raw,cal').split(','); ms = [m for m in g('--methods', 'A,B,C,D,E').split(',') if m in R]
    fig, axs = plt.subplots(1, len(lv), figsize=(4.8 * len(lv), 3.6), sharey=True, squeeze=False); nb = len(R[ms[0]]['edges']) - 1
    for j, l in enumerate(lv):
        ax = axs[0, j]; ax.axhline(1.0, color=MUTED, lw=1, ls=':')
        for i, m in enumerate(ms):
            e = R[m]['est']; ax.plot([e[f'bin{q}_phot'] for q in range(nb)], [e[f'{l}_bin{q}_pairz'] for q in range(nb)], color=C[i % len(C)], lw=2, marker='o', ms=5, label=m)
        ax.set_xlabel('photons per 6-frame window (bin median)'); ax.set_title(dict(raw='uncalibrated model sigma', cal='calibrated sigma', nb='without the photon-bin factor', f='Fisher sigma').get(l, l), loc='left', fontsize=9)
    axs[0, 0].set_ylabel('consecutive-cycle pair z (1 = sigma correct)'); axs[0, -1].legend(fontsize=8, title='variant', loc='upper left', bbox_to_anchor=(1.01, 1.0))
    if g('--title'): fig.suptitle(g('--title'), x=0.01, ha='left', fontsize=10)
    fig.tight_layout(); fig.savefig(a[2], dpi=160)
elif a[0] == 'gallery':
    from scipy.ndimage import gaussian_filter
    D = json.load(open(a[1])); tabs = [s.split('=', 1) for s in a[3:] if '=' in s and not s.startswith('--')]; n = int(g('--n', 12)); box = float(g('--box', 120)); px = 1.0
    sc = np.array(D['sites']); su = np.array(D['site_unit']); units = sorted({r['unit'] for r in D['rulers']}); rng = np.random.default_rng(1); pick = np.sort(rng.choice(units, min(n, len(units)), replace=False))
    fig, axs = plt.subplots(len(tabs), len(pick), figsize=(1.15 * len(pick), 1.15 * len(tabs) + 0.3), squeeze=False); nbx = int(box / px)
    for i, (nm, fn) in enumerate(tabs):
        T = L.load_table(fn); sh = L.global_shift(T['x_nm'], T['y_nm'], sc, rmax=15.0); x = T['x_nm'] - sh[0]; y = T['y_nm'] - sh[1]
        for j, u in enumerate(pick):
            ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); c = sc[su == u].mean(0)
            for s_ in ax.spines.values(): s_.set_visible(False)
            m = (abs(x - c[0]) < box / 2) & (abs(y - c[1]) < box / 2)
            h = gaussian_filter(np.histogram2d(y[m], x[m], bins=nbx, range=[[c[1] - box / 2, c[1] + box / 2], [c[0] - box / 2, c[0] + box / 2]])[0], 1.5 / px)
            ax.imshow(h, cmap='inferno', vmin=0, vmax=max(h.max(), 1e-9), origin='lower', interpolation='nearest')
            if j == 0: ax.set_title(LAB.get(nm, nm), loc='left', fontsize=8); ax.plot([6, 6 + 20 / px], [6, 6], color='w', lw=2)
    fig.suptitle(f'same structures for every method; {box:g} nm boxes, 1 nm bins blurred 1.5 nm, each panel scaled to its maximum; scale bar 20 nm', x=0.01, ha='left', fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05, hspace=0.25); fig.savefig(a[2], dpi=170)
print('saved', a[2] if a[0] != 'resolved' else a[1])
