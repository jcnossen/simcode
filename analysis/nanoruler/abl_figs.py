"""Nanoruler ablation: report figures and the clean ablation table (inputs: the json outputs of abl_nra.py / abl_sim.py and the method tables).

  gallery   fig_gallery.png: render gallery, the SAME rulers for every method (nra_lib.fig_gallery: 2.5 nm histogram, fixed 2.5 nm blur,
            each panel scaled to its own maximum, 80 nm scale bar): published SMLM, published SIMFLUX, baseline SIMFLUX, SIMCODE v2 pass 2 ('sel' rows)
  methods   fig_methods_photons.png: per method, versus photons (the method's own photon unit, 6 quantile bins, rulers valid in all
            methods): site-residual spread, consecutive-cycle pair precision, median reported sigma and median CRLB (where the table has one);
            x, y combined in quadrature; 95 % CI over rulers. -> fig_methods_photons.json
  ablation  fig_ablation.png + tab_ablation.md: per variant the paired difference to A (real movie: per-site spread on the common rows;
            simulated movies: RMS error against ground truth on the common rows), pass 1 and pass 2, and the sigma statistics
usage: python analysis/nanoruler/abl_figs.py gallery|methods|ablation [--pass 2]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
L = A.L
INK, MUTED, GRID, SURF = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
PUB = os.path.join(A.NR, 'analysis'); T = os.path.join(A.OUT, 'tabs')


def style():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.titlesize': 9, 'text.color': INK, 'axes.labelcolor': INK})
    return plt


def method_tables(p):
    M = [('Published SMLM', f'{PUB}/tab_pub_smlm.npz'), ('Published SIMFLUX', f'{PUB}/tab_pub_simflux.npz'), ('Baseline SIMFLUX (own DME)', f'{T}/tab_bl_simflux_dme.npz')]
    return M, [(f'SIMCODE v2 pass {q}', q) for q in ((1, 2) if p == 2 and os.path.exists(f'{T}/tab_p2_u45.npz') else (1,))]


def gallery(p):
    M, S = method_tables(p); q = S[-1][1]
    files = M + [(f'SIMCODE v2 pass {q} (sel rows)', f'{T}/tab_p{q}_sel.npz')]
    sites, rulers, _ = L.load_sites(A.SITES); Ms = {}
    for nm, fn in files:
        tab = L.load_table(fn); m = L.eval_method(tab, sites, rulers, 30.0, 10); m['T_x'] = tab['x_nm'] - m['shift'][0]; m['T_y'] = tab['y_nm'] - m['shift'][1]; Ms[nm] = m
    rsel = np.all([m['ruler_ok'] for m in Ms.values()], 0) & np.array([r['n_sites'] == 3 for r in rulers])
    pick = L.fig_gallery(f'{A.OUT}/fig_gallery.png', Ms, rulers, rsel, n=12, seed=3, ncol=12)
    print('saved fig_gallery.png; rulers', pick)


def methods(p, nboot=300, nbins=6):
    import abl_nra as N
    plt = style(); M, S = method_tables(p)
    files = M + [(f'{nm}, u <= 0.45', f'{T}/tab_p{q}_u45.npz') for nm, q in S] + [(f'{nm}, sel', f'{T}/tab_p{q}_sel.npz') for nm, q in S]
    sites, rulers, _ = L.load_sites(A.SITES); Ms = {}; tabs = {}
    for nm, fn in files:
        tabs[nm] = L.load_table(fn); Ms[nm] = L.eval_method(tabs[nm], sites, rulers, 30.0, 10)
    rsel = np.all([m['ruler_ok'] for m in Ms.values()], 0); W = N.boot_w(rsel, nboot); R = {}
    for nm, m in Ms.items():
        sig = dict(sig=(m['L_sx'], m['L_sy']))
        if 'L_crlb_x_nm' in m: sig['crlb'] = (m['L_crlb_x_nm'], m['L_crlb_y_nm'])
        edges = L.photon_edges(m, rsel, nbins); c, _ = N.calib(m, sig, edges, W)
        q2 = lambda k: [float(np.sqrt(0.5 * (c[k + '_x'][i] ** 2 + c[k + '_y'][i] ** 2))) for i in range(3)]
        R[nm] = dict(phot=[c[f'{b}|phot'][0] for b in range(nbins)], spread=[q2(f'{b}|spread') for b in range(nbins)], pair=[q2(f'{b}|pair') for b in range(nbins)],
                     sigma=[q2(f'{b}|sig_sig') for b in range(nbins)], crlb=[q2(f'{b}|crlb_sig') for b in range(nbins)] if 'crlb' in sig else None,
                     pairz=[c[f'{b}|sig_pairz'] for b in range(nbins)], ratio=[c[f'{b}|sig_ratio'] for b in range(nbins)], all=dict(spread=q2('all|spread'), pair=q2('all|pair'), sigma=q2('all|sig_sig'),
                     pairz=c['all|sig_pairz'], ratio=c['all|sig_ratio'], crlb=q2('all|crlb_sig') if 'crlb' in sig else None), n_rulers=int(rsel.sum()))
    A.jdump(R, f'{A.OUT}/fig_methods_photons.json')
    n = len(R); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow), sharey=True, squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.42))
    ser = (('spread', 'site spread', PAL[0], 'o', '-'), ('pair', 'consecutive-cycle pair precision', PAL[1], 's', '-'), ('sigma', 'reported sigma (median)', PAL[2], '^', '--'), ('crlb', 'CRLB (median)', PAL[3], 'D', ':'))
    for i, (nm, r) in enumerate(R.items()):
        a = ax[i // ncol, i % ncol]
        for k, lab, col, mk, ls in ser:
            if r[k] is None: continue
            if k == 'crlb' and np.allclose(np.array(r['crlb'])[:, 0], np.array(r['sigma'])[:, 0], rtol=1e-3): continue        # sigma = CRLB in these tables: one line
            y = np.array(r[k]); a.fill_between(r['phot'], y[:, 1], y[:, 2], color=col, alpha=0.13, lw=0)
            a.plot(r['phot'], y[:, 0], color=col, lw=2, ls=ls, marker=mk, ms=5.5, markeredgecolor=SURF, markeredgewidth=1.0, label=lab)
        crlb_is_sigma = r['crlb'] is not None and np.allclose(np.array(r['crlb'])[:, 0], np.array(r['sigma'])[:, 0], rtol=1e-3)
        a.set_title(nm + ('\n(reported sigma = CRLB)' if crlb_is_sigma else '\n '), loc='left'); a.grid(color=GRID, lw=0.6); a.set_ylim(0, None)
        a.text(0.97, 0.95, f'pair z {r["all"]["pairz"][0]:.2f}\nspread / sigma {r["all"]["ratio"][0]:.2f}', transform=a.transAxes, ha='right', va='top', fontsize=7.5)
        if i // ncol == nrow - 1: a.set_xlabel("photons per window (own unit)")
        if i % ncol == 0: a.set_ylabel('nm (x, y combined)')
    for j in range(n, nrow * ncol): ax[j // ncol, j % ncol].set_visible(False)
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc='lower right', bbox_to_anchor=(0.9, 0.08) if n % ncol else (0.9, -0.04), fontsize=8)
    fig.suptitle(f'80 nm nanorulers: scatter and stated uncertainty per method ({int(rsel.sum())} rulers valid in all methods, 95 % CI over rulers)', x=0.125, ha='left', fontsize=10)
    fig.savefig(f'{A.OUT}/fig_methods_photons.png', dpi=140, bbox_inches='tight'); print('saved fig_methods_photons.png')
    for nm, r in R.items(): print(f'{nm:34s} spread {r["all"]["spread"][0]:.2f} pair {r["all"]["pair"][0]:.2f} sigma {r["all"]["sigma"][0]:.2f} pair z {r["all"]["pairz"][0]:.2f} ratio {r["all"]["ratio"][0]:.2f}')


def ablation():
    import abl_nra as N
    plt = style(); order = ['B', 'C', 'D', 'E', 'E1', 'F', 'G', 'H', 'auto', 'autoS']
    lab = dict(N.LABEL, A='A full + c + c_a (production)')
    J = lambda f: json.load(open(f'{A.OUT}/{f}')) if os.path.exists(f'{A.OUT}/{f}') else None
    panels = [('Real movie: per-site spread', [('pass 1', J('nra_p1.json')), ('pass 2', J('nra_p2.json'))], 'real'),
              ('Sim nr_a (fluctuation put in): RMS error', [('pass 1', J('sim_sima_p1.json')), ('pass 2', J('sim_sima_p2.json'))], 'sim'),
              ('Sim nr_c0 (no fluctuation): RMS error', [('pass 1', J('sim_simc0_p1.json')), ('pass 2', J('sim_simc0_p2.json'))], 'sim')]
    def get(R, kind, n):
        V = R['views']['common']['variants']
        if n not in V: return None
        if kind == 'real': p = V[n]['paired']['xy']['median_diff_nm']; return [p['est'], *p['ci']]
        return V[n]['stats']['all|d_rms']
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    for a, (title, Rs, kind) in zip(ax, panels):
        a.axvline(0, color=MUTED, lw=1.2, ls=':')
        for j, (pn, R) in enumerate(Rs):
            if R is None: continue
            for i, n in enumerate(order):
                v = get(R, kind, n)
                if v is None: continue
                y = len(order) - 1 - i + (0.17 if j == 0 else -0.17)
                a.plot([v[1], v[2]], [y, y], color=PAL[j], lw=2, solid_capstyle='round'); a.plot([v[0]], [y], color=PAL[j], marker='os'[j], ms=6.5, markeredgecolor=SURF, markeredgewidth=1.0, lw=0, label=pn if i == 0 else None)
        a.set_title(title, loc='left'); a.grid(color=GRID, lw=0.6, axis='x'); a.set_xlabel('variant - A (nm); right of 0 = worse than A')
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([lab.get(n, n) for n in order[::-1]]); ax[0].tick_params(axis='y', colors=INK); ax[0].legend(frameon=False, loc='lower right')
    fig.suptitle('Solve variants on the same network outputs and the same windows (cycle-aligned, u <= 0.45 in A); 95 % CI over rulers', x=0.125, ha='left', fontsize=10)
    fig.savefig(f'{A.OUT}/fig_ablation.png', dpi=140, bbox_inches='tight'); print('saved fig_ablation.png')
    # ---- the table
    d3 = lambda v: '-' if v is None else '%+.3f [%+.3f, %+.3f]' % tuple(v); f2 = lambda v: '%.2f' % v
    Ls = ['# Solve-variant ablation: summary table', '', 'Same network outputs, same windows for every variant (cycle-aligned, strict_u <= 0.45 in A), each variant with its own gate-scene calibration. '
          'Differences are variant - A with 95 % CI over rulers (bootstrap). Real movie: paired per-site spread (x, y combined). Simulated movies: RMS error per axis against ground truth.', '']
    for pn, p in (('Pass 1', 'p1'), ('Pass 2', 'p2')):
        Rr, Ra, Rc = J(f'nra_{p}.json'), J(f'sim_sima_{p}.json'), J(f'sim_simc0_{p}.json')
        if Rr is None: continue
        V = Rr['views']['common']['variants']; Vs = Rr['views']['sel']['variants']; nb = Rr['nbins']
        Ls += [f'## {pn}', '', '| variant | real: site spread x / y (nm) | real: d site spread, common rows | real: d site spread, wide sel rows | nr_a: d RMS | nr_c0: d RMS | '
               'real, uncalibrated sigma: pair z dim -> bright (sel rows) | real: uncalibrated sigma relative to A, brightest bin | real, calibrated: pair z / spread over sigma | real, calibrated x g(u): pair z / spread over sigma | nr_a true z: uncalibrated / calibrated / x g(u) | nr_c0 true z: uncalibrated / calibrated / x g(u) |', '|' + '---|' * 12]
        for n in ['A'] + order:
            if n not in V: continue
            r = V[n]; c = r['calib']; cs = Vs[n]['calib']; e = r['est']
            row = [lab.get(n, n), '%.2f / %.2f' % (e['site_rsd_x_median'], e['site_rsd_y_median']), d3(get(Rr, 'real', n)) if n != 'A' else '-',
                   ('%+.3f [%+.3f, %+.3f]' % (Vs[n]['paired']['xy']['median_diff_nm']['est'], *Vs[n]['paired']['xy']['median_diff_nm']['ci'])) if n != 'A' else '-']
            for Rsim in (Ra, Rc): row.append(d3(get(Rsim, 'sim', n)) if (Rsim and n != 'A' and n in Rsim['views']['common']['variants']) else '-')
            row += ['%.2f -> %.2f (trend %s)' % (cs['0|raw_pairz'][0], cs[f'{nb - 1}|raw_pairz'][0], N.f3(cs['trend|raw_pairz'])),
                    N.f3(cs[f'rel|{nb - 1}|raw_ratio']) if n != 'A' else '-', '%.2f / %.2f' % (c['all|cal_pairz'][0], c['all|cal_ratio'][0]), '%.2f / %.2f' % (c['all|u_pairz'][0], c['all|u_ratio'][0])]
            for Rsim in (Ra, Rc):
                if Rsim and n in Rsim['views']['common']['variants']:
                    s = Rsim['views']['common']['variants'][n]['stats']; row.append('%.2f / %.2f / %.2f' % (s['all|raw_z'][0], s['all|cal_z'][0], s['all|u_z'][0]))
                else: row.append('-')
            Ls.append('| ' + ' | '.join(row) + ' |')
        Ls += ['', f'Rows: real common {V["A"]["calib"]["all|n"][0]:.0f} assigned localizations on {Rr["views"]["common"]["n_rulers"]} rulers; wide sel {Vs["A"]["calib"]["all|n"][0]:.0f} on {Rr["views"]["sel"]["n_rulers"]} rulers '
               f'(photon bins {Vs["A"]["calib"]["0|phot"][0]:.0f} ... {Vs["A"]["calib"][f"{nb - 1}|phot"][0]:.0f}).', '']
    open(f'{A.OUT}/tab_ablation.md', 'w').write('\n'.join(Ls) + '\n'); print('wrote tab_ablation.md')


if __name__ == '__main__':
    a = sys.argv[1:]; p = int(a[a.index('--pass') + 1]) if '--pass' in a else 2; A.title()
    dict(gallery=lambda: gallery(p), methods=lambda: methods(p), ablation=ablation)[a[0]]()
