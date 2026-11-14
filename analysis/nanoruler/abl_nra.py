"""Nanoruler ablation, part B3: nanoruler metrics per solve variant, paired differences to A, and the calibration picture.

Input: the variant tables of abl_solve.py (out/nanoruler/ablation/runs/<set>[_<ptag>]/<variant>/locs.npz; identical rows in every variant).
Drift: ONE trace for all variants = DME v2 on variant A's cycle-aligned max sigma <= 12 nm rows (so that only the solve differs); none for simulated sets.
Views:
  common   the same windows for every variant: the cycle-aligned rows with max sigma <= 12 nm (m12) in variant A
  own      each variant's own cycle-aligned max sigma <= 12 nm (m12) rows (its own sigma ranking)
Per variant and view (nra_lib, fixed site list, rulers valid in all variants of the view, bootstrap over rulers):
  site spread (x, y, along, across), pooled spread, ruler distance mean / sd, locs per site;
  paired to A: per-site spread difference (nra_lib.paired), pooled spread difference, ruler distance difference;
  calibration in photon bins (common photon edges from A) for four sigma levels (abl_solve.py):
      cal = calibrated, nb = without the per-photon-bin factor, raw = uncalibrated model sigma, f = Fisher sigma at the MAP
    spread = robust sd of the residuals to the site centre; ratio = spread / median sigma; z sd = robust sd of residual / sigma;
    pair z = robust sd of (difference of two localizations of one site in consecutive cycles) / sqrt(sigma1^2 + sigma2^2): free of the
    site size and of slow drift (errors common to the two cycles cancel, so it is a lower bound of the true z sd).
  The site residuals contain the physical site size and residual drift, a constant floor that by itself makes spread / sigma rise with
  photons; the pair z does not have it. 'excess' = sqrt(spread^2 - sigma^2) per bin is that floor if sigma is right.

usage: python analysis/nanoruler/abl_nra.py <set> [--ptag name] [--variants A,B,..] [--nboot 500] [--bins 8] [--ref A]
  -> out/nanoruler/ablation/nra_<set>[_<ptag>].json, tab_abl_<set>[_<ptag>].md, fig_calib_<set>[_<ptag>].png"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
import variants as V
L = A.L
VIEWS = ('common', 'own', 'sel')
LEVELS = (('cal', 'sx', 'sy', 'calibrated sigma'), ('u', 'sx_u', 'sy_u', 'calibrated sigma x g(u) (sigu rescale of the chain)'), ('nb', 'sx_nb', 'sy_nb', 'without the per-photon-bin factor'),
          ('raw', 'sx_raw', 'sy_raw', 'uncalibrated model sigma'), ('f', 'sx_f', 'sy_f', 'Fisher sigma at the MAP'))
LABEL = dict(V.LABEL, auto='auto: full + automatic c, c_a', autoS='autoS: full + automatic s, c, c_a')
G = {}


def boot_w(rsel, nboot, seed=1):
    rng = np.random.default_rng(seed); idx = np.nonzero(rsel)[0]
    return [rsel.astype(int)] + [np.bincount(rng.choice(idx, len(idx)), minlength=len(rsel)) for _ in range(nboot)]


def ci(v):
    v = np.asarray(v, float)
    return [float(v[0]), float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))] if len(v) > 1 else [float(v[0]), np.nan, np.nan]


def pairs_of(M, min_n=10):
    """index pairs (into the per-loc arrays of M) of consecutive cycles within a site"""
    j, fr = M['L_site'], M['L_frame']
    return np.nonzero((j[1:] == j[:-1]) & (fr[1:] - fr[:-1] == 1) & (M['n'][j[1:]] >= min_n))[0]


def calib(M, sig, edges, W):
    """per photon bin and overall: spread, median sigma, ratio, z sd, excess, pair z; [est, lo, hi] over the ruler bootstrap W.
    sig = dict level -> (sx, sy) per assigned loc"""
    sr = M['site_ruler']; ls = M['L_site']; ph = M['L_ph']; nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, ph, side='right') - 1, 0, nb - 1)
    pp = pairs_of(M); pb = np.clip(np.searchsorted(edges, 0.5 * (ph[pp] + ph[pp + 1]), side='right') - 1, 0, nb - 1)
    dxy = dict(x=M['L_x'][pp + 1] - M['L_x'][pp], y=M['L_y'][pp + 1] - M['L_y'][pp]); res = dict(x=M['L_rx'], y=M['L_ry'])
    acc = {}
    for w in W:
        wl = w[sr][ls]; li = np.repeat(np.arange(len(wl)), wl); wp = w[sr][ls[pp]]; pi = np.repeat(np.arange(len(pp)), wp)
        for bi in list(range(nb)) + ['all']:
            m = li if bi == 'all' else li[b[li] == bi]; mp = pi if bi == 'all' else pi[pb[pi] == bi]
            put = lambda k, v: acc.setdefault(f'{bi}|{k}', []).append(v)
            put('n', len(m)); put('phot', float(np.median(ph[m])) if len(m) else np.nan); put('n_pairs', len(mp))
            for ax in 'xy':
                r = res[ax][m]; s0 = L.rsd(r) if len(r) > 5 else np.nan; put(f'spread_{ax}', s0)
                put(f'pair_{ax}', L.rsd(dxy[ax][mp]) / np.sqrt(2) if len(mp) > 20 else np.nan)
                for lv, (sx, sy) in sig.items():
                    s = (sx if ax == 'x' else sy)
                    if len(r) <= 5:
                        for k in ('sig', 'ratio', 'z', 'excess'): put(f'{lv}_{k}_{ax}', np.nan)
                    else:
                        sm = float(np.median(s[m])); put(f'{lv}_sig_{ax}', sm); put(f'{lv}_ratio_{ax}', s0 / sm); put(f'{lv}_z_{ax}', L.rsd(r / s[m]))
                        put(f'{lv}_excess_{ax}', np.sign(s0 - sm) * np.sqrt(abs(s0 ** 2 - sm ** 2)))
                    put(f'{lv}_pairz_{ax}', L.rsd(dxy[ax][mp] / np.hypot(s[pp[mp]], s[pp[mp] + 1])) if len(mp) > 20 else np.nan)
            for lv in sig:      # x, y combined, and the bright / dim contrast is formed from these later
                for k in ('ratio', 'z', 'pairz'):
                    acc.setdefault(f'{bi}|{lv}_{k}', []).append(float(np.sqrt(0.5 * (acc[f'{bi}|{lv}_{k}_x'][-1] ** 2 + acc[f'{bi}|{lv}_{k}_y'][-1] ** 2))))
    for lv in sig:              # trend with photons: brightest / dimmest bin
        for k in ('ratio', 'z', 'pairz'):
            acc[f'trend|{lv}_{k}'] = (np.array(acc[f'{nb - 1}|{lv}_{k}']) / np.array(acc[f'0|{lv}_{k}'])).tolist()
    raw = {k: np.asarray(v, float) for k, v in acc.items() if k.split('|')[1].split('_', 1)[-1] in ('ratio', 'z', 'pairz')}
    return {k: ci(v) for k, v in acc.items()}, raw


def one(job):
    name, view = job
    T, mask, sites, rulers, drift = G['T'][name], G['mask'][view][name], G['sites'], G['rulers'], G['drift']
    tab = A.to_nra(T, mask, drift)
    M = L.eval_method(tab, sites, rulers, 30.0, 10)
    rows = tab['row'][M['L_idx']]
    sig = {lv: (T.get(kx, T['sx'])[rows].astype(np.float64) * A.PX, T.get(ky, T['sy'])[rows].astype(np.float64) * A.PX) for lv, kx, ky, _ in LEVELS}     # a table without a level falls back to the calibrated sigma
    keep = ('shift', 'n', 'centre', 'S', 'site_ruler', 'ruler_ok', 'pair_ruler', 'pair_dist', 'ruler_nsites')
    return name, view, M, sig


def finish(job):
    name, view, rsel, edges, nboot = job
    M = G['M'][(name, view)]; sig = G['sig'][(name, view)]; W = boot_w(rsel, nboot)
    est, cis = L.bootstrap(M, rsel, edges, nboot, 1)
    cal_, raw = calib(M, sig, edges, W)
    out = dict(est={k: est[k] for k in est if not k.startswith('bin')}, ci={k: cis[k] for k in cis if not k.startswith('bin')}, calib=cal_,
               n_table=M['n_table'], shift_nm=M['shift'].tolist(), _raw=raw)
    ref = G['ref']
    if name != ref:
        MA = G['M'][(ref, view)]
        out['paired'] = L.paired(M, MA, rsel, nboot, 1)
        sr = M['site_ruler']; d = {}
        for ax in ('x', 'y'):
            v = []
            for w in W:
                a = np.repeat(M['L_r' + ax], w[sr][M['L_site']]); b = np.repeat(MA['L_r' + ax], w[sr][MA['L_site']]); v.append(L.rsd(a) - L.rsd(b))
            d['pooled_' + ax] = ci(v)
        v = []; v2 = []
        for w in W:
            a = np.repeat(M['pair_dist'], w[M['pair_ruler']]); b = np.repeat(MA['pair_dist'], w[MA['pair_ruler']])
            v.append(a.mean() - b.mean()); v2.append(a.std(ddof=1) - b.std(ddof=1))
        d['dist_mean'] = ci(v); d['dist_sd'] = ci(v2); out['diff'] = d
    return name, view, out


def f3(c, d=2): return ('%.{0}f [%.{0}f, %.{0}f]'.format(d)) % tuple(c)


def markdown(R, setn):
    order = R['order']; nb = R['nbins']; ref = R['ref']; Ls = [f'# Solve-variant ablation on the nanorulers: {setn}', '', R['header'], '']
    for view, txt in (('common', 'COMMON rows (the same windows for every variant: cycle-aligned, max sigma <= 12 nm (m12) in variant %s)' % ref),
                      ('own', "each variant's OWN selection (cycle-aligned, its own max sigma <= 12 nm (m12))"),
                      ('sel', "WIDE photon range, same windows for every variant: all cycle-aligned rows of the 'sel' selection of variant %s" % ref)):
        V_ = R['views'][view]
        Ls += [f'## Positions, {txt}', '', f'{V_["n_rulers"]} rulers valid in all variants. nm; [95 % CI, bootstrap over rulers]; differences are variant - {ref}, paired on the same sites.', '',
               '| variant | locs/site med | site spread x | y | along | across | pooled x | pooled y | pair precision x | y | distance mean | distance sd | '
               f'd site spread xy (median, paired) | d pooled x | d pooled y | d distance mean | d distance sd |', '|' + '---|' * 17]
        for n in order:
            r = V_['variants'][n]; e, c = r['est'], r['ci']; g = lambda k: '%.2f [%.2f, %.2f]' % (e[k], c[k][0], c[k][1])
            row = [LABEL.get(n, n), '%.0f' % e['locs_per_site_median'], g('site_rsd_x_median'), g('site_rsd_y_median'), g('site_rsd_along_median'), g('site_rsd_across_median'),
                   g('pooled_rsd_x'), g('pooled_rsd_y'), g('pair_rsd_x'), g('pair_rsd_y'), g('dist_mean'), g('dist_sd')]
            if 'paired' in r:
                p = r['paired']['xy']['median_diff_nm']; d = r['diff']
                row += ['%+.3f [%+.3f, %+.3f]' % (p['est'], *p['ci'])] + ['%+.3f [%+.3f, %+.3f]' % tuple(d[k]) for k in ('pooled_x', 'pooled_y', 'dist_mean', 'dist_sd')]
            else: row += ['-'] * 5
            Ls.append('| ' + ' | '.join(row) + ' |')
        Ls.append('')
        for lv, _, _, txt2 in LEVELS:
            Ls += [f'### Calibration, {view} rows, {txt2} ({lv}): spread / median sigma per photon bin (x, y combined in quadrature)', '',
                   '| variant | median sigma x / y (all) | all | ' + ' | '.join('%.0f ph' % V_['variants'][ref]['calib'][f'{i}|phot'][0] for i in range(nb)) + ' | brightest / dimmest bin |', '|' + '---|' * (nb + 4)]
            for n in order:
                c = V_['variants'][n]['calib']
                Ls.append('| ' + ' | '.join([LABEL.get(n, n), '%.2f / %.2f' % (c[f'all|{lv}_sig_x'][0], c[f'all|{lv}_sig_y'][0]), f3(c[f'all|{lv}_ratio'])] + [f3(c[f'{i}|{lv}_ratio']) for i in range(nb)] + [f3(c[f'trend|{lv}_ratio'])]) + ' |')
            Ls += ['', f'Pair z sd (consecutive cycles of one site; free of site size and slow drift), {lv}:', '',
                   '| variant | all | ' + ' | '.join('%.0f ph' % V_['variants'][ref]['calib'][f'{i}|phot'][0] for i in range(nb)) + ' | brightest / dimmest bin |', '|' + '---|' * (nb + 3)]
            for n in order:
                c = V_['variants'][n]['calib']
                Ls.append('| ' + ' | '.join([LABEL.get(n, n), f3(c[f'all|{lv}_pairz'])] + [f3(c[f'{i}|{lv}_pairz']) for i in range(nb)] + [f3(c[f'trend|{lv}_pairz'])]) + ' |')
            if view != 'own' and lv in ('raw', 'f'):
                for k, kt in (('ratio', 'spread / sigma'), ('pairz', 'pair z sd')):
                    Ls += [f'{kt} of the variant relative to variant {ref} on the same rows ({lv}; > 1 = this noise model states a SMALLER sigma than {ref} for the same scatter):', '',
                           '| variant | all | ' + ' | '.join('%.0f ph' % V_['variants'][ref]['calib'][f'{i}|phot'][0] for i in range(nb)) + ' | brightest / dimmest bin |', '|' + '---|' * (nb + 3)]
                    for n in order:
                        if n == ref: continue
                        c = V_['variants'][n]['calib']
                        Ls.append('| ' + ' | '.join([LABEL.get(n, n), f3(c[f'rel|all|{lv}_{k}'])] + [f3(c[f'rel|{i}|{lv}_{k}']) for i in range(nb)] + [f3(c[f'rel|trend|{lv}_{k}'])]) + ' |')
                    Ls.append('')
            Ls.append('')
        c = V_['variants'][ref]['calib']
        Ls += [f'Variant {ref}, {view} rows, per bin: n ' + ', '.join('%d' % c[f'{i}|n'][0] for i in range(nb)) + '; spread x ' + ', '.join('%.2f' % c[f'{i}|spread_x'][0] for i in range(nb))
               + '; spread y ' + ', '.join('%.2f' % c[f'{i}|spread_y'][0] for i in range(nb)) + '; pair precision x ' + ', '.join('%.2f' % c[f'{i}|pair_x'][0] for i in range(nb))
               + '; excess over the calibrated sigma x ' + ', '.join('%.2f' % c[f'{i}|cal_excess_x'][0] for i in range(nb)) + ' nm.', '']
    return '\n'.join(Ls) + '\n'


def figure(R, fn, show=('A', 'B', 'C', 'D', 'E', 'autoS'), view='common'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    INK, MUTED, GRID, SURF = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
    COL = dict(A='#2a78d6', B='#eb6834', C='#1baf7a', D='#eda100', E='#e87ba4', autoS='#6250d6'); MK = dict(A='o', B='s', C='^', D='D', E='v', autoS='P')
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.titlesize': 9, 'text.color': INK, 'axes.labelcolor': INK})
    Vc = R['views'][view]['variants']; nb = R['nbins']; show = [n for n in show if n in Vc]
    pal = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']; mks = 'os^DvP'
    for i, n in enumerate(show): COL.setdefault(n, pal[i % 6]); MK.setdefault(n, mks[i % 6])
    cols = (('ratio', 'site spread / median sigma'), ('z', 'robust sd of residual / sigma'), ('pairz', 'consecutive-cycle pair z sd\n(no site size, no slow drift)'))
    rows = (('cal', 'Calibrated sigma'), ('u', 'Calibrated x g(u)'), ('raw', 'Uncalibrated model sigma'))
    fig, ax = plt.subplots(3, 3, figsize=(11.5, 10.0), gridspec_kw=dict(wspace=0.25, hspace=0.38), sharex=True)
    for i, (lv, rt) in enumerate(rows):
        for j, (k, ct) in enumerate(cols):
            a = ax[i, j]; a.axhline(1.0, color=MUTED, lw=1.2, ls=':')
            for n in show:
                c = Vc[n]['calib']; x = [c[f'{b}|phot'][0] for b in range(nb)]; y = np.array([c[f'{b}|{lv}_{k}'] for b in range(nb)])
                a.fill_between(x, y[:, 1], y[:, 2], color=COL[n], alpha=0.12, lw=0)
                a.plot(x, y[:, 0], color=COL[n], lw=2, marker=MK[n], ms=5.5, markeredgecolor=SURF, markeredgewidth=1.0, label=LABEL.get(n, n))
            a.grid(color=GRID, lw=0.6); a.set_title(ct, loc='left')
            if i == 2: a.set_xlabel('network window photons (bin median)')
            if j == 0: a.set_ylabel(f'{rt}\nratio (1 = sigma describes the scatter)')
    ax[2, 2].legend(frameon=False, fontsize=7.5, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(R['title'].replace('common rows', {'common': 'common rows max sigma <= 12 nm', 'sel': "all 'sel' rows"}[view]), x=0.125, ha='left', fontsize=10)
    fig.savefig(fn, dpi=140, bbox_inches='tight'); print('saved', fn)


def load_sites(S):
    if S['kind'] == 'real': return L.load_sites(A.SITES)[:2]
    import abl_sim
    return abl_sim.sim_sites(S['sim'])


def main():
    from multiprocessing import Pool
    a = sys.argv[1:]; g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    setn = a[0]; ptag = g('--ptag', ''); nboot = int(g('--nboot', 500)); nbins = int(g('--bins', 8)); ref = g('--ref', 'A'); A.title()
    S = A.SETS[setn]; base = f'{A.OUT}/runs/{setn}{"_" + ptag if ptag else ""}'; suf = f'{setn}{"_" + ptag if ptag else ""}'
    order = [n for n in ['A', 'auto', 'autoS', 'B', 'C', 'D', 'E', 'E1', 'F', 'G', 'H', 'An', 'Dn'] if os.path.exists(f'{base}/{n}/locs.npz')]
    if g('--variants'): order = [n for n in g('--variants').split(',') if os.path.exists(f'{base}/{n}/locs.npz')]   # any sub-directory names (part D: patterns)
    G['T'] = {n: A.load_table(f'{base}/{n}/locs.npz') for n in order}; G['ref'] = ref
    G['sites'], G['rulers'] = load_sites(S)
    TA = G['T'][ref]; mA = A.selection(TA, 'm12')
    dfn = f'{base}/drift_{ref}_m12.npy'       # real and simulated sets alike (the simulated movies have sample drift)
    if not os.path.exists(dfn):
        cyc = (TA['frame'][mA] // 6).astype(np.int64)
        d, rep = A.drift_report(TA['x'][mA], TA['y'][mA], cyc, TA['sx'][mA], TA['sy'][mA], S['ncyc'], name=f'{suf} variant {ref} m12', sites=(G['sites'], G['rulers']))
        np.save(dfn, d); A.jdump(rep, dfn[:-4] + '.json')
    G['drift'] = np.load(dfn)
    mS = A.selection(TA, 'sel')
    G['mask'] = dict(common={n: mA for n in order}, own={n: A.selection(G['T'][n], 'm12') for n in order}, sel={n: mS for n in order})
    jobs = [(n, v) for v in VIEWS for n in order]
    with Pool(min(12, len(jobs))) as pool: res = pool.map(one, jobs)
    G['M'] = {(n, v): M for n, v, M, s in res}; G['sig'] = {(n, v): s for n, v, M, s in res}
    R = dict(set=setn, ptag=ptag, order=order, ref=ref, nbins=nbins, views={}, title=f'Nanorulers, {suf}: does each noise model predict the scatter? (common rows, 95 % CI over rulers)',
             header=f'Set {suf} ({S["run"]}), {len(G["rulers"])} rulers / {len(G["sites"])} sites, radius 30 nm, min 10 locs per site, {nboot} bootstrap resamples over rulers. '
                    + 'One drift trace for all variants (DME v2 on variant %s, cycle-aligned max sigma <= 12 nm rows). ' % ref
                    + 'Each variant is solved with its own gate-scene calibration (abl_gate.py), no multiplier from the other setup.')
    fin = []
    for v in VIEWS:
        rsel = np.all([G['M'][(n, v)]['ruler_ok'] for n in order], 0)
        ve = 'sel' if v == 'sel' else 'common'
        MA = G['M'][(ref, ve)]; edges = L.photon_edges(MA, np.all([G['M'][(n, ve)]['ruler_ok'] for n in order], 0), nbins)
        R['views'][v] = dict(n_rulers=int(rsel.sum()), edges=[float(e) for e in edges], variants={})
        fin += [(n, v, rsel, edges, nboot) for n in order]
    with Pool(min(12, len(fin))) as pool: out = pool.map(finish, fin)
    raws = {(n, v): o.pop('_raw') for n, v, o in out}
    for n, v, o in out:
        R['views'][v]['variants'][n] = o
        if v != 'own':          # same rows and same bootstrap weights as the reference: sigma-ratio statistics relative to the reference variant
            for k, a in raws[(n, v)].items(): o['calib']['rel|' + k] = ci(a / raws[(ref, v)][k])
    A.jdump(R, f'{A.OUT}/nra_{suf}.json'); md = markdown(R, suf); open(f'{A.OUT}/tab_abl_{suf}.md', 'w').write(md)
    sh = order[:6] if g('--variants') else ('A', 'B', 'C', 'D', 'E', 'autoS')
    figure(R, f'{A.OUT}/fig_calib_{suf}.png', show=sh); figure(R, f'{A.OUT}/fig_calib_{suf}_sel.png', show=sh, view='sel')
    print(md[:3000])


if __name__ == '__main__':
    main()
