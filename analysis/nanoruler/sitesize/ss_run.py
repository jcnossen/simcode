"""Site-size study, step 2 (CPU): per movie, every DME v2 table (ss_tables.py) -> per-site spread, sigma_pair(Delta), event test, fit, bootstrap
over the sites common to all methods of the movie (ss_lib.py for the definitions). Synthetic check on the simulations ('inj': a per-event
offset of 3 nm per axis + a common AR(1) fluctuation of 2 nm per axis with a 5-cycle correlation time, added to the SIMCODE table).
  python ss_run.py <movie> ...      movies: nr80 nr40 nr_a nr_c0 s40g_a s40g_c0
Writes out/nanoruler/sitesize/res_<movie>.json, tab_<movie>.md."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ss_lib as S
NR = '/home/jc/simcode2/out/nanoruler'; OUT = f'{NR}/sitesize'; TAB = f'{OUT}/tabs'
NCYC = dict(s20_a=8000, nr80=5000, nr40=6666, nr_a=2000, nr_c0=2000, s40g_a=4000, s40g_c0=4000)
M80 = ['pub_smlm_dme', 'pub_smlm_paper', 'pub_smlm_paperdme', 'bl_smlm_dme', 'pub_simflux_dme', 'pub_simflux_paper', 'pub_simflux_paperdme',
       'bl_simflux_dme', 'simcode_m12_dme']
INJ = dict(ev_sd=3.0, ar_sd=2.0, ar_tau=5.0, seed=11)


def sites_of(mv):
    if mv == 'nr80':
        d = json.load(open(f'{NR}/analysis/sites.json')); sc = np.array([[s['x'], s['y']] for s in d['sites']])
        ax = np.array([d['rulers'][s['ruler']]['axis'] for s in d['sites']]); return dict(sc=sc, axis=ax, used=np.ones(len(sc), bool), radius=30.0, rmax=40.0)
    if mv == 's20_a':      # linear 20 nm rulers (point sites); nearest site within 10 nm: truncated, a check of the Delta / event structure only
        z = np.load(f'{NR}/nr20/sim/s20_a/sites.npz'); sc = np.c_[z['x_nm'], z['y_nm']]
        return dict(sc=sc, axis=np.c_[z['ax'], z['ay']], used=np.ones(len(sc), bool), radius=10.0, rmax=15.0)
    if mv in ('nr_a', 'nr_c0'):
        z = np.load(f'{NR}/sim/{mv}/sites.npz'); sc = np.c_[z['x_nm'], z['y_nm']]
        return dict(sc=sc, axis=np.c_[z['ax'], z['ay']], used=np.ones(len(sc), bool), radius=30.0, rmax=40.0)
    fn = f'{NR}/nr40/pairs_auto_sf.json' if mv == 'nr40' else f'{NR}/nr40/simeval/pairs_truth_s40g.json'
    d = json.load(open(fn)); sc = np.array(d['sites'], float); ax = np.zeros_like(sc); used = np.zeros(len(sc), bool)
    for r in d['rulers'][::-1]:
        for s in r['sites']: ax[s] = r['axis']; used[s] = True      # first pair of each site wins
    return dict(sc=sc, axis=ax, used=used, radius=20.0, rmax=30.0)


S20 = dict(bl_smlm_dme=f'{NR}/nr20/merged/tabs/bl_s20_a_smlm.npz', bl_simflux_dme=f'{NR}/nr20/merged/tabs/bl_s20_a_simflux.npz',
           simcode_sel_dme=f'{NR}/nr20/simeval/tabs/s20_a_p1_sel.npz')      # existing own-DME v2 tables (r24_tab.py, same abl_lib.dme call)


def load(mv, m):
    if mv == 's20_a': z = dict(np.load(S20[m])); return {k: z[k] for k in ('frame', 'x_nm', 'y_nm', 'sx_nm', 'sy_nm', 'photons')}
    if m.endswith('_nodrift'):                     # the published table without any drift correction (= DME table + its trace)
        b = m.replace('_nodrift', '_dme'); z = dict(np.load(f'{TAB}/{mv}__{b}.npz')); d = np.load(f'{TAB}/{mv}__{b}_drift.npy')
        z['x_nm'] = z['x_nm'] + d[z['frame'], 0] * 65.0; z['y_nm'] = z['y_nm'] + d[z['frame'], 1] * 65.0; return z
    return dict(np.load(f'{TAB}/{mv}__{m}.npz'))


def keep_of(R):
    k = np.zeros(len(R['n']), bool); k[R['uidx']] = R['n'][R['uidx']] >= S.MIN_N; return k


def bincount_of(R, sites):
    return R['H'][R['pos'][sites]][:, :, 0].sum((2, 3))                # pairs per site / Delta bin (+ 'all'), classes pooled


def zfix(r):
    for a in r['axes'].values():
        for k in ('sigma_all', 'sigma_loc', 's_slow', 'plateau', 's_event', 's_site', 'dvar_event'):
            a[k] = [v / S.ZSCALE / (S.ZSCALE if k == 'dvar_event' else 1) for v in a[k]]
        for k in ('sigma_pair', 'ev_same', 'ev_diff'): a[k] = {q: [v / S.ZSCALE for v in w] for q, w in a[k].items()}
        a['note'] = 'z units (robust sd of the pair z); spread / s_site_vs_spread / var fields not converted'


def run(mv):
    nc = NCYC[mv]; St = sites_of(mv)
    if mv == 'nr80': methods = M80 + ['pub_smlm_nodrift', 'pub_simflux_nodrift']
    elif mv == 'nr40': methods = list(M80)
    elif mv == 's20_a': methods = list(S20)
    else: methods = ['simcode_m12_dme', 'simcode_m12_dme+inj']
    methods += [m + x for m in methods if m.split('+')[0] == m and not m.endswith('_paper') and not m.endswith('_nodrift') and not m.endswith('_paperdme') for x in ('+int', '+z')]
    Rs = {}
    for m in methods:
        tab = load(mv, m.split('+')[0]); md = 'interior' if '+int' in m else ('z' if '+z' in m else 'all')
        Rs[m] = S.per_site(tab, St, nc, inj=INJ if m.endswith('+inj') else None, mode=md)
        print(f'{mv} {m}: {len(tab["frame"])} rows, {Rs[m]["n_assigned"]} assigned, sites n >= {S.MIN_N}: {int(((Rs[m]["n"] >= S.MIN_N) & St["used"]).sum())}', flush=True)
    core = [m for m in methods if not (m.endswith('_nodrift') or '+int' in m or '+z' in m)]
    common = np.logical_and.reduce([keep_of(Rs[m]) for m in core]); sites = np.nonzero(common)[0]
    print(f'{mv}: {len(sites)} common sites', flush=True)
    res = dict(movie=mv, ncyc=nc, n_common_sites=int(len(sites)), settings=dict(GAP=S.GAP, MIN_N=S.MIN_N, MIN_P=S.MIN_P, radius=St['radius'], inj=INJ), methods={})
    bm = np.logical_and.reduce([bincount_of(Rs[m], sites) >= S.MIN_P for m in core])
    res['n_sites_bin'] = dict(zip([str(d) for d, _, _ in Rs[core[0]]['DB']] + ['all'], bm.sum(0).tolist()))
    print(f'{mv}: sites per Delta bin (all methods) {res["n_sites_bin"]}', flush=True)
    for m in methods:
        si = sites if (m in core or '+' in m) else np.nonzero(keep_of(Rs[m]))[0]
        if '+int' in m: si = si[Rs[m]['pos'][si] >= 0]
        res['methods'][m] = S.summarize(Rs[m], si, bm if m in core else None)
        if '+z' in m: zfix(res['methods'][m])
        res['methods'][m]['global_shift_nm'] = Rs[m]['shift'].tolist()
        dj = f'{TAB}/{mv}__{m.split("+")[0].replace("_nodrift", "_dme")}_drift.json'
        if os.path.exists(dj): res['methods'][m]['drift'] = json.load(open(dj))
        print(line(m, res['methods'][m]), flush=True)
    json.dump(res, open(f'{OUT}/res_{mv}.json', 'w'), indent=1, default=float)
    open(f'{OUT}/tab_{mv}.md', 'w').write(markdown(res))


def f3(v, d=2): return f'{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]'


def line(m, r):
    a = r['axes']['xy']
    return (f'  {m}: sites {r["n_sites"]}, sigma_pair(1) {a["sigma_pair"][1][0]:.2f}, (100) {a["sigma_pair"].get(100, [np.nan])[0]:.2f}, all {a["sigma_all"][0]:.2f}, spread {a["spread"][0]:.2f}; '
            f'fit loc {a["sigma_loc"][0]:.2f} slow {a["s_slow"][0]:.2f} tau {a["tau"][0]:.1f} plateau {a["plateau"][0]:.2f} s_site {a["s_site"][0]:.2f}; s_event {a["s_event"][0]:.2f}')


def markdown(res):
    o = [f'# Site size, {res["movie"]} ({res["n_common_sites"]} common sites; nm, value [95 % bootstrap CI over sites])', '']
    for ax in ('xy', 'along', 'across'):
        o += [f'## axis: {ax}', '', '| method | sites | locs / site | sigma_loc | s_slow | tau (cycles) | plateau | sigma_all pairs | spread (median per site) | s_site = sqrt(all^2 - plateau^2) | s_site vs spread | var(10) - var(1) | var(1000) - var(10) | s_event (diff - same event, 4-12 cycles) |',
              '|' + '---|' * 14]
        for m, r in res['methods'].items():
            a = r['axes'][ax]
            o.append(f'| {m} | {r["n_sites"]} | {r["locs_per_site_median"]:.0f} | {f3(a["sigma_loc"])} | {f3(a["s_slow"])} | {f3(a["tau"], 1)} | {f3(a["plateau"])} | {f3(a["sigma_all"])} | {f3(a["spread"])} | '
                     f'{f3(a["s_site"])} | {f3(a["s_site_vs_spread"])} | {f3(a["fast_var_1_10"], 1)} | {f3(a["slow_var_10_1000"], 1)} | {f3(a["s_event"])} |')
        o.append('')
    o += ['## sigma_pair(Delta), axis xy', '', '| method | ' + ' | '.join(f'D={d:g}' for d in next(iter(res['methods'].values()))['deltas']) + ' | all |', '|' + '---|' * (2 + len(next(iter(res['methods'].values()))['deltas']))]
    for m, r in res['methods'].items():
        a = r['axes']['xy']; o.append(f'| {m} | ' + ' | '.join(f3(a['sigma_pair'][int(d)]) for d in r['deltas']) + f' | {f3(a["sigma_all"])} |')
    o += ['', '## event test, axis xy: sigma_pair of interior localizations, same event / different events at equal Delta (sites)', '',
          '| method | ' + ' | '.join(S.EBINS.__repr__() and [f'{lo}-{hi}' for lo, hi in S.EBINS]) + ' |', '|' + '---|' * (1 + len(S.EBINS))]
    for m, r in res['methods'].items():
        a = r['axes']['xy']; o.append(f'| {m} | ' + ' | '.join(f'{a["ev_same"][k][0]:.2f} / {a["ev_diff"][k][0]:.2f} ({a["ev_nsites"][k]})' for k in a['ev_same']) + ' |')
    return '\n'.join(o) + '\n'


if __name__ == '__main__':
    from simcode2.util import proctitle; proctitle.set_title('nr-sitesize')
    for mv in sys.argv[1:]: run(mv)
