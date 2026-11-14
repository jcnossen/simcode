"""Nanoruler ablation, part B1: per-variant solve calibration on the nr80 gate scenes (= analysis/ablation_cov/gate_abl.py for this setup).

Per variant (analysis/ablation_cov/variants.py A..H, E, E1, An, Dn; plus 'auto' = full Sigma_net with the automatic (c, c_a) of a set, see
abl_autoc.py) the calibration is re-fitted exactly as the chain's production one (analysis/nanoruler/ev_fitcal.py = plib.fit_cal on the
tuning seed 202, fit scenes rs05qm + iso, prior_off from iso, 65 nm px): kap_p / kap_I2 per network-photon bin from the head's z-scores with
the variant's own covariance, then the sigma factor per photon bin from the solved z-scores. sig_a / sig_g are inherited from the base.
Check: the refit of A must reproduce out/nanoruler/models/calib/calib_{sc_hd,p2L}_nr80.json.
Evaluation seeds 301-303 (gate_abl.stats): RMS / CRLB per GT-photon bin, paired difference to A, zcal, ranking.
The photon coverage of each fit (n targets per network-photon bin) is stored: the movie's photon range must lie inside it.

usage: CUDA_VISIBLE_DEVICES=g python analysis/nanoruler/abl_gate.py <front s1|s2> [variants A,B,..|all] [--auto <set>] [--noeval]
  -> out/nanoruler/ablation/calibs/calib_<front>_<v>.json, gate_<front>.json, tab_gate_<front>.md"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abl_lib as A
import plib as P
import variants as V
import gate_abl as GA


def _tr(R, v):
    """gate_abl.tr + the covariance scale s of an autoc variant (L -> sqrt(s) L; was the calibration key sig_scale, removed 2026-11-14)"""
    R = GA.tr(R, v)
    if 's' in v: R['sc_L'] = R['sc_L'] * float(v['s']) ** 0.5
    return R
from simcode import config
A.title(); P.PX = A.PX
FE = os.path.join(A.NR, 'eval', 'fe_nr80'); CD = os.path.join(A.OUT, 'calibs')
BASE = dict(s1='calib/calib_sc_hd_sc5nbb15_pm_bright.json', s2='calib/calib_p2L.json')
PROD = dict(s1=os.path.join(A.NR, 'models/calib/calib_sc_hd_nr80.json'), s2=os.path.join(A.NR, 'models/calib/calib_p2L_nr80.json'))
FIT = ('rs05qm', 'iso'); SCN = ['iso', 'pair2', 'pair1', 'rs05qm']


def variants(auto_set=None):
    """the variant table, plus 'auto' when the automatic theta of <set> exists (out/nanoruler/ablation/autoc_<set>.json)"""
    VARS = dict(V.VARS); order = list(V.ORDER)
    if auto_set:
        fn = f'{A.OUT}/autoc_{auto_set}.json'
        if os.path.exists(fn):
            th = json.load(open(fn))['mean']
            VARS['auto'] = dict(L='full', c=float(th[1]), ca=float(th[2]), theta_from=fn); order.insert(1, 'auto')
            # autoS: the whole fitted theta applied, i.e. also the scale s of the network covariance (production applies c, c_a only)
            VARS['autoS'] = dict(L='full', c=float(th[1]), ca=float(th[2]), s=float(th[0]), theta_from=fn); order.insert(2, 'autoS')
    return VARS, order


def calname(front, name, v):
    """the 'auto' calibration depends on the theta of the set it came from"""
    if name == 'auto': return f'{CD}/calib_{front}_auto_c{v["c"]:.3f}_ca{v["ca"]:.3f}.json'
    if name == 'autoS': return f'{CD}/calib_{front}_autoS_s{v["s"]:.3f}_c{v["c"]:.3f}_ca{v["ca"]:.3f}.json'
    return f'{CD}/calib_{front}_{name}.json'


def fit(front, name, v, cache={}):
    fn = calname(front, name, v)
    if os.path.exists(fn): return json.load(open(fn))
    if front not in cache:
        cache[front] = ([P.load(sc, front, seeds=(202,), d=FE) for sc in FIT], P.load('iso', front, seeds=(202,), d=FE))
    Rs, iso = cache[front]
    base = V.cal_variant(json.load(open(config.resolve(BASE[front]))), v)
    Rv = [_tr(R, v) for R in Rs]
    cal = P.fit_cal(Rv, base, iso=_tr(iso, v), verbose=False)
    ph = np.concatenate([R['sc_I'][P.targets(R)[1]].sum(1) for R in Rs])
    gph = np.concatenate([R['gt_phot'][P.targets(R)[0]] for R in Rs])
    cov = [int(((ph >= lo) & (ph < hi)).sum()) for lo, hi in zip(P.EDG[:-1], P.EDG[1:])]
    if v.get('noprior'):
        cal['kap_bins']['kap_p'] = [1000.0 * k for k in cal['kap_bins']['kap_p']]; cal.pop('sig_ph')
        zs, p2 = [], []
        for R in Rs:
            ti, oi = P.targets(R); sol = P.solve(_tr(R, v), oi, cal)
            zs.append((np.c_[sol['x'], sol['y']] - np.c_[R['gt_x'][ti], R['gt_y'][ti]]) / np.c_[sol['sx'], sol['sy']]); p2.append(R['sc_I'][oi].sum(1))
        zs, p2 = np.concatenate(zs), np.concatenate(p2); C2, F = [], []
        for lo, hi in zip(P.EDG[:-1], P.EDG[1:]):
            m = (p2 >= lo) & (p2 < hi)
            if m.sum() < 100: continue
            C2.append(float(np.median(p2[m]))); F.append(P.rob(zs[m]))
        cal['sig_ph'] = dict(ph=C2, f=F)
    cal['fit_cover'] = dict(edges=[float(e) for e in P.EDG], n=cov, net_ph_p1_p50_p99=[float(q) for q in np.percentile(ph, [1, 50, 99])],
                            gt_ph_p1_p50_p99=[float(q) for q in np.percentile(gph, [1, 50, 99])])
    cal['fit_scenes'] = [f'{FE}:{sc}' for sc in FIT]
    cal['note'] = f'nanoruler ablation variant {name} {v}: plib.fit_cal on seed 202 of {FIT} (prior_off iso), 65 nm px, base {BASE[front]}'
    os.makedirs(CD, exist_ok=True); json.dump(cal, open(fn, 'w'), indent=1)
    return cal


def table(front, res, order):
    L = [f'**nr80 gate scenes, front {front}: solve variants with their own calibration** (fit: seed 202 of {", ".join(FIT)}; evaluation seeds 301-303; '
         'RMS / multi-emitter CRLB, 95 % CI over FOVs; d = variant - A on the same targets).', '']
    for b in ('200-2000', '2000-4000', '4000-1e+09'):
        L += [f'GT photons {b}:', '', '| variant | ' + ' | '.join(f'{sc}: RMS/CRLB | d vs A | zcal' for sc in SCN) + ' |', '|---|' + '---|---|---|' * len(SCN)]
        for n in order:
            row = [V.LABEL.get(n, n + (' full + automatic s, c, c_a' if n == 'autoS' else ' full + automatic c, c_a'))]
            for sc in SCN:
                r = res['variants'].get(n, {}).get(sc, {}).get(b)
                if r is None: row += ['-', '-', '-']; continue
                row += [f'{r["rms"]:.3f} [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}] (n {r["n"]})',
                        (f'{r["d_rms"]:+.3f} [{r["d_ci"][0]:+.3f}, {r["d_ci"][1]:+.3f}]' if 'd_rms' in r else ''), f'{r["zcal"]:.2f}']
            L.append('| ' + ' | '.join(row) + ' |')
        L.append('')
    L += ['Calibration tables (network-photon bin medians of the fit; kap_I2 = variance scale of the network covariance, f = sigma factor):', '']
    for n in order:
        c = res['calibs'][n]
        L.append(f'- {n}: ph {np.round(c["ph"]).astype(int).tolist()} | kap_I2 {np.round(c["kap_I2"], 2).tolist()} | f {np.round(c["sig_f"], 2).tolist()}')
    c = res['cover']
    L += ['', f'Fit coverage: targets per network-photon bin (edges {c["edges"][:-1]} +): {c["n"]}; network photons p1 / p50 / p99 {np.round(c["net_ph_p1_p50_p99"]).tolist()}; '
          f'GT photons {np.round(c["gt_ph_p1_p50_p99"]).tolist()}. Bins with < 100 targets get no table entry (the solve interpolates in log photons and clamps outside).']
    return '\n'.join(L) + '\n'


def main():
    front = sys.argv[1]; names = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'all'
    auto = sys.argv[sys.argv.index('--auto') + 1] if '--auto' in sys.argv else None
    VARS, order = variants(auto)
    if names != 'all': order = [n for n in order if n in names.split(',') or n == 'A']
    fn = f'{A.OUT}/gate_{front}.json'
    res = json.load(open(fn)) if os.path.exists(fn) else dict(front=front, variants={}, calibs={})
    cals = {n: fit(front, n, VARS[n]) for n in order}
    if os.path.exists(PROD[front]):
        a, prod = cals['A'], json.load(open(PROD[front]))
        chk = {k: float(np.max(np.abs(np.array(a['kap_bins'][k] if k != 'sig_f' else a['sig_ph']['f']) - (prod['kap_bins'][k] if k != 'sig_f' else prod['sig_ph']['f']))))
               for k in ('kap_I2', 'kap_p', 'sig_f')}
        chk['prior_off'] = float(np.max(np.abs(np.array(a['prior_off']) - prod['prior_off'])))
        res['check_A_refit_vs_production_maxabs'] = chk; print(front, 'variant A refit vs the chain calibration, max |diff|:', chk, flush=True)
    res['cover'] = cals['A']['fit_cover']
    for n in order:
        res['calibs'][n] = dict(kap_I2=cals[n]['kap_bins']['kap_I2'], kap_p=cals[n]['kap_bins']['kap_p'], sig_f=cals[n]['sig_ph']['f'], ph=cals[n]['sig_ph']['ph'],
                                c=VARS[n]['c'], ca=VARS[n]['ca'])
        print(f'  cal {n:4s} kap_I2 {np.round(cals[n]["kap_bins"]["kap_I2"], 2)} sig_f {np.round(cals[n]["sig_ph"]["f"], 2)}', flush=True)
    if '--noeval' not in sys.argv:
        for sc in SCN:
            seeds = tuple(s for s in (301, 302, 303) if os.path.exists(f'{FE}/{sc}__s{s}_{front}.npz'))
            if not seeds: continue
            R = P.load(sc, front, seeds=seeds, d=FE); ti, oi = P.targets(R); EA = None
            for n in order:
                v = VARS[n]; E = P.errs(R, ti, oi, P.solve(_tr(R, v), oi, cals[n]))
                if n == 'A': EA = E
                st = GA.stats(R, E, None if n == 'A' else EA); res['variants'].setdefault(n, {})[sc] = st
                for b in ('200-2000', '2000-4000'):
                    if b in st:
                        r = st[b]
                        print(f'{front} {sc:7s} {n:4s} {b:9s} n {r["n"]:5d} rms/C {r["rms"]:.3f} [{r["ci"][0]:.3f},{r["ci"][1]:.3f}] '
                              + (f'd {r["d_rms"]:+.3f} [{r["d_ci"][0]:+.3f},{r["d_ci"][1]:+.3f}] ' if 'd_rms' in r else ' ' * 30)
                              + f'zcal {r["zcal"]:.2f} z>3 {r["z_gt3"]:.1f}% rms45 {r["rms45_nm"]:.2f} nm', flush=True)
            json.dump(res, open(fn, 'w'), indent=1)
    json.dump(res, open(fn, 'w'), indent=1)
    allv = [n for n in variants(auto)[1] if n in res['calibs']]
    open(f'{A.OUT}/tab_gate_{front}.md', 'w').write(table(front, res, allv))
    print('wrote', fn)


if __name__ == '__main__':
    main()
