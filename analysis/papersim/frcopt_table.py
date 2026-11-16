"""Summary tables of the selection sweep (frcopt.py) over densities.

Per set and method (SIMCODE: every threshold family): the current Fig 2 selection, the GT-FRC optimum (--crit) and the
half-split optima (first and sustained crossing), with rows kept, false positives (3-sigma own-precision gate = the Fig 2
'false positive' and the geometric one: no ON emitter within 50 nm), RMS error of the true positives, and the GT-FRC reached
at the half-split optimum (= what the real-data rule would give). Also 'SIMCODE at SIMFLUX precision': the loosest SIMCODE
(sig_p) threshold whose true-positive RMS is <= SIMFLUX's current one.
usage: python analysis/papersim/frcopt_table.py [--sets Dexp0.5_v2psf_wsm,...] [--crit gt_first]
output: out/papersim/frcopt/table_<tag>.md / .json"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

FRCD = os.path.join(C.OUT, 'frcopt')
LAB = dict(smlm='SMLM', sf='SIMFLUX', decode='DECODE', v2='SIMCODE')


def fp50(name, k, fam, thr):
    Z = np.load(os.path.join(FRCD, f'{name}_rows.npz'))
    ck = k if k != 'v2' else ('v2_' + fam if fam.startswith('sig') else 'v2_prod')
    e = np.hypot(Z[ck + '__ex'], Z[ck + '__ey']) * C.PX_NM; e = np.where(np.isfinite(e), e, np.inf)
    m = Z[ck + '__key'] <= thr
    return float(np.mean(e[m] > 50)), float(np.mean((e[m] > 20) & (e[m] <= 50)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default=','.join(f'Dexp{d}_v2psf_wsm' for d in (0.5, 1, 2, 4)))
    ap.add_argument('--crit', default='gt_first'); ap.add_argument('--tag', default='wsm')
    a = ap.parse_args()
    out = []; md = [f"# Selection sweep: FRC-optimal selection per method ({a.tag}; GT criterion {a.crit})\n",
                    "kept = % of the method's Fig 2 base set (disjoint windows, own max(σx, σy) < 16.25 nm; SIMCODE p ≥ 0.63). "
                    "GT = GT-FRC (Fig 2 definition, nm); HS = half-split FRC (no GT; first / sustained crossing). "
                    "fp3σ = Fig 2 'false positive' (no ON emitter within 3 own σ); fp50 = no ON emitter within 50 nm; "
                    "imp = error 20-50 nm; rms = true-positive RMS per axis (nm). "
                    "'GT at HS-opt' = the GT-FRC of the selection the half-split FRC picks (the real-data rule).\n",
                    "| set | method / family | current: σ thr, n, GT, fp3σ, fp50, imp, rms | GT-opt: σ thr, kept, n, GT, fp50, imp, rms | HS-opt (first): kept, GT at it | HS-opt (sust): kept, GT at it | band-mean GT-opt: kept, GT |",
                    "|---|---|---|---|---|---|---|"]
    for name in a.sets.split(','):
        fn = os.path.join(FRCD, f'{name}.json')
        if not os.path.exists(fn): print('missing', fn); continue
        R = json.load(open(fn))
        for k, M in R['methods'].items():
            fams = ['sig'] if k != 'v2' else [f for f in ('sig_p', 'sig_all', 'sig_pagg', 'p_sig12') if f in M['fam']]
            for fam in fams:
                rows = M['fam'][fam]; nb = M['n_base']
                if k == 'v2' and fam.startswith('sig'): rc = M['variants'][fam]['at12']; thr_c = 12.0
                elif k == 'v2': rc = M['variants']['sig_p']['at12']; thr_c = 12.0
                else: rc = M['current']; thr_c = rows[0]['thr'] * C.PX_NM
                O = M['opt'][fam]; o = O[a.crit]; ro = rows[o['idx']]
                if fam == 'p_sig12':
                    f50c = f50o = imc = imo = float('nan'); thr_o = f"p≥{ro['thr']:.2f}"
                else:
                    f50c, imc = fp50(name, k, fam, thr_c / C.PX_NM + 1e-9); f50o, imo = fp50(name, k, fam, ro['thr'] + 1e-12); thr_o = f"{ro['thr'] * C.PX_NM:.1f}"
                h1 = rows[O['hs_first']['idx']] if 'hs_first' in O else {}; h2 = rows[O['hs_sust']['idx']] if 'hs_sust' in O else {}
                bb = rows[O['gt_band']['idx']] if 'gt_band' in O else {}
                d = dict(set=name, method=k, family=fam, n_base=nb, current=dict(thr_nm=thr_c, n=rc['n'], gt=rc.get(a.crit), fp3s=rc.get('fp'), fp50=f50c, imp=imc, rms=rc.get('rms_tp_nm'), hs_first=rc.get('hs_first'), hs_sust=rc.get('hs_sust')),
                         gt_opt=dict(thr=ro['thr'], kept=ro['n'] / nb, n=ro['n'], gt=ro.get(a.crit), fp3s=ro.get('fp'), fp50=f50o, imp=imo, rms=ro.get('rms_tp_nm'), hs_first=ro.get('hs_first'), hs_sust=ro.get('hs_sust')),
                         hs_first_opt=dict(kept=h1.get('n', 0) / nb, gt=h1.get(a.crit), hs=h1.get('hs_first')),
                         hs_sust_opt=dict(kept=h2.get('n', 0) / nb, gt=h2.get(a.crit), hs=h2.get('hs_sust')), band_opt=dict(kept=bb.get('n', 0) / nb, gt=bb.get(a.crit)))
                out.append(d)
                c = d['current']; g = d['gt_opt']
                md.append(f"| {name} | {LAB[k]} / {fam} | {c['thr_nm']:.1f}, {c['n']}, {c['gt']:.1f}, {c['fp3s']:.3f}, {c['fp50']:.3f}, {c['imp']:.3f}, {c['rms']:.2f} | "
                          f"{thr_o}, {100 * g['kept']:.0f} %, {g['n']}, {g['gt']:.1f}, {g['fp50']:.3f}, {g['imp']:.3f}, {g['rms']:.2f} | "
                          f"{100 * d['hs_first_opt']['kept']:.0f} %, {d['hs_first_opt']['gt']:.1f} | {100 * d['hs_sust_opt']['kept']:.0f} %, {d['hs_sust_opt']['gt']:.1f} | "
                          f"{100 * d['band_opt']['kept']:.0f} %, {d['band_opt']['gt']:.1f} |")
        # SIMCODE at SIMFLUX precision
        rsf = R['methods']['sf']['current']['rms_tp_nm']; rows = R['methods']['v2']['fam']['sig_p']; nb = R['methods']['v2']['n_base']
        ok = [r for r in rows if r.get('rms_tp_nm', 99) <= rsf]
        if ok:
            r = max(ok, key=lambda r: r['n'])
            md.append(f"| {name} | SIMCODE at SIMFLUX precision (RMS ≤ {rsf:.2f} nm) | – | σ ≤ {r['thr'] * C.PX_NM:.1f}, {100 * r['n'] / nb:.0f} %, {r['n']} (SIMFLUX {R['methods']['sf']['current']['n']}), GT {r[a.crit]:.1f} (SIMFLUX {R['methods']['sf']['current'][a.crit]:.1f}), rms {r['rms_tp_nm']:.2f} | | | |")
            out.append(dict(set=name, method='v2_at_sf_precision', thr=r['thr'], n=r['n'], kept=r['n'] / nb, gt=r[a.crit], rms=r['rms_tp_nm'], sf_n=R['methods']['sf']['current']['n'], sf_gt=R['methods']['sf']['current'][a.crit]))
    fo = os.path.join(FRCD, f'table_{a.tag}')
    open(fo + '.md', 'w').write('\n'.join(md) + '\n'); json.dump(out, open(fo + '.json', 'w'), indent=1, default=float)
    print('\n'.join(md)); print('written', fo + '.md')


if __name__ == '__main__':
    main()
