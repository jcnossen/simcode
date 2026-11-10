"""critic 34 #5: repaired SIMCODE on the pass-2 gate scenes (pair1, pair1.5, rs05qm, rs25qm), eval seeds 301-303, next to
p1 / p2L of the development gate_long.py and analysis/gate/merged_d1.py.
SIMCODE = python -m simcode2.eff.simcode_gate with the paper network per density (ni6_3 ckpt 3 = 0.5 nM paper network for the
0.5 nM-preset scenes pair1 / pair1.5 / rs05qm; L64_5 ckpt 3 = 2.5 nM paper network for rs25qm), fed the gate's own patt_est
(no SIMCODE pattern / drift estimation, hence no drift-sign issue: 'repaired'); position = eq.-2 NDI + MVU merge (x_sc).
  SC     paper filter chain (crlb 0.2 px, mod_error < 0.1, |ndi - net| < 0.2 px) -> tag _scode_<net>
  SCnf   no filters, every p > 0.7 detection -> tag _scodenf_<net>
GT: the gate npz (reused from _hd_sc5nb15pm) is checked identical to the p2 dumps (same gen loop); SIMCODE outputs are put
on the p2 dump's GT. Matching / bins / CIs / zcal exactly as gate_long.py (targets: score >= 0.5, 1 px one-to-one on x_det;
resolvable; FOV bootstrap 200). SIMCODE sigma for zcal = its merged MVU sigma (s_sc). No global offset removed (SIMCODE's
offset is < 1 nm on these scenes, simcode_base/gate_paired_offset.txt).
Inputs: pass-2 gate dumps <out>/p2/fe (p1 / p2L fronts, simcode2/p2/dump.py; analysis/calib/plib.py), SIMCODE gate npz
<gate dir>/<scene>__s30[123]_scode(nf)_<net>.npz (python -m simcode2.eff.simcode_gate --model ni6_3 | L64_5 --scen ... writes both tags; gate dir =
$SIMCODE2_GATE_DIR or <out>/gate), optional <out_dir>/merged_d1.json (analysis/gate/merged_d1.py, ideal_C column).
Calibrations: calib/calib_sc_hd_sc5nbb15_pm_bright.json (p1), calib/calib_p2L.json (p2L).
Development p2/long/vs_simcode.py; snapshots analysis/results/gate/{gate_long,merged_d1}_vs_simcode.txt.
-> <out_dir>/gate_long_vs_simcode.txt/json, merged_d1_vs_simcode.txt/json (out_dir default <out>/p2/long)
usage: python analysis/gate/vs_simcode_p2.py [scenes (pair1,pair1.5,rs05qm,rs25qm)] [out_dir]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2eval as E        # analysis/gate/p2eval.py (cals, miss_anatomy)
P = E.P                   # analysis/calib/plib.py
from simcode2 import config

SCEN = (sys.argv[1] if len(sys.argv) > 1 else 'pair1,pair1.5,rs05qm,rs25qm').split(',')
D = sys.argv[2] if len(sys.argv) > 2 else config.out_dir('p2', 'long')
GD = os.environ.get('SIMCODE2_GATE_DIR') or os.path.join(config.root('out'), 'gate')
NET = {'pair1': 'ni6_3', 'pair1.5': 'ni6_3', 'rs05qm': 'ni6_3', 'rs25qm': 'L64_5'}
C = E.cals()
FR = dict(p1=('p1', 'p1'), p2L=('p2L', 'p2L'))
PAIRS = [('SC', 'p2L'), ('SC', 'p1'), ('SCnf', 'p2L')]
THR = np.r_[np.arange(0.3, 0.9, 0.02), np.arange(0.9, 1.0, 0.005)]


def unp_ci(R, e, nboot=200):
    ok = P.resolvable(R, e['ti']) & np.isfinite(e['e']).all(1)
    ph = R['gt_phot'][e['ti']]; c = R['gt_sf_m'][e['ti']]; fov = R['gt_fov'][e['ti']]
    out = {}
    for lo, hi in P.PHB:
        m = ok & (ph >= lo) & (ph < hi)
        if m.sum() < 20: continue
        st = P.stats(e['e'][m], c[m], e['s'][m])
        bs = [P.rr(e['e'][m], c[m], w) for w in P.boot_w(fov[m], nboot)]
        st['ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[f'{lo}-{hi:g}'] = st
    return out


def on_gt(Rg, Rs):
    """SIMCODE outputs Rs (gate npz, pooled) on the p2 dump GT Rg (checked identical)"""
    assert len(Rs['gt_x']) == len(Rg['gt_x']) and np.allclose(Rs['gt_x'], Rg['gt_x'], atol=1e-4) and np.array_equal(Rs['gt_fov'], Rg['gt_fov']), 'GT mismatch'
    assert int(Rs['n_fov']) == int(Rg['n_fov'])
    R = {k: v for k, v in Rg.items() if not k.startswith('sc_')}
    R.update({k: v for k, v in Rs.items() if k.startswith('sc_')})
    return R


def sc_sol(R, oi):
    return dict(x=R['sc_x_sc'][oi], y=R['sc_y_sc'][oi], sx=R['sc_s_sc'][oi, 0], sy=R['sc_s_sc'][oi, 1], xp=R['sc_x_prior'][oi], yp=R['sc_y_prior'][oi])


def anatomy(R, xo, yo, sel):
    """matched / merged / undetected (as merged_d1.py)"""
    idx = np.nonzero(sel)[0]
    og = P.match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], xo[idx], yo[idx], 1.0)
    cand_of = np.full(len(R['gt_x']), -1); cand_of[og[og >= 0]] = idx[og >= 0]
    st = (cand_of >= 0).astype(int)
    _, nbr = E.miss_anatomy(R)
    for g in np.nonzero(st == 0)[0]:
        q = nbr[g]
        if q >= 0 and cand_of[q] >= 0:
            c = cand_of[q]
            if np.hypot(xo[c] - R['gt_x'][g], yo[c] - R['gt_y'][g]) < 1.0: st[g] = 2
    return st


L = ['# critic 34 #5: repaired SIMCODE (paper networks: ni6_3 for pair1 / pair1.5 / rs05qm, L64_5 for rs25qm; gate patt_est; eq.-2 NDI + MVU)',
     '# vs pass 1 (p1) and the adopted pass-2 head (p2L), eval seeds 301-303, RMS/CRLB_SF_multi [95 % FOV-bootstrap CI], n = targets in the bin.',
     '# SC = SIMCODE with its paper filter chain, SCnf = no filters. zcal = robust SD of e / sigma (SIMCODE: merged MVU sigma).',
     '# det: recall / FP per FOV / JI at p >= 0.5 on the final positions (0.5 px, gate_long rule); p2L@FP_SC = p2L recall at the',
     '# lowest score threshold whose FP <= SIMCODE\'s FP.']
M = ['# critic 34 #5: d <= 1 px targets (>= 200 label ph, eval seeds 301-303): matched / merged / undetected (1 px one-to-one, p >= 0.5)',
     '# on the final position (SIMCODE: merged x_sc; ours: explicit solve) and on the detector point (SIMCODE: its network position).',
     '# p1 / p2L rows recomputed as merged_d1.py; ideal_C = the pattern-known NP oracle bound from merged_d1.json.']
REP, REPM = {}, {}
MD1 = json.load(open(f'{D}/merged_d1.json')) if os.path.exists(f'{D}/merged_d1.json') else {}
for sc in SCEN:
    Rf = {n: P.load(sc, f) for n, (f, _) in FR.items()}
    Rg = Rf['p2L']
    Rf['SC'] = on_gt(Rg, P.load(sc, f'scode_{NET[sc]}', d=GD))
    Rf['SCnf'] = on_gt(Rg, P.load(sc, f'scodenf_{NET[sc]}', d=GD))
    Ev, DET, FIN = {}, {}, {}
    for n, R in Rf.items():
        ti, oi = P.targets(R)
        if n.startswith('SC'):
            Ev[n] = P.errs(R, ti, oi, sc_sol(R, oi))
            xf, yf = R['sc_x_sc'].copy(), R['sc_y_sc'].copy()
            ia5 = np.nonzero(R['sc_score'] >= 0.5)[0]
        else:
            Ev[n] = P.errs(R, ti, oi, P.solve(R, oi, C[FR[n][1]], 'own'))
            ia = np.nonzero(R['sc_score'] >= 0.3)[0]
            sa = P.solve(R, ia, C[FR[n][1]], 'own')
            xf = np.full(len(R['sc_b']), np.nan); yf = xf.copy(); xf[ia] = sa['x']; yf[ia] = sa['y']
        FIN[n] = (xf, yf)
        DET[n] = P.detect(R, xf, yf, R['sc_score'] >= 0.5)
    # p2L recall at SIMCODE's FP
    R2 = Rf['p2L']; xf2, yf2 = FIN['p2L']; sweep = []
    for t in THR:
        d_ = P.detect(R2, xf2, yf2, R2['sc_score'] >= t); sweep.append((t, d_['FP'], d_['recall']))
    atfp = {}
    for n in ('SC', 'SCnf'):
        ok = [s for s in sweep if s[1] <= DET[n]['FP']]
        atfp[n] = dict(thr=ok[0][0], FP=ok[0][1], recall=ok[0][2]) if ok else None
    budget = {}
    for fb in (0.005, 0.01, 0.025, 0.05):
        ok = [s for s in sweep if s[1] <= fb * int(R2['n_fov'])]
        budget[fb] = dict(thr=ok[0][0], FP=ok[0][1], recall=ok[0][2]) if ok else None
    rep = dict(det={n: {k: v for k, v in d.items() if k != 'hit'} for n, d in DET.items()}, p2L_at_fp=atfp, p2L_at_fpfov=budget, paired={}, unpaired={})
    L.append(f'\n=== {sc} (SIMCODE net {NET[sc]}; targets {int((Rg["gt_target"] > 0).sum())}, FOVs {Rg["n_fov"]})')
    L.append('  det p>=0.5: ' + ' | '.join(f'{n} rec {d["recall"]:.3f} FP/FOV {d["fp_per_fov"]:.3f} JI {d["JI"]:.3f}' for n, d in DET.items()))
    L.append('  p2L at SIMCODE FP: ' + ' | '.join(f'@{n} (FP {DET[n]["FP"]}): ' + (f'thr {a["thr"]:.3f} FP {a["FP"]} rec {a["recall"]:.3f} (vs {DET[n]["recall"]:.3f})' if a else 'not reachable')
                                                for n, a in atfp.items()))
    L.append('  p2L at FP/FOV budget: ' + ' | '.join(f'<= {fb:g}: ' + (f'thr {a["thr"]:.3f} FP/FOV {a["FP"] / int(R2["n_fov"]):.4f} rec {a["recall"]:.3f}' if a else '-') for fb, a in budget.items()))
    L.append('  unpaired (own matched set): bin n rms [CI] zcal g4%')
    for n in ('SC', 'SCnf', 'p1', 'p2L'):
        u = unp_ci(Rf[n], Ev[n]); rep['unpaired'][n] = u
        L.append(f'    {n:4s} ' + ' | '.join(f'{b}: n {q["n"]} {q["rms"]:.3f} [{q["ci"][0]:.3f},{q["ci"][1]:.3f}] z{q["zcal"]:.2f} g{q["g4"]:.1f}' for b, q in u.items()))
    L.append('  paired (same targets): A - B, bin n rms_A vs rms_B d [CI] | zcal A / B | c5 A / B (% > 5 CRLB)')
    for a, b in PAIRS:
        pr = P.paired(Rg, Ev[a], Ev[b], bins=P.PHB, key='ph')
        rep['paired'][f'{a}-{b}'] = pr
        for bn, q in pr.items():
            L.append(f'    {a}-{b:4s} {bn:>12s} n {q["A"]["n"]:5d} {q["A"]["rms"]:.3f} vs {q["B"]["rms"]:.3f} d {q["d"]:+.3f} '
                     f'[{q["ci"][0]:+.3f},{q["ci"][1]:+.3f}] | zcal {q["A"]["zcal"]:.2f} / {q["B"]["zcal"]:.2f} | c5 {q["A"]["c5"]:.2f} / {q["B"]["c5"]:.2f}')
    REP[sc] = rep
    # ---- merged d <= 1 px
    if sc == 'pair1.5': continue
    ST = {}
    for n in ('SC', 'SCnf', 'p1', 'p2L'):
        R = Rf[n]; sel = R['sc_score'] >= 0.5
        ST[n + '|xdet'] = anatomy(R, R['sc_x_det'], R['sc_y_det'], sel)
        xf, yf = FIN[n]
        ST[n + '|fin'] = anatomy(R, xf, yf, sel & np.isfinite(xf))
    tg = (Rg['gt_target'] > 0) & (Rg['gt_phot'] >= 200)
    M.append(f'\n=== {sc} (SIMCODE net {NET[sc]})')
    for lo, hi in ((0, 0.5), (0.5, 1.0), (0, 1.0)):
        for plo, phi in ((200, 700), (700, 2000), (2000, 1e9), (200, 1e9)):
            m = tg & (Rg['gt_nn'] >= lo) & (Rg['gt_nn'] < hi) & (Rg['gt_phot'] >= plo) & (Rg['gt_phot'] < phi)
            if m.sum() < 10: continue
            r = dict(n=int(m.sum()))
            s_ = f'  d {lo:g}-{hi:g} ph {plo}-{phi:g} n {m.sum():5d}'
            for k in ('SC|fin', 'SCnf|fin', 'p1|fin', 'p2L|fin', 'SC|xdet', 'SCnf|xdet', 'p2L|xdet'):
                st = ST[k]; r[k] = [int((st[m] == 1).sum()), int((st[m] == 2).sum()), int((st[m] == 0).sum())]
                s_ += f' | {k} {r[k][0]}/{r[k][1]}/{r[k][2]}'
            q = MD1.get(f'{sc}|d{lo:g}-{hi:g}|{plo}-{phi:g}', {})
            if 'ideal_C' in q:
                r['ideal_C'] = q['ideal_C']; s_ += f' || ideal_C {q["ideal_C"]:.2f} (x n = {q["ideal_C"] * r["n"]:.0f})'
            REPM[f'{sc}|d{lo:g}-{hi:g}|{plo}-{phi:g}'] = r
            M.append(s_)
open(f'{D}/gate_long_vs_simcode.txt', 'w').write('\n'.join(L) + '\n')
json.dump(REP, open(f'{D}/gate_long_vs_simcode.json', 'w'), indent=1, default=float)
open(f'{D}/merged_d1_vs_simcode.txt', 'w').write('\n'.join(M) + '\n')
json.dump(REPM, open(f'{D}/merged_d1_vs_simcode.json', 'w'), indent=1, default=float)
print('\n'.join(L)); print('\n'.join(M))
