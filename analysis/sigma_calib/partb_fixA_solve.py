"""sigma study Part B (2026-11-17), candidate fix A in the full nonlinear solve: the pass-2 pattern solve (estimated pattern,
kinetics, fringe posterior; partb_gate.solve_full) of the pass-2 intensities with (a) the pass-2 network's own position prior
(production) or (b) the PASS-1 network's prior of the same target (pattern-free; pass-1 prior_off removed). Both without any
factor (kap = 1, sig_a = 1, sig_g = 0, sig_ph = 1; c as calibrated) and with the production pass-2 calibration. Per
network-photon bin: robust zcal (both axes), robust error sd (nm), median reported sigma (nm), start-spread variance fraction.
usage: python partb_fixA_solve.py <pass-1 dump dir> <front1> <pass-2 dump dir> <front2> <scenes> <cal1> <cal2> <out json>"""
import json, os, sys
import numpy as np
from simcode.calibration import gate_stats as P
sys.path.insert(0, os.path.dirname(__file__))
from partb_gate import solve_full, nocal, rob

PX = P.PX


def main():
    d1, f1, d2, f2, scen, c1, c2, outf = sys.argv[1:9]
    cal1, cal2 = json.load(open(c1)), json.load(open(c2))
    res = {}
    for sc in scen.split(','):
        R1 = P.load(sc, f1, d=d1); R2 = P.load(sc, f2, d=d2)
        t1, o1 = P.targets(R1); t2, o2 = P.targets(R2)
        com, i1, i2 = np.intersect1d(t1, t2, return_indices=True)
        o1, o2, ti = o1[i1], o2[i2], com
        gt = np.c_[R2['gt_x'][ti], R2['gt_y'][ti]]
        # (b): a copy of the pass-2 dump rows with the pass-1 prior (x, y, lsxy), shifted so that the pass-2 calibration's
        # prior_off (removed in the solve) leaves the pass-1 prior with the pass-1 prior_off removed
        Rb = {k: (v[o2] if (np.ndim(v) and len(v) == len(R2['sc_x'])) else v) for k, v in R2.items()}
        sh = np.array(cal2['prior_off']) - np.array(cal1['prior_off'])
        Rb['sc_x'] = R1['sc_x'][o1] + sh[0]; Rb['sc_y'] = R1['sc_y'][o1] + sh[1]; Rb['sc_lsxy'] = R1['sc_lsxy'][o1]
        Ra = {k: (v[o2] if (np.ndim(v) and len(v) == len(R2['sc_x'])) else v) for k, v in R2.items()}
        idx = np.arange(len(o2))
        S = {}
        for pr, RR in (('p2prior', Ra), ('p1prior', Rb)):
            for cn, cc in (('raw', nocal(cal2)), ('prod', cal2)):
                S[f'{pr}_{cn}'] = solve_full(RR, idx, cc)
        ph = R2['sc_I'][o2].sum(1)
        rows = []
        for lo, hi in zip(P.EDG[:-1], P.EDG[1:]):
            m = (ph >= lo) & (ph < hi)
            if m.sum() < 100: continue
            r = dict(ph_lo=lo, ph_hi=hi, n=int(m.sum()))
            for k, s in S.items():
                e = s['r'][m] - gt[m]
                r[k] = dict(zcal=rob((e / s['s_cal'][m]).ravel()), sd_nm=rob(e.ravel()) * PX, sig_med_nm=float(np.median(s['s_cal'][m])) * PX,
                            spread_frac=float(np.median((s['spread'][m] ** 2 / np.maximum(s['s_cal'][m] ** 2, 1e-12)).ravel())),
                            gross50_pct=float(100 * np.mean(np.hypot(*e.T) * PX > 50)))
            rows.append(r)
        res[sc] = rows
        print(f'== {sc}: {len(ti)} targets; zcal / error sd nm / median sigma nm / spread var frac / gross>50nm %', flush=True)
        for r in rows:
            print(f'{r["ph_lo"]:>5g}-{r["ph_hi"]:<5g}{r["n"]:5d} ' + ' | '.join(
                f'{k} {r[k]["zcal"]:.3f} {r[k]["sd_nm"]:5.2f} {r[k]["sig_med_nm"]:5.2f} {r[k]["spread_frac"]:.2f} {r[k]["gross50_pct"]:.1f}' for k in S), flush=True)
    json.dump(res, open(outf, 'w'), indent=1)


if __name__ == '__main__':
    main()
