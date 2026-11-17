"""sigma study Part B, item 3 (2026-11-17): the pass-2 solve combines the pass-2 network's position prior with the pass-2
network's intensities as if independent; their errors are correlated (partb_gate.py: rho 0.4-0.7). Candidate fix A: the
pattern-free PASS-1 prior of the same target with the pass-2 intensities. Same gate scenes / seeds, targets matched in both
dumps (gate_stats.targets, common GT index). Per network-photon bin of the pass-2 intensities:
  rho(prior error, linearised pattern-only error of the pass-2 intensities) for the pass-2 and the pass-1 prior,
  the independent combination (what the solve does) with each prior: actual robust sd, claimed sd, zcal,
  the BLUE with the measured 2 x 2 covariance (pass-2 prior) for reference.
usage: python partb_crossprior.py <pass-1 dump dir> <front1> <pass-2 dump dir> <front2> <scenes> <cal1> <cal2> <out json>"""
import json, sys
import numpy as np
from simcode.calibration import gate_stats as P
sys.path.insert(0, __import__('os').path.dirname(__file__))
from partb_gate import linear_pattern, rob, rcorr

PX = P.PX


def main():
    d1, f1, d2, f2, scen, c1, c2, outf = sys.argv[1:9]
    cal1, cal2 = json.load(open(c1)), json.load(open(c2))
    res = {}
    for sc in scen.split(','):
        R1 = P.load(sc, f1, d=d1); R2 = P.load(sc, f2, d=d2)
        assert np.array_equal(R1['gt_x'], R2['gt_x']) and np.array_equal(R1['gt_fov'], R2['gt_fov'])
        t1, o1 = P.targets(R1); t2, o2 = P.targets(R2)
        com, i1, i2 = np.intersect1d(t1, t2, return_indices=True)
        o1, o2, ti = o1[i1], o2[i2], com
        gt = np.c_[R2['gt_x'][ti], R2['gt_y'][ti]]
        e_pat, s_pat = linear_pattern(R2, ti, o2, c=float(cal2['cmult']), ca=float(cal2['cmult_ang']))
        pri = dict(p2=(np.c_[R2['sc_x'][o2], R2['sc_y'][o2]] - np.array(cal2['prior_off']) - gt, np.exp(R2['sc_lsxy'][o2])),
                   p1=(np.c_[R1['sc_x'][o1], R1['sc_y'][o1]] - np.array(cal1['prior_off']) - gt, np.exp(R1['sc_lsxy'][o1])))
        ph = R2['sc_I'][o2].sum(1)
        rows = []
        for lo, hi in zip(P.EDG[:-1], P.EDG[1:]):
            m = (ph >= lo) & (ph < hi)
            if m.sum() < 100: continue
            r = dict(ph_lo=lo, ph_hi=hi, n=int(m.sum()))
            kpat = rob((e_pat[m] / s_pat[m]).ravel())
            for k, (ep, sp) in pri.items():
                kp = rob((ep[m] / sp[m]).ravel())
                r[f'rho_{k}'] = [rcorr(ep[m, a], e_pat[m, a]) for a in range(2)]
                for tag, a_, b_ in (('raw', sp[m], s_pat[m]), ('calin', sp[m] * kp, s_pat[m] * kpat)):
                    w1, w2 = 1 / a_ ** 2, 1 / b_ ** 2
                    ec = (w1 * ep[m] + w2 * e_pat[m]) / (w1 + w2); scl = 1 / np.sqrt(w1 + w2)
                    r[f'{k}_{tag}'] = dict(sd_nm=rob(ec.ravel()) * PX, claimed_nm=float(np.median(scl)) * PX, zcal=rob((ec / scl).ravel()))
                r[f'{k}_prior_sd_nm'] = rob(ep[m].ravel()) * PX
            r['pat_sd_nm'] = rob(e_pat[m].ravel()) * PX
            ep = pri['p2'][0]
            Cs = [np.cov(np.c_[ep[m, a], e_pat[m, a]][(np.abs(ep[m, a]) < 4 * rob(ep[m, a])) & (np.abs(e_pat[m, a]) < 4 * rob(e_pat[m, a]))].T) for a in range(2)]
            bl = []
            for a in range(2):
                Ci = np.linalg.inv(Cs[a]); w = Ci.sum(1) / Ci.sum(); bl.append(rob(w[0] * ep[m, a] + w[1] * e_pat[m, a]))
            r['blue_p2_sd_nm'] = float(np.mean(bl)) * PX
            rows.append(r)
        res[sc] = rows
        print(f'== {sc}: {len(ti)} targets in both dumps')
        print(f'{"ph":>10s} {"n":>5s} {"rho p2":>11s} {"rho p1":>11s} {"prior sd p2/p1":>14s} {"pat sd":>6s} | {"p2 prior: sd/claim zcal (cal-in)":>32s} | {"p1 prior: sd/claim zcal (cal-in)":>32s} | {"BLUE p2":>7s}')
        for r in rows:
            a, b = r['p2_calin'], r['p1_calin']
            print(f'{r["ph_lo"]:>5g}-{r["ph_hi"]:<5g}{r["n"]:5d} {r["rho_p2"][0]:5.2f},{r["rho_p2"][1]:5.2f} {r["rho_p1"][0]:5.2f},{r["rho_p1"][1]:5.2f} '
                  f'{r["p2_prior_sd_nm"]:6.1f}/{r["p1_prior_sd_nm"]:6.1f} {r["pat_sd_nm"]:6.2f} | {a["sd_nm"]:6.2f}/{a["claimed_nm"]:6.2f} {a["zcal"]:5.2f} (raw {r["p2_raw"]["zcal"]:4.2f})      | '
                  f'{b["sd_nm"]:6.2f}/{b["claimed_nm"]:6.2f} {b["zcal"]:5.2f} (raw {r["p1_raw"]["zcal"]:4.2f})      | {r["blue_p2_sd_nm"]:6.2f}')
    json.dump(res, open(outf, 'w'), indent=1)


if __name__ == '__main__':
    main()
