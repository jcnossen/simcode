"""CRLB comparison (2026-11-13), stage 1: per scene x seed x pass, join the p2.dump front-end dumps of the m1809 set
(out/psfmatch/fe_m1809a/<scen>__s<seed>_s1.npz = pass 1 det_m1809 + hd_m1809, _s2 = pass 2 p2det_m1809 + p2headL_m1809)
with the full CRLB table of the same GT (out/psfmatch/gate_m1809a/<scen>__s<seed>_hd_cur.npz: sm_m, sm_i, sf_m, sf_i, fr_m;
GT checked identical), run the explicit pattern solve offline (production sim calibration of the set; variant c0 = cmult = cmult_ang = 0),
and match every GT target to the candidates (score >= 0.5) on the DETECTOR point in three ways:
  h1  Hungarian one-to-one within 1 px (the gate_report_detector_matched rule)
  h3  Hungarian one-to-one within 3 px
  n3  nearest candidate within 3 px (not one-to-one)
(all GT emitters incl. don't-care take part in the assignment; target rows kept).
Output out/crlbcmp/cache/<scen>__s<seed>_p<1|2>.npz.
usage: python analysis/crlbcmp/prep.py [scenes] [seeds] [dev]"""
import os, sys, json, time
import numpy as np, torch
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'calib'))
import plib as P
from simcode import config

FE = os.path.join(REPO, 'out', 'psfmatch', 'fe_m1809a')
GD = os.path.join(REPO, 'out', 'psfmatch', 'gate_m1809a')
OUT = os.path.join(REPO, 'out', 'crlbcmp', 'cache')
CAL = {1: 'calib/calib_sc_hd_m1809.json', 2: 'calib/calib_p2L_m1809.json'}
FRONT = {1: 's1', 2: 's2'}


def match_all(gf, gx, gy, cb, cx, cy, r, hung=True):
    """per GT row: index of the matched candidate or -1"""
    gm = np.full(len(gf), -1)
    og, oc = np.argsort(gf, kind='stable'), np.argsort(cb, kind='stable')
    gs, cs = gf[og], cb[oc]
    for f in np.unique(gs):
        a0, a1 = np.searchsorted(gs, [f, f + 1]); b0, b1 = np.searchsorted(cs, [f, f + 1])
        if b1 <= b0: continue
        ig, ic = og[a0:a1], oc[b0:b1]
        D = np.hypot(gx[ig, None] - cx[None, ic], gy[ig, None] - cy[None, ic])
        if hung:
            C = np.where(D < r, D ** 2, 1e6)
            rr, cc = linear_sum_assignment(C)
            ok = C[rr, cc] < r * r
            gm[ig[rr[ok]]] = ic[cc[ok]]
        else:
            j = D.argmin(1); ok = D[np.arange(len(ig)), j] < r
            gm[ig[ok]] = ic[j[ok]]
    return gm


def run(sc, seed, dev):
    G = dict(np.load(f'{GD}/{sc}__s{seed}_hd_cur.npz'))
    for p in (1, 2):
        fo = f'{OUT}/{sc}__s{seed}_p{p}.npz'
        if os.path.exists(fo): print('exists', fo); continue
        t0 = time.time()
        R = dict(np.load(f'{FE}/{sc}__s{seed}_{FRONT[p]}.npz'))
        assert len(R['gt_x']) == len(G['gt_x']) and np.allclose(R['gt_x'], G['gt_x'], atol=1e-4) and np.allclose(R['gt_phot'], G['gt_phot'])
        cal = json.load(open(config.resolve(CAL[p])))
        cand = np.nonzero(R['sc_score'] >= 0.5)[0]
        s = P.solve(R, cand, cal, dev=dev)
        cal0 = dict(cal, cmult=0.0, cmult_ang=0.0)
        s0 = P.solve(R, cand, cal0, dev=dev)
        st = P.solve(R, cand, cal, patt='true', dev=dev) if sc.startswith('rs') else s
        D = {k: R[k] for k in R if k.startswith('gt_') or k in ('patt_est', 'patt_true', 'n_fov')}
        D.update({k: G[k] for k in ('gt_sm_m', 'gt_sm_i', 'gt_sf_m', 'gt_sf_i', 'gt_fr_m')})
        D['cand'] = cand
        for k in ('I', 'L', 'x', 'y', 'lsxy', 'bg', 'b', 'x_det', 'y_det', 'score'):
            D['c_' + k] = R['sc_' + k][cand]
        D.update(c_xs=s['x'], c_ys=s['y'], c_sxs=s['sx'], c_sys=s['sy'], c_q=s['q'], c_xs0=s0['x'], c_ys0=s0['y'],
                 c_xst=st['x'], c_yst=st['y'])
        for nm, r, h in (('h1', 1.0, True), ('h3', 3.0, True), ('n3', 3.0, False)):
            D['m_' + nm] = match_all(R['gt_fov'], R['gt_x'], R['gt_y'], D['c_b'], D['c_x_det'], D['c_y_det'], r, h)
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(fo, **D)
        print(f'{sc} s{seed} p{p}: {len(cand)} cand, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    scen = (sys.argv[1] if len(sys.argv) > 1 else 'iso,pair2,rs05qm,rs05pm').split(',')
    seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else '301,302,303').split(',')]
    dev = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
    for sc in scen:
        for sd in seeds: run(sc, sd, dev)
