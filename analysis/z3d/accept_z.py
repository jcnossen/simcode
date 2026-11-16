"""z3d: a 3D set against its 2D paper model on the averaged-PSF gate scenes (seeds 301-303; the procedure of the 2026-11-11
PSF-matched acceptance, analysis/psf3d_nets/accept.py of b55641c), each front solved offline (analysis/calib/plib.solve, the
gate's pattern estimate) with its own sim-sigma calibration. Per scene, A (3D) vs B (2D):
  x, y   paired RMS / CRLB_SF_multi per GT photon bin on the common detector-matched targets (FOV bootstrap 95 % CI of A - B),
         zcal, gross c5 (% |e| / CRLB > 5), RMS nm
  det    recall / FP of the final positions (0.5 px) at score >= 0.5, and A's recall at B's FP count
  sel12  the paper's selection on all solved candidates: score >= 0.63 and max(sx, sy) <= 12 nm: rows, true positives (matched
         to a target, 0.5 px), FP, RMS nm of the TP
  z (A)  z RMSE (nm) after the solve and of the head's z, GT spread, z-score RMS (e / sz), per GT photon bin and GT height bin
         With <front P1> <cal P1> (pass-2 comparisons; the pass-1 dumps of A in <fe dir A>): z as the pipeline's pass-2 table
         has it since 2026-11-14 = the linked pass-1 candidate's z prior conditioned on the pass-2 solved x, y (z_conditioning.link_pass1
         on the candidates p >= CAND_PMIN of both fronts, same FOV, detector points; z_conditioning.z_from_pass1 with the pass-1
         calibration's prior_off); rows without a partner: z NaN (share reported). The pass-2 head's own z is listed as
         'z (pass-2 head)' for comparison.
usage: python accept_z.py <out prefix> <fe dir A> <front A> <cal A> <fe dir B> <front B> <cal B> <psf calib> <scenes, comma>
       [<front P1> <cal P1>] -> <out prefix>.txt / .json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode import config

THR = np.r_[np.arange(0.2, 0.9, 0.02), np.arange(0.9, 0.995, 0.005)]
ZB = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.0, 1.0)]


def run_front(sc, f, d, cal):
    R = P.load(sc, f, d=d)
    ti, oi = P.targets(R)
    sol = P.solve(R, oi, cal)
    E = P.errs(R, ti, oi, sol)
    sa = P.solve(R, np.arange(len(R['sc_score'])), cal)
    return R, E, sol, sa


def det_sweep(R, sa, fp_ref=None):
    s = R['sc_score']
    out = {'at0.5': {k: v for k, v in P.detect(R, sa['x'], sa['y'], s >= 0.5).items() if k != 'hit'}}
    if fp_ref is not None:
        best = None
        for t in THR:
            r = P.detect(R, sa['x'], sa['y'], s >= t)
            if r['FP'] <= fp_ref and (best is None or r['recall'] > best['recall']):
                best = dict({k: v for k, v in r.items() if k != 'hit'}, thr=float(t))
        out['at_fp_ref'] = best
    return out


def sel12(R, sa):
    m = (R['sc_score'] >= 0.63) & (np.maximum(sa['sx'], sa['sy']) * P.PX <= 12.0)
    r = P.detect(R, sa['x'], sa['y'], m)
    idx = np.nonzero(m)[0]
    from simcode.benchmark.gate_report import match
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], sa['x'][idx], sa['y'][idx], 0.5)
    ok = og >= 0
    e = np.c_[sa['x'][idx][ok] - R['gt_x'][og[ok]], sa['y'][idx][ok] - R['gt_y'][og[ok]]]
    return dict(n=int(m.sum()), TP=r['TP'], FP=r['FP'], recall=r['recall'], rms_nm=float(np.sqrt(np.mean(e ** 2)) * P.PX) if ok.any() else np.nan)


def zstats(R, E, sol, focus, px_um):
    ti = E['ti']; tz = (R['gt_z'][ti] - focus) / px_um
    zh = R['gt_z'][ti]; ph = R['gt_phot'][ti]
    out = {}
    for nm, m0 in [(f'ph{lo}-{hi:g}', (ph >= lo) & (ph < hi)) for lo, hi in P.PHB] + \
                  [(f'z{lo:.2f}-{hi:.2f}', (zh >= lo) & (zh < hi)) for lo, hi in ZB[:-1]]:
        m = m0 & np.isfinite(sol['z'])
        if m.sum() < 20: continue
        e = sol['z'][m] - tz[m]; ep = sol['zp'][m] - tz[m]
        out[nm] = dict(n=int(m.sum()), gt_sd_nm=float(np.std(tz[m]) * P.PX), rmse_nm=float(np.sqrt(np.mean(e ** 2)) * P.PX),
                       prior_rmse_nm=float(np.sqrt(np.mean(ep ** 2)) * P.PX), bias_nm=float(np.mean(e) * P.PX),
                       zrms=float(np.sqrt(np.mean((e / sol['sz'][m]) ** 2))), sd_med_nm=float(np.median(sol['sz'][m]) * P.PX))
    return out


def z_pass1(R2, sol, oi, R1, cal1):
    """z of the pass-2 rows oi from pass 1, as the pipeline (z_conditioning.link_pass1 + z_conditioning.z_from_pass1): pass-1 / pass-2 candidates
    with score >= CAND_PMIN, same FOV, detector points -> sol with z / sz / zp replaced, linked share of the rows oi"""
    import torch
    from simcode.emitter import model as SCM
    from simcode.solve import z_conditioning as ZC
    from simcode.processing.common import CAND_PMIN
    i1 = np.nonzero(R1['sc_score'] >= CAND_PMIN)[0]; i2 = np.nonzero(R2['sc_score'] >= CAND_PMIN)[0]
    j = ZC.link_pass1(R1['sc_b'][i1], R1['sc_x_det'][i1], R1['sc_y_det'][i1], R2['sc_b'][i2], R2['sc_x_det'][i2], R2['sc_y_det'][i2])
    J = np.full(len(R2['sc_score']), -1, np.int64); J[i2] = np.where(j >= 0, i1[np.maximum(j, 0)], -1)
    tt = lambda v: torch.tensor(np.asarray(v), device='cuda', dtype=torch.float32)
    p1 = dict(z=tt(R1['sc_z']), lsz=tt(R1['sc_lsz']), bz=tt(R1['sc_bz']),
              r0=tt(np.c_[R1['sc_x'], R1['sc_y']] - np.asarray(cal1['prior_off'])[None]))
    Z = ZC.z_from_pass1(p1, J[oi], tt(np.c_[sol['x'], sol['y']]), tt(np.c_[sol['sx'], sol['sy']]))
    out = dict(sol, z=Z['z_hy0'].cpu().numpy(), sz=Z['sz_cal'].cpu().numpy(), zp=Z['z_prior'].cpu().numpy())
    return out, float(np.mean(J[oi] >= 0))


def zlines(zs, label):
    return [f'  {label} {bn:>12s} n {v["n"]:5d}  gt sd {v["gt_sd_nm"]:5.1f}  rmse {v["rmse_nm"]:5.1f} (head prior {v["prior_rmse_nm"]:5.1f})'
            f'  bias {v["bias_nm"]:+5.1f}  z-score rms {v["zrms"]:.2f}  sd med {v["sd_med_nm"]:5.1f} nm' for bn, v in zs.items()]


def main():
    pre, dA, fA, cA, dB, fB, cB, calib, scen = sys.argv[1:10]
    f1, c1 = (sys.argv[10:12] + ['', ''])[:2]
    C1 = json.load(open(config.resolve(c1))) if f1 else None
    from simcode.psf import read_calib
    c = read_calib(config.resolve(calib)); focus, px_um = c['focus_um'], c['pixelsize_um']
    CA, CB = json.load(open(config.resolve(cA))), json.load(open(config.resolve(cB)))
    res = {}; L = [f'# 3D {fA} ({dA}, cal {cA}) vs 2D {fB} ({dB}, cal {cB}); psf {calib}']
    for sc in scen.split(','):
        Ra, Ea, Sa_t, Sa = run_front(sc, fA, dA, CA); Rb, Eb, _, Sb = run_front(sc, fB, dB, CB)
        assert len(Ra['gt_x']) == len(Rb['gt_x']) and np.allclose(Ra['gt_x'], Rb['gt_x']), 'GT differs between fronts'
        pr = P.paired(Rb, Ea, Eb)
        db = det_sweep(Rb, Sb); da = det_sweep(Ra, Sa, fp_ref=db['at0.5']['FP'])
        s12a, s12b = sel12(Ra, Sa), sel12(Rb, Sb)
        zs = zstats(Ra, Ea, Sa_t, focus, px_um) if 'z' in Sa_t else {}
        zh, lk = None, None
        if f1 and 'z' in Sa_t:      # the table's z (pass 1 | pass-2 x, y); the pass-2 head's z kept as zh for comparison
            zh = zs
            S1, lk = z_pass1(Ra, Sa_t, Ea['oi'], P.load(sc, f1, d=dA), C1)
            zs = zstats(Ra, Ea, S1, focus, px_um)
        res[sc] = dict(paired=pr, det_A=da, det_B=db, sel12_A=s12a, sel12_B=s12b, z_A=zs)
        if zh is not None: res[sc].update(z_A_p2head=zh, z_A_linked=lk)
        L.append(f'\n== {sc}: {fA} (A, 3D) vs {fB} (B, 2D)')
        for bn, v in pr.items():
            L.append(f'  {bn:>10s} n {v["A"]["n"]:5d}  RMS/CRLB A {v["A"]["rms"]:.3f} B {v["B"]["rms"]:.3f}  d {v["d"]:+.3f} '
                     f'[{v["ci"][0]:+.3f}, {v["ci"][1]:+.3f}]  zcal A {v["A"]["zcal"]:.2f} B {v["B"]["zcal"]:.2f}  '
                     f'c5% A {v["A"]["c5"]:.2f} B {v["B"]["c5"]:.2f}  nm A {v["A"]["nm"]:.2f} B {v["B"]["nm"]:.2f}')
        r5a, r5b = da['at0.5'], db['at0.5']; rf = da.get('at_fp_ref')
        L.append(f'  detect @0.5: A recall {r5a["recall"]:.4f} FP {r5a["FP"]} | B recall {r5b["recall"]:.4f} FP {r5b["FP"]}'
                 + (f' | A at FP <= {r5b["FP"]}: recall {rf["recall"]:.4f} (thr {rf["thr"]:.3f}, FP {rf["FP"]})' if rf else ' | A: no threshold reaches B FP'))
        L.append(f'  sel12 (p >= 0.63, max(sx, sy) <= 12 nm): A n {s12a["n"]} TP {s12a["TP"]} FP {s12a["FP"]} rms {s12a["rms_nm"]:.2f} nm | '
                 f'B n {s12b["n"]} TP {s12b["TP"]} FP {s12b["FP"]} rms {s12b["rms_nm"]:.2f} nm')
        if zh is not None:
            L.append(f'  z = pass 1 | pass-2 x, y (the table): {100 * lk:.2f} % of the targets linked to a pass-1 candidate (others NaN)')
        L += zlines(zs, 'z')
        if zh is not None:
            L += zlines(zh, 'z (pass-2 head)')
    os.makedirs(os.path.dirname(os.path.abspath(pre)), exist_ok=True)
    json.dump(res, open(pre + '.json', 'w'), indent=1, default=float)
    open(pre + '.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
