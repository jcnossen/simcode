"""psf3d retrain (2026-11-11): pre-registered acceptance on the gate scenes rendered with a movie's per-axis PSF (out/WORKLOG.md).
Inputs: p2.dump outputs <out>/psf3d_nets/fe_<psf>/<scene>__s<seed>_<front>.npz (seeds 301-303), fronts
  c1 = det_n1 + hd_sc5nb (current pass 1)      n1 = det_psf3dmix + hd_psf3dmix
  c2 = p2_det + p2_head_long (current pass 2)   n2 = p2det_psf3dmix + p2headL_psf3dmix     n2p = the pair variant (item 4)
each solved offline (analysis/calib/plib.solve, the gate's pattern estimate) with its own sim-sigma calibration.
Per scene and comparison (new A vs reference B), on the detector-matched targets (score >= 0.5, 1 px on x_det):
  A1  paired RMS / CRLB_SF_multi per photon bin (common targets, FOV bootstrap 95 % CI of A - B), zcal, gross c5 (|e|/CRLB > 5, %)
  A2  recall (final positions, 0.5 px) at B's FP count (B at score >= 0.5; A at the best threshold with FP <= FP_B), and both
      at score >= 0.5
usage: python accept.py <psf tag> <scenes, comma> [cal json overrides front=path,...] -> <out>/psf3d_nets/accept_<psf>.json / .txt"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode2 import config

OUT = os.path.join(config.root('out'), 'psf3d_nets')
CAL = dict(c1='calib/calib_sc_hd_sc5nbb15_pm_bright.json', c2='calib/calib_p2L.json',
           n1='calib/calib_sc_hd_psf3dmix.json', n2='calib/calib_p2L_psf3dmix.json', n2p='calib/calib_p2L_psf3dmix_pair.json')
COMP = [('n1', 'c1'), ('n2', 'c2'), ('n2p', 'n2'), ('n2p', 'c2')]
THR = np.r_[np.arange(0.2, 0.9, 0.02), np.arange(0.9, 0.995, 0.005)]


def run_front(sc, f, d, cal):
    R = P.load(sc, f, d=d)
    ti, oi = P.targets(R)
    sol = P.solve(R, oi, cal)
    E = P.errs(R, ti, oi, sol)
    # all candidates (detection sweep): solve every stored candidate
    allc = np.arange(len(R['sc_score']))
    sa = P.solve(R, allc, cal)
    return R, E, sa


def det_sweep(R, sa, fp_ref=None):
    out = {}
    s = R['sc_score']
    d5 = P.detect(R, sa['x'], sa['y'], s >= 0.5)
    out['at0.5'] = {k: v for k, v in d5.items() if k != 'hit'}
    if fp_ref is not None:
        best = None
        for t in THR:
            r = P.detect(R, sa['x'], sa['y'], s >= t)
            if r['FP'] <= fp_ref and (best is None or r['recall'] > best['recall']):
                best = dict({k: v for k, v in r.items() if k != 'hit'}, thr=float(t))
        out['at_fp_ref'] = best
    return out


def main():
    psf, scen = sys.argv[1], sys.argv[2].split(',')
    cal = dict(CAL)
    if len(sys.argv) > 3:
        for kv in sys.argv[3].split(','):
            k, v = kv.split('='); cal[k] = v
    d = os.path.join(OUT, f'fe_{psf}')
    res = {}; L = [f'# psf {psf}: acceptance (A1 paired RMS/CRLB, A2 recall at the reference FP); calibs {cal}']
    for sc in scen:
        F = {}
        for f in cal:
            if not os.path.exists(f'{d}/{sc}__s301_{f}.npz'): continue
            F[f] = run_front(sc, f, d, json.load(open(config.resolve(cal[f]))))
        res[sc] = {}
        for a, b in COMP:
            if a not in F or b not in F: continue
            Ra, Ea, Sa = F[a]; Rb, Eb, Sb = F[b]
            assert len(Ra['gt_x']) == len(Rb['gt_x']) and np.allclose(Ra['gt_x'], Rb['gt_x']), 'GT differs between fronts'
            pr = P.paired(Rb, Ea, Eb)
            db = det_sweep(Rb, Sb); da = det_sweep(Ra, Sa, fp_ref=db['at0.5']['FP'])
            res[sc][f'{a}_vs_{b}'] = dict(paired=pr, det_A=da, det_B=db)
            L.append(f'\n== {sc}: {a} (A) vs {b} (B)')
            for bn, v in pr.items():
                L.append(f'  {bn:>10s} n {v["A"]["n"]:5d}  RMS/CRLB A {v["A"]["rms"]:.3f} B {v["B"]["rms"]:.3f}  d {v["d"]:+.3f} '
                         f'[{v["ci"][0]:+.3f}, {v["ci"][1]:+.3f}]  zcal A {v["A"]["zcal"]:.2f} B {v["B"]["zcal"]:.2f}  '
                         f'c5% A {v["A"]["c5"]:.2f} B {v["B"]["c5"]:.2f}  nm A {v["A"]["nm"]:.2f} B {v["B"]["nm"]:.2f}')
            r5a, r5b = da['at0.5'], db['at0.5']; rf = da.get('at_fp_ref')
            L.append(f'  detect @0.5: A recall {r5a["recall"]:.4f} FP {r5a["FP"]} | B recall {r5b["recall"]:.4f} FP {r5b["FP"]}'
                     + (f' | A at FP <= {r5b["FP"]}: recall {rf["recall"]:.4f} (thr {rf["thr"]:.3f}, FP {rf["FP"]})' if rf else ' | A: no threshold reaches B FP'))
    json.dump(res, open(f'{OUT}/{os.environ.get("ACCEPT_PREFIX", "")}accept_{psf}.json', 'w'), indent=1, default=float)
    open(f'{OUT}/{os.environ.get("ACCEPT_PREFIX", "")}accept_{psf}.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
