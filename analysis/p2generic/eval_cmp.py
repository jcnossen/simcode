"""Generic pass 2 (2026-11-13): head-to-head of front-ends on IDENTICAL gate-scene targets from p2.dump outputs in one directory
(procedure of the psfmatch acceptance, analysis/psfmatch/accept3.py): per scene (seeds 301-303 pooled) and comparison A vs B,
  RMS / CRLB per GT-photon bin on the targets found by both (score >= 0.5, 1 px on x_det), A - B with FOV-bootstrap 95 % CI, and
  the relative difference (A - B) / B; zcal; nm
  detection: recall (final positions, 0.5 px) at score 0.5 and FP count; A's recall at the best threshold with FP <= B's FP
Each front solved with its own sim calibration (front=calib).
usage: python eval_cmp.py <dir> <scenes> <A:B,...> <front=calib,...> [out.json]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode import config
from simcode.util.proctitle import set_title
set_title('p2gen-eval')
BINS = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (200, 1e9)]
THR = np.r_[np.arange(0.2, 0.9, 0.02), np.arange(0.9, 0.995, 0.005)]


def front(d, sc, f, cal):
    R = P.load(sc, f, d=d); ti, oi = P.targets(R); E = P.errs(R, ti, oi, P.solve(R, oi, cal))
    sa = P.solve(R, np.arange(len(R['sc_score'])), cal)
    return R, E, sa


def det(R, sa, fp_ref=None):
    s = R['sc_score']; d5 = P.detect(R, sa['x'], sa['y'], s >= 0.5); out = dict(recall=d5['recall'], FP=d5['FP'])
    if fp_ref is not None:
        best = None
        for t in THR:
            r = P.detect(R, sa['x'], sa['y'], s >= t)
            if r['FP'] <= fp_ref and (best is None or r['recall'] > best['recall']): best = dict(recall=r['recall'], FP=r['FP'], thr=float(t))
        out['at_fp'] = best
    return out


if __name__ == '__main__':
    d, scen, comp, cals = sys.argv[1], sys.argv[2].split(','), [c.split(':') for c in sys.argv[3].split(',')], sys.argv[4]
    CAL = {kv.split('=')[0]: kv.split('=')[1] for kv in cals.split(',')}
    CJ = {k: json.load(open(config.resolve(v))) for k, v in CAL.items()}
    print(f'# {d}: A vs B on identical targets; calibs {CAL}')
    res = {}
    for sc in scen:
        F = {}
        for a, b in comp:
            for f in (a, b):
                if f not in F: F[f] = front(d, sc, f, CJ[f])
        for a, b in comp:
            (Ra, Ea, Sa), (Rb, Eb, Sb) = F[a], F[b]
            assert np.allclose(Ra['gt_x'], Rb['gt_x'])
            pr = P.paired(Rb, Ea, Eb, nboot=400, bins=BINS)
            print(f'\n== {sc}: {a} (A) vs {b} (B)')
            for k, v in pr.items():
                v['rel'] = v['d'] / v['B']['rms']
                print(f'  {k:>11s} n {v["A"]["n"]:5d}  RMS/CRLB A {v["A"]["rms"]:.3f} B {v["B"]["rms"]:.3f}  d {v["d"]:+.3f} [{v["ci"][0]:+.3f}, {v["ci"][1]:+.3f}]'
                      f' ({100 * v["rel"]:+.1f} % [{100 * v["ci"][0] / v["B"]["rms"]:+.1f}, {100 * v["ci"][1] / v["B"]["rms"]:+.1f}])  zcal A {v["A"]["zcal"]:.2f} B {v["B"]["zcal"]:.2f}  nm A {v["A"]["nm"]:.2f} B {v["B"]["nm"]:.2f}')
            db = det(Rb, Sb); da = det(Ra, Sa, fp_ref=db['FP']); rf = da['at_fp']
            print(f'  detect @0.5: A recall {da["recall"]:.4f} FP {da["FP"]} | B recall {db["recall"]:.4f} FP {db["FP"]}'
                  + (f' | A at FP <= {db["FP"]}: recall {rf["recall"]:.4f} (thr {rf["thr"]:.3f}, FP {rf["FP"]})' if rf else ' | A: no threshold reaches B FP'))
            res[f'{sc}|{a}-{b}'] = dict(paired=pr, detA=da, detB=db)
    if len(sys.argv) > 5: json.dump(res, open(sys.argv[5], 'w'), indent=1, default=float)
