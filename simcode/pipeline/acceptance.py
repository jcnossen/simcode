"""Stage e acceptance on the gate dumps (held-out seed 301; tuning seed 202 was used for the calibrations): per scene the
robust sigma calibration zcal (1.4826 MAD of error / calibrated sigma) of pass 1 (front s1) and pass 2 (s2), and the pass-2
over pass-1 RMS / CRLB ratio on the realistic scene, 200-2000 photons (analysis/calib/plib.py statistics).
usage: python accept.py <fe dir> <pass-1 sim calib> <pass-2 sim calib> <scenes comma> <out json>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'analysis', 'calib'))
import plib as P

fe, c1, c2, scen, outp = sys.argv[1:6]
cals = dict(s1=json.load(open(c1)), s2=json.load(open(c2)))
res = dict(zcal={}, rms={}, n={})
for sc in scen.split(','):
    for fr in ('s1', 's2'):
        R = P.load(sc, fr, seeds=(301,), d=fe)
        ti, oi = P.targets(R)
        sol = P.solve(R, oi, cals[fr])
        E = P.errs(R, ti, oi, sol)
        c = R['gt_sf_m'][ti]; ph = R['gt_phot'][ti]
        ok = np.isfinite(c).all(1) & np.isfinite(E['e']).all(1) & (ph >= 200) & (ph < 2000)
        st = P.stats(E['e'][ok], c[ok], E['s'][ok])
        res['zcal'][f'{sc}_{fr}'] = st['zcal']; res['rms'][f'{sc}_{fr}'] = st['rms']; res['n'][f'{sc}_{fr}'] = st['n']
        print(sc, fr, json.dumps(st), flush=True)
sc = scen.split(',')[-1]
res['p2_over_p1'] = res['rms'][f'{sc}_s2'] / res['rms'][f'{sc}_s1']
json.dump(res, open(outp, 'w'), indent=1)
print(json.dumps(res))
