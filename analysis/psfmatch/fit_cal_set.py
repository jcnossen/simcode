"""PSF-matched retrain (2026-11-11): sim-sigma solve calibration of a front-end on the tuning seed 202 of ONE PSF's averaged-PSF
gate scenes (= analysis/psf3d_nets/fit_cal_new.py, which pooled the 3 per-axis PSFs; same plib.fit_cal procedure, prior_off from iso
of the same PSF). The _real variant multiplies the sigma factor by sig_ph_real_mult of an existing _real calib (copied, not re-measured).
usage: python fit_cal_set.py <front> <fe tag, e.g. m1809a> <fit scenes, comma> <base calib> <out sim calib> <out real calib> <real-mult src>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode import config

OUT = os.path.join(config.root('out'), 'psfmatch')

if __name__ == '__main__':
    front, tag, scen, base, out_sim, out_real, src_real = sys.argv[1:8]
    d = f'{OUT}/fe_{tag}'
    B = json.load(open(config.resolve(base)))
    Rs = [P.load(sc, front, seeds=(202,), d=d) for sc in scen.split(',')]
    iso = P.load('iso', front, seeds=(202,), d=d)
    cal = P.fit_cal(Rs, B, iso=iso)
    cal['fit_scenes'] = [f'{tag}:{sc}' for sc in scen.split(',')]
    cal['note'] = (f'2026-11-11 PSF-matched retrain: sim-sigma calibration of front {front} (analysis/psfmatch/fit_cal_set.py = plib.fit_cal), '
                   f'tuning seed 202 of the averaged-PSF gate scenes {cal["fit_scenes"]}, prior_off from iso ({tag}); base {base}')
    os.makedirs(os.path.dirname(os.path.abspath(out_sim)), exist_ok=True)
    json.dump(cal, open(out_sim, 'w'), indent=1)
    S = json.load(open(config.resolve(src_real)))
    cr = dict(cal); M = S['sig_ph_real_mult']; cr['sig_ph_real_mult'] = M; cr['sig_ph_sim'] = cal['sig_ph']
    mult = np.interp(np.log(cal['sig_ph']['ph']), np.log(M['ph']), M['mult'])
    cr['sig_ph'] = dict(ph=cal['sig_ph']['ph'], f=[float(a * b) for a, b in zip(cal['sig_ph']['f'], mult)])
    cr['note'] = cal['note'] + f' | _real: sig_ph_real_mult copied from {src_real} (not re-measured for the new networks)'
    json.dump(cr, open(out_real, 'w'), indent=1)
    print('wrote', out_sim, out_real)
