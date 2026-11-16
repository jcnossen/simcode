"""PSF-matched retrain (2026-11-11): sim-sigma solve calibration of a front-end on the tuning seed 202 of ONE PSF's averaged-PSF
gate scenes (as the earlier fit of the mixture set, which pooled the 3 per-axis PSFs; same plib.fit_cal procedure, prior_off from iso
of the same PSF; until 2026-11-16 analysis/psfmatch/fit_cal_set.py). The _real variant multiplies the sigma factor by sig_ph_real_mult of an existing _real calib (copied, not re-measured).
usage: python -m simcode.calibration.fit_cal_set <front> <dump directory, e.g. out/psfmatch/fe_m1809a> <fit scenes, comma> <base calib> <out sim calib> <out real calib> <real-mult src>
The label in fit_scenes / note is the dump directory's basename without a leading 'fe_' (fe_m1809a -> m1809a)."""
import json, os, sys
import numpy as np
from simcode.calibration import gate_stats as P
from simcode import config

if __name__ == '__main__':
    front, d, scen, base, out_sim, out_real, src_real = sys.argv[1:8]
    label = os.path.basename(os.path.normpath(d))
    if label.startswith('fe_'): label = label[3:]
    B = json.load(open(config.resolve(base)))
    Rs = [P.load(sc, front, seeds=(202,), d=d) for sc in scen.split(',')]
    iso = P.load('iso', front, seeds=(202,), d=d)
    cal = P.fit_cal(Rs, B, iso=iso)
    cal['fit_scenes'] = [f'{label}:{sc}' for sc in scen.split(',')]
    cal['note'] = (f'2026-11-11 PSF-matched retrain: sim-sigma calibration of front {front} (simcode.calibration.fit_cal_set = gate_stats.fit_cal), '
                   f'tuning seed 202 of the averaged-PSF gate scenes {cal["fit_scenes"]}, prior_off from iso ({label}); base {base}')
    os.makedirs(os.path.dirname(os.path.abspath(out_sim)), exist_ok=True)
    json.dump(cal, open(out_sim, 'w'), indent=1)
    S = json.load(open(config.resolve(src_real)))
    cr = dict(cal); M = S['sig_ph_real_mult']; cr['sig_ph_real_mult'] = M; cr['sig_ph_sim'] = cal['sig_ph']
    mult = np.interp(np.log(cal['sig_ph']['ph']), np.log(M['ph']), M['mult'])
    cr['sig_ph'] = dict(ph=cal['sig_ph']['ph'], f=[float(a * b) for a, b in zip(cal['sig_ph']['f'], mult)])
    cr['note'] = cal['note'] + f' | _real: sig_ph_real_mult copied from {src_real} (not re-measured for the new networks)'
    json.dump(cr, open(out_real, 'w'), indent=1)
    print('wrote', out_sim, out_real)
