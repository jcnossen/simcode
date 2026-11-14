"""nr80 evaluation: sim-sigma solve calibration of a front-end on the tuning seed 202 of the nr80 gate scenes.
= analysis/psfmatch/fit_cal_set.py (same plib.fit_cal procedure, prior_off from iso, _real variant = sigma factor x the
sig_ph_real_mult of an existing _real calib, copied, not re-measured) with ONE difference: the dump directory is an explicit
argument (fit_cal_set.py hard-wires <out root>/psfmatch/fe_<tag>). The printed prior_off is in nm of 65 nm pixels.
usage: python ev_fitcal.py <front> <fe dir> <fit scenes, comma> <base calib> <out sim calib> <out real calib> <real-mult src>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode import config
from simcode.util.proctitle import set_title

if __name__ == '__main__':
    set_title('nr-eval'); P.PX = 65.0
    front, d, scen, base, out_sim, out_real, src_real = sys.argv[1:8]
    B = json.load(open(config.resolve(base)))
    for sc in scen.split(',') + ['iso']:
        assert os.path.exists(f'{d}/{sc}__s202_{front}.npz'), f'missing dump {d}/{sc}__s202_{front}.npz'
    Rs = [P.load(sc, front, seeds=(202,), d=d) for sc in scen.split(',')]
    iso = P.load('iso', front, seeds=(202,), d=d)
    cal = P.fit_cal(Rs, B, iso=iso)
    assert len(cal['sig_ph']['ph']) >= 3 and np.isfinite(cal['sig_ph']['f']).all() and np.isfinite(cal['prior_off']).all(), 'calibration fit failed'
    cal['fit_scenes'] = [f'{d}:{sc}' for sc in scen.split(',')]
    cal['note'] = (f'nr80 evaluation (analysis/nanoruler/ev_fitcal.py = plib.fit_cal): sim-sigma calibration of front {front}, tuning seed 202 of '
                   f'the nr80 gate scenes {cal["fit_scenes"]} (65 nm px, pitch 3.38 px), prior_off from iso; base {base}')
    os.makedirs(os.path.dirname(os.path.abspath(out_sim)), exist_ok=True)
    S = json.load(open(config.resolve(src_real)))
    cr = dict(cal); M = S['sig_ph_real_mult']; cr['sig_ph_real_mult'] = M; cr['sig_ph_sim'] = cal['sig_ph']
    mult = np.interp(np.log(cal['sig_ph']['ph']), np.log(M['ph']), M['mult'])
    cr['sig_ph'] = dict(ph=cal['sig_ph']['ph'], f=[float(a * b) for a, b in zip(cal['sig_ph']['f'], mult)])
    cr['note'] = cal['note'] + f' | _real: sig_ph_real_mult copied from {src_real} (measured on the other setup, not re-measured)'
    for fn, c in ((out_real, cr), (out_sim, cal)):          # the sim calib (the chain's skip marker) is written last, both atomically
        json.dump(c, open(fn + '.tmp', 'w'), indent=1); os.replace(fn + '.tmp', fn)
    print('wrote', out_sim, out_real)
