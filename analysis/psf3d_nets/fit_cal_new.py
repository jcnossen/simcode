"""psf3d retrain (2026-11-11): sim-sigma solve calibration of a retrained front-end (plib.fit_cal = the fit_cal.py procedure:
per-network-photon-bin kap_p / kap_I2 / sigma factor + prior_off from iso 700-2000 ph) on the TUNING seed 202 of the new-PSF gate
scenes, pooled over the 3 movie PSFs: rs05qm + rs05pm (1809/2 PSF), rs25qm (2.5 nM and 20 nM PSFs); prior_off from iso of all 3.
The real-data variant multiplies the sigma factor by the current calibration's sig_ph_real_mult (real M4 zcal x sim true / M4
ratio, interpolated in log photons; not re-measured for the new networks), as the shipped _real calibrations.
usage: python fit_cal_new.py <front> <base calib> <out sim calib> <out real calib> <real-mult source calib>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calib'))
import plib as P
from simcode2 import config

OUT = os.path.join(config.root('out'), 'psf3d_nets')
FIT = [('m1809', 'rs05qm'), ('m1809', 'rs05pm'), ('m25', 'rs25qm'), ('m20', 'rs25qm')]
ISO = ['m1809', 'm25', 'm20']
if os.environ.get('FIT_ONLY'):   # opt-in: restrict to one PSF tag (e.g. m1809, the per-axis / average ablation)
    FIT = [f for f in FIT if f[0] == os.environ['FIT_ONLY']]; ISO = [os.environ['FIT_ONLY']]


def cat(Rs):
    """pool dumps of different PSFs into one R (FOV ids made unique)"""
    off = 0; out = []
    for R in Rs:
        R = dict(R); R['gt_fov'] = R['gt_fov'] + off; R['sc_b'] = R['sc_b'] + off; off += int(R['n_fov']); out.append(R)
    keys = set.intersection(*[set(R) for R in out]) - {'n_fov', 'pfov'}
    M = {k: np.concatenate([R[k] for R in out]) for k in keys}; M['n_fov'] = off
    return M


if __name__ == '__main__':
    front, base, out_sim, out_real, src_real = sys.argv[1:6]
    B = json.load(open(config.resolve(base)))
    Rs = [P.load(sc, front, seeds=(202,), d=f'{OUT}/fe_{t}') for t, sc in FIT]
    iso = cat([P.load('iso', front, seeds=(202,), d=f'{OUT}/fe_{t}') for t in ISO])
    cal = P.fit_cal(Rs, B, iso=iso)
    cal['fit_scenes'] = [f'{t}:{sc}' for t, sc in FIT]
    cal['note'] = (f'2026-11-11 psf3d retrain: sim-sigma calibration of front {front} (analysis/psf3d_nets/fit_cal_new.py = plib.fit_cal), '
                   f'tuning seed 202 of the new-PSF gate scenes {cal["fit_scenes"]}, prior_off from iso (3 PSFs); base {base}')
    json.dump(cal, open(out_sim, 'w'), indent=1)
    S = json.load(open(config.resolve(src_real)))
    cr = dict(cal); M = S['sig_ph_real_mult']; cr['sig_ph_real_mult'] = M; cr['sig_ph_sim'] = cal['sig_ph']
    mult = np.interp(np.log(cal['sig_ph']['ph']), np.log(M['ph']), M['mult'])
    cr['sig_ph'] = dict(ph=cal['sig_ph']['ph'], f=[float(a * b) for a, b in zip(cal['sig_ph']['f'], mult)])
    cr['note'] = cal['note'] + f' | _real: sig_ph_real_mult copied from {src_real} (not re-measured for the new networks)'
    json.dump(cr, open(out_real, 'w'), indent=1)
    print('wrote', out_sim, out_real)
