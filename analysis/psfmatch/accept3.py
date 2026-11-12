"""PSF-matched retrain (2026-11-11): head-to-head on one PSF's averaged-PSF gate scenes, matched (s) vs mixture (x) vs current (c),
pass 1 (s1/x1/c1) and pass 2 (s2/x2/c2), each front with its production sim calibration; x1r / x2r = the mixture with a calibration
refit on the same seed-202 scenes (separates network from calibration). Metrics and procedure = analysis/psf3d_nets/accept.py
(paired delta RMS/CRLB with FOV-bootstrap 95 % CI per photon bin, zcal, gross c5; recall of A at B's FP count).
usage: python accept3.py <tag m1809a|m25a> <set m1809|m25> <scenes> -> out/psfmatch/accept_<tag>.txt / .json"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'psf3d_nets'))
import accept as A
from simcode2 import config

if __name__ == '__main__':
    tag, s, scen = sys.argv[1:4]
    A.OUT = os.path.join(config.root('out'), 'psfmatch')
    A.CAL = dict(c1='calib/calib_sc_hd_sc5nbb15_pm_bright.json', c2='calib/calib_p2L.json',
                 x1='calib/calib_sc_hd_psf3dmix.json', x2='calib/calib_p2L_psf3dmix.json',
                 s1=f'calib/calib_sc_hd_{s}.json', s2=f'calib/calib_p2L_{s}.json',
                 x1r=f'{A.OUT}/calibs/calib_x1r_{tag}.json', x2r=f'{A.OUT}/calibs/calib_x2r_{tag}.json')
    A.CAL = {k: v for k, v in A.CAL.items() if os.path.exists(config.resolve(v))}
    # x1r / x2r reuse the x1 / x2 dumps (only the calibration differs): symlink <sc>__s<seed>_x1r.npz -> ..._x1.npz
    d = f'{A.OUT}/fe_{tag}'
    for fr in ('x1r', 'x2r'):
        if fr not in A.CAL: continue
        for fn in os.listdir(d):
            if fn.endswith(f'_{fr[:2]}.npz'):
                ln = os.path.join(d, fn[:-len(f'_{fr[:2]}.npz')] + f'_{fr}.npz')
                if not os.path.lexists(ln): os.symlink(fn, ln)
    A.COMP = [('s1', 'x1'), ('s1', 'c1'), ('x1', 'c1'), ('s1', 'x1r'),
              ('s2', 'x2'), ('s2', 'c2'), ('x2', 'c2'), ('s2', 'x2r')]
    sys.argv = ['accept', tag, scen]
    A.main()
