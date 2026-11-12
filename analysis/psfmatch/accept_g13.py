"""g13 set (2026-11-12): Gaussian-PSF gate scenes out/psfmatch/fe_g13a; s1/s2 = g13 pass 1 / 2, r1/r2 = the m1809 set (reference: what
the PSF mismatch costs), each with its own sim calibration fitted on seed 202 of THESE scenes (r*: analysis/psfmatch/fit_cal_set.py too).
Same procedure as analysis/psf3d_nets/accept.py. usage: python accept_g13.py -> out/psfmatch/accept_g13a.txt"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'psf3d_nets'))
import accept as A
from simcode2 import config
A.OUT = os.path.join(config.root('out'), 'psfmatch')
A.CAL = dict(s1='calib/calib_sc_hd_g13.json', s2='calib/calib_p2L_g13.json',
             r1=f'{A.OUT}/calibs/calib_r1_g13a.json', r2=f'{A.OUT}/calibs/calib_r2_g13a.json')
A.CAL = {k: v for k, v in A.CAL.items() if os.path.exists(config.resolve(v))}
A.COMP = [('s1', 'r1'), ('s2', 'r2')]
sys.argv = ['accept', 'g13a', 'iso,pair1,pair2,rs05qm,rs25qm']
A.main()
