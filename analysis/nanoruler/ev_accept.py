"""nr80 evaluation: acceptance table on the nr80 gate scenes (out/nanoruler/eval/fe_nr80), the procedure of
analysis/psf3d_nets/accept.py as used by analysis/psfmatch/accept_g13.py. Fronts, each solved with its own sim calibration
fitted on seed 202 of THESE scenes:
  s1 / s2 = nr80 pass 1 / 2,  c1 / c2 = current production nets,  r1 / r2 = the m1809 matched set.
Comparisons (A vs B): s1-c1, s1-r1, s2-c2, s2-r2, s2-s1 and (available before the new nets exist) r1-c1, r2-c2, c2-c1.
Differences from accept_g13.py: (1) plib.PX = 65 nm (the 'nm' columns and the 'resolvable' filter CRLB < 25 nm are in real nm;
with the hard-coded 108.33 the filter would be CRLB < 15 nm); (2) extra photon band 800-2500 (the nanoruler emitters, ~1400
photons per window); bands are GT photons per 6-frame window.
usage: python ev_accept.py  -> out/nanoruler/eval/accept_nr80.txt / .json"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'psf3d_nets')); sys.path.insert(0, os.path.join(HERE, '..', 'calib'))
import accept as A
import plib as P
from simcode.util.proctitle import set_title
set_title('nr-eval')
EV = '/home/jc/simcode2/out/nanoruler/eval'; MC = '/home/jc/simcode2/out/nanoruler/models/calib'
P.PX = 65.0
if (800, 2500) not in P.PHB: P.PHB.insert(6, (800, 2500))       # in place: plib.paired's default argument is this list
A.OUT = EV
A.CAL = dict(s1=f'{MC}/calib_sc_hd_nr80.json', s2=f'{MC}/calib_p2L_nr80.json',
             c1=f'{EV}/calibs/calib_c1_nr80.json', c2=f'{EV}/calibs/calib_c2_nr80.json',
             r1=f'{EV}/calibs/calib_r1_nr80.json', r2=f'{EV}/calibs/calib_r2_nr80.json')
A.CAL = {k: v for k, v in A.CAL.items() if os.path.exists(v)}
A.COMP = [('s1', 'c1'), ('s1', 'r1'), ('s2', 'c2'), ('s2', 'r2'), ('s2', 's1'), ('r1', 'c1'), ('r2', 'c2'), ('c2', 'c1')]
sys.argv = ['accept', 'nr80', os.environ.get('EV_SCENES', 'iso,pair1,pair2,rs05qm')]
A.main()
