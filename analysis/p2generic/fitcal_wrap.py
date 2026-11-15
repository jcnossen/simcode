"""Run a calibration-fit script (analysis/nanoruler/ev_fitcal.py or analysis/p2generic/fit_cal_gen.py) with simcode.solve.solve.solve_emitters
accepting a calibration without 'sig_ph' (factor 1), as before the 2026-11-14 options clean-up made the key mandatory: plib.fit_cal
solves once with the partial calibration (kap_bins, prior_off set; sig_ph not yet) to fit sig_ph itself. In memory only.
usage: python fitcal_wrap.py <script.py> <its arguments>"""
import runpy, sys
sys.path.insert(0, '/home/jc/simcode2')
import simcode.emitter.model as SC
from simcode.solve import solve as SV
_s0 = SV.solve_emitters
def _sc_solve(o, pt_win, b, cal, *a, **k):
    if 'sig_ph' not in cal: cal = dict(cal, sig_ph=dict(ph=[1.0, 1e6], f=[1.0, 1.0]))
    return _s0(o, pt_win, b, cal, *a, **k)
SV.solve_emitters = _sc_solve
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
