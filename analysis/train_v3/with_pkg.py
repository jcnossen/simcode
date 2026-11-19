"""run a script with the frozen SIMCODE package of this retrain: imports simcode from PYTHONPATH (the snapshot src_<hash>) before
the script runs, so a script that puts its own repository first on sys.path (analysis/papersim/common.py, decode_paper.py)
still uses the snapshot's simcode (the PSF at the calibration's fit sampling). usage: python with_pkg.py <script.py> [args]"""
import sys, runpy, os
import simcode, simcode.psf, simcode.config, simcode.simulate.simulator   # noqa: F401  (the snapshot's modules, cached)
print('with_pkg: simcode from', os.path.dirname(simcode.__file__), flush=True)
script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))     # as `python <script>` does
runpy.run_path(script, run_name='__main__')
