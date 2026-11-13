"""m20check (2026-11-12): simcode2.p2.dump under the process title 'simcode:m20check' (no other change; arguments passed through).
usage: SIMCODE2_GATE_CALIB=<calib> SIMCODE2_GATE_DIR=<gate dir> python analysis/psfmatch/m20check_dump.py --scen ... --fronts ... --out ..."""
import runpy
from simcode2.util import proctitle
proctitle.set_title('m20check')
runpy.run_module('simcode2.p2.dump', run_name='__main__')
