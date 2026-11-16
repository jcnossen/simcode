"""Moved to the package on 2026-11-16: simcode/calibration/gate_stats.py (the automatic pipeline's training stage uses it).
`import plib` (with analysis/calib on sys.path) gives that module itself, so module attributes set through either name
(e.g. plib.PX) are shared."""
import sys
from simcode.calibration import gate_stats as _m

sys.modules[__name__] = _m
