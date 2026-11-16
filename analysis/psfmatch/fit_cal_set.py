"""Moved to the package on 2026-11-16: python -m simcode.calibration.fit_cal_set (same arguments). This script runs it."""
import runpy

if __name__ == '__main__':
    runpy.run_module('simcode.calibration.fit_cal_set', run_name='__main__', alter_sys=True)
