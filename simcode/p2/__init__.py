"""
Pass 2 of SIMCODE v2 (2026-11-09): pattern-aware networks that take the illumination pattern estimated after pass 1 as input.

  illum.py   perturbed pattern estimates (training / robustness) and per-pixel illumination input maps
  data.py    realistic training batches (realsim, paper brightness in label units, c ~ 0.22 intensity noise, filaments, pairs)
  det2.py    DetP2: det_n1 (DetUp up=1) + illumination input channels (zero-initialised -> starts at det_n1); training
  head2.py   SCHeadP2: SCHead4 (sc5nb) + illumination patch channels (zero-initialised -> starts at sc5nb); training
  front.py   inference: pass-2 detector + head on a window with a given pattern; pipeline pass 2 (run_uncalibrated --frontend sc2p)
  dump.py    gate-scene dumps of pass-1 / pass-2 head outputs (offline solve / calibration / reports)
The final position is always the explicit intensity-domain pattern solve (simcode/hd/sc.sc_solve -> msolve) on the network
outputs; there is no pixel-level intensity fit or pixel-level position refinement anywhere.
"""
