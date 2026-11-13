"""nr80 evaluation: python -m simcode2.run on a movie of the nanoruler setup. simcode2.run itself takes the PSF, camera and
pixel size from --calib, but three constants of the other setup are hard-coded in the package; they are overridden here IN
MEMORY (no package file is edited), then simcode2.run.main runs with the given arguments:
  1. simcode2/pipeline/pattern_est.estimate_k: pitch search annulus pitch_px = (1.6, 2.8) px. The nanoruler pitch is 3.38 px,
     outside it -> env EV_PITCH (default '2.9,3.9').
  2. simcode2/strict.py CROP = (5, 5, 110) (strict-product tiles, strict_u, coverage bins, density rule): rows outside
     5-115 px get strict_u = nan and can never be strict -> set to the --crop of this run (also the bound default
     arguments of tile_of / bin20).
  3. --crop default '5,5,110' (120 px sensor crop of the other setup) -> must be passed explicitly (x0,y0,size; the loaded
     region adds a 5 px margin, clipped to the sensor): '5,5,246' for the 256 x 256 simulated movies.
Not changed: the density -> q anchors of the strict product (strict.Q_DENSITY, fitted on the other setup's microtubule
movies; the sparse nanoruler rows fall below the lowest anchor, so q is the clamp value 0.275 pass 1 / 0.25 pass 2).
usage: python -u analysis/nanoruler/ev_run.py --movie ... --calib ... --out ... --crop 5,5,246 --pass2 --autoc 0 ..."""
import os, sys
sys.path.insert(0, '/home/jc/simcode2')
from simcode2.pipeline import pattern_est as PE
from simcode2 import strict as SP
from simcode2 import run as RUN

argv = sys.argv[1:]
assert '--crop' in argv, 'pass --crop x0,y0,size explicitly (the default 5,5,110 is the other setup)'
crop = tuple(float(v) for v in argv[argv.index('--crop') + 1].split(','))
pit = tuple(float(v) for v in os.environ.get('EV_PITCH', '2.9,3.9').split(','))
d = list(PE.estimate_k.__defaults__); assert d[0] == (1.6, 2.8), 'estimate_k signature changed'
d[0] = pit; PE.estimate_k.__defaults__ = tuple(d)
assert SP.tile_of.__defaults__[0] == SP.CROP and SP.bin20.__defaults__[-1] == SP.CROP, 'strict.py signature changed'
SP.CROP = crop
SP.tile_of.__defaults__ = (crop,) + SP.tile_of.__defaults__[1:]
SP.bin20.__defaults__ = SP.bin20.__defaults__[:-1] + (crop,)
print(f'ev_run: pitch search {pit} px, strict crop {crop}', flush=True)
# (parent, 2026-11-13) env EV_DET_BATCH: detector windows per batch (run_uncalibrated --det_batch, default 64, is not exposed by
# simcode2.run). A 400 x 400 px field needs > 12 GB at 64; the argparse default is overridden in memory.
if os.environ.get('EV_DET_BATCH'):
    import argparse as _ap
    _orig = _ap.ArgumentParser.add_argument
    def _add(self, *a, **k):
        if a and a[0] == '--det_batch': k['default'] = int(os.environ['EV_DET_BATCH'])
        return _orig(self, *a, **k)
    _ap.ArgumentParser.add_argument = _add
    print('ev_run: det_batch', os.environ['EV_DET_BATCH'], flush=True)
RUN.main(argv)
