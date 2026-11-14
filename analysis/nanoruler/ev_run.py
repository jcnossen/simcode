"""nr80 evaluation: python -m simcode.run on a movie of the nanoruler setup. simcode.run itself takes the PSF, camera and
pixel size from --calib, but three constants of the other setup are hard-coded in the package; they are overridden here IN
MEMORY (no package file is edited), then simcode.run.main runs with the given arguments:
  1. simcode/pipeline/pattern_est.estimate_k: pitch search annulus pitch_px = (1.6, 2.8) px. The nanoruler pitch is 3.38 px,
     outside it -> env EV_PITCH (default '2.9,3.9').
  2. simcode/strict.py CROP = (5, 5, 110) (default output region of the selection and of the sigma-rescale tiles) -> set to
     the --crop of this run (also the bound default arguments of tile_of / tile_rank / in_crop; as simcode/auto/runwrap.py).
     simcode.run passes its --crop to the table stage anyway.
  3. --crop default '5,5,110' (120 px sensor crop of the other setup) -> must be passed explicitly (x0,y0,size; the loaded
     region adds a 5 px margin, clipped to the sensor): '5,5,246' for the 256 x 256 simulated movies.
The model: --model out/nanoruler/models/netset_nr80.json (the nr80 model file, one entry, absolute paths); --cal_kind sim
for the sim-sigma solve calibrations (cal_sim / p2_cal_sim) that the nanoruler runs use.
usage: python -u analysis/nanoruler/ev_run.py --movie ... --calib ... --out ... --crop 5,5,246 --model <model json> --cal_kind sim ..."""
import os, sys
sys.path.insert(0, '/home/jc/simcode2')
from simcode.pipeline import pattern_est as PE
from simcode import strict as SP
from simcode import run as RUN

argv = sys.argv[1:]
assert '--crop' in argv, 'pass --crop x0,y0,size explicitly (the default 5,5,110 is the other setup)'
crop = tuple(float(v) for v in argv[argv.index('--crop') + 1].split(','))
pit = tuple(float(v) for v in os.environ.get('EV_PITCH', '2.9,3.9').split(','))
d = list(PE.estimate_k.__defaults__); assert d[0] == (1.6, 2.8), 'estimate_k signature changed'
d[0] = pit; PE.estimate_k.__defaults__ = tuple(d)
assert SP.tile_of.__defaults__[0] == SP.CROP and SP.tile_rank.__defaults__[-1] == SP.CROP and SP.in_crop.__defaults__[-1] == SP.CROP, \
    'strict.py signature changed'
SP.CROP = crop
SP.tile_of.__defaults__ = (crop,) + SP.tile_of.__defaults__[1:]
SP.tile_rank.__defaults__ = SP.tile_rank.__defaults__[:-1] + (crop,)
SP.in_crop.__defaults__ = SP.in_crop.__defaults__[:-1] + (crop,)
print(f'ev_run: pitch search {pit} px, strict crop {crop}', flush=True)
# (parent, 2026-11-13) env EV_DET_BATCH: detector windows per batch (run_uncalibrated --det_batch, default 64, is not exposed by
# simcode.run). A 400 x 400 px field needs > 12 GB at 64; the argparse default is overridden in memory.
if os.environ.get('EV_DET_BATCH'):
    import argparse as _ap
    _orig = _ap.ArgumentParser.add_argument
    def _add(self, *a, **k):
        if a and a[0] == '--det_batch': k['default'] = int(os.environ['EV_DET_BATCH'])
        return _orig(self, *a, **k)
    _ap.ArgumentParser.add_argument = _add
    print('ev_run: det_batch', os.environ['EV_DET_BATCH'], flush=True)
RUN.main(argv)
