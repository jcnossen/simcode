"""Run a simcode module (trainer, gate, dump, ...) with the RealSim presets patched for the nanoruler (original SIMFLUX) setup.

The trainers of the PSF-matched recipe take the PSF and the camera gain from --calib, but the pattern vectors, the phase-step
spacing and the read-noise map come from simcode.hd.realsim.PRESETS (DMD-SIMFLUX setup: pitch 2.07 px, Zyla maps). This
wrapper overrides those preset entries in memory (no package file is edited) from a json (env NR_PRESET, default
out/nanoruler/nr80_preset.json: {"common": {...}, "0.5nM": {...}, "2.5nM": {...}}), then runs the module as __main__:
    python analysis/nanoruler/nrwrap.py simcode.hd.sc4 train --calib ... (the module's own arguments)
Process title: env NR_TITLE (<= 20 characters with the 'simcode:' prefix)."""
import json, os, runpy, sys
sys.path.insert(0, '/home/jc/simcode2')
import simcode.hd.realsim as RS
from simcode.util.proctitle import set_title
NR = json.load(open(os.environ.get('NR_PRESET', '/home/jc/simcode2/out/nanoruler/nr80_preset.json')))
for name, p in RS.PRESETS.items():
    p.update({k: (tuple(v) if isinstance(v, list) and k != 'k' else v) for k, v in {**NR.get('common', {}), **NR.get(name, {})}.items()})
if os.environ.get('NR_TITLE'): set_title(os.environ['NR_TITLE'])
mod = sys.argv[1]; sys.argv = [mod] + sys.argv[2:]
if mod.endswith('.py'): runpy.run_path(mod, run_name='__main__')
else: runpy.run_module(mod, run_name='__main__', alter_sys=True)
