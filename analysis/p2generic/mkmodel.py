"""Write a one-model json for simcode.run --model from single network / calibration files (simcode.run has no single-file
options since 2026-11-14). Names are resolved as simcode.config.resolve does (models directory, models/calib, models/insitu)
and stored as absolute paths (cal_sim / cal_sim_pass2: a model has one solve calibration per pass since 2026-11-17). sig_u '' =
no selection-aware sigma rescale.
usage: python analysis/nanoruler/mkmodel.py <out.json> <name> <detector> <emitter> <cal> <detector_pass2> <emitter_pass2> <cal_pass2> [sig_u p1,p2]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from simcode import config

out, name, det, emitter, cal, det2, emitter2, cal2 = sys.argv[1:9]
sig_u = sys.argv[9] if len(sys.argv) > 9 else ''
r = lambda p: os.path.abspath(config.resolve(p))
for f in (det, emitter, cal, det2, emitter2, cal2):
    if not os.path.exists(r(f)): raise SystemExit(f'mkmodel: {f} not found')
M = dict(detector=r(det), emitter=r(emitter), cal_sim=r(cal), detector_pass2=r(det2), emitter_pass2=r(emitter2),
         cal_sim_pass2=r(cal2), sig_u=','.join(r(s) if s else '' for s in sig_u.split(',')) if sig_u else '')
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump({name: M, '_doc': 'written by analysis/nanoruler/mkmodel.py from single files'}, open(out, 'w'), indent=1)
print(out)
