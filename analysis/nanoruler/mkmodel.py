"""Write a one-model json for simcode.run --model from single network / calibration files (simcode.run has no single-file
options since 2026-11-14). Names are resolved as simcode.config.resolve does (models directory, models/calib, models/insitu)
and stored as absolute paths; the same calibration is used for both calibration kinds (cal_sim = cal_real, p2_cal_sim =
p2_cal_real), so the run's --cal_kind does not matter. sig_u '' = no selection-aware sigma rescale.
usage: python analysis/nanoruler/mkmodel.py <out.json> <name> <det> <head> <cal> <p2_det> <p2_head> <p2_cal> [sig_u p1,p2]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from simcode import config

out, name, det, head, cal, p2_det, p2_head, p2_cal = sys.argv[1:9]
sig_u = sys.argv[9] if len(sys.argv) > 9 else ''
r = lambda p: os.path.abspath(config.resolve(p))
for f in (det, head, cal, p2_det, p2_head, p2_cal):
    if not os.path.exists(r(f)): raise SystemExit(f'mkmodel: {f} not found')
M = dict(det=r(det), head=r(head), cal_sim=r(cal), cal_real=r(cal), p2_det=r(p2_det), p2_head=r(p2_head), p2_cal_sim=r(p2_cal),
         p2_cal_real=r(p2_cal), sig_u=','.join(r(s) if s else '' for s in sig_u.split(',')) if sig_u else '')
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump({name: M, '_doc': 'written by analysis/nanoruler/mkmodel.py from single files'}, open(out, 'w'), indent=1)
print(out)
