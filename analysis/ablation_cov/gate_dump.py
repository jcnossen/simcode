"""Covariance ablation: extra gate-scene dumps of the m1809 set on the averaged 1809/2 PSF (= the recipe of out/psfmatch/fe_m1809a,
verified bit-identical on rs05qm seed 301: simcode2.p2.dump with SIMCODE2_GATE_CALIB=insitu/calib_1809_2_psf3d_avg.json, fronts
s1 = det_m1809 + hd_m1809, s2 = p2det_m1809 + p2headL_m1809), optionally with the gate's mismatch overrides (simcode2.hd.gate POVER /
SOVER: RealSim preset values / scene() flags of the *qm scenes) for the excess-intensity-variance control and recovery tests.
usage: CUDA_VISIBLE_DEVICES=2 python gate_dump.py <scene> <seeds> <out dir> <tag> [pover k=v,..|-] [sover k=v,..|-] [fronts s1,s2] [ntarget]"""
import os, sys
os.environ['SIMCODE2_GATE_CALIB'] = 'insitu/calib_1809_2_psf3d_avg.json'
os.environ.setdefault('SIMCODE2_GATE_DIR', 'out/psfmatch/gate_m1809a')
from simcode2.util import proctitle; proctitle.set_title('covabl')
sc, seeds, out, tag = sys.argv[1:5]
pover = sys.argv[5] if len(sys.argv) > 5 else '-'; sover = sys.argv[6] if len(sys.argv) > 6 else '-'
fr = sys.argv[7] if len(sys.argv) > 7 else 's1,s2'; nt = sys.argv[8] if len(sys.argv) > 8 else '3500'
from simcode2.hd import gate as GA
for kv in filter(None, pover.strip('-').split(',')):
    k, v = kv.split('='); GA.POVER[k] = float(v)
for kv in filter(None, sover.strip('-').split(',')):
    k, v = kv.split('='); GA.SOVER[k] = v if k == 'bgmode' else bool(int(v))
print('overrides', GA.POVER, GA.SOVER, flush=True)
F = dict(s1='s1=det_m1809.pt:hd_m1809.pt', s2='s2=p2det_m1809.pt:p2headL_m1809.pt')
sys.argv = ['dump', '--scen', sc, '--seeds', seeds, '--fronts', ','.join(F[f] for f in fr.split(',')), '--ntarget', nt, '--out', out, '--tag', tag]
from simcode2.p2 import dump
dump.main()
