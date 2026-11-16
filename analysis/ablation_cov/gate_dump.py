"""Covariance ablation: extra gate-scene dumps of the m1809 set on the averaged 1809/2 PSF (= the recipe of out/psfmatch/fe_m1809a,
verified bit-identical on rs05qm seed 301: simcode.processing.gate_dump --calib insitu/calib_1809_2_psf3d_avg.json, fronts
s1 = pass 1, s2 = pass 2 of archive_2d/m1809), optionally with the gate's mismatch overrides (simcode.benchmark.gate_scenes POVER /
SOVER: AcquisitionSim preset values / scene() flags of the *qm scenes) for the excess-intensity-variance control and recovery tests.
usage: CUDA_VISIBLE_DEVICES=2 python gate_dump.py <scene> <seeds> <out dir> <tag> [pover k=v,..|-] [sover k=v,..|-] [fronts s1,s2] [ntarget]
         --gate_dir <gate npz folder, e.g. ~/simcode2/out/psfmatch/gate_m1809a>
  (--gate_dir: simcode.processing.gate_dump --gate_dir, the gate npz the CRLB is copied from; the scene PSF is fixed to
   --calib insitu/calib_1809_2_psf3d_avg.json)"""
import argparse, sys
from simcode.util import proctitle; proctitle.set_title('covabl')
ap = argparse.ArgumentParser()
ap.add_argument('scene'); ap.add_argument('seeds'); ap.add_argument('out'); ap.add_argument('tag')
ap.add_argument('pover', nargs='?', default='-'); ap.add_argument('sover', nargs='?', default='-')
ap.add_argument('fronts', nargs='?', default='s1,s2'); ap.add_argument('ntarget', nargs='?', default='3500')
ap.add_argument('--gate_dir', required=True, help='gate npz folder (simcode.processing.gate_dump --gate_dir)')
A = ap.parse_args()
sc, seeds, out, tag = A.scene, A.seeds, A.out, A.tag
pover, sover, fr, nt = A.pover, A.sover, A.fronts, A.ntarget
from simcode.benchmark import gate_scenes as GA
for kv in filter(None, pover.strip('-').split(',')):
    k, v = kv.split('='); GA.POVER[k] = float(v)
for kv in filter(None, sover.strip('-').split(',')):
    k, v = kv.split('='); GA.SOVER[k] = v if k == 'bgmode' else bool(int(v))
print('overrides', GA.POVER, GA.SOVER, flush=True)
# the 2D m1809 of the ablation (archive_2d/m1809: byte-identical to the files it was run with)
F = dict(s1='s1=archive_2d/m1809/detector_pass1.pt:archive_2d/m1809/emitter_pass1.pt', s2='s2=archive_2d/m1809/detector_pass2.pt:archive_2d/m1809/emitter_pass2.pt')
sys.argv = ['dump', '--scen', sc, '--seeds', seeds, '--fronts', ','.join(F[f] for f in fr.split(',')), '--ntarget', nt, '--out', out, '--tag', tag,
            '--calib', 'insitu/calib_1809_2_psf3d_avg.json', '--gate_dir', A.gate_dir]
from simcode.processing import gate_dump
gate_dump.main()
