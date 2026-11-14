"""z3d code-path check of the pipeline with a 3D head: the deterministic sim chunk of tests/test_fast_paths.py, pass 1
reference vs fast (z, sz, bz columns), the pattern solve reference vs fast (z_hy0, sz_cal, cov_xz / cov_yz), and the v2 table
-> Dataset with pos [n, 3]. usage: python analysis/z3d/pipe_check.py <3D head .pt> [det .pt]"""
import json, os, sys, tempfile
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests'))
import test_fast_paths as TF
from simcode.pipeline.sc_front import Pass1SC, run_chunk_pass2
from simcode.pipeline.sc_front_fast import Pass1SCFast, run_chunk_pattern_solve_fast
from simcode import config

head = sys.argv[1]; det = sys.argv[2] if len(sys.argv) > 2 else TF.DET
mov, cal = TF.sim_movie()
A = Pass1SC(TF.CALIB, head=head, det=det); B = Pass1SCFast(TF.CALIB, head=head, det=det)
B.det = A.det; B.head = A.head; B.front.det = A.det; B.front.heads = {'h': (A.head, 'sc')}
assert A.head.z3d
Ra, ia = A.run_chunk(mov, cal, 0, mov.n); Rb, ib = B.run_chunk(mov, cal, 0, mov.n)
Ra, Rb = TF.canon(Ra), TF.canon(Rb)
for k in ('z', 'sz', 'bzx', 'bzy'):
    print(f'pass1 {k}: ref vs fast max |d| {np.abs(Ra[k] - Rb[k]).max():.2e}; median {np.median(Ra[k]):+.3f} range {np.percentile(Ra[k], 1):+.3f} .. {np.percentile(Ra[k], 99):+.3f}')
pat = torch.load(TF.PATT, weights_only=False); scal = json.load(open(TF.SCAL))
Oa = run_chunk_pass2(Ra, ia, pat, scal); Ob = run_chunk_pattern_solve_fast(Ra, ia, pat, scal)
for k in ('x_hy0', 'z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior'):
    d = np.abs(Oa[k] - Ob[k]); print(f'solve {k}: n {len(d)} ref vs fast median |d| {np.median(d):.1e} p99 {np.percentile(d, 99):.1e}; finite {np.isfinite(Oa[k]).mean():.3f}')
print('sd z prior (median px)', float(np.median(Ra['sz'])), 'sd z after solve (median px)', float(np.median(Oa['sz_cal'])))
# table -> Dataset
from simcode import strict as SP
from simcode.dataset import Dataset
d = tempfile.mkdtemp()
T = dict(frame=Oa['wstart'].astype(np.int64), x=Oa['x_hy0'].astype(np.float64), y=Oa['y_hy0'].astype(np.float64), sx=Oa['sx_cal'], sy=Oa['sy_cal'],
         photons=Oa['phot'], p=Oa['p'].astype(np.float32), z=Oa['z_hy0'].astype(np.float64), sz=Oa['sz_cal'], cov_xz=Oa['cov_xz'], cov_yz=Oa['cov_yz'])
np.savez(os.path.join(d, 'locs_p1.npz'), **T, px_nm=np.array(108.33))
ds = Dataset.from_run(d, pass_=1, info=False); print('Dataset', ds['pos'].shape, ds['crlb'].shape, ds['cov'].shape)
