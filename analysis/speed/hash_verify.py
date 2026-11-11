"""Spatial hash (simcode2/util/spatial_hash) vs scipy cKDTree on the full pass-1 rows of a run dir (read-only):
exactness per query type (pattern_est.select nn / w-6 / w+6 links, fit_depth pairs, autoc iso counts), timings (serial
cKDTree, parallel cKDTree, hash CPU, hash GPU), and the whole pattern estimation (kdtree serial vs hash) with max diffs.
usage: python analysis/speed/hash_verify.py <run dir> <movie> <tag> [--sc_cal calib.json] [--skip_cpu_hash 0|1]"""
import sys, os, glob, json, time
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, gpu_state, dump
from simcode2 import config
from simcode2.util import spatial_hash as SH
from simcode2.pipeline import pattern_est as PE
from simcode2.pipeline.common import Movie
A = sys.argv[1:]
sc_cal = 'calib/calib_sc_hd_sc5nbb15_pm_bright_real.json'
if '--sc_cal' in A: i = A.index('--sc_cal'); sc_cal = A[i + 1]; del A[i:i + 2]
skip_cpu = False
if '--skip_cpu_hash' in A: i = A.index('--skip_cpu_hash'); skip_cpu = bool(int(A[i + 1])); del A[i:i + 2]
d, mv, tag = A[:3]
REP = dict(run_dir=d, gpu=torch.cuda.get_device_name(), gpu_state_start=gpu_state(), t={}, exact={})
out_f = f'{OUT}/hash_verify_{tag}.json'
fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4])); chunks = [int(f.split('_')[-1][:-4]) for f in fs]
t0 = time.time()
cols = ('wstart', 'x', 'y', 'sx', 'sy', 'I', 'phot', 'tc', 'x_det', 'y_det')
Ls = [np.load(f) for f in fs]
keep = np.concatenate([(L['wstart'] >= int(L['own'][0]) - 3) & ((L['wstart'] < int(L['own'][1]) - 3) | (f0 == chunks[-1])) for L, f0 in zip(Ls, chunks)])
R = {c: np.concatenate([L[c] for L in Ls])[keep] for c in cols}; del Ls
po = json.load(open(config.resolve(sc_cal))).get('prior_off', [0, 0])
R['x'] = R['x'] - po[0]; R['y'] = R['y'] - po[1]
R['out'] = (R['tc'] >= 1.0) & (R['tc'] < 4.0) & (R['x'] >= 5) & (R['x'] < 115) & (R['y'] >= 5) & (R['y'] < 115)
n = len(R['x']); REP['n_rows'] = n; print(f'{n} rows, load {time.time() - t0:.0f}s', flush=True)
ws = R['wstart'].astype(np.int64); x, y = R['x'].astype(np.float64), R['y'].astype(np.float64)
io = np.nonzero(R['out'])[0]; REP['n_queries'] = int(len(io))
def tm(key, f):
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time(); r = f(); REP['t'][key] = round(time.time() - t, 2); print(key, REP['t'][key], 's', flush=True); dump(REP, out_f); return r
# ---- cKDTree reference (serial = original code; parallel only timed)
P_all = np.c_[x, y, ws * 1e4]
def kd(workers):
    tr = cKDTree(P_all)
    dd, _ = tr.query(P_all[io], k=2, workers=workers)
    q = lambda dt: tr.query(np.c_[x[io], y[io], (ws[io] + dt) * 1e4], distance_upper_bound=1.0, workers=workers)
    return dd[:, 1], q(-6), q(6)
KS = tm('select_kdtree_serial', lambda: kd(1))
tm('select_kdtree_parallel', lambda: kd(-1))
def hs(dev):
    st = {}
    nn, _ = SH.nearest(x, y, ws, x[io], y[io], ws[io], PE.NN_CAP, qself=io, device=dev, stats=st)
    m = SH.nearest(x, y, ws, x[io], y[io], ws[io] - 6, 1.0, device=dev, stats=st)
    p = SH.nearest(x, y, ws, x[io], y[io], ws[io] + 6, 1.0, device=dev, stats=st)
    return nn, m, p, st
HG = tm('select_hash_gpu', lambda: hs('cuda')); REP['select_hash_gpu_peak_gb'] = round(HG[3].get('peak_gb', 0), 2); REP['select_ties_hash'] = HG[3]['ties']
if not skip_cpu: HC = tm('select_hash_cpu', lambda: hs('cpu'))
def cmp_near(dk, jk, dh, jh, qx, qy, px, py, name):
    """exact comparison; index mismatches are counted as ties iff the two chosen points are equidistant (same d^2)"""
    e = dict(dist_equal=bool(np.array_equal(dk, dh)), found_equal=bool(np.array_equal(np.isfinite(dk), np.isfinite(dh))),
             max_abs_ddist=float(np.max(np.abs(np.where(np.isfinite(dk) & np.isfinite(dh), dk - dh, 0)))) if len(dk) else 0.0)
    f = np.isfinite(dk) & np.isfinite(dh); mis = np.nonzero(f & (jk != jh))[0]
    d2k = (qx[mis] - px[jk[mis]]) ** 2 + (qy[mis] - py[jk[mis]]) ** 2; d2h = (qx[mis] - px[jh[mis]]) ** 2 + (qy[mis] - py[jh[mis]]) ** 2
    e.update(n=int(len(dk)), n_found=int(f.sum()), index_mismatch=int(len(mis)), mismatch_equidistant=int((d2k == d2h).sum()),
             mismatch_same_position=int(((px[jk[mis]] == px[jh[mis]]) & (py[jk[mis]] == py[jh[mis]])).sum()))
    REP['exact'][name] = e; print(name, e, flush=True); return e
nn_k, (dmk, jmk), (dpk, jpk) = KS
nnk_cap = np.where(nn_k < PE.NN_CAP, nn_k, np.inf)
for lab, H in (('gpu', HG),) + ((('cpu', HC),) if not skip_cpu else ()):
    nn_h, (dmh, jmh), (dph, jph), _ = H
    REP['exact'][f'nn_{lab}'] = dict(capped_equal=bool(np.array_equal(nnk_cap, nn_h)),
                                    thresholds_equal={str(t): bool(np.array_equal(nn_k >= t, nn_h >= t)) for t in (2.0, 3.0, 4.0, 5.0)},
                                    n_below_cap=int(np.isfinite(nn_h).sum()))
    print(f'nn_{lab}', REP['exact'][f'nn_{lab}'], flush=True)
    cmp_near(dmk, jmk, dmh, jmh, x[io], y[io], x, y, f'link_wm6_{lab}')
    cmp_near(dpk, jpk, dph, jph, x[io], y[io], x, y, f'link_wp6_{lab}')
# ---- fit_depth pair query on the clean set (from the reference select)
PE.NB_BACKEND = 'kdtree'; D = PE.select(R); idx = np.nonzero(D['clean'])[0]; REP['n_clean'] = int(len(idx))
dk, jk = tm('depth_kdtree_serial', lambda: PE.depth_pair_query(D, idx))
PE.NB_BACKEND = 'hash'; PE.NB_DEVICE = 'cuda'
dh, jh = tm('depth_hash_gpu', lambda: PE.depth_pair_query(D, idx))
xq, yq = D['x1'][idx], D['y1'][idx]
cmp_near(dk, jk, dh, jh, xq, yq, xq, yq, 'depth_pairs_gpu')
# ---- autoc iso counts (all rows of every chunk: key = wstart, detector points)
xd, yd = R['x_det'].astype(np.float64), R['y_det'].astype(np.float64)
ck = tm('iso_kdtree', lambda: cKDTree(np.c_[xd, yd, ws * 1e4]).query_ball_point(np.c_[xd, yd, ws * 1e4], 2.0, return_length=True, workers=-1))
ch = tm('iso_hash_gpu', lambda: SH.count(xd, yd, ws, xd, yd, ws, 2.0, device='cuda'))
REP['exact']['iso_count'] = dict(count_equal=bool(np.array_equal(ck, ch)), flag_equal=bool(np.array_equal(ck <= 1, ch <= 1)), n=int(len(ck)))
print('iso', REP['exact']['iso_count'], flush=True)
del KS, HG
# ---- whole pattern estimation: kdtree serial (original) vs hash gpu (vs hash cpu)
shape = Movie(mv).shape
res = {}
for lab, be, dev in [('kdtree_serial', 'kdtree', None), ('hash_gpu', 'hash', 'cuda')] + ([] if skip_cpu else [('hash_cpu', 'hash', 'cpu')]):
    PE.NB_BACKEND, PE.NB_DEVICE, PE.KD_WORKERS = be, dev, 1
    PE.NB_STATS.clear()
    res[lab] = tm(f'pattern_{lab}', lambda: PE.estimate_pattern(R, shape, verbose=lambda *a: None))
    REP[f'pattern_{lab}_ties'] = PE.NB_STATS.get('ties')
P0 = res['kdtree_serial'][0]
for lab in res:
    if lab == 'kdtree_serial': continue
    P1 = res[lab][0]
    REP['exact'][f'pattern_{lab}'] = {k: float((torch.as_tensor(P0[k]).double() - torch.as_tensor(P1[k]).double()).abs().max()) for k in ('k', 'phi', 'm', 'logrel', 'dphi')}
    REP['exact'][f'pattern_{lab}']['identical'] = all(torch.equal(torch.as_tensor(P0[k]), torch.as_tensor(P1[k])) for k in ('k', 'phi', 'm', 'logrel', 'dphi'))
    print(f'pattern {lab} max |diff|', REP['exact'][f'pattern_{lab}'], flush=True)
REP['gpu_state_end'] = gpu_state(); dump(REP, out_f); print('saved', out_f)
