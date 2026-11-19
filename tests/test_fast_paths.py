"""
Regression test of the fast inference paths (the production default) against the reference code paths on a fixed,
deterministic sim chunk. Run:  CUDA_VISIBLE_DEVICES=<gpu> python tests/test_fast_paths.py   (or: pytest tests)
Needs a CUDA GPU (~1 min) and the models directory (simcode/config.py). Networks and calibrations: the default model
(m1809, simcode/models.json); PSF / camera: the in-situ calibration of the 1809/2 movie; pattern: the pattern estimate of
the 1809/2 production run with that set (tests/data/pattern_est_1809_2.pt). Checks:
  pass 1 (Pass1Front vs Pass1FrontFast; queued, 1- and 2-stream, partial head batches):
    fp32 (bf16 autocast and TF32 off in both): identical candidate list, |dx|, |dy| < 1e-4 px, |dI| < 1e-2 ph
    bf16 (as in production): identical candidate list; I / x differences <= 2 x the bf16 noise reference (the original
         run with a different window batch, which changes the cuDNN algorithms) + 1e-6
  pattern solve (solve.solve_emitters vs solve_fast.solve_emitters_fast): median |dr| < 1e-5 px (compiled LM step: < 5e-5 px), p99 < 2e-3 px (the original
    solve's own sensitivity to a 1-ulp input change is p99.9 ~6e-4, max ~0.09 px), kinetic-model choice >= 99.9 % equal
  pattern-aware pass 2 (Pass2Front vs Pass2FrontFast, fp32): identical rows, |d prior| < 1e-4 px, solve positions as above; the
    z from pass 1: identical links to the pass-1 rows (> 90 % of the selection linked), z differences as the solve positions
"""
import contextlib, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simcode import config
CALIB = config.model('insitu/calib_1809_2_psf3d.json')
EMITTER = config.model(config.EMITTER)
DETECTOR = config.model(config.DETECTOR)
EMITTER_PASS2 = config.model(config.EMITTER_PASS2)
DETECTOR_PASS2 = config.model(config.DETECTOR_PASS2)
CAL = config.model(config.CAL_SIM)
CAL_PASS2 = config.model(config.CAL_SIM_PASS2)
PATT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pattern_est_1809_2.pt')   # pattern estimate of the 1809/2 movie (m1809 set)
_C = {}
# bf16 tolerances (production precision): the bf16 head output depends on the batch composition through the cuDNN algorithm
# choice. On the real 0.5 nM chunk (frames 12000-18000) the original vs itself with another window batch differs by up to
# 2.5e-3 px / 1.45 ph (median 1.1e-4 px / 0.08 ph) and the fast path vs the original by 3.1e-3 px / 1.45 ph (same medians).
BF16_X, BF16_I, BF16_DET = 5e-3, 3.0, 1e-3


def sim_movie():
    """deterministic 2.5 nM-like 120 x 120 chunk (AcquisitionSim windows laid end to end, 240 frames) in ADU"""
    if 'mov' in _C: return _C['mov']
    from simcode.processing.common import load_calib
    from simcode.psf import ArrayMovie            # in-memory ADU stack with the reader interface
    from simcode.simulate.acquisition import AcquisitionSim
    cal = load_calib(CALIB)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1234)
        rs = AcquisitionSim('2.5nM', H=120, W=120, T=6, seed=1234, calib=CALIB)
        o = rs.scene(40)
    fr = o['frames'].reshape(-1, 120, 120).cpu().numpy()
    adu = fr / cal['gain'] + cal['offset'][:120, :120][None]
    _C['mov'] = (ArrayMovie(adu.astype(np.float32)), cal)
    return _C['mov']


@contextlib.contextmanager
def fp32_mode():
    """no bf16 autocast, no TF32 (for the strict fp32 comparison)"""
    ac = torch.autocast
    tf = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.autocast = lambda *a, **k: contextlib.nullcontext()
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.autocast = ac
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf


def _keys(R):
    return np.c_[R['wstart'], R['x_det'], R['y_det']]


def canon(R):
    """rows sorted by (window, x_det, y_det): a different detector batch size changes the conv algorithms -> rounding-level
    logit differences, which can swap the top-k order of two candidates of one window (the set is unchanged)"""
    o = np.lexsort((R['y_det'], R['x_det'], R['wstart']))
    return {k: (v[o] if np.ndim(v) and len(v) == len(o) else v) for k, v in R.items()}


def same_cands(Ra, Rb, tol):
    """same candidate set: same length and windows, detector points within tol px (both canonically ordered)"""
    if len(Ra['x_det']) != len(Rb['x_det']) or not np.array_equal(Ra['wstart'], Rb['wstart']): return False, np.inf
    d = max(np.abs(Ra['x_det'] - Rb['x_det']).max(), np.abs(Ra['y_det'] - Rb['y_det']).max()) if len(Ra['x_det']) else 0.0
    return d <= tol, d


def _pass1(fp32, **kw):
    from simcode.processing.pass1_inference import Pass1Front
    from simcode.processing.inference_fast import Pass1FrontFast
    mov, cal = sim_movie()
    if 'p1' not in _C:
        A = Pass1Front(CALIB, emitter=EMITTER, detector=DETECTOR)
        B = Pass1FrontFast(CALIB, emitter=EMITTER, detector=DETECTOR)
        B.detector = A.detector; B.emitter = A.emitter; B.front.detector = A.detector
        B.front.emitters = {'h': A.emitter}
        _C['p1'] = (A, B)
    A, B = _C['p1']
    A.emitter.bf16 = not fp32
    ctx = fp32_mode() if fp32 else contextlib.nullcontext()
    with ctx:
        Ra, _ = A.run_chunk(mov, cal, 0, mov.n)
        Rb, _ = B.run_chunk(mov, cal, 0, mov.n, **kw)
        Rr = None
    A.emitter.bf16 = True
    return Ra, Rb, Rr


def _check_pass1(fp32, **kw):
    r = _pass1(fp32, **dict(kw))
    if r is None:
        print(f'  skip (compile unavailable) {kw}'); return
    Ra, Rb, _ = (canon(q) if q is not None else None for q in r)
    assert len(Ra['x']) > 1000, 'too few candidates for a meaningful test'
    if fp32:
        ok, dd = same_cands(Ra, Rb, 1e-4)
    else:
        ok, dd = same_cands(Ra, Rb, BF16_DET)
    assert ok, f'candidate lists differ ({len(Ra["x"])} vs {len(Rb["x"])}, max |d x_det| {dd:.2e}) {kw}'
    dx = max(np.abs(Ra['x'] - Rb['x']).max(), np.abs(Ra['y'] - Rb['y']).max()); dI = np.abs(Ra['I'] - Rb['I']).max()
    if fp32:
        assert dx < 1e-4, f'fp32 |dx| {dx:.2e} {kw}'
        assert dI < 1e-2, f'fp32 |dI| {dI:.2e} {kw}'
    else:
        assert dx <= BF16_X and dI <= BF16_I, f'bf16 |dx| {dx:.2e} (tol {BF16_X}) |dI| {dI:.2e} (tol {BF16_I}) {kw}'
    assert np.array_equal(Ra['out'], Rb['out']) or np.mean(Ra['out'] != Rb['out']) < 1e-4
    print(f'  ok pass1 fp32={fp32} {kw}: n {len(Ra["x"])} max|dx| {dx:.2e} max|dI| {dI:.2e}')
    return Ra


def test_pass1_fp32():
    for kw in (dict(), dict(streams=False), dict(det_batch=16, emitter_batch=1024), dict(det_batch=7, emitter_batch=333)):
        _check_pass1(True, **kw)


def test_pass1_bf16():
    _check_pass1(False)


def _solve_cmp(a, c, tag, med_tol=1e-5):
    d = (a['r'] - c['r']).abs().amax(1)
    med, p99 = float(d.median()), float(d.quantile(0.99))
    kin = float(((a['kin'] - c['kin']).abs() < 1e-2).float().mean())          # kinetic-model posterior weight
    assert med < med_tol and p99 < 2e-3 and kin >= 0.999, f'{tag}: median {med:.2e} p99 {p99:.2e} kin {kin:.4f}'
    print(f'  ok {tag}: n {len(d)} median|dr| {med:.1e} p99 {p99:.1e} max {float(d.max()):.1e} kin-weight equal {kin:.4f}')


def test_pattern_solve():
    from simcode.solve import solve_fast as MF
    from simcode.solve import solve as SV
    from simcode.processing.common import pattern_frames
    Ra, _, _ = _pass1(False)
    cal = SV.solve_cal(CAL, 0.1)                          # a nonzero brightness noise (the c path of the solve)
    pat = torch.load(PATT, weights_only=False)
    keep = np.nonzero(Ra['out'])[0][:8192]
    tt = lambda a: torch.tensor(np.asarray(a), device='cuda', dtype=torch.float32)
    uw, inv = np.unique(Ra['wstart'][keep], return_inverse=True)
    PW = pattern_frames(pat, (uw[:, None] + np.arange(6)[None]).reshape(-1), 0, 0).reshape(len(uw), 6, 5)
    o = dict(I=tt(Ra['I'][keep]), L=tt(Ra['L'][keep]), C=tt(Ra['C'][keep]), x=tt(Ra['x'][keep]), y=tt(Ra['y'][keep]),
             lsxy=torch.log(tt(np.c_[Ra['sx'][keep], Ra['sy'][keep]])), x_det=tt(Ra['x_det'][keep]), y_det=tt(Ra['y_det'][keep]))
    b = torch.tensor(inv, device='cuda')
    a = SV.solve_emitters(o, PW, b, cal)
    _solve_cmp(a, MF.solve_emitters_fast(o, PW, b, cal), 'pattern solve (eager)')
    from simcode.util import compile as torch_compile
    if torch_compile.compile_available():
        # compiled LM step: float rounding of the fused kernels; on the rows of the stored-candidate cut 0.3 (more dim rows than
        # the earlier 0.5) the median is 2.3e-5 px (8.6e-6 px on the p >= 0.5 rows; docs/CLEANUP_LOG.md 2026-11-14)
        _solve_cmp(a, MF.solve_emitters_fast(o, PW, b, cal, compile='on'), 'pattern solve (compiled)', med_tol=5e-5)


def test_pass2_p2_fp32():
    from simcode.processing.pass2_inference import Pass2Front
    from simcode.processing.inference_fast import Pass2FrontFast
    mov, cal = sim_movie()
    pat = torch.load(PATT, weights_only=False)
    A = Pass2Front(CALIB, emitter=EMITTER_PASS2, detector=DETECTOR_PASS2)
    B = Pass2FrontFast(CALIB, emitter=EMITTER_PASS2, detector=DETECTOR_PASS2, compile='off')        # eager solve (the compiled LM step: test_pattern_solve)
    B.detector = A.detector; B.emitter = A.emitter
    from simcode.processing.inference_fast import Pass1FrontFast
    P1, _ = Pass1FrontFast(CALIB, emitter=EMITTER, detector=DETECTOR).run_chunk(mov, cal, 0, mov.n)    # the stored pass-1 rows (z of pass 2)
    A.emitter.bf16 = False
    with fp32_mode():
        from simcode.solve.solve import solve_cal
        c2 = solve_cal(CAL_PASS2, 0.1)
        Ra = A.run_chunk(mov, cal, 0, mov.n, pat, c2, P1, CAL)
        Rb = B.run_chunk(mov, cal, 0, mov.n, pat, c2, P1, CAL)
    A.emitter.bf16 = True
    Ra, Rb = canon(Ra), canon(Rb)
    ok, dd = same_cands(Ra, Rb, 1e-4)
    assert ok, f'pass-2 rows differ ({len(Ra["x_det"])} vs {len(Rb["x_det"])}, {dd:.2e})'
    dp = max(np.abs(Ra['x_prior'] - Rb['x_prior']).max(), np.abs(Ra['y_prior'] - Rb['y_prior']).max())
    assert dp < 1e-4, f'pass-2 prior |d| {dp:.2e}'
    d = np.abs(np.c_[Ra['x_hy0'] - Rb['x_hy0'], Ra['y_hy0'] - Rb['y_hy0']]).max(1)
    # median 1.1e-5 px since the solve uses the network's covariances without the kap scales (2026-11-17; < 1e-5 before): about
    # 1.5 float32 ulp of a global coordinate of ~100 px (7.6e-6)
    assert np.median(d) < 2e-5 and np.quantile(d, 0.99) < 2e-3, f'pass-2 solve median {np.median(d):.2e} p99 {np.quantile(d, 0.99):.2e}'
    # z from pass 1: the same links, z of the linked pass-1 candidate | pass-2 solved x, y (differences = the solve's)
    assert np.array_equal(Ra['p1_row'], Rb['p1_row']), 'pass-1 links differ'
    lk = Ra['p1_row'] >= 0
    # linked share on this dense 2.5 nM-like chunk: 0.61 of all rows (the detectors disagree on dim p < 0.5 candidates), 0.94
    # of the paper's selection (real 2.5 nM / 0.5 nM chunks: 0.98 / 0.99 of the selection)
    s12 = (Ra['p'] >= 0.63) & (np.maximum(Ra['sx_cal'], Ra['sy_cal']) * 108.33 <= 12)
    assert lk.mean() > 0.5 and lk[s12].mean() > 0.9, f'linked to pass 1: {lk.mean():.3f} of all rows, {lk[s12].mean():.3f} of the selection'
    assert np.isnan(Ra['z_hy0'][~lk]).all() and np.isnan(Rb['z_hy0'][~lk]).all()
    assert np.isfinite(Ra['z_hy0'][lk]).all() == np.isfinite(P1['z']).all()
    assert np.array_equal(Ra['z_prior'][lk], P1['z'][Ra['p1_row'][lk]])
    dz = np.abs(Ra['z_hy0'] - Rb['z_hy0'])[lk]
    assert np.median(dz) < 1e-5 and np.quantile(dz, 0.99) < 2e-3, f'pass-2 z median {np.median(dz):.2e} p99 {np.quantile(dz, 0.99):.2e}'
    # the network columns (result table locs_p2_net): the network position - prior_off, its sd, z | that position
    po = np.array(json.load(open(CAL_PASS2))['prior_off'])
    for R_ in (Ra, Rb):
        assert np.allclose(R_['x_net'], R_['x_prior'] - po[0], atol=1e-5) and np.allclose(R_['y_net'], R_['y_prior'] - po[1], atol=1e-5)
        assert np.isnan(R_['z_net'][~lk]).all() and np.isfinite(R_['z_net'][lk]).all() == np.isfinite(P1['z']).all()
    assert np.abs(Ra['sx_net'] - Rb['sx_net']).max() < 1e-4 and np.nanmax(np.abs(Ra['z_net'] - Rb['z_net'])) < 1e-3
    print(f'  ok pass2 (p2) fp32: n {len(d)} prior max|d| {dp:.1e} solve median {np.median(d):.1e} p99 {np.quantile(d, 0.99):.1e}; '
          f'linked to pass 1 {lk.mean():.3f} (selection {lk[s12].mean():.3f}), z median |d| {np.median(dz):.1e}')


def test_prefetch_reader():
    """Prefetch (background read of the next chunk) returns the same frames as the reader, in plan order and on a miss"""
    from simcode.processing.common import chunk_read_plan
    from simcode.io.movie import Prefetch
    mov, _ = sim_movie()
    plan = chunk_read_plan(mov, [0, 60, 120, 180], 60, mov.n, (5, 5, 110), 5)
    pm = Prefetch(mov, plan)
    for k in plan + [plan[0]]:                      # 4 hits, then 1 miss
        assert np.array_equal(pm.read(*k), mov.read(*k))
    assert pm.stats['hits'] == 4 and pm.stats['misses'] == 1, pm.stats
    print(f'  ok prefetch reader {pm.stats}')


def test_spatial_hash():
    """simcode/util/spatial_hash vs scipy cKDTree (key * 1e4 embedding) on a deterministic point set with duplicates, points
    on cell borders (0.5 px grid, radius 1.0 / 2.0 / 5.0 -> many distances exactly equal to the radius), empty keys and
    keys absent from the points; CPU and (if available) CUDA. Distances / counts / found sets must be identical; nearest
    indices may differ only between equidistant points."""
    from scipy.spatial import cKDTree
    from simcode.util import spatial_hash as SH
    rng = np.random.default_rng(7)
    n = 6000
    k = rng.integers(0, 40, n) * 3; k[k == 30] = 33                     # key 30 empty
    x = rng.integers(0, 60, n) * 0.5; y = rng.integers(0, 60, n) * 0.5
    x[:300] = x[300:600]; y[:300] = y[300:600]; k[:300] = k[300:600]    # exact duplicates
    P = np.c_[x, y, k * 1e4]; t = cKDTree(P)
    qx = np.r_[x, 3.0, 100.0]; qy = np.r_[y, 3.0, -50.0]; qk = np.r_[k, 30, 0]   # + a query in an empty key, one far outside
    devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for dev in devs:
        kw = dict(device=dev, block_points=1000, cand_cap=2000)        # small blocks / pieces: exercise the chunking
        for r in (1.0, 2.0, 5.0):
            for dt in (0, 6):
                dk, jk = t.query(np.c_[qx, qy, (qk + dt) * 1e4], distance_upper_bound=r)
                dh, jh = SH.nearest(x, y, k, qx, qy, qk + dt, r, **kw)
                assert np.array_equal(dk, dh), f'{dev} r {r} dt {dt}: distances differ'
                mis = np.isfinite(dk) & (jk != jh)
                d2 = lambda j: (qx[mis] - x[j[mis]]) ** 2 + (qy[mis] - y[j[mis]]) ** 2
                assert np.array_equal(d2(jk), d2(jh)), f'{dev} r {r} dt {dt}: index differs at unequal distance'
                assert np.array_equal(jk[~np.isfinite(dk)], jh[~np.isfinite(dh)])
            dd, _ = t.query(P, k=2)
            nh, _ = SH.nearest(x, y, k, x, y, k, r, qself=np.arange(n), **kw)
            assert np.array_equal(np.where(dd[:, 1] < r, dd[:, 1], np.inf), nh), f'{dev} r {r}: nearest-other differs'
            ck = t.query_ball_point(np.c_[qx, qy, qk * 1e4], r, return_length=True)
            assert np.array_equal(ck, SH.count(x, y, k, qx, qy, qk, r, **kw)), f'{dev} r {r}: counts differ'
    print(f'  ok spatial hash vs cKDTree ({", ".join(devs)}; r 1 / 2 / 5, shifted keys, duplicates, empty keys)')


if __name__ == '__main__':
    for f in (test_pass1_fp32, test_pass1_bf16, test_pattern_solve, test_pass2_p2_fp32, test_prefetch_reader, test_spatial_hash):
        print(f.__name__, flush=True); f()
    print('ALL OK')
