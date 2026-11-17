"""
CPU tests of the localization table and the evaluation helpers on synthetic rows (no GPU, no models, no raw data).
Run: python tests/test_tables.py (or pytest tests).
  tables   the four tables (net / solve x pass 1 / 2): columns, same rows in both kinds, p1_index; select(table) = p >= 0.63,
           inside the crop, max(sx, sy) * px_nm <= 12 nm (max_precision_nm / min_score / in_region / crop / px_nm); the crop of
           the run is read from its pass-1 file; the HDF5 table file round trips and the selection is applied at load; the
           default table of a run directory is locs_p2_net; no sigma rescale
  drift    row selection of the drift estimate (simcode/util/drift.drift_rows; DME itself is not run)
  FRC      identical sets never drop below 1/7; block halves; a blurred point pattern resolves near its blur
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simcode import table as SP


def _run_dir(d, n=1000, crop=(5, 5, 110), seed=0, lo=0.0, hi=120.0, z=False):
    """a fake pipeline output directory: one pass-1 chunk (pass1_0.npz: crop, out; the pass-1 solve rows pass2_est_0.npz) and
    one pass-2 chunk of n rows (pass2_est2_0.npz) with the solve and the network columns"""
    r = np.random.default_rng(seed)
    def rows(n, p2):
        w = np.sort(r.integers(0, 2000, n)) * 3
        R = dict(wstart=w, frame=w + r.uniform(1, 4, n), x_hy0=r.uniform(lo, hi, n), y_hy0=r.uniform(lo, hi, n),
                 sx_cal=r.uniform(0.02, 0.2, n).astype(np.float32), sy_cal=r.uniform(0.02, 0.2, n).astype(np.float32),
                 phot=r.uniform(200, 3000, n).astype(np.float32), I_net=r.uniform(0, 500, (n, 6)).astype(np.float32),
                 p=r.uniform(0.3, 1, n).astype(np.float32), q_dof=r.uniform(0.5, 2, n).astype(np.float32), kin_hy=r.random(n).astype(np.float32),
                 x_prior=r.uniform(lo, hi, n), y_prior=r.uniform(lo, hi, n))
        R.update(x_net=R['x_prior'] - 0.01, y_net=R['y_prior'] + 0.01, sx_net=r.uniform(0.02, 0.2, n).astype(np.float32),
                 sy_net=r.uniform(0.02, 0.2, n).astype(np.float32))
        if z:      # 3D networks
            for k in ('z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_net', 'sz_net', 'cov_xz_net', 'cov_yz_net', 'z_prior'):
                R[k] = r.normal(0, 1, n).astype(np.float32)
        if p2: R['p1_row'] = np.where(r.random(n) < 0.9, r.integers(0, 2 * n, n), -1)
        else: R['dnll'] = r.uniform(0, 10, n).astype(np.float32)
        return R
    out = np.zeros(2 * n, bool); out[r.choice(2 * n, n, replace=False)] = True
    np.savez(f'{d}/pass1_0.npz', crop=np.array(crop), out=out)
    np.savez(f'{d}/pass2_est_0.npz', **rows(n, False))
    R = rows(n, True)
    np.savez(f'{d}/pass2_est2_0.npz', **R)
    R['_out'] = out
    return R


SOLVE_ONLY = ('q_dof', 'kin', 'dnll')
COMMON = ('frame', 't', 'x', 'y', 'sx', 'sy', 'photons', 'I', 'p', 'x_prior', 'y_prior', 'z', 'sz', 'cov_xz', 'cov_yz', 'z_prior',
          'good', 'sel', 'disjoint')


def test_strict_table():
    with tempfile.TemporaryDirectory() as d:
        R = _run_dir(d, z=True)
        assert SP.run_crop(d) == (5.0, 5.0, 110.0)
        T, info = SP.build_table(d, 'est2', kind='solve')
        for k in COMMON + SOLVE_ONLY + ('p1_index',):
            assert k in T and len(T[k]) == 1000, k
        assert 'strict_u' not in T and 'strict' not in T and 'u_max' not in info and 'sig_u' not in info      # removed rules / rescale
        assert np.array_equal(T['x'], R['x_hy0']) and np.array_equal(T['sx'], R['sx_cal']) and T['kin'].dtype == np.float32
        Tn, inn = SP.build_table(d, 'est2')                                             # kind 'net' (the default)
        assert not [k for k in SOLVE_ONLY if k in Tn] and all(k in Tn for k in COMMON) and inn['kind'] == 'net'
        assert np.array_equal(Tn['x'], R['x_net']) and np.array_equal(Tn['sy'], R['sy_net']) and np.array_equal(Tn['z'], R['z_net'].astype(np.float64))
        assert np.array_equal(Tn['frame'], T['frame']) and np.array_equal(Tn['p'], T['p'])                # same rows, same order
        # p1_index: the linked pass-1 row in the pass-1 tables (the 'out' rows of pass1_0.npz); -1 for none / not stored
        rank = np.cumsum(R['_out']) - 1
        exp = np.where((R['p1_row'] >= 0) & R['_out'][np.maximum(R['p1_row'], 0)], rank[np.maximum(R['p1_row'], 0)], -1)
        assert np.array_equal(T['p1_index'], exp) and np.array_equal(Tn['p1_index'], exp)
        inc = T['good'] & (T['x'] >= 5) & (T['x'] < 115) & (T['y'] >= 5) & (T['y'] < 115)
        assert np.array_equal(T['good'], T['p'] >= SP.P_GOOD) and info['n_good_crop'] == int(inc.sum())
        prec = np.maximum(T['sx'], T['sy']).astype(np.float64) * SP.PX_NM_DEFAULT
        main = inc & (prec <= SP.SIGMA_MAX_NM)
        assert np.array_equal(SP.select(T), main) and 'sigma_max' not in info and info['n_main'] == int(main.sum())
        assert abs(info['median_sigma_nm'] - np.median(prec[main])) < 1e-9
        assert 0 < main.sum() < inc.sum()                                               # the fixture straddles the threshold
        assert np.array_equal(SP.select(T, 8.0), inc & (prec <= 8.0))
        assert np.array_equal(SP.select(T, max_precision_nm=8.0, px_nm=100.0), inc & (prec / SP.PX_NM_DEFAULT * 100 <= 8.0))
        assert np.array_equal(SP.select(T, min_score=0.8), (T['p'] >= 0.8) & (T['x'] >= 5) & (T['x'] < 115) & (T['y'] >= 5) & (T['y'] < 115) & (prec <= 12))
        assert np.array_equal(SP.select(T, in_region=False), T['good'] & (prec <= 12))
        assert SP.select(T, None, None, False).all()
        # pass 1: no p1_index
        T1, i1 = SP.build_table(d, 'est', kind='net'); assert 'p1_index' not in T1 and len(T1['x']) == 1000
        # the table file: torch_psf Dataset HDF5; the selection at load
        fn = SP.write_table(T, info, f'{d}/locs_p2_solve')
        assert fn == f'{d}/locs_p2_solve.hdf5' and not [f for f in os.listdir(d) if 'sig' in f]       # no selection files
        from simcode.dataset import Dataset
        L = Dataset.load_table(fn, info=False); C = L.columns()
        assert L.meta['crop'] == [5.0, 5.0, 110.0] and L.pixel_nm == SP.PX_NM_DEFAULT and L.meta['tag'] == 'est2' and L.meta['kind'] == 'solve'
        eq = lambda k: np.array_equal(C[k], T[k], equal_nan=T[k].dtype.kind == 'f')
        assert all(eq(k) for k in T), [k for k in T if not eq(k)]
        assert np.array_equal(SP.select(C), main) and np.array_equal(L.mask().numpy(), main)
        # the four tables of a run; the result table is locs_p2_net
        res = SP.write_tables(d)
        assert set(res) == set(SP.TABLES) and SP.DEFAULT_TABLE == 'locs_p2_net' and res['locs_p2_net'][0]['default']
        assert all(os.path.exists(f'{d}/{n}.hdf5') and os.path.exists(f'{d}/{n}_info.json') for n in SP.TABLES)
        L = Dataset.load(d)
        assert L.meta['table'] == 'locs_p2_net' and L.meta['kind'] == 'net' and L.meta['pass'] == 2 and np.array_equal(L.columns()['x'], Tn['x'])
        assert Dataset.from_run(d, pass_=1, kind='solve').meta['table'] == 'locs_p1_solve'
    print('  ok table: net / solve columns, p1_index, select (max_precision_nm, min_score, in_region), HDF5 table file, the four tables')


def test_strict_crop():
    with tempfile.TemporaryDirectory() as d:
        _run_dir(d, crop=(0, 0, 40), lo=0.0, hi=60.0)                                   # a simulation-like field, crop from the run
        assert SP.run_crop(d) == (0.0, 0.0, 40.0)
        for kind in ('net', 'solve'):
            T, info = SP.build_table(d, 'est2', kind=kind)
            inc = T['good'] & (T['x'] >= 0) & (T['x'] < 40) & (T['y'] >= 0) & (T['y'] < 40)
            assert info['crop'] == [0.0, 0.0, 40.0] and info['n_good_crop'] == int(inc.sum())
            prec = np.maximum(T['sx'], T['sy']).astype(np.float64) * SP.PX_NM_DEFAULT
            assert np.array_equal(SP.select(T, crop=(0.0, 0.0, 40.0)), inc & (prec <= 12)) and info['n_main'] == int((inc & (prec <= 12)).sum())
        assert not hasattr(SP, 'tile_rank') and not hasattr(SP, 'model_sig_u')          # the sigma rescale g(u) is gone
    print('  ok strict crop: read from the run, the selection region follows it')


def test_drift_rows():
    from simcode.util.drift import drift_rows
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0]); y = np.ones(7)
    fr = np.array([0, 3, 6, 9, 12, 96, 97]); sx = np.array([0.1, 1.5, 0.1, 0.0, 0.1, 0.1, 0.1]); sy = np.full(7, 0.1)
    pos, t, crlb = drift_rows(x, y, fr, sx, sy, 100)
    assert t.tolist() == [3, 15, 99] and pos.dtype == np.float32 and crlb.shape == (3, 2)     # sigma >= 1, nan, sigma 0 and t >= F dropped
    assert drift_rows(x, y, fr, sx, sy, 100, t_offset=0)[1].tolist() == [0, 12, 96, 97]
    print('  ok drift row selection')


def test_frc():
    from simcode.evaluation import frc as FR
    r = np.random.default_rng(3)
    sx_, sy_ = r.uniform(10, 30, 4000), r.uniform(10, 30, 4000)                         # random binding sites (broadband structure)
    def draw(n, blur_nm):
        i = r.integers(0, len(sx_), n); s = blur_nm / 108.33
        return sx_[i] + r.normal(0, s, n), sy_[i] + r.normal(0, s, n)
    xa, ya = draw(200000, 10.0); xb, yb = draw(200000, 10.0)
    assert np.isnan(FR.frc(xa, ya, xa, ya))                                             # identical sets: never below 1/7
    res, f, c = FR.frc(xa, ya, xb, yb, return_curve=True)
    assert res == FR.frc(xa, ya, xb, yb) and len(f) == len(c) and c[3:8].mean() > 0.9
    xc, yc = draw(200000, 30.0); xd, yd = draw(200000, 30.0)
    res30 = FR.frc(xc, yc, xd, yd)
    assert 15 < res < 60 and res30 > 1.8 * res, (res, res30)                            # resolution scales with the blur
    x = np.r_[xa, xb]; y = np.r_[ya, yb]; fr = np.r_[np.zeros(len(xa), int), np.full(len(xb), 300)]
    assert FR.frc_halves(x, y, fr) == res
    assert abs(FR.frc(xa * 108.33, ya * 108.33, xb * 108.33, yb * 108.33, px=5.0, units='nm') - res) < 3
    print(f'  ok FRC: 10 nm blur -> {res:.1f} nm, 30 nm blur -> {res30:.1f} nm; identical sets nan; block halves; nm units')


if __name__ == '__main__':
    for f in (test_strict_table, test_strict_crop, test_drift_rows, test_frc):
        print(f.__name__, flush=True); f()
    print('ALL OK')
