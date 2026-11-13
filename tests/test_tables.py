"""
CPU tests of the localization table and the evaluation helpers on synthetic rows (no GPU, no models, no raw data).
Run: python tests/test_tables.py (or pytest tests).
  strict   table columns; no 'main' column: select(table) = strict_u finite and max(sx, sy) * px_nm <= the stored sigma_max
           (global precision rule); select(table, 'u') = strict_u finite and <= the stored u_max; the crop of the run is read
           from its pass-1 file and a non-default crop moves the tiles; subset files hold exactly the selected rows
  drift    row selection of the drift estimate (simcode/util/drift.drift_rows; DME itself is not run)
  FRC      identical sets never drop below 1/7; block halves; a blurred point pattern resolves near its blur
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simcode import strict as SP


def _run_dir(d, n=1000, crop=(5, 5, 110), seed=0, lo=0.0, hi=120.0):
    """a fake pipeline output directory: one pass-1 file (only its crop entry is used) and one pass-2 chunk of n rows"""
    r = np.random.default_rng(seed)
    w = np.sort(r.integers(0, 2000, n)) * 3
    R = dict(wstart=w, frame=w + r.uniform(1, 4, n), x_hy0=r.uniform(lo, hi, n), y_hy0=r.uniform(lo, hi, n),
             sx_cal=r.uniform(0.02, 0.2, n).astype(np.float32), sy_cal=r.uniform(0.02, 0.2, n).astype(np.float32),
             phot=r.uniform(200, 3000, n).astype(np.float32), I_net=r.uniform(0, 500, (n, 6)).astype(np.float32),
             p=r.uniform(0.3, 1, n).astype(np.float32), q_dof=r.uniform(0.5, 2, n).astype(np.float32), kin_hy=r.random(n) < 0.3,
             x_prior=r.uniform(lo, hi, n), y_prior=r.uniform(lo, hi, n))
    np.savez(f'{d}/pass2_est2_0.npz', **R)
    np.savez(f'{d}/pass1_0.npz', crop=np.array(crop))
    return R


def test_strict_table():
    with tempfile.TemporaryDirectory() as d:
        _run_dir(d)
        assert SP.run_crop(d) == (5.0, 5.0, 110.0)
        T, info = SP.build_table(d, 'est2')
        for k in ('frame', 't', 'x', 'y', 'sx', 'sy', 'photons', 'I', 'p', 'q_dof', 'kin', 'dnll', 'x_prior', 'y_prior', 'good', 'sel', 'disjoint',
                  'strict_u', 'strict'):
            assert k in T and len(T[k]) == 1000, k
        u = T['strict_u']; inc = T['good'] & (T['x'] >= 5) & (T['x'] < 115) & (T['y'] >= 5) & (T['y'] < 115)
        assert np.array_equal(np.isfinite(u), inc)                                     # a rank exactly for the good rows inside the crop
        assert np.array_equal(T['good'], T['p'] >= SP.P_GOOD)
        uu = np.isfinite(u) & (u <= 0.45)
        prec = np.maximum(T['sx'], T['sy']).astype(np.float64) * SP.PX_NM_DEFAULT
        main = np.isfinite(u) & (prec <= SP.SIGMA_MAX_NM)
        assert 'main' not in T and np.array_equal(SP.select(T), main) and info['sigma_max'] == SP.SIGMA_MAX_NM and info['n_main'] == int(main.sum())
        assert 0 < main.sum() < np.isfinite(u).sum()                                   # the fixture straddles the threshold
        assert np.array_equal(SP.select(T, 'u'), uu) and info['u_max'] == 0.45 and info['n_u'] == int(uu.sum())
        assert np.array_equal(SP.select(T, sigma_max=8.0), np.isfinite(u) & (prec <= 8.0))
        assert np.array_equal(SP.select(T, sigma_max=8.0, px_nm=100.0), np.isfinite(u) & (prec / SP.PX_NM_DEFAULT * 100 <= 8.0))
        try: SP.select(T, 0.2); raise AssertionError('a float rule must raise')
        except TypeError: pass
        assert np.array_equal(SP.select(T, rule='strict'), T['strict'])
        assert np.array_equal(T['strict'], np.isfinite(u) & (u <= info['q'])) and not (T['strict'] & ~T['good']).any()
        assert 0 < np.nanmin(u) and np.nanmax(u) < 1 and SP.select(T, rule='all').all()
        T2, i2 = SP.build_table(d, 'est2', u_max=0.2, sigma_max=12.5)
        assert np.array_equal(SP.select(T2, 'u', u_max=0.2), np.isfinite(u) & (u <= 0.2)) and i2['u_max'] == 0.2 and np.array_equal(T2['strict_u'], u, equal_nan=True)
        assert i2['sigma_max'] == 12.5 and i2['n_main'] == int((np.isfinite(u) & (prec <= 12.5)).sum())
        # table and subset files
        files = SP.write_table(T, info, f'{d}/locs_p2'); fn = SP.write_subset(T, f'{d}/locs_p2', 'main'); fs = SP.write_subset(T, f'{d}/locs_p2', 'strict')
        fu = SP.write_subset(T, f'{d}/locs_p2', 'u')
        assert os.path.basename(fn) == f"locs_p2_sig{SP.SIGMA_MAX_NM:g}.npz" and os.path.basename(fs) == 'locs_p2_strictonly.npz' and files == [f'{d}/locs_p2.npz']
        assert os.path.basename(fu) == 'locs_p2_u045.npz' and np.array_equal(np.load(fu)['x'], T['x'][uu])
        L = np.load(f'{d}/locs_p2.npz'); S = np.load(fn)
        assert float(L['u_max']) == 0.45 and float(L['sigma_max']) == SP.SIGMA_MAX_NM and float(L['px_nm']) == SP.PX_NM_DEFAULT and 'main' not in L.files
        assert np.array_equal(SP.select(L), main) and np.array_equal(SP.select(L, 'u'), uu)   # the stored thresholds are used when a table is read back
        SP.write_table(T2, i2, f'{d}/t2'); L2 = np.load(f'{d}/t2.npz')
        assert np.array_equal(SP.select(L2, 'u'), np.isfinite(u) & (u <= 0.2)) and np.array_equal(SP.select(L2), np.isfinite(u) & (prec <= 12.5))
        assert os.path.basename(SP.write_subset(L2, f'{d}/t2', 'main')) == 't2_sig12p5.npz'
        assert S['sel'].all() and np.array_equal(S['x'], T['x'][main]) and np.array_equal(S['frame'], T['frame'][main]) and float(S['sigma_max']) == SP.SIGMA_MAX_NM
        assert SP.subset_name('u', 0.3) == 'u030' and SP.subset_name('main', sigma_max=10.0) == 'sig10' and SP.subset_name('main', sigma_max=12.5) == 'sig12p5'
    print('  ok strict table: columns, main (global sigma) / u / strict / select, sigma_max, u_max, subset files')


def test_strict_crop():
    with tempfile.TemporaryDirectory() as d:
        _run_dir(d, crop=(0, 0, 40), lo=0.0, hi=60.0)                                   # a simulation-like field, crop from the run
        assert SP.run_crop(d) == (0.0, 0.0, 40.0)
        T, info = SP.build_table(d, 'est2')
        inc = T['good'] & (T['x'] >= 0) & (T['x'] < 40) & (T['y'] >= 0) & (T['y'] < 40)
        assert np.array_equal(np.isfinite(T['strict_u']), inc) and info['crop'] == [0.0, 0.0, 40.0]
        Td, _ = SP.build_table(d, 'est2', crop=SP.CROP)                                 # the old hard-coded crop on the same rows: a different product
        assert not np.array_equal(np.isfinite(Td['strict_u']), inc)
        # per-tile rank: within one 10-px tile the rank follows sigma
        t = SP.tile_of(T['x'], T['y'], (0.0, 0.0, 40.0)); k = np.nonzero(inc & (t == np.bincount(t[inc]).argmax()))[0]
        sig = 0.5 * (T['sx'] + T['sy']).astype(np.float64)
        assert np.array_equal(np.argsort(T['strict_u'][k], kind='stable'), np.argsort(sig[k], kind='stable'))
    print('  ok strict crop: read from the run, tiles and ranks follow it')


def test_drift_rows():
    from simcode.util.drift import drift_rows
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0]); y = np.ones(7)
    fr = np.array([0, 3, 6, 9, 12, 96, 97]); sx = np.array([0.1, 1.5, 0.1, 0.0, 0.1, 0.1, 0.1]); sy = np.full(7, 0.1)
    pos, t, crlb = drift_rows(x, y, fr, sx, sy, 100)
    assert t.tolist() == [3, 15, 99] and pos.dtype == np.float32 and crlb.shape == (3, 2)     # sigma >= 1, nan, sigma 0 and t >= F dropped
    assert drift_rows(x, y, fr, sx, sy, 100, t_offset=0)[1].tolist() == [0, 12, 96, 97]
    print('  ok drift row selection')


def test_frc():
    from simcode.evalproto import frc as FR
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
