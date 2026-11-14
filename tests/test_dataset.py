"""
CPU tests of simcode.dataset.Dataset (no models, no raw data; the two tests on files of the paper runs and the DME test are
skipped when the files / the dme package / a GPU are not available).
Run: python tests/test_dataset.py (or pytest tests/test_dataset.py).
  table      built from a fake run directory with simcode.strict; select / mask equal strict.select; indexing; filters; crop
  formats    npz (lossless, and still a v2 table for the existing readers), Picasso hdf5 (+ yaml; field names / dtypes of the
             paper's sf.hdf5), ThunderSTORM csv (1-based frames, 0.5 px origin), the preprint code's .npy, DECODE npz
  drift      apply / undo, carried through indexing and files; estimate (DME v2) equals simcode.util.drift.drift_of_table
  v1 names   the preprint class's interface (saveHDF5 / loadHDF5, saveNumpy / loadNumpy, merge, cropFromEdge, applyDrift, ...),
             addFields: extra fields through indexing, merge, npz / hdf5 / npy / csv; from_run(pass_)
  render     histogram counts and geometry;  frames  nframes, indices_per_frame;  merge
  real       out/paper2/1809_2/v2_matched/locs_p2.npz: counts of the info json; FRC through the class equals
             analysis/paper/frc_fair.py on a subset
"""
import json, os, sys, tempfile
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from simcode import strict as SP
from simcode.dataset import Dataset, Localizations
from tests.test_tables import _run_dir

REAL = os.path.join(REPO, 'out/paper2/1809_2/v2_matched/locs_p2.npz')
SF = os.path.join(REPO, 'out/paper2/1809_2/sf/sf.hdf5')


def _skip(msg):
    try:
        import pytest
    except ImportError:
        print('skipped:', msg); return
    pytest.skip(msg)


def _table(d, n=3000):
    _run_dir(d, n=n)
    T, info = SP.build_table(d, 'est2')
    SP.write_table(T, info, f'{d}/locs_p2'); json.dump(info, open(f'{d}/locs_p2_info.json', 'w'), default=str)
    return T, info


def _same(A, B, keys=None):
    for k in keys or A.columns:
        assert A[k].dtype == B[k].dtype and np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == 'f'), k


def test_table_select_index():
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        L = Dataset.load(d)                                                     # run directory -> locs_p2.npz
        assert len(L) == 3000 and set(L.columns) == set(T) and L.px_nm == SP.PX_NM_DEFAULT and L.window == 6 and L.t_offset == 3
        assert L.meta['sigma_max'] == SP.SIGMA_MAX_NM and L.meta['u_max'] == SP.U_MAX and L.meta['crop'] == [5.0, 5.0, 110.0]
        assert L.meta['info']['n_main'] == info['n_main'] and L.I.shape == (3000, 6) and np.array_equal(L.x, T['x']) and L['x'] is L.x
        Tz = np.load(f'{d}/locs_p2.npz')
        for rule, kw in (('main', {}), ('main', dict(sigma_max=8.0)), ('u', {}), ('u', dict(u_max=0.2)), ('strict', {}), ('sel', {}), ('good', {}), ('all', {})):
            m = SP.select(Tz, rule, **kw)
            assert np.array_equal(L.mask(rule, **kw), m), rule
            S = L.select(rule, **kw)
            assert len(S) == m.sum() and np.array_equal(S.x, T['x'][m]) and np.array_equal(S.I, T['I'][m]) and S.meta['selection']['rule'] == rule
        assert L.info()['n_main'] == info['n_main'] and L.info()['n_u'] == info['n_u'] and 'main selection' in repr(L)
        assert np.allclose(L.precision_nm, np.maximum(T['sx'], T['sy']).astype(np.float64) * SP.PX_NM_DEFAULT)
        # indexing: mask, slice, index array, one row; a new object every time
        m = T['good']; A = L[m]
        assert len(A) == m.sum() and len(L) == 3000 and np.array_equal(A.frame, T['frame'][m])
        assert len(L[10:20]) == 10 and np.array_equal(L[[5, 3]].x, T['x'][[5, 3]]) and len(L[7]) == 1 and L[7].x[0] == T['x'][7]
        try: L[m[:10]]; raise AssertionError('a mask of the wrong length must raise')
        except IndexError: pass
        # filters
        assert np.array_equal(L.filter_photons(500, 2000).photons, T['photons'][(T['photons'] >= 500) & (T['photons'] <= 2000)])
        s = np.maximum(T['sx'], T['sy']).astype(np.float64)
        assert len(L.filter_sigma(max_nm=10)) == (s * SP.PX_NM_DEFAULT <= 10).sum() and len(L.crlb_filter(0.15)) == (s < 0.15).sum()
        assert len(L.filter_frames(300, 600)) == ((T['frame'] >= 300) & (T['frame'] + 6 <= 600)).sum()
        R = L.filter_region(20, 30, 60, 90); assert R.x.min() >= 20 and R.x.max() < 60 and R.y.min() >= 30 and R.y.max() < 90
        C = L.crop(20, 30, size=40); assert len(C) and C.meta['region'] == [20, 30, 60, 70] and C.x.min() >= 20 and C.y.max() < 70
        C2 = L.crop(20, 30, 60, 70, shift=True); assert np.allclose(C2.x, C.x - 20) and C2.meta['origin'] == [20, 30] and C2.region() == (0, 0, 40, 40)
        L['w'] = np.arange(3000); assert np.array_equal(L[m].w, np.nonzero(m)[0])
        # frames
        assert L.nframes == T['frame'].max() + 6 and L.numFrames == L.nframes
        ipf = L.indices_per_frame(); assert len(ipf) == L.nframes and sum(len(i) for i in ipf) == 3000
        f0 = int(T['frame'][100]); assert np.array_equal(np.sort(ipf[f0]), np.nonzero(T['frame'] == f0)[0])
        # merge
        B = Dataset.concatenate([L[:100], L[100:250]], set_group=True)
        assert len(B) == 250 and np.array_equal(B.x, T['x'][:250]) and B.group[99] == 0 and B.group[100] == 1 and len(L[:5] + L[5:9]) == 9


def test_npz_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        L = Dataset.from_run(d); L.set_drift(np.random.default_rng(1).normal(0, 0.1, (L.nframes, 2)).astype(np.float32))
        L.meta['model'] = 'mtest'
        f = L.save(f'{d}/copy.npz'); M = Dataset.load(f)
        assert set(M.columns) == set(L.columns); _same(L, M)
        assert np.array_equal(M.drift, L.drift) and not M.drift_applied and M.meta['model'] == 'mtest' and M.meta['info'] == L.meta['info']
        assert M.meta['sigma_max'] == L.meta['sigma_max'] and M.px_nm == L.px_nm and M.meta['crop'] == L.meta['crop']
        # still a v2 table for the existing readers
        z = np.load(f); assert np.array_equal(SP.select(z), SP.select(T)) and float(z['px_nm']) == SP.PX_NM_DEFAULT
        from simcode.util.drift import drift_rows
        assert len(drift_rows(z['x'], z['y'], z['frame'], z['sx'], z['sy'], L.nframes)[0])
        # drift applied -> stored corrected, flag kept, undo gives the raw positions
        L.apply_drift(); M = Dataset.load(L.save(f'{d}/dc.npz'))
        assert M.drift_applied and np.array_equal(M.x, L.x) and np.allclose(M.undo_drift().x, T['x'], atol=1e-9, rtol=0)
        # column subset; subset files of strict
        P = Dataset.from_table(f'{d}/locs_p2.npz', columns=('sx', 'sy')); assert set(P.columns) == {'x', 'y', 'frame', 'sx', 'sy'} and P.meta['sigma_max'] == 12.0
        S = Dataset.load(SP.write_subset(T, f'{d}/locs_p2', 'main')); assert len(S) == info['n_main'] and S.meta['sigma_max'] == 12.0
        assert np.array_equal(S.x, L.select('main').undo_drift().x) or np.allclose(S.x, L.select('main').undo_drift().x, atol=1e-9, rtol=0)


def test_picasso_roundtrip():
    import h5py
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        L = Dataset.from_run(d); L.meta.update(imgshape=[120, 120], nframes=6100)
        f = L.save(f'{d}/a.hdf5')
        with h5py.File(f, 'r') as h:
            dt = h['locs'].dtype
            A = h['locs'][:]
        # the fields of Picasso Localize, names and dtypes as in the paper's sf.hdf5
        want = [('frame', '<u4'), ('x', '<f4'), ('y', '<f4'), ('photons', '<f4'), ('sx', '<f4'), ('sy', '<f4'), ('bg', '<f4'), ('lpx', '<f4'), ('lpy', '<f4'),
                ('ellipticity', '<f4'), ('net_gradient', '<f4')]
        assert [(k, dt[k].str) for k in dt.names[:len(want)]] == want
        assert np.array_equal(A['lpx'], T['sx']) and np.allclose(A['sx'], 1.3) and np.array_equal(A['x'], T['x'].astype(np.float32)) and not A['bg'].any()
        Y = open(f'{d}/a.yaml').read(); assert 'Frames: 6100' in Y and 'Height: 120' in Y and 'Width: 120' in Y and 'Byte Order: "<"' in Y
        try:
            import yaml
            y = yaml.safe_load(Y); assert y['Frames'] == 6100 and y['Byte Order'] == '<' and y['Pixelsize'] == SP.PX_NM_DEFAULT
        except ImportError:
            pass
        M = Dataset.load(f)
        assert M.window == 6 and M.nframes == 6100 and M.meta['imgshape'] == [120, 120] and M.px_nm == L.px_nm
        assert set(M.columns) == set(L.columns) | {'psf_sx', 'psf_sy', 'bg'}
        for k in L.columns:
            a, b = L[k], M[k]
            assert a.shape == b.shape, k
            if k in ('x', 'y', 'frame'): assert np.allclose(a, b, atol=1e-5, rtol=0), k                # float32 positions
            else: assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'), k
        assert np.array_equal(M.mask('main', sigma_max=12.0), L.mask('main'))
        f2 = L.select('main').save(f'{d}/b.hdf5', extra=False)
        with h5py.File(f2, 'r') as h: assert h['locs'].dtype.names == tuple(k for k, _ in want) and len(h['locs']) == info['n_main']


def test_picasso_paper_file():
    """the paper's SIMFLUX table (written by the preprint code): same values as that code's loader gives (frc_fair.load before)"""
    if not os.path.exists(SF): return _skip('no ' + SF)
    import h5py
    L = Dataset.from_picasso(SF, window=6)
    with h5py.File(SF, 'r') as h: A = h['locs'][:]
    assert np.array_equal(L.x, A['x']) and np.array_equal(L.sx, A['lpx']) and np.array_equal(L.sy, A['lpy']) and np.array_equal(L.psf_sx, A['sx'])
    assert np.array_equal(L.frame, A['frame']) and np.array_equal(L.photons, A['photons']) and np.array_equal(L.bg, A['bg']) and 'chisq' in L
    assert L.meta['imgshape'] == [120, 120] and L.nframes == 119995 and L.window == 6
    with tempfile.TemporaryDirectory() as d:
        S = L[:5000]; S.save(f'{d}/s.hdf5')
        with h5py.File(f'{d}/s.hdf5', 'r') as h: B = h['locs'][:]
        for k in ('frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'chisq', 'lI', 'lbg', 'roi_index'):
            assert B.dtype[k] == A.dtype[k] and np.array_equal(B[k], A[k][:5000], equal_nan=True), k


def test_thunderstorm_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d, n=500)
        L = Dataset.from_run(d)
        f = L.save(f'{d}/a.csv')
        hdr = open(f).readline().strip().split(',')
        assert hdr[:4] == ['"id"', '"frame"', '"x [nm]"', '"y [nm]"'] and '"uncertainty [nm]"' in hdr and '"intensity [photon]"' in hdr and '"sigma [nm]"' in hdr
        row = np.loadtxt(f, delimiter=',', skiprows=1)[0]
        assert row[1] == T['frame'][0] + 1 and abs(row[2] - (T['x'][0] + 0.5) * SP.PX_NM_DEFAULT) < 1e-4        # frames from 1, pixel centre at 0.5
        M = Dataset.load(f)
        assert set(M.columns) == set(L.columns) | {'psf_sx', 'psf_sy', 'bg'} and np.array_equal(M.frame, L.frame)
        assert np.allclose(M.x, L.x, atol=2e-6, rtol=0) and np.allclose(M.y, L.y, atol=2e-6, rtol=0)             # 1e-4 nm
        assert np.allclose(M.sx, L.sx, atol=1e-6, rtol=0) and np.allclose(M.photons, L.photons, rtol=1e-6) and M.I.shape == (500, 6)
        assert np.allclose(M.I, L.I, rtol=1e-6, atol=1e-6) and np.allclose(M.strict_u, L.strict_u, equal_nan=True, atol=1e-6)
        for k in ('good', 'sel', 'strict', 'disjoint', 'kin'): assert M[k].dtype == bool and np.array_equal(M[k], L[k]), k
        assert np.array_equal(M.mask('u', u_max=0.45), L.mask('u'))


def test_v1_and_decode_files():
    """the preprint code's numpy file (structured array, read without that package) and the paper's DECODE npz"""
    r = np.random.default_rng(3); n = 200
    est = np.dtype([('pos', '<f4', (2,)), ('photons', '<f4'), ('bg', '<f4'), ('ibg', '<f4', (6, 2))])
    dt = np.dtype([('roi_id', '<i4'), ('frame', '<i4'), ('estim', est), ('crlb', est), ('chisq', '<f4'), ('group', '<i4'), ('roipos', '<i4', (2,))])
    D = np.zeros(n, dt); D['frame'] = np.sort(r.integers(0, 900, n)); D['estim']['pos'] = r.uniform(5, 115, (n, 2)); D['crlb']['pos'] = r.uniform(0.02, 0.2, (n, 2))
    D['estim']['photons'] = r.uniform(300, 3000, n); D['estim']['bg'] = r.uniform(1, 9, n); D['estim']['ibg'] = r.uniform(0, 500, (n, 6, 2)); D['chisq'] = r.uniform(0, 2, n)
    with tempfile.TemporaryDirectory() as d:
        with open(f'{d}/smlm.npy', 'wb') as s:
            np.save(s, np.array([120, 120])); np.save(s, D)
        L = Dataset.load(f'{d}/smlm.npy', window=6)
        assert np.array_equal(L.x, D['estim']['pos'][:, 0]) and np.array_equal(L.y, D['estim']['pos'][:, 1]) and np.array_equal(L.sx, D['crlb']['pos'][:, 0])
        assert np.array_equal(L.sy, D['crlb']['pos'][:, 1]) and np.array_equal(L.photons, D['estim']['photons']) and np.array_equal(L.bg, D['estim']['bg'])
        assert np.array_equal(L.I, D['estim']['ibg'][..., 0]) and np.array_equal(L.I_bg, D['estim']['ibg'][..., 1]) and np.array_equal(L.chisq, D['chisq'])
        assert L.meta['imgshape'] == [120, 120] and L.window == 6 and np.array_equal(L.crlb, D['crlb']['pos'])
        assert np.array_equal(Dataset.from_smlmtorch(D.view(np.recarray)).x, L.x)
        np.savez(f'{d}/dec.npz', frame=D['frame'].astype(np.int64), x=L.x, y=L.y, z=np.zeros(n, np.float32), sx=L.sx, sy=L.sy, phot=L.photons, sphot=L.photons * 0.1,
                 bg=L.bg, prob=r.uniform(0.5, 1, n).astype(np.float32), info=np.array('decode test'))
        for E in (Dataset.load(f'{d}/dec.npz'), Dataset.from_decode(f'{d}/dec.npz')):
            assert np.array_equal(E.photons, L.photons) and 'p' in E and 'photons_sigma' in E and 'z' in E and E.window == 6 and E.meta['info'] == 'decode test'
            assert E.pos.shape == (n, 3)
        A = Dataset.from_arrays(L.x, L.y, L.frame, L.sx, L.sy, photons=L.photons, px_nm=100.0, window=1, p=np.ones(n))
        assert A.px_nm == 100.0 and A.t_offset == 0 and np.array_equal(A.pos_nm[:, 0], L.x.astype(np.float64) * 100) and A.frame.dtype == np.int64


def test_v1_interface_and_fields():
    """the preprint class's names; extra fields through indexing, merge and every format; from_run(pass_)"""
    import simcode.locs as LO
    assert Localizations is Dataset and LO.Dataset is Dataset and LO.Localizations is Dataset
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        T1, i1 = SP.build_table(d, 'est2', sigma_max=10.0); SP.write_table(T1, i1, f'{d}/locs_p1')          # a second table as "pass 1"
        ds = Dataset.from_run(d); assert ds.meta['pass'] == 2 and Dataset.from_run(d, pass_=1).meta['sigma_max'] == 10.0 and Dataset.load(d).meta['pass'] == 2
        assert Dataset.from_run(d, pass_=1).meta['pass'] == 1 and len(Dataset.from_run(d, table='locs_p1', columns=())) == 3000
        try: Dataset.from_run(d, pass_=3); raise AssertionError('a missing pass must raise')
        except FileNotFoundError: pass
        ds.meta['imgshape'] = [120, 120]
        assert ds.numFrames == ds.nframes and ds.dims == 2 and ds.imgshape == [120, 120] and not ds.isDriftCorrected and len(ds.indicesPerFrame()) == ds.numFrames
        assert ds.pos.shape == (3000, 2) and ds.crlb.shape == (3000, 2) and ds.photons is ds['photons'] and ds.frame is ds['frame']
        r = np.random.default_rng(4)
        ds.addFields(bg=r.uniform(1, 5, 3000).astype(np.float32), track=r.integers(0, 50, 3000), feat=r.normal(size=(3000, 3)).astype(np.float32), flag=r.random(3000) < 0.5)
        assert ds.background is ds.bg and {'track', 'feat', 'flag', 'p', 'strict_u', 'I', 'kin'} <= set(ds.fields)
        try: ds.addFields(bad=np.zeros(5)); raise AssertionError('a field of the wrong length must raise')
        except ValueError: pass
        m = ds.mask('main'); A = ds.filter(m); assert np.array_equal(A.track, ds.track[m]) and np.array_equal(A.feat, ds.feat[m]) and len(ds) == 3000
        B = Dataset.merge([ds[:50], ds[50:80]]); assert np.array_equal(B.feat, ds.feat[:80]) and np.array_equal((ds[:50] + ds[50:80]).track, ds.track[:80])
        E = ds.cropFromEdge(10); assert len(E) and E.x.min() >= 10 and E.x.max() < 109 and E.y.max() < 109
        assert len(ds.crlb_filter(0.1, min_photons=500)) == ((np.maximum(T['sx'], T['sy']) < 0.1) & (T['photons'] > 500)).sum()
        D = np.cumsum(r.normal(0, 0.002, (ds.numFrames, 2)), 0).astype(np.float32)
        C = ds[:100]; C.applyDrift(D); assert C.isDriftCorrected and np.array_equal(C.x, T['x'][:100] - D[T['frame'][:100] + 3, 0]); C.undoDrift(); assert not C.isDriftCorrected
        cols = set(ds.columns)
        # npz: lossless, every field with its dtype
        N = Dataset.load(ds.save(f'{d}/f.npz')); assert set(N.columns) == cols; _same(ds, N); assert N.meta['pass'] == 2
        _same(ds, Dataset.loadNumpy(ds.saveNumpy(f'{d}/g.npz')))
        # hdf5 (Picasso): extra columns, float32 positions
        H = Dataset.loadHDF5(ds.saveHDF5(f'{d}/f.hdf5')); assert set(H.columns) == cols | {'psf_sx', 'psf_sy'} and H.feat.shape == (3000, 3)
        for k in cols - {'x', 'y', 'frame'}: assert H[k].dtype == ds[k].dtype and np.array_equal(H[k], ds[k], equal_nan=ds[k].dtype.kind == 'f'), k
        assert np.allclose(H.x, ds.x, atol=1e-5, rtol=0)
        # npy (the preprint code's layout): the structured array that code reads, extra fields as top-level fields
        f = ds.save(f'{d}/f.npy')
        with open(f, 'rb') as s_:
            shp = np.load(s_); A = np.load(s_)
        assert list(shp) == [120, 120] and A.dtype.names[:7] == ('roi_id', 'frame', 'estim', 'crlb', 'chisq', 'group', 'roipos')
        assert A['estim'].dtype.names == ('pos', 'photons', 'bg', 'ibg') and A['estim']['ibg'].shape == (3000, 6, 2) and np.array_equal(A['crlb']['pos'][:, 0], T['sx'])
        V = Dataset.loadNumpy(f, window=6); assert cols <= set(V.columns) and V.imgshape == [120, 120]
        for k in cols - {'x', 'y', 'frame'}: assert V[k].dtype == ds[k].dtype and np.array_equal(V[k], ds[k], equal_nan=ds[k].dtype.kind == 'f'), k
        assert np.array_equal(V.x, ds.x.astype(np.float32)) and np.array_equal(V.frame, ds.frame) and np.array_equal(V.mask('main', sigma_max=12.0), m)
        # csv (ThunderSTORM)
        S = Dataset.loadCSVThunderStorm(ds[:200].saveCSVThunderStorm(f'{d}/f.csv')); assert cols <= set(S.columns) and np.array_equal(S.track, ds.track[:200])
        assert np.allclose(S.feat, ds.feat[:200], rtol=1e-6, atol=1e-6) and np.array_equal(S.flag > 0.5, ds.flag[:200]) and np.allclose(S.bg, ds.bg[:200], rtol=1e-6)
        F = Dataset.fromEstimates(ds.x, ds.y, ds.frame, track=ds.track); assert np.array_equal(F.track, ds.track)


def test_drift_apply_undo():
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        L = Dataset.from_run(d); F = L.nframes
        D = np.cumsum(np.random.default_rng(2).normal(0, 0.002, (F, 2)), 0).astype(np.float32)
        assert not L.is_drift_corrected
        try: L.apply_drift(); raise AssertionError('apply without a drift must raise')
        except ValueError: pass
        x0 = L.x; L.set_drift(D); S = L.select('main'); m = L.mask('main')            # the drift goes with the selection
        S.apply_drift()
        fi = np.clip(T['frame'][m] + 3, 0, F - 1)
        assert np.array_equal(S.x, T['x'][m] - D[fi, 0]) and np.array_equal(S.y, T['y'][m] - D[fi, 1]) and S.is_drift_corrected     # x - drift[frame + 3]
        assert L.x is x0 and not L.is_drift_corrected and np.array_equal(L.x, T['x'])                                               # the parent is untouched
        assert len(S[::2]) and S[::2].drift_applied and np.array_equal(S[::2].drift, D)
        for bad in (S.apply_drift, S.estimate_drift, lambda: S.set_drift(D), L.undo_drift):
            try: bad(); raise AssertionError('must raise')
            except ValueError: pass
        S.undo_drift()
        assert not S.is_drift_corrected and np.allclose(S.x, T['x'][m], atol=1e-10, rtol=0) and np.allclose(S.y, T['y'][m], atol=1e-10, rtol=0)
        L.apply_drift(); assert np.array_equal(L.x, T['x'] - D[np.clip(T['frame'] + 3, 0, F - 1), 0])
        try: Dataset.concatenate([L, S]); raise AssertionError('mixed drift states must raise')
        except ValueError: pass


def test_drift_estimate():
    """estimate_drift = simcode.util.drift.drift_of_table on the same table (DME v2; needs the dme package and a GPU)"""
    try:
        import torch
        from simcode.util import drift as DR
        DR.dme_package()
        if not torch.cuda.is_available(): raise RuntimeError('no GPU')
    except Exception as e:
        return _skip(f'DME v2 not available ({e})')
    r = np.random.default_rng(5); n, F = 40000, 3000
    site = r.uniform(10, 110, (300, 2)); k = r.integers(0, 300, n); f = np.sort(r.integers(0, F // 6, n)) * 6
    true = np.c_[np.linspace(0, 0.4, F), 0.2 * np.sin(np.arange(F) / F * 6)]
    x = site[k, 0] + r.normal(0, 0.05, n) + true[f + 3, 0]; y = site[k, 1] + r.normal(0, 0.05, n) + true[f + 3, 1]
    L = Dataset.from_arrays(x, y, f, np.full(n, 0.05, np.float32), np.full(n, 0.05, np.float32), photons=np.full(n, 1000.0))
    d = L.estimate_drift(nframes=F)
    ref = DR.drift_of_table(L.table(), F)
    assert d.shape == (F, 2) and np.allclose(d, ref, atol=1e-4, rtol=0) and L.drift is d      # (DME on the GPU is not bit-reproducible between calls) and L.meta['drift_info']['n'] == n
    e = (d - d.mean(0)) - (true - true.mean(0)); assert np.abs(e).max() < 0.05, np.abs(e).max()
    d2 = L.estimate_drift(nframes=F, rows=np.ones(n, bool)); assert np.allclose(d2, ref, atol=1e-3)
    L.apply_drift(); assert np.std(L.x - site[k, 0] - (L.x - site[k, 0]).mean()) < 0.06 and L.info()['drift'] == 'applied'


def test_render():
    x = np.array([10.0, 10.2, 10.26, 12.0, 30.0]); y = np.array([5.0, 5.0, 5.0, 7.5, 30.0])
    L = Dataset.from_arrays(x, y, np.zeros(5, int), px_nm=100.0)
    img, ext = L.render(pixel_nm=25.0, region=(10, 5, 13, 8), return_extent=True)                 # 4 bins per camera pixel
    assert img.shape == (12, 12) and img.dtype == np.float32 and img.sum() == 4 and ext == (10, 13, 5, 8)
    assert img[0, 0] == 2 and img[0, 1] == 1 and img[10, 8] == 1                                     # row = y, column = x
    assert L.render(pixel_nm=25.0, region=(10, 5, 13, 8), weights=np.arange(5.0)).sum() == 0 + 1 + 2 + 3
    b = L.render(pixel_nm=25.0, region=(10, 5, 13, 8), sigma_nm=30.0); assert abs(b.sum() - 4) < 0.2 and b.max() < 2
    assert L.region() == (10, 5, 30, 30) and L.render(pixel_nm=50.0).shape == (50, 40)
    L.meta['imgshape'] = [40, 32]; assert L.region() == (-0.5, -0.5, 31.5, 39.5) and L.render(pixel_nm=100.0).shape == (40, 32)
    assert L.render(pixel_nm=100.0)[5, 10] == 3 and L.render(pixel_nm=100.0).sum() == 5            # integer = pixel centre


def test_frc_equals_reference():
    """Dataset.frc = simcode.evalproto.frc.frc_halves = analysis/paper/frc_fair.frc_halves on synthetic rows"""
    sys.path.insert(0, os.path.join(REPO, 'analysis', 'paper'))
    import frc_fair as FF
    from simcode.evalproto.frc import frc_halves
    r = np.random.default_rng(7); n = 60000
    site = r.uniform(10, 40, (4000, 2)); k = r.integers(0, 4000, n); f = r.integers(0, 2000, n) * 3
    L = Dataset.from_arrays(site[k, 0] + r.normal(0, 0.08, n), site[k, 1] + r.normal(0, 0.08, n), f)
    v = L.frc(); res, freq, curve = L.frc(return_curve=True)
    assert np.isfinite(v) and v == frc_halves(L.x, L.y, L.frame) == res and v == FF.frc_halves(L.x, L.y, L.frame)[0]
    assert np.array_equal(curve, FF.frc_halves(L.x, L.y, L.frame)[2]) and L.frc(blk=30) == FF.frc_halves(L.x, L.y, L.frame, blk=30)[0]


def test_real_table():
    """the production pass-2 table of the 1809 movie (read only): counts of its info json, and the FRC of a subset through
    the class against analysis/paper/frc_fair.py (same rows, same drift)"""
    if not os.path.exists(REAL): return _skip('no ' + REAL)
    sys.path.insert(0, os.path.join(REPO, 'analysis', 'paper'))
    import frc_fair as FF
    info = json.load(open(REAL[:-4] + '_info.json'))
    L = Dataset.load(os.path.dirname(REAL), columns=('sx', 'sy', 'photons', 'p', 'good', 'sel', 'disjoint', 'strict', 'strict_u'))
    assert len(L) == info['n_rows'] and L.px_nm == info['px_nm'] and L.meta['info']['q'] == info['q'] and L.meta['crop'] == [5.0, 5.0, 110.0] and L.meta['pass'] == 2
    assert int(L.sel.sum()) == info['n_sel'] and int(L.good.sum()) == info['n_good'] and int(L.strict.sum()) == info['n_strict']
    z = np.load(REAL)
    T = dict(strict_u=z['strict_u'], sx=z['sx'], sy=z['sy'], px_nm=z['px_nm'], x=z['x'])
    for rule, key in (('main', 'n_main'), ('u', 'n_u')):
        m = L.mask(rule); assert np.array_equal(m, SP.select(T, rule)) and int(m.sum()) == info[key], rule
    I = L.info(); assert I['n_main'] == info['n_main'] and I['window'] == 6 and 'Dataset: 6328751 rows' in repr(L)
    # frc_fair on the first 12000 frames: its table loader, masks, and FRC; a fixed synthetic drift instead of DME (no GPU)
    F = 12000
    Tf = FF.load('v2', REAL); keep = Tf['f'] + 6 <= F; Tf = {k: v[keep] for k, v in Tf.items()}
    S = L.select('sel').filter_frames(0, F)
    assert len(S) == len(Tf['x']) and np.array_equal(S.x, Tf['x']) and np.array_equal(S.frame, Tf['f']) and np.array_equal(S.sx.astype(np.float64), Tf['sx'])
    drift = np.c_[np.linspace(0, 0.3, F), np.linspace(0, -0.2, F)].astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        np.save(f'{d}/drift.npy', drift)
        R = FF.evaluate(Tf, F, drift_cache=f'{d}/drift.npy', secondary=False)
    x0, y0, sz = FF.CROP
    A = S[S.disjoint].filter_finite().filter_region(x0, y0, x0 + sz, y0 + sz).crlb_filter(FF.SIGMA_MAX)       # the protocol's rows
    assert len(A) == R['n_frc'] and np.array_equal(A.x, Tf['x'][R['mask']])
    A.apply_drift(drift)
    assert np.array_equal(A.x, R['x_dc'][R['mask']]) and np.array_equal(A.y, R['y_dc'][R['mask']])
    v = A.frc(); assert np.isfinite(v) and v == R['frc_nm'], (v, R['frc_nm'])
    print(f'real table: {len(L)} rows, subset {len(A)} rows, FRC {v:.3f} nm (class) = {R["frc_nm"]:.3f} nm (frc_fair)')


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('ok', k)
