"""
CPU tests of simcode.dataset.Dataset, SIMCODE's layer on torch_psf.Dataset (the generic part: indexing, filters, crop, drift,
render, FRC, files, is tested in torch-psf). No models, no raw data; the tests on files of the paper runs and the DME test are
skipped when the files / the dme package / a GPU are not available.
Run: python tests/test_dataset.py (or pytest tests/test_dataset.py).
  v2 table   built from a fake run directory with simcode.table: field layout (pos, crlb, frame, n_frames = 6, score,
             intensities, the other columns), metadata, from_run(pass_) of the HDF5 tables and of the npz tables of runs
             before 2026-11-15 (columns=); select / mask equal table.select; columns(); the subclass survives indexing and
             the filters
  hdf5       save -> plain torch_psf.Dataset.load: the same fields, values, dtypes and metadata; select after the round trip
  old files  the preprint code's .npy, the paper's DECODE npz,
             Picasso (the paper's sf.hdf5)
  drift      apply at the window centre (frame + 3); estimate (DME v2) close to simcode.util.drift.drift_of_table
  frc        Dataset.frc = simcode.evaluation.frc.frc_halves
  real       out/paper2/1809_2/v2_matched/locs_p2.npz (a run before 2026-11-15): counts of the info json; FRC of a subset =
             analysis/paper/frc_fair.py
"""
import json, os, sys, tempfile
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from simcode import table as SP
from simcode.dataset import Dataset
from simcode.psf import Dataset as TorchPSFDataset
from tests.test_tables import _run_dir

REAL = os.path.join(REPO, 'out/paper2/1809_2/v2_matched/locs_p2.npz')
SF = os.path.join(REPO, 'out/paper2/1809_2/sf/sf.hdf5')
RENAMED = dict(x=None, y=None, sx=None, sy=None, frame=None, p='score', I='intensities')


def _skip(msg):
    try:
        import pytest
    except ImportError:
        print('skipped:', msg); return
    pytest.skip(msg)


def _table(d, n=3000):
    _run_dir(d, n=n)
    T, info = SP.build_table(d, 'est2')
    SP.write_table(T, info, f'{d}/locs_p2_net'); json.dump(info, open(f'{d}/locs_p2_net_info.json', 'w'), default=str)
    return T, info


def _check_layout(L, T):
    """the v2 table T as fields of L"""
    F = L.numpy()
    assert np.array_equal(F['pos'], np.stack([T['x'], T['y']], 1)) and F['pos'].dtype == np.float64
    assert np.array_equal(F['crlb'], np.stack([T['sx'], T['sy']], 1)) and F['crlb'].dtype == np.float32
    assert np.array_equal(F['frame'], T['frame']) and F['frame'].dtype == np.int32 and (F['n_frames'] == 6).all()
    for k, v in T.items():
        if np.ndim(v) == 0 or k in ('x', 'y', 'sx', 'sy', 'frame'): continue
        f = RENAMED.get(k) or k
        assert np.array_equal(F[f], v, equal_nan=v.dtype.kind == 'f'), k


def test_v2_table_select():
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        L = Dataset.load(d)                                                     # run directory -> locs_p2_net.hdf5
        assert type(L) is Dataset and len(L) == 3000 and L.n_dims == 2 and L.pixel_nm == SP.PX_NM_DEFAULT
        _check_layout(L, T)
        assert L.meta['crop'] == [5.0, 5.0, 110.0] and L.meta['pass'] == 2 and L.meta['format'] == 'simcode hdf5'
        assert L.meta['info']['n_main'] == info['n_main'] and (L.center_frame == L.frame + 3).all()
        Tz = L.columns(); assert all(np.array_equal(Tz[k], T[k], equal_nan=T[k].dtype.kind == 'f') for k in T)
        for kw in ({}, dict(max_precision_nm=8.0), dict(min_score=0.8), dict(in_region=False), dict(max_precision_nm=None, min_score=None, in_region=False)):
            m = SP.select(Tz, **kw)
            assert np.array_equal(L.mask(**kw).numpy(), m), kw
            S = L.select(**kw)
            assert type(S) is Dataset and len(S) == m.sum() and np.array_equal(S.pos[:, 0].numpy(), T['x'][m])
            assert np.array_equal(S.intensities.numpy(), T['I'][m]) and S.meta['selection']['max_precision_nm'] == kw.get('max_precision_nm', 12.0)
        assert int(L.mask().sum()) == info['n_main']
        try: L.select('main'); raise AssertionError('a rule name must raise')
        except TypeError: pass
        # the subclass and its metadata survive the generic operations (select works on the result)
        for A in (L[L.good], L[10:20], L.filter_photons(500), L.crop(20, 30, 60, 70), L.filter_frames(300, 600), L + L[:5]):
            assert type(A) is Dataset and A.meta['crop'] == L.meta['crop'] and len(A.select(in_region=False)) <= len(A)
        assert len(L.filter_crlb(max_nm=10)) == (np.maximum(T['sx'], T['sy']).astype(np.float64) * SP.PX_NM_DEFAULT <= 10).sum()
        # from_run(pass_); Dataset.load of the table file
        SP.write_table(T, info, f'{d}/locs_p1_net')
        assert Dataset.from_run(d, pass_=1).meta['pass'] == 1 and len(Dataset.load(f'{d}/locs_p1_net.hdf5').select()) == info['n_main']
        try: Dataset.from_run(d, pass_=3); raise AssertionError('a missing pass must raise')
        except FileNotFoundError: pass
    with tempfile.TemporaryDirectory() as d:
        # the npz table of a run made before 2026-11-15 (as its table stage wrote it)
        _run_dir(d, n=3000); T, info = SP.build_table(d, 'est2')
        np.savez(f'{d}/locs_p2.npz', **T, px_nm=np.array(info['px_nm']), sigma_max=np.array(12.0))
        json.dump(info, open(f'{d}/locs_p2_info.json', 'w'), default=str)
        L = Dataset.from_run(d, table='locs_p2'); _check_layout(L, T)
        try: Dataset.from_run(d); raise AssertionError('an old run directory without locs_p2_net must raise')
        except FileNotFoundError as e: assert 'before 2026-11-17' in str(e)
        assert L.meta['format'] == 'simcode npz' and L.meta['crop'] == [5.0, 5.0, 110.0] and len(L.select()) == info['n_main']
        P = Dataset.load_npz(f'{d}/locs_p2.npz', columns=('sx', 'sy'))
        assert set(P.keys()) == {'pos', 'crlb', 'frame', 'n_frames', 'id', } and P.meta['sigma_max'] == 12.0


def test_hdf5_roundtrip():
    """SIMCODE table -> save('.hdf5') -> plain torch_psf.Dataset.load: every field and the metadata; select afterwards"""
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        L = Dataset.from_run(d); L.set_drift(np.cumsum(np.random.default_rng(1).normal(0, 0.01, (L.num_frames, 2)), 0))
        f = L.save(f'{d}/a.hdf5')
        M = TorchPSFDataset.load(f)
        assert type(M) is TorchPSFDataset and set(M.keys()) == set(L.keys())
        nan = lambda t: torch.nan_to_num(t, 7.0) if t.is_floating_point() else t
        for k in L.keys():
            assert M[k].dtype == L[k].dtype and torch.equal(nan(M[k]), nan(L[k])), k
        assert torch.equal(M.drift, L.drift) and not M.drift_applied
        assert {k: v for k, v in M.meta.items() if k != 'source'} == {k: v for k, v in L.meta.items() if k != 'source'}
        N = Dataset.load(f)
        assert type(N) is Dataset and torch.equal(N.mask(), L.mask()) and int(N.mask().sum()) == info['n_main']
        N.apply_drift(); assert torch.allclose(N.pos, L.pos - L.drift[L.center_frame.long()], atol=0, rtol=0)


def test_v1_decode_picasso_files():
    """the preprint code's numpy file (structured array, read without that package), the paper's DECODE npz, Picasso"""
    r = np.random.default_rng(3); n = 200
    est = np.dtype([('pos', '<f4', (2,)), ('photons', '<f4'), ('bg', '<f4'), ('ibg', '<f4', (6, 2))])
    dt = np.dtype([('roi_id', '<i4'), ('frame', '<i4'), ('estim', est), ('crlb', est), ('chisq', '<f4'), ('group', '<i4'), ('roipos', '<i4', (2,))])
    D = np.zeros(n, dt); D['frame'] = np.sort(r.integers(0, 900, n)); D['estim']['pos'] = r.uniform(5, 115, (n, 2)); D['crlb']['pos'] = r.uniform(0.02, 0.2, (n, 2))
    D['estim']['photons'] = r.uniform(300, 3000, n); D['estim']['bg'] = r.uniform(1, 9, n); D['estim']['ibg'] = r.uniform(0, 500, (n, 6, 2)); D['chisq'] = r.uniform(0, 2, n)
    with tempfile.TemporaryDirectory() as d:
        with open(f'{d}/smlm.npy', 'wb') as s:
            np.save(s, np.array([120, 120])); np.save(s, D)
        L = Dataset.load(f'{d}/smlm.npy', window=6); F = L.numpy()
        assert np.array_equal(F['pos'], D['estim']['pos']) and np.array_equal(F['crlb'], D['crlb']['pos']) and (F['n_frames'] == 6).all()
        assert np.array_equal(F['photons'], D['estim']['photons']) and np.array_equal(F['bg'], D['estim']['bg']) and np.array_equal(F['chisq'], D['chisq'])
        assert np.array_equal(F['intensities'], D['estim']['ibg'][..., 0]) and np.array_equal(F['intensities_bg'], D['estim']['ibg'][..., 1])
        assert L.meta['image_size'] == [120, 120] and np.array_equal(Dataset.from_smlmtorch(D.view(np.recarray)).pos.numpy(), F['pos'])
        np.savez(f'{d}/dec.npz', frame=D['frame'].astype(np.int64), x=F['pos'][:, 0].astype(np.float32), y=F['pos'][:, 1].astype(np.float32),
                 z=np.full(n, 80.0, np.float32), sx=F['crlb'][:, 0], sy=F['crlb'][:, 1], phot=F['photons'], sphot=F['photons'] * 0.1, bg=F['bg'],
                 prob=r.uniform(0.5, 1, n).astype(np.float32), info=np.array('decode test'))
        for E in (Dataset.load(f'{d}/dec.npz'), Dataset.from_decode(f'{d}/dec.npz')):
            assert torch.equal(E.photons, L.photons) and 'score' in E and 'crlb_photons' in E and (E.z == 80).all()
            assert E.n_dims == 2 and (E.n_frames == 6).all() and E.meta['info'] == 'decode test' and len(E.select(in_region=False)) <= n
        P = Dataset.load(L.save_picasso(f'{d}/p.hdf5'), window=6)                          # Picasso: torch_psf's reader
        assert torch.allclose(P.pos, L.pos.float().double()) and torch.equal(P.crlb, L.crlb) and (P.n_frames == 6).all() and P.pixel_nm == SP.PX_NM_DEFAULT


def test_picasso_paper_file():
    """the paper's SIMFLUX table (written by the preprint code)"""
    if not os.path.exists(SF): return _skip('no ' + SF)
    import h5py
    L = Dataset.load(SF, window=6); F = L.numpy()
    with h5py.File(SF, 'r') as h: A = h['locs'][:]
    assert np.array_equal(F['pos'][:, 0], A['x']) and np.array_equal(F['crlb'][:, 0], A['lpx']) and np.array_equal(F['crlb'][:, 1], A['lpy'])
    assert np.array_equal(F['psf_sx'], A['sx']) and np.array_equal(F['frame'], A['frame']) and np.array_equal(F['photons'], A['photons']) and 'chisq' in F
    assert L.meta['image_size'] == [120, 120] and L.num_frames == 119995 and (F['n_frames'] == 6).all()


def test_drift_apply():
    with tempfile.TemporaryDirectory() as d:
        T, info = _table(d)
        L = Dataset.from_run(d); F = L.num_frames
        D = np.cumsum(np.random.default_rng(2).normal(0, 0.002, (F, 2)), 0)
        L.set_drift(D); m = L.mask().numpy(); S = L.select().apply_drift()                 # the drift goes with the selection
        fi = np.clip(T['frame'][m] + 3, 0, F - 1)
        assert np.array_equal(S.pos.numpy(), np.stack([T['x'][m] - D[fi, 0], T['y'][m] - D[fi, 1]], 1)) and S.drift_applied   # x - drift[frame + 3]
        assert not L.drift_applied and np.array_equal(L.pos[:, 0].numpy(), T['x'])
        S.undo_drift(); assert np.allclose(S.pos[:, 0].numpy(), T['x'][m], atol=1e-10, rtol=0)


def test_drift_estimate():
    """estimate_drift (torch_psf, DME v2) on the rows of simcode.util.drift.drift_of_table gives that drift (needs the dme
    package and a GPU)"""
    try:
        from simcode.util import drift as DR
        DR.dme_package()
        if not torch.cuda.is_available(): raise RuntimeError('no GPU')
    except Exception as e:
        return _skip(f'DME v2 not available ({e})')
    r = np.random.default_rng(5); n, F = 40000, 3000
    site = r.uniform(10, 110, (300, 2)); k = r.integers(0, 300, n); f = np.sort(r.integers(0, F // 6, n)) * 6
    true = np.c_[np.linspace(0, 0.4, F), 0.2 * np.sin(np.arange(F) / F * 6)]
    x = site[k, 0] + r.normal(0, 0.05, n) + true[f + 3, 0]; y = site[k, 1] + r.normal(0, 0.05, n) + true[f + 3, 1]
    s = np.full(n, 0.05, np.float32)
    L = Dataset.from_arrays(np.c_[x, y], f, np.c_[s, s], n_frames=np.full(n, 6), meta=dict(pixel_nm=SP.PX_NM_DEFAULT))
    d = L.estimate_drift(n_frames=F).numpy()
    ref = DR.drift_of_table(dict(x=x, y=y, frame=f, sx=s, sy=s), F)
    assert d.shape == (F, 2) and np.allclose(d, ref, atol=1e-3, rtol=0)
    e = (d - d.mean(0)) - (true - true.mean(0)); assert np.abs(e).max() < 0.05, np.abs(e).max()


def test_frc_equals_reference():
    """Dataset.frc (torch_psf) = simcode.evaluation.frc.frc_halves (the paper's estimator) on synthetic rows"""
    from simcode.evaluation.frc import frc_halves
    r = np.random.default_rng(7); n = 60000
    site = r.uniform(10, 40, (4000, 2)); k = r.integers(0, 4000, n); f = r.integers(0, 2000, n) * 3
    x, y = site[k, 0] + r.normal(0, 0.08, n), site[k, 1] + r.normal(0, 0.08, n)
    L = Dataset.from_arrays(np.c_[x, y], f, meta=dict(pixel_nm=SP.PX_NM_DEFAULT))
    v, freq, curve = L.frc(return_curve=True); w, freq2, curve2 = frc_halves(x, y, f, return_curve=True)
    # torch-psf since 7d5ded7 (pin 4d806d0) computes the FRC with float32 histograms and a complex64 rfft2 (the paper estimator: float64): the
    # curves agree to ~2e-8 (here: the same resolution)
    assert np.isfinite(v) and v == w and np.allclose(curve, curve2, rtol=0, atol=1e-6) and L.frc(block=30) == frc_halves(x, y, f, blk=30)


def test_real_table():
    """the production pass-2 table of the 1809 movie (read only): counts of its info json, and the FRC of a subset through
    the class against analysis/paper/frc_fair.py (same rows, same drift)"""
    if not os.path.exists(REAL): return _skip('no ' + REAL)
    if not os.path.exists(os.path.join(REPO, 'analysis', 'paper', 'frc_fair.py')): return _skip('no analysis/paper/frc_fair.py (paper_figures branch)')
    sys.path.insert(0, os.path.join(REPO, 'analysis', 'paper'))
    import frc_fair as FF
    info = json.load(open(REAL[:-4] + '_info.json'))
    L = Dataset.from_run(os.path.dirname(REAL), table='locs_p2', columns=('sx', 'sy', 'photons', 'p', 'good', 'sel', 'disjoint', 'strict_u'))   # a run before 2026-11-17
    assert len(L) == info['n_rows'] and L.pixel_nm == info['px_nm'] and L.meta['crop'] == [5.0, 5.0, 110.0] and L.meta['pass'] == 2
    assert int(L.sel.sum()) == info['n_sel'] and int(L.good.sum()) == info['n_good']
    prec = L.crlb.max(1).values.double() * L.pixel_nm
    m = L.mask(); assert torch.equal(m, torch.isfinite(L.strict_u) & (prec <= 12.0)) and int(m.sum()) == info['n_main']
    F = 12000
    Tf = FF.load('v2', REAL); keep = Tf['f'] + 6 <= F; Tf = {k: v[keep] for k, v in Tf.items()}
    S = L[L.sel].filter_frames(0, F)
    assert len(S) == len(Tf['x']) and np.array_equal(S.pos[:, 0].numpy(), Tf['x']) and np.array_equal(S.frame.numpy(), Tf['f'])
    drift = np.c_[np.linspace(0, 0.3, F), np.linspace(0, -0.2, F)].astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        np.save(f'{d}/drift.npy', drift)
        R = FF.evaluate(Tf, F, drift_cache=f'{d}/drift.npy', secondary=False)
    x0, y0, sz = FF.CROP
    A = S[S.disjoint].filter_finite().crop(x0, y0, x0 + sz, y0 + sz)
    A = A[A.crlb.max(1).values < FF.SIGMA_MAX]                                                       # the protocol's rows
    assert len(A) == R['n_frc'] and np.array_equal(A.pos[:, 0].numpy(), Tf['x'][R['mask']])
    A.apply_drift(drift)
    assert np.array_equal(A.pos[:, 0].numpy(), R['x_dc'][R['mask']]) and np.array_equal(A.pos[:, 1].numpy(), R['y_dc'][R['mask']])
    v = A.frc(); assert np.isfinite(v) and v == R['frc_nm'], (v, R['frc_nm'])
    print(f'real table: {len(L)} rows, subset {len(A)} rows, FRC {v:.3f} nm (class) = {R["frc_nm"]:.3f} nm (frc_fair)')


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('ok', k)
