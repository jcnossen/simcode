"""
3D position prior of the emitter network (EmitterNet, pass 1 and pass 2, 2026-11-14): layout, 2D checkpoints, the 3D NLL and the z conditioning
of the solve. CPU / GPU, no model files needed except where marked. Run: python tests/test_z3d.py (or pytest tests)
  - a 2D state dict loads with zero z rows (pad_out) and the 2D outputs do not depend on the z rows; a 3D one without the
    joint-covariance rows loads with C = 0 and its outputs do not depend on the C rows
  - nll_p (x, y) + nll_z (z | x, y) = the full 3D Gaussian NLL of prior_cov (up to the constant 1.5 log 2 pi)
  - z_given_xy = generic Gaussian conditioning of the joint (x, y, z) prior on a measurement of x, y
  - z_target_px: (z_um - focus_um) / pixelsize_um
  - link_pass1: one-to-one pass-2 row <-> pass-1 candidate (same window, detector points within 1 px, iterated mutual NN)
  - pass1_z_columns: the pass-2 z = the linked pass-1 prior (prior_off removed) conditioned on the pass-2 solved x, y; NaN
    without a partner; build_table (est2) -> Dataset: NaN z rows load, select and render; earlier pass-2 files (no p1_row)
    give no z columns
"""
import math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _head(p2=False):
    from simcode.emitter.model import EmitterNet
    torch.manual_seed(0)
    h = EmitterNet(6, 32, P=21, bf16=False, illum=p2).to(DEV)
    for prm in h.parameters():      # non-trivial weights everywhere (the zero-initialised layers included)
        prm.data.normal_(0, 0.05)
    return h.eval()


def _inputs(n=7, p2=False):
    g = torch.Generator(device='cpu').manual_seed(1)
    pt = (20 + 5 * torch.rand(n, 6, 21, 21, generator=g)).to(DEV)
    off = (torch.rand(n, 2, generator=g) - 0.5).to(DEV)
    nb = torch.rand(n, 2, 21, 21, generator=g).to(DEV)
    il = torch.rand(n, 12, 21, 21, generator=g).to(DEV) if p2 else None
    return pt, off, nb, il


def test_pad_out_and_layout():
    from simcode.emitter.model import pad_out, is_3d, is_joint, EmitterNet
    for p2 in (False, True):
        h = _head(p2); pt, off, nb, il = _inputs(p2=p2)
        sd = {k: v.clone() for k, v in h.state_dict().items()}
        assert is_3d(sd)
        nzc = EmitterNet.NZ + EmitterNet.NC
        sd2 = dict(sd); sd2['out.weight'] = sd['out.weight'][:-nzc]; sd2['out.bias'] = sd['out.bias'][:-nzc]       # 2D layout
        assert not is_3d(sd2)
        sd3 = dict(sd); sd3['out.weight'] = sd['out.weight'][:-EmitterNet.NC]; sd3['out.bias'] = sd['out.bias'][:-EmitterNet.NC]   # 3D, no C
        assert is_3d(sd3) and not is_joint(sd3) and is_joint(sd)
        with torch.no_grad():
            a = h(pt, None, off, None, nb, il)
            h.load_state_dict(pad_out(sd2, h))
            b = h(pt, None, off, None, nb, il)
            h.load_state_dict(pad_out(sd3, h))
            c = h(pt, None, off, None, nb, il)
        for k in ('I', 'L', 'dxy', 'lsxy', 'bg', 'I_lin'):
            assert torch.equal(a[k], b[k]), k          # the 2D outputs do not see the z / C rows
        for k in ('I', 'L', 'dxy', 'lsxy', 'bg', 'I_lin', 'z', 'lsz', 'bz'):
            assert torch.equal(a[k], c[k]), k          # the 3D outputs do not see the C rows
        assert a['bg'].shape == (len(pt), 6)
        assert torch.all(b['z'] == 0) and torch.all(b['bz'] == 0) and torch.allclose(b['lsz'], torch.full_like(b['lsz'], -1.25))
        assert torch.all(b['C'] == 0) and torch.all(c['C'] == 0) and c['C'].shape == (len(pt), 6, 2)


def test_nll_3d():
    from simcode.emitter.model import nll_z, prior_cov
    g = torch.Generator().manual_seed(2); n = 50
    out = dict(x=torch.randn(n, generator=g), y=torch.randn(n, generator=g), z=torch.randn(n, generator=g),
               lsxy=0.3 * torch.randn(n, 2, generator=g) - 1, lsz=0.3 * torch.randn(n, generator=g) - 0.5, bz=torch.randn(n, 2, generator=g))
    t = torch.randn(n, 3, generator=g)
    ls = out['lsxy']; ex = t[:, 0] - out['x']; ey = t[:, 1] - out['y']
    nll_p = 0.5 * (ex / ls[:, 0].exp()) ** 2 + ls[:, 0] + 0.5 * (ey / ls[:, 1].exp()) ** 2 + ls[:, 1]
    C = prior_cov(out).double()
    mu = torch.stack([out['x'], out['y'], out['z']], 1).double()
    ref = -torch.distributions.MultivariateNormal(mu, covariance_matrix=C).log_prob(t.double()) - 1.5 * math.log(2 * math.pi)
    got = (nll_p + nll_z(out, t[:, 0], t[:, 1], t[:, 2])).double()
    assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())


def test_z_given_xy():
    from simcode.emitter.model import prior_cov
    from simcode.solve.z_conditioning import z_given_xy
    g = torch.Generator().manual_seed(3); n = 40
    o = dict(x=torch.randn(n, generator=g), y=torch.randn(n, generator=g), z=torch.randn(n, generator=g),
             lsxy=0.3 * torch.randn(n, 2, generator=g) - 1, lsz=0.3 * torch.randn(n, generator=g) - 0.5, bz=torch.randn(n, 2, generator=g))
    r0 = torch.stack([o['x'], o['y']], 1)
    # measurement of x, y with sd m (the pattern); generic Kalman update of the joint prior
    C = prior_cov(o).double(); mu = torch.cat([r0, o['z'][:, None]], 1).double()
    meas = r0 + torch.randn(n, 2, generator=g) * 0.3; M = torch.diag_embed((0.1 + 0.2 * torch.rand(n, 2, generator=g)) ** 2).double()
    H = torch.tensor([[1., 0, 0], [0, 1., 0]], dtype=torch.float64)
    S = H @ C @ H.T + M; K = C @ H.T @ torch.linalg.inv(S)
    mpost = mu + (K @ (meas.double() - mu[:, :2])[..., None])[..., 0]
    Cpost = C - K @ H @ C
    # the solve's x, y posterior: mean mpost[:, :2], sd from Cpost (uncorrelated here only if C_xy is diagonal: it is)
    assert torch.allclose(Cpost[:, 0, 1], torch.zeros(n, dtype=torch.float64), atol=1e-9)
    s = Cpost[:, [0, 1], [0, 1]].sqrt()
    z, sz, cov = z_given_xy(o, mpost[:, :2].float(), r0, s.float())
    assert torch.allclose(z.double(), mpost[:, 2], atol=1e-4)
    assert torch.allclose(sz.double() ** 2, Cpost[:, 2, 2], rtol=1e-4, atol=1e-6)
    assert torch.allclose(cov[:, 1].double(), Cpost[:, 0, 2], rtol=1e-4, atol=1e-6)
    assert torch.allclose(cov[:, 2].double(), Cpost[:, 1, 2], rtol=1e-4, atol=1e-6)


def test_z_target():
    from simcode.simulate.batches import z_target_px
    class S: focus = 0.15; c = {'pixelsize_um': 0.1}
    assert abs(float(z_target_px(torch.tensor(0.05), S)) + 1.0) < 1e-6


def test_link_pass1():
    import numpy as np
    from simcode.solve.z_conditioning import link_pass1
    rng = np.random.default_rng(5); n = 500
    k1 = rng.integers(0, 20, n) * 3; x1 = rng.uniform(0, 100, n); y1 = rng.uniform(0, 100, n)
    take = rng.permutation(n)[:400]                                     # pass 2 = 400 of them, jittered, shuffled + 30 extra
    k2 = np.r_[k1[take], rng.integers(0, 20, 30) * 3]
    x2 = np.r_[x1[take] + rng.normal(0, 0.05, 400), rng.uniform(200, 300, 30)]; y2 = np.r_[y1[take] + rng.normal(0, 0.05, 400), rng.uniform(0, 100, 30)]
    j = link_pass1(k1, x1, y1, k2, x2, y2)
    assert (j[400:] == -1).all()
    # every jittered row finds its own candidate unless another candidate of the window is nearer (dense random points)
    own = j[:400] == take
    assert own.mean() > 0.97, own.mean()
    assert len(np.unique(j[j >= 0])) == (j >= 0).sum(), 'not one-to-one'
    # chain: pass-2 A (0.0), B (0.6); pass-1 a (0.35), b (-0.5): round 1 links B-a (mutual), round 2 A-b
    j = link_pass1([0, 0], [0.35, -0.5], [0, 0], [0, 0], [0.0, 0.6], [0, 0])
    assert list(j) == [1, 0], j
    # other window / beyond 1 px: no partner
    assert list(link_pass1([0], [0.0], [0.0], [3, 0], [0.0, 1.0], [0.0, 0.0])) == [-1, -1]


def test_pass1_z_columns():
    import numpy as np
    from simcode.solve.z_conditioning import pass1_z_columns, z_given_xy
    rng = np.random.default_rng(2); n1 = 50
    P1 = dict(wstart=np.repeat(np.arange(5) * 3, 10).astype(np.int64), x_det=rng.uniform(10, 60, n1), y_det=rng.uniform(10, 60, n1))
    P1.update(x=P1['x_det'] + rng.normal(0, 0.1, n1), y=P1['y_det'] + rng.normal(0, 0.1, n1), z=rng.normal(-0.5, 0.3, n1),
              sz=rng.uniform(0.1, 0.4, n1), bzx=rng.normal(0, 0.5, n1), bzy=rng.normal(0, 0.5, n1))
    P1 = {k: np.asarray(v, np.float32) if k != 'wstart' else v for k, v in P1.items()}
    sel = np.r_[3, 17, 25, 49]
    ws = np.r_[P1['wstart'][sel], 99]; xd = np.r_[P1['x_det'][sel] + 0.1, 30.0]; yd = np.r_[P1['y_det'][sel] - 0.1, 30.0]
    r = torch.tensor(np.c_[xd + 0.05, yd - 0.02], dtype=torch.float32); s = torch.full((5, 2), 0.1)
    cal1 = dict(prior_off=[0.02, -0.01])
    Z = pass1_z_columns(P1, cal1, ws, xd, yd, r, s)
    assert list(Z['p1_row'].numpy()) == list(sel) + [-1]
    o = dict(z=torch.tensor(P1['z'][sel]), lsz=torch.tensor(np.log(P1['sz'][sel])), bz=torch.tensor(np.c_[P1['bzx'][sel], P1['bzy'][sel]]))
    r0 = torch.tensor(np.c_[P1['x'][sel] - 0.02, P1['y'][sel] + 0.01])
    z, sz, cov = z_given_xy(o, r[:4], r0, s[:4])
    assert torch.allclose(Z['z_hy0'][:4], z, atol=1e-5) and torch.allclose(Z['sz_cal'][:4], sz, atol=1e-6)
    assert torch.allclose(Z['cov_xz'][:4], cov[:, 1]) and torch.allclose(Z['cov_yz'][:4], cov[:, 2])
    assert torch.allclose(Z['z_prior'][:4], o['z'])
    assert all(torch.isnan(Z[k][4]) for k in ('z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior'))


def _pass2_files(d, n, rng, p1_row=True):
    import numpy as np
    R = dict(wstart=(rng.integers(0, 100, n) * 3).astype(np.int64), x_hy0=rng.uniform(10, 100, n).astype(np.float32),
             y_hy0=rng.uniform(10, 100, n).astype(np.float32), sx_cal=rng.uniform(0.02, 0.2, n).astype(np.float32),
             sy_cal=rng.uniform(0.02, 0.2, n).astype(np.float32), phot=rng.uniform(100, 3000, n).astype(np.float32),
             p=rng.uniform(0.3, 1, n).astype(np.float32), q_dof=np.ones(n, np.float32), kin_hy=np.zeros(n, np.float32),
             I_net=rng.uniform(10, 500, (n, 6)).astype(np.float32), z_hy0=rng.normal(-0.5, 0.3, n).astype(np.float32),
             sz_cal=np.full(n, 0.3, np.float32), cov_xz=np.full(n, 1e-3, np.float32), cov_yz=np.full(n, -2e-3, np.float32),
             z_prior=rng.normal(-0.5, 0.3, n).astype(np.float32))
    R['frame'] = R['wstart'] + 2.5; R['x_prior'] = R['x_hy0']; R['y_prior'] = R['y_hy0']
    if p1_row:
        R['p1_row'] = np.arange(n); R['p1_row'][::10] = -1
        for k in ('z_hy0', 'sz_cal', 'cov_xz', 'cov_yz', 'z_prior'): R[k][::10] = np.nan
    R.update(x_net=R['x_hy0'], y_net=R['y_hy0'], sx_net=R['sx_cal'], sy_net=R['sy_cal'], z_net=R['z_hy0'], sz_net=R['sz_cal'],
             cov_xz_net=R['cov_xz'], cov_yz_net=R['cov_yz'])
    np.savez(os.path.join(d, 'pass2_est2_0.npz'), **R)
    np.savez(os.path.join(d, 'pass1_0.npz'), out=np.ones(n, bool), crop=np.array([0, 0, 128]))
    return R


def test_table_pass2_z(tmp_path=None):
    """build_table (est2): the pass-1 z columns, NaN rows without a partner -> Dataset pos [n, 3] (NaN z), selection and render
    fine; pass-2 files of a run before 2026-11-14 (no p1_row: the pass-2 head z) are refused"""
    import tempfile, numpy as np
    from simcode import table as SP
    from simcode.dataset import Dataset
    rng = np.random.default_rng(4); n = 300
    d = str(tmp_path or tempfile.mkdtemp())
    R = _pass2_files(d, n, rng)
    T, info = SP.build_table(d, 'est2', crop=(0, 0, 128))
    assert np.isnan(T['z'][::10]).all() and np.isfinite(np.delete(T['z'], np.s_[::10])).all()
    assert info['n_no_p1'] == n // 10 and 'p1_row' not in T and 'linked pass-1' in info['z']
    assert np.array_equal(T['p1_index'], R['p1_row'])                                  # all pass-1 rows stored: the same index
    ds = Dataset.load_table(SP.write_table(T, info, os.path.join(d, 'locs_p2_net')), info=False)
    assert ds['pos'].shape == (n, 3) and torch.isnan(ds['pos'][::10, 2]).all()
    img = ds.select(12, 0.63, False).render(pixel_nm=20)
    assert np.isfinite(np.asarray(img)).all()
    d2 = tempfile.mkdtemp(); _pass2_files(d2, n, rng, p1_row=False)
    try: SP.build_table(d2, 'est2', crop=(0, 0, 128)); raise AssertionError('pass-2 files without p1_row must be refused')
    except ValueError: pass


def test_table_3d(tmp_path=None):
    """a v2 table with the z columns loads as a 3D Dataset (pos / crlb [n, 3], cov (xy, xz, yz)); the selection uses x, y only;
    hdf5 round trip; a table without them (2D heads) stays 2D"""
    import tempfile, numpy as np
    from simcode.dataset import Dataset
    d = tmp_path or tempfile.mkdtemp()
    rng = np.random.default_rng(0); n = 200
    T = dict(frame=(rng.integers(0, 100, n) * 3).astype(np.int64), x=rng.uniform(10, 100, n), y=rng.uniform(10, 100, n),
             sx=rng.uniform(0.02, 0.2, n).astype(np.float32), sy=rng.uniform(0.02, 0.2, n).astype(np.float32),
             photons=rng.uniform(100, 3000, n).astype(np.float32), p=rng.uniform(0, 1, n).astype(np.float32))
    np.savez(os.path.join(str(d), 't2.npz'), **T, px_nm=np.array(108.33))
    T3 = dict(T, z=rng.normal(-0.5, 0.3, n), sz=np.full(n, 0.3, np.float32), cov_xz=np.full(n, 0.001, np.float32),
              cov_yz=np.full(n, -0.002, np.float32), z_prior=rng.normal(-0.5, 0.3, n))
    np.savez(os.path.join(str(d), 't3.npz'), **T3, px_nm=np.array(108.33))
    a = Dataset.load_npz(os.path.join(str(d), 't2.npz'), info=False); b = Dataset.load_npz(os.path.join(str(d), 't3.npz'), info=False)
    assert a['pos'].shape == (n, 2) and b['pos'].shape == (n, 3) and b['crlb'].shape == (n, 3) and b['cov'].shape == (n, 3)
    assert np.allclose(b['pos'][:, 2].numpy(), T3['z']) and np.allclose(b['cov'][:, 1].numpy(), 0.001)
    assert torch.equal(a.mask(12, 0.63, False), b.mask(12, 0.63, False))
    f = os.path.join(str(d), 't3.hdf5'); b.save(f); c = Dataset.load(f)
    assert torch.allclose(c['pos'], b['pos']) and torch.allclose(c['cov'], b['cov'])
    img = b.select(12, 0.63, False).render(pixel_nm=20)
    assert np.isfinite(np.asarray(img)).all()


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print(k, 'ok')
