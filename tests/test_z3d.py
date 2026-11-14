"""
3D position prior of the head (SCHead4 / SCHeadP2, 2026-11-14): layout, 2D checkpoints, the 3D NLL and the z conditioning
of the solve. CPU / GPU, no model files needed except where marked. Run: python tests/test_z3d.py (or pytest tests)
  - a 2D state dict loads with zero z rows (pad_out) and the 2D outputs do not depend on the z rows
  - nll_p (x, y) + nll_z (z | x, y) = the full 3D Gaussian NLL of prior_cov (up to the constant 1.5 log 2 pi)
  - z_given_xy = generic Gaussian conditioning of the joint (x, y, z) prior on a measurement of x, y
  - z_target_px: (z_um - focus_um) / pixelsize_um
"""
import math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _head(p2=False):
    from simcode.hd.sc4 import SCHead4
    from simcode.p2.head2 import SCHeadP2
    torch.manual_seed(0)
    h = (SCHeadP2 if p2 else SCHead4)(6, 32, P=21, bf16=False).to(DEV)
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
    from simcode.hd.sc4 import pad_out, is_3d, SCHead4
    for p2 in (False, True):
        h = _head(p2); pt, off, nb, il = _inputs(p2=p2)
        sd = h.state_dict()
        assert is_3d(sd)
        sd2 = dict(sd); sd2['out.weight'] = sd['out.weight'][:-SCHead4.NZ]; sd2['out.bias'] = sd['out.bias'][:-SCHead4.NZ]
        assert not is_3d(sd2)
        with torch.no_grad():
            a = h(pt, None, off, None, nb, il)
            h.load_state_dict(pad_out(sd2, h))
            b = h(pt, None, off, None, nb, il)
        for k in ('I', 'L', 'dxy', 'lsxy', 'bg', 'I_lin'):
            assert torch.equal(a[k], b[k]), k          # the 2D outputs do not see the z rows
        assert a['bg'].shape == (len(pt), 6)
        assert torch.all(b['z'] == 0) and torch.all(b['bz'] == 0) and torch.allclose(b['lsz'], torch.full_like(b['lsz'], -1.25))


def test_nll_3d():
    from simcode.hd.sc4 import nll_z, prior_cov
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
    from simcode.hd.sc4 import prior_cov
    from simcode.hd.sc import z_given_xy
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
    from simcode.hd.data import z_target_px
    class S: focus = 0.15; c = {'pixelsize_um': 0.1}
    assert abs(float(z_target_px(torch.tensor(0.05), S)) + 1.0) < 1e-6


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print(k, 'ok')
