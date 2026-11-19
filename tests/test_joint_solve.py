"""
Joint prior / intensity covariance in the pattern solve (2026-11-17). CPU, no model files. Run: python tests/test_joint_solve.py
(or pytest tests)
  - the solve's cost (fast _eval, c = 0 and c > 0) equals the explicit joint Gaussian cost e_J^T Sigma_J^-1 e_J (+ log det
    of the nuisance part) with Sigma_J built from L_J = [[diag(sp), 0], [C, L]] and the nuisance term added to its intensity block
  - nll_joint = the explicit 8 x 8 Gaussian NLL minus the prior part
  - reference and fast solve agree with a nonzero C
  - one brightness per pattern orientation (cal B_per_angle, solve two_b, 2026-11-19): reference = fast; with equal brightness of
    the two orientations it gives the one-brightness solution up to the noise; with a brightness ratio between the orientations it
    recovers both brightnesses and keeps the positions unbiased, while the one-brightness solve without c_a is biased / misfits
"""
import math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _case(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = 6
    k = 2 * math.pi / 2.1
    ang = torch.tensor([0, 0, 0, 1, 1, 1]); ph = torch.tensor([0, 2.09, 4.19, 0, 2.09, 4.19])
    pt1 = torch.stack([torch.where(ang == 0, k, 0.0), torch.where(ang == 1, k, 0.0), ph, torch.full((T,), 0.9), torch.ones(T)], 1)
    pt = pt1[None].expand(n, T, 5).contiguous()
    xy = torch.rand(n, 2, generator=g) * 3
    B = 400 + 600 * torch.rand(n, generator=g)
    from simcode.solve import pattern as C
    il, _, _ = C.illum_at(pt, xy[:, 0], xy[:, 1])
    I = B[:, None] * il + 15 * torch.randn(n, T, generator=g)
    A = torch.randn(n, T, T, generator=g) * 3
    L = torch.linalg.cholesky(A @ A.transpose(1, 2) + 200 * torch.eye(T))
    sp = 0.1 + 0.2 * torch.rand(n, 2, generator=g)
    r0 = xy + sp * torch.randn(n, 2, generator=g)
    Cj = 8 * torch.randn(n, T, 2, generator=g)
    return I, L, r0, sp, pt, Cj, B


def _explicit_cost(p, I, L, r0, sp, pt, Cj, cmult):
    from simcode.solve import pattern as C
    n, T = I.shape
    il, _, _ = C.illum_at(pt, p[:, 0], p[:, 1])
    mu = p[:, 2:3] * il                                                       # no kinetics (on-fraction 1)
    LJ = torch.zeros(n, T + 2, T + 2)
    LJ[:, 0, 0] = sp[:, 0]; LJ[:, 1, 1] = sp[:, 1]
    LJ[:, 2:, :2] = Cj                                                        # Sigma_Ip = C diag(sp)
    LJ[:, 2:, 2:] = L
    S = LJ @ LJ.transpose(1, 2)
    D = torch.diag_embed((cmult * mu.clamp(min=0)) ** 2 + (1e-6 if cmult > 0 else 0.0))
    S0 = S.clone()
    S[:, 2:, 2:] = S[:, 2:, 2:] + D
    e = torch.cat([p[:, :2] - r0, mu - I], 1)
    q = (e[:, None, :] @ torch.linalg.solve(S, e[..., None]))[:, 0, 0]
    ldx = torch.logdet(S) - torch.logdet(S0) if cmult > 0 else 0.0
    return q + ldx


def test_cost_equals_explicit_joint():
    from simcode.solve import solve_fast as MF
    I, L, r0, sp, pt, Cj, B = _case()
    n, T = I.shape
    p = torch.cat([r0 + 0.05, B[:, None], B[:, None], torch.full((n, 1), -1.0), torch.full((n, 1), T + 1.0)], 1)
    oh = torch.nn.functional.one_hot((pt[..., 0].abs() < pt[..., 1].abs()).long(), 2).float()
    Sig = L @ L.transpose(1, 2)
    for cmult in (0.0, 0.1):
        if cmult > 0:
            Lc = torch.linalg.cholesky(Sig); ld0 = 2 * torch.log(Lc.diagonal(dim1=1, dim2=2)).sum(1)
            args = (None, Sig, ld0, None)
        else:
            Lfix = torch.linalg.solve_triangular(torch.linalg.cholesky(Sig), torch.eye(T).expand(n, T, T), upper=False)
            args = (Lfix, None, None, None)
        rI, JI, rp, cost = MF._eval(p, pt, oh, I, r0, sp, Cj, *args, cmult, 0.0, False, T)
        ref = _explicit_cost(p, I, L, r0, sp, pt, Cj, cmult)
        assert torch.allclose(cost.double(), ref.double(), rtol=2e-4, atol=1e-3), (cmult, (cost - ref).abs().max())


def test_nll_joint():
    from simcode.emitter.model import nll_joint
    I, L, r0, sp, pt, Cj, B = _case(32, 1)
    n, T = I.shape
    Ig = I + 10 * torch.randn(n, T)
    ep = 0.1 * torch.randn(n, 2)                                              # true - prior
    nll, _ = nll_joint(Ig, I, L, Cj, ep / sp)
    LJ = torch.zeros(n, T + 2, T + 2); LJ[:, 0, 0] = sp[:, 0]; LJ[:, 1, 1] = sp[:, 1]; LJ[:, 2:, :2] = Cj; LJ[:, 2:, 2:] = L
    e = torch.cat([ep, Ig - I], 1)[..., None]
    z = torch.linalg.solve_triangular(LJ, e, upper=False)[..., 0]
    full = 0.5 * (z ** 2).sum(1) + torch.log(LJ.diagonal(dim1=1, dim2=2)).sum(1)
    prior = 0.5 * ((ep / sp) ** 2).sum(1) + torch.log(sp).sum(1)
    const = torch.log((Ig.clamp(min=0).sum(1) / 6 + 30).sqrt()) * 6
    assert torch.allclose(nll + const, full - prior, rtol=1e-4, atol=1e-3)


def test_reference_vs_fast():
    from simcode.solve import solve as SV, solve_fast as MF
    I, L, r0, sp, pt, Cj, B = _case(48, 2)
    Sig = L @ L.transpose(1, 2)
    a = SV.fringe_posterior(I, Sig, r0, sp, pt, [r0], Cj, cmult=0.1, cang=0.1)
    b = MF.fringe_posterior_fast(I, Sig, r0, sp, pt, [r0], Cj, cmult=0.1, cang=0.1)
    d = (a['r_pm'] - b['r_pm']).abs().amax(1)
    assert float(d.median()) < 1e-4 and float((a['s_pm'] - b['s_pm']).abs().max()) < 1e-3, (float(d.median()), float(d.max()))


def test_two_brightness():
    from simcode.solve import solve as SV, solve_fast as MF, pattern as C
    I, L, r0, sp, pt, Cj, B = _case(64, 3)
    Sig = L @ L.transpose(1, 2)
    for kw in (dict(cmult=0.0), dict(cmult=0.08)):
        a = SV.fringe_posterior(I, Sig, r0, sp, pt, [r0], Cj, two_b=True, **kw)
        b = MF.fringe_posterior_fast(I, Sig, r0, sp, pt, [r0], Cj, two_b=True, **kw)
        d = (a['r_pm'] - b['r_pm']).abs().amax(1)
        assert float(d.median()) < 1e-4 and float((a['s_pm'] - b['s_pm']).abs().max()) < 1e-3, (kw, float(d.median()))
        dn = MF.fringe_posterior_fast(I, Sig, r0, sp, pt, [r0], Cj, two_b=True, kin='none', **kw)['dof']
        assert torch.allclose(dn, torch.full_like(dn, 6 + 2 - 4))                 # x, y, B_0, B_1
        o1 = MF.fringe_posterior_fast(I, Sig, r0, sp, pt, [r0], Cj, **kw)
        # same data, equal brightness: the two solutions agree to a fraction of their sd; the extra parameter costs precision
        assert float(((b['r_pm'] - o1['r_pm']).abs() / o1['s_pm']).median()) < 0.3
        assert float((b['s_pm'] / o1['s_pm']).median()) >= 0.999
    # a brightness ratio 1.3 : 0.77 between the orientations (no position noise in I): two_b fits it, the means follow
    g = torch.Generator().manual_seed(5); n = 64
    xy = torch.rand(n, 2, generator=g) * 3
    il, _, _ = C.illum_at(pt, xy[:, 0], xy[:, 1])
    f = torch.where(torch.arange(6) < 3, 1.3, 1 / 1.3)
    I2 = 800 * f[None] * il + 5 * torch.randn(n, 6, generator=g)
    Sig2 = 25.0 * torch.eye(6).expand(n, 6, 6).contiguous(); sp2 = torch.full((n, 2), 2.0); C0 = torch.zeros(n, 6, 2)
    r2 = MF.fringe_posterior_fast(I2, Sig2, xy, sp2, pt, [xy], C0, two_b=True, kin='none')
    e2 = (r2['r'] - xy).abs().amax(1)
    assert float(e2.median()) < 0.01, float(e2.median())
    mu = r2['mu']; ratio = (mu[:, :3].sum(1) / il[:, :3].sum(1)) / (mu[:, 3:].sum(1) / il[:, 3:].sum(1))
    assert torch.allclose(ratio, torch.full_like(ratio, 1.69), rtol=0.02), ratio
    r1 = MF.fringe_posterior_fast(I2, Sig2, xy, sp2, pt, [xy], C0, kin='none')
    assert float(r1['q'].median()) > 10 * float(r2['q'].median())
    # the calibration key
    cal = dict(prior_off=[0, 0], box=1.5, mix_T=1.0, cmult=0.0, cmult_ang=0.0)
    assert SV.brightness_per_angle(cal) is False and SV.brightness_per_angle(dict(cal, B_per_angle=True)) is True


def test_kinetic_average():
    """kinetic model averaging (solve.model_average): weights in [0, 1]; an emitter on through the window -> mostly the
    non-kinetic model, one that switches off half way -> mostly the kinetic one; 'none' / 'kinetic' = one model; the average
    position is the weighted mean of the pooled modes; fast = reference"""
    from simcode.solve import solve as SV, solve_fast as MF
    from simcode.solve import pattern as C
    I, L, r0, sp, pt, Cj, B = _case(64, 3)
    n = len(I); Cj = 0.0 * Cj
    il, _, _ = C.illum_at(pt, r0[:, 0], r0[:, 1])
    on = torch.ones(n, 6); on[n // 2:, 3:] = 0.0                                # second half: off after frame 3
    Ig = 2000 * il * on + 15 * torch.randn(n, 6, generator=torch.Generator().manual_seed(5))
    Sig = L @ L.transpose(1, 2)
    kw = dict(cmult=0.05, cang=0.05)
    a = SV.fringe_posterior(Ig, Sig, r0, sp, pt, [r0], Cj, **kw)
    b = MF.fringe_posterior_fast(Ig, Sig, r0, sp, pt, [r0], Cj, **kw)
    assert ((a['kin'] >= 0) & (a['kin'] <= 1 + 1e-5)).all() and torch.allclose(a['w'].sum(1), torch.ones(n), atol=1e-5)
    assert float(a['kin'][: n // 2].median()) < 0.3 and float(a['kin'][n // 2:].median()) > 0.9, (a['kin'][: n // 2].median(), a['kin'][n // 2:].median())
    assert torch.allclose((a['w'][..., None] * torch.where(a['w'][..., None] > 0, a['R'], torch.zeros_like(a['R']))).sum(1), a['r_pm'], atol=1e-5)
    assert float((a['r_pm'] - b['r_pm']).abs().amax(1).median()) < 1e-4 and float((a['kin'] - b['kin']).abs().max()) < 1e-2
    for m, k in (('none', 0.0), ('kinetic', 1.0)):
        c = SV.fringe_posterior(Ig, Sig, r0, sp, pt, [r0], Cj, kin=m, **kw)
        assert torch.allclose(c['kin'], torch.full((n,), k)) and c['R'].shape[1] == a['R'].shape[1] // 2
    assert abs(SV.KIN_PENALTY - 2 * math.log(6)) < 1e-12


def test_solve_cal():
    """the solve calibration: the model's (no fixed c) + the dataset's brightness noise; a fixed c in a model file is refused"""
    from simcode.solve import solve as SV
    c = SV.solve_cal(dict(prior_off=[0, 0], box=1.5, mix_T=1.0), 0.12, 0.07)
    assert SV.brightness_noise(c) == (0.12, 0.07) and SV.pattern_cal(dict(box=1.5), dict(cmult=0.2, cmult_ang=0.1))['cmult'] == 0.2
    for bad in (lambda: SV.solve_cal(dict(box=1.5, cmult=0.1), 0.1, 0.1), lambda: SV.pattern_cal(dict(box=1.5), dict(k=0))):
        try: bad(); raise AssertionError('must raise')
        except (ValueError, KeyError): pass
    try: SV.brightness_noise(dict(box=1.5)); raise AssertionError('must raise')
    except KeyError: pass


if __name__ == '__main__':
    for f in (test_cost_equals_explicit_joint, test_nll_joint, test_reference_vs_fast, test_two_brightness, test_kinetic_average, test_solve_cal):
        f(); print('ok', f.__name__)
