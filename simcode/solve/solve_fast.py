"""
Fast version of the explicit intensity-domain pattern solve (2026-11-10, throughput; the pipeline's solve; the reference
simcode/solve/solve.py and simcode/solve/solve.solve_emitters are its test oracle).

pattern_solve_all_fast = solve.pattern_solve_all (same model, same LM schedule, same clamps / box / kinetic starts,
same cmult / cang nuisance covariance and log det, same outputs), restructured for speed:
  * no batched cuBLAS bmm on 6 x 6 matrices (66 % of the original solve time): the small products are broadcast sums
  * the fixed parameters (B1 always; t_on / t_off without kinetics) are removed from the normal equations: a 3 x 3 /
    5 x 5 system instead of 6 x 6 with unit rows (mathematically identical)
  * the whitening uses one triangular solve with all right-hand sides (residual + Jacobian columns) instead of the
    explicit inverse + products
  * the final covariance (pinv of J^T J) only for the rows that are returned (best kinetic start per start), by
    Cholesky inverse with an SVD-pinv fallback for ill-conditioned rows
Both kinetic models are fitted for every row and averaged (solve.model_average; 2026-11-17: the selection threshold and its
  kin_gate shortcut were removed).
compile: 'auto' | 'on' | 'off' (simcode/util/compile.py; the compiled LM step is the same code).
(2026-11-14) The approximate options (per-row early stop tol, covariance freezing freeze_cov, start pruning prune / prune_k,
cov_mode) were removed: no production or paper run used them.
"""
import torch
from simcode.solve import pattern as C

T = 6
_CACHE = {}


def _mm(A, B):
    """[M,a,b] @ [M,b,c] -> [M,a,c] without bmm"""
    return (A[:, :, :, None] * B[:, None, :, :]).sum(2)


def _gram(J):
    """J [M,R,K] -> J^T J [M,K,K]"""
    return (J[:, :, :, None] * J[:, :, None, :]).sum(1)


# ------------------------------------------------------------------------------------------------ model evaluation
def _model(p, ptr, oh, kin, Tn):
    """p [M,6] (x, y, B, B, t_on, t_off) -> mu [M,T], Jm [M,T,K] (K = 3 or 5 free columns: x, y, B (, t_on, t_off))"""
    il, dx, dy = C.illum_at(ptr, p[:, 0], p[:, 1])
    a_, d0, d1 = C.onfrac(p[:, 4], p[:, 5], Tn)
    B = p[:, 2:3]
    mu = B * a_ * il
    cols = [B * a_ * dx, B * a_ * dy, a_ * il]
    if kin:
        cols += [B * il * d0, B * il * d1]
    return mu, torch.stack(cols, -1)


def _eval(p, ptr, oh, Ir, r0r, spr, Cr, Lfix, Sigr, ld0, A2, cmult, cang, kin, Tn):
    """residual / Jacobian / cost at p (as solve.pattern_solve_all.rj; joint prior / intensity covariance: the intensities shifted
    by Cr z_p, z_p = (p_xy - r0) / sp): returns rI [M,T], JI [M,T,K], rp [M,2] (= z_p), cost [M]"""
    mu, Jm = _model(p, ptr, oh, kin, Tn)
    rp = (p[:, :2] - r0r) / spr
    Ie = Ir + (Cr * rp[:, None, :]).sum(-1)
    Jm = torch.cat([Jm[..., :2] - Cr / spr[:, None, :], Jm[..., 2:]], -1)
    rhs = torch.cat([(Ie - mu)[..., None], Jm], -1)                          # [M,T,1+K]
    ldx = 0.0
    if cmult > 0:
        mup = mu.clamp(min=0)
        Sp = Sigr + torch.diag_embed((cmult * mup) ** 2 + 1e-6)
        if cang > 0:
            Sp = Sp + cang ** 2 * (mup[:, :, None] * mup[:, None, :]) * A2
        Lp, _ = torch.linalg.cholesky_ex(Sp)
        W = torch.linalg.solve_triangular(Lp, rhs, upper=False)
        ldx = 2 * torch.log(Lp.diagonal(dim1=1, dim2=2).clamp(min=1e-12)).sum(1) - ld0
    else:
        W = _mm(Lfix, rhs)                                                    # Lfix = Sigma^-1/2 (lower), as the original
    rI = W[..., 0]; JI = -W[..., 1:]
    cost = (rI ** 2).sum(1) + (rp ** 2).sum(1) + ldx
    return rI, JI, rp, cost


def _normal(JI, rI, rp, spr):
    """J^T J [M,K,K] and J^T r [M,K] incl. the Gaussian prior rows (d rp / d xy = 1 / sp)"""
    Hm = _gram(JI)
    gr = (JI * rI[..., None]).sum(1)
    isp = 1.0 / spr
    Hm[:, 0, 0] += isp[:, 0] ** 2; Hm[:, 1, 1] += isp[:, 1] ** 2
    gr[:, 0] += rp[:, 0] * isp[:, 0]; gr[:, 1] += rp[:, 1] * isp[:, 1]
    return Hm, gr


def _clamp(p, lo_xy, hi_xy, Tn):
    x = torch.maximum(torch.minimum(p[:, :2], hi_xy), lo_xy)
    B = p[:, 2:3].clamp(min=1.0)
    t_on = p[:, 4].clamp(-1, Tn - 0.25); t_off = torch.maximum(p[:, 5], t_on + 0.25).clamp(max=Tn + 1)
    return torch.cat([x, B, B, t_on[:, None], t_off[:, None]], 1)


def _lm_step(p, rI, JI, rp, cost, lam, ptr, oh, Ir, r0r, spr, Cr, Lfix, Sigr, ld0, A2, lo_xy, hi_xy,
             cmult: float, cang: float, kin: bool, Tn: int):
    """one LM iteration of solve.pattern_solve_all (accept if the cost decreases; lam / 3 or x 4)"""
    Hm, gr = _normal(JI, rI, rp, spr)
    Hd = Hm + torch.diag_embed(lam[:, None] * Hm.diagonal(dim1=1, dim2=2) + 1e-9)
    dpf = -torch.nan_to_num(torch.linalg.solve_ex(Hd, gr[..., None])[0][..., 0])
    z = torch.zeros_like(dpf[:, :1])
    dp = torch.cat([dpf[:, :3], z, dpf[:, 3:]] if kin else [dpf, z, z, z], 1)
    pn = _clamp(p + dp, lo_xy, hi_xy, Tn)
    rIn, JIn, rpn, cn = _eval(pn, ptr, oh, Ir, r0r, spr, Cr, Lfix, Sigr, ld0, A2, cmult, cang, kin, Tn)
    acc = cn < cost
    p = torch.where(acc[:, None], pn, p); rI = torch.where(acc[:, None], rIn, rI)
    JI = torch.where(acc[:, None, None], JIn, JI); rp = torch.where(acc[:, None], rpn, rp)
    cost = torch.where(acc, cn, cost)
    lam = torch.where(acc, lam / 3, lam * 4).clamp(1e-7, 1e7)
    return p, rI, JI, rp, cost, lam


def _get_step(compile_mode):
    from simcode.util import compile as torch_compile
    key = ('step', torch_compile.resolve(compile_mode))
    if key not in _CACHE:
        _CACHE[key] = torch_compile.maybe_compile(_lm_step, compile_mode, dynamic=True) if key[1] else _lm_step
    return _CACHE[key]


# ------------------------------------------------------------------------------------------------ solve
def pattern_solve_all_fast(I, Sig, r0, sp, pt, dr, Cj, kinetics=True, iters=30, box=0.75,
                           kin_starts=((0.0, 0.0), (-0.6, 0.6), None), cmult=0.0, cang=0.0, compile='off'):
    """as solve.pattern_solve_all (nu = None)"""
    n, Tn = I.shape; dev = I.device
    _, ang = C.angle_k(pt)
    H = dr.shape[1]
    ks = list(kin_starts) if kinetics else [None]
    nk = len(ks); ns = H * nk
    Lc, info = torch.linalg.cholesky_ex(Sig)
    if (info != 0).any():
        Sig = torch.where((info != 0)[:, None, None], Sig + 1e-3 * torch.diag_embed(Sig.diagonal(dim1=1, dim2=2)), Sig)
        Lc, _ = torch.linalg.cholesky_ex(Sig)
    cmult = float(cmult); cang = float(cang)
    if cang > 0 and cmult <= 0: cmult = 1e-6
    # rows: (i, h, k) in the original order
    rows = torch.arange(n * ns, device=dev)
    ri = rows // ns; hk = rows % ns; hi_ = hk // nk; kidx = hk % nk
    M = len(rows)
    st = (r0[:, None] + dr)[ri, hi_]
    Ir, r0r, spr, ptr, angr, Cr = I[ri], r0[ri], sp[ri], pt[ri], ang[ri], Cj[ri]
    oh = torch.nn.functional.one_hot(angr, 2).float()
    A2 = oh @ oh.transpose(1, 2) if cang > 0 else None
    if cmult > 0:
        Sigr = Sig[ri]; ld0 = (2 * torch.log(Lc.diagonal(dim1=1, dim2=2).clamp(min=1e-12)).sum(1))[ri]; Lfix = None
    else:
        Sigr = ld0 = None
        Lfix = torch.linalg.solve_triangular(Lc, torch.eye(Tn, device=dev).expand(n, Tn, Tn), upper=False)[ri]
    il0 = C.illum_at(ptr, st[:, 0], st[:, 1])[0]
    B0, ton0, toff0 = C.kinetics_init(Ir, il0, Tn)
    if not kinetics:
        ton = torch.full((M,), -1.0, device=dev); toff = torch.full((M,), Tn + 1.0, device=dev)
    else:
        ton = torch.zeros(M, device=dev); toff = torch.zeros(M, device=dev)
        for j, k_ in enumerate(ks):
            m_ = kidx == j
            if k_ is None:
                ton = torch.where(m_, torch.full_like(ton, -0.5), ton); toff = torch.where(m_, torch.full_like(toff, Tn + 0.5), toff)
            else:
                ton = torch.where(m_, ton0 + k_[0], ton); toff = torch.where(m_, toff0 + k_[1], toff)
    p = torch.cat([st, B0[:, None], B0[:, None], ton[:, None], toff[:, None]], 1)
    lo_xy, hi_xy = st - box, st + box
    kin = bool(kinetics)
    p = _clamp(p, lo_xy, hi_xy, Tn); lam = torch.full((M,), 1e-2, device=dev)
    rI, JI, rp, cost = _eval(p, ptr, oh, Ir, r0r, spr, Cr, Lfix, Sigr, ld0, A2, cmult, cang, kin, Tn)
    step_fn = _get_step(compile)
    cc = [ptr, oh, Ir, r0r, spr, Cr, Lfix, Sigr, ld0, A2, lo_xy, hi_xy]
    for it in range(iters):
        p, rI, JI, rp, cost, lam = step_fn(p, rI, JI, rp, cost, lam, *cc, cmult, cang, kin, Tn)
    chi2 = (rI ** 2).sum(1) + (rp ** 2).sum(1)
    # scatter back to the full (n, H, nk) grid
    full = lambda v, fill: torch.full((n * ns,) + v.shape[1:], fill, device=dev, dtype=v.dtype).index_copy_(0, rows, v)
    costF = full(torch.nan_to_num(cost, nan=1e30), float('inf')).view(n, H, nk)
    cf, kb = costF.min(2)
    ar = torch.arange(n, device=dev)[:, None]; hh = torch.arange(H, device=dev)[None]
    sel = (ar * ns + hh * nk + kb).reshape(-1)                                # selected global row per (i, h)
    pos = torch.full((n * ns,), -1, device=dev, dtype=torch.long).index_copy_(0, rows, torch.arange(M, device=dev))[sel]
    okr = pos >= 0
    pidx = pos.clamp(min=0)
    P_ = p[pidx]
    # covariance only for the selected rows: (J^T J)^-1 of the free block
    Hm, _ = _normal(JI[pidx], rI[pidx], rp[pidx], spr[pidx])
    # = the original pinv of the 6 x 6 J^T J + fixed_diag (singular values <= 6 eps max(s_max, 1) dropped). Rows where the
    # Cholesky inverse provably equals it (lambda_min >= 1 / ||A^-1||_F > 6 eps max(trace A, 1) >= the threshold) use the
    # Cholesky inverse; the others an SVD pseudo-inverse with that threshold
    eps6 = 6 * torch.finfo(torch.float32).eps
    Kd = Hm.shape[-1]; eyeK = torch.eye(Kd, device=dev).expand_as(Hm)
    Lh, infoh = torch.linalg.cholesky_ex(Hm)
    Li = torch.linalg.solve_triangular(torch.where((infoh != 0)[:, None, None], eyeK, Lh), eyeK, upper=False)
    cov = _mm(Li.transpose(1, 2), Li)
    tr = Hm.diagonal(dim1=1, dim2=2).sum(1)
    safe = (infoh == 0) & torch.isfinite(cov).all(2).all(1) & \
           (1.0 / (cov ** 2).sum((1, 2)).sqrt().clamp(min=1e-30) > 2 * eps6 * tr.clamp(min=1.0))
    ib = torch.nonzero(~safe)[:, 0]
    if len(ib):
        U_, w_, Vh_ = torch.linalg.svd(Hm[ib])
        thr_ = eps6 * w_.amax(1, keepdim=True).clamp(min=1.0)
        iw = torch.where(w_ > thr_, 1.0 / w_, torch.zeros_like(w_))
        cov[ib] = _mm(Vh_.transpose(1, 2) * iw[:, None, :], U_.transpose(1, 2))
    sig = cov.diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt()
    st_all = (r0[:, None] + dr).reshape(-1, 2)
    P_xy = torch.where(okr[:, None], P_[:, :2], st_all)
    nfree = 3 + 2 * int(kin)
    out = dict(r=P_xy.view(n, H, 2), s=torch.where(okr[:, None], sig, torch.full_like(sig, float('nan'))).view(n, H, 2), nll=cf,
               dof=torch.full((n,), float(Tn + 2 - nfree), device=dev))
    il_, _, _ = C.illum_at(ptr[pidx], P_[:, 0], P_[:, 1]); a_f, _, _ = C.onfrac(P_[:, 4], P_[:, 5], Tn)
    out['mu'] = (P_[:, 2:3] * a_f * il_).view(n, H, Tn)
    if cmult > 0:
        c2 = torch.nan_to_num(chi2, nan=1e30)[pidx]
        out['chi2'] = torch.where(okr, c2, torch.full_like(c2, float('inf'))).view(n, H)
    return out


@torch.no_grad()
def fringe_posterior_fast(I, Sig, r0, sp, pt, bases, Cj, nf=1, box=1.5, kin='average', iters=30, temp=1.0, dedup=0.15, cmult=0.0,
                          cang=0.0, compile='off'):
    """as solve.fringe_posterior (same outputs; the posterior over the fringe starts and the kinetic model is solve.model_average)"""
    from simcode.solve.solve import start_offsets, model_average, kin_models
    dr = start_offsets(pt, r0, bases, nf)
    kw = dict(iters=iters, box=box, cmult=cmult, cang=cang, compile=compile)
    o0 = pattern_solve_all_fast(I, Sig, r0, sp, pt, dr, Cj, kinetics=False, **kw) if kin != 'kinetic' else None
    o1 = pattern_solve_all_fast(I, Sig, r0, sp, pt, dr, Cj, kinetics=True, **kw) if kin != 'none' else None
    return model_average(kin_models(kin, o0, o1), temp=temp, dedup=dedup)


@torch.no_grad()
def solve_emitters_fast(o, pt_win, b, cal, nf=1, kin='average', **opts):
    """simcode.solve.solve.solve_emitters (same calibration keys, same outputs, the optional per-row intensity-map factors
    o['relg']) with the fast fringe posterior"""
    from simcode.solve import solve as SV
    I, Sig, r0, sp, ph, Cj = SV.solve_inputs(o, cal, mm=_mm)
    c, ca = SV.brightness_noise(cal)
    s = fringe_posterior_fast(I, Sig, r0, sp, C.scale_rel(pt_win[b], o.get('relg')), [r0], Cj, nf=nf, box=cal['box'], kin=kin, temp=cal['mix_T'],
                              cmult=c, cang=ca, **opts)
    return SV.finish(s, I, Sig, r0, sp)
