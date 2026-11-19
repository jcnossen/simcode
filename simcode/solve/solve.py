"""
Multi-start intensity-domain pattern solve (critic review 28 #6; no training).
pattern_solve_all = hybrid_q.pattern_solve_cov with explicit start offsets dr [n,H,2] (relative to the prior centre r0)
that returns the per-start results (best kinetic start per start), so the same set of starts can be selected by the
posterior NLL. fringe_posterior = kinetic model selection + the posterior over the starts (the solve of the pipeline,
reference implementation; simcode/solve/solve_fast.py is the fast version). solve_emitters = the solve of the emitter-network
outputs (intensities with their full covariance, prior position, calibration) that the pipeline runs.
Start sets: base points (r_net = the prior centre, x_det = the detector point, ...) each with the (2 nf + 1)^2 fringe
shifts; box = per-start position box half-width (px).
"""
import math
import torch
from simcode.solve import pattern as C

T = 6


def pattern_solve_all(I, Sig, r0, sp, pt, dr, Cj, kinetics=True, iters=30, box=0.75,
                      kin_starts=((0.0, 0.0), (-0.6, 0.6), None), cmult=0.0):
    # (2026-11-17) joint prior / intensity covariance of the emitter network (EmitterNet: L_J = [[diag(sp), 0], [Cj, L]]):
    # Sig = L L^T is the intensity covariance GIVEN the prior error, Cj [n,T,2] the coupling. The joint Gaussian cost
    # e_J^T Sigma_J^-1 e_J (e_J = [p_xy - r0; mu - I]) = |(p_xy - r0) / sp|^2 + |L^-1 (I + Cj z_p - mu)|^2, z_p = (p_xy - r0) / sp:
    # the intensity residual uses I shifted by Cj z_p(p) (its x, y Jacobian gets - Cj / sp). The nuisance term c adds to the
    # intensity block of Sigma_J, i.e. to the conditional covariance Sig as before (the conditional mean shift is unchanged).
    # cmult (2026-11-09, critic 32 #5): nuisance-aware intensity covariance Sigma(p) = Sig + diag((cmult mu_t(p))^2) with the
    # CURRENT model mean mu_t (multiplicative per-frame emitter-brightness noise; not the observed I_t), re-whitened at every
    # LM evaluation (GLS Jacobian: d Sigma / d p ignored; the cost is the exact Gaussian NLL incl. log det Sigma(p) - log det Sig).
    # (2026-11-19: the per-orientation term c_a^2 mu mu^T and the optional brightness per orientation were removed: no effect on
    # the precision or the same-rows FRC of real or simulated data, out/model_studies.) Returned 'nll' = that cost (start /
    # kinetic selection), 'chi2' = the whitened residual + prior part.
    n, T_ = I.shape; dev = I.device
    H = dr.shape[1]
    ks = list(kin_starts) if kinetics else [None]
    nk = len(ks); ns = H * nk
    Lc, info = torch.linalg.cholesky_ex(Sig)
    if (info != 0).any():
        Sig = torch.where((info != 0)[:, None, None], Sig + 1e-3 * torch.diag_embed(Sig.diagonal(dim1=1, dim2=2)), Sig)
        Lc, _ = torch.linalg.cholesky_ex(Sig)
    Linv = torch.linalg.solve_triangular(Lc, torch.eye(T_, device=dev).expand(n, T_, T_), upper=False)
    rep = lambda t_: t_.repeat_interleave(ns, 0)
    st = (r0[:, None] + dr).repeat_interleave(nk, 1).reshape(-1, 2)
    Ir, r0r, spr, ptr, Lr, Cr = (rep(t_) for t_ in (I, r0, sp, pt, Linv, Cj))
    cmult = float(cmult)
    if cmult > 0:
        Sigr = rep(Sig); ld0 = 2 * torch.log(rep(Lc).diagonal(dim1=1, dim2=2).clamp(min=1e-12)).sum(1)
        eyeT = torch.eye(T_, device=dev)
    M = len(st)
    il0 = C.illum_at(ptr, st[:, 0], st[:, 1])[0]
    B0, ton0, toff0 = C.kinetics_init(Ir, il0, T_)
    ton = torch.empty(M, device=dev); toff = torch.empty(M, device=dev)
    kidx = torch.arange(M, device=dev) % nk
    for j, k_ in enumerate(ks):
        m_ = kidx == j
        if k_ is None:
            ton[m_] = -0.5; toff[m_] = T_ + 0.5
        else:
            ton[m_] = ton0[m_] + k_[0]; toff[m_] = toff0[m_] + k_[1]
    if not kinetics:
        ton[:] = -1.0; toff[:] = T_ + 1.0
    NP = 6
    p = torch.cat([st, B0[:, None], B0[:, None], ton[:, None], toff[:, None]], 1)
    lo_xy, hi_xy = st - box, st + box
    free = torch.tensor([1, 1, 1, 0, float(kinetics), float(kinetics)], device=dev)
    fixed_diag = torch.diag_embed((1 - free).expand(M, NP))

    def clamp(p):
        x = torch.maximum(torch.minimum(p[:, :2], hi_xy), lo_xy)
        B = p[:, 2:4].clamp(min=1.0)[:, :1].expand(-1, 2)
        t_on = p[:, 4].clamp(-1, T_ - 0.25); t_off = torch.maximum(p[:, 5], t_on + 0.25).clamp(max=T_ + 1)
        return torch.cat([x, B, t_on[:, None], t_off[:, None]], 1)

    def rj(p):
        il, dx, dy = C.illum_at(ptr, p[:, 0], p[:, 1])
        a_, d0, d1 = C.onfrac(p[:, 4], p[:, 5], T_)
        Bt = p[:, 2:3]
        mu = Bt * a_ * il
        Jm = torch.stack([Bt * a_ * dx, Bt * a_ * dy, a_ * il, torch.zeros_like(il), Bt * il * d0, Bt * il * d1], -1) * free
        ldx = 0.0
        if cmult > 0:
            mup = mu.clamp(min=0)
            Sp = Sigr + torch.diag_embed((cmult * mup) ** 2) + 1e-6 * eyeT
            Lp, _ = torch.linalg.cholesky_ex(Sp)
            Lri = torch.linalg.solve_triangular(Lp, eyeT.expand(M, T_, T_), upper=False)
            ldx = 2 * torch.log(Lp.diagonal(dim1=1, dim2=2).clamp(min=1e-12)).sum(1) - ld0
        else:
            Lri = Lr
        zp = (p[:, :2] - r0r) / spr
        Ie = Ir + (Cr * zp[:, None, :]).sum(-1)
        Jm = torch.cat([Jm[..., :2] - Cr / spr[:, None, :], Jm[..., 2:]], -1)
        rI = (Lri @ (Ie - mu)[..., None])[..., 0]
        JI = -(Lri @ Jm)
        rp = zp
        Jp = torch.zeros(M, 2, NP, device=dev); Jp[:, 0, 0] = 1 / spr[:, 0]; Jp[:, 1, 1] = 1 / spr[:, 1]
        r = torch.cat([rI, rp], 1); J = torch.cat([JI, Jp], 1)
        return r, J, (r ** 2).sum(1) + ldx, rI

    p = clamp(p); lam = torch.full((M,), 1e-2, device=dev)
    r, J, cost, rI = rj(p)
    for _ in range(iters):
        Hm = J.transpose(1, 2) @ J
        gr = J.transpose(1, 2) @ r[..., None]
        Hd = Hm + torch.diag_embed(lam[:, None] * Hm.diagonal(dim1=1, dim2=2) + 1e-9) + fixed_diag
        dp = -torch.nan_to_num(torch.linalg.solve_ex(Hd, gr)[0][..., 0]) * free
        pn = clamp(p + dp); rn, Jn, cn, rIn = rj(pn)
        acc = cn < cost
        p = torch.where(acc[:, None], pn, p); r = torch.where(acc[:, None], rn, r); J = torch.where(acc[:, None, None], Jn, J)
        cost = torch.where(acc, cn, cost); rI = torch.where(acc[:, None], rIn, rI)
        lam = torch.where(acc, lam / 3, lam * 4).clamp(1e-7, 1e7)
    cov = torch.linalg.pinv(J.transpose(1, 2) @ J + fixed_diag)
    sig = cov.diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt()
    chi2 = (r ** 2).sum(1)
    cost = torch.nan_to_num(cost, nan=1e30).view(n, H, nk)
    cf, kb = cost.min(2)                                                     # per start: best kinetic start
    ar = torch.arange(n, device=dev)[:, None]; hh = torch.arange(H, device=dev)[None]
    idx = hh * nk + kb
    P_ = p.view(n, ns, NP)[ar, idx]; S_ = sig.view(n, ns, 2)[ar, idx]
    nfree = int(free.sum().item())
    out = dict(r=P_[..., :2], s=S_, nll=cf, dof=torch.full((n,), float(T_ + 2 - nfree), device=dev))
    # 2026-11-09 (critic 33 #2, automatic c): the model mean mu_t at each start's final solution (additive output)
    il_, _, _ = C.illum_at(ptr, p[:, 0], p[:, 1]); a_f, _, _ = C.onfrac(p[:, 4], p[:, 5], T_)
    out['mu'] = (p[:, 2:3] * a_f * il_).view(n, ns, T_)[ar, idx]
    if cmult > 0: out['chi2'] = torch.nan_to_num(chi2, nan=1e30).view(n, ns)[ar, idx]
    return out


def start_offsets(pt, r0, bases, nf=1):
    """dr [n,H,2]: for each base point [n,2], base - r0 + the fringe shifts (nf)"""
    Km, _ = C.angle_k(pt)
    fs = C.fringe_shifts(Km, nf)                                             # [n,(2nf+1)^2,2]
    return torch.cat([(b - r0)[:, None] + fs for b in bases], 1)


# kinetic model (2026-11-17): the SAME procedure for every row: the non-kinetic model (x, y, B) and the kinetic model (+ t_on, t_off)
# are both fitted and averaged by their posterior weights (Bayesian model averaging; no selection threshold). The evidence of a
# model is the sum of its fringe modes' weights exp(-cost / 2) (cost = -2 log of likelihood x position prior, as the posterior over
# the fringe starts); the kinetic model's 2 extra parameters are penalised by the BIC term k ln n = 2 ln T (n = the T = 6 network
# intensities of the window): cost_kinetic + KIN_PENALTY. The table column kin = the kinetic model's posterior weight.
KIN_MODES = ('average', 'none', 'kinetic')       # 'none' / 'kinetic': one model only (analysis comparisons)
KIN_PENALTY = 2 * math.log(T)


def model_average(models, temp=1.0, dedup=0.15):
    """posterior over the fringe modes of one or more models. models: list of (o, penalty, kinetic) with o = pattern_solve_all
    output (nll [n,H], r, s [n,H,2], mu [n,H,T], chi2 [n,H] when the nuisance covariance c is on, dof [n]). Per model the starts
    are sorted by cost and deduplicated (a start whose solution lies within `dedup` px of a lower-cost start of the same model is
    the same mode, weight 0); the modes of all models are pooled with cost + penalty and weighted w ~ exp(-(cost - cost_min) /
    (2 temp)). Returns dict(r, s, q, dof, kin_map: the MAP mode (lowest penalised cost); kin = the summed weight of the kinetic
    modes; dnll = (cost of the best mode >= dedup px from the MAP - MAP cost) / 2 (inf if none); R, S [n,M,2], w [n,M] (M = all
    starts of all models); r_pm posterior mean, spread [n,2] = sqrt(sum w (R - r_map)^2), spread_pm = sqrt(sum w (R - r_pm)^2),
    s_pm = sqrt(sum w S^2); mu = the model mean at the MAP; q_nll / q (nll / chi2 part at the MAP, chi2 only with c on))"""
    Q, Qr, R, S, MU, CH, KN, DF = [], [], [], [], [], [], [], []
    for o, pen, kin in models:
        q_, order = o['nll'].sort(1)
        n, H = q_.shape; dev = q_.device
        g2 = lambda v: v.gather(1, order[..., None].expand(-1, -1, v.shape[-1]))
        r_, s_, mu_ = g2(o['r']), g2(o['s']), g2(o['mu'])
        Dm = ((r_[:, :, None] - r_[:, None, :]) ** 2).sum(-1).sqrt()
        lower = torch.tril(torch.ones(H, H, device=dev, dtype=torch.bool), -1)
        dup = ((Dm < dedup) & lower[None]).any(2) | ~torch.isfinite(q_)
        Q.append(torch.where(dup, torch.full_like(q_, float('inf')), q_ + pen)); Qr.append(q_)
        R.append(r_); S.append(s_); MU.append(mu_)
        CH.append(o['chi2'].gather(1, order) if 'chi2' in o else q_)
        KN.append(torch.full_like(q_, float(kin))); DF.append(o['dof'][:, None].expand(-1, H))
    Q, Qr, R, S, MU, CH, KN, DF = (torch.cat(v, 1) for v in (Q, Qr, R, S, MU, CH, KN, DF))
    n = Q.shape[0]; ar = torch.arange(n, device=Q.device)
    q0, bm = Q.min(1)
    w = torch.exp(-(Q - q0[:, None]) / (2 * temp)); w = torch.where(torch.isfinite(Q), w, torch.zeros_like(w))
    w = w / w.sum(1, keepdim=True)
    r_map = R[ar, bm]
    far = ((R - r_map[:, None]) ** 2).sum(-1).sqrt() >= dedup
    sec = torch.where(far, Q, torch.full_like(Q, float('inf'))).min(1).values
    Rz = torch.where(w[..., None] > 0, R, torch.zeros_like(R)); Sz = torch.where(w[..., None] > 0, S, torch.zeros_like(S))
    r_pm = (w[..., None] * Rz).sum(1)
    out = dict(r=r_map, s=S[ar, bm], q=Qr[ar, bm], dof=DF[ar, bm], kin=(w * KN).sum(1), kin_map=KN[ar, bm] > 0.5, dnll=(sec - q0) / 2,
               R=R, S=S, w=w, r_pm=r_pm, spread=(w[..., None] * (Rz - r_map[:, None]) ** 2).sum(1).sqrt(),
               spread_pm=(w[..., None] * (Rz - r_pm[:, None]) ** 2).sum(1).sqrt(), s_pm=(w[..., None] * Sz ** 2).sum(1).sqrt(),
               mu=MU[ar, bm])
    if 'chi2' in models[0][0]:
        out['q_nll'] = out['q']; out['q'] = CH[ar, bm]
    return out


def kin_models(kin, o0, o1):
    """the model list of model_average for a kinetic mode ('average', 'none', 'kinetic')"""
    if kin not in KIN_MODES: raise ValueError(f'kin {kin!r}: one of {KIN_MODES}')
    return ([(o0, 0.0, False)] if kin != 'kinetic' else []) + ([(o1, KIN_PENALTY, True)] if kin != 'none' else [])


@torch.no_grad()
def fringe_posterior(I, Sig, r0, sp, pt, bases, Cj, nf=1, box=1.5, kin='average', iters=30, temp=1.0, dedup=0.15, cmult=0.0):
    """posterior over the fringe starts and the kinetic model (model_average): both models fitted for every row (kin
    'average'), or one of them ('none', 'kinetic'). Outputs: model_average"""
    dr = start_offsets(pt, r0, bases, nf)
    kw = dict(iters=iters, box=box, cmult=cmult)
    o0 = pattern_solve_all(I, Sig, r0, sp, pt, dr, Cj, kinetics=False, **kw) if kin != 'kinetic' else None
    o1 = pattern_solve_all(I, Sig, r0, sp, pt, dr, Cj, kinetics=True, **kw) if kin != 'none' else None
    return model_average(kin_models(kin, o0, o1), temp=temp, dedup=dedup)


# ------------------------------------------------------------------------------------------------ solve of the emitter-network outputs
@torch.no_grad()
def solve_emitters(o, pt_win, b, cal, nf=1, kin='average'):
    """explicit pattern solve on the emitter network's outputs with their joint covariance (prior position, per-frame intensities;
    pattern_solve_all): posterior over the fringe starts and the kinetic model (model_average; posterior-mean position), nuisance
    covariance c (brightness noise; solve_cal).
    Returns r [n,2] (posterior mean), s [n,2] (Fisher sd of the MAP start), s_cal [n,2] = the posterior sd sqrt(s_pm^2 +
    spread_pm^2) (mixture Fisher sd over the fringe starts + their spread; no calibration factor), q / dof, p, ...
    Reference implementation (test oracle of simcode/solve/solve_fast.solve_emitters_fast, which the pipeline uses).
    o: the emitter network's outputs I, L, C, x, y, lsxy. cal = solve_cal(...): prior_off (constant prior-position bias, px,
    removed), box (start box half-width, px), mix_T (posterior temperature) of the model's calibration + cmult (the brightness
    noise c of the dataset, estimated with the pattern).
    o['relg'] (optional, 2026-11-19): per-row factors [n, 2] of the angle-0 / angle-1 relative intensities (the pattern's
    intensity map at the row's position, pattern.relmap_factors); absent or None = the pattern as given."""
    I, Sig, r0, sp, ph, Cj = solve_inputs(o, cal)
    s = fringe_posterior(I, Sig, r0, sp, C.scale_rel(pt_win[b], o.get('relg')), [r0], Cj, nf=nf, box=cal['box'], kin=kin,
                         temp=cal['mix_T'], cmult=brightness_noise(cal))
    return finish(s, I, Sig, r0, sp)


def finish(s, I, Sig, r0, sp):
    """the outputs of solve_emitters / solve_fast.solve_emitters_fast from the fringe posterior s"""
    qd = s['q'] / s['dof'].clamp(min=1)
    s_cal = (s['s_pm'] ** 2 + s['spread_pm'] ** 2).sqrt()
    return dict(R_starts=s['R'], w_starts=s['w'], r=s['r_pm'], s=s['s'], s_cal=s_cal, s_pm=s['s_pm'], spread_pm=s['spread_pm'], q=qd,
                p=C.chi2_sf(s['q'], s['dof']), kin=s['kin'], kin_map=s['kin_map'], r_map=s['r'], dnll=s['dnll'], r_prior=r0, s_prior=sp,
                mu=s['mu'], Sig=Sig, dof=s['dof'], chi2=s['q'])


BN_KEY = 'cmult'
REMOVED_CAL_KEYS = ('cmult', 'cmult_ang')      # a fixed brightness noise in a model's calibration file (removed 2026-11-17)


def solve_cal(cal, c):
    """the solve calibration of a dataset: the model's calibration (path or dict: prior_off, box, mix_T) + the brightness noise
    c (cmult: the relative per-frame scatter of the emitter brightness, Sigma = Sigma_net + c^2 diag(mu^2)) of the dataset,
    estimated with its pattern (processing/pattern_estimation.estimate_pattern -> pattern_est.pt 'cmult'; 2026-11-17).
    A model calibration that still holds a fixed brightness noise is refused (removed 2026-11-17)."""
    import json
    d = json.load(open(cal)) if isinstance(cal, str) else dict(cal)
    old = [k for k in REMOVED_CAL_KEYS if k in d]
    if old: raise ValueError(f'solve calibration {cal if isinstance(cal, str) else ""}: {old} is a fixed brightness noise, removed '
                             '2026-11-17 (c is estimated per dataset with the pattern); delete these keys')
    d[BN_KEY] = float(c)
    return d


def pattern_cal(cal, pattern):
    """solve_cal with the brightness noise stored in a pattern (pattern_est.pt dict: 'cmult'). A pattern of an earlier run
    may also hold 'cmult_ang' (the per-orientation term c_a, removed 2026-11-19): it is ignored with a warning."""
    if BN_KEY not in pattern:
        raise KeyError('the pattern has no brightness noise (cmult): pattern_est.pt of a run before 2026-11-17; re-run the pattern stage')
    if float(pattern.get('cmult_ang', 0.0) or 0.0) > 0:
        import warnings
        warnings.warn(f"pattern_est.pt with the per-orientation brightness noise c_a = {float(pattern['cmult_ang']):.3f} (cmult_ang; removed "
                      "2026-11-19): ignored, the solve uses c only (its c was estimated together with c_a; re-run the pattern stage "
                      "for the c of the current estimator)", stacklevel=2)
    return solve_cal(cal, float(pattern[BN_KEY]))


def brightness_noise(cal):
    """c of a solve calibration made by solve_cal / pattern_cal"""
    if BN_KEY not in cal:
        raise KeyError('solve calibration without the brightness noise c: build it with simcode.solve.solve.solve_cal / pattern_cal')
    return float(cal[BN_KEY])


def solve_inputs(o, cal, mm=torch.matmul):
    """emitter-network outputs + calibration -> (I [n,T], Sig [n,T,T] = L L^T + 1e-3 (the intensity covariance given the prior
    error), prior position r0 [n,2] (prior_off removed), prior sd sp [n,2] = exp(lsxy), window photons ph [n], Cj [n,T,2]
    (the joint covariance block, EmitterNet)); shared by solve_emitters and solve_fast.solve_emitters_fast. The network's own
    covariances, no scale factors (the per-photon-bin kap_p / kap_I2 were removed 2026-11-17)."""
    I = o['I'].float(); L = o['L'].float()
    ph = I.sum(1)
    Sig = mm(L, L.transpose(1, 2)) + 1e-3 * torch.eye(I.shape[1], device=I.device)
    r0 = torch.stack([o['x'], o['y']], 1).float() - torch.tensor(cal['prior_off'], device=I.device, dtype=torch.float32)[None]
    sp = o['lsxy'].float().exp()
    return I, Sig, r0, sp, ph, o['C'].float()
