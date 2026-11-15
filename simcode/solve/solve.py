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
import torch
from simcode.solve import pattern as C

T = 6


def pattern_solve_all(I, Sig, r0, sp, pt, dr, kinetics=True, iters=30, box=0.75,
                      kin_starts=((0.0, 0.0), (-0.6, 0.6), None), cmult=0.0, cang=0.0):
    # cmult (2026-11-09, critic 32 #5): nuisance-aware intensity covariance Sigma(p) = Sig + diag((cmult mu_t(p))^2) with the
    # CURRENT model mean mu_t (multiplicative per-frame emitter-brightness noise; not the observed I_t), re-whitened at every
    # LM evaluation; cang: + a per-angle common factor, Sigma += cang^2 mu mu^T within each angle's frames (GLS Jacobian: d Sigma / d p ignored; the cost is the exact Gaussian NLL incl. log det Sigma(p) - log det
    # Sig). Returned 'nll' = that cost (start / kinetic selection), 'chi2' = the whitened residual + prior part.
    n, T_ = I.shape; dev = I.device
    Km, ang = C.angle_k(pt)
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
    Ir, r0r, spr, ptr, Lr, angr = (rep(t_) for t_ in (I, r0, sp, pt, Linv, ang))
    cmult = float(cmult); cang = float(cang)
    if cang > 0 and cmult <= 0: cmult = 1e-6
    if cmult > 0:
        Sigr = rep(Sig); ld0 = 2 * torch.log(rep(Lc).diagonal(dim1=1, dim2=2).clamp(min=1e-12)).sum(1)
        eyeT = torch.eye(T_, device=dev)
    M = len(st)
    oh = torch.nn.functional.one_hot(angr, 2).float()
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
    free = torch.tensor([1, 1, 1, 0.0, float(kinetics), float(kinetics)], device=dev)
    fixed_diag = torch.diag_embed((1 - free).expand(M, NP))

    def clamp(p):
        x = torch.maximum(torch.minimum(p[:, :2], hi_xy), lo_xy)
        B = p[:, 2:4].clamp(min=1.0)[:, :1].expand(-1, 2)
        t_on = p[:, 4].clamp(-1, T_ - 0.25); t_off = torch.maximum(p[:, 5], t_on + 0.25).clamp(max=T_ + 1)
        return torch.cat([x, B, t_on[:, None], t_off[:, None]], 1)

    def rj(p):
        il, dx, dy = C.illum_at(ptr, p[:, 0], p[:, 1])
        a_, d0, d1 = C.onfrac(p[:, 4], p[:, 5], T_)
        Bt = (oh * p[:, None, 2:4]).sum(-1)
        mu = Bt * a_ * il
        dB0, dB1 = a_ * il, torch.zeros_like(il)
        Jm = torch.stack([Bt * a_ * dx, Bt * a_ * dy, dB0, dB1, Bt * il * d0, Bt * il * d1], -1) * free
        ldx = 0.0
        if cmult > 0:
            mup = mu.clamp(min=0)
            Sp = Sigr + torch.diag_embed((cmult * mup) ** 2) + 1e-6 * eyeT
            if cang > 0: Sp = Sp + cang ** 2 * (mup[:, :, None] * mup[:, None, :]) * (oh @ oh.transpose(1, 2))
            Lp, _ = torch.linalg.cholesky_ex(Sp)
            Lri = torch.linalg.solve_triangular(Lp, eyeT.expand(M, T_, T_), upper=False)
            ldx = 2 * torch.log(Lp.diagonal(dim1=1, dim2=2).clamp(min=1e-12)).sum(1) - ld0
        else:
            Lri = Lr
        rI = (Lri @ (Ir - mu)[..., None])[..., 0]
        JI = -(Lri @ Jm)
        rp = (p[:, :2] - r0r) / spr
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
    out['mu'] = ((oh * p[:, None, 2:4]).sum(-1) * a_f * il_).view(n, ns, T_)[ar, idx]
    if cmult > 0: out['chi2'] = torch.nan_to_num(chi2, nan=1e30).view(n, ns)[ar, idx]
    return out


def start_offsets(pt, r0, bases, nf=1):
    """dr [n,H,2]: for each base point [n,2], base - r0 + the fringe shifts (nf)"""
    Km, _ = C.angle_k(pt)
    fs = C.fringe_shifts(Km, nf)                                             # [n,(2nf+1)^2,2]
    return torch.cat([(b - r0)[:, None] + fs for b in bases], 1)


@torch.no_grad()
def fringe_posterior(I, Sig, r0, sp, pt, bases, nf=1, box=1.5, thr=9.21, iters=30, temp=1.0, dedup=0.15, cmult=0.0, cang=0.0):
    """posterior over the fringe starts (iteration 31, critic 30 #1): kinetic model selection (kinetics only if it lowers the
    cost by > thr), MAP = best start of the selected kinetic model, plus the per-start solutions of that model, deduplicated (a start whose solution lies within `dedup`
    px of a lower-cost start is the same mode), weighted w_m ~ exp(-(cost_m - cost_min) / (2 temp)) (cost = chi^2 incl.
    the prior = -2 log L). Returns dict(r, s, q, dof, kin: the MAP; dnll = (cost_2nd - cost_min) / 2 of the
    best distinct competitor (inf if none); R [n,H,2], S [n,H,2], w [n,H] (0 for duplicates); r_pm posterior mean,
    spread [n,2] = sqrt(sum w (R - r_map)^2) per axis, spread_pm = sqrt(sum w (R - r_pm)^2))"""
    dr = start_offsets(pt, r0, bases, nf)
    o0 = pattern_solve_all(I, Sig, r0, sp, pt, dr, kinetics=False, iters=iters, box=box, cmult=cmult, cang=cang)
    o1 = pattern_solve_all(I, Sig, r0, sp, pt, dr, kinetics=True, iters=iters, box=box, cmult=cmult, cang=cang)
    n = len(I); ar = torch.arange(n, device=I.device)
    b0 = o0['nll'].argmin(1); b1 = o1['nll'].argmin(1)
    q0 = o0['nll'][ar, b0]; q1 = o1['nll'][ar, b1]; use = (q0 - q1) > thr
    R = torch.where(use[:, None, None], o1['r'], o0['r']); S = torch.where(use[:, None, None], o1['s'], o0['s'])
    Q = torch.where(use[:, None], o1['nll'], o0['nll'])
    bm = torch.where(use, b1, b0)
    r_map = R[ar, bm]; s_map = S[ar, bm]; q = Q[ar, bm]
    # dedup: sort by cost; a start is a duplicate if a lower-cost start's solution is within dedup px
    Qs, order = Q.sort(1)
    Rs = R.gather(1, order[..., None].expand(-1, -1, 2)); Ss = S.gather(1, order[..., None].expand(-1, -1, 2))
    Dm = torch.cdist(Rs, Rs)
    H = Q.shape[1]
    lower = torch.tril(torch.ones(H, H, device=I.device, dtype=torch.bool), -1)        # [h, h'] h' < h
    dup = ((Dm < dedup) & lower[None]).any(2)
    Qm = torch.where(dup, torch.full_like(Qs, float('inf')), Qs)
    w = torch.exp(-(Qm - Qs[:, :1]) / (2 * temp)); w = w / w.sum(1, keepdim=True)
    sec = torch.where(dup[:, 1:], torch.full_like(Qs[:, 1:], float('inf')), Qs[:, 1:]).min(1).values
    dnll = (sec - Qs[:, 0]) / 2
    r_pm = (w[..., None] * Rs).sum(1)
    spread = (w[..., None] * (Rs - r_map[:, None]) ** 2).sum(1).sqrt()
    spread_pm = (w[..., None] * (Rs - r_pm[:, None]) ** 2).sum(1).sqrt()
    out = dict(r=r_map, s=s_map, q=q, dof=torch.where(use, o1['dof'], o0['dof']), kin=use, dnll=dnll, R=Rs, S=Ss, w=w,
               r_pm=r_pm, spread=spread, spread_pm=spread_pm, s_pm=(w[..., None] * Ss ** 2).sum(1).sqrt())
    out['mu'] = torch.where(use[:, None, None], o1['mu'], o0['mu'])[ar, bm]      # model mean at the MAP (automatic c)
    if cmult > 0 or cang > 0:     # q = the chi2 part at the MAP start (the nll used for selection / weights includes log det)
        out['q_nll'] = q
        out['q'] = torch.where(use[:, None], o1['chi2'], o0['chi2'])[ar, bm]
    return out


# ------------------------------------------------------------------------------------------------ solve of the emitter-network outputs
@torch.no_grad()
def solve_emitters(o, pt_win, b, cal, nf=1, thr=9.21):
    """explicit pattern solve on the emitter network's intensities with their full covariance (simcode/solve/solve.py): fringe-start
    posterior (posterior-mean position with the mixture sigma), kinetic model selection, nuisance covariance Sigma(mu).
    Returns r [n,2], s [n,2] (Fisher), s_cal (chi2-aware, calibrated), q / dof, p. Reference implementation (test oracle of
    simcode/solve/solve_fast.solve_emitters_fast, which the pipeline uses).
    Calibration keys (all required): kap_bins {'ph', 'kap_p', 'kap_I2'} (per-photon-bin prior / intensity covariance scales,
    log-photon interpolation, constant outside), prior_off (constant prior-position bias, px, removed), cmult / cmult_ang
    (nuisance covariance c, c_a), box (start box half-width, px), mix_T (posterior temperature), sig_a / sig_g (chi2-aware
    sigma), sig_ph {'ph', 'f'} (per-photon-bin sigma factor)."""
    I, Sig, r0, sp, ph = solve_inputs(o, cal)
    s = fringe_posterior(I, Sig, r0, sp, pt_win[b], [r0], nf=nf, box=cal['box'], thr=thr, temp=cal['mix_T'],
                            cmult=float(cal['cmult']), cang=float(cal['cmult_ang']))
    qd = s['q'] / s['dof'].clamp(min=1)
    f = (cal['sig_a'] * qd.clamp(min=1.0) ** cal['sig_g'])[:, None]
    s_cal = (f * s['s_pm'] ** 2 + s['spread_pm'] ** 2).sqrt() * _binval(cal['sig_ph'], 'f', ph)[:, None]
    return dict(r=s['r_pm'], s=s['s'], s_cal=s_cal, q=qd, p=C.chi2_sf(s['q'], s['dof']), kin=s['kin'], r_map=s['r'],
                s_map_cal=s['s'] * f.sqrt(), dnll=s['dnll'], r_prior=r0, s_prior=sp,
                mu=s['mu'], Sig=Sig, dof=s['dof'], chi2=s['q'])


def solve_inputs(o, cal, mm=torch.matmul):
    """head outputs + calibration -> (I [n,T], Sig [n,T,T] = L L^T kap_I2 + 1e-3, prior position r0 [n,2] (prior_off removed),
    prior sigma sp [n,2] = exp(lsxy) kap_p, window photons ph [n]); shared by solve_emitters and solve_fast.solve_emitters_fast"""
    I = o['I'].float(); L = o['L'].float()
    ph = I.sum(1)
    kI2 = _binval(cal['kap_bins'], 'kap_I2', ph)[:, None, None]
    kp = _binval(cal['kap_bins'], 'kap_p', ph)[:, None]
    Sig = mm(L, L.transpose(1, 2)) * kI2 + 1e-3 * torch.eye(I.shape[1], device=I.device)
    r0 = torch.stack([o['x'], o['y']], 1).float() - torch.tensor(cal['prior_off'], device=I.device, dtype=torch.float32)[None]
    sp = o['lsxy'].float().exp() * kp
    return I, Sig, r0, sp, ph


def _binval(tab, key, ph):
    """per-photon-bin calibration value: tab['ph'] bin centres (network window photons), tab[key] values; piecewise
    linear in log photons, constant beyond the end bins -> tensor [n]"""
    xp = torch.log(torch.tensor(tab['ph'], device=ph.device, dtype=torch.float32))
    fp = torch.tensor(tab[key], device=ph.device, dtype=torch.float32)
    x = torch.log(ph.float().clamp(min=1.0)).clamp(xp[0], xp[-1])
    i = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    w = (x - xp[i - 1]) / (xp[i] - xp[i - 1]).clamp(min=1e-9)
    return fp[i - 1] + w * (fp[i] - fp[i - 1])
