"""Automatic per-dataset (scale, c, c_a) from the explicit solve's own intensity-domain residuals (critic 33 #2).
GT-free, and pattern-free beyond the pattern the solve already uses; the same code for sim movies, gate scenes and real data.

Noise model of a window's network intensities around the model mean mu(p) (T = 6 frames):
    Sigma(theta) = s * Sigma_net + c^2 diag(mu^2) + c_a^2 (mu mu^T within each angle)            theta = (s, c, c_a)
(s = global scale of the network covariance incl. the kap_I2 bins, applied here as L -> sqrt(s) L; c = cal 'cmult', c_a = cal
'cmult_ang': exactly the Sigma(p) of msolve.pattern_solve_all.)
Residuals e = I_net - mu(p_hat) at the solve's MAP. Non-kinetic, isolated rows (no other pass-1 detection within R_ISO px in the
same window). For a GLS fit of p = (x, y, B) with the Gaussian position prior (precision P = diag(1/s_p^2), 0 for B) the
residual covariance is exactly V_e(theta) = Sigma - J (J^T Sigma^-1 J + P)^-1 J^T (J = d mu / d p at p_hat), so
  per frame      r_t = e_t / sqrt(V_e,tt)                     ~ N(0, 1)   -> PIT u = F_chi2_1(r_t^2)
  angle contrast h = w^T e / sqrt(w^T V_e w), w = +-1 by angle ~ N(0, 1)   (the within-angle phase sum is ~position-free and
                 B is shared, so h is the per-angle brightness ratio: it splits c_a from c)
binned by the frame SNR^2 = mu_t^2 / Sigma_net,tt (low SNR pins s, high SNR pins c^2 + c_a^2) and the window SNR^2 of the
contrast; theta minimises sum_bins wt (median u - 1/2)^2 (robust to FP / merged tails) at the fixed solutions; the solve is
re-run with theta and the fit repeated (fixed point, 3 iterations). Analysis estimator (2026-11-14: no longer a pipeline stage): run_dir(<pipeline output dir>) -> theta; make_calib(theta) -> a solve calibration.
The 'q/dof' of the solve uses dof = T + 2 - n_free
(5 = 6 intensities + 2 prior pseudo-observations - (x, y, B); 3 with the kinetic on/off times)."""
import json, os, glob
import numpy as np, torch
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from simcode.util import spatial_hash as SH

T = 6
PHB = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9)]
FIT_LO = 400        # the 200-400 bin (FP / selection-dominated) is reported but not used in the fit
R_ISO = 2.0
dev = 'cuda'
tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)


# ------------------------------------------------------------------------------------------------------------- sources
ISO_BACKEND = 'hash'      # (2026-11-11, speed branch) 'hash' = simcode/util/spatial_hash (exact count); 'kdtree' = scipy reference


def iso_flag(key, x, y, r=R_ISO):
    """True if no other point with the same key (window / FOV) within r px"""
    if ISO_BACKEND == 'hash':
        x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); key = np.asarray(key).astype(np.int64)
        return SH.count(x, y, key, x, y, key, r) <= 1
    P = np.c_[x, y, key.astype(np.float64) * 1e4]
    tr = cKDTree(P)
    cnt = tr.query_ball_point(P, r, return_length=True)
    return cnt <= 1


def pipeline_rows(d, chunks=None, nmax=120000, seed=0, pattern=None):
    """rows of a pipeline output dir (pass1_<f0>.npz + pattern_est.pt): the 'out' rows as run_chunk_pass2 solves them.
    Returns a list of (o-dict, PW, b, ph, iso) per chunk (subsampled to ~nmax rows in total)."""
    from simcode.pipeline.common import pattern_frames
    fs = sorted(glob.glob(f'{d}/pass1_*.npz'), key=lambda f: int(f.split('_')[-1][:-4]))
    if chunks is not None: fs = [fs[i] for i in chunks if i < len(fs)]
    Pt = torch.load(pattern or f'{d}/pattern_est.pt')
    rng = np.random.default_rng(seed); per = nmax // len(fs)
    out = []
    for f in fs:
        R = np.load(f)
        X0, Y0 = int(R['X0']), int(R['Y0'])
        iso_all = iso_flag(R['wstart'], R['x_det'], R['y_det'])
        keep = np.nonzero(R['out'].astype(bool))[0]
        if len(keep) > per: keep = np.sort(rng.choice(keep, per, replace=False))
        ws = R['wstart'][keep]
        uw, inv = np.unique(ws, return_inverse=True)
        PW = pattern_frames(Pt, (uw[:, None] + np.arange(T)[None]).reshape(-1), X0, Y0, dev).reshape(len(uw), T, 5)
        o = dict(I=tt(R['I'][keep]), L=tt(R['L'][keep]), x=tt(R['x'][keep] - X0), y=tt(R['y'][keep] - Y0),
                 lsxy=torch.log(tt(np.c_[R['sx'][keep], R['sy'][keep]])), x_det=tt(R['x_det'][keep] - X0), y_det=tt(R['y_det'][keep] - Y0))
        out.append((o, PW, torch.tensor(inv, device=dev), R['I'][keep].sum(1), iso_all[keep]))
    return out


# ------------------------------------------------------------------------------------------------------------- solve
def with_theta(cal, th):
    c = dict(cal); c['_s'] = float(th[0]); c['cmult'] = float(th[1]); c['cmult_ang'] = float(th[2])   # _s: applied to L here (solve_rows)
    return c


@torch.no_grad()
def solve_rows(rows, cal, chunk=8192, keep_all=False, fast=True):
    """pattern solve on all rows; per-row residual quantities at the MAP (torch, concatenated). Only non-kinetic isolated
    rows are kept for the estimator unless keep_all. fast: msolve_fast.sc_solve_fast (exact restructuring) instead of
    sc.sc_solve."""
    from simcode.hd import sc as SCM, msolve_fast as MF
    from simcode import core as C
    solve = MF.sc_solve_fast if fast else SCM.sc_solve
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(v)
    for (o, PW, b, ph, iso) in rows:
        n = len(o['I'])
        for c0 in range(0, n, chunk):
            sl = slice(c0, c0 + chunk)
            oo = {k: v[sl] for k, v in o.items()}
            sc_ = float(cal.get('_s', 1.0))                          # covariance scale s: L -> sqrt(s) L
            s = solve(dict(oo, L=oo['L'] * sc_ ** 0.5), PW, b[sl], cal)
            Sn = s['Sig'] / sc_                                       # network part at scale 1 (incl. the kap_I2 bins)
            pt = PW[b[sl]]
            il, dx, dy = C.illum_at(pt, s['r_map'][:, 0], s['r_map'][:, 1])
            Bh = s['mu'].sum(1) / il.sum(1).clamp(min=1e-6)
            J = torch.stack([Bh[:, None] * dx, Bh[:, None] * dy, il], -1)
            kin = s['kin'].bool()
            m = torch.ones_like(kin) if keep_all else (~kin & torch.tensor(iso[sl], device=dev))
            for k, v in dict(e=oo['I'] - s['mu'], mu=s['mu'], Sn=Sn, J=J, sp=s['s_prior'], ang=C.angle_k(pt)[1],
                             ph=tt(ph[sl]), q=s['chi2'], dof=s['dof'], kin=kin).items():
                add(k, v[m])
            add('n_all', torch.tensor([n if c0 == 0 else 0], device=dev)); add('n_kin', kin.sum()[None])
    return {k: torch.cat(v) for k, v in acc.items()}


def sigma_theta(S, th):
    s, c, ca = th
    mu = S['mu'].clamp(min=0)
    Sg = s * S['Sn'] + torch.diag_embed((c * mu) ** 2) + 1e-6 * torch.eye(T, device=dev)
    if ca > 0:
        oh = torch.nn.functional.one_hot(S['ang'], 2).float()
        Sg = Sg + ca ** 2 * (mu[:, :, None] * mu[:, None, :]) * (oh @ oh.transpose(1, 2))
    return Sg


def resid_cov(S, th):
    """V_e = Sigma - J (J^T Sigma^-1 J + P)^-1 J^T"""
    Sg = sigma_theta(S, th)
    J = S['J']
    Si_J = torch.linalg.solve(Sg, J)
    A = J.transpose(1, 2) @ Si_J
    P = torch.diag_embed(torch.stack([1 / S['sp'][:, 0] ** 2, 1 / S['sp'][:, 1] ** 2, torch.zeros_like(S['sp'][:, 0])], 1))
    A = A + P + 1e-9 * torch.eye(3, device=dev)
    return Sg - J @ torch.linalg.solve(A, J.transpose(1, 2))


def wvec(S):
    return torch.where(S['ang'] == 0, 1.0, -1.0)


def pit(S, th):
    """per-frame u [n,T] and contrast u [n]"""
    V = resid_cov(S, th)
    v = V.diagonal(dim1=1, dim2=2).clamp(min=1e-6)
    uF = torch.special.gammainc(torch.full_like(v, 0.5), S['e'] ** 2 / v / 2)
    w = wvec(S)
    vh = (w[:, None, :] @ V @ w[:, :, None])[:, 0, 0].clamp(min=1e-6)
    uC = torch.special.gammainc(torch.full_like(vh, 0.5), (w * S['e']).sum(1) ** 2 / vh / 2)
    return uF, uC


QB = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95, 0.99, 0.999, 1.0]


def snr_bins(S):
    """bin edges in log SNR^2 at the quantiles QB (equal-count bins + 3 upper-tail bins above the 87.5 / 95 / 99 / 99.9 %
    points, so that the high-SNR end, which pins c, has its own bins; bins with < 200 entries are ignored), frames and windows"""
    snrF = S['mu'].clamp(min=0) ** 2 / S['Sn'].diagonal(dim1=1, dim2=2).clamp(min=1e-6)
    w = wvec(S)
    snrC = S['mu'].clamp(min=0).sum(1) ** 2 / (w[:, None, :] @ S['Sn'] @ w[:, :, None])[:, 0, 0].clamp(min=1e-6)
    lF, lC = torch.log(snrF.clamp(min=1e-3)), torch.log(snrC.clamp(min=1e-3))
    qq = torch.tensor(QB, device=dev)
    eF = torch.quantile(lF.flatten()[::max(1, lF.numel() // 500000)], qq)
    eC = torch.quantile(lC[::max(1, lC.numel() // 500000)], qq)
    return lF, eF, lC, eC


def bin_medians(S, th, B=None):
    lF, eF, lC, eC = B if B is not None else snr_bins(S)
    uF, uC = pit(S, th)
    rF = []; rC = []
    for i in range(len(eF) - 1):
        m = (lF >= eF[i]) & (lF <= eF[i + 1])
        rF.append((float(eF[i:i + 2].mean().exp().sqrt()), int(m.sum()), float(uF[m].median()) if m.sum() >= 200 else np.nan))
        m = (lC >= eC[i]) & (lC <= eC[i + 1])
        rC.append((float(eC[i:i + 2].mean().exp().sqrt()), int(m.sum()), float(uC[m].median()) if m.sum() >= 200 else np.nan))
    return rF, rC


def fit_theta(S, th0, nmax=60000):
    if len(S['e']) > nmax:
        g = torch.Generator(device='cpu').manual_seed(0)
        ii = torch.randperm(len(S['e']), generator=g)[:nmax].to(dev)
        S = {k: (v[ii] if (torch.is_tensor(v) and v.dim() > 0 and len(v) == len(S['e'])) else v) for k, v in S.items()}
    B = snr_bins(S)
    def f(x):
        th = (float(np.exp(x[0])), abs(float(x[1])), abs(float(x[2])))
        rF, rC = bin_medians(S, th, B)
        return sum((m - 0.5) ** 2 for _, _, m in rF if np.isfinite(m)) + sum((m - 0.5) ** 2 for _, _, m in rC if np.isfinite(m))
    best = None
    for x0 in ([np.log(th0[0]), th0[1], th0[2]], [0.0, 0.05, 0.05], [0.0, 0.2, 0.1], [np.log(0.8), 0.0, 0.0]):
        r = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-8, maxiter=400))
        if best is None or r.fun < best.fun: best = r
    x = best.x
    return (float(np.exp(x[0])), abs(float(x[1])), abs(float(x[2]))), float(best.fun), B


def estimate(rows, cal0, n_iter=3, th0=(1.0, 0.10, 0.10), log=print, fast=True):
    """fixed-point estimate of theta = (s, c, c_a); returns theta, history, the last residual set"""
    th = tuple(th0); hist = []
    for it in range(n_iter):
        S = solve_rows(rows, with_theta(cal0, th), fast=fast)
        th_new, J, B = fit_theta(S, th)
        rF, rC = bin_medians(S, th_new)
        nfr = int(len(S['e']))
        hist.append(dict(it=it, theta_in=th, theta=th_new, J=J, n_rows=nfr, n_all=int(S['n_all'].sum()), n_kin=int(S['n_kin'].sum()),
                         frame_bins=rF, contrast_bins=rC))
        log(f'  it {it}: theta_in s {th[0]:.3f} c {th[1]:.3f} ca {th[2]:.3f} -> s {th_new[0]:.3f} c {th_new[1]:.3f} ca {th_new[2]:.3f} '
            f'(J {J:.1e}, rows {nfr}) med u frame [SNR]: ' + ' '.join(f'{m:.2f}[{x:.0f}]' for x, _, m in rF) +
            ' | contrast: ' + ' '.join(f'{m:.2f}[{x:.0f}]' for x, _, m in rC))
        th = th_new
    return th, hist, S


def q_table(rows, cal, fast=True):
    """q/dof p50 per window-photon bin (all rows) for a calibration"""
    S = solve_rows(rows, cal, keep_all=True, fast=fast)
    qd = S['q'] / S['dof']; ph = S['ph']
    return [(lo, float(qd[(ph >= lo) & (ph < hi)].median()) if ((ph >= lo) & (ph < hi)).sum() > 30 else np.nan) for lo, hi in PHB]


# ------------------------------------------------------------------------------------------------------------- pipeline use
def make_calib(theta, table=None, src=''):
    """solve calibration with the automatic theta: the autoc sigma table (models/calib/calib_sc_hd_sc5nbb15_pm_autoc1.json)
    with cmult = c, cmult_ang = c_a; the Sigma scale stays 1 (s is a nuisance of the fit, not applied)"""
    from simcode import config
    c = json.load(open(config.model(table or config.CAL_P1_AUTOC)))
    s, cm, ca = theta
    c.update(cmult=float(cm), cmult_ang=float(ca), autoc_theta=[float(s), float(cm), float(ca)], autoc_from=src)
    return c


def run_dir(d, nmax=120000, n_chunks=5, cal0=None, n_iter=3, log=print, fast=True):
    """automatic (s, c, c_a) of a pipeline output dir (pass1_<f0>.npz + pattern_est.pt): rows of n_chunks chunks spread over
    the movie (<= nmax rows), started at the sim-sigma calibration cal0 (default models/calib/calib_sc_hd_sc5nbb15_pm_bright.json),
    n_iter fixed-point iterations. Returns dict(theta, hist, q50, n_rows)."""
    from simcode import config
    cal0 = cal0 or config.model(config.CAL_P1_AUTOC_START)
    cal0 = json.load(open(cal0)) if isinstance(cal0, str) else dict(cal0)
    nch = len(glob.glob(d + '/pass1_*.npz'))
    rows = pipeline_rows(d, chunks=sorted(set(np.linspace(0, nch - 1, min(n_chunks, nch)).round().astype(int).tolist())), nmax=nmax)
    th, hist, _ = estimate(rows, cal0, n_iter=n_iter, log=log, fast=fast)
    qt = q_table(rows, with_theta(cal0, th), fast=fast)
    return dict(theta=th, hist=hist, q50=qt, n_rows=int(sum(len(r[3]) for r in rows)), nmax=nmax)
