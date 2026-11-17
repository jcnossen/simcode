"""Brightness noise (c, c_a) of a dataset from the explicit solve's own intensity-domain residuals (critic 33 #2; 2026-11-17: a
stage of the pattern estimation, processing/pattern_estimation.estimate_brightness_noise). GT-free; the same code for simulated and
real movies. c and c_a are, besides the PSF / camera calibration and the pattern, the only values taken from the data.

Noise model of a window's network intensities around the model mean mu(p) (T = 6 frames):
    Sigma(theta) = Sigma_net + c^2 diag(mu^2) + c_a^2 (mu mu^T within each angle)            theta = (c, c_a)
(Sigma_net = the emitter network's own intensity covariance, no scale factor; c = the solve's 'cmult', c_a = 'cmult_ang':
exactly the Sigma(p) of solve.pattern_solve_all.) c_a is one value shared by the two pattern orientations: with the brightness B
fitted per window only the sum of the two orientations' terms is identified (the contrast below), not a split per angle.
Rows: the clean emitter-windows of the pattern estimation (on through the window, isolated, >= 300 photons), solved with the
non-kinetic model (they are on through the window). Residuals e = I_net - mu(p_hat) at the solve's MAP. For a GLS fit of
p = (x, y, B) with the Gaussian position prior (precision P = diag(1/s_p^2), 0 for B) the residual covariance is exactly
V_e(theta) = Sigma - J (J^T Sigma^-1 J + P)^-1 J^T (J = d mu / d p at p_hat), so
  per frame      r_t = e_t / sqrt(V_e,tt)                     ~ N(0, 1)   -> PIT u = F_chi2_1(r_t^2)
  angle contrast h = w^T e / sqrt(w^T V_e w), w = +-1 by angle ~ N(0, 1)   (the within-angle phase sum is ~position-free and
                 B is shared, so h is the per-angle brightness ratio: it splits c_a from c)
binned by the frame SNR^2 = mu_t^2 / Sigma_net,tt (high SNR pins c^2 + c_a^2) and the window SNR^2 of the contrast; theta
minimises sum_bins (median u - 1/2)^2 (robust to FP / merged tails) at the fixed solutions; the solve is re-run with theta and
the fit repeated (fixed point, 3 iterations).
(Until 2026-11-17 the estimator also fitted a global scale s of the network covariance and was an analysis tool only; the
pipeline used a fixed c = c_a = 0.1 from the model's calibration files. Both removed.)
The 'q/dof' of the solve uses dof = T + 2 - n_free
(5 = 6 intensities + 2 prior pseudo-observations - (x, y, B); 3 with the kinetic on/off times)."""
import numpy as np, torch
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from simcode.util import spatial_hash as SH

T = 6
PHB = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9)]
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


def chunk_rows(d, pattern, rows_of_chunk):
    """estimator rows from the pass-1 chunk files of a pipeline output dir: rows_of_chunk {f0: row indices into pass1_<f0>.npz},
    pattern = pattern dict (pattern_est.pt). Returns a list of (o-dict, PW, b, ph, iso) per chunk, as the solve of the pass-1
    rows takes them (inference_fast.run_chunk_pattern_solve_fast)."""
    from simcode.processing.common import pattern_frames
    out = []
    for f0, keep in sorted(rows_of_chunk.items()):
        keep = np.sort(np.asarray(keep, np.int64))
        if not len(keep): continue
        R = np.load(f'{d}/pass1_{f0}.npz')
        X0, Y0 = int(R['X0']), int(R['Y0'])
        ws = R['wstart'][keep]
        uw, inv = np.unique(ws, return_inverse=True)
        PW = pattern_frames(pattern, (uw[:, None] + np.arange(T)[None]).reshape(-1), X0, Y0, dev).reshape(len(uw), T, 5)
        o = dict(I=tt(R['I'][keep]), L=tt(R['L'][keep]), C=tt(R['C'][keep]), x=tt(R['x'][keep] - X0), y=tt(R['y'][keep] - Y0),
                 lsxy=torch.log(tt(np.c_[R['sx'][keep], R['sy'][keep]])), x_det=tt(R['x_det'][keep] - X0), y_det=tt(R['y_det'][keep] - Y0))
        out.append((o, PW, torch.tensor(inv, device=dev), R['I'][keep].sum(1), np.ones(len(keep), bool)))
    return out


# ------------------------------------------------------------------------------------------------------------- solve
def with_theta(cal, th):
    """the model's solve calibration (no c) + theta = (c, c_a)"""
    from simcode.solve import solve as SV
    return SV.solve_cal({k: v for k, v in cal.items() if k not in SV.BN_KEYS}, th[0], th[1])


@torch.no_grad()
def solve_rows(rows, cal, chunk=8192, keep_all=False, fast=True):
    """non-kinetic pattern solve on all rows (cal from with_theta); per-row residual quantities at the MAP (torch, concatenated).
    Only the rows flagged iso are kept unless keep_all. fast: solve_fast.solve_emitters_fast instead of solve.solve_emitters."""
    from simcode.solve import solve_fast as MF
    from simcode.solve import solve as SV
    from simcode.solve import pattern as C
    solve = MF.solve_emitters_fast if fast else SV.solve_emitters
    acc = {}
    add = lambda k, v: acc.setdefault(k, []).append(v)
    for (o, PW, b, ph, iso) in rows:
        n = len(o['I'])
        for c0 in range(0, n, chunk):
            sl = slice(c0, c0 + chunk)
            oo = {k: v[sl] for k, v in o.items()}
            s = solve(oo, PW, b[sl], cal, kin='none')
            pt = PW[b[sl]]
            il, dx, dy = C.illum_at(pt, s['r_map'][:, 0], s['r_map'][:, 1])
            Bh = s['mu'].sum(1) / il.sum(1).clamp(min=1e-6)
            J = torch.stack([Bh[:, None] * dx, Bh[:, None] * dy, il], -1)
            m = torch.ones(len(Bh), dtype=torch.bool, device=dev) if keep_all else torch.tensor(iso[sl], device=dev)
            for k, v in dict(e=oo['I'] - s['mu'], mu=s['mu'], Sn=s['Sig'], J=J, sp=s['s_prior'], ang=C.angle_k(pt)[1],
                             ph=tt(ph[sl]), q=s['chi2'], dof=s['dof']).items():
                add(k, v[m])
            add('n_all', torch.tensor([n if c0 == 0 else 0], device=dev))
    return {k: torch.cat(v) for k, v in acc.items()}


def sigma_theta(S, th):
    c, ca = th
    mu = S['mu'].clamp(min=0)
    Sg = S['Sn'] + torch.diag_embed((c * mu) ** 2) + 1e-6 * torch.eye(T, device=dev)
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
        th = (abs(float(x[0])), abs(float(x[1])))
        rF, rC = bin_medians(S, th, B)
        return sum((m - 0.5) ** 2 for _, _, m in rF if np.isfinite(m)) + sum((m - 0.5) ** 2 for _, _, m in rC if np.isfinite(m))
    best = None
    for x0 in ([th0[0], th0[1]], [0.05, 0.05], [0.2, 0.1], [0.1, 0.0]):
        r = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-8, maxiter=400))
        if best is None or r.fun < best.fun: best = r
    x = best.x
    return (abs(float(x[0])), abs(float(x[1]))), float(best.fun), B


def estimate(rows, cal0, n_iter=3, th0=(0.10, 0.10), log=print, fast=True):
    """fixed-point estimate of theta = (c, c_a) (cal0 = the model's solve calibration, without c); returns theta, history, the
    last residual set"""
    th = tuple(th0); hist = []
    for it in range(n_iter):
        S = solve_rows(rows, with_theta(cal0, th), fast=fast)
        th_new, J, B = fit_theta(S, th)
        rF, rC = bin_medians(S, th_new)
        nfr = int(len(S['e']))
        hist.append(dict(it=it, theta_in=list(th), theta=list(th_new), J=J, n_rows=nfr, frame_bins=rF, contrast_bins=rC))
        log(f'  brightness noise it {it}: c {th[0]:.3f} c_a {th[1]:.3f} -> c {th_new[0]:.3f} c_a {th_new[1]:.3f} '
            f'(J {J:.1e}, rows {nfr}) med u frame [SNR]: ' + ' '.join(f'{m:.2f}[{x:.0f}]' for x, _, m in rF) +
            ' | contrast: ' + ' '.join(f'{m:.2f}[{x:.0f}]' for x, _, m in rC))
        th = th_new
    return th, hist, S


def q_table(rows, cal, fast=True):
    """q/dof p50 per window-photon bin (all rows) for a calibration"""
    S = solve_rows(rows, cal, keep_all=True, fast=fast)
    qd = S['q'] / S['dof']; ph = S['ph']
    return [(lo, float(qd[(ph >= lo) & (ph < hi)].median()) if ((ph >= lo) & (ph < hi)).sum() > 30 else np.nan) for lo, hi in PHB]
