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
(5 = 6 intensities + 2 prior pseudo-observations - (x, y, B); 3 with the kinetic on/off times).
2026-11-19: this frame-residual fit (estimate) recovers c, c_a on oracle intensities (true per-frame photons + noise of a known
covariance) but reads low (c ~ 0.002-0.08, c_a ~ 0) on the networks' intensities: the networks shrink the brightness fluctuations
and over-cover their own errors in exactly the directions the residuals live in (component section below). The pipeline uses the
component estimator (estimate_components) with a per-model network-transfer table; estimate stays as method 'residual'."""
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
def solve_rows(rows, cal, chunk=8192, keep_all=False, fast=True, extra=False):
    """non-kinetic pattern solve on all rows (cal from with_theta); per-row residual quantities at the MAP (torch, concatenated).
    Only the rows flagged iso are kept unless keep_all. fast: solve_fast.solve_emitters_fast instead of solve.solve_emitters.
    extra: also the window pattern pt, the MAP position r, the coupling Cj and the window sum of the network intensities."""
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
            vals = dict(e=oo['I'] - s['mu'], mu=s['mu'], Sn=s['Sig'], J=J, sp=s['s_prior'], ang=C.angle_k(pt)[1],
                        ph=tt(ph[sl]), q=s['chi2'], dof=s['dof'])
            if extra: vals.update(pt=pt, r=s['r_map'], Cj=oo['C'].float(), I_sum=oo['I'].float().sum(1))
            for k, v in vals.items():
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


# ------------------------------------------------------------------------------------------- component estimator (2026-11-19)
# The emitter network is trained on the per-frame photons INCLUDING the brightness fluctuation (NLL of the GT photons), i.e. it
# outputs a posterior mean: within a window it shrinks the fluctuations along the directions its training prior constrains (the
# angle contrast and the depth of each angle: gains K ~ 0.4-0.8 on held-out sims with per-frame truth) and its covariance
# over-covers its own errors there (factor a ~ 0.4-0.8), while the phase direction (= position) is nearly unshrunk. The
# residuals of the network intensities are then SMALLER than Sigma_net + c^2 mu^2 assumes, and the residual fit above returns
# c, c_a ~ 0 although it recovers them exactly on oracle intensities (out/bnoise_fix). The component estimator works with the
# three position- and brightness-free statistics of a window that carry the brightness noise,
#     h = (A_0 - A_1) / mean(A)        (angle contrast;  c_a and c)
#     rho_a = Re(Z_a e^{-i k_a.p}) / A_a   (the depth of angle a;  c)
# (A_a, Z_a = the exact 3-step solve of angle a with the estimated pattern, at the solve's position p), and models the network's
# transfer of each statistic with a per-model calibration table (K_k, a_k per window-photon bin, fitted on simulations with
# per-frame truth): var(s_k - median_bin) = a_k x_k + K_k^2 tau_k(theta)^2, x_k = g_k^T Sigma_net g_k (the network's variance
# of the statistic, marginal covariance L L^T + C C^T), tau_k^2 = g_k^T Sigma_b(theta) g_k = c^2 T_c + c_a^2 T_a (the physical
# brightness covariance Sigma_b = c^2 diag(mu^2) + c_a^2 mu mu^T within angle, g_k = d s_k / d I at the model mean mu).
# theta minimises sum over statistics and photon bins of (median PIT - 1/2)^2. Without a table K = a = 1 (= the network taken
# as an unbiased measurement with precision Sigma_net).
STATS = ('h', 'rho0', 'rho1')


def angle_maps(pt, X, Y):
    """linear maps W_a [n,3,6] (a = 0, 1): intensities -> (A_a, rad_a, tan_a) of angle a: I_t / rel_t = A + Re(Z e^{-i phase_t})
    (the exact 3-step solve of the angle's frames), Z rotated by e^{-i k_a.(X, Y)} (rad = the depth part, tan = the phase part)"""
    n = len(X)
    ang = C_angle(pt)
    Ws = []
    for a in (0, 1):
        m = (ang == a)
        cols = torch.argsort((~m).to(torch.int8), dim=1, stable=True)[:, :3]          # the angle's 3 frames (in time order)
        g = lambda q: torch.gather(q, 1, cols)
        rel, ph, kx, ky = g(pt[..., 4]), g(pt[..., 2]), g(pt[..., 0])[:, 0], g(pt[..., 1])[:, 0]
        M = torch.stack([torch.ones_like(ph), torch.cos(ph), torch.sin(ph)], 2)              # [n,3(frames),3(A,U,V)]
        Mi = torch.linalg.inv(M) / rel[:, None, :]
        th = kx * X + ky * Y
        z0, o1 = torch.zeros(n, device=pt.device), torch.ones(n, device=pt.device)
        Rr = torch.stack([torch.stack([o1, z0, z0], 1), torch.stack([z0, torch.cos(th), torch.sin(th)], 1),
                          torch.stack([z0, -torch.sin(th), torch.cos(th)], 1)], 1)
        W3 = Rr @ Mi                                                                          # [n,3,3(frames)]
        W = torch.zeros(n, 3, T, device=pt.device)
        W.scatter_(2, cols[:, None, :].expand(-1, 3, -1), W3)
        Ws.append(W)
    return Ws


def C_angle(pt):
    from simcode.solve import pattern as C
    return C.angle_k(pt)[1]


def comp_stats(I, Ws):
    """(h, rho0, rho1) [n,3] and their gradients d/dI [n,3,T] at I"""
    v0 = (Ws[0] @ I[:, :, None])[..., 0]; v1 = (Ws[1] @ I[:, :, None])[..., 0]
    A0, A1 = v0[:, 0], v1[:, 0]; Am = 0.5 * (A0 + A1)
    h = (A0 - A1) / Am
    gh = ((Ws[0][:, 0] - Ws[1][:, 0]) * Am[:, None] - (A0 - A1)[:, None] * 0.5 * (Ws[0][:, 0] + Ws[1][:, 0])) / Am[:, None] ** 2
    r0 = v0[:, 1] / A0; g0 = (Ws[0][:, 1] * A0[:, None] - v0[:, 1:2] * Ws[0][:, 0]) / A0[:, None] ** 2
    r1 = v1[:, 1] / A1; g1 = (Ws[1][:, 1] * A1[:, None] - v1[:, 1:2] * Ws[1][:, 0]) / A1[:, None] ** 2
    return torch.stack([h, r0, r1], 1), torch.stack([gh, g0, g1], 1)


def bn_terms(mu, ang, G):
    """T_c, T_a [n,3]: tau_k^2 = c^2 T_c + c_a^2 T_a for the gradients G [n,3,T] at the mean mu [n,T]"""
    gm = G * mu[:, None, :]
    Tc = (gm ** 2).sum(-1)
    Taa = [((gm * (ang == a)[:, None, :]).sum(-1)) ** 2 for a in (0, 1)]
    return Tc, Taa[0] + Taa[1], Taa


def window_components(S):
    """the component statistics of the residual set S of solve_rows (needs pt, r, Cj): s_net [n,3], x [n,3], T_c, T_a [n,3], ph [n]"""
    Ws = angle_maps(S['pt'], S['r'][:, 0], S['r'][:, 1])
    mu = S['mu'].clamp(min=1e-3)
    s_net, _ = comp_stats(S['e'] + S['mu'], Ws)                       # e + mu = the network intensities
    _, G = comp_stats(mu, Ws)
    Sm = S['Sn'] + S['Cj'] @ S['Cj'].transpose(1, 2)                  # marginal network covariance (L L^T + C C^T)
    x = torch.einsum('nki,nij,nkj->nk', G, Sm, G)
    Tc, Ta, Taa = bn_terms(mu, S['ang'], G)
    return dict(s=s_net, x=x.clamp(min=1e-8), Tc=Tc, Ta=Ta, Ta0=Taa[0], Ta1=Taa[1], ph=S['I_sum'])


def load_transfer(path):
    """the network-transfer table of a model (json: ph_edges [nb+1], K / a / a_d {stat: [nb]}, theta_train); None = K = a = 1"""
    import json, os
    if not path or not os.path.exists(path): return None
    d = json.load(open(path))
    arr = lambda D: {k: np.asarray(v, float) for k, v in D.items()}
    out = dict(d, ph_edges=np.asarray(d['ph_edges'], float), K=arr(d['K']), a=arr(d['a']), a_d=arr(d.get('a_d', d['a'])))
    if 'K_d_c' in d:     # the depth-difference gain per calibration brightness noise c (the network's transfer depends on c)
        out['K_d_c'] = dict(c=np.asarray(d['K_d_c']['c'], float), **{k: np.asarray(d['K_d_c'][k], float) for k in STATS[1:]})
    return out


def _interp_ph(tr, ph, row):
    e = tr['ph_edges']; cen = np.log(np.sqrt(np.maximum(e[:-1], 1.0) * np.minimum(e[1:], 2e4)))
    return np.interp(np.log(np.maximum(ph.detach().cpu().numpy(), 1.0)), cen, row)


def transfer_at(tr, ph, key='a'):
    """K, a (or a_d) [n,3] at window photons ph (piecewise linear in log photons between the bin centres, held outside)"""
    n = len(ph)
    if tr is None: return torch.ones(n, 3, device=dev), torch.ones(n, 3, device=dev)
    e = tr['ph_edges']; cen = np.log(np.sqrt(np.maximum(e[:-1], 1.0) * np.minimum(e[1:], 2e4)))
    lp = np.log(np.maximum(ph.detach().cpu().numpy(), 1.0))
    K = np.stack([np.interp(lp, cen, tr['K'][k]) for k in STATS], 1); a = np.stack([np.interp(lp, cen, tr[key][k]) for k in STATS], 1)
    return tt(K), tt(a)


def comp_bins(ph, nb):
    """window-photon bin index [n] (quantiles, nb bins)"""
    q = torch.quantile(ph[::max(1, len(ph) // 200000)], torch.linspace(0, 1, nb + 1, device=dev))
    return torch.bucketize(ph, q[1:-1].contiguous())


def _medians(v, bins, nb):
    out = torch.full((nb,), float('nan'), device=dev)
    for b in range(nb):
        m = bins == b
        if m.sum() >= 50: out[b] = v[m].median()
    return out


def _centred(v, bins, nb):
    med = _medians(v, bins, nb)
    return v - torch.nan_to_num(med, nan=0.0)[bins]


def comp_setup(Cm, tr, pairs, nb, ok=None):
    """the fit's fixed quantities: the contrast h of every window (centred per photon bin) with a_h x_h, K_h^2 T_c, K_h^2 T_a; the
    depth differences d_a = rho_a(w) - rho_a(w + 6) of the pairs (centred) with a_d (x(w) + x(w')), sum K^2 T_c (static depth
    variations of the emitters, e.g. the pattern's depth field, cancel in d; c_a does not enter rho)"""
    K, a = transfer_at(tr, Cm['ph'], 'a'); _, ad = transfer_at(tr, Cm['ph'], 'a_d')
    ok = torch.ones(len(Cm['ph']), dtype=torch.bool, device=dev) if ok is None else ok
    bh = comp_bins(Cm['ph'][ok], nb)
    H = dict(s=_centred(Cm['s'][ok, 0], bh, nb), n0=a[ok, 0] * Cm['x'][ok, 0], Tc=K[ok, 0] ** 2 * Cm['Tc'][ok, 0],
             Ta=K[ok, 0] ** 2 * Cm['Ta'][ok, 0], Ta0=K[ok, 0] ** 2 * Cm['Ta0'][ok, 0], Ta1=K[ok, 0] ** 2 * Cm['Ta1'][ok, 0], bins=bh,
             row=torch.nonzero(ok)[:, 0])
    out = dict(h=H, nb=nb)
    if pairs is not None and len(pairs[0]) >= 100:
        i1, i2 = pairs
        bp = comp_bins(0.5 * (Cm['ph'][i1] + Cm['ph'][i2]), nb)
        for j, k in ((1, 'rho0'), (2, 'rho1')):
            out[k] = dict(row=i1, s=_centred(Cm['s'][i1, j] - Cm['s'][i2, j], bp, nb),
                          n0=0.5 * (ad[i1, j] + ad[i2, j]) * (Cm['x'][i1, j] + Cm['x'][i2, j]),
                          Tc=K[i1, j] ** 2 * Cm['Tc'][i1, j] + K[i2, j] ** 2 * Cm['Tc'][i2, j],
                          Ta=K[i1, j] ** 2 * Cm['Ta'][i1, j] + K[i2, j] ** 2 * Cm['Ta'][i2, j], bins=bp)
            if tr is not None and 'K_d_c' in tr:      # gain of the difference as a function of c: grids [n_c, n_pairs]
                G = tr['K_d_c']
                out[k].update(cg=G['c'], Tc1=Cm['Tc'][i1, j], Tc2=Cm['Tc'][i2, j],
                              Kg1=tt(np.stack([_interp_ph(tr, Cm['ph'][i1], r) for r in G[k]])),
                              Kg2=tt(np.stack([_interp_ph(tr, Cm['ph'][i2], r) for r in G[k]])))
    return out


def comp_pit_medians(F, th):
    """median PIT u per photon bin of each statistic {name: [nb]} at theta = (c, c_a) or (c, c_a0, c_a1) (per-axis factors)"""
    c, ca = th[0], th[1]; split = len(th) == 3; out = {}
    for k in STATS:
        if k not in F: continue
        D = F[k]
        if split and k == 'h':
            V = D['n0'] + c ** 2 * D['Tc'] + th[1] ** 2 * D['Ta0'] + th[2] ** 2 * D['Ta1']
            u = torch.special.gammainc(torch.full_like(V, 0.5), D['s'] ** 2 / V.clamp(min=1e-10) / 2)
            out[k] = _medians(u, D['bins'], F['nb']); continue
        if 'cg' in D:
            g = D['cg']; i = int(np.clip(np.searchsorted(g, c) - 1, 0, len(g) - 2)) if len(g) > 1 else 0
            w = float(np.clip((c - g[i]) / (g[i + 1] - g[i]), 0, 1)) if len(g) > 1 else 0.0
            K1 = D['Kg1'][i] * (1 - w) + (D['Kg1'][i + 1] * w if len(g) > 1 else 0); K2 = D['Kg2'][i] * (1 - w) + (D['Kg2'][i + 1] * w if len(g) > 1 else 0)
            V = D['n0'] + c ** 2 * (K1 ** 2 * D['Tc1'] + K2 ** 2 * D['Tc2'])
        else:
            V = D['n0'] + c ** 2 * D['Tc'] + ca ** 2 * D['Ta']
        u = torch.special.gammainc(torch.full_like(V, 0.5), D['s'] ** 2 / V.clamp(min=1e-10) / 2)
        out[k] = _medians(u, D['bins'], F['nb'])
    return out


def comp_objective(F, th):
    return float(sum(torch.nansum((u - 0.5) ** 2) for u in comp_pit_medians(F, th).values()))


def subset(F, sel_row):
    """F restricted to the windows (h) / pairs (first member) with sel_row[row] True (bins kept)"""
    G = dict(nb=F['nb'])
    for k in STATS:
        if k not in F: continue
        m = sel_row[F[k]['row']]
        G[k] = {q: (v[m] if torch.is_tensor(v) and v.dim() >= 1 and len(v) == len(m) else
                    (v[:, m] if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == len(m) else v)) for q, v in F[k].items()}
    return G


def fit_components(F, params=('c', 'c_a'), th0=(0.1, 0.05)):
    """theta = (c, c_a) of the component model (params: the free ones among c, c_a, c_a0, c_a1; the others 0; c_a0 / c_a1 = a
    separate factor per pattern orientation). Returns theta, objective, PIT medians"""
    def unpack(x):
        x = [abs(float(v)) for v in np.atleast_1d(x)]; d = dict(zip(params, x))
        if 'c_a0' in d or 'c_a1' in d: return (d.get('c', 0.0), d.get('c_a0', 0.0), d.get('c_a1', 0.0))
        return (d.get('c', 0.0), d.get('c_a', 0.0))
    def f(x):
        U = comp_pit_medians(F, unpack(x))
        return float(sum(torch.nansum((u - 0.5) ** 2) for u in U.values()))
    best = None
    starts = [[th0[0] if p == 'c' else th0[1] for p in params]] + [[s] * len(params) for s in (0.03, 0.1, 0.2, 0.3)]
    for x0 in starts:
        r = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=2e-4, fatol=1e-10, maxiter=300))
        if best is None or r.fun < best.fun: best = r
    th = unpack(best.x)
    return th, float(best.fun), {k: v.cpu().numpy().tolist() for k, v in comp_pit_medians(F, th).items()}


def estimate_components(rows, cal0, tr, pairs=None, th_solve=(0.1, 0.05), log=print, fast=True, nb=8):
    """component estimate of theta = (c, c_a) (module section above): one non-kinetic solve of the rows (positions; theta of the
    solve th_solve, the positions barely depend on it), the statistics, the fit. pairs = (i1, i2): indices into the rows (in the
    order of `rows`, as solve_rows returns them) of photon-disjoint windows (w, w + 6) of the same emitter (the depth statistic
    uses their differences; without pairs only the contrast h is fitted). Also the fits with c only and c_a only (the objective:
    the support of each parameter). Returns theta, info"""
    S = solve_rows(rows, with_theta(cal0, th_solve), fast=fast, extra=True)
    Cm = window_components(S)
    n = len(Cm['ph'])
    ok = torch.isfinite(Cm['s']).all(1) & torch.isfinite(Cm['x']).all(1) & (Cm['x'] < 1.0).all(1)
    if pairs is not None:
        i1, i2 = (torch.as_tensor(np.asarray(v), device=dev, dtype=torch.long) for v in pairs)
        m = ok[i1] & ok[i2]; pairs = (i1[m], i2[m])
    nb_ = int(max(2, min(nb, int(ok.sum()) // 400)))
    F = comp_setup(Cm, tr, pairs, nb_, ok)
    res = {}
    # model comparison: in-sample objective and the 2-fold held-out objective (alternating blocks of 2000 rows ~ time blocks; fit on
    # one half, the objective on the other); 'c+c_a0+c_a1' = separate per-axis factors
    nrow = len(Cm['ph']); half = ((torch.arange(nrow, device=dev) // 2000) % 2) == 0
    FA, FB = subset(F, half), subset(F, ~half)
    for nm, ps in (('c+c_a', ('c', 'c_a')), ('c', ('c',)), ('c_a', ('c_a',)), ('c+c_a0+c_a1', ('c', 'c_a0', 'c_a1')), ('c_a0+c_a1', ('c_a0', 'c_a1'))):
        th, J, U = fit_components(F, ps)
        thA, _, _ = fit_components(FA, ps); thB, _, _ = fit_components(FB, ps)
        Jcv = 0.5 * (comp_objective(FB, thA) + comp_objective(FA, thB))
        res[nm] = dict(theta=list(th), J=J, J_heldout=Jcv, theta_halves=[list(thA), list(thB)], pit_medians=U)
    # identifiability of a per-axis split: the contrast's per-axis weights T_a0, T_a1 (B fitted per window: only their weighted
    # sum enters; the split is identified only through the spread of T_a0 / T_a1 over the windows)
    if pairs is not None and len(pairs[0]) >= 100:
        # the static part of the angle contrast: correlation of h between the two photon-disjoint windows of a pair (static
        # per-emitter factors: photoselection, the pattern's per-orientation intensity envelope; the per-window noise does not correlate)
        bh_ = comp_bins(Cm['ph'], nb_); hc = _centred(Cm['s'][:, 0], bh_, nb_)
        h1, h2 = hc[pairs[0]], hc[pairs[1]]
        res['h_pair_corr'] = float(((h1 - h1.mean()) * (h2 - h2.mean())).mean() / (h1.std() * h2.std()))
    r_ = (F['h']['Ta0'] / F['h']['Ta1'].clamp(min=1e-12))
    res['axis_weights'] = dict(ratio_median=float(r_.median()), ratio_cv=float(r_.std() / r_.mean()))
    th = tuple(res['c+c_a']['theta'])
    npair = int(len(pairs[0])) if pairs is not None else 0
    log(f'  brightness noise (components, {int(ok.sum())} windows, {npair} pairs, {nb_} photon bins, transfer '
        f'{"table" if tr else "none: K = a = 1"}): c {th[0]:.4f} c_a {th[1]:.4f} (J {res["c+c_a"]["J"]:.2e}); c only: '
        f'{res["c"]["theta"][0]:.4f} (J {res["c"]["J"]:.2e}); c_a only: {res["c_a"]["theta"][1]:.4f} (J {res["c_a"]["J"]:.2e}); '
        f'per axis: c {res["c+c_a0+c_a1"]["theta"][0]:.4f} c_a0 {res["c+c_a0+c_a1"]["theta"][1]:.4f} c_a1 {res["c+c_a0+c_a1"]["theta"][2]:.4f}; '
        f'h pair corr {res.get("h_pair_corr", float("nan")):.3f}; axis weight ratio T_a0/T_a1 {res["axis_weights"]["ratio_median"]:.3f} (cv {res["axis_weights"]["ratio_cv"]:.3f}); '
        'held-out J ' + ' '.join(f'{k} {v["J_heldout"]:.2e}' for k, v in res.items() if isinstance(v, dict) and 'J_heldout' in v) + '; '
        'median PIT per photon bin: ' + ' | '.join(f'{k} ' + ' '.join(f'{v:.2f}' for v in U_) for k, U_ in res['c+c_a']['pit_medians'].items()))
    return th, dict(n_windows=int(ok.sum()), n_pairs=npair, n_bins=nb_, fits=res)


def q_table(rows, cal, fast=True):
    """q/dof p50 per window-photon bin (all rows) for a calibration"""
    S = solve_rows(rows, cal, keep_all=True, fast=fast)
    qd = S['q'] / S['dof']; ph = S['ph']
    return [(lo, float(qd[(ph >= lo) & (ph < hi)].median()) if ((ph >= lo) & (ph < hi)).sum() > 30 else np.nan) for lo, hi in PHB]
