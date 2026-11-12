"""Movie-level classical 4Pi-SIMFLUX (uses cl4pi.Classic4Pi per chunk of candidate patches).

Stage A (per candidate): summed fit (x0, y0, z_ast, N, bg), per-frame counts N_t, photon asymmetry R, 4Pi moment
phase phi at z_ast.
Pattern (per file, from the data, classical positions): bright isolated on-through fits (filtering step 1 + 2) ->
k refined around a reference, step phases by the slot DFT, depth = median per-emitter modulation (analysis/fourpi/
pattern.py, as the earlier classical analysis). Then per candidate: alpha, MD, SIMFLUX two-step x*, y*.
4Pi phase field (the paper's ridge / phase-drift step): residual r = phi - phi_cal(s z_ast + z_off) of good fits;
Phi0(x, y, t) = tile map M(x, y) + block drift d(t) by alternating circular means (analysis/fourpi/solve_real4pi.
phase_field); the scale s of z_ast against the 4Pi phase and the offset are chosen by maximum residual coherence
(the ridge of the phase-vs-z_ast histogram). z_4pi = unwrap against z_ast.
Stage B: z-dependent channel calibration at z_4pi, then the 24-image pixel MLE (optional).
Filters (SI Table 2 + standard fit rejections): see FILTERS / flags().
"""
import math, os, sys
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p_ in (REPO, f'{REPO}/analysis/fourpi', os.path.dirname(os.path.abspath(__file__))):
    if p_ not in sys.path: sys.path.insert(0, p_)
from cl4pi import Classic4Pi, wrap
from pattern import refine_k, slotZ
from solve_real4pi import phase_field

PX_NM = 128.0
FILTERS = dict(R_max=0.2, MD=(0.6, 1.2), N_min=1000.0, sxy_max_nm=10.0, llr_max=500.0 * 225 / 169, C=(0.4, 1.6), chi2px_max=4.0)


def stage_a(c4, img, doff, refine=None):
    s = c4.summed_fit(img, doff, refine=refine)
    th = s['th']
    N, b, varN = c4.frame_counts(img, doff, th)
    zp = c4.zphase(img, doff, th[:, :2], z=th[:, 2])
    o = dict(x0=th[:, 0], y0=th[:, 1], z_ast=th[:, 2], Nsum=th[:, 3], bgsum=th[:, 4], sx0=s['crlb'][:, 0], sy0=s['crlb'][:, 1],
             sz_ast=s['crlb'][:, 2], chi2px=s['chi2px'], llr=s['llr'], conv=s['conv'], at_bound=s['at_bound'],
             Nt=N, bt=b, varNt=varN, phi_a=zp['phi'], C_a=zp['C'], sphi_a=zp['sphi'])
    Nx, Ny = N[:, :3].sum(1), N[:, 3:].sum(1)
    o['R'] = (Nx - Ny).abs() / (Nx + Ny).clamp(min=1e-3)
    return {k: v.cpu().numpy() for k, v in o.items()}


def estimate_pattern(A, X, Y, kref, sel, nmax=200000):
    """A: stage-A dict (numpy), X, Y global ch0 positions; kref [2,2]; sel: emitters to use."""
    idx = np.nonzero(sel)[0][:nmax]
    Xg = np.c_[X[idx], Y[idx]]
    pat = dict(n=int(len(idx)))
    for a in range(2):
        I3 = A['Nt'][idx, 3 * a:3 * a + 3].astype(np.float64)
        k = refine_k(I3, Xg, np.array(kref[a]), span=0.02, n=21)
        Z = slotZ(I3, Xg, k); psi = -np.angle(Z)
        pat[f'angle{a}'] = dict(k=k.tolist(), psi_deg=np.degrees(psi).tolist(), slot_abs=np.abs(Z).tolist())
    return pat


def psi_eff(pat, cx, cy):
    """per candidate step phases in local coordinates (relative to the ch0 candidate pixel)"""
    out = []
    for a in range(2):
        k = np.array(pat[f'angle{a}']['k']); ps = np.radians(pat[f'angle{a}']['psi_deg'])
        for s in range(3):
            out.append(ps[s] - (k[0] * cx + k[1] * cy))
    return np.stack(out, 1)


def lateral_np(A, pat, cx, cy, dev='cuda', c4=None):
    """SIMFLUX two-step on the stored per-frame counts (no patches needed)"""
    K = torch.tensor([pat['angle0']['k'], pat['angle1']['k']], device=dev, dtype=torch.float32)
    pe = torch.tensor(psi_eff(pat, cx, cy), device=dev, dtype=torch.float32)
    out = {}
    for i in range(0, len(cx), 200000):
        sl = slice(i, i + 200000)
        r = c4.lateral(torch.tensor(A['Nt'][sl], device=dev), torch.tensor(A['varNt'][sl], device=dev), K, pe[sl],
                       torch.tensor(np.c_[A['x0'][sl], A['y0'][sl]], device=dev, dtype=torch.float32))
        for k, v in r.items(): out.setdefault(k, []).append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def field_fit(c4, A, X, Y, W, sel, tile, tblk, H, Wd, scales=(0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3), dev='cuda',
              mode='poly', order=3):
    """4Pi phase field from phi - phi_cal(s z_ast); returns (f(x, y, w) -> Phi0, s, diag)"""
    za = torch.tensor(A['z_ast'], device=dev)
    best = None
    for s in scales:
        r = (torch.tensor(A['phi_a'], device=dev) - c4.phi_cal(s * za)).cpu().numpy()
        sv = np.sqrt(A['sphi_a'] ** 2 + (c4.dphi_dz(s * za).abs().cpu().numpy() * s * A['sz_ast']) ** 2)
        wgt = sel / np.maximum(sv, 0.1) ** 2
        if mode == 'tile':
            f, diag = phase_field(X[sel], Y[sel], W[sel], r[sel], wgt[sel], tile=tile, tblk=tblk, H=H, Wd=Wd)
        else:
            f, diag = phase_field_poly(X[sel], Y[sel], W[sel], r[sel], wgt[sel], order=order, tblk=tblk, H=H, Wd=Wd,
                                       tile=tile if mode == 'poly+tile' else None)
        if best is None or diag['R'] > best[2]['R']:
            best = (f, s, diag)
    return best


def _feats(x, y, H, Wd, order):
    u, v = (x - Wd / 2) / (Wd / 2), (y - H / 2) / (H / 2)
    return np.stack([u ** i * v ** j for i in range(order + 1) for j in range(order + 1 - i) if i + j > 0], 1)


def phase_field_poly(X, Y, W, phi, wgt, order=3, tblk=50, H=200, Wd=360, nit=12, tile=None, msmooth=1.5):
    """Phi0 = p(x, y) + d(t) (+ optional smoothed tile residual map): polynomial XY phase map as the published
    phaseCorrection step, per-block drift by circular means (the paper's per-segment phase-drift correction).
    p by Gauss-Newton on sum w (1 - cos(phi - d - p))."""
    from scipy import ndimage as ndi
    Fm = _feats(X, Y, H, Wd, order)
    tb = (W // tblk).astype(int); nb = tb.max() + 1
    a = np.zeros(Fm.shape[1]); d = np.zeros(nb)
    for it in range(nit):
        r = phi - Fm @ a
        Bc = np.zeros(nb, complex); np.add.at(Bc, tb, wgt * np.exp(1j * r))
        Bs = ndi.gaussian_filter1d(Bc.real, 1.0) + 1j * ndi.gaussian_filter1d(Bc.imag, 1.0)
        d = np.angle(Bs)
        for _ in range(3):
            r = np.angle(np.exp(1j * (phi - d[tb] - Fm @ a)))
            g = Fm.T @ (wgt * np.sin(r)); Hm = (Fm * (wgt * np.clip(np.cos(r), 0.05, None))[:, None]).T @ Fm
            a = a + np.linalg.solve(Hm + 1e-6 * np.eye(len(a)), g)
    M = None
    if tile is not None:
        r = np.exp(1j * (phi - d[tb] - Fm @ a)) * wgt
        tx = np.clip((X / tile).astype(int), 0, int(Wd / tile)); ty = np.clip((Y / tile).astype(int), 0, int(H / tile))
        G = np.zeros((int(H / tile) + 1, int(Wd / tile) + 1), complex); np.add.at(G, (ty, tx), r)
        G = ndi.gaussian_filter(G.real, msmooth) + 1j * ndi.gaussian_filter(G.imag, msmooth)
        M = np.angle(G)

    def f(x, y, w):
        out = _feats(x, y, H, Wd, order) @ a + d[np.clip(w // tblk, 0, nb - 1).astype(int)]
        if M is not None:
            ix = np.clip((x / tile).astype(int), 0, M.shape[1] - 1); iy = np.clip((y / tile).astype(int), 0, M.shape[0] - 1)
            out = out + M[iy, ix]
        return out
    res = np.angle(np.exp(1j * (phi - f(X, Y, W))))
    R = float(np.abs(np.sum(wgt * np.exp(1j * res))) / np.sum(wgt))
    return f, dict(R=R, a=a.tolist(), d=d, res=res)


def z4pi(c4, A, Phi0, s=1.0, dev='cuda'):
    phi = torch.tensor(A['phi_a'], device=dev); za = torch.tensor(A['z_ast'], device=dev) * s
    P0 = torch.tensor(Phi0, device=dev, dtype=torch.float32)
    return c4.unwrap_z(phi, P0, za).cpu().numpy()


def stage_b(c4, img, doff, A_sl, lat_sl, Phi0_sl, z4_sl, s, pat, cx, cy, do_pix=True, dev='cuda'):
    """z-dependent calibration at z_4pi (2 iterations) and the pixel MLE for one chunk"""
    xy = torch.tensor(np.c_[A_sl['x0'], A_sl['y0']], device=dev, dtype=torch.float32)
    za = torch.tensor(A_sl['z_ast'], device=dev) * s
    P0 = torch.tensor(Phi0_sl, device=dev, dtype=torch.float32)
    z4 = torch.tensor(z4_sl, device=dev, dtype=torch.float32)
    for _ in range(2):
        zp = c4.zphase(img, doff, xy, z=z4)
        z4 = c4.unwrap_z(zp['phi'], P0, za)
    o = dict(z4=z4, phi=zp['phi'], C=zp['C'], sphi=zp['sphi'], sz4=zp['sphi'] / c4.dphi_dz(z4).abs().clamp(min=1.0))
    if do_pix:
        K = torch.tensor([pat['angle0']['k'], pat['angle1']['k']], device=dev, dtype=torch.float32)
        pe = torch.tensor(psi_eff(pat, cx, cy), device=dev, dtype=torch.float32)
        m = torch.full((len(img),), float(pat['m']), device=dev)
        st = torch.stack([torch.tensor(lat_sl['xy'][:, 0], device=dev), torch.tensor(lat_sl['xy'][:, 1], device=dev), z4], 1)
        p = c4.pixel_mle(img, doff, st, m, pe, K, P0)
        o.update(xp=p['th'][:, 0], yp=p['th'][:, 1], zp=p['th'][:, 2], Fp=p['th'][:, 3], sxp=p['crlb'][:, 0], syp=p['crlb'][:, 1],
                 szp=p['crlb'][:, 2], chi2p=p['chi2px'], llrp=p['llr'], convp=p['conv'], boundp=p['at_bound'])
    return {k: v.cpu().numpy() for k, v in o.items()}


def flags(T, F=FILTERS):
    """filtering step 1 (kinetics: photon asymmetry, modulation depth) and step 2 (fit quality and uncertainty) of the
    published pipeline, plus the standard fit rejections. Returns an ordered dict of boolean 'reject' masks."""
    fl = {}
    fl['not converged'] = ~T['conv'].astype(bool)
    fl['parameter at bound'] = T['at_bound'].astype(bool)
    fl['chi2/px > 4 (summed fit)'] = T['chi2px'] > F['chi2px_max']
    fl['LLR > 666 (multi-emitter, SI T2 500 at 13x13 scaled to 15x15)'] = T['llr'] > F['llr_max']
    fl['photons < 1000 (SI T2)'] = T['Nsum'] < F['N_min']
    fl['lateral precision > 10 nm (SI T2; SIMFLUX sigma)'] = np.maximum(T['sxs'], T['sys']) * PX_NM > F['sxy_max_nm']
    fl['photon asymmetry R > 0.2 (step 1)'] = T['R'] > F['R_max']
    fl['modulation depth outside 0.6-1.2 (step 1)'] = ((T['MD'] < F['MD'][0]) | (T['MD'] > F['MD'][1])).any(1)
    ref = (T['conv'].astype(bool) & ~T['at_bound'].astype(bool) & (T['chi2px'] < F['chi2px_max']) & (T['Nsum'] > 1500))
    Cref = float(np.median(T['C'][ref])) if ref.sum() > 50 else 1.0
    Cr = T['C'] / Cref
    fl['interference contrast outside 0.4-1.6 x the file median'] = (Cr < F['C'][0]) | (Cr > F['C'][1])
    return fl


def flag_table(fl, base=None):
    """fraction removed by each filter alone and sequentially (in the listed order)"""
    n = len(next(iter(fl.values())))
    keep = np.ones(n, bool) if base is None else base.copy()
    n0 = keep.sum(); rows = []
    for k, m in fl.items():
        alone = float(m[keep if base is not None else slice(None)].mean()) if n0 else 0.0
        before = keep.sum(); keep &= ~m
        rows.append((k, alone, float((before - keep.sum()) / max(n0, 1))))
    return keep, rows
