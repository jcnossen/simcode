"""
Pattern FIELD maps (2026-11-10, opt-in; the default pipeline does not use them): a per-cell illumination phase map and a
smooth (Legendre-2) depth field per angle, estimated from the pass-1 (pattern-free) localizations and network
intensities, on top of a global pattern (pattern_est.py output).

Why: real data show an instrument-fixed, mostly small-scale (~1 um) pattern phase field (angle 0 ~5 nm rms, angle 1
~3 nm rms per 10 px cell) and a smooth depth field (0.75-0.95) that the global pattern (linear k, one depth per angle)
does not model (<out>/field/).

Model: illumination of angle a at position r = rel (1 + m_a f_a(r) cos(k_a.r - psi_s - eps_a(t) - phi_a(r))), with
  phi_a(r): bilinear interpolation of a cell map (cell size `cell` px, movie pixel frame), and
  f_a(r): depth factor = D_a(r) / <D_a>, D_a a Legendre-2 polynomial in u = (x - W/2)/(W/2), v = (y - H/2)/(H/2).
Estimation (same selection as pattern_est: clean on-through emitter-windows, positions averaged over w-6, w, w+6):
  delta_n = arg[Z_n exp(-i k.r_n) exp(+i eps(t))] = k.(r_true - r_prior) - phi(r): the per-cell weighted circular mean
  gives -phi. Per cell: raw = -mean(delta); smooth = weighted Legendre-2 fit of the raw cells; phi = smooth + lambda (raw -
  smooth) with the empirical-Bayes shrinkage lambda = tau^2 / (tau^2 + se^2). The phasor-weighted mean of phi over the
  clean emitters is removed (global phase gauge stays with the global pattern).
  Depth: cross-cycle estimator (fit_depth) per cell, weighted Legendre-2 fit.
GAUGE CAVEAT: phi is measured relative to the pass-1 prior positions, so any position-dependent bias of the prior is
absorbed into phi (and then into the final positions). Validate by time-split CV and cross-dataset (sensor-frame) transfer.
"""
import math
import numpy as np
from simcode2.pipeline.pattern_est import select, angle_Z, _interp_eps, T


def leg2(u, v):
    return np.stack([np.ones_like(u), u, v, 0.5 * (3 * u ** 2 - 1), u * v, 0.5 * (3 * v ** 2 - 1)], -1)


def _pattern_np(P):
    K = np.asarray(P['k'], np.float64); psi = np.asarray(P['phi'], np.float64)
    lr = np.asarray(P['logrel'], np.float64); rel = np.exp(lr - lr.mean())
    m = np.asarray(P['m'], np.float64)[[0, 3]]; eps = np.asarray(P['dphi'], np.float64); kf = int(P['knot_frames'])
    return K, psi, rel, m, eps, kf


def row_phasors(D, idx, P):
    """per clean row and angle: unit phasor of delta, weight, A (diluted amplitude)."""
    K, psi, rel, m, eps, kf = _pattern_np(P)
    w = D['w'][idx]; f = w[:, None] + np.arange(T)[None]; st = f % 6
    U = np.zeros((len(idx), 2), np.complex128); W = np.zeros((len(idx), 2))
    for a in (0, 1):
        A, Z, cols = angle_Z(D['I'][idx], st, psi, rel, a)
        e = _interp_eps(eps, kf, np.take_along_axis(f, cols, 1).mean(1))[a]
        kn = np.hypot(*K[a])
        u = Z * np.exp(-1j * (K[a, 0] * D['x'][idx] + K[a, 1] * D['y'][idx])) * np.exp(1j * e)
        U[:, a] = u / np.maximum(np.abs(u), 1e-9)
        var = kn ** 2 * D['s'][idx] ** 2 + 2.0 / (max(m[a], 0.3) ** 2 * np.maximum(A, 1.0))
        W[:, a] = np.where(A > 0, 1.0 / var, 0.0)
    return U, W


def depth_pairs(D, idx, P):
    """cross-cycle pairs (w, w+6) of the same clean emitter: per pair and angle cross = Re(Z1 Z2*), aa = A1 A2; index of pair."""
    K, psi, rel, m, eps, kf = _pattern_np(P)
    from simcode2.pipeline.pattern_est import depth_pair_query
    dd, jj = depth_pair_query(D, idx)          # (speed branch) same query as pattern_est.fit_depth (spatial hash by default)
    ok = np.isfinite(dd); i1, i2 = idx[ok], idx[jj[ok]]
    CR = np.zeros((len(i1), 2)); AA = np.zeros((len(i1), 2))
    for a in (0, 1):
        res = []
        for ii in (i1, i2):
            ww = D['w'][ii]; ff = ww[:, None] + np.arange(T)[None]; stt = ff % 6
            A, Z, cols = angle_Z(D['I'][ii], stt, psi, rel, a)
            e = _interp_eps(eps, kf, np.take_along_axis(ff, cols, 1).mean(1))[a]
            res.append((A, Z * np.exp(1j * e)))
        (A1, Z1), (A2, Z2) = res
        g = (A1 > 0) & (A2 > 0)
        CR[:, a] = np.where(g, np.real(Z1 * np.conj(Z2)), 0); AA[:, a] = np.where(g, A1 * A2, 0)
    return i1, CR, AA


def _cells(x, y, cell, ncx, ncy):
    cx = np.clip((x // cell).astype(int), 0, ncx - 1); cy = np.clip((y // cell).astype(int), 0, ncy - 1)
    return cy * ncx + cx


def fit_phase_map(x, y, U, W, cell, shape, mode='shrink', min_n=20, split=None, se_infl=None):
    """x, y (movie px), U [n] unit phasors (one angle), W [n] weights -> phi map [ncy, ncx] (rad, to ADD to the pattern phase),
    plus diagnostics. mode: 'shrink' (cells shrunk to the L2 smooth), 'raw' (cells), 'smooth' (L2 only), 'none'.
    The analytic per-cell se ignores the correlation of rows of one binding event / site (several windows) and is ~3x too
    small; it is calibrated by a split-half test: split = 0/1 labels (alternating time blocks), se^2 x c with
    c = median((raw_0 - raw_1)^2 / (se_0^2 + se_1^2)) / 0.455 (median of chi2_1), floored at 1; or a given se_infl."""
    Hh, Ww = shape; ncx, ncy = int(math.ceil(Ww / cell)), int(math.ceil(Hh / cell))
    if mode == 'none': return np.zeros((ncy, ncx)), {}
    c = _cells(x, y, cell, ncx, ncy); nc = ncx * ncy
    S = np.bincount(c, W * U.real, nc) + 1j * np.bincount(c, W * U.imag, nc)
    Wb = np.bincount(c, W, nc); W2 = np.bincount(c, W ** 2, nc); cnt = np.bincount(c, minlength=nc)
    R_ = np.abs(S) / np.maximum(Wb, 1e-12)
    raw = -np.angle(S)                                  # -mean(delta) per cell
    se = np.sqrt(np.maximum(W2 * (1 - np.minimum(R_, 0.999) ** 2) / 2, 0)) / np.maximum(R_ * Wb, 1e-12)
    good = cnt >= min_n
    infl = 1.0 if se_infl is None else float(se_infl)
    if split is not None and se_infl is None and mode in ('shrink', 'raw'):
        h = [fit_phase_map(x[split == k], y[split == k], U[split == k], W[split == k], cell, shape, mode='raw', min_n=min_n, se_infl=1.0)[1]
             for k in (0, 1)]
        g2 = (h[0]['n'] >= min_n) & (h[1]['n'] >= min_n)
        if g2.sum() >= 5:
            dd = np.angle(np.exp(1j * (h[0]['raw'] - h[1]['raw'])))[g2]
            infl = max(float(np.median(dd ** 2 / (h[0]['se'] ** 2 + h[1]['se'] ** 2)[g2]) / 0.455), 1.0)
    se = se * math.sqrt(infl)
    xc = (np.arange(ncx) + 0.5) * cell; yc = (np.arange(ncy) + 0.5) * cell
    XC, YC = np.meshgrid(xc, yc); B = leg2((XC.ravel() - Ww / 2) / (Ww / 2), (YC.ravel() - Hh / 2) / (Hh / 2))
    wg = 1 / se[good] ** 2
    coef, *_ = np.linalg.lstsq(B[good] * np.sqrt(wg)[:, None], raw[good] * np.sqrt(wg), rcond=None)
    smooth = B @ coef
    r = raw - smooth
    tau2 = max(np.average(r[good] ** 2, weights=wg) - np.average(se[good] ** 2, weights=wg), 0.0)
    lam = np.where(good, tau2 / (tau2 + se ** 2), 0.0)
    phi = {'shrink': smooth + lam * r, 'raw': np.where(good, raw, smooth), 'smooth': smooth}[mode]
    # gauge: remove the weight-averaged phi over the data (global phase stays with the global pattern)
    phi = phi - np.sum(Wb * phi) / np.sum(Wb)
    return phi.reshape(ncy, ncx), dict(raw=raw.reshape(ncy, ncx), se=se.reshape(ncy, ncx), n=cnt.reshape(ncy, ncx),
                                       smooth_coef=coef.tolist(), tau=math.sqrt(tau2), se_infl=infl,
                                       lam_med=float(np.median(lam[good])) if good.any() else 0.0)


def fit_depth_field(x, y, CR, AA, shape, cell=10):
    """weighted Legendre-2 fit of the per-cell cross-cycle depth (one angle) -> coef [6], normalisation (clean-weighted mean)."""
    Hh, Ww = shape; ncx, ncy = int(math.ceil(Ww / cell)), int(math.ceil(Hh / cell)); nc = ncx * ncy
    c = _cells(x, y, cell, ncx, ncy)
    sc = np.bincount(c, CR, nc); sa = np.bincount(c, AA, nc); cnt = np.bincount(c, minlength=nc)
    m2 = sc / np.maximum(sa, 1e-9)
    # se of m^2 from the per-pair scatter around m2 * aa
    res2 = np.bincount(c, (CR - m2[c] * AA) ** 2, nc)
    se2 = np.sqrt(res2) / np.maximum(sa, 1e-9)
    md = np.sqrt(np.maximum(m2, 1e-4)); se = se2 / (2 * md)
    good = (cnt >= 30) & (m2 > 0)
    xc = (np.arange(ncx) + 0.5) * cell; yc = (np.arange(ncy) + 0.5) * cell
    XC, YC = np.meshgrid(xc, yc); B = leg2((XC.ravel() - Ww / 2) / (Ww / 2), (YC.ravel() - Hh / 2) / (Hh / 2))
    wg = 1 / se[good] ** 2
    coef, *_ = np.linalg.lstsq(B[good] * np.sqrt(wg)[:, None], md[good] * np.sqrt(wg), rcond=None)
    # normalisation = the field averaged over the pairs (so that <f> = 1 over the data used by the global depth)
    norm = float(np.mean(leg2((x - Ww / 2) / (Ww / 2), (y - Hh / 2) / (Hh / 2)) @ coef))
    return coef, norm, dict(cell_depth=md.reshape(ncy, ncx), cell_se=se.reshape(ncy, ncx), n=cnt.reshape(ncy, ncx))


def estimate_field(R, P, shape, cell=6, mode='shrink', depth=True, nn_min=3.0, min_phot=300.0, rows=None, verbose=print, phase=True):
    """R: pass-1 rows as fed to estimate_pattern (prior_off removed, 'out' set); P: the global pattern dict.
    rows: optional boolean mask over the clean emitter-windows' window start (callable w -> bool) for split estimation.
    phase (2026-11-10, default True = unchanged): False = DEPTH-ONLY field (no phase map is estimated; F['phase'] = zeros,
    F['phase_on'] = False), the per-cell phase map being anchor-specific."""
    D = select(R, nn_min=nn_min, min_phot=min_phot)
    idx = np.nonzero(D['clean'])[0]
    if rows is not None: idx = idx[rows(D['w'][idx])]
    F = dict(cell=float(cell), shape=[int(shape[0]), int(shape[1])], mode=mode, phase=[], depth_coef=[], depth_norm=[], diag={})
    if not phase:
        Hh, Ww = shape
        F['phase'] = np.zeros((2, int(math.ceil(Hh / cell)), int(math.ceil(Ww / cell)))); F['phase_on'] = False
        U = W = None
    else:
        U, W = row_phasors(D, idx, P)
        w_ = D['w'][idx]; blk = max(150, int((w_.max() - w_.min()) / 16)); split = (w_ // blk) % 2       # se calibration halves
    for a in ((0, 1) if phase else ()):
        phi, dg = fit_phase_map(D['x'][idx], D['y'][idx], U[:, a], W[:, a], cell, shape, mode=mode, split=split)
        F['phase'].append(phi); F['diag'][f'phase{a}'] = dg
        if verbose and dg: verbose(f'  field angle {a}: cells {int((dg["n"] >= 20).sum())}, phi rms {np.std(phi) / np.hypot(*np.asarray(P["k"])[a]) * 108.33:.2f} nm, '
                                   f'tau {dg["tau"] / np.hypot(*np.asarray(P["k"])[a]) * 108.33:.2f} nm, se inflation {dg["se_infl"]:.1f}, median shrink {dg["lam_med"]:.2f}')
    if phase: F['phase'] = np.stack(F['phase'])
    if depth:
        i1, CR, AA = depth_pairs(D, idx, P)
        for a in (0, 1):
            coef, norm, dg = fit_depth_field(D['x1'][i1], D['y1'][i1], CR[:, a], AA[:, a], shape)
            F['depth_coef'].append(coef.tolist()); F['depth_norm'].append(norm); F['diag'][f'depth{a}'] = dg
            if verbose: verbose(f'  depth field angle {a}: coef {np.round(coef, 3).tolist()} norm {norm:.3f}')
    F['n_rows'] = int(len(idx))
    return F


def field_at(F, x, y):
    """dphi [n,2] (rad, added to the pattern phase) and depth factor [n,2] at movie-pixel positions x, y.
    F['map_offset'] = (dx, dy) (optional): the phase map lives in another frame (e.g. the 0.5 nM sensor frame): map coords =
    (x + dx, y + dy). The depth field is in the movie's own frame (centre W/2, H/2) unless F['depth_offset'] = (dx, dy) is
    set (2026-11-10, optional; a depth field transferred from another sample): it is then evaluated at (x + dx, y + dy) in the
    frame of the sample it was fitted on (F['shape']). F['phase_on'] = False (depth-only field): dphi = 0."""
    cell = F['cell']; Hh, Ww = F['shape']
    mo = F.get('map_offset', (0.0, 0.0)); do = F.get('depth_offset', (0.0, 0.0))
    out_p = np.zeros((len(x), 2)); out_f = np.ones((len(x), 2))
    for a in (0, 1):
        if len(F.get('depth_coef', [])) == 2:
            xd, yd = x + do[0], y + do[1]
            Dv = leg2((xd - Ww / 2) / (Ww / 2), (yd - Hh / 2) / (Hh / 2)) @ np.asarray(F['depth_coef'][a])
            out_f[:, a] = np.clip(Dv / F['depth_norm'][a], 0.3, 1.5)
        if not F.get('phase_on', True): continue
        M = np.asarray(F['phase'][a]); ncy, ncx = M.shape
        gx = np.clip((x + mo[0]) / cell - 0.5, 0, ncx - 1); gy = np.clip((y + mo[1]) / cell - 0.5, 0, ncy - 1)
        x0 = np.clip(np.floor(gx).astype(int), 0, max(ncx - 2, 0)); y0 = np.clip(np.floor(gy).astype(int), 0, max(ncy - 2, 0))
        tx = np.clip(gx - x0, 0, 1); ty = np.clip(gy - y0, 0, 1)
        x1 = np.minimum(x0 + 1, ncx - 1); y1 = np.minimum(y0 + 1, ncy - 1)
        out_p[:, a] = (M[y0, x0] * (1 - tx) * (1 - ty) + M[y0, x1] * tx * (1 - ty) + M[y1, x0] * (1 - tx) * ty + M[y1, x1] * tx * ty)
    return out_p, out_f


def depth_only(F, offset=None, R=None, P=None, shape=None, nn_min=3.0, min_phot=300.0):
    """(2026-11-10) depth-only copy of a field dict (phase map dropped). offset = (dx, dy): transfer to another sample whose
    movie px (x, y) = (x + dx, y + dy) in F's frame (F['depth_offset']). With R / P (the target's pass-1 rows and global
    pattern, as estimate_field), the normalisation is recomputed on the TARGET's clean cross-cycle pairs, so that <f> = 1 over
    the data the target's global depth was estimated on (only the SHAPE is transferred)."""
    G = {k: v for k, v in F.items() if k not in ('phase', 'diag', 'map_offset', 'map_shape')}
    G['phase'] = np.zeros_like(np.asarray(F['phase'])); G['phase_on'] = False; G['diag'] = {}
    if offset is not None: G['depth_offset'] = (float(offset[0]), float(offset[1]))
    if R is not None:
        D = select(R, nn_min=nn_min, min_phot=min_phot); idx = np.nonzero(D['clean'])[0]
        i1, CR, AA = depth_pairs(D, idx, P)
        Hh, Ww = G['shape']; do = G.get('depth_offset', (0.0, 0.0))
        B = leg2((D['x1'][i1] + do[0] - Ww / 2) / (Ww / 2), (D['y1'][i1] + do[1] - Hh / 2) / (Hh / 2))
        G['depth_norm'] = [float(np.mean(B @ np.asarray(G['depth_coef'][a]))) for a in (0, 1)]
    return G
