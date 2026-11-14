"""CRLB comparison (2026-11-13), stage 2: efficiency of the pass-1 / pass-2 head position (prior) and of the final pattern solve
against the SMLM / SIMFLUX CRLB, on the caches of prep.py (seeds pooled). All errors in nm (108.33 nm px), per GT-photon bin and axis.
Per bin / axis / estimate / matching rule:
  recall, n, bias (mean error), sd, RMSD, CRLB_rms = sqrt(mean c^2), RMSD / CRLB_rms, z_rms = sqrt(mean (e / c)^2),
  tail3 = fraction |e| > 3 c (Gaussian 0.27 %), RMSD with a miss penalty (each missed target counts PEN px per axis),
  sub-pixel bias curve b(f) (Fourier, 2 harmonics, f = GT sub-pixel coordinate), A1 (first-harmonic pull to the pixel centre),
  biased-CRLB MSE bound mean[(1 + b'(f))^2 c^2 + b(f)^2], and the Bayesian MMSE of a Gaussian likelihood (sd = c) with a
  uniform prior over the 1-px cell (genie bound: the estimator knows the cell).
Solve-gap decomposition (h3 rows, full-on emitters): per-frame intensity efficiency vs gt_fr_m, regression of the intensity errors on
the prior errors, linearised GLS (x, y, scale) of the solve from the network's empirical joint error covariance with / without the
prior-intensity cross block, and from CRLB-quality inputs (prior var = SM CRLB, intensity var = fr CRLB): architecture check vs SF CRLB.
usage: python analysis/crlbcmp/analyze.py -> out/crlbcmp/crlbcmp.json, tables.md, fig_*.png"""
import os, sys, json
import numpy as np
from scipy.stats import norm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, 'out', 'crlbcmp', 'cache')
OUT = os.path.join(REPO, 'out', 'crlbcmp')
PX = 108.33
PEN = 1.0
BINS = [100, 200, 350, 500, 1000, 2000, 4000, 10000, np.inf]
SCEN = ['iso', 'pair2', 'rs05qm', 'rs05pm']
SEEDS = (301, 302, 303)


def sub(x):
    f = x - np.round(x); f[f >= 0.5] -= 1.0; return f


def load(sc, p):
    Ds = []; off = 0; coff = 0
    for sd in SEEDS:
        fn = f'{CACHE}/{sc}__s{sd}_p{p}.npz'
        if not os.path.exists(fn): continue
        D = dict(np.load(fn)); n = len(D['c_b'])
        ff = os.path.join(OUT, 'crlbfree', f'{sc}__s{sd}.npz')
        if os.path.exists(ff):
            F = np.load(ff); assert np.allclose(F['gt_x'], D['gt_x'], atol=1e-4)
            D['gt_fr_xy'], D['gt_ts_full'] = F['fr_xy'], F['ts_full']
        else:
            D['gt_fr_xy'] = D['gt_ts_full'] = np.full((len(D['gt_x']), 2), np.nan, np.float32)
        D['gt_fov'] = D['gt_fov'] + off; D['c_b'] = D['c_b'] + off
        for m in ('m_h1', 'm_h3', 'm_n3'): D[m] = np.where(D[m] >= 0, D[m] + coff, -1)
        D['pidx'] = np.arange(int(D['n_fov'])) + off
        off += int(D['n_fov']); coff += n; Ds.append(D)
    if not Ds: return None
    return {k: np.concatenate([D[k] for D in Ds]) if np.ndim(Ds[0][k]) else None for k in Ds[0]}


# ---------------------------------------------------------------- Bayesian MMSE, uniform prior on [-0.5, 0.5), Gaussian likelihood
def _mmse_u(s):
    y = np.linspace(-0.5 - 6 * s, 0.5 + 6 * s, 4001); dy = y[1] - y[0]
    a, b = (-0.5 - y) / s, (0.5 - y) / s
    Z = norm.cdf(b) - norm.cdf(a)
    Zc = np.maximum(Z, 1e-300)
    m = (norm.pdf(a) - norm.pdf(b)) / Zc
    v = s ** 2 * (1 + (a * norm.pdf(a) - b * norm.pdf(b)) / Zc - m ** 2)
    return float(np.sum(Z * v) * dy)          # p(y) = Z (prior density 1)


_SG = np.exp(np.linspace(np.log(1e-3), np.log(3.0), 200)); _MG = np.array([_mmse_u(s) for s in _SG])
mmse_u = lambda s: np.interp(np.log(np.clip(s, 1e-3, 3.0)), np.log(_SG), _MG)


# ---------------------------------------------------------------- sub-pixel bias curve
def fourier(f, e, K=2):
    X = np.column_stack([np.ones_like(f)] + [g(2 * np.pi * k * f) for k in range(1, K + 1) for g in (np.sin, np.cos)])
    c = np.linalg.lstsq(X, e, rcond=None)[0]
    b = X @ c
    db = np.zeros_like(f)
    for k in range(1, K + 1):
        a_, b_ = c[2 * k - 1], c[2 * k]
        db += 2 * np.pi * k * (a_ * np.cos(2 * np.pi * k * f) - b_ * np.sin(2 * np.pi * k * f))
    return c, b, db


def est_xy(D, est):
    return {'prior': (D['c_x'], D['c_y']), 'solve': (D['c_xs'], D['c_ys']), 'solve_c0': (D['c_xs0'], D['c_ys0']),
            'solve_tp': (D['c_xst'], D['c_yst'])}[est]


def stats_bin(e, c, f, nt):
    """e, c, f: matched rows (px) one axis; nt: number of targets in the bin (incl. misses) and the CRLB of all targets"""
    n = len(e)
    if n < 30: return None
    rmsd = np.sqrt(np.mean(e ** 2)); cr = np.sqrt(np.mean(c ** 2))
    co, b, db = fourier(f, e, K=1)
    resid = np.sqrt(np.mean((e - b) ** 2))
    # E_f[b'^2] of the first harmonic = (2 pi)^2 (a^2 + b^2) / 2, minus its fit-noise expectation (2 pi)^2 2 s^2 / n;
    # E_f[(1 + b')^2] = 1 + E_f[b'^2] (cross term = 0 for uniform f: a periodic bias can never lower the averaged bound)
    db2 = max(0.0, (2 * np.pi) ** 2 * ((co[1] ** 2 + co[2] ** 2) / 2 - 2 * resid ** 2 / n))
    b2 = max(0.0, co[0] ** 2 + (co[1] ** 2 + co[2] ** 2) / 2 - 2 * resid ** 2 / n)
    bound_b = np.sqrt((1 + db2) * cr ** 2 + b2)
    bound_bc = np.sqrt(1 + db2) * cr
    # pointwise (f-conditional) bound with the fitted curve: min over f of (1 + b'(f))^2 = the most a pull can locally gain
    db_min = float(np.min(1 + db))
    ntot, call = nt
    pen = np.sqrt((np.sum(e ** 2) + (ntot - n) * PEN ** 2) / ntot) / np.sqrt(np.mean(call ** 2))
    return dict(n=n, recall=n / ntot, bias=np.mean(e) * PX, sd=np.std(e) * PX, rmsd=rmsd * PX, crlb=cr * PX, ratio=rmsd / cr,
                z_rms=float(np.sqrt(np.mean((e / c) ** 2))), tail3=float(np.mean(np.abs(e) > 3 * c)),
                tail5=float(np.mean(np.abs(e) > 5 * c)), ratio_trim5=float(np.sqrt(np.mean(e[np.abs(e) <= 5 * c] ** 2)) / cr),
                ratio_pen=pen, A1=-co[1] * PX, A1_se=np.sqrt(2 / n) * resid * PX, b_rms=np.sqrt(b2) * PX, dbdx_rms=float(np.sqrt(db2)),
                mean_1db2=float(1 + db2), min_1db=db_min, ratio_biased=rmsd / bound_b, resid_ratio_biased=resid / bound_bc,
                ratio_bayes_u=rmsd / np.sqrt(np.mean(mmse_u(c))), bayes_u_over_crlb=float(np.sqrt(np.mean(mmse_u(c))) / cr))


def prec_table(D, est, ckey, match):
    tg = np.nonzero(D['gt_target'] > 0)[0]
    j = D[match][tg]; ok = j >= 0
    ph = D['gt_phot'][tg]
    x, y = est_xy(D, est)
    C = D[ckey][tg]
    out = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mb = (ph >= lo) & (ph < hi) & np.isfinite(C).all(1)
        m = mb & ok
        r = {}
        for ax, (ev, gv) in enumerate(((x, D['gt_x']), (y, D['gt_y']))):
            e = ev[j[m]] - gv[tg[m]]
            s = stats_bin(e, C[m, ax], sub(gv[tg[m]]), (mb.sum(), C[mb, ax]))
            if s: r['xy'[ax]] = s
        if r: out[f'{lo:g}-{hi:g}'] = r
    return out


# ---------------------------------------------------------------- solve-gap decomposition
def illum(pt, X, Y):
    arg = pt[..., 0] * X[:, None] + pt[..., 1] * Y[:, None] - pt[..., 2]
    s = -pt[..., 4] * pt[..., 3] * np.sin(arg)
    return pt[..., 4] * (1 + pt[..., 3] * np.cos(arg)), s * pt[..., 0], s * pt[..., 1]


def gls_var(J, Sig):
    F = np.einsum('npi,npq,nqj->nij', J, np.linalg.inv(Sig), J)
    return np.linalg.inv(F)[:, [0, 1], [0, 1]]


def sandwich(J, W, S):
    """variance of the GLS estimator that uses weights W^-1 when the true error covariance is S"""
    Wi = np.linalg.inv(W)
    A = np.linalg.inv(np.einsum('npi,npq,nqj->nij', J, Wi, J))
    B = np.einsum('npi,npq,nqr,nrs,nsj->nij', J, Wi, S, Wi, J)
    return (A @ B @ A)[:, [0, 1], [0, 1]]


CALF = {1: 'calib/calib_sc_hd_m1809.json', 2: 'calib/calib_p2L_m1809.json'}


def solve_weights(D, j, Ig, pt, cal, cm, ca):
    """[n,8,8] covariance the explicit solve assumes (sc.sc_solve: kap_bins prior / intensity scales; msolve: + diag (cm mu)^2
    + ca^2 mu mu^T within each pattern angle, at the true mean mu = gt_I)"""
    ph = D['c_I'][j].sum(1)
    bv = lambda key: np.interp(np.log(np.maximum(ph, 1)), np.log(cal['kap_bins']['ph']), cal['kap_bins'][key])
    kp, kI2 = bv('kap_p'), bv('kap_I2')
    L = D['c_L'][j]
    SI = L @ L.transpose(0, 2, 1) * kI2[:, None, None] + 1e-3 * np.eye(6)
    SI = SI + np.einsum('nt,ts->nts', (cm * Ig) ** 2, np.eye(6))
    ang = (np.abs(pt[..., 0]) < np.abs(pt[..., 1])).astype(int)
    same = (ang[:, :, None] == ang[:, None, :]).astype(float)
    SI = SI + ca ** 2 * Ig[:, :, None] * Ig[:, None, :] * same
    S = np.zeros((len(j), 8, 8)); S[:, 2:, 2:] = SI
    sp = np.exp(D['c_lsxy'][j]) * kp[:, None]
    S[:, 0, 0] = sp[:, 0] ** 2; S[:, 1, 1] = sp[:, 1] ** 2
    return S


def gap(D, ckey_sm, ckey_sf, sc, p):
    cal = json.load(open(os.path.join(REPO, 'models', CALF[p])))
    tg = np.nonzero(D['gt_target'] > 0)[0]
    j = D['m_h3'][tg]; ok = j >= 0
    full = D['gt_onfr'][tg] >= 6
    if sc == 'iso': full[:] = True
    tg, j = tg[ok & full], j[ok & full]
    ph = D['gt_phot'][tg]
    ep = np.c_[D['c_x'][j] - D['gt_x'][tg], D['c_y'][j] - D['gt_y'][tg]]
    es = np.c_[D['c_xs'][j] - D['gt_x'][tg], D['c_ys'][j] - D['gt_y'][tg]]
    es0 = np.c_[D['c_xs0'][j] - D['gt_x'][tg], D['c_ys0'][j] - D['gt_y'][tg]]
    est = np.c_[D['c_xst'][j] - D['gt_x'][tg], D['c_yst'][j] - D['gt_y'][tg]]
    eI = D['c_I'][j] - D['gt_I'][tg]
    sp = np.exp(D['c_lsxy'][j]); L = D['c_L'][j]
    zI = np.linalg.solve(L, eI[..., None])[..., 0]
    W = np.c_[ep / sp, zI]
    pt = D['patt_est'][D['gt_fov'][tg].astype(int)]
    il, dx, dy = illum(pt, D['gt_x'][tg], D['gt_y'][tg])
    Ig = D['gt_I'][tg]
    J = np.zeros((len(tg), 8, 3)); J[:, 0, 0] = 1; J[:, 1, 1] = 1
    J[:, 2:, 0] = Ig * dx / il; J[:, 2:, 1] = Ig * dy / il; J[:, 2:, 2] = Ig
    M = np.zeros((len(tg), 8, 8)); M[:, 0, 0] = sp[:, 0]; M[:, 1, 1] = sp[:, 1]; M[:, 2:, 2:] = L
    sm, sf, fr = D[ckey_sm][tg], D[ckey_sf][tg], D['gt_fr_m'][tg]
    out = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ph >= lo) & (ph < hi) & np.isfinite(sm).all(1) & np.isfinite(sf).all(1) & np.isfinite(fr).all(1)
        if m.sum() < 50: continue
        Cz = W[m].T @ W[m] / m.sum()
        Czb = Cz.copy(); Czb[:2, 2:] = 0; Czb[2:, :2] = 0
        Mm = M[m]
        Sj = Mm @ Cz @ Mm.transpose(0, 2, 1); Sb = Mm @ Czb @ Mm.transpose(0, 2, 1)
        vj, vb = gls_var(J[m], Sj), gls_var(J[m], Sb)
        # CRLB-quality inputs; mixed
        Spi = np.zeros_like(Sb); Spi[:, 0, 0] = sm[m, 0] ** 2; Spi[:, 1, 1] = sm[m, 1] ** 2
        Spi[:, 2:, 2:] = np.einsum('nt,ts->nts', fr[m] ** 2, np.eye(6))
        vi = gls_var(J[m], Spi)
        Sx1 = Sb.copy(); Sx1[:, 2:, 2:] = Spi[:, 2:, 2:]            # network prior, CRLB intensities
        Sx2 = Sb.copy(); Sx2[:, :2, :2] = Spi[:, :2, :2]            # CRLB prior, network intensities
        v1, v2 = gls_var(J[m], Sx1), gls_var(J[m], Sx2)
        jm = j[m]
        Wc = solve_weights(D, jm, Ig[m], pt[m], cal, cal.get('cmult', 0.0), cal.get('cmult_ang', 0.0))
        W0 = solve_weights(D, jm, Ig[m], pt[m], cal, 0.0, 0.0)
        vw, vw0 = sandwich(J[m], Wc, Sb), sandwich(J[m], W0, Sb)
        # intensities: regression of eI on ep (slopes in photons / px), R^2
        X = np.c_[np.ones(m.sum()), ep[m]]
        cI = np.linalg.lstsq(X, eI[m], rcond=None)[0]
        res = eI[m] - X @ cI
        R2 = 1 - (res ** 2).sum(0) / ((eI[m] - eI[m].mean(0)) ** 2).sum(0)
        rho = Cz[:2, 2:] / np.sqrt(np.outer(np.diag(Cz)[:2], np.diag(Cz)[2:]))
        # solve error vs prior error (pass-through of the prior error)
        sl = [float(np.polyfit(ep[m, a], es[m, a], 1)[0]) for a in (0, 1)]
        r2 = lambda v: float(np.sqrt(np.mean(v)) * PX)
        out[f'{lo:g}-{hi:g}'] = dict(
            n=int(m.sum()),
            I_eff=float(np.sqrt(np.mean(eI[m] ** 2)) / np.sqrt(np.mean(fr[m] ** 2))),
            I_bias_rel=float(np.mean(eI[m].sum(1)) / np.mean(Ig[m].sum(1))),
            I_eff_net=float(np.sqrt(np.mean(eI[m] ** 2) / np.mean(np.einsum('ntk,ntk->nt', L[m], L[m])))),
            zI_rms=float(np.sqrt(np.mean(zI[m] ** 2))), zp_rms=float(np.sqrt(np.mean((ep[m] / sp[m]) ** 2))),
            R2_I_on_prior=float(np.mean(R2)), rho_max=float(np.abs(rho).max()), rho_rms=float(np.sqrt(np.mean(rho ** 2))),
            solve_on_prior_slope=sl,
            solve_rmsd=r2((es[m] ** 2).mean(1)), solve_c0_rmsd=r2((es0[m] ** 2).mean(1)), solve_tp_rmsd=r2((est[m] ** 2).mean(1)),
            solve_trim5=r2((es[m] ** 2)[(np.abs(es[m]) <= 5 * sf[m]).all(1)].mean(1)),
            sf_trim5=r2((sf[m] ** 2)[(np.abs(es[m]) <= 5 * sf[m]).all(1)].mean(1)), sf=r2((sf[m] ** 2).mean(1)), sm=r2((sm[m] ** 2).mean(1)), ts_full=r2((D['gt_ts_full'][tg][m] ** 2).mean(1)), fr_x=r2((D['gt_fr_xy'][tg][m] ** 2).mean(1)),
            prior_rmsd=r2((ep[m] ** 2).mean(1)),
            gls_blockdiag=r2(vb.mean(1)), gls_joint=r2(vj.mean(1)), gls_crlb_inputs=r2(vi.mean(1)),
            gls_netprior_crlbI=r2(v1.mean(1)), gls_crlbprior_netI=r2(v2.mean(1)),
            gls_solveweights=r2(vw.mean(1)), gls_solveweights_c0=r2(vw0.mean(1)))
    return out


def main():
    R = {}
    CK = {'iso': ('gt_sm_i', 'gt_sf_i'), 'pair2': ('gt_sm_m', 'gt_sf_m'), 'rs05qm': ('gt_sm_m', 'gt_sf_m'), 'rs05pm': ('gt_sm_m', 'gt_sf_m')}
    for sc in SCEN:
        for p in (1, 2):
            D = load(sc, p)
            if D is None: continue
            k = f'{sc}_p{p}'; R[k] = {}
            smk, sfk = CK[sc]
            tabs = [('prior', smk), ('prior', 'gt_fr_xy'), ('prior', sfk), ('solve', sfk), ('solve_c0', sfk)]
            if sc == 'iso': tabs += [('prior', 'gt_sm_m'), ('solve', 'gt_sf_m')]
            if sc.startswith('rs'): tabs += [('solve_tp', sfk), ('prior', 'gt_sm_i'), ('solve', 'gt_sf_i')]
            for est, ck in tabs:
                for mt in ('m_h1', 'm_h3', 'm_n3'):
                    R[k][f'{est}|{ck[3:]}|{mt[2:]}'] = prec_table(D, est, ck, mt)
            R[k]['gap'] = gap(D, smk, sfk, sc, p)
            print(k, 'done', flush=True)
    json.dump(R, open(f'{OUT}/crlbcmp.json', 'w'), indent=1, default=float)
    return R


if __name__ == '__main__':
    main()
