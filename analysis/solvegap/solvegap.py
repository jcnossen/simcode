"""Solve-gap diagnostic (2026-11-14), parts B-E on the iso gate scene, pass 2 and pass 1 (m1809a dumps, seeds 301-303), h3-matched targets
(score >= 0.5, Hungarian 3 px on the detector point; out/crlbcmp/cache), errors vs CRLB_SF (gt_sf_i).
  B   production solve (simcode/solve/solve.fringe_posterior through the sc_solve inputs), c as calibrated / c = 0
      (+ c = 0 component variants: MAP instead of the posterior mean, no kinetic model, a single start at the prior (nf = 0))
  C   ORACLE inputs: I = gt_I + N(0, Sigma_fr), prior = gt + N(0, diag fr_xy^2) with Sigma_fr / fr_xy the free-model CRLB block
      (out/crlbcmp/crlbfree cov_or), the solve given exactly that covariance (kap = 1, c = 0, prior_off = 0)
  D   network inputs, c = 0, the start set centred on the TRUE position (nf = 1, posterior mean as production) and a single
      start at the truth (nf = 0, MAP)
  E   network inputs, c = 0, covariance = the empirical (second-moment) covariance of the whitened inputs per GT-photon bin
      (prior: sp_net * sqrt(Cz_pp) per axis, intensities: L Cz_II L^T; prior-intensity cross block dropped, as the solve has none)
  L   reference: linear GLS (x, y, B) linearised at the truth on the REALISED input errors with the c = 0 solve weights
      (and on the oracle inputs), i.e. the linear estimator's actual error, not its predicted variance.
  A   classical SIMFLUX MLE (analysis/solvegap/sffit.py outputs).
Rows of B beyond 3 sigma: classified (wrong fringe = k_a . dr off by >= pi on an angle; start = fixed by D (start set at the
truth); c_term = fine at c = 0 (B only); kinetic = the kinetic model was selected; PM = the MAP is within 3 sigma but the
posterior mean is not; inputs = the linear GLS on the same inputs (L) is beyond 3 sigma too; other), with the
prior's distance from the truth.
Output out/solvegap/solvegap_p<pass>.json, rows_p<pass>.npz.  usage: CUDA_VISIBLE_DEVICES=0 python analysis/solvegap/solvegap.py [2,1]"""
import os, sys, json
import numpy as np, torch
from simcode import config; from simcode.solve import pattern as C
from simcode.emitter import model as SCM; from simcode.solve import solve as MS
from simcode.solve import solve as SV

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, 'out', 'crlbcmp', 'cache')
FREE = os.path.join(REPO, 'out', 'crlbcmp', 'crlbfree')
OUT = os.path.join(REPO, 'out', 'solvegap')
PX = 108.33
SEEDS = (301, 302, 303)
BINS = [(500, 1000), (1000, 2000), (2000, 4000), (4000, 10000), (10000, np.inf)]
EBINS = [100, 200, 350, 500, 1000, 2000, 4000, 10000, np.inf]
dev = 'cuda'
tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
cpu = lambda t: t.detach().cpu().numpy()


def fpost(I, Sig, r0, sp, pt, base, cal, nf=1, thr=9.21, cm=0.0, ca=0.0, chunk=4096):
    if nf == 0:      # single start at `base`, no kinetic model (fringe_posterior needs >= 2 starts)
        rr = [MS.pattern_solve_all(I[c0:c0 + chunk], Sig[c0:c0 + chunk], r0[c0:c0 + chunk], sp[c0:c0 + chunk], pt[c0:c0 + chunk],
                                   (base - r0)[c0:c0 + chunk, None], kinetics=False, box=cal['box'], cmult=cm, cang=ca)['r'][:, 0]
              for c0 in range(0, len(I), chunk)]
        r = cpu(torch.cat(rr)); return dict(r=r, r_pm=r, kin=np.zeros(len(r), bool))
    out = {k: [] for k in ('r_pm', 'r', 'kin')}
    for c0 in range(0, len(I), chunk):
        s = slice(c0, c0 + chunk)
        o = MS.fringe_posterior(I[s], Sig[s], r0[s], sp[s], pt[s], [base[s]], nf=nf, box=cal['box'], thr=thr, temp=cal['mix_T'],
                                cmult=cm, cang=ca)
        for k in out: out[k].append(o[k])
    return {k: cpu(torch.cat(v)) for k, v in out.items()}


def lin_gls(e_in, Ig, pt, X, Y, sp, Sig):
    """realised linear GLS error (x, y) of the (x, y, B) fit linearised at the truth; e_in [n,8] = (prior err, I err)"""
    il, dx, dy = (cpu(a) for a in C.illum_at(tt(pt), tt(X), tt(Y)))
    n = len(X)
    J = np.zeros((n, 8, 3)); J[:, 0, 0] = 1; J[:, 1, 1] = 1
    J[:, 2:, 0] = Ig * dx / il; J[:, 2:, 1] = Ig * dy / il; J[:, 2:, 2] = Ig / Ig.sum(1, keepdims=True) * 1.0
    W = np.zeros((n, 8, 8)); W[:, 0, 0] = sp[:, 0] ** 2; W[:, 1, 1] = sp[:, 1] ** 2; W[:, 2:, 2:] = Sig
    Wi = np.linalg.inv(W)
    A = np.einsum('npi,npq,nqj->nij', J, Wi, J)
    b = np.einsum('npi,npq,nq->ni', J, Wi, e_in)
    return np.linalg.solve(A, b[..., None])[..., 0][:, :2]


def per_seed(sd, cal, cal0, Cz_by_bin, rng, ps, slopes):
    D = dict(np.load(f'{CACHE}/iso__s{sd}_p{ps}.npz'))
    F = dict(np.load(f'{FREE}/iso__s{sd}.npz')); assert np.allclose(F['gt_x'], D['gt_x'], atol=1e-4)
    A = dict(np.load(f'{OUT}/sffit_s{sd}.npz'))
    tg = np.nonzero(D['gt_target'] > 0)[0]
    j = D['m_h3'][tg]; ok = j >= 0
    tg, j = tg[ok], j[ok]
    if ps == 2: assert np.array_equal(np.sort(A['row']), np.sort(tg))
    pos = {r: i for i, r in enumerate(A['row'])}
    ia = np.array([pos.get(r, -1) for r in tg]); has = ia >= 0      # A was fitted on the pass-2 h3 rows
    gt = np.c_[D['gt_x'][tg], D['gt_y'][tg]]
    R = dict(seed=np.full(len(tg), sd), ph=D['gt_phot'][tg], sf=D['gt_sf_i'][tg], gt=gt, fr_xy=F['fr_xy'][tg])
    pt_np = D['patt_est'][D['c_b'][j].astype(int)]
    assert np.allclose(D['patt_est'], D['patt_true'])
    pt = tt(pt_np)
    o = dict(I=tt(D['c_I'][j]), L=tt(D['c_L'][j]), x=tt(D['c_x'][j]), y=tt(D['c_y'][j]), lsxy=tt(D['c_lsxy'][j]))
    gtt = tt(gt)
    # ---- B: production (c calibrated) and c = 0, + variants
    I, Sig, r0, sp, ph = SV.solve_inputs(o, cal)
    b1 = fpost(I, Sig, r0, sp, pt, r0, cal, cm=cal['cmult'], ca=cal['cmult_ang'])
    I0, Sig0, r00, sp0, _ = SV.solve_inputs(o, cal0)
    b0 = fpost(I0, Sig0, r00, sp0, pt, r00, cal0)
    b0nk = fpost(I0, Sig0, r00, sp0, pt, r00, cal0, thr=1e30)
    b0n0 = fpost(I0, Sig0, r00, sp0, pt, r00, cal0, nf=0, thr=1e30)
    R['chk_B'] = np.abs(b1['r_pm'] - np.c_[D['c_xs'][j], D['c_ys'][j]]).max()
    R['chk_B0'] = np.abs(b0['r_pm'] - np.c_[D['c_xs0'][j], D['c_ys0'][j]]).max()
    R['B'] = b1['r_pm']; R['B_map'] = b1['r']; R['B_kin'] = b1['kin']
    R['B0'] = b0['r_pm']; R['B0_map'] = b0['r']; R['B0_kin'] = b0['kin']; R['B0_nokin'] = b0nk['r_pm']; R['B0_nf0'] = b0n0['r']
    R['prior'] = cpu(r00)
    # ---- B0 with de-shrunk intensities: I' = mean + (I - mean) / s_bin (s = modulation slope of the network intensities per
    # GT-photon bin), Sig' = A Sig A^T with A = P / s + 1 1^T / T (diagnostic of the modulation shrinkage; c = 0)
    sv = np.ones(len(tg))
    for (lo, hi), s_ in slopes.items(): sv[(R['ph'] >= lo) & (R['ph'] < hi)] = s_
    P6 = torch.eye(6, device=dev) - 1.0 / 6
    As = P6[None] / tt(sv)[:, None, None] + 1.0 / 6
    Ids = (As @ I0[..., None])[..., 0]; Sds = As @ Sig0 @ As.transpose(1, 2)
    R['B0_ds'] = fpost(Ids, Sds, r00, sp0, pt, r00, cal0)['r_pm']
    R['B0_ds_nokin'] = fpost(Ids, Sds, r00, sp0, pt, r00, cal0, thr=1e30)['r_pm']
    R['L_ds'] = gt + lin_gls(np.c_[cpu(r00) - gt, cpu(Ids) - D['gt_I'][tg]], D['gt_I'][tg].astype(np.float64), pt_np, gt[:, 0], gt[:, 1],
                             cpu(sp0).astype(np.float64), cpu(Sds).astype(np.float64))
    R['slope'] = sv
    # ---- D: start at the truth
    d1 = fpost(I0, Sig0, r00, sp0, pt, gtt, cal0)
    d0 = fpost(I0, Sig0, r00, sp0, pt, gtt, cal0, nf=0)
    R['D'] = d1['r_pm']; R['D_kin'] = d1['kin']; R['D_nf0'] = d0['r']
    d1c = fpost(I, Sig, r0, sp, pt, gtt, cal, cm=cal['cmult'], ca=cal['cmult_ang'])
    R['Dc'] = d1c['r_pm']
    # ---- E: empirical covariance per GT-photon bin
    spn = np.exp(D['c_lsxy'][j]); L = D['c_L'][j]
    Lp = np.zeros_like(L); spE = np.zeros_like(spn)
    for (lo, hi), Cz in Cz_by_bin.items():
        m = (R['ph'] >= lo) & (R['ph'] < hi)
        if not m.any(): continue
        SI = L[m] @ Cz[2:, 2:] @ L[m].transpose(0, 2, 1)
        Lp[m] = np.linalg.cholesky(SI + 1e-6 * np.eye(6)); spE[m] = spn[m] * np.sqrt(np.diag(Cz)[:2])
    oE = dict(o, L=tt(Lp), lsxy=tt(np.log(spE)))
    calE = dict(cal0, kap_bins=dict(ph=[1.0, 1e6], kap_p=[1.0, 1.0], kap_I2=[1.0, 1.0]))
    IE, SigE, r0E, spE_t, _ = SV.solve_inputs(oE, calE)
    e1 = fpost(IE, SigE, r0E, spE_t, pt, r0E, calE)
    e1nk = fpost(IE, SigE, r0E, spE_t, pt, r0E, calE, thr=1e30)
    R['E'] = e1['r_pm']; R['E_nokin'] = e1nk['r_pm']
    # ---- C: oracle inputs
    cov = F['cov_or'][tg].astype(np.float64)
    CI = cov[:, 2:, 2:]; CI = 0.5 * (CI + CI.transpose(0, 2, 1))
    LI = np.linalg.cholesky(CI)
    Ig = D['gt_I'][tg].astype(np.float64)
    Io = Ig + (LI @ rng.standard_normal((len(tg), 6, 1)))[..., 0]
    spo = np.sqrt(np.c_[cov[:, 0, 0], cov[:, 1, 1]])
    po = gt + spo * rng.standard_normal((len(tg), 2))
    oC = dict(I=tt(Io), L=tt(LI), x=tt(po[:, 0]), y=tt(po[:, 1]), lsxy=tt(np.log(spo)))
    calC = dict(calE, prior_off=[0.0, 0.0])
    IC, SigC, r0C, spC, _ = SV.solve_inputs(oC, calC)
    c1 = fpost(IC, SigC, r0C, spC, pt, r0C, calC)
    c1nk = fpost(IC, SigC, r0C, spC, pt, r0C, calC, thr=1e30)
    c1t = fpost(IC, SigC, r0C, spC, pt, gtt, calC, nf=0, thr=1e30)
    R['C'] = c1['r_pm']; R['C_map'] = c1['r']; R['C_nokin'] = c1nk['r_pm']; R['C_true_nf0'] = c1t['r']; R['C_kin'] = c1['kin']
    R['C_lin'] = gt + lin_gls(np.c_[po - gt, Io - Ig], Ig, pt_np, gt[:, 0], gt[:, 1], spo, CI)
    # ---- L: realised linear GLS on the network inputs, c = 0 solve weights
    e_in = np.c_[cpu(r00) - gt, D['c_I'][j] - D['gt_I'][tg]]
    R['L'] = gt + lin_gls(e_in, D['gt_I'][tg].astype(np.float64), pt_np, gt[:, 0], gt[:, 1], cpu(sp0).astype(np.float64), cpu(Sig0).astype(np.float64))
    # ---- A
    nanf = lambda v: np.where(has[:, None], v[np.maximum(ia, 0)], np.nan)
    R['A'] = nanf(A['sf'][:, :2]); R['A_true'] = nanf(A['sf_true'][:, :2]); R['A_crlb_model'] = nanf(A['crlb_model'])
    R['pt'] = pt_np
    R['zp'] = (cpu(r00) - gt) / cpu(sp0)
    return R


def mod_slopes(ps):
    """modulation slope of the network intensities per GT-photon bin (simcode/emitter/model.mod_slope), seeds pooled"""
    A, G, ph = [], [], []
    for sd in SEEDS:
        D = dict(np.load(f'{CACHE}/iso__s{sd}_p{ps}.npz'))
        tg = np.nonzero(D['gt_target'] > 0)[0]; j = D['m_h3'][tg]; ok = j >= 0; tg, j = tg[ok], j[ok]
        A.append(D['c_I'][j]); G.append(D['gt_I'][tg]); ph.append(D['gt_phot'][tg])
    A, G, ph = np.concatenate(A), np.concatenate(G), np.concatenate(ph)
    out = {}
    for lo, hi in zip(EBINS[:-1], EBINS[1:]):
        m = (ph >= lo) & (ph < hi)
        a = A[m] - A[m].mean(1, keepdims=True); g = G[m] - G[m].mean(1, keepdims=True)
        out[(lo, hi)] = float((a * g).sum() / (g * g).sum())
    return out


def empirical_cz(cal0, ps):
    """second moment of the whitened inputs (prior err / exp(lsxy), L^-1 I err) per GT-photon bin, seeds pooled"""
    W, ph = [], []
    for sd in SEEDS:
        D = dict(np.load(f'{CACHE}/iso__s{sd}_p{ps}.npz'))
        tg = np.nonzero(D['gt_target'] > 0)[0]; j = D['m_h3'][tg]; ok = j >= 0; tg, j = tg[ok], j[ok]
        po = np.array(cal0['prior_off'])
        ep = np.c_[D['c_x'][j], D['c_y'][j]] - po - np.c_[D['gt_x'][tg], D['gt_y'][tg]]
        zI = np.linalg.solve(D['c_L'][j], (D['c_I'][j] - D['gt_I'][tg])[..., None])[..., 0]
        W.append(np.c_[ep / np.exp(D['c_lsxy'][j]), zI]); ph.append(D['gt_phot'][tg])
    W, ph = np.concatenate(W), np.concatenate(ph)
    out = {}
    for lo, hi in zip(EBINS[:-1], EBINS[1:]):
        m = (ph >= lo) & (ph < hi)
        Cz = W[m].T @ W[m] / m.sum(); Cz[:2, 2:] = 0; Cz[2:, :2] = 0
        out[(lo, hi)] = Cz
    return out


def ratio_tab(R, keys):
    T = {}
    for k in keys:
        e = R[k] - R['gt']; T[k] = {}
        for lo, hi in BINS:
            m = (R['ph'] >= lo) & (R['ph'] < hi) & np.isfinite(e).all(1)
            c = R['sf'][m]
            rat = np.sqrt((e[m] ** 2).mean(0)) / np.sqrt((c ** 2).mean(0))
            T[k][f'{lo:g}-{hi:g}'] = dict(n=int(m.sum()), rx=float(rat[0]), ry=float(rat[1]),
                                          rmsd_nm=float(np.sqrt((e[m] ** 2).mean()) * PX), crlb_nm=float(np.sqrt((c ** 2).mean()) * PX),
                                          tail3=float(np.mean(np.abs(e[m]) > 3 * c)),
                                          trim=float(np.sqrt((e[m] ** 2)[(np.abs(e[m]) <= 3 * c).all(1)].mean()) / np.sqrt((c[(np.abs(e[m]) <= 3 * c).all(1)] ** 2).mean())))
    return T


def classify(R, key, Dkey='D', c0key=None):
    e = R[key] - R['gt']; c = R['sf']
    bad = (np.abs(e) > 3 * c).any(1)
    Km, _ = C.angle_k(tt(R['pt']))
    Km = cpu(Km)
    nfr = np.einsum('nak,nk->na', Km, e) / (2 * np.pi)               # fringe-period units per angle
    wrong = (np.abs(nfr) > 0.5).any(1)
    ed = R[Dkey] - R['gt']; dfix = ~(np.abs(ed) > 3 * c).any(1)
    emap = R[key + '_map'] - R['gt'] if key + '_map' in R else e
    mapok = ~(np.abs(emap) > 3 * c).any(1)
    kin = R[key + '_kin'].astype(bool)
    c0ok = ~(np.abs(R[c0key] - R['gt']) > 3 * c).any(1) if c0key else np.zeros(len(c), bool)
    linbad = (np.abs(R['L'] - R['gt']) > 3 * c).any(1)
    ep = R['prior'] - R['gt']
    period_nm = 2 * np.pi / np.linalg.norm(Km, axis=2).mean(1) * PX
    out = {}
    for lo, hi in BINS:
        m = (R['ph'] >= lo) & (R['ph'] < hi) & bad
        if not m.any(): out[f'{lo:g}-{hi:g}'] = dict(n_bad=0); continue
        cat = np.select([wrong[m], dfix[m], c0ok[m], kin[m], mapok[m], linbad[m]],
                        ['wrong_fringe', 'start', 'c_term', 'kinetic', 'pm_mix', 'inputs'], 'other')
        r = dict(n_bad=int(m.sum()), n_bin=int(((R['ph'] >= lo) & (R['ph'] < hi)).sum()))
        for cc in ('wrong_fringe', 'start', 'c_term', 'kinetic', 'pm_mix', 'inputs', 'other'): r[cc] = int((cat == cc).sum())
        r['also_D_fixes_wrong_fringe'] = int((wrong[m] & dfix[m]).sum())
        r['prior_err_nm_median'] = float(np.median(np.hypot(*ep[m].T)) * PX)
        r['prior_err_nm_max'] = float(np.max(np.hypot(*ep[m].T)) * PX)
        r['prior_z_median'] = float(np.median(np.hypot(*R['zp'][m].T)))
        r['prior_err_over_sf_median'] = float(np.median(np.hypot(*(ep[m] / c[m]).T)))
        r['prior_err_over_frxy_median'] = float(np.median(np.hypot(*(ep[m] / R['fr_xy'][m]).T)))
        r['solve_err_nm_median'] = float(np.median(np.hypot(*e[m].T)) * PX)
        r['solve_err_fringe_units_max'] = float(np.abs(nfr[m]).max())
        r['period_nm'] = float(np.median(period_nm[m]))
        # all rows of the bin, for comparison
        ma = (R['ph'] >= lo) & (R['ph'] < hi)
        r['prior_err_nm_median_all'] = float(np.median(np.hypot(*ep[ma].T)) * PX)
        r['prior_z_median_all'] = float(np.median(np.hypot(*R['zp'][ma].T)))
        r['kin_frac_all'] = float(kin[ma].mean())
        out[f'{lo:g}-{hi:g}'] = r
    return out


CALF = {1: 'archive_2d/m1809/calib/calib_emitter_pass1_sim.json', 2: 'archive_2d/m1809/calib/calib_emitter_pass2_sim.json'}


def main(ps):
    cal = json.load(open(config.resolve(CALF[ps])))
    cal0 = dict(cal, cmult=0.0, cmult_ang=0.0)
    Cz = empirical_cz(cal0, ps)
    rng = np.random.default_rng(0)
    slopes = mod_slopes(ps)
    Rs = [per_seed(sd, cal, cal0, Cz, rng, ps, slopes) for sd in SEEDS]
    R = {k: (np.concatenate([r[k] for r in Rs]) if np.ndim(Rs[0][k]) else max(r[k] for r in Rs)) for k in Rs[0]}
    print('check vs cached solve: B', R['chk_B'], 'B0', R['chk_B0'], flush=True)
    keys = ['A', 'A_true', 'B', 'B0', 'B0_map', 'B0_nokin', 'B0_nf0', 'C', 'C_map', 'C_nokin', 'C_true_nf0', 'C_lin', 'D', 'D_nf0', 'Dc', 'E', 'E_nokin', 'L', 'B0_ds', 'B0_ds_nokin', 'L_ds', 'prior']
    out = dict(tab=ratio_tab(R, keys), cls_B=classify(R, 'B', 'Dc', 'B0'), cls_B0=classify(R, 'B0', 'D'),
               chk=dict(B=float(R['chk_B']), B0=float(R['chk_B0']),
                        crlb_model_over_sf_i_median=float(np.median(R['A_crlb_model'] / R['sf'])),
                        kin_frac_B0=float(R['B0_kin'].mean()), kin_frac_C=float(R['C_kin'].mean())),
               slopes={f'{k[0]:g}-{k[1]:g}': v for k, v in slopes.items()},
               Cz_diag={f'{k[0]:g}-{k[1]:g}': np.diag(v).tolist() for k, v in Cz.items()})
    np.savez_compressed(f'{OUT}/rows_p{ps}.npz', **{k: v for k, v in R.items() if np.ndim(v)})
    json.dump(out, open(f'{OUT}/solvegap_p{ps}.json', 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    for ps in [int(a) for a in (sys.argv[1] if len(sys.argv) > 1 else '2,1').split(',')]: main(ps)
