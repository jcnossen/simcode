"""sigma calibration study, Part B (2026-11-17; out/SIGMA_WORK_BRIEF.md): is the emitter network's uncertainty calibrated, and
where does the after-the-fact sigma factor come from? On the gate-scene dumps (simulated FOVs with ground truth, the model's
own PSF; simcode.processing.gate_dump) of one model, pass 1 and pass 2 separately, per network-photon bin (the bins of the
calibration fit, simcode.calibration.gate_stats.EDG). Matched targets as in the calibration fit (gate_stats.targets).

1. emitter network alone
   zI = L^-1 (I - I_gt) (whitened intensity errors, T = 6 components): robust sd (1.4826 MAD; its square = the fitted kap_I2),
   rms, mean chi2 / T, and the eigenvalues of cov(zI) (1 for a calibrated covariance);
   zp = (r_prior - prior_off - r_gt) / exp(lsxy): robust sd (= the fitted kap_p) and rms.
2. linearised pattern information (true pattern, at the true position, exact on-fractions in gt_I): the GLS position estimate
   from the intensities alone, e_pat = [(J^T W J)^-1 J^T W (I - I_gt)]_xy with J = d mu / d(x, y, B_angle0, B_angle1), W =
   Sigma^-1 (Sigma = L L^T, optionally + the pattern-error term c^2 diag(mu^2) + c_a^2 mu mu^T within angle), and its claimed
   sd sigma_pat. Correlation rho of e_pat and the prior error e_prior per axis. The independent combination the solve makes
   (Gaussian prior + intensity likelihood) has claimed sd 1 / sqrt(1/sp^2 + 1/sigma_pat^2); its actual error and the
   prediction sqrt(1 + 2 rho s1 s2 / (s1^2 + s2^2)) (s1, s2 = the actual sds) of the overconfidence from rho alone.
3. the full nonlinear solve (estimated pattern, kinetics, fringe posterior; simcode.solve.solve.solve_emitters) with
   (a) no factor: kap = 1, sig_a = 1, sig_g = 0, sig_ph = 1 (sigma = sqrt(Fisher^2 + start spread^2)), c as calibrated;
   (b) the same with c = 0; (c) kap only (no sig_a / sig_g / sig_ph); (d) the production calibration.
   zcal = robust sd of (r - r_gt) / sigma (both axes pooled); also the Fisher-only zcal and the fraction of variance that
   is start spread.
usage: python partb_gate.py <dump dir> <front> <scenes, comma> <calibration json> <out json> [seeds, default 301,302,303]
"""
import json, sys
import numpy as np, torch
from simcode.calibration import gate_stats as P
from simcode.solve import solve as SV, pattern as C

rob = lambda v: float(1.4826 * np.median(np.abs(v[np.isfinite(v)]))) if np.isfinite(v).any() else float('nan')
rms = lambda v: float(np.sqrt(np.mean(v[np.isfinite(v)] ** 2))) if np.isfinite(v).any() else float('nan')
EDG = P.EDG


def rcorr(a, b, clip=4.0):
    """Pearson correlation of a, b on the rows where both robust z-scores are < clip (robust to the gross tail)"""
    za, zb = a / rob(a), b / rob(b)
    m = np.isfinite(za) & np.isfinite(zb) & (np.abs(za) < clip) & (np.abs(zb) < clip)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 20 else float('nan')


@torch.no_grad()
def linear_pattern(R, ti, oi, patt='patt_true', c=0.0, ca=0.0, dev='cuda'):
    """linearised GLS of the intensities alone at the truth -> e_pat [n,2] (px), sigma_pat [n,2]"""
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float64)
    Pt = tt(R[patt])[torch.tensor(R['sc_b'][oi].astype(np.int64), device=dev)]           # [n,T,5]
    mu = tt(R['gt_I'][ti])                                                              # true mean intensities (incl. on-fractions)
    ill, dx, dy = C.illum_at(Pt, tt(R['gt_x'][ti]), tt(R['gt_y'][ti]))
    ang = C.angle_index(Pt)
    oh = torch.nn.functional.one_hot(ang, 2).double()
    ratio = mu / ill.clamp(min=1e-6)                                                    # B a_t
    J = torch.stack([ratio * dx, ratio * dy, ratio * ill * oh[..., 0] / 1.0, ratio * ill * oh[..., 1]], -1)   # d mu / d(x, y, logB0, logB1)
    L = tt(R['sc_L'][oi]); Sig = L @ L.transpose(1, 2) + 1e-3 * torch.eye(6, device=dev, dtype=torch.float64)
    if c > 0: Sig = Sig + torch.diag_embed((c * mu.clamp(min=0)) ** 2)
    if ca > 0: Sig = Sig + ca ** 2 * (mu.clamp(min=0)[:, :, None] * mu.clamp(min=0)[:, None, :]) * (oh @ oh.transpose(1, 2))
    W = torch.linalg.inv(Sig)
    F = J.transpose(1, 2) @ W @ J
    cov = torch.linalg.pinv(F)                                                           # an angle with the emitter off: its B is not identified
    d = (tt(R['sc_I'][oi]) - mu)[..., None]
    th = (cov @ J.transpose(1, 2) @ W @ d)[..., 0]
    return th[:, :2].cpu().numpy(), cov.diagonal(dim1=1, dim2=2)[:, :2].clamp(min=0).sqrt().cpu().numpy()


def nocal(cal, c=None, kap=False):
    """a copy of the calibration without the after-the-fact factors (sig_ph = 1, sig_a = 1, sig_g = 0; kap = 1 unless kap)"""
    k = dict(cal); k['sig_ph'] = dict(ph=[1.0, 1e6], f=[1.0, 1.0]); k['sig_a'] = 1.0; k['sig_g'] = 0.0
    if not kap:
        b = cal['kap_bins']; k['kap_bins'] = dict(ph=b['ph'], kap_p=[1.0] * len(b['ph']), kap_I2=[1.0] * len(b['ph']))
    if c is not None: k['cmult'] = c; k['cmult_ang'] = c
    return k


@torch.no_grad()
def solve_full(R, oi, cal, chunk=8192, dev='cuda'):
    """the production solve (estimated pattern) -> r [n,2], s_cal, s_fisher (MAP), spread_pm, s_pm"""
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    Pt = tt(R['patt_est'])
    out = {k: [] for k in ('r', 's_cal', 's_pm', 'spread')}
    for c0 in range(0, len(oi), chunk):
        j = oi[c0:c0 + chunk]
        o = dict(I=tt(R['sc_I'][j]), L=tt(R['sc_L'][j]), x=tt(R['sc_x'][j]), y=tt(R['sc_y'][j]), lsxy=tt(R['sc_lsxy'][j]),
                 x_det=tt(R['sc_x_det'][j]), y_det=tt(R['sc_y_det'][j]))
        b = torch.tensor(R['sc_b'][j], device=dev).long()
        I, Sig, r0, sp, ph = SV.solve_inputs(o, cal)
        s = SV.fringe_posterior(I, Sig, r0, sp, Pt[b], [r0], nf=1, box=cal['box'], thr=9.21, temp=cal['mix_T'],
                                cmult=float(cal['cmult']), cang=float(cal['cmult_ang']))
        qd = s['q'] / s['dof'].clamp(min=1)
        f = (cal['sig_a'] * qd.clamp(min=1.0) ** cal['sig_g'])[:, None]
        s_cal = (f * s['s_pm'] ** 2 + s['spread_pm'] ** 2).sqrt() * SV._binval(cal['sig_ph'], 'f', ph)[:, None]
        out['r'].append(s['r_pm']); out['s_cal'].append(s_cal); out['s_pm'].append(s['s_pm']); out['spread'].append(s['spread_pm'])
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def main():
    d, front, scen, calf, outf = sys.argv[1:6]
    seeds = tuple(int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else '301,302,303').split(','))
    cal = json.load(open(calf)); po = np.array(cal['prior_off'])
    res = dict(dump=d, front=front, scenes=scen, seeds=seeds, cal=calf, bins_ph=EDG, scenes_out={})
    for sc in scen.split(','):
        R = P.load(sc, front, seeds=seeds, d=d)
        ti, oi = P.targets(R)
        ph = R['sc_I'][oi].sum(1)
        gt = np.c_[R['gt_x'][ti], R['gt_y'][ti]]
        zI = np.linalg.solve(R['sc_L'][oi].astype(np.float64), (R['sc_I'][oi] - R['gt_I'][ti]).astype(np.float64)[..., None])[..., 0]
        e_pr = np.c_[R['sc_x'][oi], R['sc_y'][oi]] - po - gt
        s_pr = np.exp(R['sc_lsxy'][oi])
        e_pat, s_pat = linear_pattern(R, ti, oi)
        e_patc, s_patc = linear_pattern(R, ti, oi, c=float(cal['cmult']), ca=float(cal['cmult_ang']))
        e_pate, s_pate = linear_pattern(R, ti, oi, patt='patt_est')
        S = {k: solve_full(R, oi, c_) for k, c_ in (('raw', nocal(cal)), ('raw_c0', nocal(cal, c=0.0)), ('kap', nocal(cal, kap=True)), ('prod', cal))}
        rows = []
        for lo, hi in zip(EDG[:-1], EDG[1:]):
            m = (ph >= lo) & (ph < hi)
            if m.sum() < 100: continue
            z = zI[m]
            ev = np.linalg.eigvalsh(np.cov(np.clip(z, -6, 6).T))
            r = dict(ph_lo=lo, ph_hi=hi, ph_med=float(np.median(ph[m])), n=int(m.sum()),
                     zI_rob=rob(z.ravel()), kapI2_meas=rob(z.ravel()) ** 2, zI_rms_clip6=rms(np.clip(z, -6, 6).ravel()),
                     chi2_over_T_median=float(np.median((z ** 2).sum(1)) / 6 / (1 - 2 / 27) ** 3),   # median of chi2_6 / 6 = 0.891 -> scaled to 1
                     zI_cov_eig=[float(v) for v in ev],
                     zp_rob=rob((e_pr[m] / s_pr[m]).ravel()), zp_rms_clip6=rms(np.clip(e_pr[m] / s_pr[m], -6, 6).ravel()))
            # linearised pattern-only estimate and its correlation with the prior error
            for tag, ep, sp_ in (('pat', e_pat, s_pat), ('pat_c', e_patc, s_patc), ('pat_estpatt', e_pate, s_pate)):
                r[f'z{tag}_rob'] = rob((ep[m] / sp_[m]).ravel())
            for a, ax in enumerate('xy'):
                r[f'rho_{ax}'] = rcorr(e_pr[m, a], e_pat[m, a])
                r[f'rho_{ax}_c'] = rcorr(e_pr[m, a], e_patc[m, a])
            # independent linear combination (what the solve does, linearised), with the raw and the calibrated input sigmas
            for tag, sp_, st_ in (('raw', s_pr, s_patc), ('calin', s_pr * r['zp_rob'], s_patc * r['zpat_c_rob'])):
                w1 = 1 / sp_[m] ** 2; w2 = 1 / st_[m] ** 2
                ec = (w1 * e_pr[m] + w2 * e_patc[m]) / (w1 + w2); sc_ = 1 / np.sqrt(w1 + w2)
                r[f'lin_comb_{tag}_zcal'] = rob((ec / sc_).ravel())
            s1 = np.array([rob(e_pr[m, a]) for a in range(2)]); s2 = np.array([rob(e_patc[m, a]) for a in range(2)])
            rho = np.array([r['rho_x_c'], r['rho_y_c']])
            r['pred_inflation_from_rho'] = [float(v) for v in np.sqrt(1 + 2 * rho * s1 * s2 / (s1 ** 2 + s2 ** 2))]
            r['sd_prior_nm'] = [float(v * P.PX) for v in s1]; r['sd_pat_nm'] = [float(v * P.PX) for v in s2]
            # BLUE of the two estimates with their measured covariance (what a joint covariance would give)
            Ccov = [np.cov(np.c_[e_pr[m, a], e_patc[m, a]][(np.abs(e_pr[m, a]) < 4 * s1[a]) & (np.abs(e_patc[m, a]) < 4 * s2[a])].T) for a in range(2)]
            for a, ax in enumerate('xy'):
                Ci = np.linalg.inv(Ccov[a]); w = Ci.sum(1) / Ci.sum()
                eb = w[0] * e_pr[m, a] + w[1] * e_patc[m, a]
                wi = np.array([1 / Ccov[a][0, 0], 1 / Ccov[a][1, 1]]); wi /= wi.sum()
                ei = wi[0] * e_pr[m, a] + wi[1] * e_patc[m, a]
                r[f'blue_{ax}'] = dict(w_prior=float(w[0]), sd_nm=rob(eb) * P.PX, claimed_nm=float(np.sqrt(1 / Ci.sum())) * P.PX,
                                       indep_sd_nm=rob(ei) * P.PX, indep_claimed_nm=float(np.sqrt(1 / (1 / Ccov[a][0, 0] + 1 / Ccov[a][1, 1]))) * P.PX)
            # the full nonlinear solve
            for k, s in S.items():
                e = s['r'][m] - gt[m]
                r[f'solve_{k}_zcal'] = rob((e / s['s_cal'][m]).ravel())
                r[f'solve_{k}_zcal_fisher'] = rob((e / s['s_pm'][m]).ravel())
                r[f'solve_{k}_rms_nm'] = rob(e.ravel()) * P.PX
                r[f'solve_{k}_spread_var_frac'] = float(np.median((s['spread'][m] ** 2 / np.maximum(s['s_cal'][m] ** 2, 1e-12)).ravel()))
                r[f'solve_{k}_sig_med_nm'] = float(np.median(s['s_cal'][m]) * P.PX)
            rows.append(r)
        res['scenes_out'][sc] = rows
        print(f'== {sc} {front}: {len(oi)} matched targets', flush=True)
        print(f'{"ph":>10s} {"n":>6s} {"kI2":>6s} {"chi2/T":>6s} {"kp":>6s} {"zpat":>6s} {"zpatc":>6s} {"rho_x":>6s} {"rho_y":>6s} {"infl":>11s} '
              f'{"lin_raw":>7s} {"lin_cal":>7s} {"raw":>6s} {"raw_c0":>6s} {"kap":>6s} {"prod":>6s} {"sprd":>5s}')
        for r in rows:
            print(f'{r["ph_lo"]:>5g}-{r["ph_hi"]:<5g}{r["n"]:6d} {r["kapI2_meas"]:6.3f} {r["chi2_over_T_median"]:6.3f} {r["zp_rob"]:6.3f} '
                  f'{r["zpat_rob"]:6.3f} {r["zpat_c_rob"]:6.3f} {r["rho_x_c"]:6.3f} {r["rho_y_c"]:6.3f} {r["pred_inflation_from_rho"][0]:5.3f},{r["pred_inflation_from_rho"][1]:5.3f} '
                  f'{r["lin_comb_raw_zcal"]:7.3f} {r["lin_comb_calin_zcal"]:7.3f} {r["solve_raw_zcal"]:6.3f} {r["solve_raw_c0_zcal"]:6.3f} {r["solve_kap_zcal"]:6.3f} '
                  f'{r["solve_prod_zcal"]:6.3f} {r["solve_raw_spread_var_frac"]:5.2f}', flush=True)
    json.dump(res, open(outf, 'w'), indent=1)
    print('wrote', outf)


if __name__ == '__main__':
    main()
