"""Solve-gap diagnostic (2026-11-14): where the residual of the c = 0 solve on the iso pass-2 rows sits, once B / C / D / E
(solvegap.py) put it in the network inputs. Per GT-photon bin, linearised at the truth (x, y, B), c = 0 solve weights:
  modulation slope s of the network intensities (regression of the per-emitter mean-removed I_net on the mean-removed gt_I)
  intensity-only GLS (no prior) realised error: network intensities vs oracle intensities (gt_I + N(0, Sigma_fr)), over SF CRLB
  the same for the network intensities rescaled by 1 / s about their mean (modulation-shrinkage correction)
  phase-locked bias: first-harmonic fit of the c = 0 solve error along each pattern direction vs the true pattern phase
Output out/solvegap/inputs_p<pass>.json.  usage: python analysis/solvegap/inputs.py [2,1]"""
import os, json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, 'out', 'crlbcmp', 'cache'); FREE = os.path.join(REPO, 'out', 'crlbcmp', 'crlbfree')
OUT = os.path.join(REPO, 'out', 'solvegap')
PX = 108.33
BINS = [(500, 1000), (1000, 2000), (2000, 4000), (4000, 10000), (10000, np.inf)]


def illum(pt, X, Y):
    arg = pt[..., 0] * X[:, None] + pt[..., 1] * Y[:, None] - pt[..., 2]
    s = -pt[..., 4] * pt[..., 3] * np.sin(arg)
    return pt[..., 4] * (1 + pt[..., 3] * np.cos(arg)), s * pt[..., 0], s * pt[..., 1], arg


def gls_I(eI, Ig, pt, X, Y, Sig):
    """intensity-only linear GLS (x, y, B) error"""
    il, dx, dy, _ = illum(pt, X, Y)
    J = np.stack([Ig * dx / il, Ig * dy / il, il], -1)
    Wi = np.linalg.inv(Sig)
    A = np.einsum('npi,npq,nqj->nij', J, Wi, J); b = np.einsum('npi,npq,nq->ni', J, Wi, eI)
    return np.linalg.solve(A, b[..., None])[..., 0][:, :2]


CALF = {1: 'archive_2d/m1809/calib/calib_emitter_pass1_sim.json', 2: 'archive_2d/m1809/calib/calib_emitter_pass2_sim.json'}


def main(ps):
    cal = json.load(open(os.path.join(REPO, 'models', CALF[ps])))
    rng = np.random.default_rng(1)
    rows = {k: [] for k in ('ph', 'sf', 'I', 'Ig', 'L', 'pt', 'gt', 'CI', 'es0', 'kap', 'ep')}
    for sd in (301, 302, 303):
        D = dict(np.load(f'{CACHE}/iso__s{sd}_p{ps}.npz')); F = dict(np.load(f'{FREE}/iso__s{sd}.npz'))
        tg = np.nonzero(D['gt_target'] > 0)[0]; j = D['m_h3'][tg]; ok = j >= 0; tg, j = tg[ok], j[ok]
        ph_net = D['c_I'][j].sum(1)
        kI2 = np.interp(np.log(np.maximum(ph_net, 1)), np.log(cal['kap_bins']['ph']), cal['kap_bins']['kap_I2'])
        rows['ph'].append(D['gt_phot'][tg]); rows['sf'].append(D['gt_sf_i'][tg]); rows['I'].append(D['c_I'][j]); rows['Ig'].append(D['gt_I'][tg])
        rows['L'].append(D['c_L'][j]); rows['pt'].append(D['patt_est'][D['c_b'][j].astype(int)]); rows['kap'].append(kI2)
        rows['gt'].append(np.c_[D['gt_x'][tg], D['gt_y'][tg]]); rows['CI'].append(F['cov_or'][tg][:, 2:, 2:])
        po = np.array(cal['prior_off'])
        rows['ep'].append(np.c_[D['c_x'][j], D['c_y'][j]] - po - np.c_[D['gt_x'][tg], D['gt_y'][tg]])
        rows['es0'].append(np.c_[D['c_xs0'][j], D['c_ys0'][j]] - np.c_[D['gt_x'][tg], D['gt_y'][tg]])
    R = {k: np.concatenate(v).astype(np.float64) for k, v in rows.items()}
    Sig = R['L'] @ R['L'].transpose(0, 2, 1) * R['kap'][:, None, None] + 1e-3 * np.eye(6)
    CI = 0.5 * (R['CI'] + R['CI'].transpose(0, 2, 1))
    Io = R['Ig'] + (np.linalg.cholesky(CI) @ rng.standard_normal((len(CI), 6, 1)))[..., 0]
    X, Y = R['gt'][:, 0], R['gt'][:, 1]
    il, dx, dy, arg = illum(R['pt'], X, Y)
    ang = (np.abs(R['pt'][..., 0]) < np.abs(R['pt'][..., 1])).astype(int)
    out = {}
    for lo, hi in BINS:
        m = (R['ph'] >= lo) & (R['ph'] < hi)
        a = R['I'][m] - R['I'][m].mean(1, keepdims=True); g = R['Ig'][m] - R['Ig'][m].mean(1, keepdims=True)
        s = float((a * g).sum() / (g * g).sum())
        res = a - s * g
        cr = lambda e: [float(v) for v in np.sqrt((e ** 2).mean(0)) / np.sqrt((R['sf'][m] ** 2).mean(0))]
        e_net = gls_I(R['I'][m] - R['Ig'][m], R['Ig'][m], R['pt'][m], X[m], Y[m], Sig[m])
        e_or = gls_I(Io[m] - R['Ig'][m], R['Ig'][m], R['pt'][m], X[m], Y[m], CI[m])
        e_nor = gls_I(Io[m] - R['Ig'][m], R['Ig'][m], R['pt'][m], X[m], Y[m], Sig[m])     # oracle data, network weights
        Ic = R['I'][m].mean(1, keepdims=True) + a / s                                     # modulation-shrinkage corrected
        e_cor = gls_I(Ic - R['Ig'][m], R['Ig'][m], R['pt'][m], X[m], Y[m], Sig[m] / s ** 2)
        # per-frame: mean-removed residual (after the slope) vs the free-model CRLB of the mean-removed intensities
        P = np.eye(6) - 1.0 / 6
        cr_res = np.sqrt(np.einsum('ij,njk,ki->n', P, CI[m], P).mean() / 6)
        # phase-locked bias of the c = 0 solve error along each pattern direction (first harmonic in the angle's mean phase)
        pl = {}
        for A_ in (0, 1):
            sel = ang[m] == A_
            k = (R['pt'][m][..., :2] * sel[..., None]).sum(1) / sel.sum(1, keepdims=True)
            kn = k / np.linalg.norm(k, axis=1, keepdims=True)
            e_par = (R['es0'][m] * kn).sum(1)
            ph0 = np.angle(np.exp(1j * arg[m])[sel].reshape(m.sum(), -1).mean(1))     # phase of the angle's first frame set
            Xd = np.c_[np.ones(m.sum()), np.sin(ph0), np.cos(ph0)]
            c = np.linalg.lstsq(Xd, e_par, rcond=None)[0]
            pl[f'ang{A_}'] = dict(amp_nm=float(np.hypot(c[1], c[2]) * PX), rms_e_par_nm=float(np.sqrt((e_par ** 2).mean()) * PX))
        # prior vs intensity-only position: correlation, and the best linear combination of the two with their empirical
        # 2 x 2 covariance per axis (what a solve that knows the prior-intensity correlation could reach)
        ep = R['ep'][m]; rho, comb, indep = [], [], []
        for ax in (0, 1):
            Cm = np.cov(np.c_[ep[:, ax], e_net[:, ax]].T, bias=True) + np.outer([ep[:, ax].mean(), e_net[:, ax].mean()], [ep[:, ax].mean(), e_net[:, ax].mean()])
            rho.append(float(Cm[0, 1] / np.sqrt(Cm[0, 0] * Cm[1, 1])))
            w = np.linalg.solve(Cm, np.ones(2)); w = w / w.sum()
            comb.append(float(np.sqrt(np.mean((w[0] * ep[:, ax] + w[1] * e_net[:, ax]) ** 2)) / np.sqrt(np.mean(R['sf'][m][:, ax] ** 2))))
            wi = np.array([1 / Cm[0, 0], 1 / Cm[1, 1]]); wi = wi / wi.sum()
            indep.append(float(np.sqrt(np.mean((wi[0] * ep[:, ax] + wi[1] * e_net[:, ax]) ** 2)) / np.sqrt(np.mean(R['sf'][m][:, ax] ** 2))))
        prior_r = cr(ep)
        out[f'{lo:g}-{hi:g}'] = dict(n=int(m.sum()), prior=prior_r, rho_prior_Ionly=rho, comb_opt=comb, comb_indep=indep, mod_slope=s, resid_rms_over_crlb_fr=float(np.sqrt((res ** 2).mean()) / cr_res),
                                     Ionly_net=cr(e_net), Ionly_oracle=cr(e_or), Ionly_oracle_netweights=cr(e_nor),
                                     Ionly_net_slopecorr=cr(e_cor), phase_bias=pl)
    json.dump(out, open(f'{OUT}/inputs_p{ps}.json', 'w'), indent=1)
    for k, v in out.items(): print(k, json.dumps(v))


if __name__ == '__main__':
    import sys
    for ps in [int(a) for a in (sys.argv[1] if len(sys.argv) > 1 else '2,1').split(',')]: main(ps)
