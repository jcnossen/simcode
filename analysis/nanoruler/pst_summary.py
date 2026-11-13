"""
Collects all numbers of the per-step pattern analysis into out/nanoruler/perstep/perstep.json (fit_summary.json, jitter.json,
lagcov.json, simcheck.json, cfree.json, compare.json + derived headline numbers), and the sensitivity of c to the additive
noise term.
"""
import sys, json, math
import numpy as np, torch
from scipy.optimize import minimize_scalar
sys.path.insert(0, '/home/jc/simcode2/analysis/nanoruler')
import pst_lib as L, pst_jitter as J, pst_cfree as C

NM = [219.9 / (2 * math.pi), 219.5 / (2 * math.pi)]
ld = lambda n: json.load(open(L.OUT + n))

if __name__ == '__main__':
    R = dict(fit=ld('fit_summary.json'), jitter=ld('jitter.json'), lagcov=ld('lagcov.json'), simcheck=ld('simcheck.json'),
             cfree=ld('cfree.json'), compare=ld('compare.json'), pairs=ld('pairs.json'))
    H = {}
    # ---- c versus the assumed additive noise a0 (a1 = 1), phase fixed (1 parameter per group) and phase free (2 parameters)
    F = torch.load(L.OUT + 'fits.pt', weights_only=False); D = L.load_data(); I = D['I']; g = np.unique(D['g'], return_inverse=True)[1]
    sens = {}
    for nm, cfg in (('shared', R['jitter']['shared_cfg']), ('perstep', R['jitter']['perstep_cfg'])):
        P = L.Pattern.from_state(F[cfg]['full']['state']); x, y, th = J.model_xy(D, P); C.Y['y'] = y
        for lab, lf in (('phase_fixed', J.loo), ('phase_free', C.loo2)):
            for a0 in (0.0, 300.0, 700.0, 1400.0):
                def nll(c2, keep=None):
                    o = lf(I, x, g, (a0, 1.0, c2)); k = o['ok'] if keep is None else (o['ok'] & keep)
                    return float((np.log(o['var'][k]) + o['r'][k] ** 2 / o['var'][k] / 0.9892).mean())     # 0.9892: 4-sigma truncation
                c2 = 0.05
                for _ in range(3):
                    o = lf(I, x, g, (a0, 1.0, c2)); keep = np.abs(o['r']) < 4 * np.sqrt(o['var'])
                    c2 = minimize_scalar(lambda v: nll(v, keep), bounds=(1e-4, 0.3), method='bounded').x
                sens[f'{nm}/{lab}/a0={a0:.0f}'] = dict(c=float(math.sqrt(c2)), nll=nll(c2, keep))
                print(nm, lab, a0, round(math.sqrt(c2), 4), round(nll(c2, keep), 4), flush=True)
    H['c_vs_a0'] = sens

    # ---- frame-common phase fluctuation (shared_3000 pattern; cross-site covariance), with the simulation calibration
    lc = R['lagcov']['shared_3000']['phase']; sim = R['simcheck']['common_ar1']
    H['phase_fluctuation'] = dict(
        cov_same_step_by_lag={k: lc[k]['same'] for k in lc}, cov_cross_step_by_lag={k: lc[k]['cross'] for k in lc},
        cov_cross_orientation_by_lag={k: lc[k]['xori'] for k in lc},
        sim_ar1_var0p003_tau5={k: dict(truth=sim[k]['truth_common'], same=sim[k]['same'], cross=sim[k]['cross']) for k in sim if k != 'truth'},
        note='the estimator over-reads a fluctuation common to the steps by about 1.1-1.45x at lags 0-5 (simcheck); the real curve '
             'is matched by an AR(1) fluctuation of variance ~0.003-0.004 rad^2 and correlation time ~5 cycles')
    v0 = lc['0-0']['same'][0]; r_ = [sim[k]['same'][0] / sim[k]['truth_common'] for k in ('0-0', '1-1', '2-2', '3-5')]
    est = [lc[k]['same'][0] / (sim[k]['same'][0] / sim[k]['truth_common']) / (sim[k]['truth_common'] / 0.003) for k in ('0-0', '1-1', '2-2', '3-5')]
    se = [lc[k]['same'][1] / (sim[k]['same'][0] / 0.003) for k in ('0-0', '1-1', '2-2', '3-5')]
    wv = 1 / np.array(se) ** 2; var0 = float(np.sum(wv * est) / wv.sum()); var0_se = float(max(1 / math.sqrt(wv.sum()), 0.0) * 1.5)
    H['phase_fluctuation'].update(var_lag0_calibrated=var0, var_lag0_calibrated_se=var0_se,
                                  sd_rad=math.sqrt(max(var0, 0)), sd_rad_se=var0_se / (2 * math.sqrt(max(var0, 1e-9))),
                                  sd_nm=math.sqrt(max(var0, 0)) * NM[0], sd_nm_se=var0_se / (2 * math.sqrt(max(var0, 1e-9))) * NM[0],
                                  var_lag0_note='lags 0-5 of the same-step covariance scaled to lag 0 with the simulated AR(1) (tau 5 cycles) response; '
                                                'se inflated 1.5x for the correlation between lag bins',
                                  white_part_lag0_minus_lag1=[lc['0-0']['same'][0] - lc['1-1']['same'][0], math.hypot(lc['0-0']['same'][1], lc['1-1']['same'][1])])
    # ---- step-relative variance: raw (2/3)(same - cross) per lag bin, pooled over the clean lags >= 3, calibrated by the simulation
    simr = R['simcheck']['common_ar1_plus_rel']; clean = ['3-5', '6-10', '11-20', '21-50', '51-100', '101-250', '251-500']
    resp = float(np.mean([simr[k]['var_rel'][0] for k in clean]) / simr['truth']['rel_var_actual'])
    simn = R['simcheck']['none']; bias0 = float(np.mean([simn[k]['var_rel'][0] for k in clean]))
    rel = {}
    for cfg in ('shared_3000', 'shared_1500'):
        for a in (0, 1):
            v = np.array([R['lagcov'][cfg]['phase'][k][f'var_rel{a}'][0] for k in clean]); s_ = np.array([R['lagcov'][cfg]['phase'][k][f'var_rel{a}'][1] for k in clean])
            w = 1 / s_ ** 2; raw = float(np.sum(w * v) / w.sum())
            se_ind = float(1 / math.sqrt(w.sum())); se_cons = float(s_.min())          # independent bins (optimistic) / fully correlated (conservative)
            up = max(raw + 1.645 * se_cons, 0) / resp
            rel[f'{cfg}/o{a}'] = dict(raw_var=raw, raw_se_independent_bins=se_ind, raw_se_conservative=se_cons, response=resp,
                                      var_calibrated=raw / resp, var_calibrated_se=se_cons / resp,
                                      sd_upper95_rad=math.sqrt(up), sd_upper95_nm=math.sqrt(up) * NM[a])
    H['step_relative'] = dict(lagcov=rel, response_from_sim=resp, sim_null_var_rel=bias0,
                              note='response: the leave-one-out amplitude absorbs part of a step-relative phase error, the raw (2/3)(same - cross) '
                                   'reads only this fraction of an injected slow step-relative variance (simcheck common_ar1_plus_rel)')
    # ---- implied relative intensity errors
    m = 0.88; f = J.implied_c(m, 1.0)
    sd = H['phase_fluctuation']['sd_rad']
    H['implied_c'] = dict(depth_assumed=m, factors=f,
                          from_phase_fluctuation={k: f[f'factor_{k}'] * sd for k in ('uniform', 'photon_weighted', 'selected')},
                          from_step_relative_upper95={k_: {k: f[f'factor_{k}'] * v['sd_upper95_rad'] for k in ('uniform', 'photon_weighted', 'selected')} for k_, v in rel.items()},
                          measured_c=dict(phase_fixed_shared=R['jitter']['noise']['shared']['ml_trim4_a1eq1']['c'],
                                          phase_fixed_perstep=R['jitter']['noise']['perstep']['ml_trim4_a1eq1']['c'],
                                          shared_minus_perstep=R['jitter']['noise']['shared_minus_perstep'],
                                          phase_free_shared=[R['cfree']['shared']['phase_free']['c'], R['cfree']['shared']['phase_free']['c_se']],
                                          phase_free_perstep=[R['cfree']['perstep']['phase_free']['c'], R['cfree']['perstep']['phase_free']['c_se']]))
    # ---- amplitude: same-step covariance at short lags minus its long-lag level (the per-emitter ratio bias is lag-independent)
    la = R['lagcov']['shared_3000']['amp']
    base = np.mean([la[k]['same'][0] for k in ('101-250', '251-500')]); base_se = 0.5 * math.hypot(la['101-250']['same'][1], la['251-500']['same'][1])
    va = la['0-0']['same'][0] - base; va_se = math.hypot(la['0-0']['same'][1], base_se)
    H['amplitude_jitter'] = dict(var=float(va), var_se=float(va_se), sd_upper95=float(math.sqrt(max(va + 1.645 * va_se, 0))),
                                 baseline_long_lag=float(base), cross_orientation_level=la['0-0']['xori'],
                                 note='baseline = squared mean per-emitter ratio bias of d = I / (A_loo x) - 1 (present at every lag and across orientations)')
    R['headline'] = H
    json.dump(R, open(L.OUT + 'perstep.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in H.items() if k != 'c_vs_a0'}, indent=1)[:6000])
