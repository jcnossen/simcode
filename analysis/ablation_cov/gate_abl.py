"""Covariance ablation, part 2a: the solve variants (variants.py) on the gate scenes of the m1809 set (dumps out/psfmatch/fe_m1809a,
dense scene rs25qm in out/ablation_cov/fe_m1809a), fronts s1 (pass 1) and s2 (pass 2), no retraining.
Per variant the calibration is re-fitted exactly as the production one (analysis/psfmatch/fit_cal_set.py = plib.fit_cal: tuning seed 202 of
rs05qm + rs05pm, prior_off from iso; kap_p / kap_I2 per network-photon bin from the head's z-scores computed with the variant's own
covariance, then the sigma factor per photon bin from the solved z-scores); sig_a / sig_g (the chi2 inflation) are inherited from the base
calibration as in production. Check: variant A's refit must reproduce models/calib/calib_{sc_hd,p2L}_m1809.json.
Evaluation seeds 301-303 (302 max for rs25qm): detector-matched targets, resolvable (multi-emitter CRLB < 25 nm), per GT-photon bin:
RMS / CRLB (+ FOV-bootstrap 95 % CI), paired difference vs A on common targets (FOV bootstrap), trimmed, gross (|e| > 5 CRLB), bias (nm),
zcal (robust sd of e / sigma), sd of the clipped z, fraction |z| > 3, and the ranking use of sigma: RMS (nm) of the 45 % of rows with the
smallest sigma, Spearman(sigma, |e|).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/ablation_cov/gate_abl.py [fronts s2,s1] -> out/ablation_cov/gate_abl_<front>.json, calibs/"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'calib')); sys.path.insert(0, HERE)
import plib as P
import variants as V
from simcode import config
from simcode.util import proctitle
from scipy.stats import spearmanr
proctitle.set_title('covabl')
OUT = os.path.join(REPO, 'out', 'ablation_cov'); FE = os.path.join(REPO, 'out', 'psfmatch', 'fe_m1809a'); FE2 = f'{OUT}/fe_m1809a'
PROD = dict(s1='calib/calib_sc_hd_m1809.json', s2='calib/calib_p2L_m1809.json')
SCN = ['iso', 'pair2', 'pair1', 'rs05qm', 'rs05pm', 'rs25qm']
BINS = [(200, 2000), (200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9)]


def tr(R, v):
    R = dict(R); R['sc_L'] = V.L_variant(R['sc_L'], v['L']); R['sc_lsxy'] = V.lsxy_variant(R['sc_lsxy'], v)
    return R


def fit(front, name, v, cache={}):
    fn = f'{OUT}/calibs/calib_{front}_{name}.json'
    if os.path.exists(fn): return json.load(open(fn))
    if front not in cache:
        cache[front] = ([P.load(sc, front, seeds=(202,), d=FE) for sc in ('rs05qm', 'rs05pm')], P.load('iso', front, seeds=(202,), d=FE))
    Rs, iso = cache[front]
    base = V.cal_variant(json.load(open(config.resolve(PROD[front]))), v)
    cal = P.fit_cal([tr(R, v) for R in Rs], base, iso=tr(iso, v), verbose=False)
    if v.get('noprior'):       # prior sigma x 1000 (flat within the +-1.5 px box of each fringe start), then the sigma factor refitted as in fit_cal
        cal['kap_bins']['kap_p'] = [1000.0 * k for k in cal['kap_bins']['kap_p']]; cal.pop('sig_ph')
        zs, ph = [], []
        for R in Rs:
            ti, oi = P.targets(R); sol = P.solve(tr(R, v), oi, cal)
            zs.append((np.c_[sol['x'], sol['y']] - np.c_[R['gt_x'][ti], R['gt_y'][ti]]) / np.c_[sol['sx'], sol['sy']]); ph.append(R['sc_I'][oi].sum(1))
        zs, ph = np.concatenate(zs), np.concatenate(ph); C2, F = [], []
        for lo, hi in zip(P.EDG[:-1], P.EDG[1:]):
            m = (ph >= lo) & (ph < hi)
            if m.sum() < 100: continue
            C2.append(float(np.median(ph[m]))); F.append(P.rob(zs[m]))
        cal['sig_ph'] = dict(ph=C2, f=F)
    cal['note'] = f'covariance ablation variant {name} {v}: plib.fit_cal on seed 202 rs05qm + rs05pm (prior_off iso), base {PROD[front]}'
    os.makedirs(f'{OUT}/calibs', exist_ok=True); json.dump(cal, open(fn, 'w'), indent=1)
    return cal


def stats(R, E, A=None, nboot=200):
    ti = E['ti']; c = R['gt_sf_m'][ti]
    ok = np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * P.PX < 25) & np.isfinite(E['e']).all(1)
    ph = R['gt_phot'][ti]; fov = R['gt_fov'][ti]; out = {}
    for lo, hi in BINS:
        m = ok & (ph >= lo) & (ph < hi)
        if m.sum() < 50: continue
        e, s, cc = E['e'][m], E['s'][m], c[m]
        st = P.stats(e, cc, s)
        W = P.boot_w(fov[m], nboot)
        bs = [P.rr(e, cc, w) for w in W]
        st['ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        z = (e / s).ravel(); zc = z[np.abs(z) < 5]
        st.update(bias_nm=(e.mean(0) * P.PX).tolist(), bias_se_nm=(e.std(0) / np.sqrt(len(e)) * P.PX).tolist(), z_sd_clip5=float(zc.std()), z_gt3=float(100 * np.mean(np.abs(z) > 3)),
                  crlb_nm=float(np.sqrt(np.mean(cc ** 2)) * P.PX), med_sigma_nm=float(np.median(s) * P.PX))
        sg = s.mean(1); er = np.hypot(*e.T); k = sg <= np.quantile(sg, 0.45)
        st['rms45_nm'] = float(np.sqrt(np.mean(e[k] ** 2)) * P.PX); st['spearman'] = float(spearmanr(sg, er)[0])
        b45 = []
        for w in W[:100]:
            ww = w.astype(float); o = np.argsort(sg); cw = np.cumsum(ww[o]); kk = o[cw <= 0.45 * cw[-1]]
            b45.append(np.sqrt(np.sum(ww[kk, None] * e[kk] ** 2) / (2 * ww[kk].sum())) * P.PX)
        st['rms45_se'] = float(np.std(b45))
        if A is not None:      # paired on common targets (identical here: same candidates), same FOV bootstrap weights
            ea = A['e'][m]
            d = [P.rr(e, cc, w) - P.rr(ea, cc, w) for w in W]
            st['d_rms'] = P.rr(e, cc) - P.rr(ea, cc); st['d_ci'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            g = lambda x: 100 * np.mean(np.hypot(*(x / cc).T) > 5)
            st['d_c5'] = float(g(e) - g(ea))
        out[f'{lo:g}-{hi:g}'] = st
    return out


def main():
    fronts = (sys.argv[1] if len(sys.argv) > 1 else 's2,s1').split(',')
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    for front in fronts:
        prod = json.load(open(config.resolve(PROD[front])))
        res = dict(front=front, variants={}, calibs={})
        if only: res = json.load(open(f'{OUT}/gate_abl_{front}.json'))
        cals = {n: fit(front, n, V.VARS[n]) for n in V.ORDER}
        a = cals['A']
        chk = dict(kap_I2=float(np.max(np.abs(np.array(a['kap_bins']['kap_I2']) - prod['kap_bins']['kap_I2']))),
                   kap_p=float(np.max(np.abs(np.array(a['kap_bins']['kap_p']) - prod['kap_bins']['kap_p']))),
                   sig_f=float(np.max(np.abs(np.array(a['sig_ph']['f']) - prod['sig_ph']['f']))), prior_off=float(np.max(np.abs(np.array(a['prior_off']) - prod['prior_off']))))
        res['check_A_refit_vs_production_maxabs'] = chk
        print(front, 'variant A refit vs production calibration, max |diff|:', chk, flush=True)
        for n in V.ORDER:
            res['calibs'][n] = dict(kap_I2=cals[n]['kap_bins']['kap_I2'], kap_p=cals[n]['kap_bins']['kap_p'], sig_f=cals[n]['sig_ph']['f'], ph=cals[n]['sig_ph']['ph'])
            print(f'  cal {n:3s} kap_I2 {np.round(cals[n]["kap_bins"]["kap_I2"], 2)} sig_f {np.round(cals[n]["sig_ph"]["f"], 2)}', flush=True)
        for sc in SCN:
            d = FE2 if sc.startswith('rs25') else FE
            seeds = tuple(s for s in (301, 302, 303) if os.path.exists(f'{d}/{sc}__s{s}_{front}.npz'))
            if not seeds: continue
            R = P.load(sc, front, seeds=seeds, d=d); ti, oi = P.targets(R)
            EA = None
            for n in V.ORDER:
                if only and n != 'A' and n not in only: continue
                v = V.VARS[n]
                E = P.errs(R, ti, oi, P.solve(tr(R, v), oi, cals[n]))
                if n == 'A': EA = E
                st = stats(R, E, None if n == 'A' else EA)
                res['variants'].setdefault(n, {})[sc] = st
                for b in ('200-2000', '2000-4000'):
                    if b in st:
                        r = st[b]
                        print(f'{front} {sc:7s} {n:3s} {b:9s} n {r["n"]:5d} rms/C {r["rms"]:.3f} [{r["ci"][0]:.3f},{r["ci"][1]:.3f}] '
                              + (f'd {r["d_rms"]:+.3f} [{r["d_ci"][0]:+.3f},{r["d_ci"][1]:+.3f}] ' if 'd_rms' in r else ' ' * 30)
                              + f'trim {r["trim"]:.3f} c5 {r["c5"]:.2f}% zcal {r["zcal"]:.2f} z>3 {r["z_gt3"]:.1f}% bias {r["bias_nm"][0]:+.2f},{r["bias_nm"][1]:+.2f} '
                                f'rms45 {r["rms45_nm"]:.2f}+-{r["rms45_se"]:.2f} nm rho {r["spearman"]:.2f} nm {r["nm"]:.2f}', flush=True)
            json.dump(res, open(f'{OUT}/gate_abl_{front}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
