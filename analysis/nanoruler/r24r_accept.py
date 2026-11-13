"""R24 reuse check, part C (analysis): acceptance numbers of the nr80 network front on gate scenes rendered with ANOTHER movie's calib
(out/nanoruler/<name>/reuse/fe_<name>, made by r24r_gates.sh) against the same front on the nr80 scenes (out/nanoruler/eval/fe_nr80).
UNPAIRED (different scenes): per front its own detector-matched targets, as analysis/psfmatch/m20check_gates.py own() /
analysis/nanoruler/ev_accept.py (plib: targets = score >= 0.5 matched 1 px on the detector point, explicit pattern solve, RMS /
CRLB_SF_multi of the resolvable targets (CRLB < 25 nm), FOV-bootstrap 95 % CI; plib.PX = 65 nm; bands = GT photons per window).
Calibrations:  <f>  = the nr80 sim calibration as is;  <f>r = the same dumps with a calibration REFIT on seed 202 of THESE scenes
(plib.fit_cal on rs05qm + iso, prior_off from iso, as ev_fitcal.py with EV_FIT=rs05qm,iso). zcal(<f>) vs zcal(<f>r) and the ratio
of the fitted tables tell whether the sigma calibration transfers.
modes (env R24R_FRONT = s1 (default, pass 1) | s2 (pass 2); GPU for the solve unless R24R_DEV=cpu):
  python r24r_accept.py fit   <name>            -> out/nanoruler/<name>/reuse/calib_<f>_refit_<name>.json
  python r24r_accept.py scene <name|nr80> <sc>  -> out/nanoruler/<name>/reuse/accept_<f>_<sc>.json (nr80: out/nanoruler/nr40/reuse/accept_ref_nr80_<f>_<sc>.json)
  python r24r_accept.py table <name>            -> out/nanoruler/<name>/reuse/accept_<f>_<name>.txt (CPU, from the jsons)"""
import functools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/jc/simcode2'); sys.path.insert(0, os.path.join(HERE, '..', 'calib')); sys.path.insert(0, os.path.join(HERE, '..', 'psf3d_nets'))
import plib as P
import accept as A
from simcode2 import config
from simcode2.util.proctitle import set_title
set_title('nr-r24r')
P.PX = 65.0
P.solve = functools.partial(P.solve, dev=os.environ.get('R24R_DEV', 'cuda'))
ROOT = '/home/jc/simcode2/out/nanoruler'
F = os.environ.get('R24R_FRONT', 's1')
CAL80 = {'s1': f'{ROOT}/models/calib/calib_sc_hd_nr80.json', 's2': f'{ROOT}/models/calib/calib_p2L_nr80.json'}[F]
BASE = {'s1': 'calib/calib_sc_hd_sc5nbb15_pm_bright.json', 's2': 'calib/calib_p2L.json'}[F]
BINS = ['200-2000', '800-2500', '200-400', '400-700', '700-1200', '1200-2000', '2000-4000']
SCN = ['iso', 'rs05qm', 'pair1', 'pair2']
fe = lambda n: f'{ROOT}/eval/fe_nr80' if n == 'nr80' else f'{ROOT}/{n}/reuse/fe_{n}'
refit_fn = lambda n: f'{ROOT}/{n}/reuse/calib_{F}_refit_{n}.json'
scene_fn = lambda n, sc: f'{ROOT}/nr40/reuse/accept_ref_nr80_{F}_{sc}.json' if n == 'nr80' else f'{ROOT}/{n}/reuse/accept_{F}_{sc}.json'


def own(R, E, nboot=200):
    ti = E['ti']; c = R['gt_sf_m'][ti]; ok = np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * P.PX < 25) & np.isfinite(E['e']).all(1)
    ph = R['gt_phot'][ti]; fov = R['gt_fov'][ti]; out = {}
    for b in BINS:
        lo, hi = [float(v) for v in b.split('-')]
        m = ok & (ph >= lo) & (ph < hi)
        if m.sum() < 20: continue
        s = P.stats(E['e'][m], c[m], E['s'][m])
        W = P.boot_w(fov[m], nboot)
        bs = [P.rr(E['e'][m], c[m], w) for w in W]
        s['ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; s['se'] = float(np.std(bs))
        zb = [P.rob(np.repeat(E['e'][m] / E['s'][m], w, 0)) for w in W[:100]]
        s['zcal_ci'] = [float(np.percentile(zb, 2.5)), float(np.percentile(zb, 97.5))]
        s['crlb_nm'] = float(np.sqrt(np.mean(c[m] ** 2)) * P.PX); s['bias_nm'] = (E['e'][m].mean(0) * P.PX).tolist()
        s['sigma_nm_median'] = float(np.median(E['s'][m]) * P.PX)
        out[b] = s
    return out


def main():
    mode, name = sys.argv[1], sys.argv[2]
    if mode == 'fit':
        d = fe(name); B = json.load(open(config.resolve(BASE)))
        Rs = [P.load(sc, F, seeds=(202,), d=d) for sc in ('rs05qm', 'iso')]
        cal = P.fit_cal(Rs, B, iso=Rs[1])
        cal['note'] = (f'R24 reuse check (r24r_accept.py fit = plib.fit_cal as ev_fitcal.py): sim-sigma calibration of the nr80 front {F} REFIT on seed 202 of the '
                       f'{name} gate scenes (rs05qm + iso; prior_off from iso); base {BASE}. Diagnostic only: NOT a production calibration.')
        json.dump(cal, open(refit_fn(name), 'w'), indent=1); print('wrote', refit_fn(name)); return
    if mode == 'scene':
        sc = sys.argv[3]; d = fe(name); res = {}
        cal = json.load(open(CAL80))
        R, E, S = A.run_front(sc, F, d, cal)
        st = own(R, E); dt = A.det_sweep(R, S)
        g = P.detect(R, S['x'], S['y'], R['sc_score'] >= 0.63); dt['at0.63'] = {k: v for k, v in g.items() if k != 'hit'}
        tgt = R['gt_target'] > 0
        res[F] = dict(stats=st, det=dt, n_fov=int(R['n_fov']), n_targets=int(tgt.sum()), n_cand=int(len(R['sc_score'])),
                      target_phot_pct=np.percentile(R['gt_phot'][tgt], [5, 50, 95]).tolist())
        if name != 'nr80' and os.path.exists(refit_fn(name)):
            ti, oi = P.targets(R)
            E2 = P.errs(R, ti, oi, P.solve(R, oi, json.load(open(refit_fn(name)))))
            res[F + 'r'] = dict(stats=own(R, E2))
        json.dump(res, open(scene_fn(name, sc), 'w'), indent=1, default=float); print('wrote', scene_fn(name, sc)); return
    # table
    L = [f'# {name}: the nr80 front {F} on gate scenes rendered with calib_{name}.json (PSF, read-noise map, k), eval seeds 301-303, against the same front on the nr80 scenes.',
         f'# UNPAIRED; {F} = nr80 sim calibration ({os.path.basename(CAL80)}) as is, {F}r = calibration refit on seed 202 of the {name} scenes. RMS/CRLB = RMS error / multi-emitter SIMFLUX CRLB, '
         'FOV-bootstrap 95 % CI; d = this - nr80 +- 2 se. Bands = GT photons per 6-frame window. Detection on the final positions (0.5 px).']
    out = {}
    for sc in SCN:
        if not (os.path.exists(scene_fn(name, sc)) and os.path.exists(scene_fn('nr80', sc))): L.append(f'\n== {sc}: missing'); continue
        a = json.load(open(scene_fn(name, sc))); r = json.load(open(scene_fn('nr80', sc))); out[sc] = dict(this=a, nr80=r)
        L.append(f'\n== {sc}  ({name}: {a[F]["n_targets"]} targets in {a[F]["n_fov"]} FOVs, target photons p5/p50/p95 {np.round(a[F]["target_phot_pct"]).tolist()}; nr80: {r[F]["n_targets"]} in {r[F]["n_fov"]})')
        for b in BINS:
            x = a[F]['stats'].get(b); y = r[F]['stats'].get(b); xr = a.get(F + 'r', {}).get('stats', {}).get(b)
            if not x or not y: continue
            d = x['rms'] - y['rms']; se = float(np.hypot(x['se'], y['se']))
            L.append(f'  {b:>9s}  {name} n {x["n"]:5d} RMS/CRLB {x["rms"]:.3f} [{x["ci"][0]:.3f}, {x["ci"][1]:.3f}] ({x["nm"]:.2f} nm, CRLB {x["crlb_nm"]:.2f}, c5 {x["c5"]:.2f} %)'
                     f' | nr80 n {y["n"]:5d} {y["rms"]:.3f} [{y["ci"][0]:.3f}, {y["ci"][1]:.3f}] ({y["nm"]:.2f} nm, CRLB {y["crlb_nm"]:.2f}, c5 {y["c5"]:.2f} %)'
                     f' | d {d:+.3f} +- {2 * se:.3f}'
                     f' | zcal {name} {x["zcal"]:.3f} [{x["zcal_ci"][0]:.3f}, {x["zcal_ci"][1]:.3f}]  refit {xr["zcal"] if xr else float("nan"):.3f}  nr80 {y["zcal"]:.3f} [{y["zcal_ci"][0]:.3f}, {y["zcal_ci"][1]:.3f}]'
                     + (f' | refit RMS/CRLB {xr["rms"]:.3f}' if xr else ''))
        for thr in ('at0.5', 'at0.63'):
            x, y = a[F]['det'][thr], r[F]['det'][thr]
            L.append(f'  detect {thr[2:]:>4s}: {name} recall {x["recall"]:.4f} FP {x["FP"]} ({x["fp_per_fov"]:.3f} / FOV) | nr80 recall {y["recall"]:.4f} FP {y["FP"]} ({y["fp_per_fov"]:.3f} / FOV)'
                     f' | d recall {x["recall"] - y["recall"]:+.4f}')
    if os.path.exists(refit_fn(name)):
        c0 = json.load(open(CAL80)); c1 = json.load(open(refit_fn(name)))
        L.append(f'\n== calibration tables: refit on the {name} scenes / nr80 calibration (network-photon bins of the nr80 table; log-photon interpolation of the refit)')
        ph = np.array(c0['sig_ph']['ph']); it = lambda c, t, k: np.interp(np.log(ph), np.log(c[t]['ph']), c[t][k])
        L.append('  network photons   ' + '  '.join(f'{p:7.0f}' for p in ph))
        for t, k in (('sig_ph', 'f'), ('kap_bins', 'kap_p'), ('kap_bins', 'kap_I2')):
            L.append(f'  {k:8s} nr80     ' + '  '.join(f'{v:7.3f}' for v in it(c0, t, k)))
            L.append(f'  {k:8s} refit    ' + '  '.join(f'{v:7.3f}' for v in it(c1, t, k)))
            L.append(f'  {k:8s} ratio    ' + '  '.join(f'{v:7.3f}' for v in it(c1, t, k) / it(c0, t, k)))
        L.append(f'  prior_off nm: nr80 {(np.array(c0["prior_off"]) * 65).round(3).tolist()}  refit {(np.array(c1["prior_off"]) * 65).round(3).tolist()}')
        out['calib'] = dict(nr80=c0, refit=c1)
    fn = f'{ROOT}/{name}/reuse/accept_{F}_{name}'
    json.dump(out, open(fn + '.json', 'w'), indent=1, default=float); open(fn + '.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
