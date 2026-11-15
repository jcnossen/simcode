"""Apply the p_valid model (pvalid.py, net set m1809) to a real movie's pass-2 table and compare the selections p_valid >= t
with the paper's rule sigma <= 12 nm (simcode.table.select; it replaced u <= 0.45, removed 2026-11-14): fraction kept, FRC (analysis/paper/frc_fair.py protocol incl. each
selection's own DME drift, crop 5-115 px, disjoint windows, sigma < 0.15 px) and the GT-free revisit consistency per p_valid bin
(add_pvalid.revisit). The same revisit statistic on a simulated held-out movie, next to its GT validity, shows what the
statistic means. On real data the calibration of p_valid itself cannot be verified (no ground truth).
usage: CUDA_VISIBLE_DEVICES=2 python analysis/papersim/pvalid_real.py <locs_p2.npz> <nframes> [--model ...] [--sim D2]
output: out/papersim/selection/pvalid_real_<dataset>.json"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'analysis', 'paper'))
import common as C
import pvalid as PV
import add_pvalid as AP


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('locs'); ap.add_argument('nframes', type=int)
    ap.add_argument('--model', default=os.path.join(PV.OUTD, 'pvalid_m1809.joblib')); ap.add_argument('--sim', default='D2'); ap.add_argument('--name', default=None)
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('psim-pvreal')
    import frc_fair
    name = a.name or os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(a.locs)))) + '_' + os.path.basename(os.path.dirname(os.path.abspath(a.locs)))
    z, pv, M = AP.apply(a.locs, a.model)
    out = dict(locs=a.locs, model=a.model, n_rows=int(len(pv)), p_valid_quantiles=dict(zip(['p5', 'p25', 'p50', 'p75', 'p95'], np.percentile(pv, [5, 25, 50, 75, 95]).round(4).tolist())),
               hist=np.histogram(pv, bins=np.linspace(0, 1, 21))[0].tolist(), selections={})
    T = dict(x=z['x'].astype(np.float64), y=z['y'].astype(np.float64), f=z['frame'].astype(np.int64), sx=z['sx'].astype(np.float64),
             sy=z['sy'].astype(np.float64), photons=z['photons'].astype(np.float64))
    sels = {f'p_valid>={t}': pv >= t for t in (0.9, 0.95, 0.99)}
    from simcode import table as SP
    sels['sigma<=12nm'] = SP.select(z, SP.SIGMA_MAX_NM, crop=SP.run_crop(os.path.dirname(os.path.abspath(a.locs))))
    sels['sel (v2 all, reference)'] = z['sel'].astype(bool)
    for k, m in sels.items():
        Ts = {kk: v[m] for kk, v in T.items()}
        R = frc_fair.evaluate(Ts, a.nframes, drift_cache=os.path.join(PV.OUTD, 'cache', f'drift_{name}_{k.replace(">=", "ge").replace("<=", "le").replace(" ", "_")}.npy'))
        out['selections'][k] = dict(kept=float(m.mean()), n=int(m.sum()), mean_p_valid=float(pv[m].mean()), frc_nm=R['frc_nm'], n_frc=R['n_frc'],
                                    frc_nm_nofilter=R.get('frc_nm_nofilter'))
        print(name, k, json.dumps(out['selections'][k]), flush=True)
    out['revisit_real'] = AP.revisit(z, pv)
    # the same statistic on a simulated held-out movie, with its GT validity per bin
    if a.sim:
        ds = PV.datasets('m1809')[a.sim]
        zs, Xs, Ls = PV.load(ds, os.path.join(PV.OUTD, 'cache'))
        import joblib; pvs = PV.predict(joblib.load(a.model)['valid'], Xs)
        rv = AP.revisit(zs, pvs)
        for r in rv:
            lo, hi = [float(v) for v in r['p_valid_min'].split('-')]
            m = (pvs >= lo) & (pvs < (hi if hi < 1 else 1.0001))
            r['gt_valid_rows_in_bin'] = float(Ls['valid'][m].mean())
        out['revisit_sim'] = dict(movie=a.sim, bins=rv)
    json.dump(out, open(os.path.join(PV.OUTD, f'pvalid_real_{name}.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('selections',)}, default=float)[:3000])


if __name__ == '__main__':
    main()
