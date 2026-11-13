"""Global precision threshold sweep on the three real movies (fair FRC protocol of analysis/paper/frc_fair.py).

Base rows: out/selection_global/cache/<ds>.npz (extract.py) = pass-2 rows with a per-tile rank (p >= 0.63, inside the crop).
Rules compared, each on those base rows:
  u045            strict_u <= 0.45 (the earlier paper rule)
  max <= s        max(sx, sy) <= s nm (calibrated sigma after the sig_u rescale)
  mean <= s       (sx + sy) / 2 <= s nm
  match_max/mean  the global threshold that keeps the same count as u045
FRC: frc_fair.evaluate (disjoint windows, max sigma < 0.15 px, crop, alternating 300-frame halves). Drift: for the sweep ONE drift
per movie, DME v2 on the u045 set (cached; the drift is a property of the sample, so the sweep isolates the selection); the
chosen value is re-evaluated with its own drift by the figure script. Also: occupied 20-nm bins of the FRC set (coverage),
median sigma, median photons.
usage: CUDA_VISIBLE_DEVICES=1 python analysis/selection_global/sweep_real.py -> out/selection_global/sweep_real.json"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'analysis', 'paper')]
import frc_fair as FF
from simcode.util.proctitle import set_title

PX = FF.PX
NFR = {'1809_2': 120000, '2.5nM_2': 180000, '20nM_2': 120000}
GRID = np.round(np.arange(5.0, 16.01, 0.5), 2)
OUT = f'{REPO}/out/selection_global'


def table(z, m):
    return dict(x=z['x'][m].astype(np.float64), y=z['y'][m].astype(np.float64), f=z['frame'][m].astype(np.int64),
                sx=z['sx'][m].astype(np.float64), sy=z['sy'][m].astype(np.float64), photons=z['photons'][m].astype(np.float64))


def evaluate(T, F, drift):
    keep = T['f'] + 6 <= F
    T = {k: v[keep] for k, v in T.items()}
    R = FF.evaluate(T, F, drift_cache=drift, secondary=False)
    m = R['mask']
    nb = int(PX * 110 / 20)
    ix = np.clip(((R['x_dc'][m] - 5) / 110 * nb).astype(int), 0, nb - 1); iy = np.clip(((R['y_dc'][m] - 5) / 110 * nb).astype(int), 0, nb - 1)
    s = 0.5 * (T['sx'][m] + T['sy'][m]) * PX
    return dict(frc_nm=float(R['frc_nm']), n_all=int(len(T['x'])), n_frc=int(m.sum()), bins20=int(len(np.unique(iy * nb + ix))),
                median_sigma_nm=float(np.median(s)) if m.any() else np.nan,
                median_photons=float(np.median(T['photons'][m])) if m.any() else np.nan)


def main():
    set_title('sel-sweepreal')
    res = {}
    fn = f'{OUT}/sweep_real.json'
    if os.path.exists(fn): res = json.load(open(fn))
    for ds, F in NFR.items():
        z = dict(np.load(f'{OUT}/cache/{ds}.npz'))
        smean = 0.5 * (z['sx'] + z['sy']) * PX; smax = np.maximum(z['sx'], z['sy']) * PX
        u45 = z['strict_u'] <= 0.45
        drift = f'{OUT}/cache/{ds}_u045_drift.npy'
        R = res.get(ds, {})
        t0 = time.time()
        if 'u045' not in R: R['u045'] = evaluate(table(z, u45), F, drift)
        n = int(u45.sum())
        for meas, s in (('max', smax), ('mean', smean)):
            thr = float(np.sort(s)[n - 1])
            if f'match_{meas}' not in R:
                m = s <= thr
                R[f'match_{meas}'] = dict(evaluate(table(z, m), F, drift), thr_nm=thr, overlap_u045=float((m & u45).sum() / n))
            for t in GRID:
                k = f'{meas}<={t:g}'
                if k in R: continue
                m = s <= t
                R[k] = dict(evaluate(table(z, m), F, drift), thr_nm=float(t), overlap_u045=float((m & u45).sum() / max(n, 1)),
                            frac_base=float(m.mean()))
                print(ds, k, json.dumps(R[k]), flush=True)
        res[ds] = R
        json.dump(res, open(fn, 'w'), indent=1)
        print(ds, 'done', f'{time.time() - t0:.0f} s', json.dumps(R['u045']), json.dumps(R['match_max']), json.dumps(R['match_mean']), flush=True)


if __name__ == '__main__':
    main()
