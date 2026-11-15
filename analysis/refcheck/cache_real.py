"""Referee checks, real data: cache each method's fair-FRC rows of Fig 3 (1809_2) / Fig 4 (2.5nM_2) once (the tables on
/mnt/data are read a single time).

Rows = exactly the figure's rows (analysis/paper/fig_exp_new.py via out/paper_v2psf/run_ds.sh): frc_fair.load of the figure's
table paths (values json 'paths'; SIMCODE with v2_sigma_max = 12 nm), window end within the movie, the disjoint windows of the
common crop (frc_fair.masks); flag `filt` = the FRC set (max sigma < 0.15 px); the drift = the figure's own cached DME v2
drift per method (drift_info of the values json), applied at window start + 3 as frc_fair.evaluate.
Check: the 300-frame-halves FRC of the cached FRC set reproduces the figure's frc_nm.
usage: python analysis/refcheck/cache_real.py <1809_2|2.5nM_2> [--extra name=kind:path ...]
output: out/refcheck/cache/<ds>_<method>.npz (x, y, f, sx, sy, photons, xdc, ydc, filt; disjoint rows only), <ds>_check.json"""
import os, sys, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'analysis', 'paper')]
import frc_fair as FF

OUT = os.path.join(REPO, 'out', 'refcheck', 'cache')
FIG = os.path.join(REPO, 'out', 'paper_figs_v2')
VALUES = {'1809_2': f'{FIG}/1809_2_fig3_psfv2_values.json', '2.5nM_2': f'{FIG}/2.5nM_2_fig4_psfv2_values.json'}
KINDS = ['smlm', 'sf', 'decode', 'v2']


def cache_method(ds, name, kind, path, F, drift_file, v2_sigma_max=12.0):
    t0 = time.time()
    T = FF.load(kind, path, **(dict(v2_sigma_max=v2_sigma_max) if kind == 'v2' else {}))
    keep = T['f'] + 6 <= F
    T = {k: v[keep] for k, v in T.items()}
    disj, filt = FF.masks(T)
    drift = np.load(drift_file) if drift_file and os.path.exists(drift_file) else None
    ti = np.clip(T['f'] + 3, 0, (len(drift) if drift is not None else F) - 1)
    dx, dy = (drift[ti, 0], drift[ti, 1]) if drift is not None else (0 * T['x'], 0 * T['y'])
    i = np.nonzero(disj)[0]
    Z = dict(x=T['x'][i], y=T['y'][i], f=T['f'][i], sx=T['sx'][i], sy=T['sy'][i], photons=T['photons'][i],
             xdc=(T['x'] - dx)[i], ydc=(T['y'] - dy)[i], filt=filt[i])
    fn = os.path.join(OUT, f'{ds}_{name}.npz'); np.savez(fn, **Z)
    m = Z['filt']
    res = FF.frc_halves(Z['xdc'][m], Z['ydc'][m], Z['f'][m])[0]
    print(f'{ds} {name}: n_all {len(T["x"])} disj {len(i)} frc set {m.sum()} FRC {res:.3f} nm ({time.time() - t0:.0f} s)', flush=True)
    return dict(path=path, drift=drift_file, n_all=int(len(T['x'])), n_disjoint=int(len(i)), n_frc=int(m.sum()), frc_nm=float(res))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ds', choices=list(VALUES))
    ap.add_argument('--extra', nargs='*', default=[], help='name=kind:path[@drift.npy] (no figure drift: drift estimated here)')
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    V = json.load(open(VALUES[a.ds])); F = V['nframes']
    chk_fn = os.path.join(OUT, f'{a.ds}_check.json')
    chk = json.load(open(chk_fn)) if os.path.exists(chk_fn) else {}
    for j, k in enumerate(KINDS):
        if a.only and k not in a.only.split(','): continue
        r = cache_method(a.ds, k, k, V['paths'][k], F, V['drift_info'][j]['cached'])
        r['figure_frc_nm'] = V['frc_nm'][j]; r['figure_n_frc'] = V['n_frc'][j]
        r['match'] = bool(abs(r['frc_nm'] - V['frc_nm'][j]) < 1e-6 and r['n_frc'] == V['n_frc'][j])
        chk[k] = r
    for spec in a.extra:
        name, kp = spec.split('=', 1); kind, path = kp.split(':', 1)
        path, _, dfile = path.partition('@')
        chk[name] = cache_method(a.ds, name, kind, path, F, dfile or None)
    json.dump(chk, open(chk_fn, 'w'), indent=1)


if __name__ == '__main__':
    main()
