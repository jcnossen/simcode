"""Global precision threshold sweep on the paper simulations with ground truth (SIMCODE v2 pass-2 tables locs_p2_sim.npz).

Base rows = rows with a per-tile rank (strict_u finite: p >= 0.63, inside the field) = the rows u <= 0.45 ranks. Rules: u045,
max(sx, sy) <= s, (sx + sy) / 2 <= s (calibrated sigma, nm). Metrics as analysis/papersim/metrics.py density_metrics:
  n_all (rows of the rule), n_frc (fair FRC set: disjoint windows, max sigma < 0.15 px, whole FOV, no drift), FRC nm,
  Jaccard (preprint definition, 10 nm, all rows of the rule, whole movie = metrics' jaccard_end), GT match on the FRC set
  (RMS x / y nm, fraction of errors > 30 nm, recall of GT emitters >= 150 ph).
Moving movies (--moving): per-1000-frame-block Jaccard at 10 / 20 / 40 nm and per-block FRC (moving_metrics), coarse grid.
usage: python analysis/selection_global/sweep_sim.py [--runs D1,...] [--moving] -> out/selection_global/sweep_sim.json"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'analysis', 'papersim'), os.path.join(REPO, 'analysis', 'paper')]
import metrics as M, common as C, frc_fair
from simcode.util.proctitle import set_title

PS = f'{REPO}/out/papersim'
OUT = f'{REPO}/out/selection_global'
DENS = [f'{p}{d}{s}' for p, s, ns in (('D', '', 'm1809'), ('Dexp', '', 'm1809'), ('D', '_gauss', 'g13')) for d in ('0.5', '1', '2', '4')]
MOV = ['moving', 'moving_exp', 'moving_gauss']
NETSET = lambda r: 'g13' if r.endswith('_gauss') else 'm1809'
GRID = np.round(np.arange(5.0, 16.01, 0.5), 2)
GRID_MOV = [10.0, 12.0, 13.0, 14.0, 16.0]


def rows(z, m):
    return dict(x=z['x'][m].astype(np.float64), y=z['y'][m].astype(np.float64), f=z['frame'][m].astype(np.int64),
                sx=z['sx'][m].astype(np.float64), sy=z['sy'][m].astype(np.float64), photons=z['photons'][m].astype(np.float64))


def rules(z, grid, mean=True):
    u = z['strict_u'].astype(np.float64); base = np.isfinite(u)
    smax = np.maximum(z['sx'], z['sy']).astype(np.float64) * C.PX_NM; smean = 0.5 * (z['sx'] + z['sy']).astype(np.float64) * C.PX_NM
    out = {'u045': base & (u <= 0.45)}
    for t in grid:
        out[f'max<={t:g}'] = base & (smax <= t)
        if mean: out[f'mean<={t:g}'] = base & (smean <= t)
    return out


def dens_eval(run):
    md = f'{PS}/{run}'; ns = NETSET(run)
    g = json.load(open(f'{md}/gen.json')); W = g['W']; nf = g['nframes']
    gt = dict(np.load(f'{md}/gt.npz')); pts = gt['pts']
    GW = M.gt_windows(gt, nf, np.arange(0, nf - 5, 6))
    z = dict(np.load(f'{md}/full/v2_{ns}/locs_p2_sim.npz'))
    R = {}
    for k, m in rules(z, GRID).items():
        T = rows(z, m)
        full, disj, filt = M.frc_set(T, W, nf)
        r = dict(n_all=int(full.sum()), n_frc=int(filt.sum()))
        r['frc_nm'] = float(frc_fair.frc_halves(T['x'][filt], T['y'][filt], T['f'][filt])[0]) if filt.sum() > 50 else np.nan
        r['jaccard'] = float(M.compute_jaccard_points(pts[:, :2], np.c_[T['x'][full], T['y'][full]], 10 / C.PX_NM)) if full.sum() else np.nan
        gm = M.gt_match(T, filt, GW, W) if filt.sum() else {}
        r.update(rms_nm=float(np.mean(gm['rms_nm'])) if gm.get('rms_nm') else np.nan, err_gt30=gm.get('frac_err_gt30', np.nan),
                 recall=gm.get('recall_gt_ph150', np.nan),
                 med_sigma_nm=float(np.median(0.5 * (T['sx'][filt] + T['sy'][filt])) * C.PX_NM) if filt.any() else np.nan)
        R[k] = r
    return R


def mov_eval(run, bs=1000, thr_nm=(10, 20, 40)):
    md = f'{PS}/{run}'; ns = NETSET(run)
    g = json.load(open(f'{md}/gen.json')); nf = g['nframes']; tstep = int(g['traj_step'])
    traj = np.load(f'{md}/traj.npy', mmap_mode='r'); nb = nf // bs
    z = dict(np.load(f'{md}/full/v2_{ns}/locs_p2_sim.npz'))
    R = {}
    for k, m in rules(z, GRID_MOV, mean=False).items():
        T = rows(z, m)
        J = np.full((len(thr_nm), nb), np.nan); fb = np.full(nb, np.nan)
        for i in range(nb):
            sel = (T['f'] > i * bs) & (T['f'] < (i + 1) * bs)
            if sel.sum():
                P = np.asarray(traj[i * bs // tstep:(i + 1) * bs // tstep]).reshape(-1, 2)
                for j, t in enumerate(thr_nm): J[j, i] = M.compute_jaccard_points(np.c_[T['x'][sel], T['y'][sel]], P, t / C.PX_NM)
            s2 = (T['f'] >= i * bs) & (T['f'] < (i + 1) * bs) & (T['f'] % 6 == 0) & (np.maximum(T['sx'], T['sy']) < 0.15)
            if s2.sum() > 200: fb[i] = frc_fair.frc_halves(T['x'][s2], T['y'][s2], T['f'][s2], blk=60)[0]
        R[k] = dict(n_per_block=float(len(T['x']) / nb), jaccard_mean=np.nanmean(J, 1).tolist(),
                    frc_block_median_nm=float(np.nanmedian(fb)) if np.isfinite(fb).any() else np.nan)
        print(run, k, json.dumps(R[k]), flush=True)
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default=''); ap.add_argument('--moving', action='store_true')
    a = ap.parse_args()
    set_title('sel-sweepsim')
    fn = f'{OUT}/sweep_sim{"_moving" if a.moving else ""}{"_" + a.runs.replace(",", "_") if a.runs else ""}.json'
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for run in (a.runs.split(',') if a.runs else (MOV if a.moving else DENS)):
        if run in res: continue
        t0 = time.time()
        res[run] = mov_eval(run) if a.moving else dens_eval(run)
        json.dump(res, open(fn, 'w'), indent=1)
        if not a.moving:
            u = res[run]['u045']; print(run, f'{time.time() - t0:.0f} s', 'u045', json.dumps(u), flush=True)
            for t in (8, 10, 12, 14):
                print('   max<=%g' % t, json.dumps(res[run][f'max<={t:g}']), flush=True)


if __name__ == '__main__':
    main()
