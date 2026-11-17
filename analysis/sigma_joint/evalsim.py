"""joint prior / intensity covariance (2026-11-17): acceptance on full simulated movies with ground truth (pipeline runs of
out/sigma_layer/sims and the paper simulations; the networks were fine-tuned on the trainer's own simulated batches, so every movie
here is held out). For each run label (e.g. id = production until now, k1 = old networks without kap, joint = the fine-tuned networks
with the joint solve) and pass: the rows with p >= 0.63 inside the crop, matched one-to-one (1 px, per window) to the true events
ON in the row's window, minus the per-movie median error (gauge offset). Reported per network-photon bin and per sigma-rank bin u
(per-10-px-tile rank of the mean sigma, simcode.table.tile_rank):
  z_core = rms(e / sigma) on |z| < 4 (truncation-corrected), z_rob = 1.4826 MAD, gross = |z| >= 4 %   (sigma = the run's sx, sy;
           for the id runs the production sigma until 2026-11-17 (prod_sigma, the removed factors; their x, y = production) and
           'id_g' = that x the model's old g(u) = the production tables; for k1 the raw solve sd without kap)
  precision: robust error sd (1.4826 MAD, nm), rms of |e| < 50 nm, gross > 50 nm %; on the rows matched in all compared runs
           (same movie: the matched sets differ slightly between runs, the common GT events are used)
usage: python evalsim.py <out json> <model> <labels comma, first = reference> <sim> [<sim> ...]"""
import json, os, subprocess, sys
import numpy as np

PX = 108.33
PH = [0, 300, 500, 800, 1200, 2000, 3000, 5000, 1e9]
UE = [0.0, 0.05, 0.20, 0.50, 0.80, 1.0]
TRIM = 4.0
SL = os.path.expanduser('~/simcode2/out/sigma_layer')
SJ = os.path.expanduser('~/simcode2/out/sigma_joint')
RUNDIR = dict(id=f'{SL}/runs/{{m}}_id', k1=f'{SL}/runs/{{m}}_k1')      # other labels: out/sigma_joint/runs/<m>_<label>
rundir = lambda lab: RUNDIR.get(lab, f'{SJ}/runs/{{m}}_{lab}')
rob = lambda v: float(1.4826 * np.median(np.abs(v))) if len(v) else float('nan')


def gt_of(sim):
    for p in (f'{SL}/sims/{sim}/sim_gt.npz', os.path.expanduser(f'~/simcode2/out/papersim/{sim}/gt.npz')):
        if os.path.exists(p): return p
    return dict(ah1=os.path.expanduser('~/simcode2/out/psf_v2/eval/sims/m05r3v2/sim_gt.npz'),
                bh1=os.path.expanduser('~/simcode2/out/psf_v2/eval/sims/m25rc3v2/sim_gt.npz'))[sim]


def load_gt(path):
    G = np.load(path, allow_pickle=True)
    if 'ev_x' in G.files:
        return G['ev_x'].astype(float), G['ev_y'].astype(float), G['ev_t0'].astype(float), G['ev_t1'].astype(float), int(G['nframes'])
    return G['x'].astype(float), G['y'].astype(float), G['t0'].astype(float), G['t1'].astype(float), int(G['nf'])


def gt_windows(x, y, t0, t1, nf, grid=3):
    """(window start w % grid == 0, event) pairs with the event ON in [w, w + 6) -> [m, 4] (w, event, x, y)"""
    wlo = np.maximum(np.ceil((t0 - 6) / grid + 1e-9), 0).astype(np.int64) * grid
    whi = np.minimum(np.floor(t1 / grid - 1e-9), (nf - 6) // grid).astype(np.int64) * grid
    n = np.maximum((whi - wlo) // grid + 1, 0)
    ev = np.repeat(np.arange(len(x)), n)
    w = wlo[ev] + grid * (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    ok = (np.minimum(t1[ev], w + 6) - np.maximum(t0[ev], w)) > 0
    return np.c_[w, ev, x[ev], y[ev]][ok]


def old_files(model, ps):
    """the production calibration and g(u) of a model and pass until 2026-11-17 (git)"""
    br = 'paper' if model in ('m1809', 'm25') else 'master'
    g = lambda p: json.loads(subprocess.check_output(['git', '-C', os.path.expanduser('~/simcode2'), 'show', f'{br}:models/{model}/{p}']))
    M = g('model.json')[model]
    return g(M['cal_sim' if ps == 'est' else 'cal_sim_pass2']), g(M['sig_u'].split(',')[0 if ps == 'est' else 1])


def prod_sigma(cal, R, ii):
    """the production sigma until 2026-11-17 from the raw solve quantities stored by the id runs: sqrt(f s_pm^2 + spread_pm^2)
    x sig_ph(photons), f = sig_a max(q/dof, 1)^sig_g"""
    two = lambda k: np.c_[R[k + '_x'], R[k + '_y']][ii].astype(np.float64)
    ph = R['phot'][ii]; tab = cal['sig_ph']
    sph = np.interp(np.log(np.maximum(ph, 1.0)), np.log(np.array(tab['ph'])), np.array(tab['f']))
    f = cal['sig_a'] * np.maximum(R['q_dof'][ii], 1.0) ** cal['sig_g']
    return np.sqrt(f[:, None] * two('s_pm') ** 2 + two('spread_pm') ** 2) * sph[:, None]


def load(run, tag, GW, G=None, cal=None):
    """matched rows of a run: e [n,2] px, s [n,2] (sigma: the run's sx, sy, or the production sigma of cal (prod_sigma); x g(u) if
    G), photons, u (sigma rank), gt index"""
    from simcode import table as SP
    from simcode.evaluation import metrics as E
    R = SP.load_rows(run, tag); crop = SP.run_crop(run)
    ii = np.nonzero((R['p'] >= 0.63) & SP.in_crop(R['x_hy0'], R['y_hy0'], crop))[0]
    s = prod_sigma(cal, R, ii) if cal is not None else np.c_[R['sx_cal'], R['sy_cal']][ii].astype(np.float64)
    u = SP.tile_rank(R['x_hy0'][ii], R['y_hy0'][ii], s.mean(1), crop)
    if G is not None:
        s = s * np.interp(u, G['u_mid'], G['g'])[:, None]
        u = SP.tile_rank(R['x_hy0'][ii], R['y_hy0'][ii], s.mean(1), crop)
    gi = E.match_fast(GW, R['wstart'][ii].astype(np.int64), R['x_hy0'][ii].astype(np.float64), R['y_hy0'][ii].astype(np.float64), r=1.0)
    m = gi >= 0
    e = np.c_[R['x_hy0'][ii][m] - GW[gi[m], 2], R['y_hy0'][ii][m] - GW[gi[m], 3]]
    e -= np.median(e, 0)
    return dict(e=e, s=s[m], ph=R['phot'][ii][m], u=u[m], gi=gi[m], n_base=len(ii), cut16=int((s.max(1) * PX <= 16.25).sum()))


def zst(z):
    v = z.ravel(); c = v[np.abs(v) < TRIM]
    return dict(n=int(len(z)), z_core=float(np.sqrt(np.mean(c ** 2)) / 0.99893) if len(c) else float('nan'), z_rob=rob(v),
                gross_pct=float(100 * np.mean(np.abs(v) >= TRIM)) if len(v) else float('nan'))


def prec(e):
    r = np.hypot(*e.T) * PX; v = e.ravel() * PX; c = e[r < 50] * PX
    return dict(n=int(len(e)), sd_rob=rob(v), rms50=float(np.sqrt(np.mean(c ** 2))) if len(c) else float('nan'), gross50_pct=float(100 * np.mean(r >= 50)))


def main():
    outj, model, labels, sims = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4:]
    res = dict(model=model, labels=labels, sims=sims, passes={})
    for ps in ('est', 'est2'):
        D = {lab: [] for lab in labels + (['id_g'] if 'id' in labels else [])}
        cal0, G = old_files(model, ps)
        common = {lab: [] for lab in labels}
        per_sim = {}
        for sim in sims:
            GW = gt_windows(*load_gt(gt_of(sim)))
            Ls = {lab: load(rundir(lab).format(m=model) + '/' + sim, ps, GW, cal=cal0 if lab == 'id' else None) for lab in labels}
            if 'id' in labels: Ls['id_g'] = load(rundir('id').format(m=model) + '/' + sim, ps, GW, G, cal=cal0)
            for lab, L in Ls.items(): D[lab].append(L)
            # precision on the GT windows matched in every run
            gs = None
            for lab in labels:
                gs = Ls[lab]['gi'] if gs is None else np.intersect1d(gs, Ls[lab]['gi'])
            srt = lambda L, v: v[np.isin(L['gi'], gs)][np.argsort(L['gi'][np.isin(L['gi'], gs)])]     # rows of the common GT windows, by window
            ph0 = srt(Ls[labels[0]], Ls[labels[0]]['ph'])                                              # photons of the reference run
            for lab in labels:
                common[lab].append(dict(e=srt(Ls[lab], Ls[lab]['e']), ph=ph0))
            per_sim[sim] = {lab: dict(photons=[dict(lo=lo, hi=hi, **zst(L['e'][(L['ph'] >= lo) & (L['ph'] < hi)] / L['s'][(L['ph'] >= lo) & (L['ph'] < hi)]))
                                               for lo, hi in zip(PH[:-1], PH[1:]) if ((L['ph'] >= lo) & (L['ph'] < hi)).sum() >= 200],
                                      n_base=L['n_base'], cut16=L['cut16']) for lab, L in Ls.items()}
            print(f'# {ps} {sim}: ' + ', '.join(f'{lab} matched {len(L["e"])} / base {L["n_base"]}, <=16.25 nm {L["cut16"]}' for lab, L in Ls.items()), flush=True)
        R = dict(calib={}, precision={}, per_sim=per_sim)
        for lab, Ls in D.items():
            e = np.concatenate([L['e'] for L in Ls]); s = np.concatenate([L['s'] for L in Ls])
            ph = np.concatenate([L['ph'] for L in Ls]); u = np.concatenate([L['u'] for L in Ls]); z = e / s
            R['calib'][lab] = dict(all=zst(z), cut16=int(sum(L['cut16'] for L in Ls)), n_base=int(sum(L['n_base'] for L in Ls)),
                                   photons=[dict(lo=lo, hi=hi, sigma_med_nm=float(np.median(s[(ph >= lo) & (ph < hi)]) * PX), **zst(z[(ph >= lo) & (ph < hi)]))
                                            for lo, hi in zip(PH[:-1], PH[1:]) if ((ph >= lo) & (ph < hi)).sum() >= 200],
                                   u=[dict(lo=lo, hi=hi, **zst(z[(u >= lo) & (u < hi)])) for lo, hi in zip(UE[:-1], UE[1:])])
        for lab in labels:
            e = np.concatenate([c['e'] for c in common[lab]]); ph = np.concatenate([c['ph'] for c in common[lab]])
            R['precision'][lab] = [dict(lo=lo, hi=hi, **prec(e[(ph >= lo) & (ph < hi)])) for lo, hi in zip(PH[:-1], PH[1:]) if ((ph >= lo) & (ph < hi)).sum() >= 200]
        res['passes'][ps] = R
        for lab, C in R['calib'].items():
            print(f'{model} {ps} {lab:6s} z_core ' + ' '.join(f"{b['lo']:g}:{b['z_core']:.2f}" for b in C['photons']) + ' | u ' +
                  ' '.join(f"{b['z_core']:.2f}" for b in C['u']) + f" | all {C['all']['z_core']:.2f} (rob {C['all']['z_rob']:.2f}) | <=16.25 nm {C['cut16']} / {C['n_base']}", flush=True)
        for lab, P in R['precision'].items():
            print(f'{model} {ps} {lab:6s} sd_rob nm ' + ' '.join(f"{b['lo']:g}:{b['sd_rob']:.2f}" for b in P) + ' | rms<50 ' +
                  ' '.join(f"{b['rms50']:.2f}" for b in P) + ' | gross% ' + ' '.join(f"{b['gross50_pct']:.2f}" for b in P), flush=True)
    json.dump(res, open(outj, 'w'), indent=1)


if __name__ == '__main__':
    main()
