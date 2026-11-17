"""joint prior / intensity covariance (2026-11-17): where does the ~1.2x overconfidence of the solve's sigma on full simulated movies
come from? Pass 2 of a run on a simulated movie regenerated with its per-frame truth (sim_truth.py): the pass-2 front is re-run
on the movie (same networks, estimated pattern and pass-1 rows as the run; the rows are checked against the run's chunk file) and
the rows with p >= 0.63 inside the crop are matched to the true events (1 px, per window).
(1) network level: the whitened joint residual z_J = L_J^-1 [true - prior (x, y; prior_off removed); I_true - I] with the network's
    joint covariance (L_J = [[diag(sx, sy), 0], [C, L]]), I_true = the emitted photons per frame (truth_frames.npz): eigenvalues of its
    sample covariance (rows with all |z| < 6), rms of the position and intensity components
(2) solve level, the same rows: the estimated pattern (as the pipeline) vs the TRUE pattern of each row's frames (GT pattern + the
    pattern-phase / depth field at the true position; 'truemult': + the true per-frame brightness factors of the emitter), each
    with c as calibrated (0.1 / 0.1), c = 0 and c = 0.2 / 0.1; z = (x, y - true) / sigma (rms over
    |z| < 4, per-variant median error removed)
both per network-photon bin and split into isolated / crowded (another true emitter ON in the window within 3 px) and
on-through-window / blinking (ON in all 6 frames or not).
usage: python movie_diag.py <run dir> <truth dir (sim_truth.py output)> <model folder> <calib json> <out json>"""
import json, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalsim import gt_windows, load_gt

PX = 108.33
PH = [0, 300, 500, 800, 1200, 2000, 3000, 1e9]
dev = 'cuda'


def core(z):
    v = np.asarray(z).ravel(); c = v[np.abs(v) < 4]
    return float(np.sqrt(np.mean(c ** 2)) / 0.99893) if len(c) > 20 else float('nan')


def main():
    run, tdir, mdir, calib, outj = sys.argv[1:6]
    from simcode import config, table as SP
    from simcode.io.movie import open_movie
    from simcode.processing.common import region, load_calib
    from simcode.processing.inference_fast import Pass2FrontFast
    from simcode.solve import solve_fast as MF
    from simcode.evaluation import metrics as E
    M = config.model_set(mdir)
    movie = open_movie(f'{tdir}/sim_movie.tif'); cal = load_calib(calib)
    crop = tuple(int(v) for v in SP.run_crop(run)); margin = 5
    X0, Y0, S = region(movie, crop, margin)
    c1 = json.load(open(config.resolve(M['cal_sim']))); c2 = json.load(open(config.resolve(M['cal_sim_pass2'])))
    P1 = dict(np.load(f'{run}/pass1_0.npz')); Pt = torch.load(f'{run}/pattern_est.pt', weights_only=False)
    p2 = Pass2FrontFast(calib, emitter=M['emitter_pass2'], detector=M['detector_pass2'], compile='off')
    R, Hh = p2.run_chunk(movie, cal, 0, movie.n, Pt, c2, P1, c1, crop=crop, margin=margin, return_head=True)
    ref = np.load(f'{run}/pass2_est2_0.npz')
    same = len(ref['x_hy0']) == len(R['x_hy0']) and float(np.abs(ref['x_hy0'] - R['x_hy0']).max()) < 1e-3
    print('rows', len(R['x_hy0']), 'same as the run:', same, flush=True)
    o, PWall, bw = Hh['o'], Hh['PWall'], Hh['bw']
    # truth
    G = np.load(f'{tdir}/sim_gt.npz'); TF = np.load(f'{tdir}/truth_frames.npz')
    x, y, t0, t1, nf = load_gt(f'{tdir}/sim_gt.npz')
    GW = gt_windows(x, y, t0, t1, nf)
    ii = np.nonzero((R['p'] >= 0.63) & SP.in_crop(R['x_hy0'], R['y_hy0'], crop))[0]
    gi = E.match_fast(GW, R['wstart'][ii].astype(np.int64), R['x_hy0'][ii].astype(np.float64), R['y_hy0'][ii].astype(np.float64), r=1.0)
    k = gi >= 0; rows = ii[k]; gw = GW[gi[k]]
    w = gw[:, 0].astype(np.int64); ev = gw[:, 1].astype(np.int64)
    key = TF['ev'] * nf + TF['frame']; o_ = np.argsort(key); key, phv = key[o_], TF['ph'][o_]
    q = ev[:, None] * nf + w[:, None] + np.arange(6)[None]
    j = np.clip(np.searchsorted(key, q), 0, len(key) - 1)
    Ig = np.where(key[j] == q, phv[j], 0.0).astype(np.float32)
    tr = np.c_[x[ev] - X0, y[ev] - Y0]
    # crowding: another event ON in the same window within 3 px; on-through: ON in all 6 frames
    T_ = cKDTree(np.c_[GW[:, 2], GW[:, 3], GW[:, 0] * 1e4])
    d, nb = T_.query(np.c_[x[ev], y[ev], w * 1e4], k=2, distance_upper_bound=3.0)
    crowded = np.isfinite(d[:, 1])
    onthru = (t0[ev] <= w) & (t1[ev] >= w + 6)
    tt = lambda a: torch.as_tensor(np.asarray(a), device=dev, dtype=torch.float32)
    sel = torch.as_tensor(rows, device=dev)
    O = {kk: v[sel] for kk, v in o.items()}
    ph_net = O['I'].sum(1).cpu().numpy()
    # (1) network level
    po = tt(c2['prior_off'])
    sp = O['lsxy'].exp(); prior = torch.stack([O['x'], O['y']], 1) - po[None]
    n = len(rows); T = 6
    LJ = torch.zeros(n, T + 2, T + 2, device=dev)
    LJ[:, 0, 0] = sp[:, 0]; LJ[:, 1, 1] = sp[:, 1]; LJ[:, 2:, :2] = O['C']; LJ[:, 2:, 2:] = O['L']
    e = torch.cat([tt(tr) - prior, tt(Ig) - O['I']], 1)[..., None]
    zJ = torch.linalg.solve_triangular(LJ, e, upper=False)[..., 0].cpu().numpy()
    # (2) solve variants
    patt = G['patt'].astype(np.float32)                                       # [nf,5] global coordinates
    fr = w[:, None] + np.arange(6)[None]
    pw = patt[fr].copy()
    pw[..., 2] = pw[..., 2] - pw[..., 0] * X0 - pw[..., 1] * Y0                # region coordinates
    pwf = pw.copy()
    if 'dph' in TF.files:
        ang = (fr % 6) // 3
        pwf[..., 2] = pwf[..., 2] + TF['dph'][ev[:, None], ang]
        pwf[..., 3] = np.minimum(pwf[..., 3] * TF['dmf'][ev[:, None], ang], 1.0)
    # the per-frame brightness factors of the truth (per-angle / per-step emitter factors and the per-frame variation of the
    # simulator): mult_t = emitted photons / (brightness x true illumination x on-fraction); 'truemult' = the true pattern with
    # rel_t x mult_t, i.e. the solve's model mu_t = B a_t il_t made exact
    il_true = pwf[..., 4] * (1 + pwf[..., 3] * np.cos(pwf[..., 0] * tr[:, :1] + pwf[..., 1] * tr[:, 1:] - pwf[..., 2]))
    onf = np.clip(np.minimum(t1[ev, None], fr + 1) - np.maximum(t0[ev, None], fr), 0, 1)
    den = G['bright'][ev, None] * il_true * onf
    mult = np.where(den > 1e-3, Ig / np.maximum(den, 1e-3), 1.0)
    pwm = pwf.copy(); pwm[..., 4] = pwm[..., 4] * mult
    variants = {}
    bi = torch.arange(n, device=dev)
    for nm, (PW, b) in dict(est=(PWall, bw[sel]), true=(tt(pwf), bi), true_nofield=(tt(pw), bi), truemult=(tt(pwm), bi)).items():
        for cn, cc in (('c', c2), ('c0', dict(c2, cmult=0.0, cmult_ang=0.0)), ('c02', dict(c2, cmult=0.2, cmult_ang=0.1))):
            if nm in ('true_nofield',) and cn == 'c02': continue
            acc = {'r': [], 's': []}
            for c0 in range(0, n, 16384):
                s_ = slice(c0, c0 + 16384)
                so = MF.solve_emitters_fast({kk: v[s_] for kk, v in O.items()}, PW, b[s_], cc, compile='off')
                acc['r'].append(so['r']); acc['s'].append(so['s_cal'])
            r = torch.cat(acc['r']).cpu().numpy(); s = torch.cat(acc['s']).cpu().numpy()
            er = r - tr; er -= np.median(er, 0)
            variants[f'{nm}_{cn}'] = dict(z=er / s, e=er, s=s)
    res = dict(run=run, truth=tdir, n=int(n), rows_same_as_run=bool(same), splits={})
    splits = dict(all=np.ones(n, bool), isolated=~crowded, crowded=crowded, onthru=onthru, blinking=~onthru,
                  iso_onthru=~crowded & onthru)
    for sn, m in splits.items():
        out = []
        for lo, hi in zip(PH[:-1], PH[1:]):
            b = m & (ph_net >= lo) & (ph_net < hi)
            if b.sum() < 100: continue
            zz = zJ[b]; ok = (np.abs(zz) < 6).all(1)
            ev_ = np.linalg.eigvalsh(np.cov(zz[ok].T))
            r = dict(lo=lo, hi=hi, n=int(b.sum()), net_eig_min=float(ev_.min()), net_eig_max=float(ev_.max()),
                     net_zp=float(np.sqrt(np.mean(zz[ok][:, :2] ** 2))), net_zI=float(np.sqrt(np.mean(zz[ok][:, 2:] ** 2))))
            for vn, V in variants.items():
                r[f'z_{vn}'] = core(V['z'][b]); r[f'sd_{vn}_nm'] = float(1.4826 * np.median(np.abs(V['e'][b]))) * PX
            out.append(r)
        res['splits'][sn] = dict(frac=float(m.mean()), bins=out)
        print(f'== {sn} ({m.mean() * 100:.0f} % of {n} rows)', flush=True)
        for r in out:
            print(f"  {r['lo']:>5g}-{r['hi']:<6g} n {r['n']:6d} | net eig {r['net_eig_min']:.2f}..{r['net_eig_max']:.2f} zp {r['net_zp']:.2f} zI {r['net_zI']:.2f} | "
                  + ' '.join(f"{vn} {r['z_' + vn]:.2f}" for vn in variants) + f" | sd est_c / true_c / truemult_c nm {r['sd_est_c_nm']:.2f} / {r['sd_true_c_nm']:.2f} / {r['sd_truemult_c_nm']:.2f}", flush=True)
    json.dump(res, open(outj, 'w'), indent=1)


if __name__ == '__main__':
    main()
