"""Kinetic model and brightness noise (2026-11-17, final method): on held-out simulated movies with per-frame truth, runs of the final
pipeline (python -m simcode.run: brightness noise c, c_a estimated with the pattern; kinetic models averaged) in
out/sigadopt/runs/<model>/<sim>.
Per GT window matched to the run's pass-2 rows (p >= 0.63, inside the crop, 1 px on the averaged solve; the window's true event),
the pass-2 front is re-run (return_head) and the pass-2 solve repeated with
  c       'est' = the run's estimated (c, c_a) | 'fix' = the removed fixed (0.1, 0.1)
  kinetic 'avg' = posterior average of both models (BIC penalty 2 ln 6; the pipeline) | 'thr' = the removed rule (kinetic only if
          it lowers the cost by > 9.21) | 'none' = non-kinetic only | 'kin' = kinetic only
and compared with the pass-2 network (net_p2; locs_p2_net). Per photon bin (detected photons of the window), always-on GT rows
(ON through the window) and blinking rows separately: robust sd nm (1.4826 MAD, x and y pooled, errors minus the per-movie
median), z = rms(err / reported sigma) (|z| < 4), as analysis/sigma_joint/net_vs_solve.py.
True brightness variation of the sim: gt_brightness.py (GT photons vs B il_t(x) with the true illumination incl. the pattern
field, B per window; and the simulator's knobs frame_sd, step_sd, ang_sd).
usage: python kin_c_compare.py <out json> <model>:<sim> [...]"""
import json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, 'analysis', 'baselines'))
from evalsim import gt_windows, load_gt, SJ
from kap_ideal import movie_of
from net_vs_solve import matched, crlbs, rob, zcore, PSFSPEC, PH, PX
from scipy.spatial import cKDTree

RUNS = os.path.expanduser('~/simcode2/out/sigadopt/runs')
dev = 'cuda'


@torch.no_grad()
def solves(o, PWall, bw, cal_model, c, ca):
    """the pass-2 solve of the head outputs o with brightness noise (c, ca): kinetic modes avg / thr / none / kin -> {mode: (r [n,2], s [n,2], kinw [n])}"""
    from simcode.solve import solve as SV, solve_fast as MF
    cal = SV.solve_cal({k: v for k, v in cal_model.items() if k not in SV.BN_KEYS}, c, ca)
    out = {k: ([], [], []) for k in ('avg', 'thr', 'none', 'kin')}
    for c0 in range(0, len(o['I']), 8192):
        sl = slice(c0, c0 + 8192)
        oc = {k: v[sl] for k, v in o.items()}
        I, Sig, r0, sp, ph, Cj = SV.solve_inputs(oc, cal, mm=MF._mm)
        pt = PWall[bw[sl]]
        dr = SV.start_offsets(pt, r0, [r0], 1)
        kw = dict(iters=30, box=cal['box'], cmult=c, cang=ca, compile='off')
        o0 = MF.pattern_solve_all_fast(I, Sig, r0, sp, pt, dr, Cj, kinetics=False, **kw)
        o1 = MF.pattern_solve_all_fast(I, Sig, r0, sp, pt, dr, Cj, kinetics=True, **kw)
        T = cal['mix_T']
        F = {m: SV.finish(SV.model_average(SV.kin_models(m, o0, o1), temp=T), I, Sig, r0, sp) for m in ('average', 'none', 'kinetic')}
        use = (o0['nll'].min(1).values - o1['nll'].min(1).values) > 9.21          # the removed selection rule
        for m, f in (('avg', F['average']), ('none', F['none']), ('kin', F['kinetic'])):
            out[m][0].append(f['r']); out[m][1].append(f['s_cal']); out[m][2].append(f['kin'])
        w = lambda a, b: torch.where(use[:, None], a, b)
        out['thr'][0].append(w(F['kinetic']['r'], F['none']['r'])); out['thr'][1].append(w(F['kinetic']['s_cal'], F['none']['s_cal']))
        out['thr'][2].append(use.float())
    return {k: tuple(torch.cat(v).cpu().numpy() for v in vv) for k, vv in out.items()}


@torch.no_grad()
def collect(model, sim, psf):
    from simcode import config, table as SP
    from simcode.io.movie import open_movie
    from simcode.processing.common import region, load_calib
    from simcode.processing.inference_fast import Pass2FrontFast
    M = config.model_set(model)
    c1 = json.load(open(config.resolve(M['cal_sim']))); c2 = json.load(open(config.resolve(M['cal_sim_pass2'])))
    run = f'{RUNS}/{model}/{sim}'; tdir = f'{SJ}/truth/{sim}'
    mov, calp = movie_of(sim, model)
    movie = open_movie(mov); cal = load_calib(calp)
    crop = tuple(int(v) for v in SP.run_crop(run)); X0, Y0, S = region(movie, crop, 5)
    Pt = torch.load(f'{run}/pattern_est.pt', weights_only=False)
    cE, caE = float(Pt['cmult']), float(Pt['cmult_ang'])
    p2 = Pass2FrontFast(calp, emitter=M['emitter_pass2'], detector=M['detector_pass2'], compile='off')
    P1 = dict(np.load(f'{run}/pass1_0.npz'))
    from simcode.solve.solve import pattern_cal
    R, H = p2.run_chunk(movie, cal, 0, movie.n, Pt, pattern_cal(c2, Pt), P1, c1, crop=crop, margin=5, return_head=True)
    o, PWall, bw = H['o'], H['PWall'], H['bw']
    del p2; torch.cuda.empty_cache()
    V = {f'est_{k}': v for k, v in solves(o, PWall, bw, c2, cE, caE).items()}
    V.update({f'fix_{k}': v for k, v in solves(o, PWall, bw, c2, 0.1, 0.1).items()})
    dx = float(np.abs(V['est_avg'][0][:, 0] + X0 - R['x_hy0']).max())
    print(sim, 'pass-2 rows', len(R['x_hy0']), f'c {cE:.4f} c_a {caE:.4f}', 'max |avg solve - run| px', dx, flush=True)
    G = np.load(f'{tdir}/sim_gt.npz'); TF = np.load(f'{tdir}/truth_frames.npz')
    gxa, gya, t0, t1, nf = load_gt(f'{tdir}/sim_gt.npz')
    GW = gt_windows(gxa, gya, t0, t1, nf)
    xs, ys = V['est_avg'][0][:, 0] + X0, V['est_avg'][0][:, 1] + Y0
    b2 = (R['p'] >= 0.63) & SP.in_crop(xs, ys, crop)
    g2 = np.full(len(b2), -1); g2[b2] = matched(R['wstart'][b2], xs[b2], ys[b2], GW)
    r2 = np.nonzero(g2 >= 0)[0]; com = g2[r2]
    w = GW[com, 0].astype(np.int64); ev = GW[com, 1].astype(np.int64); tx, ty = GW[com, 2], GW[com, 3]
    tz = G['z'][ev]
    key = TF['ev'] * nf + TF['frame']; oo = np.argsort(key); key, phv = key[oo], TF['ph'][oo]
    q = ev[:, None] * nf + w[:, None] + np.arange(6)[None]
    j = np.clip(np.searchsorted(key, q), 0, len(key) - 1)
    Ig = np.where(key[j] == q, phv[j], 0.0)
    fr = w[:, None] + np.arange(6)[None]
    pt = G['patt'][fr].astype(np.float64)
    arg = pt[..., 0] * tx[:, None] + pt[..., 1] * ty[:, None] - pt[..., 2]
    il = pt[..., 4] * (1 + pt[..., 3] * np.cos(arg)); dd = -pt[..., 4] * pt[..., 3] * np.sin(arg)
    bg = o['bg'][torch.as_tensor(r2, device=dev)].float().cpu().numpy().clip(min=0.05)
    _, csf, coll = crlbs(psf, tx, ty, tz, Ig, bg, il, dd * pt[..., 0], dd * pt[..., 1])
    onthru = (t0[ev] <= w) & (t1[ev] >= w + 6)
    po2 = np.array(c2['prior_off']); sel = torch.as_tensor(r2, device=dev)
    E = dict(net_p2=(np.c_[o['x'][sel].cpu().numpy() + X0 - po2[0], o['y'][sel].cpu().numpy() + Y0 - po2[1]], o['lsxy'][sel].exp().cpu().numpy()))
    for k, (r, s, kw_) in V.items():
        E[k] = (r[r2] + [X0, Y0], s[r2])
    out = dict(ph=Ig.sum(1) * coll, onthru=onthru, crlb_sf=csf, kinw=V['est_avg'][2][r2], kinthr=V['est_thr'][2][r2])
    for k, (r, s) in E.items():
        err = r - np.c_[tx, ty]; err -= np.median(err, 0)
        out['e_' + k] = err; out['s_' + k] = s
    import gt_brightness as GB
    info = dict(sim=sim, model=model, c_est=cE, c_a_est=caE, n_matched=int(len(r2)), n_rows=int(len(R['x_hy0'])), max_dx_avg_vs_run_px=dx)
    info.update({k: v for k, v in GB.one(sim).items() if k != 'sim'})
    return out, info


KEYS = ('net_p2', 'fix_thr', 'est_thr', 'est_avg', 'est_none', 'est_kin', 'fix_avg')


def main():
    outj, specs = sys.argv[1], sys.argv[2:]
    import zpsf
    res = dict(sims={}, tables={})
    L = []; pr = lambda t: (print(t, flush=True), L.append(t))
    by_model = {}
    for sp_ in specs:
        model, sim = sp_.split(':')
        psf = zpsf.load(PSFSPEC[model], zmax=0.4)
        D, info = collect(model, sim, psf)
        res['sims'][sim] = info
        by_model.setdefault(model, []).append(D)
        torch.cuda.empty_cache()
    for model, parts in by_model.items():
        D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        sims = [s for s, i in res['sims'].items() if i['model'] == model]
        pr(f'\n# {model}: held-out sims {sims}; matched GT windows {len(D["ph"])}')
        for s in sims:
            i = res['sims'][s]
            pr(f'  {s}: estimated c {i["c_est"]:.3f}, c_a {i["c_a_est"]:.3f} | GT (gt_brightness.py) c {i["c_gt"]:.3f}, c_a {i["c_a_gt"]:.3f}'
               + (f' | simulator knobs sqrt(frame_sd^2 + step_sd^2) {i["c_knobs"]:.3f}, ang_sd/sqrt2 {i.get("c_a_knobs", float("nan")):.3f}' if 'c_knobs' in i else ''))
        rows = []
        for sub, m0 in (('always_on', D['onthru']), ('blinking', ~D['onthru'])):
            pr(f'\n## {sub} ({100 * m0.mean():.0f} %): robust sd nm (z)   [fix = fixed c 0.1 / 0.1, est = estimated c; thr = 9.21 rule, avg = posterior average]')
            pr('   photons       n  ' + '  '.join(f'{k:>14s}' for k in KEYS) + '  kin weight (avg)  kin share (thr)')
            for lo, hi in zip(PH[:-1], PH[1:]):
                m = m0 & (D['ph'] >= lo) & (D['ph'] < hi)
                if m.sum() < 100: continue
                r = dict(subset=sub, lo=lo, hi=hi, n=int(m.sum()), ph_med=float(np.median(D['ph'][m])), crlb_sf_nm=float(np.median(D['crlb_sf'][m])) * PX,
                         kin_weight_mean=float(D['kinw'][m].mean()), kin_thr_share=float(D['kinthr'][m].mean()))
                for k in KEYS:
                    e = D['e_' + k][m]; r[k] = dict(sd_nm=rob(e.ravel()) * PX, z=zcore(e / D['s_' + k][m]), sigma_med_nm=float(np.median(D['s_' + k][m])) * PX)
                rows.append(r)
                pr(f'   {lo:>5g}-{hi:<6g}{r["n"]:7d}  ' + '  '.join(f'{r[k]["sd_nm"]:7.2f} ({r[k]["z"]:.2f})' for k in KEYS)
                   + f'  {r["kin_weight_mean"]:.3f}  {r["kin_thr_share"]:.3f}')
        res['tables'][model] = rows
    json.dump(res, open(outj, 'w'), indent=1, default=float)
    open(outj.replace('.json', '.txt'), 'w').write('\n'.join(L) + '\n')
    print('wrote', outj)


if __name__ == '__main__':
    main()
