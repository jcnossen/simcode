"""Covariance ablation, part 2b: the solve variants on whole movies (pass 2 = the product), from the captured head outputs (capture.py).
Per variant, everything downstream of the head is redone the production way:
  solve    msolve_fast.sc_solve_fast (the pipeline's solve, batches of 16384) with the variant's covariance and its own calibration
           (out/ablation_cov/calibs/calib_s2_<v>.json from gate_abl.py; real movies: sigma factor x sig_ph_real_mult copied from the production
           real calibration, as analysis/psfmatch/fit_cal_set.py) -> runs/<set>/<v>/pass2_est2_<f0>.npz (pipeline format)
  sig_u    the selection-aware sigma rescale g(strict_u) re-fitted per variant on the sim movie m05r3 frames 0-2400 (analysis/psf3d_nets/sigcal.py fit,
           the production recipe of models/calib/sigu/sigu_p2_m1809.json) -> calibs/sigu_p2_<v>.json; held-out check on frames 2400-4800
  table    simcode2.strict.build_table (per-tile sigma rank strict_u, density-rule q, sig_u) -> locs_p2.npz
  metrics  sim (papersim D1 / D4): the 'v2q45' row of analysis/papersim/metrics.py (rows strict_u <= 0.45; fair FRC, Jaccard 10 nm, GT match)
           real: analysis/paper/frc_fair.py (rows strict_u <= 0.45 & sel, disjoint windows, sigma < 0.15 px, own DME v2 drift, 300-frame halves)
  repeat noise: the same metrics on 4 consecutive quarters of the movie (FRC of a quarter is coarser than of the whole movie; the paired
           difference to variant A per quarter gives mean +- se over the 4 quarters)
usage: CUDA_VISIBLE_DEVICES=2 python analysis/ablation_cov/movie_abl.py <set m05r3|D1|D4|real> [variants A,B,..] [stage solve,table]"""
import glob, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE, os.path.join(REPO, 'analysis', 'papersim'), os.path.join(REPO, 'analysis', 'paper'), os.path.join(REPO, 'analysis', 'psf3d_nets')):
    sys.path.insert(0, p)
import variants as V
from simcode2 import config
from simcode2.util import proctitle
proctitle.set_title('covabl')
OUT = os.path.join(REPO, 'out', 'ablation_cov'); PX = 108.33
SETS = dict(m05r3=dict(kind='fit', sim=os.path.join(REPO, 'out/psf3d_nets/sims/m05r3')),
            D1=dict(kind='sim', md=os.path.join(REPO, 'out/papersim/D1'), nframes=60000),
            D4=dict(kind='sim', md=os.path.join(REPO, 'out/papersim/D4'), nframes=60000),
            real=dict(kind='real', nframes=120000))
PROD_REAL = 'calib/calib_p2L_m1809_real.json'


def calib(name, real):
    cal = json.load(open(f'{OUT}/calibs/calib_s2_{name}.json'))
    if real:
        M = json.load(open(config.resolve(PROD_REAL)))['sig_ph_real_mult']
        mult = np.interp(np.log(cal['sig_ph']['ph']), np.log(M['ph']), M['mult'])
        cal['sig_ph_sim'] = cal['sig_ph']; cal['sig_ph'] = dict(ph=cal['sig_ph']['ph'], f=[float(a * b) for a, b in zip(cal['sig_ph']['f'], mult)])
    return cal


@torch.no_grad()
def solve(setn, name):
    from simcode2.hd import msolve_fast as MF
    v = V.VARS[name]; cal = calib(name, SETS[setn]['kind'] == 'real'); d = f'{OUT}/runs/{setn}/{name}'; os.makedirs(d, exist_ok=True)
    fs = sorted(glob.glob(f'{OUT}/head_{setn}/head_*.npz'), key=lambda f: int(f.rsplit('_', 1)[1][:-4]))
    t0 = time.time(); dmax = 0.0
    for f in fs:
        f0 = f.rsplit('_', 1)[1][:-4]; fn = f'{d}/pass2_est2_{f0}.npz'
        if os.path.exists(fn): continue
        H = np.load(f); tt = lambda a: torch.tensor(a, device='cuda', dtype=torch.float32)
        o = dict(I=tt(H['o_I']), L=tt(V.L_variant(H['o_L'], v['L'])), x=tt(H['o_x']), y=tt(H['o_y']), lsxy=tt(V.lsxy_variant(H['o_lsxy'], v)),
                 x_det=tt(H['o_x_det']), y_det=tt(H['o_y_det']))
        PW = tt(H['PW']); bw = torch.tensor(H['bw'], device='cuda').long(); X0, Y0 = float(H['X0']), float(H['Y0'])
        acc = {k: [] for k in ('r', 's_cal', 'q', 'kin')}
        for c0 in range(0, len(bw), 16384):
            sl = slice(c0, c0 + 16384)
            so = MF.sc_solve_fast({kk: vv[sl] for kk, vv in o.items()}, PW, bw[sl], cal, compile='auto')
            for k in acc: acc[k].append(so[k].float().cpu().numpy())
        so = {k: np.concatenate(a) for k, a in acc.items()}
        R = {k: H[k] for k in ('wstart', 'frame', 'p', 'phot', 'x_prior', 'y_prior', 'x_det', 'y_det', 'I_net')}
        R.update(x_hy0=so['r'][:, 0] + X0, y_hy0=so['r'][:, 1] + Y0, sx_cal=so['s_cal'][:, 0], sy_cal=so['s_cal'][:, 1], sig_hy0=so['s_cal'].mean(1),
                 q_dof=so['q'], kin_hy=so['kin'].astype(np.float32))
        if name == 'A':
            dmax = max(dmax, float(np.abs(R['x_hy0'] - H['x_hy0']).max()), float(np.abs(R['sx_cal'] - H['sx_cal']).max()))
        np.savez(fn, **R)
    if name == 'A' and dmax >= 0:
        print(f'{setn} A: max |x - production x|, |sx - production sx| over the solved chunks = {dmax:.2e} px', flush=True)
    print(f'{setn} {name}: solved {len(fs)} chunks in {time.time() - t0:.0f}s', flush=True)
    return d


def fit_sigu(name):
    """sigcal.py fit on m05r3 [0, 2400), test on [2400, 4800)"""
    import simcode2.strict as SP, sigcal
    d = f'{OUT}/runs/m05r3/{name}'
    T, info = SP.build_table(d, 'est2', PX); SP.write_table(T, info, f'{d}/locs_p2', ('npz',))
    sim = SETS['m05r3']['sim']
    D = sigcal.load(f'{d}:{sim}:est2:0:2400')
    mid, g, n = [], [], []
    for lo, hi in zip(sigcal.UE[:-1], sigcal.UE[1:]):
        m = (D['u'] >= lo) & (D['u'] < hi); mid.append(float(np.median(D['u'][m]))); g.append(sigcal.zc(D['e'][m], D['s'][m])); n.append(int(m.sum()))
    fit = dict(u_edges=sigcal.UE, u_mid=mid, g=g, n=n, note=f'covariance ablation variant {name}: sigcal fit on m05r3 [0, 2400)')
    json.dump(fit, open(f'{OUT}/calibs/sigu_p2_{name}.json', 'w'), indent=1)
    H = sigcal.load(f'{d}:{sim}:est2:2400:4800'); gs = sigcal.g_of(fit, H['u'])[:, None]; st = H['strict']; k45 = H['u'] <= 0.45
    r = dict(g=g, q=info['q'], heldout=dict(n_good=H['n_good'], n_match=H['n_match'], zc_strict_before=sigcal.zc(H['e'][st], H['s'][st]),
             zc_strict=sigcal.zc(H['e'][st], H['s'][st] * gs[st]), zc_all=sigcal.zc(H['e'], H['s'] * gs), zc_u45=sigcal.zc(H['e'][k45], (H['s'] * gs)[k45]),
             rms_u45_nm=float(np.sqrt(np.mean(H['e'][k45] ** 2)) * PX), rms_strict_nm=float(np.sqrt(np.mean(H['e'][st] ** 2)) * PX),
             rms_all_nm=float(np.sqrt(np.mean(H['e'] ** 2)) * PX), n_u45=int(k45.sum())))
    # calibration of sigma across the u ranking on the held-out half: zcal per u bin after the rescale
    r['heldout']['zc_by_u'] = [sigcal.zc(H['e'][(H['u'] >= lo) & (H['u'] < hi)], (H['s'] * gs)[(H['u'] >= lo) & (H['u'] < hi)]) for lo, hi in zip(sigcal.UE[:-1], sigcal.UE[1:])]
    return r


def frc2(x, y, f):
    """(protocol FRC = first frequency bin below 1/7, as frc_fair.frc_halves; the same crossing linearly interpolated between the two
    frequency bins). The protocol value is quantised (bin step 0.16 nm at 26 nm, 0.24 nm at 17 nm); the interpolated one is not."""
    import frc_fair
    h = (f // 300) % 2 == 0
    res, freq, c = frc_fair.frc_curve(x[h], y[h], x[~h], y[~h])
    b = np.nonzero(c[3:] < 1 / 7)[0]
    if not len(b): return float(res), float('nan')
    i = b[0] + 3; fs = freq[i - 1] + (c[i - 1] - 1 / 7) / max(c[i - 1] - c[i], 1e-12) * (freq[i] - freq[i - 1])
    return float(res), float(1 / fs)


def quarters(nf): return [(i * nf // 4, (i + 1) * nf // 4) for i in range(4)]


def sim_metrics(setn, name):
    import simcode2.strict as SP, metrics as M, frc_fair
    S = SETS[setn]; md = S['md']; nf = S['nframes']; d = f'{OUT}/runs/{setn}/{name}'
    g = json.load(open(f'{md}/gen.json')); W = int(g['W']); crop = (0.0, 0.0, float(W))
    SP.CROP = crop; SP.tile_of.__defaults__ = (crop, 10.0); SP.bin20.__defaults__ = (crop,)
    T_, info = SP.build_table(d, 'est2', g['px_nm'], sig_u=f'{OUT}/calibs/sigu_p2_{name}.json')
    SP.write_table(T_, info, f'{d}/locs_p2_sim', ('npz',))
    T = M.load_method('v2q45', f'{d}/locs_p2_sim.npz')
    gt = dict(np.load(f'{md}/gt.npz')); pts = gt['pts']
    if 'GW' not in S: S['GW'] = M.gt_windows(gt, nf, np.arange(0, nf - 5, 6))
    GW = S['GW']
    full, disj, filt = M.frc_set(T, W, nf)
    r = dict(n_all=int(len(T['x'])), n_disjoint=int(disj.sum()), n_frc=int(filt.sum()), q_strict=info['q'], n_good=info['n_good'])
    r['frc_nm'], r['frc_i_nm'] = frc2(T['x'][filt], T['y'][filt], T['f'][filt])
    pos = np.c_[T['x'], T['y']]
    r['jaccard_end'] = float(M.compute_jaccard_points(pts[:, :2], pos[full], 10 / M.C.PX_NM))
    r['jaccard_frcset_end'] = float(M.compute_jaccard_points(pts[:, :2], pos[filt], 10 / M.C.PX_NM))
    gm = M.gt_match(T, filt, GW, W); r['gt_match'] = gm
    r['rms_nm'] = float(np.sqrt(np.mean(np.array(gm['rms_nm']) ** 2))); r['rms_trim2_nm'] = float(np.sqrt(np.mean(np.array(gm['rms_trim2_nm']) ** 2)))
    Q = []
    for lo, hi in quarters(nf):
        k = (T['f'] >= lo) & (T['f'] < hi); fq = filt & k
        gq = M.gt_match(T, fq, GW, W)
        Q.append(dict(frc_nm=frc2(T['x'][fq], T['y'][fq], T['f'][fq])[0], frc_i_nm=frc2(T['x'][fq], T['y'][fq], T['f'][fq])[1], n_frc=int(fq.sum()),
                      jaccard=float(M.compute_jaccard_points(pts[:, :2], pos[full & k], 10 / M.C.PX_NM)),
                      rms_nm=float(np.sqrt(np.mean(np.array(gq['rms_nm']) ** 2))), err30=gq['frac_err_gt30'], zmed=float(np.mean(gq['median_abs_err_over_sigma']))))
    r['quarters'] = Q
    return r


def real_metrics(name):
    import simcode2.strict as SP, frc_fair
    nf = SETS['real']['nframes']; d = f'{OUT}/runs/real/{name}'
    T_, info = SP.build_table(d, 'est2', PX, sig_u=f'{OUT}/calibs/sigu_p2_{name}.json')
    u = np.nan_to_num(T_['strict_u'].astype(np.float64), nan=2.0); m = (u <= 0.45)      # as the production locs_p2_u045.npz FRC row (n_all 1404302 = all u <= 0.45 rows)
    np.savez(f'{d}/locs_p2_u045.npz', **{k: v[(u <= 0.45)] for k, v in T_.items()})
    T = dict(x=T_['x'][m].astype(np.float64), y=T_['y'][m].astype(np.float64), f=T_['frame'][m].astype(np.int64), sx=T_['sx'][m].astype(np.float64),
             sy=T_['sy'][m].astype(np.float64), photons=T_['photons'][m].astype(np.float64))
    R = frc_fair.evaluate(T, nf, drift_cache=f'{d}/drift_u045.npy')
    r = dict(frc_fair.summary(R), q_strict=info['q'], n_good=info['n_good'], n_u045=int((u <= 0.45).sum()), n_strict=info['n_strict'],
             median_sigma_nm=float(np.median(0.5 * (T['sx'] + T['sy'])[R['mask']]) * PX), median_photons=float(np.median(T['photons'][R['mask']])))
    r['drift_info'] = str(r.get('drift_info')); r['frc_i_nm'] = frc2(R['x_dc'][R['mask']], R['y_dc'][R['mask']], T['f'][R['mask']])[1]
    Q = []
    for lo, hi in quarters(nf):
        k = R['mask'] & (T['f'] >= lo) & (T['f'] < hi)
        Q.append(dict(frc_nm=frc2(R['x_dc'][k], R['y_dc'][k], T['f'][k])[0], frc_i_nm=frc2(R['x_dc'][k], R['y_dc'][k], T['f'][k])[1], n_frc=int(k.sum())))
    r['quarters'] = Q
    np.savez(f'{d}/frc_rows.npz', x=R['x_dc'][R['mask']].astype(np.float32), y=R['y_dc'][R['mask']].astype(np.float32), f=T['f'][R['mask']].astype(np.int32))
    return r


def main():
    setn = sys.argv[1]; names = (sys.argv[2] if len(sys.argv) > 2 else ','.join(V.ORDER)).split(',')
    stages = (sys.argv[3] if len(sys.argv) > 3 else 'solve,table').split(',')
    fn = f'{OUT}/movie_abl_{setn}.json'; res = json.load(open(fn)) if os.path.exists(fn) else {}
    for name in names:
        t0 = time.time()
        if 'solve' in stages: solve(setn, name)
        if 'table' in stages:
            k = SETS[setn]['kind']
            res[name] = fit_sigu(name) if k == 'fit' else sim_metrics(setn, name) if k == 'sim' else real_metrics(name)
            json.dump(res, open(fn, 'w'), indent=1, default=float)
            r = res[name]
            if k == 'fit': print(f'{setn} {name}: g(u) {np.round(r["g"], 3).tolist()} q {r["q"]:.3f} held-out {json.dumps({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r["heldout"].items() if kk != "zc_by_u"})}', flush=True)
            else: print(f'{setn} {name}: ' + json.dumps({kk: vv for kk, vv in r.items() if kk not in ('quarters', 'gt_match')}, default=float)[:600] + ' | quarters FRC ' + str([round(q['frc_nm'], 2) for q in r['quarters']]), flush=True)
        print(f'{setn} {name} done {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
