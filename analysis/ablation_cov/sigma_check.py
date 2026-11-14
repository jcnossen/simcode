"""Covariance ablation, part 4: does the c term matter for the reported UNCERTAINTY (2026-11-13).
gate  pass-1 dumps of rs05qm seed 301 (out/ablation_cov/fe_ctl, gate_dump.py) with a known emitter fluctuation: ctl (none, exact model),
      f10 / f20 (per-frame sd 0.10 / 0.20), f10a10 (+ per-angle 0.10), full (production mismatch scene), mis (PSF/camera/pattern error only).
      Variants A, F, H, C, D (variants.py) with their own pass-1 calibrations (calibs/calib_s1_<v>.json, fitted on tuning seed 202 of
      rs05qm + rs05pm = the 'full' fluctuation level), and Aauto = A with (c, c_a) set to the automatic estimate of that scene
      (autoc_est.json, gate:<tag>). Per GT-photon bin: robust sd of error / sigma (zcal, 1 = calibrated) for the reported sigma (s_cal: chi2
      inflation + per-photon-bin factor) and for the plain Fisher sigma of the solve (s, no rescale at all), % |z| > 3, RMS (nm),
      median q/dof (RMS in nm: the fe_ctl dumps have no CRLB); FOV-bootstrap 95 % CI of zcal(variant) - zcal(A).
real  revisit test without ground truth on 1809_2 pass 2 (runs/real/<v>/pass2_est2_*.npz, all rows p >= 0.5, before g(u)): pairs of
      localisations of the same binding event in consecutive NON-overlapping windows (window start + 6 frames, nearest within 0.5 px):
      z = (x2 - x1) / sqrt(sx1^2 + sx2^2); robust sd per mean-photon bin (1 = sigma calibrated, assuming the emitter did not move), and the
      solve's median q/dof per photon bin. Validated on the simulated movies D1 (constant brightness) and m05r3 (fluctuating) where the
      same pairs also have a ground-truth zcal.
usage: CUDA_VISIBLE_DEVICES=2 python analysis/ablation_cov/sigma_check.py [gate,real] -> out/ablation_cov/sigma_check_{gate,real}.json"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'analysis', 'calib')); sys.path.insert(0, HERE)
import variants as V
from simcode.util import proctitle
proctitle.set_title('covabl')
OUT = os.path.join(REPO, 'out', 'ablation_cov'); PX = 108.33
rob = lambda z: float(1.4826 * np.median(np.abs(np.asarray(z) - np.median(z))))
GB = [(200, 700), (700, 1200), (1200, 2000), (2000, 3000), (3000, 1e9)]
TAGS = ['ctl', 'f10', 'f20', 'f10a10', 'full', 'mis']
VS = ['A', 'F', 'H', 'C', 'D', 'Aauto']


def solve_full(R, oi, cal, v):
    import torch
    from simcode.hd import sc as SCM
    dev = 'cuda'; tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    P_ = tt(R['patt_est']); out = {k: [] for k in ('x', 'y', 's', 's_cal', 'q')}
    with torch.no_grad():
        for c0 in range(0, len(oi), 8192):
            j = oi[c0:c0 + 8192]
            o = dict(I=tt(R['sc_I'][j]), L=tt(V.L_variant(R['sc_L'][j], v['L'])), x=tt(R['sc_x'][j]), y=tt(R['sc_y'][j]), lsxy=tt(R['sc_lsxy'][j]),
                     x_det=tt(R['sc_x_det'][j]), y_det=tt(R['sc_y_det'][j]))
            s = SCM.sc_solve(o, P_, torch.tensor(R['sc_b'][j], device=dev).long(), cal)
            out['x'].append(s['r'][:, 0]); out['y'].append(s['r'][:, 1]); out['s'].append(s['s']); out['s_cal'].append(s['s_cal']); out['q'].append(s['q'])
    return {k: torch.cat(w).cpu().numpy() for k, w in out.items()}


def gate():
    import plib as P
    AE = json.load(open(f'{OUT}/autoc_est.json')); res = {}
    for tag in TAGS:
        R = P.load('rs05qm', 's1', seeds=(301,), tag='_' + tag, d=f'{OUT}/fe_ctl'); ti, oi = P.targets(R)
        ok = np.ones(len(ti), bool)          # these dumps carry no CRLB (gt_sf_m NaN): all detector-matched targets
        ph = R['gt_phot'][ti]; fov = R['gt_fov'][ti]; tr = np.c_[R['gt_x'][ti], R['gt_y'][ti]]
        E = {}
        for n in VS:
            v = V.VARS['A' if n == 'Aauto' else n]; cal = json.load(open(f'{OUT}/calibs/calib_s1_{"A" if n == "Aauto" else n}.json'))
            if n == 'Aauto': th = AE['gate:' + tag]['theta']; cal['cmult'], cal['cmult_ang'] = float(th[1]), float(th[2])
            else: cal = V.cal_variant(cal, v)
            S = solve_full(R, oi, cal, v); E[n] = dict(e=np.c_[S['x'], S['y']] - tr, s=S['s_cal'], f=S['s'], q=S['q'])
        res[tag] = {}
        for lo, hi in GB:
            m = ok & (ph >= lo) & (ph < hi)
            if m.sum() < 50: continue
            W = P.boot_w(fov[m], 200); b = {}
            for n in VS:
                e, s, f = E[n]['e'][m], E[n]['s'][m], E[n]['f'][m]; z = e / s
                zb = [rob(np.repeat(z, w, 0).ravel()) for w in W]
                b[n] = dict(n=int(m.sum()), zcal=rob(z.ravel()), zcal_ci=[float(np.percentile(zb, 2.5)), float(np.percentile(zb, 97.5))], zcal_fisher=rob((e / f).ravel()),
                            z_gt3=float(100 * np.mean(np.abs(z) > 3)), rms_nm=float(np.sqrt(np.mean(e ** 2)) * PX), q_med=float(np.median(E[n]['q'][m])), zb=zb)
            for n in VS:
                if n != 'A':
                    d = np.array(b[n]['zb']) - np.array(b['A']['zb']); b[n]['dzcal_ci_vs_A'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            for n in VS: b[n].pop('zb')
            res[tag][f'{lo:g}-{hi:g}'] = b
            print(tag, f'{lo:g}-{hi:g}', ' '.join(f'{n}: zc {b[n]["zcal"]:.2f} [{b[n]["zcal_ci"][0]:.2f},{b[n]["zcal_ci"][1]:.2f}] zF {b[n]["zcal_fisher"]:.2f} >3 {b[n]["z_gt3"]:.1f}% q {b[n]["q_med"]:.2f}' for n in VS), flush=True)
        json.dump(res, open(f'{OUT}/sigma_check_gate.json', 'w'), indent=1)


RB = [(200, 500), (500, 800), (800, 1200), (1200, 1800), (1800, 2600), (2600, 1e9)]


def pairs(d, gt=None):
    """consecutive non-overlapping windows of the same event -> per photon bin robust sd of the normalised difference, median q/dof"""
    from scipy.spatial import cKDTree
    fs = sorted(glob.glob(f'{d}/pass2_est2_*.npz'), key=lambda s: int(s.split('_')[-1][:-4]))
    X = {k: [] for k in ('w', 'x', 'y', 'sx', 'sy', 'ph', 'q')}
    for fn in fs:
        D = np.load(fn); k = D['p'] >= 0.5
        for a, b in (('w', 'wstart'), ('x', 'x_hy0'), ('y', 'y_hy0'), ('sx', 'sx_cal'), ('sy', 'sy_cal'), ('ph', 'phot'), ('q', 'q_dof')): X[a].append(D[b][k])
    X = {k: np.concatenate(v).astype(np.float64) for k, v in X.items()}; w = X['w'].astype(np.int64)
    order = np.argsort(w, kind='stable'); X = {k: v[order] for k, v in X.items()}; w = w[order]
    uw, st = np.unique(w, return_index=True); en = np.r_[st[1:], len(w)]; idx = {int(a): (s, e) for a, s, e in zip(uw, st, en)}
    I1, I2 = [], []
    for a, (s, e) in idx.items():
        if a + 6 not in idx: continue
        s2, e2 = idx[a + 6]
        t = cKDTree(np.c_[X['x'][s2:e2], X['y'][s2:e2]]); dd, j = t.query(np.c_[X['x'][s:e], X['y'][s:e]], distance_upper_bound=0.5)
        k = np.isfinite(dd)
        I1.append(np.arange(s, e)[k]); I2.append(s2 + j[k])
    I1, I2 = np.concatenate(I1), np.concatenate(I2)
    u, cnt = np.unique(I2, return_counts=True); dup = np.isin(I2, u[cnt > 1]); I1, I2 = I1[~dup], I2[~dup]
    zx = (X['x'][I2] - X['x'][I1]) / np.hypot(X['sx'][I1], X['sx'][I2]); zy = (X['y'][I2] - X['y'][I1]) / np.hypot(X['sy'][I1], X['sy'][I2])
    ph = 0.5 * (X['ph'][I1] + X['ph'][I2]); out = dict(n_pairs=int(len(I1)), bins={})
    rng = np.random.default_rng(0)
    for lo, hi in RB:
        m = (ph >= lo) & (ph < hi)
        if m.sum() < 200: continue
        z = np.r_[zx[m], zy[m]]; bs = [rob(z[rng.integers(0, len(z), len(z))]) for _ in range(50)]
        qq = np.r_[X['q'][I1][m], X['q'][I2][m]]
        out['bins'][f'{lo:g}-{hi:g}'] = dict(n=int(m.sum()), ph=float(np.median(ph[m])), z_rob=rob(z), z_se=float(np.std(bs)), q_med=float(np.median(qq)),
                                            sig_nm=float(np.median(np.hypot(X['sx'][I1][m], X['sx'][I2][m])) / np.sqrt(2) * PX))
    return out


def real():
    res = {}
    for setn in ('real', 'D1', 'm05r3'):
        for n in ('A', 'F', 'H', 'C', 'D'):
            d = f'{OUT}/runs/{setn}/{n}'
            if not glob.glob(f'{d}/pass2_est2_*.npz'): continue
            r = pairs(d); res[f'{setn}:{n}'] = r
            print(setn, n, r['n_pairs'], ' | '.join(f'{k}: z {b["z_rob"]:.3f}+-{b["z_se"]:.3f} q {b["q_med"]:.2f} sig {b["sig_nm"]:.1f}' for k, b in r['bins'].items()), flush=True)
            json.dump(res, open(f'{OUT}/sigma_check_real.json', 'w'), indent=1)


def table():
    G = json.load(open(f'{OUT}/sigma_check_gate.json')); R = json.load(open(f'{OUT}/sigma_check_real.json'))
    D = dict(ctl='exact model, no fluctuation', mis='PSF / camera / pattern error only', f10='per-frame sd 0.10', f10a10='per-frame 0.10 + per-angle 0.10',
             full='production mismatch (= calibration tuning level)', f20='per-frame sd 0.20')
    L = ['\n**Gate rs05qm seed 301, pass 1: robust sd of error / sigma per GT-photon bin (1 = calibrated; > 1 = sigma too small).** Reported sigma '
         '(after each variant\'s own chi2 inflation + per-photon-bin factor) for A, F, H, C, D, Aauto [95 % FOV-bootstrap CI for A]; Fisher = the solve\'s '
         'plain sigma without any rescale, for A and C; % |z| > 3 (Gaussian 0.27 % per coordinate) for A, C, Aauto; median q/dof for A and C.\n',
         '| scene | photons | n | A | F | H | C | D | Aauto | Fisher A | Fisher C | >3 A / C / Aauto % | q/dof A / C |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for t in ('ctl', 'mis', 'f10', 'f10a10', 'full', 'f20'):
        for b, r in G[t].items():
            a = r['A']
            L.append(f'| {D[t]} | {b} | {a["n"]} | {a["zcal"]:.2f} [{a["zcal_ci"][0]:.2f}, {a["zcal_ci"][1]:.2f}] | ' + ' | '.join(f'{r[n]["zcal"]:.2f}' for n in ('F', 'H', 'C', 'D', 'Aauto'))
                     + f' | {a["zcal_fisher"]:.2f} | {r["C"]["zcal_fisher"]:.2f} | {a["z_gt3"]:.1f} / {r["C"]["z_gt3"]:.1f} / {r["Aauto"]["z_gt3"]:.1f} | {a["q_med"]:.2f} / {r["C"]["q_med"]:.2f} |')
    L += ['\n**Revisit test without ground truth (pass 2, all rows p >= 0.5, sigma before g(u)): robust sd of the normalised difference of consecutive '
          'non-overlapping windows of the same event, per mean-photon bin (± bootstrap se), and median q/dof.** Absolute values are biased low '
          'by errors shared between adjacent windows (see text); compare variants and the trend with brightness.\n',
          '| data | variant | ' + ' | '.join(R['real:A']['bins']) + ' |', '|---|---|' + '---|' * len(R['real:A']['bins'])]
    for k, r in R.items():
        st, n = k.split(':')
        L.append(f'| {st} | {n} | ' + ' | '.join(f'{v["z_rob"]:.3f} (q {v["q_med"]:.2f})' for v in r['bins'].values()) + ' |')
    L.append('\nPairs: ' + ', '.join(f'{k} {r["n_pairs"]}' for k, r in R.items() if k.endswith(':A')) + '; photon-bin counts in sigma_check_real.json.')
    open(f'{OUT}/tab_sigma.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    w = (sys.argv[1] if len(sys.argv) > 1 else 'gate,real,table').split(',')
    if 'gate' in w: gate()
    if 'real' in w: real()
    if 'table' in w: table()
