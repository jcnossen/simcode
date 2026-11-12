"""Offline pattern-solve evaluation on the dumped gate head outputs (dump_fe.py): re-run simcode.hd.sc.sc_solve with a
calibration dict, match targets as gate_report_xdet (x_det, 1 px, score >= 0.5, resolvable CRLB < 25 nm), per-GT-photon-bin
RMS / CRLB_SF_multi, zcal, the oracle convex mix with the prior (per-bin optimal alpha), paired FOV-bootstrap CIs."""
import glob, json, os
from simcode import config as _cfg
import numpy as np, torch
from simcode.hd.gate_report import match

PX = 108.33
FE = os.path.join(_cfg.root('out'), 'bright', 'fe')
GATE = os.path.join(_cfg.root('out'), 'gate')
PHB = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (2000, 1e9)]


def load(sc, seeds=(301, 302, 303), name='sc5nb'):
    out = []; off = 0
    for sd in seeds:
        D = dict(np.load(f'{FE}/{sc}__s{sd}_{name}.npz'))
        if 'gt_sf_m' not in D:
            for f in sorted(glob.glob(f'{GATE}/{sc}__s{sd}_hd_*.npz')):
                R = np.load(f)
                if len(R['gt_x']) == len(D['gt_x']) and np.allclose(R['gt_x'], D['gt_x']) and 'gt_sf_m' in R.files:
                    D['gt_sf_m'] = R['gt_sf_m']; break
        if 'gt_sf_m' not in D: D['gt_sf_m'] = np.full((len(D['gt_x']), 2), np.nan, np.float32)
        D['gt_fov'] = D['gt_fov'] + off; D['sc_b'] = D['sc_b'] + off
        off += int(D['n_fov'])
        out.append(D)
    keys = set.intersection(*[set(D) for D in out]) - {'n_fov'}
    R = {k: np.concatenate([D[k] for D in out]) for k in keys}
    R['n_fov'] = off
    return R


def targets(R, thr=0.5):
    idx = np.nonzero(R['sc_score'] >= thr)[0]
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], R['sc_b'][idx], R['sc_x_det'][idx], R['sc_y_det'][idx], 1.0)
    ti = og[og >= 0]; oi = idx[og >= 0]
    k = R['gt_target'][ti] > 0
    return ti[k], oi[k]


@torch.no_grad()
def solve(R, oi, cal, patt='est', chunk=8192, dev='cuda'):
    from simcode.hd import sc as SCM
    P = torch.tensor(R['patt_est' if patt == 'est' else 'patt_true'], device=dev, dtype=torch.float32)
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    res = {k: [] for k in ('x', 'y', 'sx', 'sy', 'q', 'p', 'xp', 'yp', 'xmap', 'ymap')}
    for c0 in range(0, len(oi), chunk):
        j = oi[c0:c0 + chunk]
        o = dict(I=tt(R['sc_I'][j]), L=tt(R['sc_L'][j]), x=tt(R['sc_x'][j]), y=tt(R['sc_y'][j]), lsxy=tt(R['sc_lsxy'][j]),
                 x_det=tt(R['sc_x_det'][j]), y_det=tt(R['sc_y_det'][j]))
        b = torch.tensor(R['sc_b'][j], device=dev).long()
        s = SCM.sc_solve(o, P, b, cal)
        res['x'].append(s['r'][:, 0]); res['y'].append(s['r'][:, 1]); res['sx'].append(s['s_cal'][:, 0]); res['sy'].append(s['s_cal'][:, 1])
        res['q'].append(s['q']); res['p'].append(s['p'])
        rp = s.get('r_prior', torch.stack([o['x'], o['y']], 1))
        res['xp'].append(rp[:, 0]); res['yp'].append(rp[:, 1])
        rm = s.get('r_map', s['r']); res['xmap'].append(rm[:, 0]); res['ymap'].append(rm[:, 1])
    return {k: torch.cat(v).cpu().numpy() for k, v in res.items()}


def rms_r(e, c, w=None):
    if w is not None: e, c = np.repeat(e, w, 0), np.repeat(c, w, 0)
    return float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(c ** 2)))


def boot_idx(fov, nboot=200, seed=0):
    rng = np.random.default_rng(seed)
    uf, inv = np.unique(fov, return_inverse=True)
    W = []
    for _ in range(nboot):
        cnt = np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf))
        W.append(cnt[inv])
    return W


ALPHA = np.linspace(0, 1, 41)


def evaluate(R, ti, sol, base=None, nboot=200, bins=PHB):
    """per bin: n, rms/C, trim, zcal, oracle-mix gain (alpha*), and the paired diff vs base (another sol) with CI"""
    tr = np.c_[R['gt_x'][ti], R['gt_y'][ti]]
    c = R['gt_sf_m'][ti]; ph = R['gt_phot'][ti]; fov = R['gt_fov'][ti]
    ok = np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * PX < 25)
    e = np.c_[sol['x'], sol['y']] - tr; ep = np.c_[sol['xp'], sol['yp']] - tr; s = np.c_[sol['sx'], sol['sy']]
    ok &= np.isfinite(e).all(1)
    eb = np.c_[base['x'], base['y']] - tr if base is not None else None
    out = {}
    for lo, hi in bins:
        m = ok & (ph >= lo) & (ph < hi)
        if m.sum() < 20: continue
        em, cm, epm = e[m], c[m], ep[m]
        zc = np.hypot(*(em / cm).T)
        r = dict(n=int(m.sum()), rms=rms_r(em, cm), trim=rms_r(em[zc <= 5], cm[zc <= 5]),
                 zcal=float(1.4826 * np.median(np.abs(em / s[m]))), gross_s4=float(100 * np.mean(np.hypot(*(em / s[m]).T) > 4)),
                 rms_prior=rms_r(epm, cm), bias_nm=(np.mean(em, 0) * PX).tolist(), bias_prior_nm=(np.mean(epm, 0) * PX).tolist())
        mix = [rms_r(a * epm + (1 - a) * em, cm) for a in ALPHA]
        k = int(np.argmin(mix)); r['alpha'] = float(ALPHA[k]); r['mix_gain'] = r['rms'] - mix[k]
        W = boot_idx(fov[m], nboot)
        gb = []
        for w in W:
            mm = [rms_r(a * epm + (1 - a) * em, cm, w) for a in (ALPHA[k],)]
            gb.append(rms_r(em, cm, w) - mm[0])
        r['mix_gain_ci'] = [float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]
        if eb is not None:
            ebm = eb[m]
            d = [rms_r(em, cm, w) - rms_r(ebm, cm, w) for w in W]
            r['d_rms'] = r['rms'] - rms_r(ebm, cm); r['d_ci'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        out[f'{lo}-{hi:.0f}'] = r
    return out


def fmt(rep, name):
    L = []
    for k, r in rep.items():
        s = (f'  {name:22s} {k:>10s} n {r["n"]:5d} rms {r["rms"]:.3f} trim {r["trim"]:.3f} zcal {r["zcal"]:.2f} g4 {r["gross_s4"]:.2f}% '
             f'prior {r["rms_prior"]:.2f} mix a*={r["alpha"]:.3f} gain {r["mix_gain"]:+.3f} [{r["mix_gain_ci"][0]:+.3f},{r["mix_gain_ci"][1]:+.3f}]')
        if 'd_rms' in r: s += f' | d {r["d_rms"]:+.3f} [{r["d_ci"][0]:+.3f},{r["d_ci"][1]:+.3f}]'
        L.append(s)
    return '\n'.join(L)
