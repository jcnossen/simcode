"""Offline evaluation of the pass-2 benchmark dumps (simcode/processing/gate_dump.py --out <dir>); until 2026-11-16 analysis/calib/plib.py
(the paper_figures branch keeps that name as an alias of this module):
  load(scene, front, d=<dir>)   seeds pooled, FOV ids made unique
  targets(R)                    gate_scenes_report_detector_matched matching: candidates score >= 0.5 matched one-to-one (1 px) to targets on the
                                DETECTOR point x_det
  solve(R, oi, cal, prior)      (3D head dumps: + z, sz after the solve, zp = the head's z) explicit pattern solve (simcode/solve/solve.solve_emitters; fringe / kinetic posterior; c = c_a = 0: the gate scenes' constant brightness) on
                                the dumped head outputs; prior 'own' (the head's prior) or 'pf' (the pattern-free prior of pretrain_emitter_pass1 on
                                the same candidates, pass-2 dumps only); 'direct' = the head prior itself (variant b) with
                                sigma = s_prior
  fit_cal(...)                  prior_off from iso (700-2000 ph)
  prec(...)                     RMS / CRLB_SF_multi, trimmed, zcal, gross, per photon / nn bin; paired diff with FOV bootstrap
  detect(...)                   recall / FP / JI on the final positions (0.5 px, gate_scenes_report.detection rule), matched-FP sweep
"""
import glob, json, os
import numpy as np, torch
from simcode.benchmark.gate_scenes_report import match

PX = 108.33
S = 64
PHB = [(200, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 4000), (4000, 1e9), (200, 2000), (200, 1e9)]
NNB = [(0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 1e9)]
EDG = [100, 300, 500, 800, 1200, 2000, 3000, 5000, 1e9]
rob = lambda v: float(1.4826 * np.median(np.abs(v[np.isfinite(v)])))


def load(sc, front, seeds=(301, 302, 303), tag='', *, d):
    out = []; off = 0
    for sd in seeds:
        fn = f'{d}/{sc}__s{sd}_{front}{tag}.npz'
        if not os.path.exists(fn): continue
        D = dict(np.load(fn))
        D['gt_fov'] = D['gt_fov'] + off; D['sc_b'] = D['sc_b'] + off
        D['pfov'] = np.arange(int(D['n_fov'])) + off
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
def solve(R, oi, cal, prior='own', patt='est', chunk=8192, dev='cuda'):
    """(cal: the model's solve calibration; the gate scenes' brightness noise c = c_a = 0 of simcode.simulate.crlb is added)"""
    from simcode.emitter import model as SCM
    from simcode.solve import solve as SV
    from simcode.solve import z_conditioning as ZC
    from simcode.simulate.crlb import BRIGHTNESS_NOISE
    cal = SV.solve_cal(cal, *BRIGHTNESS_NOISE)
    ph = R['sc_I'][oi].sum(1)
    pre = 'sc_pf_' if prior == 'pf' else 'sc_'
    if prior == 'direct':
        x, y = R['sc_x'][oi], R['sc_y'][oi]
        po = np.array(cal.get('prior_off', [0.0, 0.0]))
        s = np.exp(R['sc_lsxy'][oi])
        return dict(x=x - po[0], y=y - po[1], sx=s[:, 0], sy=s[:, 1], q=np.zeros(len(oi)), xp=x - po[0], yp=y - po[1])
    P = torch.tensor(R['patt_est' if patt == 'est' else 'patt_true'], device=dev, dtype=torch.float32)
    tt = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
    z3 = 'sc_z' in R and prior != 'pf'          # 3D head dump: z after the solve (z_conditioning.z_columns)
    res = {k: [] for k in ('x', 'y', 'sx', 'sy', 'q', 'xp', 'yp') + (('z', 'sz', 'zp') if z3 else ())}
    for c0 in range(0, len(oi), chunk):
        j = oi[c0:c0 + chunk]
        o = dict(I=tt(R['sc_I'][j]), L=tt(R['sc_L'][j]), C=tt(R['sc_C'][j]), x=tt(R[pre + 'x'][j]), y=tt(R[pre + 'y'][j]), lsxy=tt(R[pre + 'lsxy'][j]),
                 x_det=tt(R['sc_x_det'][j]), y_det=tt(R['sc_y_det'][j]))
        if z3: o.update(z=tt(R['sc_z'][j]), lsz=tt(R['sc_lsz'][j]), bz=tt(R['sc_bz'][j]))
        b = torch.tensor(R['sc_b'][j], device=dev).long()
        s = SV.solve_emitters(o, P, b, cal)
        if z3:
            zc = ZC.z_columns(o, s); res['z'].append(zc['z_hy0']); res['sz'].append(zc['sz_cal']); res['zp'].append(o['z'])
        res['x'].append(s['r'][:, 0]); res['y'].append(s['r'][:, 1]); res['sx'].append(s['s_cal'][:, 0]); res['sy'].append(s['s_cal'][:, 1])
        res['q'].append(s['q'])
        rp = s.get('r_prior', torch.stack([o['x'], o['y']], 1)); res['xp'].append(rp[:, 0]); res['yp'].append(rp[:, 1])
    return {k: torch.cat(v).cpu().numpy() for k, v in res.items()}


def fit_cal(Rs, base, prior='own', iso=None, verbose=True):
    """solve calibration of a pass except its sigma layer: prior_off (the constant prior-position bias) from the iso dump
    (exact pattern, 700-2000 ph); box, mix_T from base (a fixed brightness noise cmult / cmult_ang of base is dropped: it is
    estimated per dataset with the pattern since 2026-11-17). Rs (tuning-seed dumps of the fit scenes) are
    no longer used for a fit since the per-photon-bin solve weights kap_p / kap_I2 were removed (2026-11-17; the solve uses the
    network's own covariances); kept in the signature for the callers. The reported sigma is the solve's posterior sd with the
    emitter network's joint prior / intensity covariance (no sigma factor)."""
    cal = {k: v for k, v in base.items() if k not in ('kap_bins', 'prior_off', 'cmult', 'cmult_ang')}
    pre = 'sc_pf_' if prior == 'pf' else 'sc_'
    po = np.zeros(2)
    if iso is not None:
        ti, oi = targets(iso)
        ph = iso['gt_phot'][ti]; m = (ph >= 700) & (ph < 2000)
        e = np.c_[iso[pre + 'x'][oi], iso[pre + 'y'][oi]] - np.c_[iso['gt_x'][ti], iso['gt_y'][ti]]
        po = np.mean(e[m], 0)
    cal['prior_off'] = po.tolist()
    if verbose:
        print('  prior_off nm', (po * PX).round(2), flush=True)
    return cal


def boot_w(fov, nboot=200, seed=0):
    rng = np.random.default_rng(seed)
    uf, inv = np.unique(fov, return_inverse=True)
    return [np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf))[inv] for _ in range(nboot)]


def rr(e, c, w=None):
    if w is not None: e, c = np.repeat(e, w, 0), np.repeat(c, w, 0)
    return float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(c ** 2)))


def stats(e, c, s):
    zc = np.hypot(*(e / c).T)
    return dict(n=int(len(e)), rms=rr(e, c), trim=rr(e[zc <= 5], c[zc <= 5]) if (zc <= 5).any() else np.nan,
                zcal=rob(e / s) if s is not None else np.nan, g4=float(100 * np.mean(np.hypot(*(e / s).T) > 4)) if s is not None else np.nan,
                c5=float(100 * np.mean(zc > 5)), nm=float(np.sqrt(np.mean(e ** 2)) * PX))


def resolvable(R, ti):
    c = R['gt_sf_m'][ti]
    return np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * PX < 25)


def paired(R, A, B, nboot=200, bins=PHB, key='ph'):
    """A, B: dicts ti -> (e [2], s [2]) as arrays: A = dict(ti=, e=, s=); paired on common ti; returns per bin stats of both + d"""
    common, ia, ib = np.intersect1d(A['ti'], B['ti'], return_indices=True)
    c = R['gt_sf_m'][common]; ok = np.isfinite(c).all(1) & (np.hypot(*c.T) / np.sqrt(2) * PX < 25)
    ea, eb, sa, sb = A['e'][ia], B['e'][ib], A['s'][ia], B['s'][ib]
    ok &= np.isfinite(ea).all(1) & np.isfinite(eb).all(1)
    v = R['gt_phot'][common] if key == 'ph' else R['gt_nn'][common]
    fov = R['gt_fov'][common]
    out = {}
    for lo, hi in bins:
        m = ok & (v >= lo) & (v < hi)
        if m.sum() < 20: continue
        W = boot_w(fov[m], nboot)
        d = [rr(ea[m], c[m], w) - rr(eb[m], c[m], w) for w in W]
        out[f'{lo}-{hi:g}'] = dict(A=stats(ea[m], c[m], sa[m]), B=stats(eb[m], c[m], sb[m]),
                                  d=rr(ea[m], c[m]) - rr(eb[m], c[m]), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    return out


def errs(R, ti, oi, sol):
    tr = np.c_[R['gt_x'][ti], R['gt_y'][ti]]
    return dict(ti=ti, oi=oi, e=np.c_[sol['x'], sol['y']] - tr, s=np.c_[sol['sx'], sol['sy']], ep=np.c_[sol['xp'], sol['yp']] - tr)


def detect(R, xf, yf, sel, r=0.5):
    """recall / FP / JI of the outputs `sel` (bool over all candidates) at positions xf, yf (arrays over all candidates)"""
    tgt = R['gt_target'] > 0
    ob, ox, oy = R['sc_b'][sel], xf[sel], yf[sel]
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], ob, ox, oy, r)
    inter = (ox >= 9) & (ox <= S - 10) & (oy >= 9) & (oy <= S - 10)
    tp = (og >= 0) & tgt[np.maximum(og, 0)]
    fp = (og < 0) & inter
    TP, FP, NT = int(tp.sum()), int(fp.sum()), int(tgt.sum())
    return dict(TP=TP, FP=FP, recall=TP / max(NT, 1), JI=TP / max(TP + FP + NT - TP, 1), fp_per_fov=FP / max(R['n_fov'], 1),
                hit=og[tp])
