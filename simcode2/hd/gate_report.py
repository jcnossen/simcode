"""
Gate metrics for simcode2/hd/gate.py outputs (all seeds of a scene pooled; bootstrap over FOVs).
Detection (per score threshold): Hungarian matching of the front-end's FINAL positions to all active GT within
0.5 px per FOV; TP = matched to a visible interior target, matches to dim / edge GT are ignored, unmatched interior
outputs = FP. precision, recall, JI = TP / (TP + FP + FN), FP per FOV.
Precision (500-5k photons, at an operating point): targets matched one-to-one within 1.0 px; ratio = RMSD / RMS(CRLB_SF
multi) over resolvable targets (CRLB < 25 nm), trimmed (errors <= 50 nm) ratio, zmed (robust |e| / CRLB), gross % (> 50 nm),
z calibration (robust e / sigma), per nearest-neighbour bin; 95 % bootstrap CIs (200 FOV resamples).
Operating points: 'thr0.5' (each front-end's own p > 0.5) and 'matchedFP' (each front-end at the lowest threshold whose
FP / FOV <= the det front-end's FP / FOV at p > 0.5).
usage: python -m simcode2.hd.gate_report --dir <out>/gate --scen iso,pair1.5,... --out <out>/gate_report.json
"""
import argparse, glob, json, os
from simcode2 import config
import numpy as np
from scipy.optimize import linear_sum_assignment

PX = 108.33
S = 64
# user decision 2026-11-09: only the network (sc) path is reported (the linear-fit fronts pf / det / hd stay on disk, unused);
# the exact-PSF single-emitter MLE ('mle', iso only) is a sanity line
METH = {'sc': ['sc'], 'mle': ['mle']}
THRS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NNB = [(0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 1e9)]


def load(d, sc, tag=''):
    import re
    pat = re.compile(re.escape(sc) + r'__s\d+' + re.escape(tag) + r'\.npz$')
    fs = sorted(f for f in glob.glob(f'{d}/{sc}__s*.npz') if pat.match(os.path.basename(f)))
    Ds = [dict(np.load(f)) for f in fs]
    off = 0; out = []
    for D in Ds:       # make FOV ids unique across seeds
        D = dict(D)
        for k in list(D):
            if k == 'gt_fov' or k.endswith('_b'): D[k] = D[k] + off
        off += int(D['n_fov']) + 1
        out.append(D)
    keys = set.intersection(*[set(D) for D in out])
    return {k: np.concatenate([D[k] for D in out]) if out[0][k].ndim else np.array(sum(int(D[k]) for D in out)) for k in keys}, len(fs)


def match(gb, gx, gy, ob, ox, oy, r):
    """one-to-one per FOV within r px -> og [n_out] index of GT or -1"""
    og = np.full(len(ob), -1)
    order_g = np.argsort(gb, kind='stable'); order_o = np.argsort(ob, kind='stable')
    ug, sg = np.unique(gb[order_g], return_index=True); eg = np.r_[sg[1:], len(gb)]
    uo, so = np.unique(ob[order_o], return_index=True); eo = np.r_[so[1:], len(ob)]
    gi = dict(zip(ug, zip(sg, eg)))
    for f, a, b in zip(uo, so, eo):
        if f not in gi: continue
        i_o = order_o[a:b]; a2, b2 = gi[f]; i_g = order_g[a2:b2]
        D = (ox[i_o, None] - gx[None, i_g]) ** 2 + (oy[i_o, None] - gy[None, i_g]) ** 2
        near = D < r * r
        if not near.any(): continue
        Cc = np.where(near, D, 1e6)
        rr, cc = linear_sum_assignment(Cc)
        ok = Cc[rr, cc] < r * r
        og[i_o[rr[ok]]] = i_g[cc[ok]]
    return og


def detection(R, f, m, thr):
    gb, gx, gy = R['gt_fov'], R['gt_x'], R['gt_y']
    tgt = R['gt_target'] > 0
    sel = (R[f'{f}_score'] >= thr) & np.isfinite(R[f'{f}_x_{m}']) & np.isfinite(R[f'{f}_y_{m}'])
    ob, ox, oy = R[f'{f}_b'][sel], R[f'{f}_x_{m}'][sel], R[f'{f}_y_{m}'][sel]
    og = match(gb, gx, gy, ob, ox, oy, 0.5)
    inter = (ox >= 9) & (ox <= S - 10) & (oy >= 9) & (oy <= S - 10)
    tp = (og >= 0) & tgt[np.maximum(og, 0)]
    fp = (og < 0) & inter
    TP, FP = int(tp.sum()), int(fp.sum()); NT = int(tgt.sum())
    nf = len(np.unique(gb))
    return dict(thr=thr, TP=TP, FP=FP, recall=TP / max(NT, 1), precision=TP / max(TP + FP, 1), JI=TP / max(TP + FP + NT - TP, 1),
                fp_per_fov=FP / max(nf, 1))


def precision_metrics(R, f, m, thr, lo=500, hi=5000, nboot=200, seed=0):
    gb, gx, gy = R['gt_fov'], R['gt_x'], R['gt_y']
    sel = (R[f'{f}_score'] >= thr) & np.isfinite(R[f'{f}_x_{m}'])
    idx = np.nonzero(sel)[0]
    og = match(gb, gx, gy, R[f'{f}_b'][idx], R[f'{f}_x_{m}'][idx], R[f'{f}_y_{m}'][idx], 1.0)
    ti = og[og >= 0]; oi = idx[og >= 0]
    tgt = R['gt_target'][ti] > 0; ti, oi = ti[tgt], oi[tgt]
    ph = R['gt_phot'][ti]; inb = (ph >= lo) & (ph < hi); ti, oi = ti[inb], oi[inb]
    e = np.stack([R[f'{f}_x_{m}'][oi] - R['gt_x'][ti], R[f'{f}_y_{m}'][oi] - R['gt_y'][ti]], 1)
    c = R['gt_sf_m'][ti]
    s = R[f'{f}_s_{m}'][oi] if f'{f}_s_{m}' in R else None
    sc = R[f'{f}_scal_{m}'][oi] if f'{f}_scal_{m}' in R else None
    ok = np.isfinite(e).all(1) & np.isfinite(c).all(1)
    res = ok & (np.hypot(c[:, 0], c[:, 1]) / np.sqrt(2) * PX < 25)
    fovs = R['gt_fov'][ti]
    nt = int(((R['gt_target'] > 0) & (R['gt_phot'] >= lo) & (R['gt_phot'] < hi)).sum())

    def stats(msk):
        ee, cc = e[msk], c[msk]
        en = np.hypot(ee[:, 0], ee[:, 1]) * PX
        g = en <= 50
        cr = np.sqrt(np.mean(cc ** 2))
        return dict(ratio=float(np.sqrt(np.mean(ee ** 2)) / cr), trim=float(np.sqrt(np.mean(ee[g] ** 2)) / np.sqrt(np.mean(cc[g] ** 2))),
                    zmed=float(1.4826 * np.median(np.abs(ee / cc))), gross=float(100 * (~g).mean()))
    out = dict(n=int(res.sum()), recall1px=float(len(ti) / max(nt, 1)), **stats(res))
    # bootstrap over FOVs
    rng = np.random.default_rng(seed)
    uf = np.unique(fovs[res]); B = []
    fi = {u: np.nonzero((fovs == u) & res)[0] for u in uf}
    for _ in range(nboot):
        pick = rng.choice(uf, len(uf)); ii = np.concatenate([fi[u] for u in pick])
        msk = np.zeros(len(res), bool); w = np.bincount(ii, minlength=len(res))
        ee, cc = np.repeat(e, w, 0), np.repeat(c, w, 0)
        en = np.hypot(ee[:, 0], ee[:, 1]) * PX
        B.append([np.sqrt(np.mean(ee ** 2)) / np.sqrt(np.mean(cc ** 2)), 100 * (en > 50).mean()])
    B = np.array(B)
    out['ratio_ci'] = [float(np.percentile(B[:, 0], 2.5)), float(np.percentile(B[:, 0], 97.5))]
    out['gross_ci'] = [float(np.percentile(B[:, 1], 2.5)), float(np.percentile(B[:, 1], 97.5))]
    if s is not None:
        z = (e / s)[res]; out['zcal_sigma'] = float(1.4826 * np.median(np.abs(z[np.isfinite(z).all(1)])))
    if sc is not None:
        z = (e / sc)[res]; out['zcal_scal'] = float(1.4826 * np.median(np.abs(z[np.isfinite(z).all(1)])))
    nb_ = {}
    nn = R['gt_nn'][ti]
    for a, b in NNB:
        mk = res & (nn >= a) & (nn < b)
        tall = (R['gt_target'] > 0) & (R['gt_phot'] >= lo) & (R['gt_phot'] < hi) & (R['gt_nn'] >= a) & (R['gt_nn'] < b)
        if mk.sum() >= 20:
            nb_[f'{a:g}-{b:g}'] = dict(n=int(mk.sum()), recall1px=float(((nn >= a) & (nn < b)).sum() / max(tall.sum(), 1)), **stats(mk))
    out['nn_bins'] = nb_
    return out


def intensity_metrics(R, f, thr, bins=((300, 1000), (1000, 3000), (3000, 20000), (500, 5000))):
    """network per-frame intensities of matched targets (1 px): modulation slope (regression of the mean-removed I_net on
    the mean-removed I_gt), bias, efficiency var(I_net - I_gt) / CRLB_free(multi)^2 per frame, z-calibration of sI_net."""
    sel = R[f'{f}_score'] >= thr
    idx = np.nonzero(sel)[0]
    og = match(R['gt_fov'], R['gt_x'], R['gt_y'], R[f'{f}_b'][idx], R[f'{f}_x_{f if f != "sc" else "sc"}'][idx] if False else R[f'{f}_x_det'][idx],
               R[f'{f}_y_det'][idx], 1.0)
    ti = og[og >= 0]; oi = idx[og >= 0]
    tgt = R['gt_target'][ti] > 0; ti, oi = ti[tgt], oi[tgt]
    out = {}
    for lo, hi in bins:
        m = (R['gt_phot'][ti] >= lo) & (R['gt_phot'][ti] < hi)
        if m.sum() < 30: continue
        I = R[f'{f}_I_net'][oi[m]]; G = R['gt_I'][ti[m]]; Cf = R['gt_fr_m'][ti[m]]; sI = R[f'{f}_sI_net'][oi[m]]
        a = I - I.mean(1, keepdims=True); g = G - G.mean(1, keepdims=True)
        ok = np.isfinite(Cf).all(1)
        e = I - G
        out[f'{lo}-{hi}'] = dict(n=int(m.sum()), slope=float((a * g).sum() / (g * g).sum()),
                                 rel_bias=float(I.sum() / G.sum() - 1),
                                 eff=float(np.mean(e[ok] ** 2) / np.mean(Cf[ok] ** 2)),
                                 eff_robust=float(np.median((e[ok] / Cf[ok]) ** 2) / 0.4549),
                                 zcal=float(1.4826 * np.median(np.abs(e / sI))))
    return out


def report(d, sc, tag='', extra=()):
    R, nfiles = load(d, sc, tag)
    for etag, ename in extra:      # other runs on the same scenes (tag): their 'hd' outputs as front-end ename
        try:
            E, _ = load(d, sc, etag)
        except Exception:
            continue
        if len(E['gt_x']) != len(R['gt_x']) or not np.allclose(E['gt_x'], R['gt_x']):
            print('  WARNING GT mismatch', sc, etag); continue
        src = 'sc' if 'sc_b' in E else 'hd'
        for k in E:
            if k.startswith(src + '_'): R[ename + k[len(src):]] = E[k]
        METH[ename] = ['sc'] if src == 'sc' else ['hdf']
    fronts = [f for f in METH if f + '_b' in R]
    rep = dict(n_seeds=nfiles, n_targets=int((R['gt_target'] > 0).sum()), n_fov=int(len(np.unique(R['gt_fov']))))
    curves = {}
    for f in fronts:
        for m in METH[f]:
            if f'{f}_x_{m}' not in R: continue
            curves[f'{f}.{m}'] = [detection(R, f, m, t) for t in THRS]
    rep['pr'] = curves
    ref = next((c for c in curves.get('sc.sc', []) if c['thr'] == 0.5), None)     # matched FP: FP / FOV of sc at 0.5
    ops = {}
    for key, cur in curves.items():
        f, m = key.split('.')
        o = {'thr0.5': 0.5}
        if ref is not None:
            okc = [c['thr'] for c in cur if c['fp_per_fov'] <= ref['fp_per_fov'] + 1e-9]
            o['matchedFP'] = min(okc) if okc else max(THRS)
        ops[key] = {}
        for nm, t in o.items():
            det_ = next(c for c in cur if c['thr'] == t)
            ops[key][nm] = dict(thr=t, det=det_, prec=precision_metrics(R, f, m, t))
            if f'{f}_I_net' in R:
                ops[key][nm]['intensity'] = intensity_metrics(R, f, t)
    rep['ops'] = ops
    return rep


def line(sc, key, nm, q):
    p = q['prec']; dd = q['det']
    return (f'{sc:8s} {key:8s} {nm:9s} thr {q["thr"]:.1f} | P {dd["precision"]:.3f} R {dd["recall"]:.3f} JI {dd["JI"]:.3f} '
            f'FP/fov {dd["fp_per_fov"]:.2f} | ratio {p["ratio"]:.2f} [{p["ratio_ci"][0]:.2f},{p["ratio_ci"][1]:.2f}] trim {p["trim"]:.2f} '
            f'zmed {p["zmed"]:.2f} gross {p["gross"]:.2f}% [{p["gross_ci"][0]:.2f},{p["gross_ci"][1]:.2f}] '
            f'zcal {p.get("zcal_scal", p.get("zcal_sigma", float("nan"))):.2f} n {p["n"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(config.root('out'), 'gate'))
    ap.add_argument('--scen', default='iso,pair1,pair1.5,pair2,pair3,rs05x,rs05,rs25x,rs25,rs25s')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--extra', default='')     # e.g. _hd_c:hdc,_hd_c0:hdc0
    a = ap.parse_args()
    extra = [tuple(e.split(':')) for e in a.extra.split(',')] if a.extra else []
    REP = {}
    for sc in a.scen.split(','):
        if not glob.glob(f'{a.dir}/{sc}__s*{a.tag}.npz'): continue
        REP[sc] = report(a.dir, sc, a.tag, extra)
        for key, ops in REP[sc]['ops'].items():
            for nm, q in ops.items():
                print(line(sc, key, nm, q), flush=True)
    if a.out: json.dump(REP, open(a.out, 'w'), indent=1)
