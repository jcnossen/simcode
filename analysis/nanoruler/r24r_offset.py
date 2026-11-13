"""R24 reuse check, part D (2026-11-13): camera offset of the nr40 / nr20 movies, 99 or 100 ADU?  CPU only, reads frames of the
reordered caches (read-only) and the quick-SMLM detection tables; nr80 is the control (offset 100 taken as right).

Pixel model (gain g = 0.45 e/ADU ASSUMED for all movies): ADU = offset_p + [Poisson(lambda) + N(0, rn_p^2)] / g, so for emitter-free
pixel-frames, per pixel:  mean m = offset + lambda / g,  variance v = (rn^2 + lambda) / g^2,  third central moment k3 = lambda / g^3.
  1. raw statistics: pooled percentiles / mode of all pixel values and of the emitter-free pixel-frames
  2. differential test against nr80 on the SAME sensor pixels (same camera, same ROI; needs equal gain and read noise, not their
     values):  offset_X - offset_nr80 = (m_X - m_80) - g (v_X - v_80)
  3. absolute test from the skewness (needs only g): lambda = k3 g^3, offset = m - k3 g^2, rn^2 = v g^2 - lambda
  4. histogram likelihood: pooled histogram of (value - pixel mean) of the free pixel-frames against the model, offset scanned
     (per pixel class: variance class x fractional part of the mean; lambda = (m - offset) g, rn^2 = v g^2 - lambda)
  5. what each offset implies: background photons / px / frame = (m - offset) g and read noise sqrt(v g^2 - bg)
Emitter-free = not within +-R px (square) of any quick-SMLM detection (all detections, not only the good ones) in frames
f - 2 .. f + 7 of its 6-frame window start f; R varied (contamination check). 'site-free' additionally excludes pixels within
R of ANY detection of the whole quick table (binding sites that are dim / undetected in other frames).
usage: python analysis/nanoruler/r24r_offset.py [f0 nframes]  -> out/nanoruler/nr40/reuse/offset_check.json / .txt"""
import json, os, sys, time
import numpy as np, tifffile, h5py
from scipy import ndimage as ndi
from scipy.special import ndtr, gammaln
sys.path.insert(0, '/home/jc/simcode2')
from simcode2.util import proctitle
proctitle.set_title('nr-r24r')
ROOT = '/home/jc/simcode2/out/nanoruler'; BIG = '/mnt/data/dmdsf/simcode2_out/nanoruler'
MOV = dict(nr80=(f'{ROOT}/raw/sim4_1_reord.tif', f'{ROOT}/quick_smlm_w6.hdf5'),
           nr40=(f'{BIG}/raw_nr40/movie_reord.tif', f'{ROOT}/insitu_nr40/quick_w6.hdf5'),
           nr20=(f'{BIG}/raw_nr20/movie_reord.tif', f'{ROOT}/insitu_nr20/quick_w6.hdf5'))
G = 0.45
F0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
TAG = '' if (F0, NF) == (0, 3000) else f'_f{F0}_{NF}'
RADII = (5, 8, 12)
REF = 103.0
DB = np.arange(-40.5, 80.5, 1.0)          # bins of value - pixel mean


def read(path):
    with tifffile.TiffFile(path) as t:
        return np.stack([t.pages[i].asarray() for i in range(F0, F0 + NF)])


def masks(L, shape, r):
    """busy [NF, H, W] (time mask) and site [H, W] (any detection of the whole table within r)"""
    nf, H, W = shape
    cx = np.clip(np.round(L['x']).astype(int), 0, W - 1); cy = np.clip(np.round(L['y']).astype(int), 0, H - 1); fr = L['frame'].astype(int) - F0
    site = np.zeros((H, W), bool); site[cy, cx] = True
    site = ndi.maximum_filter(site, size=2 * r + 1)
    busy = np.zeros(shape, bool)
    for df in range(-2, 8):
        f = fr + df; k = (f >= 0) & (f < nf)
        busy[f[k], cy[k], cx[k]] = True
    busy = ndi.maximum_filter1d(busy, 2 * r + 1, axis=1); busy = ndi.maximum_filter1d(busy, 2 * r + 1, axis=2)
    return busy, site


def moments(X, free):
    """per-pixel n, mean, variance, third central moment over the free frames (float64 power sums of X - REF)"""
    n = np.zeros(X.shape[1:]); s1 = np.zeros_like(n); s2 = np.zeros_like(n); s3 = np.zeros_like(n)
    for i in range(0, len(X), 250):
        d = (X[i:i + 250].astype(np.float64) - REF) * free[i:i + 250]
        n += free[i:i + 250].sum(0); s1 += d.sum(0); s2 += (d ** 2).sum(0); s3 += (d ** 3).sum(0)
    nn = np.maximum(n, 1)
    m1 = s1 / nn; m2 = s2 / nn - m1 ** 2; m3 = s3 / nn - 3 * m1 * s2 / nn + 2 * m1 ** 3
    return n, m1 + REF, m2 * nn / np.maximum(nn - 1, 1), m3


def dev_hist(X, free, m):
    h = np.zeros(len(DB) - 1)
    for i in range(0, len(X), 250):
        d = (X[i:i + 250].astype(np.float32) - m[None].astype(np.float32))[free[i:i + 250]]
        h += np.histogram(d, DB)[0]
    return h


def model_hist(m, v, w, off, nv=24, nfr=10):
    """expected pooled pmf of (integer value - pixel mean) for offset off: pixels grouped by variance (nv quantile classes) and
    fractional part of the mean (nfr classes), weights w (free samples per pixel); common lambda from the weighted mean level"""
    mm = np.average(m, weights=w); lam = max((mm - off) * G, 1e-3)
    vq = np.quantile(v, np.linspace(0, 1, nv + 1)); vc = np.clip(np.searchsorted(vq, v) - 1, 0, nv - 1)
    fr = m - np.floor(m); fc = np.minimum((fr * nfr).astype(int), nfr - 1)
    k = np.arange(0, 40); pk = np.exp(k * np.log(lam) - lam - gammaln(k + 1))
    pm = np.zeros(len(DB) - 1); ctr = 0.5 * (DB[1:] + DB[:-1])
    for a in range(nv):
        for b in range(nfr):
            s = (vc == a) & (fc == b); ws = w[s].sum()
            if ws == 0: continue
            vv = np.average(v[s], weights=w[s]) - 1.0 / 12; f = np.average(fr[s], weights=w[s])
            sd = np.sqrt(max(vv * G * G - lam, 0.05)) / G             # read-noise sd in ADU
            # integer values n = floor(m) + j; deviation d = j - f; value mean = m: electrons centred: (k - lam) / G
            j = np.arange(-45, 90)
            mu = f + (k[None] - lam) / G                               # position of the k-electron peak relative to floor(m)
            p = (pk[None] * (ndtr((j[:, None] + 0.5 - mu) / sd) - ndtr((j[:, None] - 0.5 - mu) / sd))).sum(1)
            pm += ws * np.histogram(j - f, DB, weights=p)[0]
    return pm / pm.sum(), lam


def main():
    t0 = time.time(); res = dict(gain=G, f0=F0, nframes=NF); PIX = {}
    L_ = [f'# camera offset check (r24r_offset.py): frames {F0}..{F0 + NF - 1} of the reordered caches, gain {G} e/ADU assumed; nr80 = control']
    for name, (mv, lf) in MOV.items():
        X = read(mv); H, W = X.shape[1:]
        Lq = h5py.File(lf)['locs'][:]
        r = dict(all=dict(mean=float(X.mean()), median=float(np.median(X[::10])), mode=int(np.bincount(X[::10].ravel()).argmax()),
                          **{f'p{p}': float(np.percentile(X[::10], p)) for p in (0.1, 1, 5, 25, 75)}),
                 frame_mean_sd=float(X.reshape(NF, -1).mean(1).std()))
        r['n_det_in_block'] = int(((Lq['frame'] >= F0) & (Lq['frame'] < F0 + NF)).sum())
        for R in RADII:
            busy, site = masks(Lq, X.shape, R)
            for kind in ('time', 'site'):
                free = ~busy if kind == 'time' else (~busy & ~site[None])
                n, m, v, k3 = moments(X, free)
                ok = n >= 0.5 * NF
                if ok.sum() < 500:
                    r[f'{kind}_R{R}'] = dict(frac_free=float(free.mean()), n_pix=int(ok.sum())); continue
                w = n[ok]
                d = dict(frac_free=float(free.mean()), n_pix=int(ok.sum()), mean=float(np.average(m[ok], weights=w)), var=float(np.average(v[ok], weights=w)),
                         var_median=float(np.median(v[ok])), k3=float(np.average(k3[ok], weights=w)), k3_median=float(np.median(k3[ok])),
                         k3_se=float(k3[ok].std() / np.sqrt(ok.sum())))
                # 3: skewness-based
                d['lam_k3'] = d['k3'] * G ** 3; d['offset_k3'] = d['mean'] - d['k3'] * G ** 2; d['rn_k3'] = float(np.sqrt(max((d['var'] - 1 / 12) * G * G - d['lam_k3'], 0)))
                d['offset_k3_medianpix'] = float(np.median(m[ok]) - d['k3_median'] * G ** 2)
                for off in (99.0, 100.0):
                    bg = (d['mean'] - off) * G
                    d[f'bg_off{off:.0f}'] = bg; d[f'rn_off{off:.0f}'] = float(np.sqrt(max((d['var'] - 1 / 12) * G * G - bg, 0)))
                    d[f'rn_medianpix_off{off:.0f}'] = float(np.sqrt(max((d['var_median'] - 1 / 12) * G * G - bg, 0))); d[f'k3_pred_off{off:.0f}'] = bg / G ** 3
                if R == 8:
                    h = dev_hist(X, free & ok[None], m)
                    offs = np.arange(96.0, 102.01, 0.25); ll = []
                    for off in offs:
                        pm, _ = model_hist(m[ok], v[ok], w, off)
                        ll.append(float((h * np.log(np.maximum(pm, 1e-12))).sum() / h.sum()))
                    ll = np.array(ll); i = int(ll.argmax())
                    if 0 < i < len(offs) - 1:     # parabola through the three points around the maximum
                        a_, b_, c_ = ll[i - 1], ll[i], ll[i + 1]; best = offs[i] + 0.25 * 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_)
                    else: best = offs[i]
                    emp = h / h.sum(); ent = float((emp[emp > 0] * np.log(emp[emp > 0])).sum())
                    d['hist'] = dict(offsets=offs.tolist(), loglik_per_sample=ll.tolist(), best_offset=float(best), kl_at_best=float(ent - ll.max()),
                                     kl_off99=float(ent - ll[list(offs).index(99.0)]), kl_off100=float(ent - ll[list(offs).index(100.0)]), n_samples=float(h.sum()))
                    # percentiles of value - pixel mean: data vs model
                    cs = np.cumsum(emp); ctr = DB[1:]
                    d['hist']['dev_pct_data'] = {str(p): float(np.interp(p / 100, cs, ctr)) for p in (0.1, 1, 5, 50, 95, 99, 99.9)}
                    for off in (99.0, 100.0):
                        pm, _ = model_hist(m[ok], v[ok], w, off); cm = np.cumsum(pm)
                        d['hist'][f'dev_pct_model_off{off:.0f}'] = {str(p): float(np.interp(p / 100, cm, ctr)) for p in (0.1, 1, 5, 50, 95, 99, 99.9)}
                    PIX[name, kind] = (ok, m, v)
                r[f'{kind}_R{R}'] = d
            del busy
        res[name] = r
        print(name, f'{time.time() - t0:.0f}s', json.dumps(r['all']), flush=True)
        del X
    # 2: differential against nr80 on common free pixels (R = 8)
    for name in ('nr40', 'nr20'):
        for kind in ('time', 'site'):
            if (name, kind) not in PIX or ('nr80', kind) not in PIX: continue
            ok = PIX[name, kind][0] & PIX['nr80', kind][0]
            if ok.sum() < 500: continue
            dm = PIX[name, kind][1][ok] - PIX['nr80', kind][1][ok]; dv = PIX[name, kind][2][ok] - PIX['nr80', kind][2][ok]
            rng = np.random.default_rng(0); bl = (np.arange(400)[:, None] // 40 * 10 + np.arange(400)[None] // 40)[ok]   # 100 spatial blocks for the CI
            est = lambda a, b: float(np.mean(a) - G * np.mean(b)); estm = lambda a, b: float(np.median(a) - G * np.median(b))
            bs, bsm = [], []
            ub = np.unique(bl); idx = [np.flatnonzero(bl == u) for u in ub]
            for _ in range(200):
                j = np.concatenate([idx[i] for i in rng.integers(0, len(ub), len(ub))])
                bs.append(est(dm[j], dv[j])); bsm.append(estm(dm[j], dv[j]))
            res[f'diff_{name}_{kind}'] = dict(n_pix=int(ok.sum()), d_mean_adu=float(dm.mean()), d_var_adu2=float(dv.mean()), d_var_median=float(np.median(dv)),
                                              d_offset=est(dm, dv), d_offset_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                              d_offset_median=estm(dm, dv), d_offset_median_ci=[float(np.percentile(bsm, 2.5)), float(np.percentile(bsm, 97.5))],
                                              corr_pixel_var=float(np.corrcoef(PIX[name, kind][2][ok], PIX['nr80', kind][2][ok])[0, 1]),
                                              corr_pixel_mean=float(np.corrcoef(PIX[name, kind][1][ok], PIX['nr80', kind][1][ok])[0, 1]))
    out = f'{ROOT}/nr40/reuse/offset_check{TAG}'
    json.dump(res, open(out + '.json', 'w'), indent=1)
    L_.append('\n## 1 raw pixel values (all pixel-frames, every 10th frame for the percentiles)')
    for n in MOV:
        a = res[n]['all']; L_.append(f'{n}: mean {a["mean"]:.3f}  median {a["median"]:.0f}  mode {a["mode"]}  p0.1 {a["p0.1"]:.0f}  p1 {a["p1"]:.0f}  p5 {a["p5"]:.0f}  p25 {a["p25"]:.0f}  p75 {a["p75"]:.0f}'
                                     f'  | sd of the frame means {res[n]["frame_mean_sd"]:.3f} ADU | detections in the block {res[n]["n_det_in_block"]}')
    L_.append('\n## 3 / 5 emitter-free pixel-frames (kind time = time mask only, site = also away from every detection of the whole table; R = mask half-width px)')
    L_.append('movie kind R | free fraction, pixels | mean ADU | var ADU^2 (mean / median pixel) | k3 ADU^3 (mean / median pixel) | skewness: offset, bg e, rn e | offset 100: bg, rn (mean / median pixel), k3 predicted | offset 99: bg, rn, k3 predicted')
    for n in MOV:
        for R in RADII:
            for kind in ('time', 'site'):
                d = res[n][f'{kind}_R{R}']
                if 'mean' not in d: L_.append(f'{n} {kind} R{R} | {d["frac_free"]:.3f}, {d["n_pix"]} px: too few pixels'); continue
                L_.append(f'{n} {kind} R{R} | {d["frac_free"]:.3f}, {d["n_pix"]} | {d["mean"]:.3f} | {d["var"]:.3f} / {d["var_median"]:.3f} | {d["k3"]:.2f} / {d["k3_median"]:.2f} | '
                          f'{d["offset_k3"]:.2f} (median px {d["offset_k3_medianpix"]:.2f}), {d["lam_k3"]:.3f}, {d["rn_k3"]:.3f} | {d["bg_off100"]:.3f}, {d["rn_off100"]:.3f} / {d["rn_medianpix_off100"]:.3f}, {d["k3_pred_off100"]:.2f} | '
                          f'{d["bg_off99"]:.3f}, {d["rn_off99"]:.3f} / {d["rn_medianpix_off99"]:.3f}, {d["k3_pred_off99"]:.2f}')
    L_.append('\n## 4 histogram of (value - pixel mean), free pixel-frames R = 8: best offset, KL divergence data || model (nats per sample) at the best offset, at 99 and at 100; percentiles of the deviation, data vs model')
    for n in MOV:
        for kind in ('time', 'site'):
            d = res[n][f'{kind}_R8'].get('hist')
            if not d: continue
            L_.append(f'{n} {kind}: best offset {d["best_offset"]:.2f}  KL best {d["kl_at_best"]:.2e}  KL(99) {d["kl_off99"]:.2e}  KL(100) {d["kl_off100"]:.2e}  ({d["n_samples"]:.3g} samples)')
            L_.append('     pct          ' + '  '.join(f'{p:>7s}' for p in d['dev_pct_data']))
            for k_ in ('dev_pct_data', 'dev_pct_model_off99', 'dev_pct_model_off100'):
                L_.append(f'     {k_[8:]:12s} ' + '  '.join(f'{v:+7.2f}' for v in d[k_].values()))
    L_.append('\n## 2 differential against nr80 on the same sensor pixels (R = 8): offset_X - offset_nr80 = d mean - g d var (95 % CI: bootstrap over 100 spatial blocks)')
    for k_, d in res.items():
        if k_.startswith('diff_'):
            L_.append(f'{k_[5:]}: {d["n_pix"]} px  d mean {d["d_mean_adu"]:+.3f} ADU  d var {d["d_var_adu2"]:+.3f} ADU^2 (median px {d["d_var_median"]:+.3f})  ->  d offset {d["d_offset"]:+.3f} [{d["d_offset_ci"][0]:+.3f}, {d["d_offset_ci"][1]:+.3f}] ADU'
                      f'  (medians: {d["d_offset_median"]:+.3f} [{d["d_offset_median_ci"][0]:+.3f}, {d["d_offset_median_ci"][1]:+.3f}])  | per-pixel corr with nr80: var {d["corr_pixel_var"]:.2f}, mean {d["corr_pixel_mean"]:.2f}')
    open(out + '.txt', 'w').write('\n'.join(L_) + '\n'); print('\n'.join(L_))


if __name__ == '__main__':
    main()
