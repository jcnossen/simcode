"""Referee checks, real data (Fig 3 = 1809_2, Fig 4 = 2.5nM_2), on the cached figure rows (cache_real.py; the FRC set = the
fair protocol of analysis/paper/frc_fair.py, each method's own DME v2 drift of the figure).
(e) FRC uncertainty: paired block bootstrap (the same block draws for every method; blocks of 6 frames = the windows, and of 300
    frames), B resamples, FRC with 300-frame and 30-frame halves; random assignment of the 300-frame blocks to the halves (R
    repeats); FRC vs half-block length; temporal correlation of the localizations (P(another row within r at window f + lag),
    simcode.evaluation.metrics.revisit) = how long one binding event lasts vs the half-block length.
(f) FRC at a matched number of localizations: every method subsampled to SIMFLUX's FRC-set count, by random whole windows and
    by random rows (5 repeats each), 300-frame halves; plus the n best rows by the method's own reported precision.
(h) RSP (paper_common.compute_pc = pearson_corr.compute_correlation: render zoom 20, Gaussian blur 1.3 camera px, block average to
    the camera grid, 2 px border, Pearson with the widefield mean image of the raw movie) variants: as the figure (drift-corrected
    render); the uncorrected positions (the same drift as the widefield mean); drift-corrected + the mean drift; the figure's
    optional integer-pixel alignment; blur scaled by the method's resolution (sqrt(1.3^2 - (FRC / 2.355 / 108.33 nm)^2) px); the
    best sub-pixel shift (+-0.5 px grid, 0.1 px) x blur (0.8-2.0 px); matched count (random windows to SIMFLUX's count).
usage: CUDA_VISIBLE_DEVICES=0 python analysis/refcheck/real_checks.py <1809_2|2.5nM_2> [--B 200] [--parts e,f,h]
output: out/refcheck/real_<ds>.json"""
import os, sys, json, argparse, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'analysis', 'paper')]
import frc_gpu as FG

CACHE = os.path.join(REPO, 'out', 'refcheck', 'cache'); OUTD = os.path.join(REPO, 'out', 'refcheck')
KINDS = [('smlm', 'SMLM'), ('sf', 'SIMFLUX'), ('decode', 'DECODE'), ('v2', 'SIMCODE')]
MEAN = {'1809_2': f'{REPO}/out/paper2/1809_2/movie/simflux_5.5ms_500mW.tif_mean.npy',
        '2.5nM_2': f'{REPO}/out/paper2/2.5nM_2/movie/cover1_600mW_5.5ms_cf660.tif_mean.npy'}
NF = {'1809_2': 120000, '2.5nM_2': 180000}


def ci(v):
    v = np.asarray(v, float)
    return dict(median=float(np.nanmedian(v)), lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)),
                sd=float(np.nanstd(v)), nan=int(np.isnan(v).sum()), n=int(len(v)))


def load(ds, names):
    T = {}
    for k in names:
        Z = np.load(os.path.join(CACHE, f'{ds}_{k}.npz')); m = Z['filt']
        T[k] = dict(x=Z['xdc'][m], y=Z['ydc'][m], f=Z['f'][m], xr=Z['x'][m], yr=Z['y'][m], sx=Z['sx'][m], sy=Z['sy'][m],
                    ph=Z['photons'][m])
    return T


# ------------------------------------------------------------------------------------------------ (e)
def part_e(T, fr, nf, B, R, rng, kinds):
    out = dict(point={}, boot={}, split={}, blk_curve={}, revisit={})
    blks = (6, 12, 30, 60, 150, 300, 600, 1500, 3000, 6000)
    for k in kinds:
        t = T[k]
        out['point'][k] = dict(h300=fr[k].halves(t['x'], t['y'], t['f'], 300), h30=fr[k].halves(t['x'], t['y'], t['f'], 30), n=int(len(t['x'])))
        out['blk_curve'][k] = {str(b): fr[k].halves(t['x'], t['y'], t['f'], b) for b in blks}
    for blk in (6, 300):
        nb = int(np.ceil(nf / blk)); V = {k: dict(h300=[], h30=[]) for k in kinds}; t0 = time.time()
        for it in range(B):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            for k in kinds:
                t = T[k]; wr = w[t['f'] // blk]
                V[k]['h300'].append(fr[k].halves(t['x'], t['y'], t['f'], 300, w=wr))
                V[k]['h30'].append(fr[k].halves(t['x'], t['y'], t['f'], 30, w=wr))
        o = {k: {h: ci(V[k][h]) for h in V[k]} for k in kinds}
        for h in ('h300', 'h30'):
            others = np.array([V[k][h] for k in kinds if k != 'v2']); sc = np.array(V['v2'][h])
            best = np.nanmin(others, 0); sf = np.array(V['sf'][h])
            o[f'margin_{h}'] = dict(sf_minus_simcode_nm=ci(sf - sc), next_best_minus_simcode_nm=ci(best - sc),
                                    p_simcode_best=float(np.mean(sc < best)), p_simcode_lt_sf=float(np.mean(sc < sf)),
                                    p_tie_sf=float(np.mean(np.abs(sc - sf) < 1e-9)))
        out['boot'][f'blk{blk}'] = o
        print(f'boot blk {blk}: {time.time() - t0:.0f} s', json.dumps({k: (o[k]['h300']['lo'], o[k]['h300']['hi']) for k in kinds}), flush=True)
    nb = int(np.ceil(nf / 300)); V = {k: [] for k in kinds}
    for it in range(R):
        assign = rng.permutation(np.r_[np.ones(nb // 2, bool), np.zeros(nb - nb // 2, bool)])
        for k in kinds:
            t = T[k]; V[k].append(fr[k].halves(t['x'], t['y'], t['f'], 300, assign=assign))
    out['split'] = {k: ci(V[k]) for k in kinds}
    from simcode.evaluation.metrics import revisit
    lags = (6, 12, 18, 30, 60, 120, 300, 600, 3000, 30000)
    for k in kinds:
        t = T[k]
        out['revisit'][k] = {str(l): v for l, v in revisit(t['x'], t['y'], t['f'], lags=lags, r=0.25, nq=40000).items()}
    return out


def jk(v, n, d, point):
    """delete-d jackknife over blocks: SE = sqrt((n - d) / d * var(theta_j)); 95 % CI = point +- 1.96 SE"""
    v = np.asarray(v, float); se = float(np.sqrt((n - d) / d * np.nanvar(v)))
    return dict(point=float(point), se=se, lo=float(point - 1.96 * se), hi=float(point + 1.96 * se), mean_sub=float(np.nanmean(v)), J=int(len(v)))


def part_j(T, fr, nf, J, rng, kinds, keep=0.9):
    """paired delete-d jackknife (10 % of the blocks dropped without replacement, the same blocks for every method)"""
    out = {}
    pt = {k: dict(h300=fr[k].halves(T[k]['x'], T[k]['y'], T[k]['f'], 300), h30=fr[k].halves(T[k]['x'], T[k]['y'], T[k]['f'], 30)) for k in kinds}
    for blk in (6, 300):
        nb = int(np.ceil(nf / blk)); d = nb - int(round(keep * nb)); V = {k: dict(h300=[], h30=[]) for k in kinds}; t0 = time.time()
        for it in range(J):
            kb = np.ones(nb, bool); kb[rng.choice(nb, d, replace=False)] = False
            for k in kinds:
                t = T[k]; m = kb[t['f'] // blk]
                V[k]['h300'].append(fr[k].halves(t['x'][m], t['y'][m], t['f'][m], 300))
                V[k]['h30'].append(fr[k].halves(t['x'][m], t['y'][m], t['f'][m], 30))
        o = {k: {h: jk(V[k][h], nb, d, pt[k][h]) for h in V[k]} for k in kinds}
        for h in ('h300', 'h30'):
            others = [k for k in kinds if k != 'v2']
            best = np.nanmin(np.array([V[k][h] for k in others]), 0); sc = np.array(V['v2'][h]); sf = np.array(V['sf'][h])
            o[f'margin_{h}'] = dict(sf_minus_simcode_nm=jk(sf - sc, nb, d, pt['sf'][h] - pt['v2'][h]),
                                    next_best_minus_simcode_nm=jk(best - sc, nb, d, min(pt[k][h] for k in others) - pt['v2'][h]),
                                    frac_subsamples_simcode_lt_sf=float(np.mean(sc < sf - 1e-9)), frac_subsamples_tie_sf=float(np.mean(np.abs(sc - sf) < 1e-9)))
        out[f'blk{blk}'] = o
        print(f'jack blk {blk}: {time.time() - t0:.0f} s', json.dumps({k: (round(o[k]['h300']['lo'], 2), round(o[k]['h300']['hi'], 2)) for k in kinds}), flush=True)
    return out


# ------------------------------------------------------------------------------------------------ (f)
def window_subsample(f, n_target, rng):
    u, inv, cnt = np.unique(f, return_inverse=True, return_counts=True)
    p = rng.permutation(len(u)); c = np.cumsum(cnt[p]); j = int(np.searchsorted(c, n_target)) + 1
    keep = np.zeros(len(u), bool); keep[p[:j]] = True
    return keep[inv]


def part_f(T, fr, rng, kinds, reps=5):
    n_t = len(T['sf']['x']); out = dict(n_target=int(n_t), methods={})
    for k in kinds:
        t = T[k]; n = len(t['x']); o = dict(n=int(n))
        if n <= n_t:
            o['note'] = 'already at or below the target'; o['windows'] = o['rows'] = ci([fr[k].halves(t['x'], t['y'], t['f'])])
        else:
            vw, vr, nw = [], [], []
            for r in range(reps):
                m = window_subsample(t['f'], n_t, rng); nw.append(int(m.sum())); vw.append(fr[k].halves(t['x'][m], t['y'][m], t['f'][m]))
                m = np.zeros(n, bool); m[rng.choice(n, n_t, replace=False)] = True; vr.append(fr[k].halves(t['x'][m], t['y'][m], t['f'][m]))
            o['windows'] = dict(ci(vw), values=vw, n=nw, frac_windows=None); o['rows'] = dict(ci(vr), values=vr)
            s = np.maximum(t['sx'], t['sy']); m = np.zeros(n, bool); m[np.argsort(s, kind='stable')[:n_t]] = True
            o['best_sigma'] = dict(frc_nm=fr[k].halves(t['x'][m], t['y'][m], t['f'][m]), sigma_max_nm=float(s[m].max() * 108.33))
        out['methods'][k] = o
        print('f', k, json.dumps({kk: (v['median'] if isinstance(v, dict) and 'median' in v else v) for kk, v in o.items() if kk != 'note'}), flush=True)
    return out


# ------------------------------------------------------------------------------------------------ (h)
def render(x, y, zoom=20, shape=(120, 120)):
    """= smlmtorch Dataset.render(zoom=20, sigma=1) up to the 1-zoomed-pixel Gaussian of each spot: histogram on the zoomed grid
    (Draw puts a spot at pos * zoom with pixel centres on the integers -> histogram bin = round(pos * zoom))"""
    H = np.zeros((shape[0] * zoom, shape[1] * zoom))
    ix = np.round(x * zoom).astype(int); iy = np.round(y * zoom).astype(int)
    ok = (ix >= 0) & (ix < H.shape[1]) & (iy >= 0) & (iy < H.shape[0])
    np.add.at(H, (iy[ok], ix[ok]), 1.0)
    return H


def _norm_d(d_img):
    d = d_img - d_img.min(); return d / (d.max() + 1e-15)


def blurred(sr, zoom, sigma=1.3, extra_sigma=1.0):
    from pearson_corr import fft_gaussian_blur
    return fft_gaussian_blur(sr / (sr.max() + 1e-15), np.sqrt((sigma * zoom) ** 2 + extra_sigma ** 2))


def score(b, d, border=2, shift=(0, 0)):
    from pearson_corr import block_average_downsample, pearson_correlation
    if shift != (0, 0): b = np.roll(b, shift, axis=(0, 1))
    r = block_average_downsample(b, d.shape)
    return float(pearson_correlation(r[border:-border, border:-border], d[border:-border, border:-border]))


def rsp(sr, d_img, sigma=1.3, border=2, shift=(0, 0), extra_sigma=1.0):
    """pearson_corr.compute_correlation(sr, d, sigma, border, align=False), with an optional shift (zoomed px) of the blurred
    render and the per-spot sigma of the render (1 zoomed px) folded into the blur"""
    zoom = sr.shape[0] // d_img.shape[0]
    return score(blurred(sr, zoom, sigma, extra_sigma), _norm_d(d_img), border, shift)


def part_h(ds, T, drift, frc_nm, rng, kinds):
    sys.path.append(os.path.expanduser('~/smlmtorch/scripts'))
    from pearson_corr import compute_correlation
    d_img = np.load(MEAN[ds]); out = dict(methods={})
    n_t = len(T['sf']['x'])
    for k in kinds:
        t = T[k]; o = {}
        H = render(t['x'], t['y'])
        o['figure_like'] = rsp(H, d_img)
        o['figure_like_align'] = float(compute_correlation(H, d_img, sigma=1.3, align=True)[0])
        o['uncorrected_positions'] = rsp(render(t['xr'], t['yr']), d_img)
        dm = drift[k].mean(0)
        o['mean_drift_added'] = rsp(render(t['x'] + dm[0], t['y'] + dm[1]), d_img); o['mean_drift_px'] = dm.tolist()
        sl = frc_nm[k] / 2.355 / 108.33; o['res_scaled_sigma_px'] = float(np.sqrt(1.3 ** 2 - sl ** 2))
        o['res_scaled_blur'] = rsp(H, d_img, sigma=o['res_scaled_sigma_px'])
        best = (-1, None, None); grid = {}; dn = _norm_d(d_img)
        for sg in (0.8, 1.0, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 4.0):
            b = blurred(H, 20, sg)
            for dy in range(-10, 11, 2):
                for dx in range(-10, 11, 2):
                    v = score(b, dn, shift=(dy, dx))
                    if v > best[0]: best = (v, sg, (dy / 20, dx / 20))
            grid[str(sg)] = score(b, dn)
        o['blur_scan_noshift'] = grid
        o['best_shift_blur'] = dict(pearson=best[0], sigma_px=best[1], shift_px_yx=best[2])
        if len(t['x']) > n_t:
            v = []
            for r in range(3):
                m = window_subsample(t['f'], n_t, rng); v.append(rsp(render(t['x'][m], t['y'][m]), d_img))
            o['matched_count_windows'] = ci(v)
        out['methods'][k] = o
        print('h', k, json.dumps({kk: vv for kk, vv in o.items() if kk != 'blur_scan_noshift'}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('ds', choices=list(NF))
    ap.add_argument('--B', type=int, default=200); ap.add_argument('--R', type=int, default=50)
    ap.add_argument('--parts', default='e,f,h'); ap.add_argument('--kinds', default='smlm,sf,decode,v2')
    ap.add_argument('--tag', default=''); ap.add_argument('--J', type=int, default=100)
    a = ap.parse_args(); rng = np.random.default_rng(20261115)
    kinds = a.kinds.split(','); T = load(a.ds, kinds)
    chk = json.load(open(os.path.join(CACHE, f'{a.ds}_check.json')))
    fr = {k: FG.FRC(T[k]['x'], T[k]['y']) for k in kinds}
    for k in kinds:
        v = FG.check(T[k]['x'], T[k]['y'], T[k]['f'], fr[k]); assert abs(v - chk[k]['frc_nm']) < 1e-9, (k, v, chk[k])
    fn = os.path.join(OUTD, f'real_{a.ds}{a.tag}.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    res.update(dataset=a.ds, nframes=NF[a.ds], kinds=kinds, n_frc={k: int(len(T[k]['x'])) for k in kinds},
               frc_nm={k: chk[k]['frc_nm'] for k in kinds}, protocol=__doc__)
    parts = a.parts.split(',')
    if 'e' in parts:
        res['e'] = part_e(T, fr, NF[a.ds], a.B, a.R, rng, kinds); json.dump(res, open(fn, 'w'), indent=1, default=float)
    if 'j' in parts:
        res['j'] = part_j(T, fr, NF[a.ds], a.J, rng, kinds); json.dump(res, open(fn, 'w'), indent=1, default=float)
    if 'f' in parts:
        res['f'] = part_f(T, fr, rng, kinds); json.dump(res, open(fn, 'w'), indent=1, default=float)
    if 'h' in parts:
        drift = {k: np.load(chk[k]['drift']) for k in kinds}
        res['h'] = part_h(a.ds, T, drift, res['frc_nm'], rng, kinds); json.dump(res, open(fn, 'w'), indent=1, default=float)
    print('written', fn)


if __name__ == '__main__':
    main()
