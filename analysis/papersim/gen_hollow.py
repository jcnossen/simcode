"""Hollow-microtubule movies for the reviewers' question (are hollow microtubules resolvable at our photon counts?).

Ground truth: the seed-1 tubule layout of the density sims (common.preprint_tubules, smlmtorch 5a4fda9), every point moved onto
a hollow tube of radius R around its centreline as smlmtorch scripts/generate_tubules.py project_points_to_tube (038d312): a
random angle around the local tangent, offset R (cos a perp1 + sin a perp2); for in-plane tubules perp1 is in the image plane
and perp2 along z, so the 2D projection is an offset R cos a across the tubule (the arcsine profile: density peaks at +-R).
The z offset (+-R, <= 25 nm) is not rendered (2D sims, fixed emitter plane).
Experimental conditions (out/paper2/decode_real_stats.json: SIMCODE v2 pass-1 rows of the real movies, p >= 0.63, disjoint
windows; photons in the PSF model's units = the units of this simulator):
  --cond 1809_2   window photons median 817 (p5 392, p95 1695, log-sd 0.45); bg per 6-sum median 30.8 photons / px (p1-p99 25.9-36.3)
  --cond 2.5nM_2  window photons median 961 (log-sd 0.51); bg per 6-sum median 52.9 (p1-p99 33.7-79.7)
  Per binding event a brightness I6 (photons per fully-on 6-frame window) ~ lognormal(med0, sd0); (med0, sd0) are fitted so that
  the GT window photons of the windows with >= 300 photons (the real rows' p1 is ~300) have the measured median and log-sd.
  --phot_scale multiplies every event's brightness (photon sweep at the same background).
  Background: flat over the FOV, one level per 6-frame block drawn from the measured bg-per-6-sum distribution (quantile
  interpolation), divided by 6 per frame.
Kinetics: the preprint's Markov blinking (common.blink_events) with avg_on_time 20 frames (the real movies: linked disjoint windows
3.5 / 2.0 sums mean, a lower bound; the preprint sims' 6 frames are much shorter), density 1 emitter ON / um^2 / frame (real
1809/2: 8.2 detected emitters per 40 x 40 px per 6-sum; this movie ~ 1.3 x that), pitch 224 nm, m 0.9, PSF / z / camera as common.py.
usage: CUDA_VISIBLE_DEVICES=2 python analysis/papersim/gen_hollow.py --radius_nm 25 --phot_scale 1 [--cond 1809_2] [--nframes 30000]
outputs (out/papersim/hollow/<cond>_R<r>_x<s>/): movie.tif, gt.npz (pts = tube points, centre = centreline points, tub = tubule id,
events with ev_I6), calib_sim.json, camera_sim.npz, gt_pattern.pt, mp_gt.pickle, gen.json"""
import argparse, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STATS = os.path.join(C.REPO, 'out', 'paper2', 'decode_real_stats.json')
HOLLOW = os.path.join(C.OUT, 'hollow')


def tube_ids(pts):
    """tubule index of the stacked centreline points (the generator stacks 20 tubules; a jump > 0.5 px starts a new one)"""
    d = np.hypot(*np.diff(pts[:, :2], axis=0).T)
    return np.concatenate([[0], np.cumsum(d > 0.5)])


def project_to_tube(pts, tub, R, rng):
    """= generate_tubules.project_points_to_tube (038d312) per tubule (tangents by central differences inside each tubule)"""
    out = pts.copy(); ang = rng.uniform(0, 2 * np.pi, len(pts))
    for t in np.unique(tub):
        i = np.nonzero(tub == t)[0]; c = pts[i]
        tg = np.zeros_like(c); tg[1:-1] = c[2:] - c[:-2]; tg[0] = c[1] - c[0]; tg[-1] = c[-1] - c[-2]
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
        ref = np.where(np.abs(tg[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        p1 = ref - (ref * tg).sum(1, keepdims=True) * tg; p1 /= np.linalg.norm(p1, axis=1, keepdims=True)
        p2 = np.cross(tg, p1)
        out[i] = c + R * (np.cos(ang[i])[:, None] * p1 + np.sin(ang[i])[:, None] * p2)
    return out


def quantile_sampler(q):
    """inverse-CDF sampler from the stored quantiles {p: value} (linear between, clamped at p1 / p99)"""
    p = np.array(sorted(float(k) for k in q)); v = np.array([q[k] for k in sorted(q, key=float)])
    return lambda u: np.interp(u, p, v)


def window_photons(t0, t1, I6, nframes):
    """GT photons of every (event, disjoint window) with on-time > 0: I6 x on-frames / 6 (the illumination averages out
    over the 3 steps of each angle)"""
    w0 = np.floor(t0 / 6).astype(int); w1 = np.ceil(t1 / 6).astype(int); n = w1 - w0
    ev = np.repeat(np.arange(len(t0)), n); w = np.repeat(w0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    on = np.clip(np.minimum(t1[ev], 6 * w + 6) - np.maximum(t0[ev], 6 * w), 0, 6)
    ok = (on > 0) & (w < nframes // 6)
    return I6[ev[ok]] * on[ok] / 6


def fit_brightness(t0, t1, nframes, med_t, lsd_t, rng, thr=300.0):
    z = rng.standard_normal(len(t0))
    best = None
    for sd in np.linspace(0.15, 0.6, 19):
        lo, hi = 50.0, 5000.0
        for _ in range(40):
            m = np.sqrt(lo * hi)
            ph = window_photons(t0, t1, m * np.exp(sd * z), nframes); ph = ph[ph >= thr]
            if np.median(ph) < med_t: lo = m
            else: hi = m
        ph = window_photons(t0, t1, m * np.exp(sd * z), nframes); ph = ph[ph >= thr]
        err = abs(np.std(np.log(ph)) - lsd_t)
        if best is None or err < best[0]: best = (err, m, sd, float(np.median(ph)), float(np.std(np.log(ph))))
    return dict(med0=best[1], sd0=best[2], fit_median=best[3], fit_logsd=best[4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--radius_nm', type=float, required=True)
    ap.add_argument('--phot_scale', type=float, default=1.0)
    ap.add_argument('--cond', default='1809_2', choices=['1809_2', '2.5nM_2'])
    ap.add_argument('--nframes', type=int, default=30000)
    ap.add_argument('--density', type=float, default=1.0)
    ap.add_argument('--avg_on_time', type=float, default=20.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--block', type=int, default=600)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from simcode2.util.proctitle import auto; auto('psim-hollow')
    import torch
    d = a.out or os.path.join(HOLLOW, f'{a.cond}_R{a.radius_nm:g}_x{a.phot_scale:g}')
    os.makedirs(d, exist_ok=True)
    t00 = time.time()
    S = json.load(open(STATS))[a.cond]
    W = 40
    centre, knots = C.preprint_tubules()
    tub = tube_ids(centre)
    rng = np.random.default_rng([a.seed, int(a.radius_nm * 10)])          # tube angles: same for every photon level
    pts = project_to_tube(centre, tub, a.radius_nm / C.PX_NM, rng)
    # kinetics + brightness: identical events for every radius and photon level (same seed), brightness scaled
    rng = np.random.default_rng([a.seed, 77])
    area = (W * C.PX_NM / 1000) ** 2
    ids, t0e, t1e = C.blink_events(len(pts), a.nframes, a.avg_on_time, a.density * area / len(pts), rng)
    bf = fit_brightness(t0e, t1e, a.nframes, S['photons_sum6']['0.5'], S['log_photons_sd'], np.random.default_rng([a.seed, 78]))
    I6 = bf['med0'] * np.exp(bf['sd0'] * np.random.default_rng([a.seed, 78]).standard_normal(len(ids))) * a.phot_scale
    bgq = quantile_sampler(S['bg_sum6_per_px'])
    bg6 = bgq(np.random.default_rng([a.seed, 79]).uniform(0.01, 0.99, a.nframes // 6 + 1)) / 6     # per frame, per 6-frame block
    P = C.gt_pattern([0, 90])
    fa = np.floor(t0e).astype(np.int64); fb = np.ceil(t1e).astype(np.int64); n = fb - fa
    ev = np.repeat(np.arange(len(ids)), n)
    fr = np.repeat(fa, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    onf = np.clip(np.minimum(t1e[ev], fr + 1) - np.maximum(t0e[ev], fr), 0, 1)
    ok = (onf > 0) & (fr < a.nframes); ev, fr, onf = ev[ok], fr[ok], onf[ok]
    x = pts[ids[ev], 0]; y = pts[ids[ev], 1]
    ph = I6[ev] / 6 * C.illum(P, x, y, fr) * onf
    o = np.argsort(fr, kind='stable'); fr, x, y, ph = fr[o], x[o], y[o], ph[o]
    R = C.Renderer(W, W)
    gen = torch.Generator(device='cuda'); gen.manual_seed(a.seed * 7919 + int(a.radius_nm * 10) * 31 + int(a.phot_scale * 100))
    bounds = np.searchsorted(fr, np.arange(0, a.nframes + a.block, a.block))
    bgt = torch.as_tensor(np.repeat(bg6, 6)[:a.nframes], device='cuda', dtype=torch.float32)

    def blocks():
        for i, f0 in enumerate(range(0, a.nframes, a.block)):
            nf = min(a.block, a.nframes - f0); s, e = bounds[i], bounds[i + 1]
            mu = R.expected(f0, nf, fr[s:e], x[s:e], y[s:e], ph[s:e]) + bgt[f0:f0 + nf, None, None]
            yield R.adu(mu, gen)
    C.write_movie(os.path.join(d, 'movie.tif'), blocks())
    wp = window_photons(t0e, t1e, I6, a.nframes)
    np.savez_compressed(os.path.join(d, 'gt.npz'), pts=pts, centre=centre, tub=tub, knots=knots, ev_id=ids, ev_x=pts[ids, 0], ev_y=pts[ids, 1],
                        ev_t0=t0e, ev_t1=t1e, ev_I6=I6, bg6=bg6, k=P['k'], phi=P['phi'], m=P['m'], rel=P['rel'], z_um=R.z,
                        nframes=a.nframes, W=W, radius_px=a.radius_nm / C.PX_NM)
    calib = C.write_sim_calib(d, W, W, P)
    C.save_pattern_files(P, a.nframes, d)
    info = dict(kind='hollow', cond=a.cond, radius_nm=a.radius_nm, phot_scale=a.phot_scale, density=a.density, nframes=a.nframes, W=W,
                px_nm=C.PX_NM, avg_on_time=a.avg_on_time, n_gt_points=int(len(pts)), n_events=int(len(ids)),
                brightness_fit=bf, window_photons_ge300=dict(zip(['p5', 'p50', 'p95'], np.percentile(wp[wp >= 300], [5, 50, 95]).round(1).tolist())),
                window_photons_all=dict(zip(['p5', 'p50', 'p95'], np.percentile(wp, [5, 50, 95]).round(1).tolist())),
                bg_per_frame=dict(mean=float(bg6.mean()), p1=float(np.percentile(bg6, 1)), p99=float(np.percentile(bg6, 99))),
                real_stats=dict(photons_sum6=S['photons_sum6'], log_photons_sd=S['log_photons_sd'], bg_sum6_per_px=S['bg_sum6_per_px']),
                mean_on_per_frame=float(onf.sum() / a.nframes), z_um=R.z, psf=C.PSF_CALIB, calib=calib, seed=a.seed,
                gt_points_md5=hashlib.md5(np.ascontiguousarray(pts).tobytes()).hexdigest(), s=round(time.time() - t00, 1))
    json.dump(info, open(os.path.join(d, 'gen.json'), 'w'), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
