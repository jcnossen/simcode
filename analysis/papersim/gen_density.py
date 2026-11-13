"""S1 / S2 (main Fig 2 + SI S1 / S2) movie generator: the preprint's random-microtubule ground truth (smlmtorch 5a4fda9, seed 1;
common.preprint_tubules) at one density, rendered with the simcode simulator and the averaged 1809/2 in-situ PSF.

Preprint settings kept (5a4fda9 scripts/fig_densities2.py): 40 x 40 px of 108.33 nm, 60000 frames, density in emitters ON per
um^2 per frame (on_fraction = density x area / number of GT points), avg_on_time 6 frames (10 sub-steps), intensity 500 photons
per 6 frames, background 2.5 photons / px / frame, pitch 224 nm, depth 0.9, angles 0 / 90 deg. Changes: PSF, emitter z fixed
at the calib's emitter plane, pattern step order [[0,1,2],[3,4,5]], camera written as ADU (gain 1, offset 100; Poisson only).
See common.py.

usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/gen_density.py --density 1 [--nframes 60000] [--out out/papersim/D1]
outputs (<out>): movie.tif (uint16 ADU), gt.npz (pts = GT point set [N,3] px, knots, events: id / x / y / t0 / t1 frames,
pattern), calib_sim.json + camera_sim.npz, gt_pattern.pt, mp_gt.pickle, gen.json"""
import argparse, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--density', type=float, required=True)
    ap.add_argument('--nframes', type=int, default=60000)
    ap.add_argument('--bg', type=float, default=2.5)
    ap.add_argument('--intensity', type=float, default=C.I6)
    ap.add_argument('--avg_on_time', type=float, default=6.0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=None)
    ap.add_argument('--block', type=int, default=600)
    ap.add_argument('--psf', default='vectorial', help="'vectorial' (default) or e.g. 'gauss1.3' (control: the preprint's PSF; calib_sim.json still names the vectorial PSF)")
    a = ap.parse_args()
    from simcode.util.proctitle import auto; auto('psim-gen')
    import torch
    d = a.out or C.movie_dir(f'D{a.density:g}')
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    W = 40
    pts, knots = C.preprint_tubules()
    area = (W * C.PX_NM / 1000) ** 2
    on_fraction = a.density * area / len(pts)
    rng = np.random.default_rng([a.seed, int(round(a.density * 1000))])
    ids, t0e, t1e = C.blink_events(len(pts), a.nframes, a.avg_on_time, on_fraction, rng)
    P = C.gt_pattern([0, 90])
    # rows (event, frame) with on-fraction > 0
    fa = np.floor(t0e).astype(np.int64); fb = np.ceil(t1e).astype(np.int64)            # frames fa .. fb - 1
    n = fb - fa
    ev = np.repeat(np.arange(len(ids)), n)
    fr = np.repeat(fa, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    onf = np.clip(np.minimum(t1e[ev], fr + 1) - np.maximum(t0e[ev], fr), 0, 1)
    ok = (onf > 0) & (fr < a.nframes)
    ev, fr, onf = ev[ok], fr[ok], onf[ok]
    x = pts[ids[ev], 0]; y = pts[ids[ev], 1]
    ph = a.intensity / 6 * C.illum(P, x, y, fr) * onf
    o = np.argsort(fr, kind='stable'); ev, fr, x, y, ph = ev[o], fr[o], x[o], y[o], ph[o]
    R = C.Renderer(W, W, psf=a.psf)
    gen = torch.Generator(device='cuda'); gen.manual_seed(a.seed * 7919 + int(round(a.density * 1000)))
    bounds = np.searchsorted(fr, np.arange(0, a.nframes + a.block, a.block))

    def blocks():
        for i, f0 in enumerate(range(0, a.nframes, a.block)):
            nf = min(a.block, a.nframes - f0); s, e = bounds[i], bounds[i + 1]
            mu = R.expected(f0, nf, fr[s:e], x[s:e], y[s:e], ph[s:e]) + a.bg
            yield R.adu(mu, gen)
    C.write_movie(os.path.join(d, 'movie.tif'), blocks())
    np.savez_compressed(os.path.join(d, 'gt.npz'), pts=pts, knots=knots, ev_id=ids, ev_x=pts[ids, 0], ev_y=pts[ids, 1], ev_t0=t0e, ev_t1=t1e,
                        k=P['k'], phi=P['phi'], m=P['m'], rel=P['rel'], z_um=R.z, nframes=a.nframes, W=W)
    calib = C.write_sim_calib(d, W, W, P, psf=a.psf)
    C.save_pattern_files(P, a.nframes, d)
    info = dict(kind='density', density=a.density, nframes=a.nframes, W=W, px_nm=C.PX_NM, bg=a.bg, intensity_per_6frames=a.intensity,
                avg_on_time=a.avg_on_time, on_fraction=on_fraction, n_gt_points=int(len(pts)),
                gt_points_md5=hashlib.md5(np.ascontiguousarray(pts).tobytes()).hexdigest(), n_events=int(len(ids)),
                mean_on_per_frame=float(onf.sum() / a.nframes), expected_on_per_frame=float(a.density * area), z_um=R.z,
                pattern=dict(angles_deg=P['angles_deg'], pitch_nm=C.PITCH_NM, depth=C.DEPTH, order='[[0,1,2],[3,4,5]]'),
                camera=dict(gain=C.GAIN, offset=C.OFFSET, read_noise=0), psf=C.PSF_CALIB if a.psf == 'vectorial' else a.psf, calib=calib, seed=a.seed,
                gt_source=f'smlmtorch {C.GT_COMMIT} scripts/generate_tubules.py + fig_densities2.py (seed 1)', s=round(time.time() - t0, 1))
    json.dump(info, open(os.path.join(d, 'gen.json'), 'w'), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
