"""S3 (SI S8) movie generator: moving microtubules (Brownian motion + spring constraints), the preprint's simulation
(smlmtorch scripts/fig_moving_tubules.py, unchanged since 93c3e2a; read here from commit 5f7fdcb) rendered with the simcode2
simulator and the averaged 1809/2 PSF.

Ported verbatim from the old script (compiled from its git source, common.load_functions): generate_microtubule_points
(moving variant, return_list), update_tubule, center_tubules; the frame loop is simulate_brownian_microtubules' loop (same
order of RNG draws: np.random.seed(0), the tubule generation, then per frame center_tubules, record, per tubule
np.random.normal(0, 1, (N, 3)) with z zeroed and update_tubule) without keeping all 65000 frames in memory.
Kept: seed 0, W = 50 px, 15 tubules, linedensity 20 / px, spl_degree 3, margin 0.01, 65000 steps, dt 0.01, k_spring 40,
sigma_interior 0.2, sigma_endpoints 0.15, the first 1000 frames skipped (-> 64000 frames), on_fraction 0.01, avg_on_time 6,
intensity 500 per 6 frames, bg 4 photons / px / frame, pitch 224 nm, depth 0.9, angles 1 / 91 deg.
Changes: PSF / z / camera / step order as common.py; pixel size 108.33 nm instead of the old script's 100 nm (the PSF is
calibrated at 108.33 nm; the motion and the GT are in px, so the structure is 8 % larger in nm); update_tubule compiled with
numba parallel=False (the old prange loops race on neighbouring points, so the parallel version is not reproducible).
usage: CUDA_VISIBLE_DEVICES=0 python analysis/papersim/gen_moving.py [--nframes 64000] [--out out/papersim/moving]
outputs: movie.tif, gt_rows.npz (frame, spot, x, y, photons of every ON emitter-frame), traj.npy (all points every
traj_step frames, [n, N, 2] px float32, for the block Jaccard), calib_sim.json, camera_sim.npz, gt_pattern.pt, mp_gt.pickle,
gen.json"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def ported():
    import numba
    nj = lambda *a, **k: numba.njit(**{**k, 'parallel': False})
    from scipy.interpolate import InterpolatedUnivariateSpline
    return C.load_functions(C.MOVING_COMMIT, 'scripts/fig_moving_tubules.py',
                            ['generate_microtubule_points', 'update_tubule', 'center_tubules'],
                            dict(njit=nj, jit=nj, prange=range, InterpolatedUnivariateSpline=InterpolatedUnivariateSpline))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nframes', type=int, default=64000, help='frames after the 1000 skipped ones (old: 65000 - 1000)')
    ap.add_argument('--skip', type=int, default=1000)
    ap.add_argument('--bg', type=float, default=4.0)
    ap.add_argument('--on_fraction', type=float, default=0.01)
    ap.add_argument('--traj_step', type=int, default=10)
    ap.add_argument('--block', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--cond', default='preprint', choices=['preprint', '1809_2'],
                    help="'preprint' (I 500 / 6 frames, bg 4, on-time 6) or '1809_2' (experimental photons / bg as gen_hollow.py, on-time 20)")
    ap.add_argument('--avg_on_time', type=float, default=None)
    ap.add_argument('--psf', default='vectorial', help="'vectorial' or e.g. 'gauss1.3' (the preprint's PSF; calib_sim.json still names the vectorial PSF)")
    a = ap.parse_args()
    from simcode2.util.proctitle import auto; auto('psim-genmov')
    import torch
    d = a.out or C.movie_dir('moving'); os.makedirs(d, exist_ok=True)
    t00 = time.time()
    F = ported()
    W = 50
    np.random.seed(a.seed)
    tubules = F['generate_microtubule_points'](width=W, depth=0, linedensity=20, numtubules=15, spl_knots=4, spl_degree=3,
                                               return_knots=False, return_list=True, margin=0.01)
    rest = [np.sqrt(np.sum((t[1:] - t[:-1]) ** 2, axis=1)) for t in tubules]
    sig_i = 0.2 * np.sqrt(2 * 0.01); sig_e = 0.15 * np.sqrt(2 * 0.01)
    Npts = sum(len(t) for t in tubules)
    rng = np.random.default_rng([a.seed, 64])
    ont = a.avg_on_time or (6.0 if a.cond == 'preprint' else 20.0)
    ids, t0e, t1e = C.blink_events(Npts, a.nframes, ont, a.on_fraction, rng)            # on_fraction = the ON density (same in both conditions)
    if a.cond == 'preprint':
        I6ev = np.full(len(ids), C.I6); bgf = np.full(a.nframes, a.bg); bfit = None
    else:
        import gen_hollow as GH
        St = json.load(open(GH.STATS))[a.cond]
        bfit = GH.fit_brightness(t0e, t1e, a.nframes, St['photons_sum6']['0.5'], St['log_photons_sd'], np.random.default_rng([a.seed, 78]))
        I6ev = bfit['med0'] * np.exp(bfit['sd0'] * np.random.default_rng([a.seed, 78]).standard_normal(len(ids)))
        bgf = np.repeat(GH.quantile_sampler(St['bg_sum6_per_px'])(np.random.default_rng([a.seed, 79]).uniform(0.01, 0.99, a.nframes // 6 + 1)) / 6, 6)[:a.nframes]
    fa = np.floor(t0e).astype(np.int64); fb = np.ceil(t1e).astype(np.int64); n = fb - fa
    ev = np.repeat(np.arange(len(ids)), n)
    fr = np.repeat(fa, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    onf = np.clip(np.minimum(t1e[ev], fr + 1) - np.maximum(t0e[ev], fr), 0, 1)
    ok = (onf > 0) & (fr < a.nframes); ev, fr, onf = ev[ok], fr[ok], onf[ok]
    o = np.argsort(fr, kind='stable'); ev, fr, onf = ev[o], fr[o], onf[o]
    spot = ids[ev]; I6r = I6ev[ev]
    fb_ = np.searchsorted(fr, np.arange(a.nframes + 1))
    P = C.gt_pattern([1, 91])
    R = C.Renderer(W, W, psf=a.psf)
    gen = torch.Generator(device='cuda'); gen.manual_seed(a.seed * 7919 + 64)
    gx = np.zeros(len(fr), np.float32); gy = np.zeros(len(fr), np.float32)
    ntraj = a.nframes // a.traj_step
    traj = np.lib.format.open_memmap(os.path.join(d, 'traj.npy'), mode='w+', dtype=np.float32, shape=(ntraj, Npts, 2))
    total = a.nframes + a.skip

    def blocks():
        buf_f, buf_x, buf_y, buf_p = [], [], [], []
        f_blk0 = 0
        for step in range(total):
            F['center_tubules'](tubules, W)
            fi = step - a.skip
            if fi >= 0:
                allp = np.concatenate(tubules)
                if fi % a.traj_step == 0 and fi // a.traj_step < ntraj: traj[fi // a.traj_step] = allp[:, :2]
                s, e = fb_[fi], fb_[fi + 1]
                x = allp[spot[s:e], 0]; y = allp[spot[s:e], 1]
                gx[s:e] = x; gy[s:e] = y
                buf_f.append(fr[s:e]); buf_x.append(x); buf_y.append(y)
                buf_p.append(I6r[s:e] / 6 * C.illum(P, x, y, fr[s:e]) * onf[s:e])
                if fi - f_blk0 + 1 == a.block or fi == a.nframes - 1:
                    nf = fi - f_blk0 + 1
                    mu = R.expected(f_blk0, nf, np.concatenate(buf_f), np.concatenate(buf_x), np.concatenate(buf_y), np.concatenate(buf_p)) \
                         + torch.as_tensor(bgf[f_blk0:f_blk0 + nf], device='cuda', dtype=torch.float32)[:, None, None]
                    yield R.adu(mu, gen)
                    buf_f, buf_x, buf_y, buf_p = [], [], [], []; f_blk0 = fi + 1
            for i, tub in enumerate(tubules):          # simulate_brownian_microtubules: update after recording the frame
                rnd = np.random.normal(0.0, 1, size=(tub.shape[0], 3)); rnd[:, 2] = 0.0
                F['update_tubule'](tub, rest[i], rnd, 0.01, 40.0, 0, sig_i, sig_e)
    C.write_movie(os.path.join(d, 'movie.tif'), blocks())
    traj.flush(); del traj
    ph = I6r / 6 * C.illum(P, gx, gy, fr) * onf
    np.savez_compressed(os.path.join(d, 'gt_rows.npz'), frame=fr.astype(np.int32), spot=spot.astype(np.int32), x=gx, y=gy, photons=ph.astype(np.float32),
                        ev_id=ids, ev_t0=t0e, ev_t1=t1e, ev_I6=I6ev, k=P['k'], phi=P['phi'], m=P['m'], rel=P['rel'])
    calib = C.write_sim_calib(d, W, W, P, psf=a.psf)
    C.save_pattern_files(P, a.nframes, d)
    info = dict(kind='moving', cond=a.cond, render_psf=a.psf, avg_on_time_used=ont, brightness_fit=bfit, bg_mean=float(bgf.mean()), nframes=a.nframes, skip=a.skip, W=W, px_nm=C.PX_NM, bg=a.bg, intensity_per_6frames=C.I6, avg_on_time=6.0,
                on_fraction=a.on_fraction, n_points=int(Npts), n_tubules=len(tubules), n_events=int(len(ids)),
                mean_on_per_frame=float(onf.sum() / a.nframes), traj_step=a.traj_step, z_um=R.z,
                frac_rows_outside_fov=float(np.mean((gx < 0) | (gx >= W) | (gy < 0) | (gy >= W))),
                pattern=dict(angles_deg=P['angles_deg'], pitch_nm=C.PITCH_NM, depth=C.DEPTH, order='[[0,1,2],[3,4,5]]'),
                camera=dict(gain=C.GAIN, offset=C.OFFSET, read_noise=0), psf=C.PSF_CALIB, calib=calib, seed=a.seed,
                source=f'smlmtorch {C.MOVING_COMMIT} scripts/fig_moving_tubules.py', s=round(time.time() - t00, 1))
    json.dump(info, open(os.path.join(d, 'gen.json'), 'w'), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
