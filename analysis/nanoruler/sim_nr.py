"""Simulated DNA-PAINT NANORULER movie with ground truth, in the format of the cached (reordered) real nanoruler movie.
Adapted from analysis/simreal/sim_movie_r.py (same output file names and sim_gt.npz keys) for the original SIMFLUX setup:
65 nm pixels, in-situ vectorial PSF of out/nanoruler/calib_nr80.json, camera gain / offset / per-pixel maps of that
calibration, pattern block of the calibration (pitch about 220 nm, 2 orientations x 3 phase steps).

usage: CUDA_VISIBLE_DEVICES=2 python analysis/nanoruler/sim_nr.py <out_dir> [n_frames=12000] [seed=1] [options]
       (every option --name can also be given as environment variable NR_SIM_NAME; the command line wins)

TIME BASES (see sim_nr_lib.py)
  ACQUISITION frame t: pattern index t % 6, orientation (t % 6) % 2, phase step (t % 6) // 2 (orientations interleaved, as on
      the real microscope). Binding kinetics (on / off times), per-frame brightness fluctuation, pattern phase drift, sample
      drift, background fluctuation (AR(1) processes) and camera noise are all generated in acquisition time and order.
  STORED frame f: the movie is WRITTEN with every 6-frame cycle in the order PERM = [0, 2, 4, 1, 3, 5]
      (stored frame 6 c + j = acquisition frame 6 c + PERM[j]), exactly as analysis/nanoruler/reorder.py caches the real movie.
      In stored order the pattern follows the calib convention: step s = f % 6, angle a = s // 3,
          I = A relint_s (1 + depth_s cos(k_a . r(f) - phase_s - dphi_a(f))),   r(f) = r + drift(f)
      Only stored windows starting at a multiple of 6 are 6 consecutive acquisition frames.

GEOMETRY  nanorulers: 3 (default 60 %) or 2 binding sites on a line, 80 nm apart, random position and in-plane orientation,
  all sites >= --min_dist nm from the sites of other rulers and >= --margin px inside the FOV; z uniform in [0, --zmax_nm].
  Every site binds repeatedly and independently (alternating renewal process: exponential dark time; on-time --on_model 'mix' =
  two exponentials (fraction --on_flong with mean --on_tlong, the rest with mean --on_tshort) or 'lognormal' (--on_median /
  --on_mean); one imager per site at a time, sites of one ruler can be on simultaneously). The number of rulers and
  the binding rate follow from --dens (ON emitters per px per frame) and --ev_per_site_30k (binding events per site per 30000
  frames); the per-site rate is then adjusted so that the expected ON density is exactly --dens for the drawn rulers.
DEFAULTS  tuned (2026-11-13) so that the quick Gaussian SMLM pass (quick_smlm.py --sum 6) and the mean frame level of the sim
  match the real movie, see out/nanoruler/sim/logs/check_nr_a.log. They differ from the nominal figures of the real movie
  (ON density 8.5e-5, on-time mean 23 / median 11 frames, window photons 1400, bg 1.65, 125 events per site per 30000 frames):
  the nominal movie (--dens 8.5e-5 --ev_per_site_30k 125 --on_model lognormal --win_med 1400 --bright_sd 0.36 --bg 1.65
  --bg_fluct -1) gave Gaussian-fit photons 850 instead of 1158 (the Gaussian fit on 11 x 11 px returns about 56 % of the full
  integral of this PSF), a mean level 0.45 ADU too high, Gaussian-fit bg 10.85 instead of 9.81, 1.6 x too much temporal noise
  of 15 x 15 px block sums (AcquisitionSim fluct 0.15 at this low bg), 4 x too few rulers with 3.5 x too high duty each, and too few
  long events (detection pair survival at 96 frames 0.17 instead of 0.38).
BRIGHTNESS  per event lognormal (--win_med = median photons of a fully-on 6-frame window at mean illumination, --bright_sd);
  variation as in AcquisitionSim: per-event angle factor exp(ang_sd N) (normalized to mean 1 over the two angles), per-event step
  factor 1 + step_sd N (stored step index), per-frame factor 1 + frame_sd N (clamped at 0.2). --no_var sets all three to 0;
  the random draws are separate streams, so the same seed gives the same rulers / events / background / pattern with or
  without variation (camera-noise realizations differ where the expected image differs).
PSF  AcquisitionSim LUT bank (calib PSF + --npsf - 1 weakly jittered LUTs, one LUT per SITE) + weak halo, as in sim_movie_r.py.
  The PSF image is rendered per 60-frame block at the block-centre drift position (drift within a block << 0.1 nm); the
  illumination uses the per-frame position.
CAMERA  Poisson + per-pixel read noise (crop of camera_maps_nr80.npz), ADU = round(e / gain_true + offset map + residual),
  gain_true = gain (1 + gain_err N), residual offset error off_err ADU per pixel (static); uint16 TIFF.
BACKGROUND  preset level (photons / px / frame), weak static structure, per-angle factor, temporal fluctuation (global AR(1)
  + smooth field AR(1), AcquisitionSim 'fluct' model) evolving in acquisition order.

OUTPUTS (all under <out_dir>)
  sim_movie.tif   [NF, H, W] uint16 ADU, STORED frame order
  sim_gt.npz      keys of sim_movie_r.py (site variant):
      x, y [N] px     event position (site + linker jitter), drift-free, pixel centres at integers; N = binding events
      z [N] um, bright [N] (expected photons per frame at relint 1 / mean illumination, fully on), dim [N] (0)
      t0, t1 [N]      on / off time in ACQUISITION frames (continuous; t0 may be < 0, t1 > NF). NOTE: consumers that take the
                      overlap of [t0, t1] with a stored frame index (evalproto.lib.gt_windows_fast) are exact for windows in
                      which the event is fully on and approximate (within-cycle permutation ignored) at the event edges;
                      sim_nr_lib.gt_windows_exact gives the exact stored on-fractions
      ea [N, 2], es [N, 6]   angle / step factors, STORED step convention (angle = s // 3)
      patt [NF, 5]    kx, ky, phase, depth, relint per STORED frame (phase = phase_s + dphi_a at that frame's acquisition time)
      ph6 [N, NF/6]   expected photons per cycle c (stored frames 6 c .. 6 c + 5 = acquisition frames 6 c .. 6 c + 5), all
                      factors incl. the per-frame fluctuation and on-fractions
      site [N], site_x, site_y, site_z, site_a, site_b, site_w [n_site], grp [N]  (site_a / site_b = -1e9 / 1e9, site_w = 1:
                      sites are active for the whole movie with equal rates; kept for format compatibility)
      gain_true, frame_sd, step_sd, nf, preset, knobs (json of all arguments);
      halo2, halo2_sig, filstat, struct = 0 and med0 / sd = window median / 6 and log-sd (not meaningful otherwise)
    additional keys:
      ang_sd, shape (H, W), pixel_nm, crop (x0, y0 of the camera-map crop), psf_sum (mean PSF sum on the ROI)
      fr_ev, fr_frame, fr_acq, fr_onf, fr_ph   sparse per-frame table of every (event, frame) with on-fraction > 0:
                      event index, STORED frame, ACQUISITION frame, on-fraction, expected photons (all factors)
      drift [NF, 2] px per STORED frame, drift_acq [NF, 2] px per ACQUISITION frame (sample drift, added to x, y)
      dphi_acq [2, NF] rad pattern phase drift per angle in ACQUISITION time
      bg_frame [NF] mean expected background (photons / px) per STORED frame, bg_map [H, W] static background (float32)
  gt_pattern.pt   common pattern format (k, phi, m, logrel, dphi [2, NF] per STORED frame, knot_frames 1, angle_of_step)
  sites.npz       per site: site, ruler, idx (position in the ruler), x, y (px), x_nm, y_nm, z_um, ax, ay (ruler axis unit
                  vector), n_events, n_full6 (fully-on cycle-aligned windows); per ruler: ruler_cx, ruler_cy (px),
                  ruler_angle (rad), ruler_nsites; pixel_nm
  calib_sim.json + camera_maps_sim.npz   the calibration with the camera maps cropped to the simulated FOV
  sim_info.json   arguments and summary statistics"""
import argparse, json, math, os, sys
sys.path.insert(0, '/home/jc/simcode2'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, tifffile
from simcode.util.proctitle import set_title
from sim_nr_lib import PERM, INV, acq_of_stored

ND = '/home/jc/simcode2/out/nanoruler/'
dev = 'cuda'


def args_():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    E = lambda n, d, t=float: t(os.environ.get('NR_SIM_' + n.upper(), d))
    ap.add_argument('out'); ap.add_argument('n_frames', nargs='?', type=int, default=12000); ap.add_argument('seed', nargs='?', type=int, default=1)
    def A(n, d, t=float, h=''):
        ap.add_argument('--' + n, type=t, default=E(n, d, t), help=h)
    A('calib', ND + ('calib_nr80.json' if os.path.exists(ND + 'calib_nr80.json') else 'calib_nr80_draft.json'), str)
    A('preset_json', os.environ.get('NR_PRESET', ND + 'nr80_preset.json'), str, 'AcquisitionSim preset patch (as nrwrap.py)')
    A('H', 256, int); A('W', 256, int); A('crop_x0', -1, int, 'camera-map crop origin (-1: centred)'); A('crop_y0', -1, int)
    A('dens', 9.5e-5, h='ON emitters per px per frame'); A('ev_per_site_30k', 40.0, h='binding events per site per 30000 frames')
    A('n_rulers', 0, int, '0: from dens'); A('frac3', 0.6, h='fraction of three-site rulers'); A('site_nm', 80.0)
    A('min_dist', 400.0, h='nm between sites of different rulers'); A('margin', 8.0, h='px, sites from the border')
    A('site_jit', 0.0, h='nm, per-event position jitter per axis'); A('zmax_nm', 20.0)
    A('on_model', 'mix', str, "'mix': two exponentials (on_flong, on_tshort, on_tlong); 'lognormal': on_median, on_mean")
    A('on_flong', 0.2, h='mix: fraction of long events'); A('on_tshort', 6.0, h='mix: mean on-time of the short events (frames)'); A('on_tlong', 160.0)
    A('on_median', 11.0, h='lognormal model'); A('on_mean', 23.0, h='lognormal model'); A('on_max', 3000.0)
    A('win_med', 2150.0, h='median photons of a fully-on 6-frame window (full PSF integral)'); A('bright_sd', 0.30, h='log-sd')
    A('frame_sd', 0.10); A('step_sd', -1.0, h='-1: preset'); A('ang_sd', -1.0, h='-1: preset')
    ap.add_argument('--no_var', action='store_true', default=bool(int(os.environ.get('NR_SIM_NO_VAR', '0'))), help='frame_sd = step_sd = ang_sd = 0')
    A('bg', 1.45, h='photons / px / frame (-1: preset)'); A('bg_sd', -1.0); A('bg_hp', -1.0); A('bg_fluct', 0.05, h='AcquisitionSim fluct scale (-1: preset)')
    A('drift_nm', 2.0, h='sample drift rms per axis over the movie (0: off)'); A('pdrift_nm', 1.5, h='pattern phase drift rms per angle (0: off)')
    A('npsf', 4, int, '1: exact calib PSF for all sites'); A('halo', -1.0, h='-1: preset')
    A('gain_err', 0.01); A('off_err', 0.3, h='ADU'); A('rn_mult', 1.0)
    return ap.parse_args()


def smooth_curves(rng, n, T, rms):
    """n smooth curves over t = 0 .. T - 1: linear trend + 3 slow sinusoids (0.5-3 periods over the movie), zero at t = 0,
    rms about the mean = rms"""
    t = np.arange(T) / max(T, 1); out = np.zeros((n, T))
    if rms <= 0: return out
    for i in range(n):
        c = rng.randn() * t
        for j in range(3):
            c = c + rng.randn() / (j + 1) * np.sin(2 * np.pi * (rng.uniform(0.5, 3.0) * t + rng.rand()))
        c = c - c[0]; out[i] = c * rms / max(np.std(c), 1e-12)
    return out


def place_rulers(rng, a, n_target):
    H, W = a.H, a.W; px = a.pixel_nm; d = a.site_nm / px; md = a.min_dist / px
    S = []; R = []
    for it in range(200000):
        if len(R) >= n_target: break
        ns = 3 if rng.rand() < a.frac3 else 2
        th = rng.uniform(0, math.pi); c = np.array([rng.uniform(0, W - 1), rng.uniform(0, H - 1)])
        u = np.array([math.cos(th), math.sin(th)])
        pts = c[None] + (np.arange(ns) - (ns - 1) / 2)[:, None] * d * u[None]
        if (pts[:, 0] < a.margin).any() or (pts[:, 0] > W - 1 - a.margin).any() or (pts[:, 1] < a.margin).any() or (pts[:, 1] > H - 1 - a.margin).any(): continue
        if S and np.min(np.hypot(*(np.concatenate([s[0] for s in S])[:, None] - pts[None]).transpose(2, 0, 1))) < md: continue
        S.append((pts, len(R))); R.append((c, th, ns))
    assert len(R) == n_target, 'could not place the rulers (density too high for min_dist)'
    return S, R


def sfield(n, H, W, sig, g):
    """unit-sd smooth random fields [n, H, W] (as acquisition.smooth_field, with an explicit generator)"""
    k = int(3 * sig); o = torch.arange(-k, k + 1, device=dev, dtype=torch.float32)
    q = torch.exp(-0.5 * (o / sig) ** 2); q = q / q.sum()
    x = torch.randn(n, 1, H + 2 * k, W + 2 * k, device=dev, generator=g)
    x = torch.nn.functional.conv2d(x, q.view(1, 1, 1, -1)); x = torch.nn.functional.conv2d(x, q.view(1, 1, -1, 1))[:, 0]
    return (x - x.mean((1, 2), keepdim=True)) / x.std((1, 2), keepdim=True).clamp(min=1e-6)


def main():
    a = args_(); set_title('nr-sim')
    out, NF, seed = a.out, a.n_frames, a.seed
    assert NF % 6 == 0
    assert os.path.abspath(out).startswith('/home/jc/simcode2/out/nanoruler/sim'), 'outputs only under out/nanoruler/sim/'
    os.makedirs(out, exist_ok=True)
    import simcode.simulate.acquisition as RS
    NR = json.load(open(a.preset_json))                      # in-memory preset patch, as analysis/nanoruler/nrwrap.py
    for name, p_ in RS.PRESETS.items():
        p_.update({k: (tuple(v) if isinstance(v, list) and k != 'k' else v) for k, v in {**NR.get('common', {}), **NR.get(name, {})}.items()})
    PRE = '0.5nM'
    C = json.load(open(a.calib)); a.pixel_nm = 1000 * C['pixelsize_um']
    H, W = a.H, a.W
    rs = RS.AcquisitionSim(PRE, H, W, 6, npsf=a.npsf, seed=seed, calib=a.calib)
    p = rs.p; sim = rs.sim; R = sim.R
    assert not isinstance(rs.luts[0], tuple), 'per-angle PSF calib not supported here'
    for key in ('step_sd', 'ang_sd', 'bg', 'bg_sd', 'bg_hp', 'halo'):
        if getattr(a, key) < 0: setattr(a, key, float(p[key]))
    if a.bg_fluct < 0: a.bg_fluct = float(p['fluct'])
    if a.no_var: a.frame_sd = a.step_sd = a.ang_sd = 0.0
    print('arguments', vars(a), flush=True)
    rng = np.random.RandomState(seed)                        # geometry, kinetics, brightness, drift (CPU)
    gv = torch.Generator(device=dev).manual_seed(1000 + seed)   # emitter variation
    gb = torch.Generator(device=dev).manual_seed(2000 + seed)   # background
    gc = torch.Generator(device=dev).manual_seed(3000 + seed)   # camera
    gp = torch.Generator(device=dev).manual_seed(4000 + seed)   # PSF group / halo

    # ---------------------------------------------------------------- rulers, sites
    rate = a.ev_per_site_30k / 30000.0                       # events per site per frame
    assert a.on_model in ('mix', 'lognormal')
    on_mean = a.on_flong * a.on_tlong + (1 - a.on_flong) * a.on_tshort if a.on_model == 'mix' else a.on_mean
    duty = on_mean * rate
    nsite_t = a.dens * H * W / duty
    n_rul = a.n_rulers if a.n_rulers > 0 else max(1, int(round(nsite_t / (3 * a.frac3 + 2 * (1 - a.frac3)))))
    S, RU = place_rulers(rng, a, n_rul)
    sxy = np.concatenate([s[0] for s in S]); srul = np.concatenate([np.full(len(s[0]), s[1]) for s in S]); sidx = np.concatenate([np.arange(len(s[0])) for s in S])
    M = len(sxy)
    sz = rng.uniform(0, a.zmax_nm * 1e-3, M)
    duty = a.dens * H * W / M; rate = duty / on_mean       # adjusted so that E[ON density] = dens for the drawn sites
    mean_off = 1 / rate - on_mean
    assert mean_off > 0
    print(f'{n_rul} rulers, {M} sites; per-site duty {duty:.4f}, events / site / 30000 frames {rate * 30000:.1f}, mean dark time {mean_off:.1f} frames', flush=True)
    # ---------------------------------------------------------------- binding events (acquisition time)
    ls = math.sqrt(2 * math.log(a.on_mean / a.on_median)); lm = math.log(a.on_median)
    def on_time():
        if a.on_model == 'mix': return rng.exponential(a.on_tlong if rng.rand() < a.on_flong else a.on_tshort)
        return float(np.exp(lm + ls * rng.randn()))
    ev_s, ev_t0, ev_t1 = [], [], []
    for s in range(M):
        t = -20 * (on_mean + mean_off) - rng.exponential(mean_off)
        while True:
            t0 = t + rng.exponential(mean_off); t1 = t0 + min(on_time(), a.on_max)
            if t0 >= NF: break
            if t1 > 0: ev_s.append(s); ev_t0.append(t0); ev_t1.append(t1)
            t = t1
    site = np.array(ev_s); t0n = np.array(ev_t0); t1n = np.array(ev_t1); N = len(site)
    jit = a.site_jit / a.pixel_nm
    xn = sxy[site, 0] + jit * rng.randn(N); yn = sxy[site, 1] + jit * rng.randn(N)
    brn = a.win_med / 6.0 * np.exp(a.bright_sd * rng.randn(N))
    drift_acq = smooth_curves(rng, 2, NF, a.drift_nm / a.pixel_nm).T                       # [NF, 2] px, acquisition time
    k = np.array(C['pattern']['k_rad_per_px'], np.float64)                                 # [2, 2] rows = angle
    phi = np.array(C['pattern']['phase']); m = np.array(C['pattern']['depth']); rel = np.array(C['pattern']['relint'])
    pitch_px = 2 * np.pi / np.hypot(k[:, 0], k[:, 1])
    dphi_acq = smooth_curves(rng, 2, NF, 1.0) * (2 * np.pi * (a.pdrift_nm / a.pixel_nm) / pitch_px)[:, None] if a.pdrift_nm > 0 else np.zeros((2, NF))

    T_ = lambda v, dt=torch.float32: torch.tensor(np.asarray(v), device=dev, dtype=dt)
    x, y, z, br, t0, t1 = T_(xn), T_(yn), T_(sz[site]), T_(brn), T_(t0n), T_(t1n)
    sitet = T_(site, torch.long)
    ea = torch.exp(a.ang_sd * torch.randn(N, 2, device=dev, generator=gv)); ea = ea / ea.mean(1, keepdim=True)
    es = 1 + a.step_sd * torch.randn(N, 6, device=dev, generator=gv)
    sgrp = torch.randint(0, len(rs.luts), (M,), device=dev, generator=gp); grp = sgrp[sitet]
    hh = a.halo * torch.rand(N, 1, 1, device=dev, generator=gp)
    Dacq = T_(drift_acq); DPH = T_(dphi_acq); Kt = T_(k); PHI = T_(phi); Mt = T_(m); RELt = T_(rel)
    INVt = T_(INV, torch.long)

    def mkpsf(ix_, xs, ys):
        P_ = torch.zeros(len(ix_), R, R, device=dev); base_ = sim.lut
        for gi, lut in enumerate(rs.luts):
            m_ = grp[ix_] == gi
            if m_.any():
                sim.lut = lut; P_[m_] = sim.render_psf(xs[m_], ys[m_], z[ix_][m_])
        sim.lut = base_
        o_ = torch.arange(R, device=dev, dtype=torch.float32) - R // 2
        fx_, fy_ = xs - torch.round(xs), ys - torch.round(ys)
        gb_ = torch.exp(-0.5 * (((o_[None, None, :] - fx_[:, None, None]) ** 2 + (o_[None, :, None] - fy_[:, None, None]) ** 2) / 16.0))
        gb_ = gb_ / gb_.sum((1, 2), keepdim=True) * P_.sum((1, 2), keepdim=True)
        return (1 - hh[ix_]) * P_ + hh[ix_] * gb_

    # ---------------------------------------------------------------- camera (crop of the calibration maps)
    cam = dict(np.load(p['cam']))
    FH, FW = cam['readnoise_e'].shape
    x0 = (FW - W) // 2 if a.crop_x0 < 0 else a.crop_x0; y0 = (FH - H) // 2 if a.crop_y0 < 0 else a.crop_y0
    assert x0 + W <= FW and y0 + H <= FH
    crop = {k_: (v[y0:y0 + H, x0:x0 + W] if getattr(v, 'shape', ()) == (FH, FW) else v) for k_, v in cam.items()}
    np.savez_compressed(f'{out}/camera_maps_sim.npz', **crop)
    C2 = json.loads(json.dumps(C)); C2['camera']['maps'] = os.path.abspath(f'{out}/camera_maps_sim.npz')
    C2['source'] = f'sim_nr.py: {a.calib} with the camera maps cropped to x0 {x0}, y0 {y0}, {W} x {H}'
    json.dump(C2, open(f'{out}/calib_sim.json', 'w'), indent=1)
    rn = T_(crop['readnoise_e']) * a.rn_mult
    gain_true = sim.gain * (1 + a.gain_err * torch.randn(1, device=dev, generator=gc).item())
    off_true = T_(crop['offset_adu']) + a.off_err * torch.randn(H, W, device=dev, generator=gc)
    # ---------------------------------------------------------------- background (static part)
    st = (a.bg + a.bg_sd * sfield(1, H, W, 8.0, gb)[0] + a.bg_hp * sfield(1, H, W, 2.5, gb)[0]).clamp(min=0.3)
    bg_ang = T_(p['bg_ang'])
    gl = torch.randn(1, device=dev, generator=gb); F = sfield(1, H, W, 4.3, gb)[0]; a1, a2 = 0.4, 0.5

    hr = R // 2; PD = hr + 2
    oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
    ADU = np.empty((NF, H, W), np.uint16)
    PH = torch.zeros(N, NF // 6, device=dev)
    FR = []; bg_frame = np.zeros(NF); psf_sum = []
    BL = 60
    for b0 in range(0, NF, BL):
        ta = torch.arange(b0, min(b0 + BL, NF), device=dev); Tb = len(ta); fr = ta.float()
        i6 = ta % 6; ang = i6 % 2; s = INVt[i6]                 # stored step = 3 * orientation + phase step
        fs = 6 * (ta // 6) + s                                  # stored frame index of every acquisition frame
        act = torch.nonzero((t1 > b0) & (t0 < b0 + Tb))[:, 0]
        img = torch.zeros(Tb, H + 2 * PD, W + 2 * PD, device=dev)
        if len(act):
            onf = (torch.minimum(t1[act, None], fr[None] + 1) - torch.maximum(t0[act, None], fr[None])).clamp(0, 1)
            xe = x[act, None] + Dacq[ta, 0][None]; ye = y[act, None] + Dacq[ta, 1][None]
            phs = PHI[s] + DPH[ang, ta]
            il = RELt[s][None] * (1 + Mt[s][None] * torch.cos(Kt[ang, 0][None] * xe + Kt[ang, 1][None] * ye - phs[None]))
            mult = (ea[act][:, ang] * es[act][:, s] * (1 + a.frame_sd * torch.randn(len(act), Tb, device=dev, generator=gv))).clamp(min=0.2)
            ph = br[act, None] * il * onf * mult
            PH.index_put_((act[:, None].expand(-1, Tb), (ta // 6)[None].expand(len(act), -1)), ph, accumulate=True)
            ii, jj = torch.nonzero(onf > 0, as_tuple=True)
            FR.append(torch.stack([act[ii].double(), fs[jj].double(), ta[jj].double(), onf[ii, jj].double(), ph[ii, jj].double()], 1).cpu())
            dm = Dacq[b0 + Tb // 2]
            xr, yr = x[act] + dm[0], y[act] + dm[1]
            pa = mkpsf(act, xr, yr); psf_sum.append(pa.sum((1, 2)).cpu())
            cx = torch.round(xr).long() + PD; cy = torch.round(yr).long() + PD
            iy = (cy[:, None, None] - hr + oy).reshape(len(act), -1); ix = (cx[:, None, None] - hr + ox).reshape(len(act), -1)
            Hp, Wp = img.shape[-2], img.shape[-1]
            lin = torch.arange(Tb, device=dev)[None, :, None] * Hp * Wp + (iy * Wp + ix)[:, None, :]
            img.view(-1).index_add_(0, lin.reshape(-1), (ph[:, :, None] * pa.reshape(len(act), 1, -1)).reshape(-1))
        # background of the block, acquisition order
        af = bg_ang[ang] * (1 + 0.01 * torch.randn(Tb, device=dev, generator=gb))
        nz = torch.randn(Tb, device=dev, generator=gb); Fn = sfield(Tb, H, W, 4.3, gb)
        bt = torch.empty(Tb, H, W, device=dev)
        for j in range(Tb):
            gl = a1 * gl + math.sqrt(1 - a1 ** 2) * nz[j]; F = a2 * F + math.sqrt(1 - a2 ** 2) * Fn[j]
            bt[j] = st * af[j] * (1 + 0.027 * a.bg_fluct * gl) + 1.8 * a.bg_fluct * F
        bt = bt.clamp(min=0.3)
        mu = img[:, PD:PD + H, PD:PD + W] + bt
        e = torch.poisson(mu, generator=gc) + torch.randn(mu.shape, device=dev, generator=gc) * rn
        fsn = fs.cpu().numpy()
        ADU[fsn] = torch.round(e / gain_true + off_true).clamp(0, 65535).cpu().numpy().astype(np.uint16)   # WRITTEN in stored order
        bg_frame[fsn] = bt.mean((1, 2)).cpu().numpy()
    tifffile.imwrite(f'{out}/sim_movie.tif', ADU, bigtiff=ADU.nbytes > 3.9e9)

    # ---------------------------------------------------------------- ground truth
    fst = np.arange(NF); tacq = acq_of_stored(fst); sst = fst % 6; ast = sst // 3
    patt = np.stack([k[ast, 0], k[ast, 1], phi[sst] + dphi_acq[ast, tacq], m[sst], rel[sst]], 1).astype(np.float32)
    FRn = torch.cat(FR).numpy() if FR else np.zeros((0, 5))
    ph6 = PH.cpu().numpy()
    cyc = np.arange(NF // 6)
    full6 = (t0n[:, None] <= 6 * cyc[None]) & (t1n[:, None] >= 6 * cyc[None] + 6)                 # [N, NF/6] fully on in cycle c
    psf_sum = float(torch.cat(psf_sum).mean()) if psf_sum else float('nan')
    knobs = json.dumps({k_: v for k_, v in vars(a).items()})
    np.savez_compressed(
        f'{out}/sim_gt.npz', dim=np.zeros(N, np.float32), knobs=knobs, x=xn.astype(np.float32), y=yn.astype(np.float32), z=sz[site].astype(np.float32),
        bright=brn.astype(np.float32), t0=t0n.astype(np.float32), t1=t1n.astype(np.float32), ea=ea.cpu().numpy(), es=es.cpu().numpy(), patt=patt,
        ph6=ph6, gain_true=gain_true, halo2=0.0, halo2_sig=0.0, frame_sd=a.frame_sd, step_sd=a.step_sd, filstat=0.0, struct=0.0, nf=NF, preset=PRE,
        med0=a.win_med / 6.0, sd=a.bright_sd,
        site=site.astype(np.int64), site_x=sxy[:, 0].astype(np.float32), site_y=sxy[:, 1].astype(np.float32), site_z=sz.astype(np.float32),
        site_a=np.full(M, -1e9, np.float32), site_b=np.full(M, 1e9, np.float32), site_w=np.ones(M, np.float32), grp=grp.cpu().numpy(),
        ang_sd=a.ang_sd, shape=np.array([H, W]), pixel_nm=a.pixel_nm, crop=np.array([x0, y0]), psf_sum=psf_sum,
        fr_ev=FRn[:, 0].astype(np.int64), fr_frame=FRn[:, 1].astype(np.int64), fr_acq=FRn[:, 2].astype(np.int64),
        fr_onf=FRn[:, 3].astype(np.float32), fr_ph=FRn[:, 4].astype(np.float32),
        drift=drift_acq[tacq].astype(np.float32), drift_acq=drift_acq.astype(np.float32), dphi_acq=dphi_acq.astype(np.float32),
        bg_frame=bg_frame.astype(np.float32), bg_map=st.cpu().numpy())
    G = dict(k=torch.tensor(k, dtype=torch.float32), phi=torch.tensor(phi, dtype=torch.float32), m=torch.tensor(m), logrel=torch.tensor(np.log(rel)),
             dphi=torch.tensor(dphi_acq[:, tacq], dtype=torch.float32), knot_frames=1.0, angle_of_step=[0, 0, 0, 1, 1, 1])
    torch.save(G, f'{out}/gt_pattern.pt')
    u = np.stack([np.cos([r[1] for r in RU]), np.sin([r[1] for r in RU])], 1)
    np.savez(f'{out}/sites.npz', site=np.arange(M), ruler=srul, idx=sidx, x=sxy[:, 0], y=sxy[:, 1], x_nm=sxy[:, 0] * a.pixel_nm, y_nm=sxy[:, 1] * a.pixel_nm,
             z_um=sz, ax=u[srul, 0], ay=u[srul, 1], n_events=np.bincount(site, minlength=M), n_full6=np.bincount(site, weights=full6.sum(1), minlength=M).astype(np.int64),
             ruler_cx=np.array([r[0][0] for r in RU]), ruler_cy=np.array([r[0][1] for r in RU]), ruler_angle=np.array([r[1] for r in RU]),
             ruler_nsites=np.array([r[2] for r in RU]), pixel_nm=a.pixel_nm)
    # ---------------------------------------------------------------- summary
    tt = np.arange(NF)[None]; on = np.clip(np.minimum(t1n[:, None], tt + 1) - np.maximum(t0n[:, None], tt), 0, 1).sum(0) if N * NF < 3e8 else None
    inm = (t0n >= 0) & (t1n <= NF); dur = (t1n - t0n)
    wp = ph6[full6]
    info = dict(args=vars(a), n_rulers=n_rul, n_sites=int(M), n_events=int(N), gain_true=gain_true, psf_sum=psf_sum,
                on_per_frame=float(on.mean()) if on is not None else None, on_density=float(on.mean() / (H * W)) if on is not None else None,
                on_time_mean=float(dur[inm].mean()), on_time_median=float(np.median(dur[inm])),
                events_per_site=float(N / M), full6_per_site=float(full6.sum() / M), n_full6=int(full6.sum()),
                full6_photons_p5_p50_p95=[float(v) for v in np.percentile(wp, [5, 50, 95])] if len(wp) else None,
                bg_mean=float(bg_frame.mean()), bg_angle_ratio=[float(bg_frame[sst < 3].mean() / bg_frame.mean()), float(bg_frame[sst >= 3].mean() / bg_frame.mean())],
                mean_adu=float(ADU[::10].mean()), drift_range_nm=[float(v) for v in np.ptp(drift_acq, 0) * a.pixel_nm],
                pattern_drift_range_nm=[float(v) for v in np.ptp(dphi_acq, 1) / (2 * np.pi) * pitch_px * a.pixel_nm])
    json.dump(info, open(f'{out}/sim_info.json', 'w'), indent=1)
    print(json.dumps({k_: v for k_, v in info.items() if k_ != 'args'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
