"""Shared pieces of the paper-simulation regeneration (analysis/papersim/): constants, the preprint's ground truth, blinking,
the movie renderer, the simulated-camera calibration and the per-movie file layout.

Settings (preprint, smlmtorch 5a4fda9 scripts/fig_densities2.py and fig_moving_tubules.py; see STATUS.md in the output dir for
every deviation):
  PSF       in-situ vectorial PSF of the 1809/2 movie averaged over the two pattern axes (models/insitu/calib_1809_2_psf3d_avg.json),
            rendered with the simcode simulator's LUT (simcode/simulate/simulator.py: VectorialPSF, 21 x 21 px ROI, 33 x 33
            sub-pixel knots, bilinear) at the fixed emitter height Z_UM = the calib's emitter_z_median_um (0.055 um; a z knot,
            so no z interpolation). Every emitter of every simulation sits at that height (the preprint sims were 2D).
  pattern   2 angles x 3 steps, frame f has step f % 6, angle = step // 3 (the experimental order [[0,1,2],[3,4,5]]; the
            preprint sims used the interleaved [[0,2,4],[1,3,5]], which the v2 networks and the real pipeline do not support).
            illumination rel (1 + m cos(k_a . r - phi_s)), pitch 224 nm, m 0.9, phi = 0, 2 pi / 3, 4 pi / 3 per angle,
            rel 1, no phase drift. Angles 0 / 90 deg (density movies) and 1 / 91 deg (moving movie), as the preprint.
  photons   intensity I6 = 500 photons per 6 frames for an emitter on for the whole window: per frame I6 / 6 x illumination x
            on-fraction (= smlmtorch simulate(): intensity x relint 1/6 x (1 + depth sin(.)) x on-time).
  camera    pure Poisson photons (no read noise, as the preprint), written as uint16 ADU = photons / GAIN + OFFSET with
            GAIN 1 e/ADU and a flat OFFSET 100 ADU (so the counts are exact); calib_sim.json (+ camera_sim.npz: per-pixel
            offset map 100, read-noise map 0) gives the real pipelines the same camera model (simcode.run, paper_pipeline,
            DECODE all read gain + offset map).
  blinking  smlmtorch.simflux.simulate.blinking_spots: a two-state Markov chain on every GT point with 10 sub-steps per
            frame, k_off = 1 / (avg_on_time x 10), k_on = k_off (1 - p_off) / p_off, p_off = 1 - on_fraction, initial state
            Bernoulli(on_fraction); per-frame on-fraction = on sub-steps / 10. Sampled here exactly as alternating geometric
            run lengths (same distribution, vectorised); the random stream differs from the preprint's.
Coordinates everywhere: px, pixel centre = integer, x = column (simcode / smlmtorch / fastpsf / DECODE convention).

v2-PSF movie sets (2026-11-15, opt-in psf='v2'; the default above is unchanged): see V2 below.
  PSF       in-situ PSF v2 of 1809/2 (models/insitu/calib_1809_2_psf_v2.json: vectorial Zernike 5-21, focus 0.20 um, no blur, default
            dipole model, so no per-angle terms), rendered on a 41 x 41 px ROI (as the training simulator AcquisitionSim; the
            21 x 21 ROI of the old movies holds only 87-94 % of the v2 PSF's light) with LUT planes every 10 nm over 0-0.40 um
            (linear in z, bilinear sub-pixel, 32 sub-steps).
  heights   one height per GT point (a labelling site does not move): the v2 calibration rule of the training simulator
            (simcode.simulate.acquisition.height_quantiles: piecewise-linear inverse CDF of emitter_z_quantiles_um = the
            non-pinned fitted heights of the held-out spots, clipped to emitter_z_range_um = [0, 0.40] um); no TIRF factor.
  photons   DETECTED photons (coordinator decision 2026-11-15): the condition's numbers (500 per window for D, the real-data
            medians for Dexp) are the photons that reach the camera. The vectorial PSF's intensity parameter collects only
            0.96 / 0.72 / 0.54 at z = 0 / 0.116 / 0.40 um (supercritical-angle light decays with z), so every emitter's
            intensity parameter is the condition's photons / collected(z), collected(z) = the sum of the rendered 41 x 41 px LUT
            image at height z (RendererV2.collected, linear in z as the render). (The training simulator AcquisitionSim draws
            its photons as the intensity parameter, i.e. detected = photons x collected(z).)
  camera    the v2 camera of 1809/2 on a fixed 40 x 40 crop (rows / cols 40-80 of models/insitu/camera_maps_1809_2_v2.npz):
            gain 0.4256 e/ADU, offset map (fixed pattern + 103.93 ADU, sd 1.69 ADU), per-pixel Gaussian read noise
            (median 0.90 e, rms 1.34 e); ADU = round((Poisson(mu) + N(0, rn^2)) / gain + offset). calib_sim.json /
            camera_sim.npz carry exactly these maps (the pipelines know the camera, as in the old sims)."""
import ast, json, math, os, subprocess, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path: sys.path.insert(0, REPO)
OUT = os.path.join(REPO, 'out', 'papersim')
PSF_CALIB = os.path.join(REPO, 'models', 'insitu', 'calib_1809_2_psf3d_avg.json')
PX_NM = 108.33
PITCH_NM = 224.0
DEPTH = 0.9
GAIN = 1.0
OFFSET = 100.0
I6 = 500.0
SMLMTORCH = os.path.expanduser('~/smlmtorch')
GT_COMMIT = '5a4fda9'
MOVING_COMMIT = '5f7fdcb'        # fig_moving_tubules.py: last change 93c3e2a; identical at 5f7fdcb (the server's head)
DENSITIES = [0.5, 1, 2, 4]
# v2-PSF movie sets (2026-11-15)
PSF_CALIB_V2 = os.path.join(REPO, 'models', 'insitu', 'calib_1809_2_psf_v2.json')
CAM_V2 = os.path.join(REPO, 'models', 'insitu', 'camera_maps_1809_2_v2.npz')
CAM_V2_CROP = (40, 40)            # top-left (row, col) of the 40 x 40 crop of the v2 camera maps
GAIN_OLD, GAIN_V2 = 0.56, 0.4256
OFFSET_SHIFT_V2 = 4.6256          # v2 offset map - stored 1809/2 offset map (ADU, constant over the sensor ROI)
PHOT_V2 = GAIN_V2 / GAIN_OLD      # photons (intensity) old units -> v2 units: x 0.76


def bg_v2(b_old, nframes=1):
    """background old photon units -> v2 units (photons / px over nframes summed frames): (b / 0.56 - nframes x 4.6256 ADU) x 0.4256
    = 0.76 b - nframes x 1.969"""
    return (np.asarray(b_old, float) / GAIN_OLD - nframes * OFFSET_SHIFT_V2) * GAIN_V2


def heights_v2(n, rng, calib=PSF_CALIB_V2):
    """[n] emitter heights (um) by the v2 calibration rule (simcode.simulate.acquisition.height_quantiles / draw_z, numpy draw)"""
    import torch
    import simcode.simulate.acquisition as acq
    from simcode.psf import read_calib
    c = read_calib(calib)
    d0 = acq.dev; acq.dev = torch.device('cpu')            # the quantile nodes only (CPU-only generation works too)
    try: P, Z = (t.numpy().astype(np.float64) for t in acq.height_quantiles(c))
    finally: acq.dev = d0
    z = np.interp(rng.random(n), P, Z)
    lo, hi = c['emitter_z_range_um']
    return np.clip(z, lo, hi)


def camera_v2(H=40, W=40):
    """(gain, offset_adu [H,W], readnoise_e [H,W]) of the v2-PSF movies: the crop with top-left CAM_V2_CROP (40 x 40 for the density
    movies, 50 x 50 for the moving movie moving_exp_v2psf)"""
    m = np.load(CAM_V2); r, c = CAM_V2_CROP
    assert r + H <= m['offset_adu'].shape[0] and c + W <= m['offset_adu'].shape[1]
    return float(m['gain']), m['offset_adu'][r:r + H, c:c + W].astype(np.float32), m['readnoise_e'][r:r + H, c:c + W].astype(np.float32)


def z_um():
    from simcode.psf import read_calib
    return float(read_calib(PSF_CALIB).get('emitter_z_median_um', 0.055))


# ------------------------------------------------------------------------------------------------ preprint sources
def git_source(commit, path):
    return subprocess.run(['git', '-C', SMLMTORCH, 'show', f'{commit}:{path}'], check=True, capture_output=True, text=True).stdout


def load_functions(commit, path, names, extra_globals=None):
    """the named top-level functions of an (old, notebook-style) smlmtorch script, compiled from its git source without
    running the script's top-level code"""
    src = git_source(commit, path)
    tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef,)) and n.name in names]
    g = {'np': np, 'numpy': np, '__name__': 'papersim_port'}
    g.update(extra_globals or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), f'{commit}:{path}', 'exec'), g)
    return {n: g[n] for n in names}


def preprint_tubules():
    """the preprint's GT point set: 5a4fda9 scripts/fig_densities2.py lines 'np.random.seed(1); generate_microtubule_points(
    W=40, depth=0, numtubules=20, linedensity=0.5 / nm x 108.33, nudge_factor=0.1, margin=0.05, return_knots=True)' with
    generate_microtubule_points of 5a4fda9 scripts/generate_tubules.py. -> pts [N,3] px (z = 0), knots"""
    from scipy.interpolate import InterpolatedUnivariateSpline   # noqa: imported inside the old function too
    f = load_functions(GT_COMMIT, 'scripts/generate_tubules.py', ['generate_microtubule_points'])['generate_microtubule_points']
    np.random.seed(1)
    pts, _, knots = f(40, depth=0, numtubules=20, linedensity=0.5 * PX_NM, nudge_factor=0.1, margin=0.05, return_knots=True)
    return pts, knots


# ------------------------------------------------------------------------------------------------ blinking
def blink_events(n, nframes, avg_on_time, on_fraction, rng, sub=10):
    """exact sampler of smlmtorch blinking_spots (see module doc) -> (spot id, t0, t1) arrays, t in frames (multiples of 1/sub)"""
    on_fraction = min(0.999, on_fraction)
    p_off = 1 - on_fraction
    k_off = 1 / (avg_on_time * sub)
    k_on = min((k_off - p_off * k_off) / p_off, 1.0)
    S = nframes * sub
    state = rng.random(n) < on_fraction                       # x_{-1}
    t = np.zeros(n, np.int64)                                 # first sub-step whose state is not yet assigned
    L = rng.geometric(np.where(state, k_off, k_on)) - 1       # memoryless: further sub-steps in the initial state
    ids, a0, b0 = [], [], []
    active = np.ones(n, bool)
    while active.any():
        idx = np.nonzero(active)[0]
        on = state[idx]; a = t[idx]; b = np.minimum(a + L[idx], S)
        s = on & (b > a)
        ids.append(idx[s]); a0.append(a[s]); b0.append(b[s])
        t[idx] = a + L[idx]
        state[idx] = ~on
        L[idx] = rng.geometric(np.where(state[idx], k_off, k_on))
        active[idx] = t[idx] < S
    ids = np.concatenate(ids); a0 = np.concatenate(a0); b0 = np.concatenate(b0)
    o = np.lexsort((ids, a0))
    return ids[o], a0[o] / sub, b0[o] / sub


# ------------------------------------------------------------------------------------------------ pattern
def gt_pattern(angles_deg):
    """-> dict k [2,2] (rows = angle, rad / px), phi [6], m [6], rel [6]"""
    kk = 2 * np.pi / PITCH_NM * PX_NM
    th = np.deg2rad(np.asarray(angles_deg, float))
    k = np.stack([kk * np.cos(th), kk * np.sin(th)], 1)
    phi = np.tile(np.arange(3) * 2 * np.pi / 3, 2)
    return dict(k=k, phi=phi, m=np.full(6, DEPTH), rel=np.ones(6), angles_deg=list(map(float, angles_deg)), pitch_nm=PITCH_NM)


def illum(P, x, y, frames):
    """illumination of emitters at (x, y) in the given frames (same length arrays)"""
    s = np.asarray(frames) % 6; a = s // 3
    return P['rel'][s] * (1 + P['m'][s] * np.cos(P['k'][a, 0] * x + P['k'][a, 1] * y - P['phi'][s]))


def save_pattern_files(P, nframes, d):
    """gt_pattern.pt (simcode common pattern format, for the GT-pattern diagnostic of run_simcode.py) and mp_gt.pickle (smlmtorch
    ModulationPattern, sin convention: phase_smlmtorch = phi - pi / 2, relint 1/6, pattern_frames [[0,1,2],[3,4,5]])"""
    import torch
    G = dict(k=torch.tensor(P['k'], dtype=torch.float32), phi=torch.tensor(P['phi'], dtype=torch.float32),
             m=torch.tensor(P['m'], dtype=torch.float32), logrel=torch.zeros(6), dphi=torch.zeros(2, 2),
             knot_frames=float(nframes), angle_of_step=[0, 0, 0, 1, 1, 1])
    torch.save(G, os.path.join(d, 'gt_pattern.pt'))
    import smlmtorch.simflux.pattern_estimator as pe
    pf = np.array([[0, 1, 2], [3, 4, 5]])
    mod = np.zeros(6, dtype=pe.ModulationType)
    for s in range(6):
        a = s // 3
        mod['k'][s, :2] = P['k'][a]; mod['k'][s, 2] = 0
        mod['phase'][s] = P['phi'][s] - np.pi / 2
        mod['depth'][s] = P['m'][s]; mod['relint'][s] = 1 / 6
    pe.ModulationPattern(pf, mod).save(os.path.join(d, 'mp_gt.pickle'))


# ------------------------------------------------------------------------------------------------ camera / calibration
GAUSS_CALIB = os.path.join(REPO, 'models', 'insitu', 'calib_gauss1.3.json')   # opt-in key psf_gauss_sigma_um (torch_psf.PSF.load -> GaussianPSF)


def write_sim_calib(d, H, W, P, psf='vectorial'):
    """calib_sim.json (the PSF that rendered the movie + the simulated camera) and camera_sim.npz in movie dir d.
    psf 'vectorial' = the averaged 1809/2 PSF; 'gauss1.3' = models/insitu/calib_gauss1.3.json (pixel-integrated Gaussian,
    psf_gauss_sigma_um), so SIMCODE v2's head template and solve use the movie's PSF"""
    if psf == 'v2':
        return _write_sim_calib_v2(d, H, W, P)
    from simcode.psf import read_calib
    c = read_calib(GAUSS_CALIB if psf.startswith('gauss') else PSF_CALIB)
    if psf.startswith('gauss'):
        assert abs(c['psf_gauss_sigma_um'] / c['pixelsize_um'] - float(psf[5:])) < 1e-3, 'calib_gauss sigma != movie sigma'
    np.savez(os.path.join(d, 'camera_sim.npz'), offset_adu=np.full((H, W), OFFSET, np.float32), readnoise_e=np.zeros((H, W), np.float32),
             gain=np.float32(GAIN))
    c['source'] = ('papersim simulated movie: PSF of ' + os.path.basename(GAUSS_CALIB if psf.startswith('gauss') else PSF_CALIB) + '; camera = pure Poisson, gain %.2f e/ADU, '
                   'flat offset %.1f ADU, no read noise' % (GAIN, OFFSET))
    c['camera'] = dict(gain_e_per_adu=GAIN, offset_adu=OFFSET, readnoise_e=0.0, maps='camera_sim.npz',
                       note='simulated: Poisson only, flat offset map, read-noise map 0')
    c['pattern'] = dict(k_rad_per_px=P['k'].tolist(), pitch_nm=[PITCH_NM, PITCH_NM], depth=P['m'].tolist(), phase=P['phi'].tolist(),
                        note='ground truth of the simulation (the pipeline never reads this entry)')
    p = os.path.join(d, 'calib_sim.json')
    json.dump(c, open(p, 'w'), indent=1)
    return p


def _write_sim_calib_v2(d, H, W, P):
    """calib_sim.json of a v2-PSF movie: the v2 calibration (PSF, emitter-height keys) with the movie's camera (camera_v2 crop)"""
    from simcode.psf import read_calib
    c = read_calib(PSF_CALIB_V2)
    g, off, rn = camera_v2(H, W)
    np.savez(os.path.join(d, 'camera_sim.npz'), offset_adu=off, readnoise_e=rn, gain=np.float32(g))
    c['source'] = ('papersim simulated movie (v2 PSF set): PSF of ' + os.path.basename(PSF_CALIB_V2) + '; camera = the v2 1809/2 camera '
                   'on the %d x %d crop at (row, col) %s of %s: gain %.4f e/ADU, offset map, per-pixel read noise' % (H, W, CAM_V2_CROP, os.path.basename(CAM_V2), g))
    c['camera'] = dict(gain_e_per_adu=g, offset_adu=float(np.median(off)), readnoise_e=float(np.median(rn)), maps='camera_sim.npz',
                       note='simulated: Poisson + per-pixel Gaussian read noise (readnoise_e map), offset map; = the maps that rendered the movie')
    c['pattern'] = dict(k_rad_per_px=P['k'].tolist(), pitch_nm=[PITCH_NM, PITCH_NM], depth=P['m'].tolist(), phase=P['phi'].tolist(),
                        note='ground truth of the simulation (the pipeline never reads this entry)')
    p = os.path.join(d, 'calib_sim.json')
    json.dump(c, open(p, 'w'), indent=1)
    return p


class RendererV2:
    """v2-PSF movies: rows (frame, x, y, z, photons) -> expected images with the v2 PSF LUT (41 x 41 px, 0-0.40 um), + bg,
    Poisson + read noise, ADU with the v2 camera crop (see the module doc)"""
    def __init__(self, H, W, dev='cuda'):
        import torch
        from simcode.simulate.simulator import Simulator
        self.t = torch; self.dev = torch.device(dev); self.H, self.W = H, W
        self.sim = Simulator(PSF_CALIB_V2, H=H, W=W, T=6, device=dev, psf_roi=41, psf_lut_z=41, psf_lut_sub=32, zmin=0.0, zmax=0.40)
        self.R = self.sim.R
        g, off, rn = camera_v2(H, W)
        self.gain = g; self.off = torch.tensor(off, device=self.dev); self.rn = torch.tensor(rn, device=self.dev)
        L = self.sim.lut
        self.coll = L[:, L.shape[1] // 2, L.shape[2] // 2].sum((-2, -1)).double().cpu().numpy()    # [nz] collected fraction per LUT plane
        self.coll_z = self.sim.zs.double().cpu().numpy()

    def collected(self, z):
        """fraction of an emitter's intensity parameter that the rendered PSF image holds at height z (um; linear between the
        LUT planes, as the render)"""
        return np.interp(np.asarray(z, float), self.coll_z, self.coll)

    def expected(self, f0, nf, fr, x, y, z, ph):
        torch = self.t; dev = self.dev; H, W, R = self.H, self.W, self.R; hr = R // 2
        pad = R + 1
        img = torch.zeros(nf, H + 2 * pad, W + 2 * pad, device=dev)
        x = torch.as_tensor(x, device=dev, dtype=torch.float32); y = torch.as_tensor(y, device=dev, dtype=torch.float32)
        z = torch.as_tensor(z, device=dev, dtype=torch.float32)
        keep = (x > -hr) & (x < W + hr - 1) & (y > -hr) & (y < H + hr - 1)
        if keep.any():
            fr = torch.as_tensor(fr, device=dev).long()[keep] - f0; ph = torch.as_tensor(ph, device=dev, dtype=torch.float32)[keep]
            x = x[keep]; y = y[keep]; z = z[keep]
            oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
            Hp, Wp = img.shape[-2:]
            for s in range(0, len(x), 8000):
                sl = slice(s, s + 8000)
                psf = self.sim.render_psf(x[sl], y[sl], z[sl])
                cx = torch.round(x[sl]).long() - hr + pad; cy = torch.round(y[sl]).long() - hr + pad
                lin = fr[sl, None] * Hp * Wp + ((cy[:, None, None] + oy) * Wp + cx[:, None, None] + ox).reshape(len(psf), -1)
                img.view(-1).index_add_(0, lin.reshape(-1), (ph[sl, None, None] * psf).reshape(-1))
        return img[:, pad:pad + H, pad:pad + W]

    def adu(self, mu, gen):
        torch = self.t
        e = torch.poisson(mu, generator=gen) + self.rn * torch.randn(mu.shape, device=self.dev, generator=gen)
        return torch.round(e / self.gain + self.off).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)


# ------------------------------------------------------------------------------------------------ renderer
class Renderer:
    """rows (frame, x, y, photons) -> expected images with the simulator's PSF LUT at the fixed z, + bg, Poisson, ADU"""
    def __init__(self, H, W, dev='cuda', psf='vectorial'):
        import torch
        from simcode.simulate.simulator import Simulator
        self.t = torch; self.dev = torch.device(dev); self.H, self.W = H, W
        self.z = z_um()
        # z knots 0, z, 2z: the emitter plane is a knot (no z interpolation)
        self.sim = Simulator(PSF_CALIB, H=H, W=W, T=6, device=dev, psf_roi=21, psf_lut_z=3, psf_lut_sub=32, zmax=2 * self.z)
        self.R = self.sim.R
        self.gauss = None
        if psf.startswith('gauss'):      # control: the preprint's Gaussian PSF (pixel-integrated, sigma in px), e.g. 'gauss1.3'
            sys.path.insert(0, os.path.join(REPO, 'analysis', 'baselines')); import zpsf
            self.gauss = zpsf.GaussPSF(float(psf[5:]), dev)

    def _psf(self, x, y):
        if self.gauss is None:
            return self.sim.render_psf(x, y, self.t.full_like(x, self.z))
        return self.gauss.eval(x - self.t.round(x), y - self.t.round(y), self.R)[0]

    def expected(self, f0, nf, fr, x, y, ph):
        """expected photon images [nf,H,W] (signal only) of rows with frame fr in [f0, f0 + nf)"""
        torch = self.t; dev = self.dev; H, W, R = self.H, self.W, self.R; hr = R // 2
        pad = R + 1                                        # canvas index = pixel + pad
        img = torch.zeros(nf, H + 2 * pad, W + 2 * pad, device=dev)
        x = torch.as_tensor(x, device=dev, dtype=torch.float32); y = torch.as_tensor(y, device=dev, dtype=torch.float32)
        keep = (x > -hr) & (x < W + hr - 1) & (y > -hr) & (y < H + hr - 1)
        if keep.any():
            fr = torch.as_tensor(fr, device=dev).long()[keep] - f0; ph = torch.as_tensor(ph, device=dev, dtype=torch.float32)[keep]
            x = x[keep]; y = y[keep]
            for s in range(0, len(x), 20000):
                sl = slice(s, s + 20000)
                psf = self._psf(x[sl], y[sl])                                                    # [n,R,R] centred on round(x)
                cx = torch.round(x[sl]).long() - hr + pad; cy = torch.round(y[sl]).long() - hr + pad   # canvas index of PSF px 0
                oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
                Hp, Wp = img.shape[-2:]
                lin = fr[sl, None] * Hp * Wp + ((cy[:, None, None] + oy) * Wp + cx[:, None, None] + ox).reshape(len(psf), -1)
                img.view(-1).index_add_(0, lin.reshape(-1), (ph[sl, None, None] * psf).reshape(-1))
        return img[:, pad:pad + H, pad:pad + W]

    def adu(self, mu, gen):
        torch = self.t
        e = torch.poisson(mu, generator=gen)
        return torch.round(e / GAIN + OFFSET).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)


def write_movie(path, frames_iter):
    import tifffile
    with tifffile.TiffWriter(path, bigtiff=False) as tw:
        for blk in frames_iter:
            for im in blk:
                tw.write(im, contiguous=True)


def movie_dir(name):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True); return d


def status_line(d, msg):
    with open(os.path.join(d, 'log.txt'), 'a') as f:
        f.write(msg.rstrip() + '\n')
