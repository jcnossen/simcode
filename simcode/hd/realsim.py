"""
Experiment-matched simulator (hd step 0). Wraps simcode/sim/simulator.py (PSF LUT rendering, calibration) without
editing it. Everything is taken from existing calibration / summary files only (no movie is read):
  PSF       calibrated vectorial PSF (analysis/calib_0.5nM.json); per-emitter PSF jitter = one of NPSF perturbed LUTs
            (width x U(0.97, 1.03), Zernike jitter 0.008 waves, focus +-15 nm; psfcheck: 2-3 % coherent residual)
  pattern   k of the estimated patterns (pattern8 / pipeline pattern_est: pitch 2.070 px, angles 1.17-1.22 / 91.35 deg),
            randomized: rotation +-0.05 deg, pitch +-0.1 %; step spacing 2.09-2.11 rad (measured, not exactly 2 pi / 3);
            depth per preset; step relint and angle ratio per preset; within-window phase drift eps(t) random walk
            0.25 nm / frame (measured eps rms 4.5-7 nm over 10^5 frames, smooth); random global phases
  estimate  the pattern the localizer gets (patt_est): per-angle common phase error sd 0.06 rad (~2.1 nm), per-step
            0.01 rad, linear phase-field ramp across the FOV up to +-0.08 rad (pattern_compare k_field 1.4-3.8 nm rms),
            depth error sd 0.02, relint error sd 0.02
  camera    gain 0.56 e/ADU (+ per-FOV gain error sd 1 %), per-pixel read-noise maps (camera_maps_0.5nM /
            hd/camera_maps_2p5nM_m2), per-pixel residual offset error sd 0.3 ADU, ADU rounding
  bg        per-angle levels, static smooth structure (two scales) matched to bgstats.json (median / spatial sd),
            temporal fluctuation (hybrid25 bgmap_steps: global AR(1) sd 2.7 %, smooth field sd 1.8 ph, corr length
            4.3 px, lag-1 0.5) -- scaled for the 0.5 nM presets
  emitters  Poisson DNA-PAINT events (ON density per frame per px), on-time lognormal (median tau), z lognormal
            (median ~50 nm, clipped 0-200 nm), brightness lognormal x TIRF exp(-z / 82 nm), per-emitter angle factor
            (log-sd 0.18), per-emitter per-step factor (sd 0.05), per-frame multiplicative jitter (sd 0.03);
            geometry: uniform or microtubule-like smooth filaments (fraction per preset)
Ground truth gt['phot'] [N,T] = the actual per-frame photons (all factors included).
Presets: '0.5nM' (0.5 nM / 1809_2-like, low density), '2.5nM' (2.5 nM movie-2-like).
Iteration 27 (critic review 26): fine static bg scaled to the measured high-pass sd, PSF rendered on 41 x 41, filament
density per area, frame_sd 0.10, halo jitter (0-5 % in a sigma-4 px component), fluct amplitude reduced (x0.15 / x0.35 of
the measured upper estimate: at full amplitude the pf front-end found 36-44 % false positives and 2-3x the real
low-photon detection fraction); dens / bg_sd re-tuned uncapped (validate_sim k = 1500, real-like crop + selection). Values were tuned against
the real-output statistics realstats.json with the development scripts realstats.py / validate_sim.py. Those ran the legacy
pattern-free + hybrid (pixel-fit) front-end on real and simulated windows; that front-end is not part of simcode, so the
tuning check is not reproducible here. Snapshots of its statistics and of the final sim-vs-real tables:
analysis/results/realism/ (realstats.json, validate_{0.5,2.5}nM_final.txt). The realism check against the simcode front-end
itself (det_n1 + sc5nb + pattern solve, same statistics on real and sim pass-1 outputs) is analysis/figures/scorecard.py
(-> realism_scorecard.png).
API: RealSim(preset, H, W, T).scene(nb, em=None, **overrides) -> dict as crlb_dense.make_scene (+ patt_est,
illum_est, shift_est, bg_static); .emitters(nb, **kw) -> em dict.
(2026-11-10/11 critic 37 #2, opt-in; absent keys = unchanged, no extra random draws) smooth illumination envelope across the
FOV, measured on the real movies (<out>/evalproto/envmap.py): keys env_x0 / env_y0 (centre px), env_s1 / env_s2
(Gaussian sd px along the principal axes), env_th (angle rad of axis 1) -> RealSim.envelope(x, y) = exp(-0.5 q) normalized to
geometric mean 1 over the 5-115 px crop (multiply emitter brightness by it); env_bg = fraction of the static bg that follows
the envelope (background(): st x (1 - env_bg + env_bg En), En = envelope normalized to mean 1 over the crop; FOV-mean bg kept).
"""
import json, math
import numpy as np, torch
from simcode.sim.simulator import Simulator
from simcode import config as _cfg

dev = 'cuda'
_M = lambda f: _cfg.model('insitu/' + f)
PRESETS = {
    '0.5nM': dict(calib=_M('calib_0.5nM.json'), cam=_M('camera_maps_0.5nM.npz'),
                  k=[[3.03634, 0.06331], [-0.07080, 3.03422]], spacing=(2.09, 2.11), depth=(0.93, 0.955), depth_sd=0.015,
                  relint=(1.0, 1.0), angle_ratio=(0.96, 1.10), step_rel_sd=0.03,
                  bg=6.1, bg_sd=0.5, bg_hp=0.25, bg_ang=(1.03, 0.97), fluct=0.15, dens=0.0085, tau=65.0, bright=140.0, bright_sd=0.4,
                  z_med=0.050, z_sd=0.45, struct_frac=0.5, ang_sd=0.14, step_sd=0.07, frame_sd=0.10, halo=0.05),
    '2.5nM': dict(calib=_M('calib_0.5nM.json'), cam=_M('camera_maps_2p5nM_m2.npz'),
                  k=[[3.03503, 0.06479], [-0.07334, 3.03187]], spacing=(2.088, 2.118), depth=(0.76, 0.80), depth_sd=0.03,
                  relint=(1.10, 0.90), angle_ratio=(1.0, 1.0), step_rel_sd=0.02,
                  bg=10.8, bg_sd=3.0, bg_hp=0.45, bg_ang=(1.09, 0.91), fluct=0.35, dens=0.024, tau=65.0, bright=185.0, bright_sd=0.35,
                  z_med=0.055, z_sd=0.40, struct_frac=0.85, ang_sd=0.2, step_sd=0.07, frame_sd=0.10, halo=0.05),
}
PX_NM = 108.33


def lu(lo, hi, *n):
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(*n, device=dev))


def psf_blur(P_, sx, sy):
    """(2026-11-11) anisotropic Gaussian image blur (px) of PSF images [..., R, R] (FFT, zero-padded, centred kernel)"""
    R_ = P_.shape[-1]; sh = P_.shape
    fy_, fx_ = torch.meshgrid(torch.fft.fftfreq(2 * R_, device=P_.device), torch.fft.fftfreq(2 * R_, device=P_.device), indexing='ij')
    K_ = torch.exp(-2 * math.pi ** 2 * (sx ** 2 * fx_ ** 2 + sy ** 2 * fy_ ** 2))
    out = []
    for c in P_.reshape(-1, R_, R_).split(4096):
        out.append(torch.fft.ifft2(torch.fft.fft2(torch.nn.functional.pad(c, (0, R_, 0, R_))) * K_).real[:, :R_, :R_].clamp(min=0))
    return torch.cat(out).reshape(sh)


def psf_div_luts(rs, dv, seed=0):
    """(2026-11-11 PSF-shape mismatch, opt-in RealSim key psfdiv = dict) PSF-diversity bank: n LUTs, each = calib + Gaussian
    Zernike jitter (sd zsd waves on the Noll indices `noll`) + base_dz {noll: waves} + focus shift base_df + U(+-fr/2) um, width x
    1 + wr (U - 0.5), then an anisotropic Gaussian image blur sx = max(0, base_blur[0] + blur_r (2U - 1)), sy likewise.
    keep_old: prepend the existing (calib + weak jitter) LUTs, unblurred.  Own generator (seed 4321 + seed): no global draws."""
    from simcode.det.psfpert import build_lut
    g = torch.Generator().manual_seed(4321 + seed)
    zl = rs.sim.c['zernike_noll']; nz = len(zl)
    out = list(rs.luts) if dv.get('keep_old', True) else []
    for i in range(int(dv.get('n', 12))):
        u = torch.rand(5, generator=g)
        zd = torch.zeros(nz)
        for j in dv.get('noll', [5, 6, 7, 8]): zd[zl.index(j)] = float(torch.randn(1, generator=g)) * dv.get('zsd', 0.04)
        for j, v in dv.get('base_dz', {}).items(): zd[zl.index(int(j))] += v
        fs = dv.get('base_df', 0.0) + (float(u[0]) - 0.5) * dv.get('fr', 0.06)
        w = 1.0 + dv.get('wr', 0.0) * (float(u[1]) - 0.5)
        lut = build_lut(rs.sim, w, zd.to(dev), fs, rs.zmax)[0]
        bb = dv.get('base_blur', [0.0, 0.0]); br = dv.get('blur_r', 0.0)
        sx = max(0.0, bb[0] + br * (2 * float(u[2]) - 1)); sy = max(0.0, bb[1] + br * (2 * float(u[3]) - 1))
        if sx > 0 or sy > 0: lut = psf_blur(lut, sx, sy)
        out.append(lut)
    return out


def add_halo2(psf, x, y, frac, sig, R=None):
    """(2026-11-09, sim realism, critic 32 #3) broad emitter-correlated halo: a fraction `frac` of each emitter's light moved
    into an exponential disc exp(-r / sig) (px) centred on the emitter (sub-pixel), normalized to the PSF's own sum on the ROI.
    psf [N,R,R] rendered at (x, y) (ROI centre = round(x), round(y)). frac may be a scalar or [N]."""
    R = psf.shape[-1]
    o_ = torch.arange(R, device=psf.device, dtype=torch.float32) - R // 2
    fx_, fy_ = x - torch.round(x), y - torch.round(y)
    r = torch.sqrt((o_[None, None, :] - fx_[:, None, None]) ** 2 + (o_[None, :, None] - fy_[:, None, None]) ** 2)
    h = torch.exp(-r / sig)
    h = h / h.sum((1, 2), keepdim=True) * psf.sum((1, 2), keepdim=True)
    f = torch.as_tensor(frac, device=psf.device, dtype=torch.float32)
    f = f.reshape(-1, 1, 1) if f.dim() else f
    return (1 - f) * psf + f * h


def filament_static_map(pts, H, W, sig=1.5, up=8):
    """(2026-11-09, critic 32 #3) unit map of filament-correlated dim static light (unresolved / out-of-focus binding along the
    microtubules): the realsim filament curves (emitters(...)['fil_pts'][fov] = [nfil, ns, 2] px) rasterized as lines
    (points densified x`up`, bilinear splat, equal weight per unit length), blurred with a Gaussian of `sig` px, normalized
    to mean 1 over the H x W FOV (all-zero map if there are no filaments). Static light = A * D added to the bg; the flat
    bg is lowered by A so the FOV-mean bg is unchanged. Deterministic (no random draws)."""
    if pts is None: return torch.zeros(H, W)
    P = pts.float().cpu()                                                      # [nfil, ns, 2]
    t = torch.linspace(0, 1, up + 1)[:-1]
    d = (P[:, :-1, None] + (P[:, 1:] - P[:, :-1])[:, :, None] * t[None, None, :, None]).reshape(-1, 2)
    k = int(4 * sig) + 1; Pd = k + 2
    h = torch.zeros(H + 2 * Pd, W + 2 * Pd)
    xx, yy = d[:, 0] + Pd, d[:, 1] + Pd
    x0, y0 = torch.floor(xx).long(), torch.floor(yy).long(); fx, fy = xx - x0, yy - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy; ok = (xi >= 0) & (xi < W + 2 * Pd) & (yi >= 0) & (yi < H + 2 * Pd)
        h.index_put_((yi[ok], xi[ok]), w[ok], accumulate=True)
    o = torch.arange(-k, k + 1, dtype=torch.float32); g = torch.exp(-0.5 * (o / sig) ** 2); g = g / g.sum()
    h = torch.nn.functional.conv2d(h[None, None], g.view(1, 1, 1, -1), padding=(0, k))
    h = torch.nn.functional.conv2d(h, g.view(1, 1, -1, 1), padding=(k, 0))[0, 0]
    D = h[Pd:Pd + H, Pd:Pd + W]
    return D / D.mean() if float(D.mean()) > 0 else D


def smooth_field(n, H, W, sig):
    k = int(3 * sig); o = torch.arange(-k, k + 1, device=dev, dtype=torch.float32)
    g = torch.exp(-0.5 * (o / sig) ** 2); g = g / g.sum()
    x = torch.randn(n, 1, H + 2 * k, W + 2 * k, device=dev)
    x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, -1)); x = torch.nn.functional.conv2d(x, g.view(1, 1, -1, 1))
    x = x[:, 0]
    return (x - x.mean((1, 2), keepdim=True)) / x.std((1, 2), keepdim=True).clamp(min=1e-6)


class RealSim:
    def __init__(self, preset='0.5nM', H=64, W=64, T=6, npsf=4, psf_jitter=True, seed=0, psf_roi=41, **over):
        self.p = dict(PRESETS[preset], **over); self.preset = preset
        # PSF rendered on a 41 x 41 ROI (critic review 26 #3: 19 % of the real PSF energy lies outside 11 x 11, 8 % beyond 21 x 21)
        self.sim = Simulator(self.p['calib'], H=H, W=W, T=T, device=dev, psf_roi=psf_roi)
        self.H, self.W, self.T = H, W, T
        self.zmax = self.sim.zmax; self.focus = self.sim.focus
        self.rn_map = torch.tensor(np.load(self.p['cam'])['readnoise_e'], device=dev).float()
        self.luts = [self.sim.lut]
        # (2026-11-11, opt-in per-angle PSF calib) every LUT-bank entry is a tuple (angle-0 LUT, angle-1 LUT); same jitter draws
        ang = self.sim.lut_angle is not None
        if ang: self.luts = [tuple(self.sim.lut_angle)]
        if psf_jitter and npsf > 1:
            from simcode.det.psfpert import build_lut
            g = torch.Generator().manual_seed(1234 + seed)
            nz = len(self.sim.c['zernike_waves'])
            for i in range(npsf - 1):
                u = torch.rand(2, generator=g)
                # (2026-11-10 simreal2) opt-in jitter widths: psf_wr (width range, 0.06), psf_fr (focus range um, 0.03), psf_zs (Zernike sd waves, 0.008)
                w = 1.0 + self.p.get('psf_wr', 0.06) * (float(u[0]) - 0.5); fs = (float(u[1]) - 0.5) * self.p.get('psf_fr', 0.03)
                zd = (torch.randn(nz, generator=g) * self.p.get('psf_zs', 0.008)).to(dev)
                self.luts.append(tuple(build_lut(self.sim, w, zd, fs, self.zmax, angle=k)[0] for k in range(2)) if ang
                                 else build_lut(self.sim, w, zd, fs, self.zmax)[0])
        if self.p.get('psfdiv'):
            assert not ang, 'psfdiv bank: no per-angle PSF support'    # (2026-11-11 PSF-shape mismatch, opt-in) PSF-diversity LUT bank, see psf_div_luts
            self.luts = psf_div_luts(self, self.p['psfdiv'], seed)

    # ----------------------------------------------------------------------------------------------- emitters
    def emitters(self, nb, dens=None, tau=None, struct_frac=None, bright=None, margin=6, pair_frac=0.0,
                 pair_sep=(0.5, 3.0), pair_curr=False, fil_pts_in=None, use_sites=True, pair_sub=None):
        # (2026-11-10 simreal2, opt-in; defaults unchanged) fil_pts_in: reuse these filament curves [nb, nfil, ns, 2] (e.g. a dim-binder
        # population on the same microtubules); preset keys fil_on (fraction of events on a filament, 0.85), fil_w (filament
        # width sd nm, 12), fil_resample (redraw the filament point up to 12 times when it falls outside the canvas instead of
        # placing the event uniformly), fil_L (filament length px, 256)
        p = self.p; H, W, T = self.H, self.W, self.T
        g = lambda *s: torch.rand(*s, device=dev)
        dens = p['dens'] if dens is None else dens
        d = torch.as_tensor(dens, device=dev, dtype=torch.float32).expand(nb).clone() if not torch.is_tensor(dens) or dens.dim() == 0 else dens
        tau = torch.full((nb,), p['tau'] if tau is None else tau, device=dev) if tau is None or not torch.is_tensor(tau) else tau
        area = (H + 2 * margin) * (W + 2 * margin)
        # per-event on-time lognormal around tau (sd 0.5): mean events start rate from the mean on-time
        # (2026-11-10/11 evaluation protocol, opt-in; absent = unchanged draws) heavy on-time tail: a fraction tau_tf of the
        # events has mean on-time tau x tau_tm; the start rate is scaled so that the ON density stays `dens`
        tf, tm = float(p.get('tau_tf', 0.0)), float(p.get('tau_tm', 1.0))
        mix = tf > 0 and tm != 1.0
        t_lo = -6 * tau * (max(tm, 1.0) if mix else 1.0)
        n_ev = torch.poisson(d * area * (T - t_lo) / tau / ((1 - tf + tf * tm) if mix else 1.0)).long()
        fov = torch.repeat_interleave(torch.arange(nb, device=dev), n_ev); N = len(fov)
        t0 = t_lo[fov] + (T - t_lo[fov]) * g(N)
        dur = -torch.log(g(N).clamp(min=1e-9)) * tau[fov]
        if mix:
            dur = torch.where(g(N) < tf, dur * tm, dur)
        t1 = t0 + dur
        keep = (t1 > 0) & (t0 < T); fov, t0, t1 = fov[keep], t0[keep], t1[keep]; N = len(fov)
        x = -margin + (W + 2 * margin) * g(N); y = -margin + (H + 2 * margin) * g(N)
        sf = p['struct_frac'] if struct_frac is None else struct_frac
        structured = g(nb) < sf
        fil_pts, fil_struct = None, structured & False
        if structured.any() and N:
            # microtubule-like: nfil smooth curves (random-walk direction, persistence ~ 20 px), emitters within 12 nm
            # filament length per area fixed (0.22 px / px^2 = 5 filaments of 256 px in a 76 x 76 px canvas), independent
            # of the FOV size (critic review 26 #1)
            L = float(p.get('fil_L', 256.0)); ns = 128
            nfil = max(1, int(round(p.get('fil_density', 0.22) * (W + 2 * margin) * (H + 2 * margin) / L)))
            p0 = torch.stack([-margin + (W + 2 * margin) * g(nb, nfil), -margin + (H + 2 * margin) * g(nb, nfil)], -1)
            th0 = g(nb, nfil) * 2 * math.pi
            dth = torch.cumsum(torch.randn(nb, nfil, ns, device=dev) * 0.12, -1) + th0[..., None]
            step = L / ns
            pts = p0[..., None, :] + torch.cumsum(torch.stack([torch.cos(dth), torch.sin(dth)], -1) * step, -2) - L / 2 * torch.stack([torch.cos(th0), torch.sin(th0)], -1)[..., None, :]
            if fil_pts_in is not None:
                pts = fil_pts_in.to(dev); nfil = pts.shape[1]; ns = pts.shape[2]
            fi = torch.randint(0, nfil, (N,), device=dev); s = g(N) * (ns - 1)
            si = s.floor().long(); fr = (s - si)[:, None]
            P0 = pts[fov, fi, si]; P1 = pts[fov, fi, (si + 1).clamp(max=ns - 1)]
            q = P0 + (P1 - P0) * fr + torch.randn(N, 2, device=dev) * (p.get('fil_w', 12.0) / PX_NM)
            inside = (q[:, 0] > -margin) & (q[:, 0] < W + margin) & (q[:, 1] > -margin) & (q[:, 1] < H + margin)
            if p.get('fil_resample', False):
                for _ in range(12):
                    o_ = ~inside
                    if not o_.any(): break
                    fi2 = torch.randint(0, nfil, (N,), device=dev); s2 = g(N) * (ns - 1); si2 = s2.floor().long(); fr2 = (s2 - si2)[:, None]
                    Q0 = pts[fov, fi2, si2]; Q1 = pts[fov, fi2, (si2 + 1).clamp(max=ns - 1)]
                    q2 = Q0 + (Q1 - Q0) * fr2 + torch.randn(N, 2, device=dev) * (p.get('fil_w', 12.0) / PX_NM)
                    q = torch.where(o_[:, None], q2, q)
                    inside = (q[:, 0] > -margin) & (q[:, 0] < W + margin) & (q[:, 1] > -margin) & (q[:, 1] < H + margin)
            use = structured[fov] & inside & (g(N) < p.get('fil_on', 0.85))
            x = torch.where(use, q[:, 0], x); y = torch.where(use, q[:, 1], y)
            fil_pts, fil_struct = pts, structured
        if pair_frac > 0 and N:
            pi = torch.nonzero(g(N) < pair_frac)[:, 0]; M = len(pi)
            if M:
                sep = pair_sep[0] + (pair_sep[1] - pair_sep[0]) * g(M); th = g(M) * 2 * math.pi
                if pair_curr:   # curriculum: half the partners at 1.5-2.5 px, 30 % of the directions along a pattern k
                    sep = torch.where(g(M) < 0.5, 1.5 + g(M), sep)
                    along = g(M) < 0.3
                    th = torch.where(along, torch.randint(0, 4, (M,), device=dev).float() * (math.pi / 2) + 0.021 + 0.05 * torch.randn(M, device=dev), th)
                if pair_sub is not None and pair_sub[0] > 0:   # (2026-11-11 item 4, opt-in; None = unchanged draws) close-pair enrichment:
                    sep = torch.where(g(M) < pair_sub[0], pair_sub[1] + (pair_sub[2] - pair_sub[1]) * g(M), sep)   # (frac, lo, hi) px
                tp = tau[fov[pi]]
                pt0 = -tp + (T - 0.5 + tp) * g(M); pt1 = pt0 + 0.5 - torch.log(g(M).clamp(min=1e-9)) * tp
                fov = torch.cat([fov, fov[pi]]); t0 = torch.cat([t0, pt0]); t1 = torch.cat([t1, pt1])
                x = torch.cat([x, x[pi] + sep * torch.cos(th)]); y = torch.cat([y, y[pi] + sep * torch.sin(th)])
                ok = (x > -margin) & (x < W + margin) & (y > -margin) & (y < H + margin)
                fov, t0, t1, x, y = fov[ok], t0[ok], t1[ok], x[ok], y[ok]; N = len(fov)
        site = None
        if use_sites and p.get('site_ev', 0) > 0 and N:
            x, y, site, z_site, site_info = self._sites(nb, fov, t0, x, y, fil_pts, fil_struct, margin)
        z = (p['z_med'] * torch.exp(p['z_sd'] * torch.randn(N, device=dev))).clamp(0.0, self.zmax)
        if site is not None:
            z = z_site[site]
        Bm = p['bright'] if bright is None else bright
        Bm = torch.as_tensor(Bm, device=dev, dtype=torch.float32)
        Bm = Bm[fov] if Bm.dim() == 1 else Bm
        br = Bm * torch.exp(p['bright_sd'] * torch.randn(N, device=dev)) * torch.exp(-(z - p['z_med']) / self.sim.tirf)
        out = dict(fov=fov, x=x, y=y, z=z, bright=br, t0=t0, t1=t1, fil_pts=fil_pts, fil_struct=fil_struct)
        if site is not None:
            out['site'] = site; out.update(site_info)
        return out

    def _sites(self, nb, fov, t0, x, y, fil_pts, fil_struct, margin):
        """(2026-11-10/11 evaluation protocol, opt-in via preset key site_ev > 0; never called otherwise) discrete DNA-PAINT
        docking sites with repeated binding: n_site = N_events / site_ev sites per batch (same geometry as the events:
        filaments with fil_on / fil_w / fil_resample, uniform otherwise); each event is assigned to a site active at its start
        (probability ~ per-site weight, lognormal log-sd site_wsd); per-site activity window (exponential length, mean site_L
        frames; 0 = whole movie); event position = site + N(0, site_jit nm) (true dye position, GT); z drawn per SITE.
        Returns x, y, site index [N], z_site [n_site], dict(site_x, site_y, site_z, site_fov, site_a, site_b)."""
        p = self.p; H, W, T = self.H, self.W, self.T
        g = lambda *s: torch.rand(*s, device=dev)
        N = len(fov)
        ns_f = torch.bincount(fov, minlength=nb).float() / float(p['site_ev'])
        n_s = torch.poisson(ns_f).long().clamp(min=1)
        sfov = torch.repeat_interleave(torch.arange(nb, device=dev), n_s); M = len(sfov)
        sx = -margin + (W + 2 * margin) * g(M); sy = -margin + (H + 2 * margin) * g(M)
        if fil_pts is not None and bool(fil_struct.any()):
            nfil, ns = fil_pts.shape[1], fil_pts.shape[2]
            q = torch.zeros(M, 2, device=dev); inside = torch.zeros(M, dtype=torch.bool, device=dev)
            for it in range(13 if p.get('fil_resample', False) else 1):
                fi = torch.randint(0, nfil, (M,), device=dev); s = g(M) * (ns - 1); si = s.floor().long(); fr = (s - si)[:, None]
                Q0 = fil_pts[sfov, fi, si]; Q1 = fil_pts[sfov, fi, (si + 1).clamp(max=ns - 1)]
                q2 = Q0 + (Q1 - Q0) * fr + torch.randn(M, 2, device=dev) * (p.get('fil_w', 12.0) / PX_NM)
                q = torch.where(inside[:, None], q, q2)
                inside = (q[:, 0] > -margin) & (q[:, 0] < W + margin) & (q[:, 1] > -margin) & (q[:, 1] < H + margin)
            use = fil_struct[sfov] & inside & (g(M) < p.get('fil_on', 0.85))
            sx = torch.where(use, q[:, 0], sx); sy = torch.where(use, q[:, 1], sy)
        w = torch.exp(p.get('site_wsd', 0.0) * torch.randn(M, device=dev))
        L = p.get('site_L', 0.0)
        if L > 0:
            Ls = -torch.log(g(M).clamp(min=1e-9)) * L
            sa = -Ls + (T + Ls) * g(M); sb = sa + Ls
        else:
            sa = torch.full((M,), -1e9, device=dev); sb = torch.full((M,), 1e9, device=dev)
        # per-FOV weighted site draw (cumulative weights, offset by the FOV's first site), rejection on activity at t0
        first = torch.cumsum(n_s, 0) - n_s
        cw = torch.zeros(M, device=dev)
        for b in range(nb):
            a_, c_ = int(first[b]), int(first[b] + n_s[b])
            cw[a_:c_] = torch.cumsum(w[a_:c_], 0) / w[a_:c_].sum()
        site = torch.full((N,), -1, dtype=torch.long, device=dev)
        t0c = t0.clamp(0, T)
        for it in range(200):
            todo = torch.nonzero(site < 0)[:, 0]
            if len(todo) == 0: break
            fb = fov[todo]; u = g(len(todo))
            # searchsorted within each FOV's block of cw (global offset trick: cw + FOV index is increasing)
            key = cw + sfov.float()
            j = torch.searchsorted(key.contiguous(), (u + fb.float()).contiguous()).clamp(max=M - 1)
            ok = (sa[j] <= t0c[todo]) & (sb[j] > t0c[todo]) & (sfov[j] == fb)
            site[todo[ok]] = j[ok]
        miss = site < 0
        if miss.any():      # no active site found: nearest-in-time active-window site of the FOV (rare)
            j = torch.randint(0, M, (int(miss.sum()),), device=dev); site[miss] = j
        jit = p.get('site_jit', 0.0) / PX_NM
        x = sx[site] + jit * torch.randn(N, device=dev); y = sy[site] + jit * torch.randn(N, device=dev)
        z_site = (p['z_med'] * torch.exp(p['z_sd'] * torch.randn(M, device=dev))).clamp(0.0, self.zmax)
        return x, y, site, z_site, dict(site_x=sx, site_y=sy, site_z=z_site, site_fov=sfov, site_a=sa, site_b=sb, site_w=w)

    # ----------------------------------------------------------------------------------------------- pattern
    def pattern(self, nb, exact_est=False):
        p = self.p; T = self.T; H, W = self.H, self.W
        g = lambda *s: torch.rand(*s, device=dev)
        k0 = torch.tensor(p['k'], device=dev)
        rot = (g(nb, 2) - 0.5) * 2 * math.radians(0.05); sc = 1 + (g(nb, 2) - 0.5) * 2e-3
        k = torch.stack([k0[:, 0] * torch.cos(rot) - k0[:, 1] * torch.sin(rot),
                         k0[:, 0] * torch.sin(rot) + k0[:, 1] * torch.cos(rot)], -1) * sc[..., None]   # [nb,2,2]
        sp = p['spacing'][0] + (p['spacing'][1] - p['spacing'][0]) * g(nb, 2, 1)
        phi = (g(nb, 2, 1) * 2 * math.pi + torch.arange(3, device=dev) * sp).reshape(nb, 6)
        dm = torch.tensor([p['depth'][0]] * 3 + [p['depth'][1]] * 3, device=dev)
        m = (dm + torch.randn(nb, 1, device=dev) * p['depth_sd'] + torch.randn(nb, 6, device=dev) * 0.005).clamp(0, 1)
        ar = p['angle_ratio'][0] + (p['angle_ratio'][1] - p['angle_ratio'][0]) * g(nb)        # angle 0 / angle 1
        base = torch.tensor([p['relint'][0]] * 3 + [p['relint'][1]] * 3, device=dev)
        rel = base * torch.cat([ar.sqrt()[:, None].expand(nb, 3), ar.rsqrt()[:, None].expand(nb, 3)], 1)
        rel = rel * (1 + torch.randn(nb, 6, device=dev) * p['step_rel_sd'])
        rel = rel / rel.mean(1, keepdim=True)
        step0 = torch.randint(0, 2, (nb,), device=dev) * 3       # windows start at a step-0 or step-3 frame
        steps = (step0[:, None] + torch.arange(T, device=dev)[None]) % 6
        angle = steps // 3
        kk = k[torch.arange(nb, device=dev)[:, None], angle]                                    # [nb,T,2]
        eps = torch.cumsum(torch.randn(nb, T, device=dev), 1) * (0.25 / PX_NM * 2 * math.pi / 2.07)   # drift (rad)
        ph = phi.gather(1, steps) + eps; mm = m.gather(1, steps); rr = rel.gather(1, steps)
        patt = torch.stack([kk[..., 0], kk[..., 1], ph, mm, rr], -1)                             # [nb,T,5]
        # estimate errors
        pe = patt.clone()
        e_ang = torch.zeros(nb, 2, device=dev); ramp = torch.zeros(nb, 2, 2, device=dev)
        if not exact_est:
            e_ang = torch.randn(nb, 2, device=dev) * 0.06
            pe[..., 2] = ph + e_ang.gather(1, angle) + torch.randn(nb, T, device=dev) * 0.01 - eps + eps.mean(1, keepdim=True)
            pe[..., 3] = (mm + torch.randn(nb, 1, device=dev) * 0.02).clamp(0, 1)
            pe[..., 4] = rr * (1 + torch.randn(nb, T, device=dev) * 0.02)
            # linear phase-field ramp error across the FOV: d(phase) = g . (r - centre), |g| <= 0.08 rad / (H / 2)
            ramp = (g(nb, 2, 2) - 0.5) * 2 * 0.08 / (H / 2)
            gr = ramp[torch.arange(nb, device=dev)[:, None], angle]                               # [nb,T,2]
            # phase(r) error = gr . (r - c): absorb as a k error and a phase offset
            cx, cy = (W - 1) / 2, (H - 1) / 2
            pe[..., 0] = kk[..., 0] + gr[..., 0]; pe[..., 1] = kk[..., 1] + gr[..., 1]
            pe[..., 2] = pe[..., 2] + gr[..., 0] * cx + gr[..., 1] * cy
        # common per-angle error = translation of the pattern (errors are measured in the estimated-pattern frame)
        Km = k
        shift = torch.linalg.solve(Km, e_ang[..., None])[..., 0]
        return dict(patt=patt, patt_est=pe, steps=steps, angle=angle, k=k, shift_est=shift)

    # ----------------------------------------------------------------------------------------------- background
    def _env_raw(self, x, y):
        p = self.p
        dx, dy = x - p['env_x0'], y - p['env_y0']
        c, s_ = math.cos(p['env_th']), math.sin(p['env_th'])
        u = c * dx + s_ * dy; v = -s_ * dx + c * dy
        return torch.exp(-0.5 * (u ** 2 / p['env_s1'] ** 2 + v ** 2 / p['env_s2'] ** 2))

    def _env_grid(self):
        g = torch.arange(5, 115, device=dev, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(g, g, indexing='ij')
        return self._env_raw(gx, gy)

    def envelope(self, x, y):
        """(critic 37 #2, opt-in) illumination envelope factor at emitter positions (geometric mean 1 over the 5-115 crop);
        None when the env_* keys are absent"""
        if 'env_s1' not in self.p: return None
        return self._env_raw(x, y) / torch.exp(torch.log(self._env_grid()).mean())

    def background(self, nb, steps, bg=None, bg_sd=None, fluct=None, mode='struct', bg_hp=None):
        p = self.p; H, W, T = self.H, self.W, self.T
        b0 = torch.as_tensor(p['bg'] if bg is None else bg, device=dev, dtype=torch.float32).expand(nb).reshape(nb, 1, 1)
        sd = p['bg_sd'] if bg_sd is None else bg_sd
        if torch.is_tensor(sd): sd = sd.reshape(nb, 1, 1)
        fl = p['fluct'] if fluct is None else fluct
        if mode == 'flat':
            st = b0.expand(nb, H, W)
        else:
            # static structure: large-scale field (sigma 8 px, sd bg_sd) + PSF-scale fine field (sigma 2.5 px) scaled to the
            # measured high-pass sd (bgstats spatial_std_hp 0.28 / 0.54 ph, an upper bound: the temporal median keeps some
            # emitter light) -- critic review 26 #1: the old 0.6-weight fine field put ~2 ph/px of PSF-like structure in
            hp = p.get('bg_hp', 0.0) if bg_hp is None else bg_hp
            if torch.is_tensor(hp): hp = hp.reshape(nb, 1, 1)
            st = (b0 + sd * smooth_field(nb, H, W, 8.0) + hp * smooth_field(nb, H, W, 2.5)).clamp(min=0.3) * torch.ones(1, H, W, device=dev)
            if 'env_s1' in p and p.get('env_bg', 0.0) > 0:   # (critic 37 #2, opt-in) bg fraction following the envelope
                yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + 0.5, torch.arange(W, device=dev, dtype=torch.float32) + 0.5, indexing='ij')
                En = self._env_raw(xx, yy) / self._env_grid().mean()
                st = st * (1 - p['env_bg'] + p['env_bg'] * En)
        af = torch.tensor(p['bg_ang'], device=dev)[steps // 3] * (1 + 0.01 * torch.randn(nb, T, device=dev))
        stat = st[:, None] * af[..., None, None]
        if mode in ('flat', 'struct'):
            return stat, stat
        fl = torch.as_tensor(fl, device=dev, dtype=torch.float32).expand(nb).reshape(nb, 1, 1, 1)
        gl = torch.zeros(nb, T, device=dev); F = torch.zeros(nb, T, H, W, device=dev)
        a1, a2 = 0.4, 0.5
        gl[:, 0] = torch.randn(nb, device=dev); F[:, 0] = smooth_field(nb, H, W, 4.3)
        for t in range(1, T):
            gl[:, t] = a1 * gl[:, t - 1] + math.sqrt(1 - a1 ** 2) * torch.randn(nb, device=dev)
            F[:, t] = a2 * F[:, t - 1] + math.sqrt(1 - a2 ** 2) * smooth_field(nb, H, W, 4.3)
        bt = stat * (1 + 0.027 * fl * gl[..., None, None]) + 1.8 * fl * F
        return bt.clamp(min=0.3), stat

    # ----------------------------------------------------------------------------------------------- render
    def scene(self, nb, em=None, bgmode='fluct', exact_est=False, emitter_var=True, cam_err=None, psf_var=None, **kw):
        cam_err = emitter_var if cam_err is None else cam_err
        psf_var = emitter_var if psf_var is None else psf_var
        H, W, T = self.H, self.W, self.T
        p = self.p
        bgkw = {k_: kw.pop(k_) for k_ in ('bg', 'bg_sd', 'fluct', 'bg_hp') if k_ in kw}
        em = self.emitters(nb, **kw) if em is None else em
        fov, x, y, z, bright, t0, t1 = (em[k_] for k_ in ('fov', 'x', 'y', 'z', 'bright', 't0', 't1')); N = len(fov)
        fr = torch.arange(T, device=dev, dtype=torch.float32)
        onf = (torch.minimum(t1[:, None], fr[None] + 1) - torch.maximum(t0[:, None], fr[None])).clamp(0, 1)
        P = self.pattern(nb, exact_est)
        patt, steps = P['patt'], P['steps']
        pt_e = patt[fov]
        pfr, pfd = p.get('pf_rad', 0.0), p.get('pf_dep', 0.0)
        if (pfr > 0 or pfd > 0) and N:
            # (2026-11-10 realgen, opt-in; absent keys = unchanged, no extra draws) static per-FOV pattern phase / depth
            # field (the instrument's measured phase field, critic 35: random field of similar amplitude / scale, not the
            # measured map): per FOV and angle phase phi_a(r) = U(0, pf_rad) x unit smooth field (sigma U(pf_sig) px) and
            # depth factor f_a(r) = 1 + U(0, pf_dep) x unit smooth field (sigma pf_dsig px); il = rel (1 + m f cos(. - phi));
            # the estimate (patt_est) stays the global pattern
            m_ = 6; Hc, Wc = H + 2 * m_, W + 2 * m_
            sg = p.get('pf_sig', (5.0, 12.0)); sg = sg[0] + (sg[1] - sg[0]) * torch.rand(1).item()
            phF = torch.stack([smooth_field(nb, Hc, Wc, sg) for _ in range(2)], 1) * (pfr * torch.rand(nb, 2, 1, 1, device=dev))
            dfF = 1 + torch.stack([smooth_field(nb, Hc, Wc, p.get('pf_dsig', 25.0)) for _ in range(2)], 1) * (pfd * torch.rand(nb, 2, 1, 1, device=dev))
            gx = (x + m_).clamp(0, Wc - 1.001); gy = (y + m_).clamp(0, Hc - 1.001)
            x0, y0 = gx.floor().long(), gy.floor().long(); fx, fy = (gx - x0)[:, None], (gy - y0)[:, None]
            bil = lambda F_: (F_[fov, :, y0, x0] * (1 - fx) * (1 - fy) + F_[fov, :, y0, x0 + 1] * fx * (1 - fy)
                              + F_[fov, :, y0 + 1, x0] * (1 - fx) * fy + F_[fov, :, y0 + 1, x0 + 1] * fx * fy)       # [N,2]
            ang_e = (steps // 3)[fov]
            dph = bil(phF).gather(1, ang_e); dmf = bil(dfF).clamp(0.3, 1.5).gather(1, ang_e)
            il = pt_e[..., 4] * (1 + (pt_e[..., 3] * dmf).clamp(max=1.0) * torch.cos(pt_e[..., 0] * x[:, None] + pt_e[..., 1] * y[:, None] - pt_e[..., 2] - dph))
        else:
            il = pt_e[..., 4] * (1 + pt_e[..., 3] * torch.cos(pt_e[..., 0] * x[:, None] + pt_e[..., 1] * y[:, None] - pt_e[..., 2]))
        mult = torch.ones(N, T, device=dev)
        if emitter_var and N:
            ang = (steps // 3)[fov]
            ea = torch.exp(p['ang_sd'] * torch.randn(N, 2, device=dev)); ea = ea / ea.mean(1, keepdim=True)
            es = 1 + p['step_sd'] * torch.randn(N, 6, device=dev)
            mult = ea.gather(1, ang) * es.gather(1, steps[fov]) * (1 + p['frame_sd'] * torch.randn(N, T, device=dev))
            mult = mult.clamp(min=0.2)
        phot = bright[:, None] * il * onf * mult
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        mk = lambda q: q[..., 4, None, None] * (1 + q[..., 3, None, None] * torch.cos(q[..., 0, None, None] * xx + q[..., 1, None, None] * yy - q[..., 2, None, None]))
        sim = self.sim; R = sim.R; hr = R // 2; margin = 6
        img = torch.zeros(nb, T, H + 2 * (margin + hr) + 2, W + 2 * (margin + hr) + 2, device=dev)
        xr_, yr_ = x, y
        if N > 0 and p.get('ev_shift', 0.0) > 0:   # (2026-11-11, opt-in) per-event emission offset: PSF rendered at r + d, d ~ N(0, ev_shift nm)
            xr_ = x + p['ev_shift'] / PX_NM * torch.randn(N, device=dev); yr_ = y + p['ev_shift'] / PX_NM * torch.randn(N, device=dev)   # per axis; illumination / GT stay at r
        if N > 0:
            grp = torch.randint(0, len(self.luts) if psf_var else 1, (N,), device=dev)
            ang = isinstance(self.luts[0], tuple)
            psf = torch.zeros(N, R, R, device=dev)
            if ang: psf1 = torch.zeros(N, R, R, device=dev)       # angle-1 PSF (psf = angle 0)
            base = sim.lut
            for gi, lut in enumerate(self.luts):
                m_ = grp == gi
                if m_.any():
                    sim.lut = lut[0] if ang else lut; psf[m_] = sim.render_psf(xr_[m_], yr_[m_], z[m_])
                    if ang: sim.lut = lut[1]; psf1[m_] = sim.render_psf(xr_[m_], yr_[m_], z[m_])
            sim.lut = base
            # halo-amplitude jitter: a fraction h ~ U(0, halo) of each emitter's light in a broad Gaussian (sigma 4 px)
            hmax = p.get('halo', 0.0) if emitter_var else 0.0
            if hmax > 0:
                o_ = torch.arange(R, device=dev, dtype=torch.float32) - R // 2
                fx_, fy_ = xr_ - torch.round(xr_), yr_ - torch.round(yr_)
                gb = torch.exp(-0.5 * (((o_[None, None, :] - fx_[:, None, None]) ** 2 + (o_[None, :, None] - fy_[:, None, None]) ** 2) / 16.0))
                gb = gb / gb.sum((1, 2), keepdim=True) * psf.sum((1, 2), keepdim=True)
                hh = hmax * torch.rand(N, 1, 1, device=dev)
                psf = (1 - hh) * psf + hh * gb
                if ang: psf1 = (1 - hh) * psf1 + hh * gb
            if p.get('halo2', 0.0) > 0:    # broad halo (default off; see add_halo2)
                psf = add_halo2(psf, xr_, yr_, p['halo2'], p.get('halo2_sig', 4.0))
                if ang: psf1 = add_halo2(psf1, xr_, yr_, p['halo2'], p.get('halo2_sig', 4.0))
            cx = torch.round(xr_).long() + margin + 1; cy = torch.round(yr_).long() + margin + 1
            oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
            iy = (cy[:, None, None] + oy).reshape(N, -1); ix = (cx[:, None, None] + ox).reshape(N, -1)
            Wp, Hp = img.shape[-1], img.shape[-2]
            lin = fov[:, None, None] * T * Hp * Wp + torch.arange(T, device=dev)[None, :, None] * Hp * Wp + (iy * Wp + ix)[:, None, :]
            if ang:   # each frame's PSF follows its pattern angle (step // 3)
                sel_ = ((steps // 3)[fov] == 0)[..., None]
                val_ = phot[:, :, None] * torch.where(sel_, psf.reshape(N, 1, -1), psf1.reshape(N, 1, -1))
            else:
                val_ = phot[:, :, None] * psf.reshape(N, 1, -1)
            img.view(-1).index_add_(0, lin.reshape(-1), val_.reshape(-1))
        off = margin + hr + 1
        img = img[:, :, off:off + H, off:off + W]
        bgt, bst = self.background(nb, steps, mode=bgmode, **bgkw)
        mu = img + bgt
        Hm, Wm = self.rn_map.shape
        oy_ = torch.randint(0, Hm - H + 1, (nb,)).tolist(); ox_ = torch.randint(0, Wm - W + 1, (nb,)).tolist()
        rn = torch.stack([self.rn_map[y0:y0 + H, x0:x0 + W] for y0, x0 in zip(oy_, ox_)])
        flip = torch.rand(nb, device=dev) < 0.5
        rn = torch.where(flip[:, None, None], rn.flip(-1), rn)[:, None]
        e = torch.poisson(mu) + torch.randn_like(mu) * rn
        gain = sim.gain
        adu = torch.round(e / gain + sim.offset).clamp(0, 65535)
        if cam_err:   # residual offset error (static per pixel) and gain error of the conversion
            adu = adu - 0.3 * torch.randn(nb, 1, H, W, device=dev)
            gain = gain * (1 + 0.01 * torch.randn(nb, 1, 1, 1, device=dev))
        frames = (adu - sim.offset) * gain
        gt = dict(fov=fov, x=x, y=y, z=z, bright=bright, t0=t0, t1=t1, onf=onf, phot=phot)
        return dict(frames=frames, illum=mk(patt), illum_est=mk(P['patt_est']), patt=patt, patt_est=P['patt_est'],
                    shift_est=P['shift_est'], rn=rn, bg=bgt, bg_static=bst, steps=steps, gt=gt, nb=nb)
