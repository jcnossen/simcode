"""
GPU simulator of modulated (DMD-SIMFLUX) DNA-PAINT movies from the calibrated physical model.

Per batch: nb fields of view of HxW pixels and T frames (T multiple of 6, frame f has pattern step (f0+f)%6).
 - Binding events: Poisson in time/space; each event = emitter at a fixed position with on-interval [t0, t1]
   (exponential duration); per-frame on-fraction = exact overlap with the frame. Positions uniform, or on
   random filaments (structured data, overlapping emitters).
 - Emitter z in [0, zmax], TIRF excitation exp(-z/d). Brightness lognormal.
 - Illumination: relint_s * (1 + m_s cos(k_a . r - phi_s)), randomized per FOV around the calibration.
 - Background: smooth random field x per-angle factor; camera: Poisson + Gaussian read noise, gain/offset,
   ADU rounding; output is converted back to photons exactly as real data would be ((ADU - offset) * gain).
Ground truth: emitter table with per-frame photons (= brightness * illumination * on-fraction).
"""
import json, math
import numpy as np
import torch
from simcode import psf as spsf
from simcode.psf import has_angle_psf


class Simulator:
    def __init__(self, calib_path, H=64, W=64, T=12, device='cuda', psf_roi=21, zmax=0.2, tirf_depth=None, zmin=0.0):
        from simcode.config import resolve
        calib_path = resolve(calib_path)
        self.c = spsf.read_calib(calib_path)          # calib json or psf.zip
        self.H, self.W, self.T = H, W, T
        self.dev = torch.device(device)
        # emitter heights zmin .. zmax um the simulator draws / renders (z is clamped to it)
        self.zmin, self.zmax = zmin, zmax
        self.tirf = tirf_depth or self.c['tirf_depth_um']
        self.R = psf_roi
        # calibrated PSF (2026-11-19): THE PSF renderer, simcode.psf.calib_psf = the integrated cubic B-spline of the calibration
        # (torch_psf IntegratedSplinePSF; read from a psf.zip that carries it, else built from the vectorial model on load; a
        # GaussianPSF for 'psf_gauss_sigma_um'). self.psf = the angle average (the mean of the angle images: pattern-free consumers,
        # templates, sums); self.psf_angle = [angle 0, angle 1] for a per-pattern-angle calibration (frame f: angle f % 6 // 3)
        self.calib_path = calib_path
        # (heights below the coverslip, zmin < 0, or above SPLINE's: a spline over the wider range, built from the vectorial model)
        self.psf = spsf.calib_psf(calib_path, psf_roi, device=device, z_range=(zmin, zmax))
        self.psf_angle = ([spsf.calib_psf(calib_path, psf_roi, angle=k, device=device, z_range=(zmin, zmax)) for k in range(2)]
                          if has_angle_psf(self.c) else None)
        self.focus = self.c['focus_um']
        cam = self.c['camera']
        self.gain, self.offset, self.rn = cam['gain_e_per_adu'], cam['offset_adu'], cam['readnoise_e']
        self.rn_map = None
        if cam.get('maps'):
            self.rn_map = torch.tensor(spsf.camera_maps(calib_path, self.c)['readnoise_e'], device=self.dev)   # per-pixel read noise [e]
        p = self.c['pattern']
        self.k0 = torch.tensor(p['k_rad_per_px'], device=self.dev, dtype=torch.float32)   # [2,2]
        self.depth0 = torch.tensor(p['depth'], device=self.dev)

    # ------------------------------------------------------------------
    def render_psf_angles(self, x, y, z, psf=None):
        """(per-angle PSF) [2,N,R,R] of the two pattern angles (psf: a pair of models, default self.psf_angle); [1,N,R,R] of
        self.psf for a calibration without per-angle terms"""
        pa = self.psf_angle if psf is None else psf
        if pa is None:
            return self.render_psf(x, y, z)[None]
        return torch.stack([self.render_psf(x, y, z, p) for p in pa])

    def render_psf(self, x, y, z, psf=None):
        """PSF images [N,R,R] (psf: a model, default self.psf) of emitters at (x, y) px, height z um, on the R x R ROI centred
        on their nearest pixel (round(x), round(y)); z clamped to zmin .. zmax"""
        p = self.psf if psf is None else psf
        pos = torch.stack([x - torch.round(x), y - torch.round(y), z.clamp(self.zmin, self.zmax)], 1)
        with torch.no_grad():
            return p.render(pos)

    # ------------------------------------------------------------------
    def sample(self, nb, density=(0.0005, 0.012), ontime=(2.0, 20.0), photons=(150, 1500), bg=(1.0, 30.0),
               depth=(0.5, 1.0), structured_frac=0.5, return_adu=False, phase_err=(0.0, 0.05), depth_err=0.1):
        dev, H, W, T = self.dev, self.H, self.W, self.T
        g = lambda *s: torch.rand(*s, device=dev)
        margin = 6
        # --- per-FOV parameters (log-uniform where it spans decades)
        lu = lambda lo, hi, n: torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * g(n))
        dens = lu(*density, nb)                 # mean emitters ON per frame per pixel
        tau = lu(*ontime, nb)                   # mean on-time [frames]
        Bmed = lu(*photons, nb)                 # median photons per frame (full on, mean illumination, z=0)
        # number of events overlapping the window [0, T]: rate * area * (T + tau) / tau ... events start in [-4tau, T]
        area = (H + 2 * margin) * (W + 2 * margin)
        t_lo = -4 * tau
        n_ev = torch.poisson(dens * area * (T - t_lo) / tau).long()
        fov = torch.repeat_interleave(torch.arange(nb, device=dev), n_ev)
        N = len(fov)
        t0 = t_lo[fov] + (T - t_lo[fov]) * g(N)
        t1 = t0 + -torch.log(g(N).clamp(min=1e-9)) * tau[fov]
        keep = t1 > 0
        fov, t0, t1 = fov[keep], t0[keep], t1[keep]; N = len(fov)
        # positions: uniform or on random filaments
        x = -margin + (W + 2 * margin) * g(N); y = -margin + (H + 2 * margin) * g(N)
        structured = g(nb) < structured_frac
        if structured.any():
            nfil = 6
            p0 = torch.stack([-margin + (W + 2 * margin) * g(nb, nfil), -margin + (H + 2 * margin) * g(nb, nfil)], -1)
            ang = g(nb, nfil) * math.pi
            fi = torch.randint(0, nfil, (N,), device=dev)
            s = (g(N) - 0.5) * 2 * (W + H)
            a_ = ang[fov, fi]
            xs = p0[fov, fi, 0] + s * torch.cos(a_) + torch.randn(N, device=dev) * 0.1
            ys = p0[fov, fi, 1] + s * torch.sin(a_) + torch.randn(N, device=dev) * 0.1
            inside = (xs > -margin) & (xs < W + margin) & (ys > -margin) & (ys < H + margin)
            use = structured[fov] & inside & (g(N) < 0.8)
            x = torch.where(use, xs, x); y = torch.where(use, ys, y)
        z = self.zmax * g(N) ** 1.5
        bright = Bmed[fov] * torch.exp(0.35 * torch.randn(N, device=dev)) * torch.exp(-z / self.tirf)
        # per-frame on fraction
        fr = torch.arange(T, device=dev, dtype=torch.float32)
        onf = (torch.minimum(t1[:, None], fr[None] + 1) - torch.maximum(t0[:, None], fr[None])).clamp(0, 1)   # [N,T]

        # --- illumination pattern per FOV (randomized around calibration)
        step0 = torch.randint(0, 6, (nb,), device=dev)
        steps = (step0[:, None] + torch.arange(T, device=dev)[None]) % 6              # [nb,T]
        rot = (g(nb, 2) - 0.5) * math.radians(10)
        scale = 1 + (g(nb, 2) - 0.5) * 0.06
        k = torch.stack([self.k0[:, 0] * torch.cos(rot) - self.k0[:, 1] * torch.sin(rot),
                         self.k0[:, 0] * torch.sin(rot) + self.k0[:, 1] * torch.cos(rot)], -1) * scale[..., None]  # [nb,2,2]
        phi = (g(nb, 2) * 2 * math.pi)[:, :, None] + torch.tensor([0, 2, 4], device=dev) * math.pi / 3 \
              + (g(nb, 2, 3) - 0.5) * 0.4                                              # [nb,2,3]
        phi = phi.reshape(nb, 6)
        m = depth[0] + (depth[1] - depth[0]) * g(nb, 1) + (g(nb, 6) - 0.5) * 0.1
        m = m.clamp(0, 1)
        rel = 1 + (g(nb, 6) - 0.5) * 0.2   # +-10% per step (real step relint up to ~7% off, angle ratio ~4-9%)
        angle = steps // 3
        kk = k[torch.arange(nb, device=dev)[:, None], angle]                            # [nb,T,2]
        ph = phi.gather(1, steps); mm = m.gather(1, steps); rr = rel.gather(1, steps)   # [nb,T]
        # illumination at emitters
        kre = (kk[fov] * torch.stack([x, y], -1)[:, None]).sum(-1)                       # [N,T]
        illum_e = rr[fov] * (1 + mm[fov] * torch.cos(kre - ph[fov]))
        phot = bright[:, None] * illum_e * onf                                           # [N,T] photons per frame
        # illumination maps (network input)
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        kr_map = kk[..., 0, None, None] * xx + kk[..., 1, None, None] * yy               # [nb,T,H,W]
        illum = rr[..., None, None] * (1 + mm[..., None, None] * torch.cos(kr_map - ph[..., None, None]))
        # imperfect pattern knowledge (what the network gets): common phase error per angle (drift-like, shared by
        # the 3 steps) + small per-step error, depth error
        sig = phase_err[0] + (phase_err[1] - phase_err[0]) * g(nb, 1)
        e_ang = torch.randn(nb, 2, device=dev) * sig
        e_ph = e_ang.gather(1, angle) + torch.randn(nb, T, device=dev) * 0.03
        e_m = (mm * (1 + (g(nb, 1) - 0.5) * 2 * depth_err)).clamp(0, 1)
        illum_est = rr[..., None, None] * (1 + e_m[..., None, None] * torch.cos(kr_map - (ph + e_ph)[..., None, None]))
        # a common per-angle phase error e_a is a global translation of the pattern: an emitter at r looks like one at
        # r + D in the estimated pattern frame, with k_a . D = e_a  ->  D = K^-1 e  (training targets use this frame)
        shift_est = torch.linalg.solve(k, e_ang[..., None])[..., 0]          # [nb,2] px (x,y)
        # pattern parameters per frame (kx, ky, phase, depth, relint): true and as estimated (analytic, no resampling)
        patt = torch.stack([kk[..., 0], kk[..., 1], ph, mm, rr], -1)
        patt_est = torch.stack([kk[..., 0], kk[..., 1], ph + e_ph, e_m, rr], -1)

        # --- render
        R = self.R; hr = R // 2
        img = torch.zeros(nb, T, H + 2 * (margin + hr) + 2, W + 2 * (margin + hr) + 2, device=dev)
        if N > 0:
            psf = self.render_psf(x, y, z)                                                # [N,R,R]
            cx = torch.round(x).long() + margin + 1; cy = torch.round(y).long() + margin + 1
            oy, ox = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing='ij')
            iy = (cy[:, None, None] + oy).reshape(N, -1); ix = (cx[:, None, None] + ox).reshape(N, -1)
            Wp = img.shape[-1]; Hp = img.shape[-2]
            lin = (fov[:, None, None] * T * Hp * Wp + torch.arange(T, device=dev)[None, :, None] * Hp * Wp
                   + (iy * Wp + ix)[:, None, :])                                          # [N,T,R*R]
            if self.psf_angle is None:
                val = phot[:, :, None] * psf.reshape(N, 1, -1)
            else:   # each frame's PSF follows its pattern angle
                psfA = self.render_psf_angles(x, y, z).reshape(2, N, 1, -1)
                ang_e = angle[fov][..., None]                                             # [N,T,1]
                val = phot[:, :, None] * torch.where(ang_e == 0, psfA[0], psfA[1])
            img.view(-1).index_add_(0, lin.reshape(-1), val.reshape(-1))
        off = margin + hr + 1
        img = img[:, :, off:off + H, off:off + W]
        # background: smooth field * per-angle factor
        b0 = lu(*bg, nb)
        low = torch.nn.functional.interpolate(g(nb, 1, 4, 4), size=(H, W), mode='bicubic', align_corners=False)
        bgmap = b0[:, None, None, None] * (0.7 + 0.6 * low).clamp(min=0.2)
        angf = 1 + (g(nb, 2) - 0.5) * 0.3
        bgf = angf.gather(1, angle)[..., None, None]
        mu = img + bgmap * bgf
        # camera
        if self.rn_map is not None:   # random crops / flips of the measured per-pixel read-noise map (sCMOS)
            Hm, Wm = self.rn_map.shape
            oy = torch.randint(0, Hm - H + 1, (nb,)).tolist(); ox = torch.randint(0, Wm - W + 1, (nb,)).tolist()
            rn = torch.stack([self.rn_map[y0:y0 + H, x0:x0 + W] for y0, x0 in zip(oy, ox)])
            flip = torch.rand(nb, device=dev) < 0.5
            rn = torch.where(flip[:, None, None], rn.flip(-1), rn)[:, None]
        else:
            rn = self.rn
        e = torch.poisson(mu) + torch.randn_like(mu) * rn
        adu = torch.round(e / self.gain + self.offset).clamp(0, 65535)
        frames = (adu - self.offset) * self.gain
        gt = dict(fov=fov, x=x, y=y, z=z, phot=phot, onf=onf, bright=bright)
        out = dict(frames=frames, illum=illum, illum_est=illum_est, shift_est=shift_est, rn=rn, patt=patt, patt_est=patt_est, bg=bgmap * bgf, gt=gt, steps=steps, k=k, phi=phi, depth=m, rel=rel)
        if return_adu: out['adu'] = adu
        return out
