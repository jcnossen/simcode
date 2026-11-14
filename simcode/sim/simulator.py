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
from simcode.psf import has_angle_psf, render


class Simulator:
    def __init__(self, calib_path, H=64, W=64, T=12, device='cuda', psf_roi=21, psf_lut_z=21, psf_lut_sub=32,
                 zmax=0.2, tirf_depth=None):
        from simcode.config import resolve
        calib_path = resolve(calib_path)
        self.c = json.load(open(calib_path))
        self.H, self.W, self.T = H, W, T
        self.dev = torch.device(device)
        self.zmax = zmax
        self.tirf = tirf_depth or self.c['tirf_depth_um']
        self.R = psf_roi
        # calibrated PSF (torch_psf.PSF.load of the calib json: VectorialPSF with the opt-in blur / dipole terms, angle
        # average; a GaussianPSF for 'psf_gauss_sigma_um')
        self.calib_path = calib_path
        self.psf = spsf.PSF.load(calib_path, roisize=psf_roi, oversample=3, device=device)
        self.focus = self.c['focus_um']
        # PSF lookup table over (z, subpixel x, subpixel y) for fast rendering; bilinear in subpixel, linear in z
        self.lz, self.ls = psf_lut_z, psf_lut_sub
        zs = torch.linspace(0, zmax, psf_lut_z, device=self.dev)
        sub = (torch.arange(psf_lut_sub + 1, device=self.dev) / psf_lut_sub) - 0.5     # -0.5 .. 0.5 inclusive
        Z, SY, SX = torch.meshgrid(zs, sub, sub, indexing='ij')
        pos = torch.stack([SX.reshape(-1), SY.reshape(-1), Z.reshape(-1)], 1)
        with torch.no_grad():
            lut = torch.cat([render(self.psf, p) for p in pos.split(1024)])
        self.lut = lut.reshape(psf_lut_z, psf_lut_sub + 1, psf_lut_sub + 1, psf_roi, psf_roi)
        # (2026-11-11, opt-in) per-pattern-angle PSF (calib 'psf_dipole_M_angle' / 'psf_blur_angle_px2'): one LUT per angle;
        # self.lut = the exact average of the two angle LUTs (pattern-free consumers: templates, sums); self.psf = the
        # angle-averaged model of torch_psf.PSF.load(angle=None) (mean dipole tensor: exact; mean blur covariance: approximate)
        self.lut_angle = None
        if has_angle_psf(self.c):
            self.lut_angle = []
            for k in range(2):
                pk = spsf.PSF.load(calib_path, roisize=psf_roi, oversample=3, device=device, angle=k)
                with torch.no_grad():
                    la = torch.cat([render(pk, p) for p in pos.split(1024)])
                self.lut_angle.append(la.reshape(psf_lut_z, psf_lut_sub + 1, psf_lut_sub + 1, psf_roi, psf_roi))
            self.lut = 0.5 * (self.lut_angle[0] + self.lut_angle[1])   # exact angle average (equal light per angle)
        self.zs = zs
        cam = self.c['camera']
        self.gain, self.offset, self.rn = cam['gain_e_per_adu'], cam['offset_adu'], cam['readnoise_e']
        self.rn_map = None
        if cam.get('maps'):
            import numpy as _np, os as _os
            mp = cam['maps'] if _os.path.isabs(cam['maps']) else _os.path.join(_os.path.dirname(_os.path.abspath(calib_path)), cam['maps'])
            self.rn_map = torch.tensor(_np.load(mp)['readnoise_e'], device=self.dev)   # per-pixel read noise [e]
        p = self.c['pattern']
        self.k0 = torch.tensor(p['k_rad_per_px'], device=self.dev, dtype=torch.float32)   # [2,2]
        self.depth0 = torch.tensor(p['depth'], device=self.dev)

    # ------------------------------------------------------------------
    def render_psf_angles(self, x, y, z):
        """(per-angle PSF) [2,N,R,R] rendered with lut_angle[0 / 1]; [1,N,R,R] with self.lut otherwise"""
        if self.lut_angle is None:
            return self.render_psf(x, y, z)[None]
        base = self.lut; out = []
        for la in self.lut_angle:
            self.lut = la; out.append(self.render_psf(x, y, z))
        self.lut = base
        return torch.stack(out)

    def render_psf(self, x, y, z):
        """Interpolated PSF images [N,R,R] for emitters with subpixel offsets relative to their pixel center."""
        fx, fy = x - torch.round(x), y - torch.round(y)                     # [-0.5, 0.5]
        tx = (fx + 0.5) * self.ls; ty = (fy + 0.5) * self.ls
        ix = tx.floor().long().clamp(0, self.ls - 1); iy = ty.floor().long().clamp(0, self.ls - 1)
        wx = (tx - ix)[:, None, None]; wy = (ty - iy)[:, None, None]
        tz = (z / self.zmax).clamp(0, 1) * (self.lz - 1)
        iz = tz.floor().long().clamp(0, self.lz - 2); wz = (tz - iz)[:, None, None]
        def at(dz, dy, dx):
            return self.lut[iz + dz, iy + dy, ix + dx]
        out = 0
        for dz, w_z in ((0, 1 - wz), (1, wz)):
            out = out + w_z * ((1 - wy) * ((1 - wx) * at(dz, 0, 0) + wx * at(dz, 0, 1)) +
                               wy * ((1 - wx) * at(dz, 1, 0) + wx * at(dz, 1, 1)))
        return out

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
            if self.lut_angle is None:
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
