"""Coherent two-arm vectorial 4Pi PSF with polarization-split interference channels.

Built on torch_psf.VectorialPSF.pupil_fields().

Geometry (index-matched objectives, focus of both arms at z = 0):
  top arm    : the standard VectorialPSF field of a dipole mu at (x, y, z), pupil aberration W_top.
  bottom arm : the same objective type viewing from -z. In the bottom frame (rotation by pi about the x axis) the
               emitter is at (x, -y, -z) with dipole (mu_x, -mu_y, -mu_z); its image is mirrored back (y -> -y,
               E_y -> -E_y) by the detection optics so that it overlays the top image. Own aberration W_bot, lateral
               offset (dx, dy) px and axial offset dz (um) of the bottom arm, amplitude ratio a.
               (mirror='x' variant: mirror about the y axis instead.)
  channel c  : polarization pol_c (0 = E_x, 1 = E_y; pairs (0, 2) and (1, 3)), phase phi_c, transmission tau_c,
               coherence gamma (global, <= 1):
               I_c = tau_c [ |E_t|^2 + a^2 |E_b|^2 + 2 a gamma Re(E_t . conj(E_b) e^{i phi_c}) ]_pol_c
               summed over dipole components with the emission-dipole tensor M (isotropic = I/3).
Normalization: the sum over the 4 channels with tau = 1/4 each and the interference term averaged out integrates
to ~1 (in-focus, aberration-free); tau_c are relative channel weights (sum to 1 by construction -> 4 tau_c mean 1).

forward(pos [B,3] (x, y px rel. ROI centre, z um), M=None [3,3] or [B,3,3], phi_extra=None [B]) ->
  [B, 4, R, R] expected photons per photon emitted (times intensity if given).
"""
import math
import torch
from simcode.psf import VectorialPSF, gauss_blur_os, dipole_tensor

ZIDX = tuple(range(5, 22))


class PSF4Pi(torch.nn.Module):
    def __init__(self, R, px=0.128, NA=1.35, n=1.406, lam=0.60, npupil=48, os_=2, device='cpu', mirror='y',
                 zidx=ZIDX, pol=(0, 1, 0, 1)):
        super().__init__()
        self.R, self.os, self.mirror = R, os_, mirror
        mk = lambda: VectorialPSF(R, pixel_um=px, NA=NA, n_imm=n, n_med=n, wavelength_um=lam, zernike_indices=zidx,
                                  npupil=npupil, oversample=os_, device=device)
        self.top, self.bot = mk(), mk()
        self.px = px
        self.register_buffer('pol', torch.tensor(pol, device=device))
        self.phi = torch.nn.Parameter(torch.tensor([0.0, 1.5 * math.pi, math.pi, 0.5 * math.pi], device=device))
        self.tau_logit = torch.nn.Parameter(torch.zeros(4, device=device))
        self.a_log = torch.nn.Parameter(torch.tensor(0.0, device=device))          # bottom / top amplitude ratio
        self.gamma_logit = torch.nn.Parameter(torch.tensor(2.0, device=device))    # coherence
        self.d_bot = torch.nn.Parameter(torch.zeros(3, device=device))             # dx, dy (px), dz (um)
        self.logblur = torch.nn.Parameter(torch.tensor(math.log(0.1), device=device))
        self.meta = dict(R=R, px=px, NA=NA, n=n, lam=lam, npupil=npupil, os=os_, mirror=mirror, zidx=list(zidx), pol=list(pol))

    @property
    def tau(self):
        return torch.softmax(self.tau_logit, 0)

    def _fields(self, vp, pos, sgn_mu):
        """image fields [B, 3 (dipole), 2 (Ex, Ey), ns, ns] for the arm vp at pos (px, px, um)"""
        P, kz_med, kz_imm, kx, ky = vp.pupil_fields()
        B = pos.shape[0]
        W = torch.einsum('j,jnm->nm', vp.zcoef, vp.Z) * (2 * math.pi)
        x = pos[:, 0, None, None] * self.px
        y = pos[:, 1, None, None] * self.px
        ze = pos[:, 2, None, None]
        ph = torch.exp(1j * (W[None] - kx * x - ky * y)) * torch.exp(1j * kz_med * ze)
        Pb = P[None] * ph[:, None]
        pad = VectorialPSF.BLUR_PAD
        ns = (self.R + 2 * pad) * self.os
        u = (torch.arange(ns, device=P.device) - (ns - 1) / 2) / self.os * self.px
        Fx = torch.exp(1j * u[:, None] * kx[0][None, :])
        Fy = torch.exp(1j * u[:, None] * ky[:, 0][None, :])
        E = (Fy @ Pb.to(torch.complex64) @ Fx.T).reshape(B, 3, 2, ns, ns)
        E = E * sgn_mu.to(E.real.dtype)[None, :, None, None, None]
        dk = (kx[0, 1] - kx[0, 0]) * (ky[1, 0] - ky[0, 0])
        energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        return E * (dk / (2 * math.pi) ** 2), energy

    def forward(self, pos, M=None, phi_extra=None, intensity=None, bg=None):
        B = pos.shape[0]
        dev = pos.device
        one = torch.ones(3, device=dev)
        Et, energy = self._fields(self.top, pos, one)
        pb = pos + self.d_bot[None]
        if self.mirror == 'y':
            pb = torch.stack([pb[:, 0], -pb[:, 1], -pb[:, 2]], 1)
            Eb, _ = self._fields(self.bot, pb, torch.tensor([1.0, -1.0, -1.0], device=dev))
            Eb = torch.flip(Eb, dims=[-2])
            Eb = torch.stack([Eb[:, :, 0], -Eb[:, :, 1]], 2)
        else:
            pb = torch.stack([-pb[:, 0], pb[:, 1], -pb[:, 2]], 1)
            Eb, _ = self._fields(self.bot, pb, torch.tensor([-1.0, 1.0, -1.0], device=dev))
            Eb = torch.flip(Eb, dims=[-1])
            Eb = torch.stack([-Eb[:, :, 0], Eb[:, :, 1]], 2)
        a = torch.exp(self.a_log)
        g = torch.sigmoid(self.gamma_logit)
        Mt = dipole_tensor(M, None, dev)
        if Mt is None:
            Mt = torch.eye(3, device=dev) / 3
        if Mt.dim() == 2:
            Mt = Mt[None].expand(B, 3, 3)
        out = []
        for c in range(4):
            p = int(self.pol[c])
            et, eb = Et[:, :, p], Eb[:, :, p]                       # [B,3,ns,ns]
            ph = self.phi[c] + (phi_extra if phi_extra is not None else 0.0)
            ph = torch.as_tensor(ph, device=dev)
            e_ph = torch.exp(1j * ph.to(torch.float32))
            e_ph = e_ph.reshape(-1, 1, 1, 1) if e_ph.dim() > 0 else e_ph
            # coherent part: field f = et + a e^{i phi} eb, incoherent leak (1 - g) handled by mixing
            Mc = Mt.to(et.real.dtype)
            def quad(x, y):   # sum_ij M_ij Re[x_i conj(y_j)]
                yr = torch.einsum('bij,bjuv->biuv', Mc, y.real); yi = torch.einsum('bij,bjuv->biuv', Mc, y.imag)
                return (x.real * yr + x.imag * yi).sum(1)
            Itt, Ibb = quad(et, et), quad(eb, eb)
            ebp = eb * e_ph
            Itb = quad(et, ebp)
            I = Itt + a * a * Ibb + 2 * a * g * Itb
            out.append(I)
        I = torch.stack(out, 1) / (1 + a * a)                       # [B,4,ns,ns]
        bv = torch.exp(self.logblur)
        cov = torch.stack([bv, bv, torch.zeros_like(bv)])
        I = gauss_blur_os(I.reshape(B * 4, *I.shape[-2:]), cov, self.os, VectorialPSF.BLUR_PAD)
        R, o = self.R, self.os
        I = I.reshape(B, 4, R, o, R, o).mean((3, 5)) * self.px ** 2 / energy
        I = I * (4 * self.tau)[None, :, None, None] / 4 * 2      # channel weights; x2: one polarization per channel
        if intensity is not None:
            I = I * intensity[:, None, None, None]
        if bg is not None:
            I = I + (bg[:, :, None, None] if bg.dim() == 2 else bg[:, None, None, None])
        return I

    def state(self):
        return dict(meta=self.meta, zt=self.top.zcoef.tolist(), zb=self.bot.zcoef.tolist(), phi=self.phi.tolist(),
                    tau=self.tau.tolist(), a=float(torch.exp(self.a_log)), gamma=float(torch.sigmoid(self.gamma_logit)),
                    d_bot=self.d_bot.tolist(), blur_px2=float(torch.exp(self.logblur)))

    @classmethod
    def from_state(cls, s, device='cpu', **kw):
        m = s['meta']
        p = cls(m['R'] if 'R' not in kw else kw.pop('R'), px=m['px'], NA=m['NA'], n=m['n'], lam=m['lam'],
                npupil=m['npupil'], os_=m['os'], device=device, mirror=m['mirror'], zidx=tuple(m['zidx']),
                pol=tuple(m['pol']), **kw)
        with torch.no_grad():
            p.top.zcoef.copy_(torch.tensor(s['zt'])); p.bot.zcoef.copy_(torch.tensor(s['zb']))
            p.phi.copy_(torch.tensor(s['phi'])); p.tau_logit.copy_(torch.log(torch.tensor(s['tau'])))
            p.a_log.fill_(math.log(s['a'])); p.gamma_logit.fill_(math.log(s['gamma'] / max(1 - s['gamma'], 1e-6)))
            p.d_bot.copy_(torch.tensor(s['d_bot'])); p.logblur.fill_(math.log(s['blur_px2']))
            set_arm_terms(p, s)
        return p


def set_arm_terms(p, s):
    """optional per-arm terms of an in-situ state (key 'arms' = {'top' | 'bot': {NA, n_med, apod, saf, pupil_phase,
    pupil_amp}}; analysis/fourpi_plain/insitu4pi.py) copied into the arms' VectorialPSF; states without it are unchanged"""
    for nm in ('top', 'bot'):
        for k, v in (s.get('arms') or {}).get(nm, {}).items():
            t = getattr(getattr(p, nm), k)
            t.copy_(torch.as_tensor(v, dtype=t.dtype).reshape(t.shape).to(t.device))


class PSF4PiA(PSF4Pi):
    """Asymmetric-arm coherent 4Pi PSF (2026-11-11, plan item 2): each arm has its own objective (NA, immersion index) and
    the sample medium index n_med (DNA-PAINT buffer ~1.33). Default: top = oil 1.5 NA / 1.518 (near-TIRF, supercritical
    collection), bottom = silicone 1.35 NA / 1.406.
    With index mismatch, moving the emitter and moving the focus differ, so the signed z of an arm (+z top, -z bottom,
    as in PSF4Pi.forward) is interpreted by `zmode`:
      'emitter': the emitter moves in the medium:  phase = kz_med * (depth_arm + z)          (movies)
      'stage'  : the sample (bead at a fixed depth) moves through the focus: phase = kz_med * depth_arm - kz_imm * z
                 (bead z-stacks; z = stage displacement)
    depth_arm (um) = emitter distance from that arm's coverslip at z = 0 (fitted; sets the supercritical / evanescent part
    and the depth-dependent spherical aberration). Noll 4 (defocus) is fitted too (focus offset of each arm)."""
    def __init__(self, R, px=0.128, lam=0.60, npupil=48, os_=2, device='cpu', mirror='y', zidx=(4,) + ZIDX,
                 pol=(0, 1, 0, 1), NA_t=1.5, n_imm_t=1.518, NA_b=1.35, n_imm_b=1.406, n_med=1.33, zmode='emitter',
                 depth=(0.05, 2.0)):
        super().__init__(R, px=px, NA=NA_b, n=n_imm_b, lam=lam, npupil=npupil, os_=os_, device=device, mirror=mirror,
                         zidx=zidx, pol=pol)
        mk = lambda NA, ni: VectorialPSF(R, pixel_um=px, NA=NA, n_imm=ni, n_med=n_med, wavelength_um=lam, zernike_indices=zidx,
                                         npupil=npupil, oversample=os_, device=device)
        self.top, self.bot = mk(NA_t, n_imm_t), mk(NA_b, n_imm_b)
        self.zmode = zmode
        self.depth = torch.nn.Parameter(torch.tensor([float(depth[0]), float(depth[1])], device=device))
        self.meta.update(asym=True, NA_t=NA_t, n_imm_t=n_imm_t, NA_b=NA_b, n_imm_b=n_imm_b, n_med=n_med, zmode=zmode)

    def _fields(self, vp, pos, sgn_mu):
        P, kz_med, kz_imm, kx, ky = vp.pupil_fields()
        B = pos.shape[0]
        W = torch.einsum('j,jnm->nm', vp.zcoef, vp.Z) * (2 * math.pi)
        x = pos[:, 0, None, None] * self.px
        y = pos[:, 1, None, None] * self.px
        z = pos[:, 2, None, None]
        dep = self.depth[0] if vp is self.top else self.depth[1]
        if self.zmode == 'emitter':
            zph = torch.exp(1j * kz_med * (dep + z).clamp(min=0.0))
        else:
            zph = torch.exp(1j * kz_med * dep.clamp(min=0.0)) * torch.exp(-1j * kz_imm * z)
        ph = torch.exp(1j * (W[None] - kx * x - ky * y)) * zph
        Pb = P[None] * ph[:, None]
        pad = VectorialPSF.BLUR_PAD
        ns = (self.R + 2 * pad) * self.os
        u = (torch.arange(ns, device=P.device) - (ns - 1) / 2) / self.os * self.px
        Fx = torch.exp(1j * u[:, None] * kx[0][None, :])
        Fy = torch.exp(1j * u[:, None] * ky[:, 0][None, :])
        E = (Fy @ Pb.to(torch.complex64) @ Fx.T).reshape(B, 3, 2, ns, ns)
        E = E * sgn_mu.to(E.real.dtype)[None, :, None, None, None]
        dk = (kx[0, 1] - kx[0, 0]) * (ky[1, 0] - ky[0, 0])
        energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        return E * (dk / (2 * math.pi) ** 2), energy

    def state(self):
        s = super().state(); s['depth'] = self.depth.tolist(); return s

    @classmethod
    def from_state(cls, s, device='cpu', **kw):
        m = s['meta']
        p = cls(kw.pop('R', m['R']), px=m['px'], lam=m['lam'], npupil=m['npupil'], os_=m['os'], device=device,
                mirror=m['mirror'], zidx=tuple(m['zidx']), pol=tuple(m['pol']), NA_t=m['NA_t'], n_imm_t=m['n_imm_t'],
                NA_b=m['NA_b'], n_imm_b=m['n_imm_b'], n_med=m['n_med'], zmode=kw.pop('zmode', m['zmode']))
        with torch.no_grad():
            p.top.zcoef.copy_(torch.tensor(s['zt'])); p.bot.zcoef.copy_(torch.tensor(s['zb']))
            p.phi.copy_(torch.tensor(s['phi'])); p.tau_logit.copy_(torch.log(torch.tensor(s['tau'])))
            p.a_log.fill_(math.log(s['a'])); p.gamma_logit.fill_(math.log(s['gamma'] / max(1 - s['gamma'], 1e-6)))
            p.d_bot.copy_(torch.tensor(s['d_bot'])); p.logblur.fill_(math.log(s['blur_px2']))
            p.depth.copy_(torch.tensor(s['depth']))
            set_arm_terms(p, s)
        return p


def psf_from_state(s, device='cpu', **kw):
    """PSF4Pi or PSF4PiA from a stored state"""
    return (PSF4PiA if s['meta'].get('asym') else PSF4Pi).from_state(s, device=device, **kw)
