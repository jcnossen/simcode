"""
Vectorial pupil-function PSF model (PyTorch, batched, differentiable).

Physics:
 - Emitter: freely rotating dipole (incoherent sum over x,y,z dipole orientations) at height z_e above the coverslip,
   in a medium with index n_med. Coverslip is assumed index-matched to the immersion (n_imm).
 - Fresnel transmission at the medium/coverslip interface (s and p), including supercritical-angle fluorescence
   (evanescent components for NA > n_med decay with emitter height).
 - Aplanatic apodization 1/sqrt(cos(theta_imm)).
 - Aberrations: Zernike phase (Noll index, in waves), optionally field dependent (supplied per emitter).
 - Defocus: stage/focus position f (focal plane position relative to the coverslip, in the sample's coordinate),
   phase = k0 * (n_med cos(theta_med) z_e - n_imm cos(theta_imm) f).
 - Image: chirp-z/matrix Fourier transform from pupil to an oversampled pixel grid, integrated over pixels.
 - Normalization: PSF integrates to 1 over the full image plane for an aberration-free emitter at z_e=0,
   i.e. `intensity` = detected photons for an emitter at the coverslip.

Units: micrometers.
"""
import math
import torch


def noll_to_nm(j):
    """Noll index -> (n, m); m>0 cosine, m<0 sine term."""
    n, j1 = 0, j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def zernike(j, rho, phi):
    """Noll-normalized Zernike polynomial Z_j(rho, phi) (rms = 1 over the unit disk)."""
    n, m = noll_to_nm(j)
    am = abs(m)
    R = torch.zeros_like(rho)
    for s in range((n - am) // 2 + 1):
        c = (-1) ** s * math.factorial(n - s) / (
            math.factorial(s) * math.factorial((n + am) // 2 - s) * math.factorial((n - am) // 2 - s))
        R = R + c * rho ** (n - 2 * s)
    if m == 0:
        return math.sqrt(n + 1) * R
    f = torch.cos(am * phi) if m > 0 else torch.sin(am * phi)
    return math.sqrt(2 * (n + 1)) * R * f


class VectorialPSF(torch.nn.Module):
    def __init__(self, roisize, pixelsize=0.10833, NA=1.49, n_imm=1.518, n_med=1.33, wavelength=0.68,
                 zernike_indices=tuple(range(5, 22)), npupil=64, oversample=3, device='cuda', fit_NA=False, fit_nmed=False):
        super().__init__()
        self.roisize = roisize
        self.pixelsize = pixelsize
        self.npupil = npupil
        self.os = oversample
        self.zidx = list(zernike_indices)
        self.n_imm, self.wavelength = n_imm, wavelength
        self.dev = torch.device(device)
        self.NA = torch.nn.Parameter(torch.tensor(float(NA), device=self.dev), requires_grad=fit_NA)
        self.n_med = torch.nn.Parameter(torch.tensor(float(n_med), device=self.dev), requires_grad=fit_nmed)
        self.zcoef = torch.nn.Parameter(torch.zeros(len(self.zidx), device=self.dev))  # waves (rms)
        # radial pupil amplitude (objective transmission): A(rho) = exp(sum_n c_n rho^(2n)), n=1..3
        self.apod = torch.nn.Parameter(torch.zeros(3, device=self.dev), requires_grad=False)
        # supercritical-annulus asymmetry (critic review 14): on rho > n_med/NA only, phase tilt (waves per unit radius) and
        # amplitude modulation along x and y: [t_x, t_y, a_x, a_y]
        self.saf = torch.nn.Parameter(torch.zeros(4, device=self.dev), requires_grad=False)
        # (2026-11-11 PSF ellipticity, critic 39; opt-in) detection-path Ex/Ey diattenuation d: the pupil field's Ex components
        # are scaled by sqrt(1 + d), Ey by sqrt(1 - d) (intensity transmission ratio (1 + d) / (1 - d)).  None / 0 = unchanged.
        # A plain attribute (float or tensor, may carry grad), not a parameter / buffer: state dicts are unaffected.
        self.diat = None

        # normalized pupil grid (unit disk), fixed; physical k scales with NA
        g = (torch.arange(npupil, device=self.dev) - (npupil - 1) / 2) / (npupil / 2)
        self.register_buffer('px', g[None, :].expand(npupil, npupil).clone())
        self.register_buffer('py', g[:, None].expand(npupil, npupil).clone())
        rho = torch.sqrt(self.px ** 2 + self.py ** 2)
        self.register_buffer('rho', rho)
        self.register_buffer('phi', torch.atan2(self.py, self.px))
        self.register_buffer('mask', (rho <= 1).float())
        Z = torch.stack([zernike(j, rho.clamp(max=1), self.phi) for j in self.zidx]) * self.mask
        self.register_buffer('Z', Z)  # [nz, N, N]

        # image sample positions (oversampled), relative to ROI center, in um
        ns = roisize * oversample
        u = (torch.arange(ns, device=self.dev) - (ns - 1) / 2) / oversample * pixelsize
        self.register_buffer('u', u)

    def pupil_fields(self):
        """Returns per-dipole/polarization pupil amplitude [6,N,N] (complex), cos terms and kx,ky."""
        NA = self.NA
        k0 = 2 * math.pi / self.wavelength
        s_imm = NA * self.rho / self.n_imm
        c_imm = torch.sqrt((1 - s_imm ** 2).clamp(min=1e-6))
        s_med = (NA * self.rho / self.n_med).to(torch.complex64)
        c_med = torch.sqrt(1 - s_med ** 2)  # complex; imag>0 for supercritical angles
        c_med = torch.where(c_med.imag < 0, -c_med, c_med)
        n1, n2 = self.n_med.to(torch.complex64), self.n_imm
        ts = 2 * n1 * c_med / (n1 * c_med + n2 * c_imm)
        tp = 2 * n1 * c_med / (n2 * c_med + n1 * c_imm)
        cphi, sphi = torch.cos(self.phi), torch.sin(self.phi)
        # far field of a dipole near an interface (Novotny & Hecht ch.10): t * (n2/n1) * cos2/cos1,
        # times aplanatic apodization 1/sqrt(cos2)
        apod = self.mask / torch.sqrt(c_imm) * (n2 / n1) * c_imm / c_med
        r2 = self.rho ** 2
        apod = apod * torch.exp(self.apod[0] * r2 + self.apod[1] * r2 ** 2 + self.apod[2] * r2 ** 3)
        ann = (self.rho * NA > self.n_med).float()
        apod = apod * (1 + ann * (self.saf[2] * self.px + self.saf[3] * self.py))
        apod = apod * torch.exp(1j * 2 * math.pi * ann * (self.saf[0] * self.px + self.saf[1] * self.py))
        # p-vector in medium (cos_med cos phi, cos_med sin phi, -sin_med), s-vector (-sin phi, cos phi, 0)
        # pupil field = tp (mu.p) (cos phi, sin phi) + ts (mu.s) (-sin phi, cos phi)
        Ex_x = tp * c_med * cphi * cphi + ts * sphi * sphi
        Ey_x = tp * c_med * cphi * sphi - ts * sphi * cphi
        Ex_y = tp * c_med * sphi * cphi - ts * cphi * sphi
        Ey_y = tp * c_med * sphi * sphi + ts * cphi * cphi
        Ex_z = -tp * s_med * cphi
        Ey_z = -tp * s_med * sphi
        P = torch.stack([Ex_x, Ey_x, Ex_y, Ey_y, Ex_z, Ey_z]) * apod
        d = getattr(self, 'diat', None)   # getattr: modules pickled before 2026-11-11 lack the attribute
        if d is not None and (torch.is_tensor(d) or d != 0):
            d = d if torch.is_tensor(d) else torch.tensor(float(d), device=self.dev)
            wx, wy = torch.sqrt(1 + d), torch.sqrt(1 - d)
            P = P * torch.stack([wx, wy, wx, wy, wx, wy]).to(P.real.dtype)[:, None, None]
        kx = k0 * NA * self.px
        ky = k0 * NA * self.py
        return P, k0 * self.n_med * c_med, k0 * self.n_imm * c_imm, kx, ky

    def forward(self, pos, focus, zcoef_offset=None, intensity=None, bg=None):
        """
        pos: [B,3] (x, y relative to ROI center in pixels, z_e emitter height in um)
        focus: [B] focal plane position in um (relative to coverslip)
        zcoef_offset: optional [B,nz] per-emitter Zernike offsets (field dependence)
        Returns [B, roisize, roisize] expected image (normalized PSF * intensity + bg if given).
        """
        B = pos.shape[0]
        P, kz_med, kz_imm, kx, ky = self.pupil_fields()
        zc = self.zcoef[None].expand(B, -1)
        if zcoef_offset is not None:
            zc = zc + zcoef_offset
        W = torch.einsum('bj,jnm->bnm', zc, self.Z) * (2 * math.pi)  # aberration phase
        x = pos[:, 0, None, None] * self.pixelsize
        y = pos[:, 1, None, None] * self.pixelsize
        ze = pos[:, 2, None, None]
        f = focus[:, None, None] if torch.is_tensor(focus) and focus.dim() > 0 else focus
        # complex phase; kz_med complex -> evanescent decay for z_e>0
        ph = torch.exp(1j * (W - kx * x - ky * y - kz_imm * f)) * torch.exp(1j * kz_med * ze)
        Pb = P[None] * ph[:, None]  # [B,6,N,N]
        # matrix Fourier transform: E(u,v) = sum_k P(kx,ky) exp(i (kx u + ky v))
        kx1 = kx[0]  # [N]
        ky1 = ky[:, 0]
        Fx = torch.exp(1j * self.u[:, None] * kx1[None, :])  # [ns, N]
        Fy = torch.exp(1j * self.u[:, None] * ky1[None, :])
        E = Fy @ Pb.to(torch.complex64) @ Fx.T  # [B,6,ns,ns]
        I = (E.real ** 2 + E.imag ** 2).sum(1) / 3
        # normalization: Parseval energy of aberration-free emitter at z=0 -> sum over plane = 1
        dk = (kx1[1] - kx1[0]) * (ky1[1] - ky1[0])
        energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        # continuous field E = sum(...) * dk/(2pi)^2, so |E|^2 gets that factor squared
        I = I * (dk / (2 * math.pi) ** 2) ** 2
        # pixel integration: average over oversampled subpixels * pixel area
        R, o = self.roisize, self.os
        I = I.reshape(B, R, o, R, o).mean((2, 4)) * self.pixelsize ** 2
        psf = I / energy
        if intensity is not None:
            psf = psf * intensity[:, None, None]
        if bg is not None:
            psf = psf + bg[:, None, None]
        return psf
