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


def gauss_blur_os(I, cov, os_, pad):
    """Anisotropic Gaussian blur of oversampled intensity images I [B, n, n] (n = (R + 2 pad) os) with covariance
    cov = [sxx, syy, sxy] (px^2; [3] or [B, 3]), by multiplication with exp(-2 pi^2 f^T C f) in the Fourier domain (exact for
    the band-limited intensity; os >= 2 samples it above Nyquist), then cropped by `pad` px per side -> [B, n - 2 pad os, ...].
    The padding keeps the circular wrap out of the ROI. A negative-definite cov (deblur) is allowed but its gain is capped."""
    B, n = I.shape[0], I.shape[-1]
    cov = torch.as_tensor(cov, device=I.device, dtype=torch.float32)
    cov = cov.reshape(-1, 3)
    f = torch.fft.fftfreq(n, d=1.0 / os_, device=I.device)            # cycles / px
    fr = torch.fft.rfftfreq(n, d=1.0 / os_, device=I.device)
    fy, fx = f[:, None], fr[None, :]
    q = (cov[:, 0, None, None] * fx ** 2 + cov[:, 1, None, None] * fy ** 2 + 2 * cov[:, 2, None, None] * fx * fy)
    H = torch.exp((-2 * math.pi ** 2 * q).clamp(max=3.0))
    J = torch.fft.irfft2(torch.fft.rfft2(I) * H, s=(n, n))
    c = pad * os_
    return J[:, c:n - c, c:n - c] if c > 0 else J


def dipole_tensor(M=None, w=None, dev=None):
    """emission-dipole second-moment tensor (trace 1) from M ([3,3] or [B,3,3]) or diagonal weights w ([3], mean 1); None if both None"""
    if M is not None:
        return torch.as_tensor(M, device=dev, dtype=torch.float32)
    if w is not None:
        return torch.diag_embed(torch.as_tensor(w, device=dev, dtype=torch.float32)) / 3
    return None


def photosel_tensor(gamma, e_inplane, eps=0.0, k_dir=None, ck=0.153):
    """M = (1 - gamma) I / 3 + gamma K, K = excitation polarisation tensor (trace 1):
    K = (1 - eps) e e^T + eps (ck kk^T + (1 - ck) zz^T) (eps = p-polarised fraction of the evanescent excitation intensity;
    ck = |E_x|^2 / |E|^2 of the evanescent p wave = (n_eff^2 - n_med^2) / (2 n_eff^2 - n_med^2), 0.153 at n_eff 1.47, n_med 1.33).
    e_inplane, k_dir: in-plane unit 2-vectors (x = column, y = row)."""
    dev = gamma.device if torch.is_tensor(gamma) else None
    e = torch.zeros(3, device=dev); e[:2] = torch.as_tensor(e_inplane, device=dev, dtype=torch.float32)
    K = e[:, None] * e[None, :]
    if k_dir is not None:
        kk = torch.zeros(3, device=dev); kk[:2] = torch.as_tensor(k_dir, device=dev, dtype=torch.float32)
        z = torch.zeros(3, device=dev); z[2] = 1
        Kp = ck * kk[:, None] * kk[None, :] + (1 - ck) * z[:, None] * z[None, :]
        K = (1 - eps) * K + eps * Kp
    return (1 - gamma) * torch.eye(3, device=dev) / 3 + gamma * K


ANGLE_KEYS = ('psf_dipole_M_angle', 'psf_blur_angle_px2')


def has_angle_psf(c):
    """True if the calib dict c has a per-pattern-angle PSF (angle = step // 3 = frame % 6 // 3)"""
    return any(c.get(k) is not None for k in ANGLE_KEYS)


def set_psf_extras(psf, c, angle=None):
    """Opt-in PSF terms of a calib json dict c -> attributes of a VectorialPSF (absent keys: unchanged). Returns psf.
    Global keys: 'psf_blur_px2' [sxx, syy, sxy] px^2 (image-plane blur), 'psf_dipole_M' (3 x 3, trace 1) / 'psf_dipole_w' [wx, wy, wz]
    (emission-dipole tensor), 'saf' [tx, ty, ax, ay] (supercritical-annulus tilts).
    Per pattern angle: 'psf_dipole_M_angle' [2][3][3], 'psf_blur_angle_px2' [2][3] (total blur of that angle, replaces the
    global blur). angle = 0 / 1 selects that angle's PSF; angle None = the angle-averaged PSF: the mean dipole tensor (exact for
    equal light per angle, the image is linear in M) and the mean blur covariance (an approximation)."""
    dv = psf.u.device
    t = lambda v: torch.tensor(v, device=dv, dtype=torch.float32)
    if c.get('psf_blur_px2') is not None:
        psf.blur = t([float(v) for v in c['psf_blur_px2']])
    if c.get('psf_dipole_M') is not None:
        psf.dipM = t(c['psf_dipole_M'])
    if c.get('psf_dipole_w') is not None:
        psf.dipw = t([float(v) for v in c['psf_dipole_w']])
    if c.get('saf') is not None:
        with torch.no_grad(): psf.saf.copy_(t([float(v) for v in c['saf']]))
    if c.get('psf_dipole_M_angle') is not None:
        M = t(c['psf_dipole_M_angle'])
        psf.dipM = M.mean(0) if angle is None else M[angle]
    if c.get('psf_blur_angle_px2') is not None:
        Bc = t(c['psf_blur_angle_px2'])
        psf.blur = Bc.mean(0) if angle is None else Bc[angle]
    if c.get('psf_gauss_sigma_um') is not None:   # (2026-11-12, opt-in) pixel-integrated Gaussian PSF instead of the vectorial model
        psf.gauss_sigma_um = float(c['psf_gauss_sigma_um'])
    return psf


def angle_kwargs(c, dev):
    """per-pattern-angle forward() kwargs [kw0, kw1] (dipM / blur of that angle) of a calib dict, or None if it has none"""
    if not has_angle_psf(c):
        return None
    out = []
    for k in range(2):
        kw = {}
        if c.get('psf_dipole_M_angle') is not None: kw['dipM'] = torch.tensor(c['psf_dipole_M_angle'][k], device=dev, dtype=torch.float32)
        if c.get('psf_blur_angle_px2') is not None: kw['blur'] = torch.tensor(c['psf_blur_angle_px2'][k], device=dev, dtype=torch.float32)
        out.append(kw)
    return out


def copy_psf_extras(dst, src):
    """copy the opt-in attributes (blur, diat, dipw; saf is a parameter, copied by the callers / here) of VectorialPSF src
    to dst (the LUT / CRLB copies of a calibrated PSF)"""
    with torch.no_grad(): dst.saf.copy_(src.saf)
    for k in ('blur', 'diat', 'dipw', 'dipM', 'gauss_sigma_um'):
        v = getattr(src, k, None)
        if v is not None:
            setattr(dst, k, v.detach().clone() if torch.is_tensor(v) else v)
    return dst


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
        # (2026-11-11 in-situ 3D PSF, opt-in) image-plane Gaussian blur, applied to the oversampled intensity before pixel
        # integration: covariance [sxx, syy, sxy] in px^2 (list / tensor, may carry grad; [B,3] = per emitter).  Plain
        # attribute like diat; None = unchanged (bit-identical).  calib json key 'psf_blur_px2' (set_psf_extras).
        self.blur = None
        # (2026-11-11, opt-in) emission-dipole second-moment tensor M (3 x 3, trace 1; isotropic = I / 3), e.g. photoselection
        # by a polarised excitation: the image is sum_ij M_ij Re[E_i . E_j*] (E_i = image field of a unit dipole along i) instead
        # of sum_i |E_i|^2 / 3.  dipw [wx, wy, wz] (mean 1) is the diagonal shortcut M = diag(w) / 3.  None = unchanged.
        # calib keys 'psf_dipole_M' / 'psf_dipole_w' (global) and 'psf_dipole_M_angle' (per pattern angle, see simulator).
        self.dipw = None
        self.dipM = None

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

    BLUR_PAD = 3   # px of padding per side for the image-plane blur (kernel sd <~ 0.7 px)

    def forward(self, pos, focus, zcoef_offset=None, intensity=None, bg=None, blur=None, dipw=None, dipM=None):
        """
        pos: [B,3] (x, y relative to ROI center in pixels, z_e emitter height in um)
        focus: [B] focal plane position in um (relative to coverslip)
        zcoef_offset: optional [B,nz] per-emitter Zernike offsets (field dependence)
        blur: optional image-plane blur covariance [3] or [B,3] (sxx, syy, sxy; px^2), overrides self.blur
        Returns [B, roisize, roisize] expected image (normalized PSF * intensity + bg if given).
        """
        gs = getattr(self, 'gauss_sigma_um', None)
        if gs is not None:   # (2026-11-12, opt-in calib key 'psf_gauss_sigma_um'; absent = unchanged) the preprint's PSF:
            # pixel-integrated isotropic Gaussian, sigma = gs / pixelsize px (= analysis/baselines/zpsf.GaussPSF), independent of z,
            # focus and the Zernike / blur / dipole terms. A width-scaled copy (det/psfpert: pixelsize / w) gets sigma x w.
            s = (gs / self.pixelsize) * math.sqrt(2)
            o = torch.arange(self.roisize, device=pos.device, dtype=torch.float32) - (self.roisize - 1) / 2
            ax = lambda c_: 0.5 * (torch.erf((o[None] + 0.5 - c_[:, None]) / s) - torch.erf((o[None] - 0.5 - c_[:, None]) / s))
            psf = ax(pos[:, 1].float())[:, :, None] * ax(pos[:, 0].float())[:, None, :]
            if intensity is not None: psf = psf * intensity[:, None, None]
            if bg is not None: psf = psf + bg[:, None, None]
            return psf
        B = pos.shape[0]
        P, kz_med, kz_imm, kx, ky = self.pupil_fields()
        Mt = dipole_tensor(dipM if dipM is not None else (None if dipw is not None else getattr(self, 'dipM', None)),
                           dipw if dipw is not None else getattr(self, 'dipw', None), self.dev)
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
        bl = blur if blur is not None else getattr(self, 'blur', None)   # getattr: modules pickled before 2026-11-11
        u, pad = self.u, 0
        if bl is not None:   # blur: evaluate on a grid padded by `pad` px per side, so the kernel sees the PSF beyond the ROI
            pad = self.BLUR_PAD
            ns_p = (self.roisize + 2 * pad) * self.os
            u = (torch.arange(ns_p, device=self.dev) - (ns_p - 1) / 2) / self.os * self.pixelsize
        Fx = torch.exp(1j * u[:, None] * kx1[None, :])  # [ns, N]
        Fy = torch.exp(1j * u[:, None] * ky1[None, :])
        E = Fy @ Pb.to(torch.complex64) @ Fx.T  # [B,6,ns,ns]
        if Mt is None:
            I = (E.real ** 2 + E.imag ** 2).sum(1) / 3
        else:   # sum_ij M_ij Re[E_i . E_j*], E_i = (Ex, Ey) of dipole i = channels (2i, 2i+1)
            Ed = E.reshape(B, 3, 2, *E.shape[-2:])
            Mc = Mt.to(Ed.real.dtype)
            F = torch.einsum('ij,bjcuv->bicuv', Mc, Ed.real) + 1j * torch.einsum('ij,bjcuv->bicuv', Mc, Ed.imag) \
                if Mc.dim() == 2 else torch.einsum('bij,bjcuv->bicuv', Mc, Ed.real) + 1j * torch.einsum('bij,bjcuv->bicuv', Mc, Ed.imag)
            I = (Ed.real * F.real + Ed.imag * F.imag).sum((1, 2))
        # normalization: Parseval energy of aberration-free emitter at z=0 -> sum over plane = 1
        dk = (kx1[1] - kx1[0]) * (ky1[1] - ky1[0])
        energy = (P.real ** 2 + P.imag ** 2).sum() / 3 * dk / (2 * math.pi) ** 2
        # continuous field E = sum(...) * dk/(2pi)^2, so |E|^2 gets that factor squared
        I = I * (dk / (2 * math.pi) ** 2) ** 2
        if bl is not None:
            I = gauss_blur_os(I, bl, self.os, pad)
        # pixel integration: average over oversampled subpixels * pixel area
        R, o = self.roisize, self.os
        I = I.reshape(B, R, o, R, o).mean((2, 4)) * self.pixelsize ** 2
        psf = I / energy
        if intensity is not None:
            psf = psf * intensity[:, None, None]
        if bg is not None:
            psf = psf + bg[:, None, None]
        return psf
