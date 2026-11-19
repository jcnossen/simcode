"""Classical single-emitter 4Pi-SIMFLUX localization (no network), batched on GPU.

Input per candidate: the 24 sub-images img [n, 6 frames, 4 channels, P, P] (e), the per-channel sub-pixel offset of the
candidate in each channel patch doff [n, 4, 2] (patch centre = round(mapped position)), the SIM pattern in local patch
coordinates (k [2 angles, 2] rad/px, psi_eff [n, 6]: frame t modulation 1 + m cos(k_a . u - psi_eff_t), u relative to
the ch0 candidate pixel), depth m [n], and later the 4Pi phase offset Phi0 [n] (from the in-situ field).

Steps (the published procedure, SI Note 2 / Extended Data Fig. 5, re-implemented with the bead-fitted 4Pi PSF model):
 1. summed fit: Poisson MLE of the 24-image sum with the incoherent part of the 4Pi PSF (channel c at u + doff_c,
    i.e. registration without interpolation): x, y, z_ast (astigmatism), N, bg. Replaces the paper's elliptical
    Gaussian + width-vs-z calibration curve: same information, model-exact for this PSF.
 2. per-frame photon counts N_t (channel sum, position fixed, 2-parameter MLE per frame), per angle the exact 3-step
    solve -> pattern phase alpha_a, modulation depth MD_a, offset B_a; photon asymmetry R = |N_x - N_y| / (N_x + N_y).
    Lateral SIMFLUX position by the paper's unwrapping against the rough position: r* = r_RL + K^-1 wrap(alpha - K r_RL).
 3. 4Pi phase: Gaussian-weighted zeroth moments per channel and angle (frames of an angle summed), linear LS
    N_ca = B_a tau_c (1 + Re(e_c e^{i delta_a} Z)) with the channel responses e_c calibrated on the PSF model (as the
    paper calibrates the channel phase shift on a bead z-stack) -> phi = arg Z, contrast C = |Z|.
    z from phi via the calibrated phase-vs-z curve phi_cal(z) (same estimator applied to the model z-stack), unwrapped
    against the astigmatic z: phi_unw = phi_cal(z_ast) + wrap(phi - Phi0 - phi_cal(z_ast)), z_4pi = phi_cal^-1(phi_unw).
    Phi0(x, y, t) (interferometer phase field + drift) comes from the data (see run scripts).
 4. statistically optimal variant: pixel MLE of all 24 sub-images with the coherent 4Pi PSF, the pattern and Phi0,
    on-through emitter (x, y, z, F, bg x 4), 3 fringe starts around z_4pi.
"""
import math
import numpy as np
import torch
from simcode.psf import fourpi_channel
from pixel_lm import lm_poisson_j, fisher_j, crlb

V_PX = 1.14 ** 2 + 0.2 ** 2          # read noise + residual offset error variance per pixel and image (e^2)


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _best(a, b):
    """keep the higher-likelihood of two (th, ll, mu) results"""
    k = b[1] > a[1]
    return (torch.where(k[:, None], b[0], a[0]), torch.where(k, b[1], a[1]), torch.where(k[:, None], b[2], a[2]))


class Classic4Pi:
    def __init__(self, lut, delta=(0.0, math.radians(-10.4)), wsd=1.2, dev='cuda'):
        self.lut, self.dev, self.P = lut, dev, lut.P
        self.delta = delta
        self.wsd = wsd
        o = torch.arange(self.P, device=dev, dtype=torch.float32) - (self.P - 1) / 2
        self.oy, self.ox = torch.meshgrid(o, o, indexing='ij')
        border = torch.zeros(self.P, self.P, dtype=torch.bool, device=dev)
        border[[0, -1], :] = True; border[:, [0, -1]] = True
        self.border = border
        self._calibrate()

    # ------------------------------------------------------------------ calibration of the 4Pi phase read-out
    def _calibrate(self, zmin=-0.58, zmax=0.58, dz=0.002):
        """channel responses e_c (complex, at z = 0) and the phase-vs-z curve phi_cal(z) of the moment estimator,
        computed on the PSF model (a noise-free 'bead z-stack' at x = y = 0, Phi0 = 0)"""
        dev = self.dev
        zs = torch.arange(zmin, zmax + 1e-9, dz, device=dev)
        n = len(zs)
        pos = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), zs], 1)
        w = self._weights(torch.zeros(n, 4, 2, device=dev), torch.zeros(n, 2, device=dev))      # [n,4,P,P]
        I, R, Q = [], [], []
        for c in range(4):
            inc, cr, ci = (row[c].render(pos) for row in self.lut.luts)
            I.append((inc * w[:, c]).sum((-1, -2))); R.append((cr * w[:, c]).sum((-1, -2))); Q.append((ci * w[:, c]).sum((-1, -2)))
        I, R, Q = torch.stack(I, 1), torch.stack(R, 1), torch.stack(Q, 1)                         # [nz,4]
        i0 = int(torch.argmin(zs.abs()))
        self.tau = I[i0] / I[i0].sum()
        self.e = torch.complex(R[i0], Q[i0]) / I[i0]                                               # [4]
        # same estimator on the model stack (Phi0 = 0, both angles at delta 0)
        Z = [self._phasor(I + R, torch.ones_like(I))[0]]          # moments at Phi0 = 0: I + cos0 R - sin0 Q
        ph = torch.angle(Z[0]).cpu().numpy()
        phu = np.unwrap(ph)
        self.zcal = zs.cpu().numpy()
        self.phical = phu - phu[i0]
        self.phical0 = float(phu[i0])            # phase of the estimator at z = 0, Phi0 = 0 (absorbed in Phi0 conventions)
        self.Ccal = torch.abs(Z[0]).cpu().numpy()
        slope = np.gradient(self.phical, self.zcal)
        self.monotone = bool((np.sign(slope) == np.sign(slope[i0])).all())
        self.fringe_um = float(2 * math.pi / abs(slope[i0]))
        self._zc = torch.tensor(self.zcal, device=dev, dtype=torch.float32)
        self._pc = torch.tensor(self.phical * np.sign(slope[i0]), device=dev, dtype=torch.float32)   # increasing
        self._sgn = float(np.sign(slope[i0]))
        # z-dependent channel calibration with the common phase psi(z) removed: e~_c(z) = e_c(z) exp(-i psi(z)),
        # psi(z) = estimator phase with the z = 0 calibration; using e~_c at the emitter z then returns Phi0 + phi_cal(z)
        ez = torch.complex(R, Q) / I
        psi = torch.tensor(phu, device=dev, dtype=torch.float32)
        self.tau_z = I / I.sum(1, keepdim=True)                                                   # [nz,4]
        self.e_z = ez * torch.exp(torch.complex(torch.zeros_like(psi), -psi))[:, None]
        self.e_z = self.e_z * torch.exp(torch.complex(torch.tensor(0.0, device=dev), torch.tensor(self.phical0, device=dev)))

    def phi_cal(self, z):
        """calibrated phase at z (torch), relative to the estimator phase at z = 0"""
        t = ((z - self._zc[0]) / (self._zc[1] - self._zc[0])).clamp(0, len(self._zc) - 1.001)
        i = t.floor().long(); f = t - i
        return self._sgn * (self._pc[i] * (1 - f) + self._pc[i + 1] * f)

    def z_of_phi(self, ph):
        """inverse of phi_cal (monotone branch); ph unwrapped"""
        p = self._sgn * ph
        i = torch.searchsorted(self._pc, p.contiguous()).clamp(1, len(self._pc) - 1)
        p0, p1 = self._pc[i - 1], self._pc[i]
        f = ((p - p0) / (p1 - p0).clamp(min=1e-9)).clamp(-0.5, 1.5)
        return self._zc[i - 1] + f * (self._zc[1] - self._zc[0])

    def dphi_dz(self, z):
        h = 0.004
        return (self.phi_cal(z + h) - self.phi_cal(z - h)) / (2 * h)

    # ------------------------------------------------------------------ helpers
    def _weights(self, doff, xy):
        """Gaussian weights per channel centred at the emitter position in each channel patch: [n,4,P,P]"""
        px = xy[:, None, 0, None, None] + doff[:, :, 0, None, None]
        py = xy[:, None, 1, None, None] + doff[:, :, 1, None, None]
        return torch.exp(-((self.ox - px) ** 2 + (self.oy - py) ** 2) / (2 * self.wsd ** 2))

    def _inc_sum(self, x, y, z, doff, grad=False, cs=None):
        """sum over channels of the PSF, channel c at (x, y) + doff_c: [n,P,P] (+ d/dx, d/dy, d/dz). cs = (c, s) [n]:
        include the residual coherent part c * sum cr - s * sum ci (c = cos Phi, s = sin Phi; the four channels do not
        cancel it exactly). With grad and cs, also returns the two coherent images (for d/dc, d/ds)."""
        acc = None
        for c in range(4):
            pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
            comps = [self.lut.luts[q][c].render(pc, grad) for q in range(3 if cs is not None else 1)]
            if not grad: comps = [(r,) for r in comps]
            acc = [list(r) for r in comps] if acc is None else [[a + b for a, b in zip(A_, r)] for A_, r in zip(acc, comps)]
        if cs is None:
            return acc[0] if grad else acc[0][0]
        cc, ss = cs[0][:, None, None], cs[1][:, None, None]
        tot = [acc[0][q] + cc * acc[1][q] - ss * acc[2][q] for q in range(len(acc[0]))]
        if not grad: return tot[0]
        return tot + [acc[1][0], -acc[2][0]]

    def _phasor(self, Mc, var, tau=None, e=None):
        """weighted LS for one angle: Mc [n,4] = B tau_c (1 + Re(e_c Z)) -> Z [n] complex, B [n], cov of (B, BZr, BZi)"""
        tau = self.tau.expand_as(Mc) if tau is None else tau
        e = self.e.expand_as(Mc) if e is None else e
        A = torch.stack([tau, tau * e.real, -tau * e.imag], -1)   # [n,4,3]
        wt = 1.0 / var.clamp(min=1e-3)
        H = torch.einsum('nci,nc,ncj->nij', A, wt, A)
        g = torch.einsum('nci,nc,nc->ni', A, wt, Mc)
        Hi = torch.linalg.inv(H + 1e-9 * torch.eye(3, device=Mc.device))
        s = torch.einsum('nij,nj->ni', Hi, g)
        B = s[:, 0]
        Z = torch.complex(s[:, 1], s[:, 2]) / torch.where(B.abs() > 1e-6, B, torch.full_like(B, 1e-6))
        return Z, s, Hi

    # ------------------------------------------------------------------ 1. summed fit
    def summed_fit(self, img, doff, iters=30, zstarts=(-0.3, 0.0, 0.3), refine=None):
        """x, y, z, N, bg, c, s: N (inc + c cr - s ci) + bg summed over the 4 channels (c, s = linear nuisance amplitudes
        of the residual coherent sum, |c|, |s| <= 1.2)"""
        n = len(img); dev = self.dev
        S = img.sum((1, 2))                                                   # [n,P,P]
        d = S.reshape(n, -1)
        v = 24 * V_PX

        def fj(th):
            x, y, z, N, b, cc, ss = th.unbind(1)
            g, gx, gy, gz, gc, gs = self._inc_sum(x, y, z, doff, True, cs=(cc, ss))
            Np = N[:, None, None]
            mu = Np * g + b[:, None, None]
            J = torch.stack([Np * gx, Np * gy, Np * gz, g, torch.ones_like(g), Np * gc, Np * gs], -1)
            return mu.reshape(n, -1), J.reshape(n, -1, 7)

        bg0 = S[:, self.border].median(1).values.clamp(min=0.1)
        N0 = (d - bg0[:, None]).sum(1).clamp(min=50)
        lo = torch.tensor([-3, -3, -0.6, 1, 0, -1.2, -1.2], device=dev, dtype=torch.float32)
        hi = torch.tensor([3, 3, 0.6, 1e6, 1e3, 1.2, 1.2], device=dev, dtype=torch.float32)
        best = None
        z0 = torch.zeros(n, device=dev)
        if zstarts == 'grid':      # one start: best z of a 9-point likelihood grid at the centroid
            Sb = (S - bg0[:, None, None]).clamp(min=0); tot = Sb.sum((1, 2)).clamp(min=1e-3)
            xc = ((Sb * self.ox).sum((1, 2)) / tot).clamp(-1.5, 1.5); yc = ((Sb * self.oy).sum((1, 2)) / tot).clamp(-1.5, 1.5)
            bl = None
            from pixel_lm import loglik
            for zs in torch.linspace(-0.48, 0.48, 9).tolist():
                th0 = torch.stack([xc, yc, torch.full((n,), zs, device=dev), N0, bg0, z0, z0], 1)
                g = self._inc_sum(xc, yc, th0[:, 2], doff)
                ll = loglik(d, (N0[:, None, None] * g + bg0[:, None, None]).reshape(n, -1), v)
                bl = (th0, ll) if bl is None else (torch.where((ll > bl[1])[:, None], th0, bl[0]), torch.maximum(ll, bl[1]))
            best = lm_poisson_j(fj, d, bl[0], v, lo=lo, hi=hi, iters=iters)
        else:
            for zs in zstarts:
                th0 = torch.stack([z0, z0, torch.full((n,), zs, device=dev), N0, bg0, z0, z0], 1)
                r = lm_poisson_j(fj, d, th0, v, lo=lo, hi=hi, iters=iters if refine is None else refine[0])
                best = r if best is None else _best(best, r)
            if refine is not None:
                best = lm_poisson_j(fj, d, best[0], v, lo=lo, hi=hi, iters=refine[1])
        th, ll, mu = best
        th_c, _, _ = lm_poisson_j(fj, d, th, v, lo=lo, hi=hi, iters=8)
        conv = ((th_c[:, :2] - th[:, :2]).abs().max(1).values < 0.01) & ((th_c[:, 2] - th[:, 2]).abs() < 0.002)
        cr = crlb(fisher_j(fj, th, v))
        chi2px = (((d - mu) ** 2) / (mu.clamp(min=1e-6) + v)).mean(1)
        dd = (d + v).clamp(min=1e-6); mm = (mu + v).clamp(min=1e-6)
        dev2 = 2 * (dd * torch.log(dd / mm) - (dd - mm)).sum(1)                 # deviance (log-likelihood ratio)
        at_bound = (th[:, :2].abs() >= 2.95).any(1) | (th[:, 2].abs() >= 0.59) | (th[:, 3] <= 1.5) | (th[:, 4] <= 1e-3) | (th[:, 4] >= 999)
        return dict(th=th, crlb=cr, chi2px=chi2px, llr=dev2, conv=conv, at_bound=at_bound)

    # ------------------------------------------------------------------ 2. per-frame counts, lateral two-step
    def frame_counts(self, img, doff, th, iters=8):
        """per frame t: channel-summed image, N_t g + b_t with g = summed incoherent PSF at the summed-fit position;
        2-parameter Poisson MLE (Newton). -> N [n,6], b [n,6], var(N) [n,6]"""
        n = len(img)
        g = self._inc_sum(th[:, 0], th[:, 1], th[:, 2], doff, cs=(th[:, 5], th[:, 6]))[:, None]     # [n,1,P,P]
        D = img.sum(2)                                                           # [n,6,P,P]
        v = 4 * V_PX
        b = D[:, :, self.border].median(-1).values.clamp(min=0.05)
        N = (D - b[..., None, None]).sum((-1, -2)).clamp(min=1.0)
        for _ in range(iters):
            mu = N[..., None, None] * g + b[..., None, None]
            w = 1.0 / (mu + v).clamp(min=1e-3)
            r = (D - mu) * w
            gN = (r * g).sum((-1, -2)); gb = r.sum((-1, -2))
            hNN = (g * g * w).sum((-1, -2)); hNb = (g * w).sum((-1, -2)); hbb = w.sum((-1, -2))
            det = hNN * hbb - hNb ** 2
            N = (N + (hbb * gN - hNb * gb) / det).clamp(min=0.1)
            b = (b + (hNN * gb - hNb * gN) / det).clamp(min=0.01)
        mu = N[..., None, None] * g + b[..., None, None]
        w = 1.0 / (mu + v).clamp(min=1e-3)
        hNN = (g * g * w).sum((-1, -2)); hNb = (g * w).sum((-1, -2)); hbb = w.sum((-1, -2))
        varN = hbb / (hNN * hbb - hNb ** 2)
        return N, b, varN

    def lateral(self, N, varN, k, psi_eff, xy):
        """3-step solve per angle with the step phases psi_eff (local coordinates). Returns dict with alpha [n,2],
        MD [n,2], B [n,2], sigma_alpha [n,2], R (photon asymmetry), and the unwrapped SIMFLUX position xy* [n,2]
        and its sigma (from the alpha errors)."""
        al, md, Bs, sal = [], [], [], []
        for a in range(2):
            I = N[:, 3 * a:3 * a + 3]; ps = psi_eff[:, 3 * a:3 * a + 3]
            M = torch.stack([torch.ones_like(ps), torch.cos(ps), torch.sin(ps)], -1)        # [n,3,3]
            Mi = torch.linalg.inv(M)
            s = torch.einsum('nij,nj->ni', Mi, I)
            B, C, S_ = s.unbind(1)
            al.append(torch.atan2(S_, C)); md.append(torch.hypot(C, S_) / B.clamp(min=1e-3)); Bs.append(B)
            r2 = (C ** 2 + S_ ** 2).clamp(min=1e-6)
            gI = (C[:, None] * Mi[:, 2] - S_[:, None] * Mi[:, 1]) / r2[:, None]               # d alpha / d I_j
            sal.append(torch.sqrt((gI ** 2 * varN[:, 3 * a:3 * a + 3]).sum(1)))
        alpha = torch.stack(al, 1); MD = torch.stack(md, 1); sa = torch.stack(sal, 1)
        Nx, Ny = N[:, :3].sum(1), N[:, 3:].sum(1)
        R = (Nx - Ny).abs() / (Nx + Ny).clamp(min=1e-3)
        K = k.to(N.device)                                                     # [2,2]: rows = angles
        Ki = torch.linalg.inv(K)
        res = wrap(alpha - xy @ K.T)
        xys = xy + res @ Ki.T
        sxy = torch.sqrt((sa[:, None, :] ** 2 * Ki[None] ** 2).sum(-1))         # [n,2]
        return dict(alpha=alpha, MD=MD, B=torch.stack(Bs, 1), salpha=sa, R=R, xy=xys, sxy=sxy)

    # ------------------------------------------------------------------ 3. 4Pi phase from channel moments
    def calib_at(self, z):
        i = ((z - self._zc[0]) / (self._zc[1] - self._zc[0])).round().long().clamp(0, len(self._zc) - 1)
        return self.tau_z[i], self.e_z[i]

    def zphase(self, img, doff, xy, z=None):
        """Gaussian-weighted zeroth moments per channel and angle, background from the patch border; phasor LS per
        angle, combined with the per-angle offset delta. -> phi [n], C [n], sphi [n], Mca [n,2,4]"""
        w = self._weights(doff, xy)                                            # [n,4,P,P]
        cal = None
        if z is not None:            # model moments of each channel at the emitter's own position (x + doff_c, z)
            I_, R_, Q_ = [], [], []
            for c in range(4):
                pc = torch.stack([xy[:, 0] + doff[:, c, 0], xy[:, 1] + doff[:, c, 1], z], 1)
                inc, cr, ci = (row[c].render(pc) for row in self.lut.luts)
                I_.append((inc * w[:, c]).sum((-1, -2))); R_.append((cr * w[:, c]).sum((-1, -2))); Q_.append((ci * w[:, c]).sum((-1, -2)))
            I_, R_, Q_ = torch.stack(I_, 1), torch.stack(R_, 1), torch.stack(Q_, 1)
            psi = self.phi_cal(z) + self.phical0
            ec = torch.complex(R_, Q_) / I_.clamp(min=1e-9) * torch.exp(torch.complex(torch.zeros_like(psi), self.phical0 - psi))[:, None]
            cal = (I_ / I_.sum(1, keepdim=True), ec)
        Z, Wsum = [], []
        cov = []
        Ms = []
        sphi_parts = []
        for a in range(2):
            D = img[:, 3 * a:3 * a + 3].sum(1)                                  # [n,4,P,P]
            b = D[:, :, self.border].median(-1).values                          # [n,4]
            Mc = ((D - b[..., None, None]) * w).sum((-1, -2))
            var = ((D.clamp(min=0) + 3 * V_PX) * w * w).sum((-1, -2))
            Za, s, Hi = self._phasor(Mc, var, *(cal if cal is not None else (None, None)))
            rot = complex(math.cos(-self.delta[a]), math.sin(-self.delta[a]))
            Wc = torch.complex(s[:, 1], s[:, 2]) * rot                           # B_a Z rotated to angle 0
            Z.append(Wc); Wsum.append(s[:, 0]); Ms.append(Mc)
            # covariance of (Re, Im) of Wc: rotate the 2x2 block of Hi
            c_, s_ = math.cos(-self.delta[a]), math.sin(-self.delta[a])
            Rm = torch.tensor([[c_, -s_], [s_, c_]], device=img.device, dtype=img.dtype)
            cov.append(Rm @ Hi[:, 1:, 1:] @ Rm.T)
        W = Z[0] + Z[1]; B = Wsum[0] + Wsum[1]
        phi = torch.angle(W) - (self.phical0 if z is None else 0.0)
        C = W.abs() / B.clamp(min=1e-3)
        Cv = cov[0] + cov[1]
        gr = torch.stack([-W.imag, W.real], -1) / (W.abs() ** 2).clamp(min=1e-9)[:, None]
        sphi = torch.sqrt(torch.einsum('ni,nij,nj->n', gr, Cv, gr).clamp(min=0))
        return dict(phi=wrap(phi), C=C, sphi=sphi, M=torch.stack(Ms, 1), B=B)

    def unwrap_z(self, phi, Phi0, z_ast):
        """z from the 4Pi phase: nearest fringe to the astigmatic z along the calibrated curve"""
        pc = self.phi_cal(z_ast)
        pu = pc + wrap(phi - Phi0 - pc)
        z = self.z_of_phi(pu)
        return z

    # ------------------------------------------------------------------ 4. 24-image pixel MLE
    def pixel_mle(self, img, doff, start, m, psi_eff, k, Phi0, iters=25, fringe=None, ret_fisher=False, nstart=3, refine=None):
        """start [n,3] (x, y, z); returns th [n,8] (x, y, z, F, bg0..3), crlb, ll, chi2px, llr, conv, at_bound"""
        n = len(img); dev = self.dev; lut = self.lut
        d = img.reshape(n, -1)
        K = k.to(dev)
        ang = [0, 0, 0, 1, 1, 1]

        def fj(th):
            x, y, z, F = th[:, 0], th[:, 1], th[:, 2], th[:, 3]
            arg = torch.stack([K[ang[t], 0] * x + K[ang[t], 1] * y - psi_eff[:, t] for t in range(6)], 1)   # [n,6]
            pat = 1 + m[:, None] * torch.cos(arg)
            dpat = -m[:, None] * torch.sin(arg)
            dpx = dpat * torch.tensor([K[a_, 0] for a_ in ang], device=dev)[None]
            dpy = dpat * torch.tensor([K[a_, 1] for a_ in ang], device=dev)[None]
            mus, Js = [], []
            ai = torch.tensor(ang, device=dev)
            for c in range(4):
                pc = torch.stack([x + doff[:, c, 0], y + doff[:, c, 1], z], 1)
                G = [fourpi_channel(lut, pc, Phi0 + self.delta[a], c, grad=True) for a in range(2)]
                g, gx, gy, gz = (torch.stack([G[0][q], G[1][q]], 1)[:, ai] for q in range(4))           # [n,6,P,P]
                Fp = F[:, None, None, None]; pt = pat[:, :, None, None]
                mu_c = Fp * pt * g + th[:, 4 + c, None, None, None]
                jx = Fp * (dpx[:, :, None, None] * g + pt * gx)
                jy = Fp * (dpy[:, :, None, None] * g + pt * gy)
                jz = Fp * pt * gz
                jF = pt * g
                jb = [torch.full_like(g, 1.0 if cc == c else 0.0) for cc in range(4)]
                mus.append(mu_c); Js.append(torch.stack([jx, jy, jz, jF] + jb, -1))
            return torch.stack(mus, 2).reshape(n, -1), torch.stack(Js, 2).reshape(n, -1, 8)

        bgc = img[:, :, :, self.border].median(-1).values.median(1).values.clamp(min=0.01)    # [n,4]
        F0 = ((img.sum((2, 3, 4)) - (bgc.sum(1) * self.P ** 2)[:, None]).mean(1)).clamp(min=20)
        lo = torch.tensor([-3, -3, -0.6, 1, 0, 0, 0, 0], device=dev, dtype=torch.float32)
        hi = torch.tensor([3, 3, 0.6, 1e5, 50, 50, 50, 50], device=dev, dtype=torch.float32)
        fr = fringe if fringe is not None else self.fringe_um
        best = None
        it0 = iters if refine is None else refine[0]
        for dz in ((-fr, 0.0, fr) if nstart == 3 else (0.0,)):
            th0 = torch.cat([start[:, :2], (start[:, 2:3] + dz).clamp(-0.57, 0.57), F0[:, None], bgc], 1)
            r = lm_poisson_j(fj, d, th0, V_PX, lo=lo, hi=hi, iters=it0)
            best = r if best is None else _best(best, r)
        if refine is not None:      # short starts, then more iterations on the best only
            best = lm_poisson_j(fj, d, best[0], V_PX, lo=lo, hi=hi, iters=refine[1])
        th, ll, mu = best
        th_c, _, _ = lm_poisson_j(fj, d, th, V_PX, lo=lo, hi=hi, iters=6)
        conv = ((th_c[:, :2] - th[:, :2]).abs().max(1).values < 0.01) & ((th_c[:, 2] - th[:, 2]).abs() < 0.002)
        Fi = fisher_j(fj, th, V_PX)
        cr = crlb(Fi)
        chi2px = (((d - mu) ** 2) / (mu.clamp(min=1e-6) + V_PX)).mean(1)
        dd = (d + V_PX).clamp(min=1e-6); mm = (mu + V_PX).clamp(min=1e-6)
        llr = 2 * (dd * torch.log(dd / mm) - (dd - mm)).sum(1)
        at_bound = (th[:, :2].abs() >= 2.95).any(1) | (th[:, 2].abs() >= 0.59) | (th[:, 3] <= 1.5) | (th[:, 3] >= 0.99e5)
        out = dict(th=th, crlb=cr, ll=ll, chi2px=chi2px, llr=llr, conv=conv, at_bound=at_bound)
        if ret_fisher: out['fj'] = fj
        return out


def run_chunk(c4, img, doff, k, psi_eff, m, Phi0=None, do_pix=True, z_override=None):
    """steps 1-3 (and 4 if Phi0 is given and do_pix) for one chunk; returns a dict of torch tensors"""
    s = c4.summed_fit(img, doff)
    th = s['th']
    N, b, varN = c4.frame_counts(img, doff, th)
    lat = c4.lateral(N, varN, k, psi_eff, th[:, :2])
    zp = c4.zphase(img, doff, th[:, :2])
    out = dict(x0=th[:, 0], y0=th[:, 1], z_ast=th[:, 2], Nsum=th[:, 3], bgsum=th[:, 4], sx0=s['crlb'][:, 0], sy0=s['crlb'][:, 1],
               sz_ast=s['crlb'][:, 2], chi2px=s['chi2px'], llr=s['llr'], conv=s['conv'], at_bound=s['at_bound'],
               Nt=N, bt=b, alpha=lat['alpha'], MD=lat['MD'], R=lat['R'], xs=lat['xy'][:, 0], ys=lat['xy'][:, 1],
               sxs=lat['sxy'][:, 0], sys=lat['sxy'][:, 1], phi=zp['phi'], C=zp['C'], sphi=zp['sphi'])
    if Phi0 is not None:
        z4 = c4.unwrap_z(zp['phi'], Phi0, th[:, 2])
        for _ in range(2):          # z-dependent channel calibration at the current z
            zp = c4.zphase(img, doff, th[:, :2], z=z4)
            z4 = c4.unwrap_z(zp['phi'], Phi0, th[:, 2])
        out.update(phi=zp['phi'], C=zp['C'], sphi=zp['sphi'])
        out['z4'] = z4
        out['sz4'] = zp['sphi'] / c4.dphi_dz(z4).abs().clamp(min=1.0)
        if do_pix:
            st = torch.stack([lat['xy'][:, 0], lat['xy'][:, 1], z4], 1)
            p = c4.pixel_mle(img, doff, st, m, psi_eff, k, Phi0)
            out.update(xp=p['th'][:, 0], yp=p['th'][:, 1], zp=p['th'][:, 2], Fp=p['th'][:, 3], sxp=p['crlb'][:, 0], syp=p['crlb'][:, 1],
                       szp=p['crlb'][:, 2], chi2p=p['chi2px'], llrp=p['llr'], convp=p['conv'], boundp=p['at_bound'])
    return out
