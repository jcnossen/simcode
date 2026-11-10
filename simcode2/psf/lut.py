"""
Cached PSF for the fast LM: an oversampled z-stack of the calibrated VectorialPSF, interpolated with the user's
conv2d-based Catmull-Rom 3D spline (as in torch-spline-3d / torch_convspline, J. Cnossen, MIT licence; the only piece
needed, the Catmull-Rom geometry matrix, is vendored below), which also gives d/d(z, y, x).

The pixel-integrated PSF is shift invariant: psf(x, y, z)[i, j] = g(o_j - x, o_i - y, z), o = pixel offset from the ROI
centre. g is sampled once on a uniform grid (spacing 1/S px in u, v; hz um in z) by evaluating the same VectorialPSF
(identical NA / n_med / Zernikes / apodization / SAF terms, larger ROI) at S x S sub-pixel shifts: an S-times
oversampled z-stack [nz, nv, nu]. Output pixels lie S knots apart and share the fractional knot offset, so the ROI is
torch_convspline's Catmull-Rom 4x4x4 kernels (compute_weights_deriv: value, d/dz, d/dy, d/dx) applied with the same
grouped conv2d as eval_image_spline3D_deriv, but with stride S, to the [4, (P-1)S+4, (P-1)S+4] block of knots around the
emitter. eval_image_spline3D_deriv itself is not called: it is stride 1 (S^2 x wasted output) and calls .contiguous()
on the per-emitter expanded z-stack (copies the whole stack per emitter). The knot block is gathered with one linear
index (bit-identical to torch_convspline's slice_volumes for in-range blocks, ~3x faster; coordinates are clamped so
blocks are always in range).

PSFLUT(psf, focus, zmax) is callable like VectorialPSF: lut(pos [N,3], focus [N]) -> [N,P,P];
lut.evaluate(pos, grad=True) also returns d/dx, d/dy (px) and d/dz (um).
"""
import math
import torch
from simcode2.psf.vectorial import VectorialPSF


def geometry_matrix(device, tau: float = 0.5):
    """Catmull-Rom basis as in torch-spline-3d (torch_convspline.conv_spline.geometry_matrix, MIT licence, (c) 2023/2024
    Jelmer Cnossen), vendored unchanged: per-axis weights w = [1, t, t^2, t^3] @ G."""
    return torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [-tau, 0.0, tau, 0.0],
        [2 * tau, tau - 3, 3 - 2 * tau, -tau],
        [-tau, 2 - tau, tau - 2, tau]
    ], device=device, dtype=torch.float32)


class PSFLUT:
    def __init__(self, psf, focus, zmax=0.2, sub=6, hz=0.01, zpad=3, xmax=None, chunk=256):
        self.P = P = psf.roisize
        self.roisize = P
        self.focus = float(focus)
        self.dev = psf.u.device
        self.S = S = sub
        self.xmax = float(xmax if xmax is not None else P / 2 + 1)      # |x|,|y| supported (clamped beyond)
        c = int(math.ceil((P - 1) / 2 + self.xmax)) + 2                  # half-width of the big ROI (px)
        Pl = 2 * c + 1
        big = VectorialPSF(Pl, pixelsize=psf.pixelsize, NA=float(psf.NA), n_imm=psf.n_imm, n_med=float(psf.n_med),
                           wavelength=psf.wavelength, zernike_indices=psf.zidx, npupil=psf.npupil, oversample=psf.os,
                           device=self.dev)
        with torch.no_grad():
            big.zcoef.copy_(psf.zcoef); big.apod.copy_(psf.apod); big.saf.copy_(psf.saf)
        nz = int(round(zmax / hz)) + 1 + 2 * zpad
        self.z0, self.hz, self.nz = -zpad * hz, hz, nz
        zs = self.z0 + hz * torch.arange(nz, device=self.dev, dtype=torch.float64)
        a = torch.arange(S, device=self.dev, dtype=torch.float64) / S
        Zg, B, A = torch.meshgrid(zs, a, a, indexing='ij')                # shift (fx, fy) = (A, B)
        pos = torch.stack([A.reshape(-1), B.reshape(-1), Zg.reshape(-1)], 1).float()
        with torch.no_grad():
            imgs = torch.cat([big(p, torch.full((len(p),), self.focus, device=self.dev)) for p in pos.split(chunk)])
        imgs = imgs.reshape(nz, S, S, Pl, Pl)                             # [z, b, a, i, j]
        # g at u = (j - c) - a/S -> knot index j*S + (S-1-a) (contiguous 0 .. Pl*S-1); same for v with (i, b)
        G = imgs.flip(1).flip(2).permute(0, 3, 1, 4, 2).reshape(nz, Pl * S, Pl * S)   # [z, v, u]
        self.u0 = -(c * S + S - 1) / S                                    # u at knot index 0
        self.zstack = G.contiguous()                                      # [nz, nv, nu] knots
        D, H, W = self.zstack.shape; self.HW, self.W = H * W, W
        self.Lw = Lw = (P - 1) * S + 4
        r = lambda n_: torch.arange(n_, device=self.dev)
        self.pat = (r(4)[:, None, None] * H * W + r(Lw)[None, :, None] * W + r(Lw)[None, None, :]).reshape(-1)
        self.G = geometry_matrix(self.dev)                                # Catmull-Rom basis (tau 0.5)
        e = torch.arange(4, device=self.dev)
        self.e, self.em1 = e, (e - 1).clamp_min(0)
        o = torch.arange(P, device=self.dev, dtype=torch.float32) - (P - 1) / 2
        self.oS0 = float((o[0] - self.u0) * S)                            # knot index of pixel 0 at x=0

    def _weights(self, offs, grad):
        """Same as torch_convspline.conv_spline.compute_weights_deriv (spline_weights / spline_weights_deriv outer
        products), but with the geometry matrix cached: the library rebuilds it with torch.tensor(..., device) on every
        call, i.e. 6 blocking host->device copies per evaluation."""
        wa = (offs[..., None] ** self.e) @ self.G                          # [N,3,4] per-axis weights (z, y, x)
        we = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
        if not grad:
            return we
        da = (self.e * offs[..., None] ** self.em1) @ self.G
        w0 = da[:, 0, :, None, None] * wa[:, 1, None, :, None] * wa[:, 2, None, None, :]
        w1 = wa[:, 0, :, None, None] * da[:, 1, None, :, None] * wa[:, 2, None, None, :]
        w2 = wa[:, 0, :, None, None] * wa[:, 1, None, :, None] * da[:, 2, None, None, :]
        return torch.cat([we, w0, w1, w2], dim=1).view(-1, 4, 4, 4)

    def evaluate(self, pos, grad=False):
        """pos [N,3] (x, y px rel. ROI centre; z um). Returns psf [N,P,P] (and d/dx, d/dy, d/dz if grad)."""
        N, P, S = pos.shape[0], self.P, self.S
        x = pos[:, 0].clamp(-self.xmax, self.xmax); y = pos[:, 1].clamp(-self.xmax, self.xmax)
        tz = ((pos[:, 2] - self.z0) / self.hz).clamp(1, self.nz - 2 - 1e-4)
        zyx = torch.stack([tz, self.oS0 - y * S, self.oS0 - x * S], 1)  # knot-index coords of output pixel (0, 0)
        ix = torch.floor(zyx); offs = zyx - ix
        Lw = self.Lw
        i0 = ix.long() - 1                                                # block corner (as slice_volumes(.., ix - 1, ..))
        base = i0[:, 0] * self.HW + i0[:, 1] * self.W + i0[:, 2]
        sub = self.zstack.view(-1)[base[:, None] + self.pat[None]]       # [N, 4*Lw*Lw]
        w = self._weights(offs, grad)                                     # [4N or N,4,4,4]: value(, d/dz, d/dy, d/dx)
        r = torch.nn.functional.conv2d(sub.reshape(1, 4 * N, Lw, Lw), w, groups=N, stride=S)
        if not grad:
            return r.view(N, P, P)
        r = r.view(N, 4, P, P)
        return r[:, 0], -S * r[:, 3], -S * r[:, 2], r[:, 1] / self.hz

    def __call__(self, pos, focus=None, zcoef_offset=None, intensity=None, bg=None):
        assert zcoef_offset is None
        if focus is not None and not torch.is_tensor(focus):      # (tensor focus not checked: avoids a GPU sync)
            assert abs(float(focus) - self.focus) < 1e-6, 'PSFLUT built for another focus'
        with torch.no_grad():
            out = self.evaluate(pos)
        if intensity is not None: out = out * intensity[:, None, None]
        if bg is not None: out = out + bg[:, None, None]
        return out


def make_lut(sim, P):
    """PSFLUT of the simulator's calibrated vectorial PSF on a P x P ROI (same normalisation as crlb.models)."""
    p0 = sim.psf
    rp = type(p0)(P, pixelsize=p0.pixelsize, NA=float(p0.NA), n_imm=p0.n_imm, n_med=float(p0.n_med),
                  wavelength=p0.wavelength, oversample=3, device=p0.u.device)
    with torch.no_grad():
        rp.zcoef.copy_(p0.zcoef); rp.apod.copy_(p0.apod)
    return PSFLUT(rp, sim.focus, sim.zmax)
